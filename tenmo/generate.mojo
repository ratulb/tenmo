"""Sampling + uncached generation for the Mojo LLM path.

`sample` is the helper (temperature + top-k over `Tensor.multinomial`);
`generate` is the uncached autoregressive loop: full forward over the
whole prefix each step, append, slide at `n_ctx`. Deliberately O(T²) — v1
correctness baseline; the KV-cache follow-up must reproduce its output
exactly. Both run grad-free; `generate` forces `eval()` on entry and
restores `train()` on exit (same contract as `eval_epoch`).
"""

from std.utils.numerics import neg_inf
from .gpt import GPTModel, KVCache
from .tensor import Tensor
from .shared.shapes import Shape

def sample[dtype: DType](
    logits: Tensor[dtype],
    temperature: Float32 = 0.8,
    top_k: Int = 50,
    init_seed: Optional[Int] = None,
) raises -> Int where dtype.is_floating_point():
    """Draw one token id from 1-D `logits` (V,).

    - `temperature <= 0` or `top_k == 1` → greedy argmax (deterministic).
    - Otherwise: keep the top-`k` entries (`0 < k < V`; `k <= 0` or `k >= V`
      disables the mask), scale the survivors by `1/τ`, softmax, single
      `multinomial` draw.

    Seeding:
    - `init_seed == None` → `multinomial` draws fresh entropy per call, so
      repeated calls are (almost surely) different. This is the default.
    - `init_seed == Some(s)` → the draw is reproducible for a given `s`.

    Scaling by positive `1/τ` is monotonic, so it commutes with top-k
    selection: we select on raw logits and scale only the survivors.
    """
    var V = logits.numels()
    if V == 0:
        raise Error("sample: empty logits")

    # ---- Greedy shortcuts -------------------------------------------------
    # τ <= 0 means "no randomness"; top_k == 1 means "only one candidate".
    if temperature <= 0.0 or top_k == 1:
        return Int(logits.argmax(axis=0).item())

    # Precompute 1/τ once; applied only to survivors (or to all V when
    # truncation is disabled).
    var inv_tau = Scalar[dtype](1.0) / Scalar[dtype](temperature)

    # Softmax normalizes along the only axis of a 1-D tensor.
    var axes = List[Int]()
    axes.append(0)

    var probs: Tensor[dtype]

    if top_k > 1 and top_k < V:
        # ---- Top-k truncation --------------------------------------------
        # Manual selection: repeatedly take the current max and mask it with
        # -inf so the next argmax finds the runner-up. O(top_k * V) total.
        # At k=50, V=256 that is ~12.8k comparisons — negligible. If V or k
        # grows, swap this for a min-heap (O(V log k)) or quickselect
        # (O(V) avg). Selection runs on raw logits (see docstring) and we
        # snapshot the *scaled* value at pick time, so logits is never
        # re-read and no full-V division pass is needed.
        var work = logits.clone()
        var kept = List[Int](capacity=top_k)
        var kept_values = List[Scalar[dtype]](capacity=top_k)
        for _ in range(top_k):
            var m = Int(work.argmax(axis=0).item())
            kept.append(m)
            kept_values.append(work[m] * inv_tau)
            work[m] = neg_inf[dtype]()

        # Build the masked logit vector: -inf everywhere, scaled survivors
        # in their original positions. exp(-inf) == 0, so softmax gives the
        # masked entries exactly zero probability.
        var masked = Tensor[dtype].full(Shape(V), neg_inf[dtype]())
        for i in range(top_k):
            masked[kept[i]] = kept_values[i]

        probs = masked.softmax[track_grad=False](axes)
    else:
        # ---- No truncation -----------------------------------------------
        # top_k <= 1 is handled above; top_k >= V means "keep everything".
        if temperature != 1.0:
            var scaled = logits.clone() * inv_tau
            probs = scaled.softmax[track_grad=False](axes)
        else:
            # τ == 1 is a no-op; skip the clone and softmax the input directly.
            probs = logits.softmax[track_grad=False](axes)

    # ---- Single draw ------------------------------------------------------
    # One sample, so `replacement` is inert here; kept for callers who later
    # bump num_samples. `init_seed == None` → fresh entropy inside
    # `multinomial`; `init_seed != None` → reproducible draw.
    var drawn = probs.multinomial(
        num_samples=1, replacement=False, init_seed=init_seed
    )
    return Int(drawn.item())

def generate[OutT: DType](
    mut model: GPTModel[OutT],
    prompt: Tensor[DType.int64],
    max_new_tokens: Int,
    temperature: Float32 = 0.8,
    top_k: Int = 50,
    end_id: Int = -1,
    init_seed: Optional[Int] = None,
) raises -> Tensor[DType.int64] where OutT.is_floating_point():
    """Uncached autoregressive generation: prompt + up to `max_new_tokens`.

    Each step runs the full model over the last-`n_ctx` window, samples the
    final position, appends. Stops early on `end_id` (>= 0).

    Seeding:
    - `init_seed == None` → every step passes `None` to `sample`, so each
      step (and each whole run) draws fresh entropy. Non-reproducible; the
      default.
    - `init_seed == Some(s)` → step `m` uses seed `s + m`, so a whole run is
      reproducible while each step's Gumbel noise is independent. `s + m`
      must not overflow `Int` (i.e. `s + max_new_tokens` must fit).

    Returns the 1-D id stream (prompt prefix intact).
    """
    if prompt.numels() == 0:
        raise Error("generate: empty prompt")
    if max_new_tokens <= 0:
        raise Error("generate: max_new_tokens must be >= 1")

    model.eval()
    var ids = List[Int](capacity=prompt.numels() + max_new_tokens)
    for i in range(prompt.numels()):
        var v = Int(prompt.get(i))
        if v < 0:
            model.train()
            raise Error("generate: negative prompt id")
        ids.append(v)

    # `base_seed` is only read when `have_seed` is true; the `else 0` is a
    # placeholder so the variable is always initialized.
    var base_seed = init_seed.value() if init_seed else 0
    var have_seed = init_seed != None
    var made = 0
    while made < max_new_tokens:
        var total = len(ids)
        var T = min(total, model.n_ctx)
        var window = List[Scalar[DType.int64]](capacity=T)
        for i in range(total - T, total):
            window.append(Scalar[DType.int64](ids[i]))
        var batch = Tensor[DType.int64].from_list[DType.int64](window)
        var flat = batch.reshape(1, T)

        var logits = model(flat)  # (1, T, V), grad-free under eval()
        var V = logits.shape()[2]
        var last = logits.slice(T - 1, T, 1, axis=1)  # (1, 1, V) view
        var last_1d = last.reshape(V)

        # Seeded run: step m uses `base_seed + m` so each step's Gumbel
        # noise is independent while the whole run stays reproducible.
        # Unseeded run: pass None through, so `multinomial` draws fresh
        # entropy per step.
        var step_seed = Optional[Int](None)
        if have_seed:
            step_seed = Optional[Int](base_seed + made)
        var next_id = sample(
            last_1d, temperature, top_k, init_seed=step_seed
        )
        ids.append(next_id)
        made += 1
        if end_id >= 0 and next_id == end_id:
            break

    model.train()
    var out = List[Scalar[DType.int64]](capacity=len(ids))
    for i in range(len(ids)):
        out.append(Scalar[DType.int64](ids[i]))
    return Tensor[DType.int64].from_list[DType.int64](out)

def generate_cached[OutT: DType](
    mut model: GPTModel[OutT],
    prompt: Tensor[DType.int64],
    max_new_tokens: Int,
    temperature: Float32 = 0.8,
    top_k: Int = 50,
    end_id: Int = -1,
    init_seed: Optional[Int] = None,
) raises -> Tensor[DType.int64] where OutT.is_floating_point():
    """KV-cached generation: same contract as `generate`, O(T) steps.

    Prefill feeds prompt ids one at a time through `forward_step`
    (positions 0..P-1); the prefill-LAST logits yield the first id, then
    each decode step feeds the last id at `pos = len(ids) - 1` (= cache
    length — each step appends exactly one) and samples position 0 of
    the `(1, 1, V)` output. Same `init_seed + made` schedule as
    `generate`, so seeded runs are comparable id-for-id with the
    uncached loop. Single-batch (B=1): `prompt` is 1-D.
    Eval-entry/train-restore like `generate`.

    v1 constraint: unlike `generate` (which slides the window), the cache
    never drops entries, so `len(prompt) + max_new_tokens <= n_ctx` is
    required — raises otherwise (cache sliding is follow-up work).
    """
    if prompt.numels() == 0:
        raise Error("generate_cached: empty prompt")
    if max_new_tokens <= 0:
        raise Error("generate_cached: max_new_tokens must be >= 1")

    model.eval()
    if prompt.numels() + max_new_tokens > model.n_ctx:
        model.train()
        raise Error(
            "generate_cached: prompt + max_new_tokens exceeds n_ctx "
            "(v1 has no cache sliding)"
        )
    var ids = List[Int](capacity=prompt.numels() + max_new_tokens)
    for i in range(prompt.numels()):
        var v = Int(prompt.get(i))
        if v < 0:
            model.train()
            raise Error("generate_cached: negative prompt id")
        ids.append(v)

    var cache = KVCache[OutT]()
    # Prefill: build the cache token by token (positions 0..P-1). The
    # prefill-LAST logits produce the first id — exactly the query the
    # uncached loop samples (its window's last position), so the streams
    # agree id-for-id. Only the last step's logits are kept.
    for p in range(len(ids) - 1):
        var one = List[Scalar[DType.int64]](capacity=1)
        one.append(Scalar[DType.int64](ids[p]))
        var x_1 = Tensor[DType.int64].from_list[DType.int64](one).reshape(
            1, 1
        )
        _ = model.forward_step(x_1, p, cache)
    var last_one = List[Scalar[DType.int64]](capacity=1)
    last_one.append(Scalar[DType.int64](ids[len(ids) - 1]))
    var x_last = Tensor[DType.int64].from_list[DType.int64](
        last_one
    ).reshape(1, 1)
    var prefill_last = model.forward_step(x_last, len(ids) - 1, cache)

    var base_seed = init_seed.value() if init_seed else 0
    var have_seed = True if init_seed else False
    var made = 0
    # First id from the prefill-last logits (no new forward).
    var first_logits = prefill_last.reshape(prefill_last.shape()[2])
    var first_seed = Optional[Int](None)
    if have_seed:
        first_seed = Optional[Int](base_seed)
    var first_id = sample(
        first_logits, temperature, top_k, init_seed=first_seed
    )
    ids.append(first_id)
    made += 1
    if not (end_id >= 0 and first_id == end_id):
        while made < max_new_tokens:
            var one = List[Scalar[DType.int64]](capacity=1)
            one.append(Scalar[DType.int64](ids[len(ids) - 1]))
            var x_1 = Tensor[DType.int64].from_list[DType.int64](
                one
            ).reshape(1, 1)
            var logits = model.forward_step(
                x_1, len(ids) - 1, cache
            )  # (1, 1, V)
            var last_1d = logits.reshape(logits.shape()[2])

            var step_seed = Optional[Int](None)
            if have_seed:
                step_seed = Optional[Int](base_seed + made)
            var next_id = sample(
                last_1d, temperature, top_k, init_seed=step_seed
            )
            ids.append(next_id)
            made += 1
            if end_id >= 0 and next_id == end_id:
                break

    model.train()
    var out = List[Scalar[DType.int64]](capacity=len(ids))
    for i in range(len(ids)):
        out.append(Scalar[DType.int64](ids[i]))
    return Tensor[DType.int64].from_list[DType.int64](out)
