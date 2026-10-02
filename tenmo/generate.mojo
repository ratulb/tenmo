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


def sample[dtype: DType](
    logits: Tensor[dtype],
    temperature: Float32 = 0.8,
    top_k: Int = 50,
    init_seed: Optional[Int] = None,
) raises -> Int where dtype.is_floating_point():
    """Draw one token id from 1-D `logits` (V,).

    - `temperature <= 0` or `top_k == 1` → greedy argmax (deterministic).
    - Otherwise: scale by `1/τ`, keep the top-`k` entries (`0 < k < V`;
      `k <= 0` or `k >= V` disables the mask), softmax, single
      `multinomial` draw. `init_seed` pins the draw for tests.
    """
    var V = logits.numels()
    if V == 0:
        raise Error("sample: empty logits")

    if temperature <= 0.0 or top_k == 1:
        return Int(logits.argmax(axis=0).item())

    var work = logits.clone()
    if temperature != 1.0:
        work = work / Scalar[dtype](temperature)

    var probs: Tensor[dtype]
    if top_k > 1 and top_k < V:
        var kept = List[Int](capacity=top_k)
        for _ in range(top_k):
            var m = Int(work.argmax(axis=0).item())
            kept.append(m)
            work[m] = neg_inf[dtype]()
        var masked = Tensor[dtype].zeros(V) + neg_inf[dtype]()
        for k in range(top_k):
            masked[kept[k]] = logits[kept[k]] / Scalar[dtype](temperature) if temperature != 1.0 else logits[kept[k]]
        var axes = List[Int]()
        axes.append(0)
        probs = masked.softmax[track_grad=False](axes)
    else:
        var axes = List[Int]()
        axes.append(0)
        probs = work.softmax[track_grad=False](axes)

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
    final position, appends. Stops early on `end_id` (>= 0). Per-step seeds
    derive from `init_seed + step` (a fixed seed must still vary the draws).
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

    var base_seed = init_seed.value() if init_seed else 0
    var have_seed = True if init_seed else False
    var made = 0
    while made < max_new_tokens:
        var total = len(ids)
        var T = total if total < model.n_ctx else model.n_ctx
        var window = List[Scalar[DType.int64]](capacity=T)
        for i in range(total - T, total):
            window.append(Scalar[DType.int64](ids[i]))
        var batch = Tensor[DType.int64].from_list[DType.int64](window^)
        var flat = batch.reshape(1, T)

        var logits = model(flat)  # (1, T, V), grad-free under eval()
        var V = logits.shape()[2]
        var last = logits.slice(T - 1, T, 1, axis=1)  # (1, 1, V) view
        var last_1d = last.reshape(V)

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
    return Tensor[DType.int64].from_list[DType.int64](out^)


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
        var x_1 = Tensor[DType.int64].from_list[DType.int64](one^).reshape(
            1, 1
        )
        _ = model.forward_step(x_1, p, cache)
    var last_one = List[Scalar[DType.int64]](capacity=1)
    last_one.append(Scalar[DType.int64](ids[len(ids) - 1]))
    var x_last = Tensor[DType.int64].from_list[DType.int64](
        last_one^
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
                one^
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
    return Tensor[DType.int64].from_list[DType.int64](out^)
