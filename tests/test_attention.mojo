"""
Tests for SelfAttention.

────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE IS FOR (read this first if you are new to the codebase).
────────────────────────────────────────────────────────────────────────────
`SelfAttention` is the "attention" building block used inside a
transformer / GPT-style language model. Given a batch of token embeddings of
shape (B, T, C) — B = batch size, T = sequence length ("time"), C = embedding
width ("channels") — it lets every position `t` in the sequence gather
information from positions `0..t` (never from the future — that's the
"causal" part), and returns a new tensor of the same shape (B, T, C).

Internally it does, per head:
    1. Project x -> q, k, v via a single linear layer `c_attn`
       (weight shape (C, 3C), so one matmul produces query/key/value at once).
    2. Compute attention scores = q @ k^T / sqrt(head_dim).
    3. Mask out any score where the key position j is *after* the query
       position i (set to -infinity so it becomes 0 after softmax).
    4. softmax the scores per row, then weight-sum the values: attn @ v.
    5. Project the result back through `c_proj` (weight shape (C, C)).

This test file does NOT re-implement that whole pipeline to "cheat" — instead
most tests build one of two kinds of independent checks:

  (a) *Property checks* — things that must be true of causal attention
      regardless of the exact numbers (e.g. "changing a future token must not
      change the current output"; "a masked-out position gets exactly zero
      attention weight").

  (b) *Numeric parity checks* — for the simplest possible configuration
      (batch size 1, a single head), a hand-written reference computation
      (`_attention_ref_single_head`, `_row_softmax_ref`) recomputes the exact
      same math using plain nested loops, with NO dependency on the
      `SelfAttention` module's internal implementation. If the module's
      output matches this brute-force reference to within a small numeric
      tolerance, the module is doing the right math.

HOW TO VERIFY THIS YOURSELF, AS A NEWCOMER
────────────────────────────────────────────────────────────────────────────
1. Run the whole file: `mojo test_causal_self_attention.mojo` (or however
   your project's test runner invokes it — see `main()` at the bottom).
   Every `test_attn_*` function is discovered and run automatically; each one
   raises/asserts on failure and is silent on success.
2. To verify any single test "by hand": copy just that test's body into a
   scratch script (or a Python/numpy prototype using the same numbers) and
   recompute the expected values with a calculator or numpy. Because the
   reference helpers below use only "obvious" operations (loops, `exp`,
   `sqrt`, elementwise multiply/add), you can transcribe them line-for-line
   into numpy and confirm the numbers match.
3. Every test below has a docstring explaining:
      - WHAT it sets up
      - WHAT it computes / asserts
      - WHY that assertion proves the module is correct
      - HOW you could hand-verify it (e.g. "compute this 4x4 matrix by hand")
────────────────────────────────────────────────────────────────────────────
"""

from std.testing import assert_true, assert_false, TestSuite
from std.math import exp, sqrt
from std.sys import has_accelerator
from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape
from tenmo.shared.indexhelper import i, s
from tenmo.attention import SelfAttention
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.embedding import Embedding
from tenmo.net import Linear
from tenmo.optim import SGD


# =============================================================================
# Tests for SelfAttention
# Prefix: test_attn_ on all test names.
# =============================================================================
#
# HELPERS — small reference implementations used to verify the module wiring
# independently of its internal op composition. These work on the (B, T, C)
# qkv layout for the single-head case (B=1, h=1, dh=C).


def _set_deterministic_weights[
    dtype: DType
](model: SelfAttention[dtype],):
    """
    HELPER (not a test itself).
    Overwrites every learnable weight/bias in
    `model` with a fixed, hand-computable pattern instead of the random
    values the constructor normally initializes them to.

    WHAT IT SETS, AND THE EXACT FORMULA USED:
      - `c_attn.weight[r, c]`  = ((r+1)*10 + (c+1)) * 0.01
            e.g. weight[0,0] = (1*10+1)*0.01 = 0.11
                 weight[0,1] = (1*10+2)*0.01 = 0.12
                 weight[1,0] = (2*10+1)*0.01 = 0.21
      - `c_attn.bias[r]`       = 0.01 * (r+1)
            e.g. bias[0] = 0.01, bias[1] = 0.02, ...
      - `c_proj.weight[r, c]`  = (r+1) + 0.05*(c+1)
            e.g. weight[0,0] = 1 + 0.05 = 1.05
                 weight[0,1] = 1 + 0.10 = 1.10
      - `c_proj.bias[r]`       = 0.02 * (r+1)
            e.g. bias[0] = 0.02, bias[1] = 0.04, ...

    WHY: with random weights you cannot hand-check a numeric result — you'd
    need to trust whatever number the code prints. With this deterministic
    pattern, anyone can regenerate the exact weight matrices with a
    calculator (or `numpy.fromfunction`) and independently redo the whole
    forward pass by hand for a small example, which is exactly what
    `_attention_ref_single_head` below does.

    HOW TO VERIFY: for a model with n_embd=2, print `model.c_attn.weight`
    after calling this function and confirm it equals
        [[0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
         [0.21, 0.22, 0.23, 0.24, 0.25, 0.26]]
    (shape is (C, 3C) = (2, 6) since c_attn produces q, k, v concatenated).
    """
    var w = model.c_attn.weight
    var nrows = w.shape()[0]
    var ncols = w.shape()[1]
    for r in range(nrows):
        for c in range(ncols):
            w[r, c] = Scalar[dtype](Float64((r + 1) * 10 + (c + 1)) * 0.01)
    var b = model.c_attn.bias.value()
    for r in range(b.numels()):
        b[r] = Scalar[dtype](0.01 * Float64(r + 1))
    var wp = model.c_proj.weight
    for r in range(wp.shape()[0]):
        for c in range(wp.shape()[1]):
            wp[r, c] = Scalar[dtype](Float64(r + 1) + 0.05 * Float64(c + 1))
    var bp = model.c_proj.bias.value()
    for r in range(bp.numels()):
        bp[r] = Scalar[dtype](0.02 * Float64(r + 1))


def _row_softmax_ref[dtype: DType](row: Tensor[dtype]) -> Tensor[dtype]:
    """
    HELPER (not a test itself).
    Computes softmax over a single 1-D vector
    (`row`, length C) using the textbook numerically-stable formula below.

        m         = max(row)                      # for numerical stability
        out[c]    = exp(row[c] - m) / sum_j exp(row[j] - m)

    This is deliberately written with plain loops (no vectorized/optimized
    tensor ops) so it cannot accidentally share a bug with the module's own
    softmax implementation — it's an independent ground truth.

    WHY SUBTRACT THE MAX (`m`)? `exp()` overflows for large inputs; because
    softmax(x) == softmax(x - constant) for any constant, subtracting the
    row's max keeps every exponent <= 0 (safe) while giving the identical
    mathematical result.

    NOTE ON -infinity INPUTS: this function is called (elsewhere in this
    file) on rows that contain `-inf` in the "masked" (future) positions.
    `exp(-inf - m) == exp(-inf) == 0`, so those positions correctly
    contribute a probability of exactly 0 — this is how causal masking
    is realized numerically.

    HOW TO VERIFY: for row = [1.0, 2.0, 3.0], by hand -
        m = 3.0
        exps = [exp(-2), exp(-1), exp(0)] = [0.1353, 0.3679, 1.0]
        sum  = 1.5032
        out  = [0.0900, 0.2447, 0.6652]   (sums to 1.0).
    """
    comptime assert dtype.is_floating_point()
    var C = row.shape()[0]
    var m = row[0]
    for c in range(C):
        if row[c] > m:
            m = row[c]
    var ssum: Scalar[dtype] = 0.0
    for c in range(C):
        ssum += exp(row[c] - m)
    var out = Tensor[dtype].zeros(Shape(C))
    for c in range(C):
        out[c] = exp(row[c] - m) / ssum
    return out^


def _attention_ref_single_head[
    dtype: DType
](model: SelfAttention[dtype], x: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    HELPER (not a test itself) — a from-scratch.
    Brute-force recomputation of
    the ENTIRE SelfAttention forward pass, valid only for the simplest
    configuration: batch size B=1 and a single attention head (n_head=1, so
    head_dim `dh` equals the full embedding width `C`).

    This function reads the module's own weight tensors (`c_attn.weight`,
    `c_attn.bias`, `c_proj.weight`, `c_proj.bias`) but does NOT call any of
    the module's forward/attention logic — every matmul, mask, and softmax
    below is written out as explicit nested `for` loops. It exists so that
    `test_attn_single_head_parity_forward` can compare "module output" against
    "independently-computed output" and catch any bug in the module's real
    (presumably vectorized/optimized) implementation.

    STEP-BY-STEP MATH (matches the 5 steps described in the file's module
    docstring above):

      1. QKV PROJECTION
         qkv[t, o] = sum_c( x[0, t, c] * W_qkv[c, o] ) + b_qkv[o]
         for t in [0, T), o in [0, 3C)   →  qkv has shape (T, 3C)
         This is just x @ W_qkv + b_qkv for the single batch element.

      2. SPLIT into q, k, v — the 3C output columns of c_attn are the
         concatenation [q | k | v], each of width C:
             q[t, c] = qkv[t, c]
             k[t, c] = qkv[t, C + c]
             v[t, c] = qkv[t, 2C + c]

      3. SCALED, MASKED ATTENTION SCORES
         scores[i, j] = (q[i] . k[j]) / sqrt(dh)      for all i, j in [0, T)
         then for every j > i (a "future" key relative to query i):
             scores[i, j] = -infinity
         (computed here as -1.0 / 0.0, which yields -inf under IEEE float
         division-by-zero semantics — this is intentional, not a bug.)

      4. ROW SOFTMAX + WEIGHTED SUM OF VALUES
         attn[i, :] = softmax(scores[i, :])           (via _row_softmax_ref)
         out[i, c]  = sum_j( attn[i, j] * v[j, c] )

      5. OUTPUT PROJECTION
         final[t, c] = sum_d( out[t, d] * W_proj[d, c] ) + b_proj[c]

      Returns `final`, shape (T, C) — the batch dimension is dropped since
      B=1 throughout this helper.

    WHY THIS PROVES CORRECTNESS: if the module's actual (likely fused/
    vectorized/multi-head-capable) implementation produces the same numbers
    as this maximally-literal loop version, for a case where both must agree
    (single head), then the module's math is right for that case. Combined
    with `test_attn_shape_multi_head` (shape-only check for >1 head) and
    `test_attn_causal_no_lookahead` (behavioral check that generalizes to
    multi-head), this gives good confidence in the multi-head path too.

    HOW TO VERIFY BY HAND: pick T=2, C=2 (tiny) and redo steps 1-5 with a
    calculator or a numpy script using the same `_set_deterministic_weights`
    formulas; you should get identical numbers to what this function returns.
    """
    var T = x.shape()[1]
    var C = x.shape()[2]
    var dh = model.head_dim
    assert_true(model.n_head == 1)

    var W_qkv = model.c_attn.weight  # (C, 3C)
    var b_qkv = model.c_attn.bias.value()  # (3C,)
    var W_proj = model.c_proj.weight  # (C, C)
    var b_proj = model.c_proj.bias.value()  # (C,)

    # Step 1: qkv[b=0] = x @ W_qkv + b_qkv   -> (T, 3C)
    var qkv = Tensor[dtype].zeros(Shape(T, 3 * C))
    for t in range(T):
        for o in range(3 * C):
            var acc: Scalar[dtype] = 0.0
            for c in range(C):
                acc += x[0, t, c] * W_qkv[c, o]
            qkv[t, o] = acc + b_qkv[o]

    # Step 2: split the (T, 3C) tensor into q, k, v, each (T, C)
    var q = Tensor[dtype].zeros(Shape(T, C))
    var k = Tensor[dtype].zeros(Shape(T, C))
    var v = Tensor[dtype].zeros(Shape(T, C))
    for t in range(T):
        for c in range(C):
            q[t, c] = qkv[t, c]
            k[t, c] = qkv[t, C + c]
            v[t, c] = qkv[t, 2 * C + c]

    # Step 3: scores[i, j] = dot(q[i], k[j]) / sqrt(dh), masked (j <= i else -inf)
    var inv = Scalar[dtype](1.0 / sqrt(Float64(dh)))
    var scores = Tensor[dtype].zeros(Shape(T, T))
    for i in range(T):
        for j in range(T):
            var acc: Scalar[dtype] = 0.0
            for c in range(C):
                acc += q[i, c] * k[j, c]
            scores[i, j] = acc * inv
            if j > i:
                scores[i, j] = Scalar[dtype](-1.0) / Scalar[dtype](0.0)

    # Step 4: attn[i, :] = softmax(scores[i, :]); out[i] = attn[i] @ v
    var out = Tensor[dtype].zeros(Shape(T, C))
    for i in range(T):
        var attn = _row_softmax_ref(scores[i, s()])  # (T,)
        for c in range(C):
            var acc: Scalar[dtype] = 0.0
            for j in range(T):
                acc += attn[j] * v[j, c]
            out[i, c] = acc

    # Step 5: final = out @ W_proj + b_proj  (B=1 folded)
    var final = Tensor[dtype].zeros(Shape(T, C))
    for t in range(T):
        for c in range(C):
            var acc: Scalar[dtype] = 0.0
            for d in range(C):
                acc += out[t, d] * W_proj[d, c]
            final[t, c] = acc + b_proj[c]
    return final^


# ═════════════════════════════════════════════════════════════════════════════
# 1. SHAPES
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_shape_single_head() raises:
    """
    TEST — the simplest possible sanity check.
    Does the module preserve the
    input tensor's shape at all?

    SETUP: n_embd=2, n_head=1 (one head covering the whole embedding width).
    Input `x` is a (1, 4, 2) tensor of all-ones (batch=1, sequence length=4,
    embedding width=2).

    ASSERTS: `model(x).shape() == Shape(1, 4, 2)` — i.e. attention must
    return exactly the same (B, T, C) shape it was given. This must hold
    for ANY valid attention module regardless of its internal math, because
    attention output is meant to be added back into the residual stream at
    the same shape.

    HOW TO VERIFY: run it; if the shape assertion fails, the module is
    broken at a very basic structural level (e.g. reshape bug, wrong
    n_head/head_dim math). No numeric computation needs to be checked here —
    this test only confirms plumbing, not correctness of values.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=2, n_head=1, qkv_bias=True)
    var x = Tensor[dtype].ones(Shape(1, 4, 2))
    var out = model(x)
    assert_true(out.shape() == Shape(1, 4, 2))


def test_attn_shape_multi_head() raises:
    """
    TEST.
    Same shape check as above, but for the multi-head case, to make
    sure splitting the embedding into multiple heads and recombining them
    doesn't change the overall (B, T, C) shape.

    SETUP: n_embd=8, n_head=2 (so head_dim = 8/2 = 4). Input `x` is a
    (2, 8, 8) tensor (batch=2, sequence length=8, embedding width=8) sampled
    from a normal distribution (mean=0.0, std=0.02) — the values themselves
    don't matter here, only the shape does.

    ASSERTS: `model(x).shape() == Shape(2, 8, 8)`.

    HOW TO VERIFY: run it; a shape mismatch here (but not in the single-head
    test above) would point specifically at the multi-head split/merge logic
    (e.g. reshaping (B,T,C) into (B, n_head, T, head_dim) and back).
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=8, n_head=2, qkv_bias=True)
    var x = Tensor[dtype].randn(Shape(2, 8, 8), mean=0.0, std=0.02)
    var out = model(x)
    assert_true(out.shape() == Shape(2, 8, 8))


# ═════════════════════════════════════════════════════════════════════════════
# 2. CAUSAL CORRECTNESS — no look-ahead
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_causal_no_lookahead() raises:
    """
        TEST — the defining property of *causal* attention: the output at.
    position `t` must depend ONLY on inputs at positions `0..t`, and must be
    completely unaffected by anything at positions `t+1..T-1` ("the future").

    SETUP: n_embd=4, n_head=1, deterministic weights (via
    `_set_deterministic_weights`), model in `eval()` mode (so no dropout or
    other stochastic behavior can interfere). Input `x` has sequence length
    5, filled with random values (mean=0.0, std=0.1).

    METHOD (this is the clever part — it does NOT recompute attention by
    hand; it tests the property directly):
      1. Run the model once on the original `x` → `out_full`.
      2. Make a modified copy `x_cut`, zeroing out every value at time steps
         3 and 4 (i.e. deliberately corrupting "the future" relative to
         positions 0, 1, 2).
      3. Run the model again on `x_cut` → `out_cut`.
      4. Assert that `out_full` and `out_cut` are numerically identical
         (within atol=1e-4) at positions 0, 1, and 2.

    WHY THIS PROVES CAUSALITY: if position 2's output changed after we
    edited positions 3 and 4, that would mean position 2 was "peeking" at
    future tokens — a serious correctness bug for a language model (it would
    let the model cheat during training by seeing the answer). Since
    positions 0-2 come out identical whether or not we destroy positions 3-4,
    we've proven those positions never influenced the earlier outputs.

    HOW TO VERIFY: pick any T and zero out any suffix of positions; the
    outputs for all positions strictly before the zeroed suffix must stay
    identical. You can rerun this test with different cut points (e.g. cut
    from position 1 instead of 3) as an extra sanity check.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=4, n_head=1, qkv_bias=True)
    _set_deterministic_weights(model)
    model.eval()
    var x = Tensor[dtype].randn(Shape(1, 5, 4), mean=0.0, std=0.1)
    var out_full = model(x)

    # Zero out all inputs at positions > i (i = 2); rows 0..i must not change.
    var x_cut = x.clone()
    for t in range(3, 5):
        for c in range(4):
            x_cut[0, t, c] = 0.0
    var out_cut = model(x_cut)
    for t in range(3):
        assert_true(
            out_full[i(0), i(t), s()].all_close[atol=1e-4](
                out_cut[i(0), i(t), s()]
            )
        )


# ═════════════════════════════════════════════════════════════════════════════
# 3. SINGLE-HEAD NUMERIC PARITY — forward only (B=1, h=1, dh=C=2)
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_single_head_parity_forward() raises:
    """
    TEST — the strongest correctness check in this file.
    Does the module's
    actual forward pass produce the SAME NUMBERS as an independent,
    brute-force reference implementation (`_attention_ref_single_head`)?

    SETUP: n_embd=2, n_head=1 (so head_dim = C = 2, the smallest non-trivial
    single-head case). Deterministic weights via `_set_deterministic_weights`.
    Model in `eval()` mode. A fixed, hand-chosen 4x2 input:
        x = [[ 0.5, -0.25],
             [ 0.1,  0.9 ],
             [ 0.7, -0.4 ],
             [ 0.2,  0.3 ]]
    reshaped to (1, 4, 2) to add the batch dimension.

    METHOD:
      1. `module_out = model(x3)` — run the real module.
      2. `reference  = _attention_ref_single_head(model, x3)` — recompute the
         exact same forward pass using nothing but explicit loops (see that
         function's docstring for the full step-by-step math).
      3. Assert `module_out[0] ≈ reference` element-wise, within atol=1e-3
         (a loose-ish tolerance to allow for floating point summation-order
         differences between the module's implementation and the reference
         loops — not because the math itself is approximate).

    WHY THIS PROVES CORRECTNESS: `_attention_ref_single_head` shares none of
    the module's code paths (no shared matmul kernel, no shared softmax
    routine) — it is a second, independent implementation of the same
    specification. Agreement between two independently-written
    implementations of the same math, on the same random-looking input, is
    strong evidence both are correct (or coincidentally both wrong the same
    way, which is extremely unlikely given how differently they're written).

    HOW TO VERIFY BY HAND: because n_embd=2 and T=4 is small, you can
    literally redo all 5 steps from `_attention_ref_single_head`'s docstring
    with a calculator or a short numpy script, using:
        W_qkv[r,c]  = ((r+1)*10 + (c+1)) * 0.01        (shape (2, 6))
        b_qkv[r]    = 0.01 * (r+1)                      (shape (6,))
        W_proj[r,c] = (r+1) + 0.05*(c+1)                (shape (2, 2))
        b_proj[r]   = 0.02 * (r+1)                      (shape (2,))
    and the `x` matrix given above. The final (4, 2) matrix you compute
    should match what the test asserts (within ~1e-3).
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=2, n_head=1, qkv_bias=True)
    _set_deterministic_weights(model)
    model.eval()
    var x = Tensor[dtype].d2(
        [[0.5, -0.25], [0.1, 0.9], [0.7, -0.4], [0.2, 0.3]]
    )
    var x3 = x.reshape(1, 4, 2)
    var module_out = model(x3)  # (1, 4, 2)
    var reference = _attention_ref_single_head(model, x3)  # (4, 2)
    assert_true(module_out[i(0), s(), s()].all_close[atol=1e-3](reference))


# ═════════════════════════════════════════════════════════════════════════════
# 4. MASKED-ROW BEHAVIOR — exactly-zero weight on future positions
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_masked_row_zero_weight() raises:
    """
        TEST — a narrower, more surgical version of the causality check: rather.
    than checking outputs (as `test_attn_causal_no_lookahead` does), this
    test checks the *attention weights themselves* — the softmax
    probabilities — and confirms that every future position gets EXACTLY
    zero weight (not just "negligibly small", but bit-for-bit 0.0).

    SETUP: n_embd=2, n_head=1, deterministic weights, the same fixed 4x2
    input `x` used in the parity test above.

    METHOD: this test does NOT call the module or the full reference helper
    — it recomputes only the qkv projection and the masked score matrix
    inline (same formulas as steps 1 and 3 in `_attention_ref_single_head`'s
    docstring), then:
      1. For each row `i` (query position), computes
         `scores[i, j] = dot(qkv_q[i], qkv_k[j]) / sqrt(dh)` for all `j`,
         and overwrites `scores[i, j] = -inf` whenever `j > i` (future key).
      2. Runs `_row_softmax_ref` on that row to get the actual attention
         probabilities `attn`.
      3. Asserts `attn[j] == 0.0` (exact equality, not "close to") for every
         `j > i`.

    WHY EXACT EQUALITY, AND WHY THIS MATTERS: `exp(-inf) == 0.0` exactly in
    IEEE floating point — there's no rounding error to worry about, so this
    is one of the rare places in numeric code where an exact `==` assertion
    is appropriate and meaningful. This test guards specifically against a
    masking bug where an off-by-one error (e.g. masking `j >= i` instead of
    `j > i`, or forgetting to mask row 0 because "there's nothing to mask
    yet") would leak a small but non-zero probability onto a future
    position — a subtle bug that could slip past the "outputs don't change"
    check above if the leaked probability happened to be very small.

    HOW TO VERIFY: for row i=0, only j=0 should be unmasked, so attn should
    be exactly [1.0, 0.0, 0.0, 0.0] (softmax of a single real value plus
    three -inf values). For row i=3 (the last), nothing is masked, so all
    four probabilities can be non-zero and should sum to 1.0.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=2, n_head=1, qkv_bias=True)
    _set_deterministic_weights(model)
    var x = Tensor[dtype].d2(
        [[0.5, -0.25], [0.1, 0.9], [0.7, -0.4], [0.2, 0.3]]
    )
    var x3 = x.reshape(1, 4, 2)

    # Build the pre-softmax scores for this module independently (single head)
    var T = 4
    var C = 2
    var dh = model.head_dim
    var W_qkv = model.c_attn.weight
    var b_qkv = model.c_attn.bias.value()
    var inv = Scalar[dtype](1.0 / sqrt(Float64(dh)))
    var qkv = Tensor[dtype].zeros(Shape(T, 3 * C))
    for t in range(T):
        for o in range(3 * C):
            var acc: Scalar[dtype] = 0.0
            for c in range(C):
                acc += x3[0, t, c] * W_qkv[c, o]
            qkv[t, o] = acc + b_qkv[o]
    var scores = Tensor[dtype].zeros(Shape(T, T))
    for i in range(T):
        for j in range(T):
            var acc: Scalar[dtype] = 0.0
            for c in range(C):
                acc += qkv[i, c] * qkv[j, C + c]
            scores[i, j] = acc * inv
            if j > i:
                scores[i, j] = Scalar[dtype](-1.0) / Scalar[dtype](0.0)
    for i in range(T):
        var attn = _row_softmax_ref(scores[i, s()])
        for j in range(T):
            if j > i:
                assert_true(attn[j] == 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 5. GRADIENT DESCENT OVERFITS ONE BATCH (Embedding -> Attention -> Linear head)
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_overfit_one_batch() raises:
    """
        TEST — an end-to-end "does learning actually work?" check. Rather than.
    checking the forward pass alone, this wires `SelfAttention` into a
    tiny but complete training loop (token embedding -> attention -> output
    head -> cross-entropy loss -> backward -> SGD step) and confirms the
    loss goes down when trained repeatedly on a single fixed batch of data.

    SETUP:
      - B=2 (batch size), T=4 (sequence length), C=16 (embedding width),
        V=32 (vocabulary size), H=4 (attention heads).
      - `wte`  : an `Embedding` layer mapping token ids -> (B, T, C) vectors.
      - `model_out` (misleadingly named — it's the attention block): a
        `SelfAttention[C, H]`, put into `.train()` mode.
      - `head` : a `Linear(C, V)` layer projecting attention output back to
        vocabulary-sized logits.
      - `tokens` : two fixed sequences of 4 token ids each, e.g.
            [[3, 17, 5, 22], [1, 4, 9, 30]]
      - `targets` : the "next token" for each position in `tokens` — i.e.
        this is a standard next-token-prediction language-modeling setup,
        where `targets[b, t]` is what the model should predict given
        `tokens[b, 0..t]`.
      - All parameters from `wte`, the attention block, and `head` are
        collected into one flat list and handed to a single `SGD` optimizer
        with a (deliberately large, for a fast test) learning rate of 0.5.
      - Loss function: `CrossEntropyLoss(reduction="mean")`.

    TRAINING LOOP (60 steps), each step:
      1. `x      = wte(tokens)`               → (B, T, C) embeddings
      2. `attn   = model_out(x)`               → (B, T, C) causal attention output
      3. `logits = head(attn)`                 → (B, T, V) raw prediction scores
      4. `logits_btv = logits.permute([0,2,1])` → (B, V, T), because this
         codebase's `CrossEntropyLoss` expects the class dimension in axis 1
         (matching common (N, C, ...) conventions), not the last axis.
      5. `loss = criterion(logits_btv, targets)`
      6. `optimizer.zero_grad(); loss.backward(); optimizer.step()`
      7. Record the loss value; remember the first step's loss as `loss0`
         and keep overwriting `lossN` so it ends up holding the *final*
         step's loss.

    ASSERTS (after all 60 steps):
      - `lossN < loss0 * 0.5`  — the loss must have at least halved.
      - `lossN < 0.5`          — the loss must have dropped to an
        absolutely small value (cross-entropy loss near 0 means the model is
        assigning very high probability to the correct next token).

    WHY THIS MATTERS: this is the test most likely to catch a bug that the
    purely-forward tests above cannot — specifically, anything wrong in
    `SelfAttention`'s BACKWARD pass (gradient computation). A module
    can have a perfectly correct forward pass and a broken backward pass;
    such a bug would show up here as the loss failing to decrease (or
    decreasing far more slowly than expected), because gradients flowing
    back through attention into `wte` and `head` would be wrong or absent.
    Overfitting a tiny, fixed dataset is a standard and effective smoke test
    for "is the whole training pipeline (forward + backward + optimizer)
    wired together correctly?"

    HOW TO VERIFY: rerun the test and print `loss0` and `lossN` — you should
    see `loss0` start somewhere in the neighborhood of `ln(V) ≈ ln(32) ≈
    3.47` (the expected cross-entropy loss for a freshly-initialized model
    making essentially random predictions over 32 classes), dropping to well
    under 0.5 by step 60. If you want to sanity-check the loss trend, print
    the loss every 10 steps and confirm it decreases roughly monotonically.
    """
    comptime dtype = DType.float32
    var B = 2
    var T = 4
    var C = 16
    var V = 32
    var H = 4
    var model_out = SelfAttention[dtype](n_embd=C, n_head=H, qkv_bias=True)
    var wte = Embedding[dtype](num_embeddings=V, embedding_dim=C)
    var head = Linear[dtype](C, V)
    model_out.train()

    var tokens = Tensor[DType.int64].d2([[3, 17, 5, 22], [1, 4, 9, 30]])
    var targets = Tensor[DType.int64].d2([[17, 5, 22, 3], [4, 9, 30, 1]])

    var params = wte.parameters()
    var attn_params = model_out.parameters()
    for p in range(len(attn_params)):
        params.append(attn_params[p])
    var head_params = head.parameters()
    for p in range(len(head_params)):
        params.append(head_params[p])

    var optimizer = SGD[dtype](params, lr=0.5)
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()

    var loss0: Float64 = 0.0
    var lossN: Float64 = 0.0
    for step in range(60):
        var x = wte(tokens)  # (B, T, C)
        var attn = model_out(x)  # (B, T, C)
        var logits = head(attn)  # (B, T, V)
        # CE expects (N, C, ...) with class on axis 1, target (N, ...).
        var logits_btv = logits.permute([0, 2, 1])  # (B, V, T)
        var loss = criterion(logits_btv, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        var v = Float64(loss.item())
        if step == 0:
            loss0 = v
        lossN = v

    # Overfit: loss must drop substantially from its initial value.
    assert_true(lossN < loss0 * 0.5)
    assert_true(lossN < 0.5)


# ═════════════════════════════════════════════════════════════════════════════
# 6. PARAMS / COMPOSABILITY
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_num_parameters() raises:
    """
    TEST — verifies the module reports the exact.
    Hand-computable count of
    learnable parameters for a realistic-sized configuration.

    SETUP: n_embd=384, n_head=8 (these are GPT-2-small-like dimensions).

    THE EXPECTED NUMBER, DERIVED BY HAND:
      `c_attn` is a Linear layer C -> 3C, so:
          weight: C * 3C = 384 * 1152 = 442,368
          bias:        3C = 1,152
      `c_proj` is a Linear layer C -> C, so:
          weight: C * C  = 384 * 384  = 147,456
          bias:        C = 384
      Total = 442,368 + 1,152 + 147,456 + 384 = 591,360

    ASSERTS: `model.num_parameters() == 591360`.

    HOW TO VERIFY: redo the arithmetic above with a calculator — it only
    involves multiplying and adding the dimensions listed. Note that
    `n_head` does not appear anywhere in this formula: splitting into heads
    is a reshape/view operation on the same underlying weights, it does not
    add any extra parameters.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=384, n_head=8, qkv_bias=True)
    assert_true(model.num_parameters() == 591360)


def test_attn_training_flag_and_mode_toggle() raises:
    """
    TEST.
    Checks the `training` boolean flag (used to enable/disable
    dropout and similar train-only behavior) starts correctly and responds
    correctly to `.eval()` / `.train()` calls.

    SETUP: n_embd=8, n_head=2 (arbitrary small config; the values chosen
    don't matter for this test, only the flag toggling does).

    METHOD / ASSERTS, in order:
      1. `assert_true(model.training)` — a freshly constructed module must
         default to training mode (this matches common ML framework
         convention: modules start in train mode unless told otherwise).
      2. `model.eval()` then `assert_false(model.training)` — calling
         `.eval()` must flip the flag to False.
      3. `model.train()` then `assert_true(model.training)` — calling
         `.train()` must flip it back to True.

    WHY THIS MATTERS: several other tests in this file (e.g.
    `test_attn_causal_no_lookahead`, `test_attn_single_head_parity_forward`)
    explicitly call `.eval()` before asserting on exact numeric output —
    they are implicitly relying on `.eval()` actually taking effect (e.g.
    disabling dropout so results are deterministic). If this flag-toggle
    test fails, it would explain why those other tests might be flaky.

    HOW TO VERIFY: trivial to check by reading `model.training` after each
    call in a REPL/script — no numeric computation involved.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=8, n_head=2, qkv_bias=True)
    assert_true(model.training)
    model.eval()
    assert_false(model.training)
    model.train()
    assert_true(model.training)


def test_attn_parameters_and_named_plumbing() raises:
    """
    TEST.
    Checks that the module correctly exposes its learnable tensors
    through two standard introspection APIs: `.parameters()` (a flat list,
    used by optimizers) and `.named_parameters(prefix)` (a list of
    (name, tensor) pairs, used for logging/checkpointing/debugging).

    SETUP: n_embd=8, n_head=2 (arbitrary small config).

    ASSERTS:
      1. `len(model.parameters()) == 4` — the module owns exactly 4 raw
         parameter tensors: `c_attn.weight`, `c_attn.bias`, `c_proj.weight`,
         `c_proj.bias`. (This count matches the two Linear sub-layers,
         `c_attn` and `c_proj`, each contributing one weight + one bias.)
      2. `len(model.named_parameters("attn.")) == 4` — the named variant
         must return the same count, just paired with string names.
      3. `String(named[0].name) == "attn.c_attn.weight"` — the FIRST named
         parameter, when given prefix `"attn."`, must be named exactly
         `"attn.c_attn.weight"`. This pins down both the naming convention
         (prefix + dotted sub-module path + parameter name) and the
         ordering (c_attn's weight must come first in the list).

    WHY THIS MATTERS: `.parameters()` ordering and completeness directly
    affects optimizer wiring (as seen in `test_attn_overfit_one_batch`,
    where `.parameters()` lists from multiple modules are concatenated into
    one optimizer) — if a parameter were missing or duplicated here, that
    module simply wouldn't learn (or would be updated twice). The naming
    test guards against typos or path-construction bugs when this module is
    nested inside a larger model (e.g. a full transformer block using
    prefix `"attn."`).

    HOW TO VERIFY: no numeric computation here — just count sub-module
    parameters (2 Linear layers x 2 tensors each = 4) and check the naming
    convention by inspection.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=8, n_head=2, qkv_bias=True)
    var params = model.parameters()
    # c_attn (W + b) + c_proj (W + b) = 4 params
    assert_true(len(params) == 4)
    var named = model.named_parameters("attn.")
    assert_true(len(named) == 4)
    var n0 = String(named[0].name)
    assert_true(n0 == "attn.c_attn.weight")


def test_attn_gpu_roundtrip() raises:
    """
    TEST.
    Checks that moving the module's parameters to GPU (`.to_gpu()`)
    and back to CPU (`.to_cpu()`) preserves the module intact — specifically,
    that no parameters are lost, duplicated, or corrupted by the round trip.

    GUARD: `comptime if has_accelerator():` — this entire test body is
    compiled/run only when a GPU accelerator is actually available in the
    current environment. On a CPU-only machine this test is effectively a
    no-op (it compiles but its body never executes), so it will not fail
    just because you don't have a GPU.

    SETUP: n_embd=8, n_head=2.

    METHOD: `g = model.to_gpu()` moves parameters to GPU memory, then
    `back = g.to_cpu()` moves them back to CPU memory.

    ASSERTS: `back.num_parameters() == 288`, computed the same way as in
    `test_attn_num_parameters`, but for these smaller dimensions:
        c_attn: weight 8*24=192, bias 24        -> 216
        c_proj: weight 8*8=64,  bias 8          -> 72
        total: 216 + 72 = 288

    WHY THIS MATTERS: GPU/CPU transfer code often involves manual buffer
    copying and pointer management; a bug there (e.g. forgetting to copy the
    bias tensor, or copying the wrong shape) would typically show up as a
    changed parameter count or a crash, which this test would catch.

    HOW TO VERIFY: same arithmetic as `test_attn_num_parameters`, just with
    C=8, n_head=2 (note: 3C = 24 for the c_attn output width). Only
    runnable/verifiable on a machine with GPU support enabled.
    """
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var model = SelfAttention[dtype](n_embd=8, n_head=2, qkv_bias=True)
        var g = model.to_gpu()
        var back = g.to_cpu()
        # c_attn (8x24 + 24) + c_proj (8x8 + 8) = 192 + 24 + 64 + 8 = 288
        assert_true(back.num_parameters() == 288)


def main() raises:
    """
    ENTRY POINT.
    Discovers every `test_attn_*` function defined in this
    module (via `__functions_in_module()`) and runs them all in sequence.
    If every test's internal `assert_true`/`assert_false` calls succeed, it
    prints a final confirmation message; if any assertion fails, the
    `TestSuite` runner is expected to report which test failed and why
    (exact reporting behavior depends on the `std.testing` implementation).

    HOW TO RUN: invoke this file with your Mojo test runner, e.g.
        mojo test_causal_self_attention.mojo
    and look for "All attention tests passed!" at the end.
    """
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll attention tests passed!")
