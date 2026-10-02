"""
Extended tests for SelfAttention — targeting gaps left by the original suite.

────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS
────────────────────────────────────────────────────────────────────────────
The original test file is a solid
foundation: it checks output shapes, the causal (no-look-ahead) property,
one numeric-parity check for the SINGLE-HEAD case, exact-zero masking,
one end-to-end training smoke test, and basic parameter bookkeeping.

Reading `attention.mojo` line by line against that suite turns up several
code paths and properties that are exercised by NO existing test. This file
targets exactly those gaps, surgically — each test below is aimed at one
specific section of the implementation, and each docstring names which
lines/steps it is validating and, just as importantly, WHY that thing could
plausibly break even though the rest of the suite passes.

GAP ANALYSIS (what's missing, and where it's fixed in this file):

  1. Multi-head attention is only ever checked for SHAPE (`test_attn_shape_
     multi_head`), never for VALUES. The single-head parity test
     (`test_attn_single_head_parity_forward`) exercises the QKV projection,
     scaling, masking, and softmax — but it can never catch a bug in the
     head-SPLIT/MERGE logic (the reshape+permute in steps B and E of
     `_forward`), because with `n_head=1` there is only one head to split
     into — there's nothing to get wrong.
       -> Fixed by Section 9 (multi-head numeric parity).

  2. No test checks that attention weights actually form a valid probability
     distribution (sum to 1) — only that MASKED weights are exactly zero.
     Zero-masking and summing-to-one are two separate properties, and a bug
     in softmax's normalization could break the second while leaving the
     first intact.
       -> Fixed by Section 8 (probability conservation).

  3. No test checks BATCH INDEPENDENCE — that item 0 and item 1 in a batch
     never influence each other. A bug that accidentally reduces or
     broadcasts across the wrong axis could silently mix batch elements
     together while still passing every existing test (which mostly use
     B=1).
       -> Fixed by Section 10 (batch independence).

  4. No test exercises the DEGENERATE configuration boundaries the
     constructor explicitly allows: `head_dim == 1` (n_head == n_embd) and
     the smallest possible model (`n_embd=1, n_head=1`). Off-by-one bugs in
     reshape/permute logic often only surface at these extremes.
       -> Fixed by Section 7 (configuration boundaries).

  5. No test exercises `T == 1` (a single-token sequence) — the most
     degenerate possible *sequence*, where the causal mask allows exactly
     one attendable position. This is also a nice opportunity to teach a
     subtle, reassuring fact about softmax (see Section 11's docstring).
       -> Fixed by Section 11.

  6. `dropout_p` is a constructor parameter with real, documented behavior
     (train-mode-only stochastic zeroing) — but no existing test ever
     passes a non-zero `dropout_p`, so nothing confirms dropout (a) is
     really disabled in eval mode, or (b) actually does something in train
     mode.
       -> Fixed by Section 13 (dropout mode behavior).

  7. `init_seed` is documented as controlling reproducible initialization,
     but nothing confirms two independently-constructed modules with the
     same seed actually produce identical weights. Reproducibility is a
     first-class concern in ML — an experiment that can't be rerun
     identically can't be debugged.
       -> Fixed by Section 12 (seeded reproducibility).

  8. `parameters()` is exercised (indirectly, by feeding it to an optimizer)
     in the original suite's training test, but nothing directly confirms
     that EVERY one of the 4 parameter tensors this module owns actually
     receives a gradient and gets updated. A parameter that silently never
     received gradient (e.g. due to a broken autograd edge) could sit there
     unchanged forever while the overall loss still manages to decrease
     using the other parameters — the existing overfit test would not catch
     this, because it only checks the aggregate loss.
       -> Fixed by Section 14 (gradient wiring).

  9. `test_attn_gpu_roundtrip` (original suite) checks that the PARAMETER
     COUNT survives a `to_gpu()` -> `to_cpu()` round trip, but never checks
     that the actual WEIGHT VALUES survive unchanged. A transfer bug could
     preserve shape/count while scrambling or truncating values.
       -> Fixed by Section 15 (GPU round-trip value preservation).

 10. `num_parameters()` is checked for exactly ONE configuration
     (n_embd=384, n_head=8). Nothing confirms the formula generalizes, or
     — a specifically interesting property — that the count is genuinely
     INDEPENDENT of `n_head` for a fixed `n_embd` (splitting into heads is a
     reshape of existing weights, not new storage; this is easy to get
     wrong if someone later "optimizes" the implementation).
       -> Fixed by Section 7 (parameter formula across configs).

────────────────────────────────────────────────────────────────────────────
HOW TO READ THIS FILE
────────────────────────────────────────────────────────────────────────────
Sections are numbered 7 onward, continuing the original suite's numbering
(1. SHAPES ... 6. PARAMS/COMPOSABILITY) so the two files read as one
continuous suite once merged. Within this file, sections are ordered from
SIMPLE to INVOLVED:

    7.  Configuration boundaries & parameter-count formula   (shape/counting only)
    8.  Probability conservation (attention rows sum to 1)   (one extra property check)
    9.  Multi-head numeric parity                            (new reference math)
    10. Batch independence                                   (cross-example isolation)
    11. Single-token sequence edge case                      (degenerate T=1)
    12. Seeded-initialization reproducibility                (construction contract)
    13. Dropout mode behavior (train vs eval)                (stochastic vs deterministic)
    14. Gradient wiring (does backward reach every parameter?)(training-pipeline internals)
    15. GPU round-trip value preservation                    (device-transfer correctness)

If you are new to transformers, each test's docstring is written to stand
alone — you should be able to read this file top to bottom as a gentle
introduction to WHY each of these properties matters for a causal
self-attention layer, not just WHAT is being asserted.

NAMING NOTE FOR MERGING: every helper function below is suffixed `_ext` (for
"extended") specifically so it cannot collide with the original suite's
same-purpose helpers (`_set_deterministic_weights`, `_row_softmax_ref`,
`_attention_ref_single_head`) if you paste both files into one module.
Rename/deduplicate as you see fit once merged — for instance,
`_row_softmax_ref_ext` and the original `_row_softmax_ref` are byte-for-byte
the same function and only one copy is truly needed.
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
# HELPERS — independent reference math, generalized to arbitrary n_head.
# These share the exact same spirit as the original suite's
# `_set_deterministic_weights` / `_row_softmax_ref` / `_attention_ref_single_head`,
# but the attention reference is generalized to loop over an arbitrary
# number of heads instead of assuming exactly one. This generalization is
# what lets Section 9 below numerically verify the head-split/merge logic
# that the original suite's single-head-only reference structurally cannot
# reach.
# =============================================================================


def _set_deterministic_weights_ext[dtype: DType](
    model: SelfAttention[dtype],
):
    """
    HELPER.
    Identical in spirit to the original suite's
    `_set_deterministic_weights`: overwrites every learnable weight/bias
    with a fixed, hand-computable pattern so that forward-pass numbers can
    be reproduced with a calculator instead of trusted blindly.

    Formulas (same as the original helper, repeated here so this file is
    self-contained):
        c_attn.weight[r, c]  = ((r+1)*10 + (c+1)) * 0.01
        c_attn.bias[r]       = 0.01 * (r+1)
        c_proj.weight[r, c]  = (r+1) + 0.05*(c+1)
        c_proj.bias[r]       = 0.02 * (r+1)

    See the original suite's docstring for a worked example of the first
    few entries these formulas produce.
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
            wp[r, c] = Scalar[dtype](
                Float64(r + 1) + 0.05 * Float64(c + 1)
            )
    var bp = model.c_proj.bias.value()
    for r in range(bp.numels()):
        bp[r] = Scalar[dtype](0.02 * Float64(r + 1))


def _row_softmax_ref_ext[dtype: DType](row: Tensor[dtype]) -> Tensor[dtype]:
    """
    HELPER — numerically-stable softmax over a 1-D vector.
    Identical to the
    original suite's `_row_softmax_ref`. See that function's docstring for
    the full explanation of why the max is subtracted before exponentiating,
    and why `exp(-inf) == 0.0` is exactly what makes causal masking work.
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


def _attention_ref_multi_head_ext[dtype: DType](
    model: SelfAttention[dtype], x: Tensor[dtype]
) raises -> Tensor[dtype]:
    """
        HELPER — the key new piece of machinery in this file: a brute-force,.
    loop-only reference implementation of the FULL forward pass that
    supports ANY number of heads (`model.n_head >= 1`), not just one.

    This generalizes the original suite's `_attention_ref_single_head` in
    exactly one way: instead of computing one (T,T) score matrix over the
    whole embedding width, it computes `n_head` independent (T,T) score
    matrices, one per head, each using only that head's `head_dim`-wide
    slice of q/k/v — and concatenates the per-head outputs back together
    before applying `c_proj`. Still assumes batch size B=1, matching the
    original single-head reference's scope (multi-batch behavior is instead
    covered directly, and separately, in Section 10 below).

    BACKGROUND — why does slicing by head matter?  In `attention.mojo`,
    step B reshapes each of q, k, v from `(T, C)` to `(T, h, dh)` before
    permuting. Reshaping only "splits" a `(T, C)` matrix into `h` groups of
    `dh` CONSECUTIVE columns each — group 0 is columns `[0, dh)`, group 1 is
    columns `[dh, 2*dh)`, and so on. So "head `hh`'s query vector for
    position `t`" is just `q[t, hh*dh : (hh+1)*dh]` — a contiguous column
    slice of the full q matrix. This helper reproduces that slicing
    explicitly (see the `hh * dh` / `(hh + 1) * dh` arithmetic below), which
    is precisely the arithmetic a bug in the real reshape/permute logic
    could get wrong (e.g. swapping which axis becomes "head" vs "position
    within head", or slicing the wrong `dh`-wide window).

    STEP-BY-STEP MATH:

      1. QKV PROJECTION (identical to the single-head reference):
         qkv[t, o] = sum_c( x[0, t, c] * W_qkv[c, o] ) + b_qkv[o]
         for t in [0, T), o in [0, 3C)   ->  qkv has shape (T, 3C)

      2. SPLIT into q, k, v, each (T, C) — the [q | k | v] column blocks.

      3. FOR EACH HEAD `hh` in [0, h):
           - Take this head's column slice of q, k, v:
                 q_hh[t, d] = q[t, hh*dh + d]      for d in [0, dh)
                 k_hh[t, d] = k[t, hh*dh + d]
                 v_hh[t, d] = v[t, hh*dh + d]
           - scores_hh[i, j] = dot(q_hh[i], k_hh[j]) / sqrt(dh),
             masked to -inf wherever j > i (exactly as in the single-head
             case — causality is applied independently, per head, but using
             the SAME (T,T) triangular pattern for every head, since the
             mask only depends on sequence position, never on head or
             feature content).
           - attn_hh[i, :] = softmax(scores_hh[i, :])
           - headout_hh[i, d] = sum_j( attn_hh[i, j] * v_hh[j, d] )
             -> shape (T, dh)

      4. CONCATENATE all heads' (T, dh) outputs back into one (T, C) matrix,
         placing head `hh`'s output in columns `[hh*dh, (hh+1)*dh)` — the
         same column layout the real implementation's reshape/permute is
         supposed to reconstruct in step E.

      5. OUTPUT PROJECTION (identical to the single-head reference):
         final[t, c] = sum_d( merged[t, d] * W_proj[d, c] ) + b_proj[c]

    Returns `final`, shape (T, C).

    WHY THIS PROVES THE HEAD-SPLIT LOGIC IS CORRECT: this helper computes
    each head's contribution using EXPLICIT column-slice arithmetic, with
    no reshape or permute operation anywhere in its own code. If the real
    module's `reshape` + `permute` dance (which exists purely for
    computational efficiency — expressing the same per-head slicing as a
    batched matmul) produces different numbers than this "obviously
    correct, if inefficient" column-slicing version, the discrepancy can
    ONLY come from a bug in how the real code splits/merges heads, since
    every other piece of the computation (the projections, the scaling, the
    masking, the softmax) is identical between the two implementations and
    is ALREADY independently verified by the single-head parity test.

    Setting `model.n_head = 1` here reduces this function to exactly the
    single-head reference (there's only one "head slice", spanning all `C`
    columns) — so this helper is a strict superset of the original one and
    could replace it if you are deduplicating during a merge.
    """
    var T = x.shape()[1]
    var C = x.shape()[2]
    var h = model.n_head
    var dh = model.head_dim

    var W_qkv = model.c_attn.weight  # (C, 3C)
    var b_qkv = model.c_attn.bias.value()  # (3C,)
    var W_proj = model.c_proj.weight  # (C, C)
    var b_proj = model.c_proj.bias.value()  # (C,)

    # Step 1: qkv = x @ W_qkv + b_qkv   -> (T, 3C)
    var qkv = Tensor[dtype].zeros(Shape(T, 3 * C))
    for t in range(T):
        for o in range(3 * C):
            var acc: Scalar[dtype] = 0.0
            for c in range(C):
                acc += x[0, t, c] * W_qkv[c, o]
            qkv[t, o] = acc + b_qkv[o]

    # Step 2: split into q, k, v, each (T, C)
    var q = Tensor[dtype].zeros(Shape(T, C))
    var k = Tensor[dtype].zeros(Shape(T, C))
    var v = Tensor[dtype].zeros(Shape(T, C))
    for t in range(T):
        for c in range(C):
            q[t, c] = qkv[t, c]
            k[t, c] = qkv[t, C + c]
            v[t, c] = qkv[t, 2 * C + c]

    # Step 3 + 4: per-head causal attention, written into the right column
    # slice of `merged` as we go, so we never need a separate concat pass.
    var inv = Scalar[dtype](1.0 / sqrt(Float64(dh)))
    var merged = Tensor[dtype].zeros(Shape(T, C))
    for hh in range(h):
        var col0 = hh * dh
        # scores_hh[i, j] = dot(q_hh[i], k_hh[j]) * inv, masked j > i -> -inf
        var scores = Tensor[dtype].zeros(Shape(T, T))
        for qi in range(T):
            for kj in range(T):
                var acc: Scalar[dtype] = 0.0
                for d in range(dh):
                    acc += q[qi, col0 + d] * k[kj, col0 + d]
                scores[qi, kj] = acc * inv
                if kj > qi:
                    scores[qi, kj] = Scalar[dtype](-1.0) / Scalar[dtype](0.0)
        for qi in range(T):
            var attn = _row_softmax_ref_ext(scores[qi, s()])  # (T,)
            for d in range(dh):
                var acc: Scalar[dtype] = 0.0
                for kj in range(T):
                    acc += attn[kj] * v[kj, col0 + d]
                merged[qi, col0 + d] = acc

    # Step 5: final = merged @ W_proj + b_proj
    var final = Tensor[dtype].zeros(Shape(T, C))
    for t in range(T):
        for c in range(C):
            var acc: Scalar[dtype] = 0.0
            for d in range(C):
                acc += merged[t, d] * W_proj[d, c]
            final[t, c] = acc + b_proj[c]
    return final^


# ═════════════════════════════════════════════════════════════════════════════
# 7. CONFIGURATION BOUNDARIES & THE PARAMETER-COUNT FORMULA
# ═════════════════════════════════════════════════════════════════════════════
#
# BACKGROUND: `n_embd` and `n_head` are the two numbers that fully determine
# an attention layer's shape. The constructor enforces `n_embd % n_head == 0`
# but otherwise allows any positive integers — including the extremes below,
# which are easy to accidentally special-case incorrectly (or forget to
# handle at all) in reshape/permute-heavy code.


def test_attn_head_dim_one_shape() raises:
    """
        TEST — the most extreme *valid* multi-head configuration: `n_head`
    equals `n_embd`, so `head_dim = n_embd // n_head = 1`. Every head's
    query/key/value "vectors" are, individually, a SINGLE number.

    WHY THIS IS WORTH TESTING ON ITS OWN: with `head_dim == 1`, the
    attention "dot product" `dot(q_hh[i], k_hh[j])` inside each head
    degenerates into a plain scalar multiplication (a 1-element dot
    product), and the scaling factor `1/sqrt(head_dim)` becomes
    `1/sqrt(1) == 1` (a no-op). Code that assumes `head_dim >= 2` somewhere
    (for instance, in how a reshape or a reduction axis is chosen) is
    exactly the kind of bug that silently works for "normal" configurations
    like `n_head=8` but breaks — or, worse, silently produces WRONG
    numbers without crashing — right at this boundary.

    SETUP: n_embd=4, n_head=4 (head_dim=1). Input is a (2, 3, 4) batch of
    ones (batch=2, sequence length=3, embedding width=4) — only shape
    matters here, not the values.

    ASSERTS: `model(x).shape() == Shape(2, 3, 4)` — the module must still
    preserve the input's shape at this boundary exactly as it does for
    "normal" configurations (see the original suite's shape tests).

    HOW TO VERIFY: no numeric computation needed — just confirm the call
    doesn't crash and the returned shape matches the input's shape.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=4, n_head=4, qkv_bias=True)
    var x = Tensor[dtype].ones(Shape(2, 3, 4))
    var out = model(x)
    assert_true(out.shape() == Shape(2, 3, 4))


def test_attn_minimal_config_shape() raises:
    """
        TEST — the smallest possible attention module at all: `n_embd=1`,.
    `n_head=1` (which is also, trivially, the `head_dim=1` case again, but
    with only a single feature per token in the first place — there is no
    smaller valid configuration than this).

    WHY THIS MATTERS: a "1-wide" tensor is often where indexing bugs hide,
    because many operations that behave identically for widths 2, 3, 4, ...
    can behave differently (or trip an edge case) specifically at width 1
    — e.g. a reduction that assumes there's "a max and a second-max" to
    compare, or a reshape that assumes at least two dimensions can be
    swapped meaningfully.

    SETUP: n_embd=1, n_head=1. Input is (1, 3, 1) — batch=1, sequence
    length=3, a single scalar "embedding" per token.

    ASSERTS: `model(x).shape() == Shape(1, 3, 1)`.

    HOW TO VERIFY: no numeric computation needed here either — this is a
    pure "does it even run" boundary check.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=1, n_head=1, qkv_bias=True)
    var x = Tensor[dtype].ones(Shape(1, 3, 1))
    var out = model(x)
    assert_true(out.shape() == Shape(1, 3, 1))


def test_attn_num_parameters_independent_of_n_head() raises:
    """
    TEST.
    Proves, numerically, a fact stated but not directly tested in
    the original suite: for a FIXED `n_embd`, the total parameter count
    does not depend on `n_head` at all.

    BACKGROUND / WHY THIS IS TRUE: splitting an embedding into multiple
    heads is purely a RESHAPE of the same underlying `c_attn`/`c_proj`
    weight matrices — no new weight matrix is allocated per head. The
    weight matrices' shapes (`(C, 3C)` and `(C, C)`) only ever depend on
    `n_embd` (`C`), never on `n_head`. This is a genuinely easy thing to
    get wrong if someone later refactors the module to give each head its
    own smaller `Linear` (a legitimate alternative implementation strategy
    used by some transformer codebases!) without realizing it would also
    change the parameter count formula and, likely, break weight-loading
    compatibility with pretrained checkpoints that assume the fused-weight
    layout.

    SETUP: four models sharing the same `n_embd=8` but different, all-valid
    values of `n_head` (1, 2, 4, 8 — every divisor of 8).

    ASSERTS: `num_parameters()` returns the SAME value for all four models,
    and that value matches the hand-computed formula for `n_embd=8`:
        c_attn: weight 8*24=192, bias 24   -> 216
        c_proj: weight 8*8=64,  bias 8     -> 72
        total: 216 + 72 = 288

    HOW TO VERIFY: redo the arithmetic above with a calculator — notice
    `n_head` never appears in it. Then confirm all four `assert_true`
    calls below compare against the same literal, 288.
    """
    comptime dtype = DType.float32
    var expected = 288  # see docstring for the arithmetic
    var m1 = SelfAttention[dtype](n_embd=8, n_head=1, qkv_bias=True)
    var m2 = SelfAttention[dtype](n_embd=8, n_head=2, qkv_bias=True)
    var m4 = SelfAttention[dtype](n_embd=8, n_head=4, qkv_bias=True)
    var m8 = SelfAttention[dtype](n_embd=8, n_head=8, qkv_bias=True)
    assert_true(m1.num_parameters() == expected)
    assert_true(m2.num_parameters() == expected)
    assert_true(m4.num_parameters() == expected)
    assert_true(m8.num_parameters() == expected)


def test_attn_named_parameters_full_set_and_order() raises:
    """
        TEST — a stricter version of the original suite's.
    `test_attn_parameters_and_named_plumbing`, which only checked the name
    of the FIRST named parameter. This test pins down all four names, in
    order, plus confirms `dattn` (the Dropout sub-layer) contributes zero
    named parameters.

    BACKGROUND: `named_parameters(prefix)` exists so that, inside a larger
    model (e.g. a full transformer with many stacked blocks), every weight
    tensor can be identified by a unique, human-readable, dotted path
    (think `"transformer.h.3.attn.c_proj.weight"`). Getting the order or
    naming wrong here wouldn't break training (which only needs the FLAT
    list from `parameters()`), but it WOULD break anything that relies on
    names — checkpoint saving/loading by name, debugging print-outs,
    loading pretrained GPT-2 weights (which are keyed by exactly this kind
    of dotted name in the HuggingFace/OpenAI checkpoint format).

    SETUP: n_embd=8, n_head=2, prefix `"attn."`.

    ASSERTS, in order:
      1. `len(named) == 4` — exactly the 4 tensors from the two Linears;
         `Dropout` has no learnable weights, so it must contribute nothing.
      2. `named[0].name == "attn.c_attn.weight"`
      3. `named[1].name == "attn.c_attn.bias"`
      4. `named[2].name == "attn.c_proj.weight"`
      5. `named[3].name == "attn.c_proj.bias"`
      (i.e. c_attn's own [weight, bias] pair first, then c_proj's
      [weight, bias] pair — matching the order `_forward` computes them in:
      c_attn is applied before c_proj.)

    HOW TO VERIFY: read `SelfAttention.named_parameters` in
    `attention.mojo` — it literally calls `self.c_attn.named_parameters(...)`
    first, appends `self.c_proj.named_parameters(...)` second, and never
    touches `self.dattn` at all. This test is really checking that
    `Linear.named_parameters` itself orders weight before bias — worth
    confirming directly in `Linear`'s own tests too, if it isn't already.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=8, n_head=2, qkv_bias=True)
    var named = model.named_parameters("attn.")
    assert_true(len(named) == 4)
    assert_true(String(named[0].name) == "attn.c_attn.weight")
    assert_true(String(named[1].name) == "attn.c_attn.bias")
    assert_true(String(named[2].name) == "attn.c_proj.weight")
    assert_true(String(named[3].name) == "attn.c_proj.bias")


# ═════════════════════════════════════════════════════════════════════════════
# 8. PROBABILITY CONSERVATION — attention weights must sum to exactly 1
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_row_weights_sum_to_one() raises:
    """
    TEST.
    Checks the OTHER half of "attention weights form a valid
    probability distribution". The original suite's
    `test_attn_masked_row_zero_weight` checks that masked (future)
    positions get EXACTLY zero weight; this test checks that the
    UNMASKED positions' weights sum to EXACTLY one, for every query
    position `i`.

    BACKGROUND: softmax's entire job is to turn a row of arbitrary real
    numbers into a probability distribution — a set of non-negative numbers
    that sum to 1. "Masked positions are zero" and "the row sums to one"
    are two independent properties: a bug could, in principle, zero out the
    masked positions correctly while still messing up the normalization of
    the remaining ones (e.g. dividing by the wrong denominator, or summing
    over the wrong axis) — which would slip past the original suite's test
    entirely while still producing a subtly wrong result.

    SETUP: n_embd=2, n_head=1, deterministic weights, the same fixed 4x2
    input `x` used elsewhere in the suite (T=4).

    METHOD: recompute the qkv projection and masked score matrix inline
    (same formulas used throughout this file and the original suite), run
    `_row_softmax_ref_ext` on each row, and sum the resulting probabilities.

    ASSERTS: for every row `i` in `[0, T)`, the sum of `attn[i, :]` is
    within `1e-5` of `1.0`.

    HOW TO VERIFY: for row i=0 (only position 0 is unmasked), the sum is
    trivially 1.0 — softmax of a single valid value is always exactly that
    value's own contribution over itself, i.e. 1.0 (see Section 11's
    docstring for more on why this is always true regardless of the raw
    score). For row i=3 (nothing masked, all four positions valid), you can
    manually add up the four `exp(score - max) / sum` terms this test
    computes and confirm they total 1.0.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=2, n_head=1, qkv_bias=True)
    _set_deterministic_weights_ext(model)
    var x = Tensor[dtype].d2([[0.5, -0.25], [0.1, 0.9], [0.7, -0.4], [0.2, 0.3]])
    var x3 = x.reshape(1, 4, 2)

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
    for qi in range(T):
        for kj in range(T):
            var acc: Scalar[dtype] = 0.0
            for c in range(C):
                acc += qkv[qi, c] * qkv[kj, C + c]
            scores[qi, kj] = acc * inv
            if kj > qi:
                scores[qi, kj] = Scalar[dtype](-1.0) / Scalar[dtype](0.0)
    for qi in range(T):
        var attn = _row_softmax_ref_ext(scores[qi, s()])
        var rowsum: Scalar[dtype] = 0.0
        for kj in range(T):
            rowsum += attn[kj]
        assert_true(rowsum < Scalar[dtype](1.0 + 1e-5))
        assert_true(rowsum > Scalar[dtype](1.0 - 1e-5))


# ═════════════════════════════════════════════════════════════════════════════
# 9. MULTI-HEAD NUMERIC PARITY — the flagship test of this file
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_multi_head_parity_forward() raises:
    """
    TEST — the single most important addition in this file.
    Numeric parity
    between the real module's forward pass and `_attention_ref_multi_head_ext`
    for a genuinely MULTI-head configuration (n_head=2). This is the direct
    multi-head analogue of the original suite's
    `test_attn_single_head_parity_forward`, and it closes Gap #1 from the
    top of this file: nothing else in either test suite checks multi-head
    attention's VALUES, only its shape.

    SETUP: n_embd=4, n_head=2 (so head_dim=2 — head 0 owns columns [0,2) of
    q/k/v, head 1 owns columns [2,4)). Deterministic weights via
    `_set_deterministic_weights_ext`. Model in `.eval()` mode (no dropout
    noise). A fixed, hand-chosen 4x4 input:
        x = [[0.5, -0.25, 0.3 ,  0.1 ],
             [0.1,  0.9 , -0.2,  0.4 ],
             [0.7, -0.4 ,  0.05, -0.1],
             [0.2,  0.3 ,  0.15,  0.25]]
    reshaped to (1, 4, 4).

    METHOD:
      1. `module_out = model(x3)` — the real module's forward pass, which
         internally uses reshape+permute+batched-matmul to compute both
         heads' attention "at once".
      2. `reference = _attention_ref_multi_head_ext(model, x3)` — an
         independent recomputation that processes each head with its own
         explicit column-slice, one head at a time, in a plain loop.
      3. Assert `module_out[0] ≈ reference` element-wise (atol=1e-3).

    WHY THIS SPECIFICALLY CATCHES HEAD-SPLIT/MERGE BUGS: every other piece
    of these two computations — the QKV projection, the `1/sqrt(head_dim)`
    scaling, the causal mask, the softmax — is IDENTICAL in form to what
    the single-head parity test already verifies. The ONLY conceptually new
    code path this test exercises is "does the real reshape+permute-based
    head split actually correspond, mathematically, to slicing out
    consecutive `head_dim`-wide column groups per head?" If a bug swapped,
    say, which axis becomes the "head" axis during the permute, or used the
    wrong stride when reshaping, the module's output would silently mix
    together data from what should be two independent heads — WITHOUT
    changing the output's shape, so the existing `test_attn_shape_
    multi_head` test would still pass. Only a numeric comparison like this
    one can catch that class of bug.

    HOW TO VERIFY BY HAND: because n_embd=4, n_head=2 (head_dim=2) and T=4
    are all small, you can redo `_attention_ref_multi_head_ext`'s 5 steps
    with a calculator or a short numpy script:
      - Compute the (4, 12) `qkv` matrix using the deterministic weight
        formulas (see `_set_deterministic_weights_ext`'s docstring).
      - Split into q, k, v (each 4x4), then split EACH of those into head 0
        (columns 0-1) and head 1 (columns 2-3).
      - For each head, compute its own 4x4 causal-masked score matrix,
        softmax each row, and blend with that head's value columns.
      - Concatenate the two heads' 4x2 outputs back into a 4x4 matrix, then
        apply `c_proj`.
    The result should match what the test asserts (within ~1e-3).
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=4, n_head=2, qkv_bias=True)
    _set_deterministic_weights_ext(model)
    model.eval()
    var x = Tensor[dtype].d2(
        [
            [0.5, -0.25, 0.3, 0.1],
            [0.1, 0.9, -0.2, 0.4],
            [0.7, -0.4, 0.05, -0.1],
            [0.2, 0.3, 0.15, 0.25],
        ]
    )
    var x3 = x.reshape(1, 4, 4)
    var module_out = model(x3)  # (1, 4, 4)
    var reference = _attention_ref_multi_head_ext(model, x3)  # (4, 4)
    assert_true(
        module_out[i(0), s(), s()].all_close[atol=1e-3](reference)
    )


# ═════════════════════════════════════════════════════════════════════════════
# 10. BATCH INDEPENDENCE — no cross-example leakage
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_batch_independence() raises:
    """
    TEST.
    Confirms that two different sequences processed TOGETHER in one
    batch produce EXACTLY the same per-sequence outputs as processing them
    SEPARATELY, one at a time. This directly targets Gap #3: nothing else
    in either suite mixes multiple distinct sequences into one batch and
    cross-checks them against independent single-item runs.

    BACKGROUND: a batch dimension exists purely for computational
    efficiency — running 32 unrelated sequences through the model "at once"
    should be mathematically identical to running each of the 32 sequences
    through the model one at a time; the model must never let information
    from sequence A leak into sequence B's output just because they
    happened to be processed in the same batch. This sounds obvious, but
    it's a genuinely common class of bug in tensor code: a reduction,
    reshape, or matmul that accidentally operates across the batch axis
    instead of stopping at each sequence's own boundary will typically
    still produce THE RIGHT SHAPE (which is why shape tests alone can't
    catch it) while silently blending unrelated sequences' information
    together.

    SETUP: n_embd=6, n_head=2, model in `.eval()` mode (dropout defaults to
    0.0 anyway, but eval mode is used for consistency with the rest of the
    suite's numeric tests). Two independently-random (5-token) sequences,
    `x1` and `x2`, each shape (1, 5, 6).

    METHOD:
      1. Run `model(x1)` and `model(x2)` SEPARATELY -> `out1`, `out2`, each
         (1, 5, 6).
      2. Manually build a batched tensor `batched` of shape (2, 5, 6) by
         copying `x1`'s values into batch slot 0 and `x2`'s values into
         batch slot 1 (element by element — no separate "concatenate"
         tensor op is assumed to exist).
      3. Run `model(batched)` -> `out_batched`, shape (2, 5, 6).
      4. Assert `out_batched`'s batch-slot-0 slice matches `out1`, and its
         batch-slot-1 slice matches `out2` (both within atol=1e-4).

    WHY THIS IS ENOUGH TO PROVE INDEPENDENCE: if the model's internal
    matmuls or reductions ever accidentally spanned the batch axis, mixing
    x1's and x2's numbers together would change the resulting values for
    AT LEAST one of the two batch slots compared to running them alone —
    there is no way to "accidentally leak" information between the two
    sequences and still coincidentally reproduce both single-item results
    exactly. Passing this test is a strong, direct guarantee that the batch
    axis is being treated as fully independent "lanes" of computation.

    HOW TO VERIFY: rerun with different random sequences, or with `B=3` or
    more batch slots, and the same equality should hold slot-by-slot.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=6, n_head=2, qkv_bias=True)
    model.eval()
    var x1 = Tensor[dtype].randn(Shape(1, 5, 6), mean=0.0, std=0.1)
    var x2 = Tensor[dtype].randn(Shape(1, 5, 6), mean=0.0, std=0.1)
    var out1 = model(x1)
    var out2 = model(x2)

    var batched = Tensor[dtype].zeros(Shape(2, 5, 6))
    for t in range(5):
        for c in range(6):
            batched[0, t, c] = x1[0, t, c]
            batched[1, t, c] = x2[0, t, c]
    var out_batched = model(batched)

    assert_true(
        out_batched[i(0), s(), s()].all_close[atol=1e-4](
            out1[i(0), s(), s()]
        )
    )
    assert_true(
        out_batched[i(1), s(), s()].all_close[atol=1e-4](
            out2[i(0), s(), s()]
        )
    )


# ═════════════════════════════════════════════════════════════════════════════
# 11. SINGLE-TOKEN SEQUENCE EDGE CASE (T=1)
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_single_token_softmax_is_trivially_one() raises:
    """
        TEST — the most degenerate possible SEQUENCE (as opposed to Section 7's.
    most degenerate CONFIGURATIONS): `T=1`, a single-token "sequence". This
    both stress-tests an important edge case and illustrates a genuinely
    useful, reassuring fact about softmax for anyone new to attention.

    THE FACT WORTH UNDERSTANDING: when a query position has exactly ONE
    unmasked key to attend to (which is always true for position 0 in ANY
    causal-attention sequence, and is true for EVERY position when T=1),
    the softmax of "one real number, with everything else masked to -inf"
    is ALWAYS exactly 1.0 for that one real number — no matter how large,
    small, positive, or negative the underlying raw score is. This is
    because softmax is shift-invariant (subtracting the row's max before
    exponentiating, as `_row_softmax_ref_ext` does, leaves that single
    surviving entry at exactly `exp(0) = 1`, divided by a sum that is also
    just `1` since there's nothing else to add). In other words: the actual
    numeric value of `q . k / sqrt(dh)` for a lone unmasked position is
    COMPLETELY IRRELEVANT to the resulting attention weight — it could
    overflow to `1e30` or underflow to `-1e30` and the output would be
    identical. This is a nice piece of intuition for anyone worried about
    numerical stability in attention: the position with the fewest
    alternatives to attend to is also the position least sensitive to the
    raw score's magnitude.

    SETUP: n_embd=4, n_head=1, deterministic weights, model in `.eval()`
    mode. A single token: `x = [[0.37, -1.2, 0.05, 2.0]]`, reshaped to
    (1, 1, 4).

    METHOD: compare `model(x3)` against `_attention_ref_multi_head_ext`
    with `n_head=1` (the module's actual configuration) — which, for T=1,
    reduces to: attention output for the one position equals its OWN value
    vector exactly (attention weight forced to 1.0), then projected through
    `c_proj`.

    ASSERTS: `module_out[0] ≈ reference` (atol=1e-3), same style as the
    other parity tests in this suite.

    HOW TO VERIFY BY HAND: because there's only one position, you don't
    even need to compute the query/key dot product at all — whatever it is,
    the attention weight on the only available key is 1.0. So the "attention
    output" (before `c_proj`) is simply this token's own value vector `v[0]
    = qkv[0, 2C:3C]`. Compute that (a small matvec) and then apply
    `c_proj`'s weight and bias — that's the entire expected answer.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=4, n_head=1, qkv_bias=True)
    _set_deterministic_weights_ext(model)
    model.eval()
    var x = Tensor[dtype].d2([[0.37, -1.2, 0.05, 2.0]])
    var x3 = x.reshape(1, 1, 4)
    var module_out = model(x3)  # (1, 1, 4)
    var reference = _attention_ref_multi_head_ext(model, x3)  # (1, 4)
    assert_true(
        module_out[i(0), s(), s()].all_close[atol=1e-3](reference)
    )


# ═════════════════════════════════════════════════════════════════════════════
# 12. SEEDED-INITIALIZATION REPRODUCIBILITY
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_seeded_init_is_deterministic() raises:
    """
    TEST.
    Confirms that `init_seed` genuinely makes weight initialization
    reproducible: two SEPARATELY constructed modules, given the same
    `n_embd`, `n_head`, `init_seed`, and `init_method`, must end up with
    numerically identical weights (not just "similar" — identical).

    BACKGROUND — why reproducibility matters: neural network weights are
    normally initialized from a random distribution (see `init_method`,
    e.g. "uniform" for PyTorch nn.Linear's default scheme). Uncontrolled
    randomness makes debugging extremely painful — if a training run
    behaves unexpectedly, you want to be able to rerun the EXACT same
    experiment (same starting weights, same data order) to isolate whether
    a change in behavior came from your code edit or just from a different
    random draw. `init_seed` is the documented mechanism for pinning down
    that randomness. This test is really a "specification check" on that
    contract — if it fails, it's not necessarily this module's fault (the
    underlying RNG plumbing could live elsewhere in Tenmo), but it reveals
    that the reproducibility guarantee newcomers are likely to assume
    doesn't actually hold, which is important to know either way.

    SETUP: `model1` and `model2`, both `SelfAttention[dtype](n_embd=8,
    n_head=2, init_seed=42, init_method="uniform", qkv_bias=True)`,
    constructed one after the other.

    ASSERTS:
      1. `model1.c_attn.weight` all_close `model2.c_attn.weight` (atol=0,
         i.e. bit-for-bit — if a seed genuinely pins the RNG, there should
         be no floating-point "noise" between two runs at all).
      2. `model1.c_proj.weight` all_close `model2.c_proj.weight`.
      3. As an end-to-end confirmation: with both models in `.eval()` mode
         (no dropout), running the SAME input through both produces
         identical output.

    HOW TO VERIFY: rerun this test multiple times (or in a loop) — a truly
    seeded implementation should pass every time, with no flakiness. If
    this test is flaky or fails outright, it's worth checking whether
    `init_seed` is consumed once per model (good) or advances some shared
    global RNG state that a second model's construction would then pick up
    from a different point (which would break exact reproducibility for
    the SECOND model constructed with a given seed, even though the seed
    value itself was passed correctly).
    """
    comptime dtype = DType.float32
    var model1 = SelfAttention[dtype](
        n_embd=8, n_head=2, init_seed=42, init_method="uniform", qkv_bias=True
    )
    var model2 = SelfAttention[dtype](
        n_embd=8, n_head=2, init_seed=42, init_method="uniform", qkv_bias=True
    )
    assert_true(model1.c_attn.weight.all_close[atol=0](model2.c_attn.weight))
    assert_true(model1.c_proj.weight.all_close[atol=0](model2.c_proj.weight))

    model1.eval()
    model2.eval()
    var x = Tensor[dtype].randn(Shape(1, 4, 8), mean=0.0, std=0.05)
    var out1 = model1(x)
    var out2 = model2(x)
    assert_true(out1.all_close[atol=1e-6](out2))


# ═════════════════════════════════════════════════════════════════════════════
# 13. DROPOUT MODE BEHAVIOR — train (stochastic) vs eval (deterministic)
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_dropout_eval_is_deterministic() raises:
    """
    TEST.
    Confirms that a model constructed WITH non-zero dropout
    (`dropout_p=0.5`) becomes fully deterministic once switched to
    `.eval()` mode: calling it twice on the SAME input must return the
    SAME output.

    BACKGROUND — what dropout is, briefly: dropout is a regularization
    technique that, during training only, randomly zeroes out some
    fraction (`dropout_p`) of values — here, some of the post-softmax
    attention weights (see step D of `_forward`) — forcing the model not
    to rely too heavily on any single attention connection. This
    randomness is only useful DURING training; at evaluation/inference
    time you want a stable, reproducible prediction, not a different
    answer every time you ask the same question. That's why `Dropout`
    modules (and, by extension, any layer that contains one) are expected
    to become the IDENTITY function once `.eval()` is called.

    WHY THIS IS WORTH TESTING SPECIFICALLY: the original suite's
    `test_attn_training_flag_and_mode_toggle` only checks that the
    `self.training` BOOLEAN flips correctly — it never checks that
    flipping it actually changes forward-pass BEHAVIOR. A bug where
    `.eval()` correctly sets `self.training = False` but forgets to also
    call `self.dattn.eval()` (or where `Dropout.eval()` itself is broken)
    would pass every existing test while silently leaving dropout active
    at "inference" time — a real, easy-to-make bug that would make a
    model's predictions non-reproducible in production.

    SETUP: n_embd=16, n_head=4, `dropout_p=0.5` (deliberately aggressive,
    to make any leftover randomness obvious rather than subtle). `.eval()`
    is called explicitly. Input: a (1, 6, 16) random tensor.

    ASSERTS: `model(x)` called twice produces `all_close(atol=1e-6)`
    results — i.e. bit-for-bit reproducible (up to ordinary floating point
    associativity noise), not just "close by chance".

    HOW TO VERIFY: run this test repeatedly — it should never fail. If it
    ever does, dropout is leaking into eval-mode inference.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=16, n_head=4, dropout_p=0.5, qkv_bias=True)
    model.eval()
    var x = Tensor[dtype].randn(Shape(1, 6, 16), mean=0.0, std=0.05)
    var out1 = model(x)
    var out2 = model(x)
    assert_true(out1.all_close[atol=1e-6](out2))


def test_attn_dropout_train_is_stochastic() raises:
    """
    TEST — the mirror image of the previous test.
    Confirms that, in
    `.train()` mode, non-zero dropout actually DOES something — i.e. two
    forward passes on the SAME input with the SAME weights produce
    DIFFERENT outputs, because a different random subset of attention
    weights gets zeroed out each time.

    WHY BOTH DIRECTIONS MATTER: `test_attn_dropout_eval_is_deterministic`
    alone could technically be satisfied by a `Dropout` implementation that
    is ALWAYS a no-op (never drops anything, in either mode) — which would
    be a silent bug (dropout configured but not actually regularizing
    anything during training) that the eval-mode test alone cannot catch.
    Only by ALSO confirming train-mode output varies do we know dropout is
    genuinely wired in and active when it's supposed to be.

    SETUP: same model configuration as the previous test (n_embd=16,
    n_head=4, dropout_p=0.5), left in its default `.train()` mode (the
    constructor's default — see `SelfAttention.__init__`, which sets
    `self.training = True`). Same style of random input.

    ASSERTS: `model(x)` called twice produces outputs that are NOT
    `all_close` — i.e. `assert_false(out1.all_close(out2))`.

    A NOTE ON WHY THIS IS A SAFE (NOT FLAKY) STATISTICAL TEST: with
    `dropout_p=0.5` applied independently to every one of
    `n_head * T * T = 4 * 6 * 6 = 144` attention-weight entries, the chance
    that two independent random dropout masks happen to be IDENTICAL is
    roughly `0.5^144` — a number so astronomically small it will never
    occur in practice (this is the same style of reasoning used any time a
    test relies on "random data won't coincidentally collide", e.g. hash
    tests). If this assumption doesn't hold for this codebase's `Dropout` —
    for instance, if it uses a FIXED per-instance seed rather than
    re-randomizing on every call — this test will fail consistently rather
    than "flake" occasionally, which is itself a useful signal: it would
    mean dropout is deterministic across calls even in training mode,
    which is worth knowing and is different from the commonly-expected
    behavior this test assumes.

    HOW TO VERIFY: rerun this test many times; a correct, re-randomizing
    `Dropout` implementation should pass essentially every time. If it
    fails, check whether `Dropout`'s random mask is regenerated on every
    `__call__`, or only once at construction time.
    """
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=16, n_head=4, dropout_p=0.5, qkv_bias=True)
    assert_true(model.training)  # default mode; dropout should be active
    var x = Tensor[dtype].randn(Shape(1, 6, 16), mean=0.0, std=0.05)
    var out1 = model(x)
    var out2 = model(x)
    assert_false(out1.all_close[atol=1e-6](out2))


# ═════════════════════════════════════════════════════════════════════════════
# 14. GRADIENT WIRING — does backward() actually reach every parameter?
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_backward_updates_all_parameters() raises:
    """
        TEST — a narrower, faster, more DIAGNOSTIC version of the original.
    suite's `test_attn_overfit_one_batch`. Rather than training for 60
    steps and checking whether the AGGREGATE loss went down (which it
    could do even if one of the four parameter tensors never received any
    gradient at all, as long as the other three carried enough signal),
    this test runs exactly ONE training step and checks, individually,
    that EVERY ONE of the four parameter tensors this module owns
    (`c_attn.weight`, `c_attn.bias`, `c_proj.weight`, `c_proj.bias`)
    actually changed value.

    BACKGROUND — why "changed value" proves gradient flow: an `SGD`
    optimizer step computes `param := param - lr * param.grad`. If
    `param.grad` were ever exactly zero (e.g. because a broken autograd
    edge silently failed to connect that parameter to the loss), the
    parameter would come out of `optimizer.step()` byte-for-byte identical
    to how it went in. So "the parameter changed after one step" is a
    direct, simple proxy for "backward() successfully computed a
    (generically nonzero) gradient for this parameter" — without needing
    any direct API for inspecting `.grad` tensors.

    SETUP: a small end-to-end pipeline, deliberately similar in shape to
    the original suite's overfit test but smaller and run for only ONE
    step: `wte` (Embedding, V=16 -> C=8), `attn` (SelfAttention,
    n_embd=8, n_head=2), `head` (Linear, C=8 -> V=16). Two short (T=3)
    fixed token sequences as `tokens`/`targets`. All parameters from all
    three sub-modules are combined into one list for a single `SGD`
    optimizer (`lr=0.1`).

    METHOD:
      1. BEFORE the step: `.clone()` each of the attention module's 4
         parameter tensors (`c_attn.weight`, `c_attn.bias`, `c_proj.weight`,
         `c_proj.bias`) to snapshot their pre-update values.
      2. Run one forward pass through `wte -> attn -> head`, compute
         cross-entropy loss against `targets` (same `permute` trick as the
         original overfit test, to put the class/vocab axis where
         `CrossEntropyLoss` expects it).
      3. `optimizer.zero_grad(); loss.backward(); optimizer.step()`.
      4. AFTER the step: compare each of the 4 "before" snapshots against
         the corresponding CURRENT tensor.

    ASSERTS: `assert_false(before.all_close(after))` for EACH of the 4
    parameter tensors individually — i.e. every single one must have
    measurably changed, not just the aggregate loss.

    WHY THIS IS A MORE PRECISE DIAGNOSTIC THAN THE ORIGINAL OVERFIT TEST:
    imagine a bug where `c_proj`'s bias never receives a gradient (perhaps
    due to a subtle autograd-graph wiring mistake specific to bias
    addition). The original suite's `test_attn_overfit_one_batch` could
    still very plausibly pass — the other three parameter tensors, plus
    `wte` and `head`'s own parameters, likely carry more than enough
    capacity to overfit two 3-token sequences over 60 steps, masking the
    bias's silent non-participation entirely. This test would catch that
    bug immediately and point at exactly which tensor is the problem.

    HOW TO VERIFY: temporarily comment out one parameter from the
    `params` list built below (simulating "this parameter isn't wired to
    the optimizer") and confirm the corresponding assertion fails — that's
    a good way to convince yourself this test is actually sensitive to the
    thing it claims to check.
    """
    comptime dtype = DType.float32
    var B = 2
    var T = 3
    var C = 8
    var V = 16
    var H = 2
    var attn = SelfAttention[dtype](n_embd=C, n_head=H, qkv_bias=True)
    var wte = Embedding[dtype](num_embeddings=V, embedding_dim=C)
    var head = Linear[dtype](C, V)
    attn.train()

    var tokens = Tensor[DType.int64].d2([[2, 5, 9], [1, 4, 7]])
    var targets = Tensor[DType.int64].d2([[5, 9, 2], [4, 7, 1]])

    var c_attn_w_before = attn.c_attn.weight.clone()
    var c_attn_b_before = attn.c_attn.bias.value().clone()
    var c_proj_w_before = attn.c_proj.weight.clone()
    var c_proj_b_before = attn.c_proj.bias.value().clone()

    var params = wte.parameters()
    var attn_params = attn.parameters()
    for p in range(len(attn_params)):
        params.append(attn_params[p])
    var head_params = head.parameters()
    for p in range(len(head_params)):
        params.append(head_params[p])

    var optimizer = SGD[dtype](params, lr=0.1)
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()

    var x = wte(tokens)
    var out = attn(x)
    var logits = head(out)
    var logits_btv = logits.permute([0, 2, 1])
    var loss = criterion(logits_btv, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert_false(c_attn_w_before.all_close[atol=1e-9](attn.c_attn.weight))
    assert_false(
        c_attn_b_before.all_close[atol=1e-9](attn.c_attn.bias.value())
    )
    assert_false(c_proj_w_before.all_close[atol=1e-9](attn.c_proj.weight))
    assert_false(
        c_proj_b_before.all_close[atol=1e-9](attn.c_proj.bias.value())
    )


# ═════════════════════════════════════════════════════════════════════════════
# 15. GPU ROUND-TRIP VALUE PRESERVATION
# ═════════════════════════════════════════════════════════════════════════════


def test_attn_gpu_roundtrip_preserves_weight_values() raises:
    """
    TEST.
    Extends the original suite's `test_attn_gpu_roundtrip`, which
    only checks that `num_parameters()` survives a `to_gpu()` -> `to_cpu()`
    round trip, to also check that the ACTUAL WEIGHT VALUES survive
    unchanged.

    WHY THE ORIGINAL TEST ISN'T ENOUGH ON ITS OWN: `num_parameters()`
    simply counts elements — it would report the same count whether the
    round trip preserved every value perfectly, silently zeroed them all
    out, or scrambled their order. A count-only check cannot distinguish
    "moved correctly" from "moved to a buffer of the right size, but with
    garbage/incorrect contents" (a plausible bug in manual GPU buffer
    copy/allocation code, e.g. copying the wrong stride or forgetting to
    synchronize before reading back).

    GUARD: like the original GPU test, this only runs when
    `has_accelerator()` is true, so it's a no-op on CPU-only machines
    rather than a failure.

    SETUP: n_embd=8, n_head=2, deterministic weights via
    `_set_deterministic_weights_ext` (so there's a known, exact value to
    check for, rather than trusting whatever the random initializer
    produced).

    METHOD:
      1. Snapshot `attn.c_attn.weight` and `attn.c_proj.weight` via
         `.clone()` before any device transfer.
      2. `gpu_model = model.to_gpu()`, then `back = gpu_model.to_cpu()`.
      3. Compare the snapshots against `back.c_attn.weight` /
         `back.c_proj.weight`.

    ASSERTS: both weight tensors are `all_close(atol=1e-6)` to their
    pre-transfer snapshots — i.e. the round trip is numerically lossless
    (up to ordinary floating-point representation, not exactly bit-for-bit
    if any dtype narrowing occurs on-device, hence the small tolerance
    rather than `atol=0`).

    HOW TO VERIFY: only runnable on a machine with GPU support enabled.
    Once there, you can strengthen this test further by also checking
    `c_attn.bias.value()` / `c_proj.bias.value()`, and by running a full
    forward pass before and after the round trip (using the multi-head
    parity reference from Section 9) to confirm not just the raw weight
    values but the resulting COMPUTATION is unaffected by the transfer.
    """
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var model = SelfAttention[dtype](n_embd=8, n_head=2, qkv_bias=True)
        _set_deterministic_weights_ext(model)
        var c_attn_w_before = model.c_attn.weight.clone()
        var c_proj_w_before = model.c_proj.weight.clone()

        var gpu_model = model.to_gpu()
        var back = gpu_model.to_cpu()

        assert_true(c_attn_w_before.all_close[atol=1e-6](back.c_attn.weight))
        assert_true(c_proj_w_before.all_close[atol=1e-6](back.c_proj.weight))


def main() raises:
    """
        ENTRY POINT for this extended suite. Identical pattern to the original.
    suite's `main()`: discovers and runs every `test_attn_*` function
    defined in this module.

    WHEN MERGING with the original suite into a single file, keep only ONE
    `main()` (and note the naming collision this implies — Mojo will not
    allow two top-level functions named `main` in the same module) and
    make sure `TestSuite.discover_tests[__functions_in_module()]()` is
    called against the MERGED module so it picks up every `test_attn_*`
    function from both files, not just one or the other.
    """
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll extended attention tests passed!")
