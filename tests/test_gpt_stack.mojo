"""
Tests for the GPT stack.

────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE IS FOR (read this first if you are new to the codebase).
────────────────────────────────────────────────────────────────────────────
This file tests the full GPT-model stack: MLP, TransformerBlock,
GPTEmbedding, and GPTModel.  Each test function below corresponds to one of
the ten gate cases.

HOW TO RUN:
    ./execute.sh gpt_stack

The test suite discovers all `test_gpt_*` functions automatically via
`TestSuite.discover_tests[__functions_in_module()]().run()`.
────────────────────────────────────────────────────────────────────────────
"""

from std.testing import assert_true, assert_false, assert_raises, TestSuite
from std.sys import has_accelerator
from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape
from tenmo.shared.indexhelper import i, s
from tenmo.gpt import MLP, TransformerBlock, GPTEmbedding, GPTModel
from tenmo.positional import PositionalEmbedding
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.optim import SGD
from tenmo.embedding import Embedding
from tenmo.net import Linear
from tenmo.shared.intarray import IntArray


# ═════════════════════════════════════════════════════════════════════════════
# Case 1 — Shapes
# ═════════════════════════════════════════════════════════════════════════════


def test_gpt_shapes() raises:
    """
    TEST.
    Verify output shapes for MLP, TransformerBlock, GPTEmbedding,
    and GPTModel across several input configurations.

    ASSERTS:
      - MLP(Shape(2,3,8)) → (2,3,8)
      - TransformerBlock(Shape(2,3,8)) → (2,3,8)
      - GPTEmbedding(Shape(2,5)) → (2,5,8)
      - GPTModel(Shape(2,5)) → (2,5,V)
      - GPTModel(Shape(2,8)) → (2,8,V)   (B=2, T=8 variant).
    """
    comptime dtype = DType.float32
    var C = 8
    var V = 16
    var n_ctx = 8

    # MLP
    var mlp = MLP[dtype](n_embd=C)
    mlp.eval()
    var x_mlp = Tensor[dtype].rand(Shape(2, 3, C), init_seed=1)
    var out_mlp = mlp(x_mlp)
    assert_true(out_mlp.shape() == Shape(2, 3, C))

    # TransformerBlock
    var block = TransformerBlock[dtype](n_embd=C, n_head=2)
    block.eval()
    var x_block = Tensor[dtype].rand(Shape(2, 3, C), init_seed=2)
    var out_block = block(x_block)
    assert_true(out_block.shape() == Shape(2, 3, C))

    # GPTEmbedding — (B, T) → (B, T, C)
    var emb = GPTEmbedding[dtype](V, n_ctx, C)
    emb.eval()
    var tokens_5 = Tensor[DType.int64].d2([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    var out_emb = emb(tokens_5)
    assert_true(out_emb.shape() == Shape(2, 5, C))

    # GPTModel — (B, T) → (B, T, V) with T=5
    var gpt = GPTModel[dtype](V, n_ctx, C, n_head=2, n_layer=2)
    gpt.eval()
    var out_gpt = gpt(tokens_5)
    assert_true(out_gpt.shape() == Shape(2, 5, V))

    # GPTModel — B=2, T=8 variant
    var tokens_8 = Tensor[DType.int64].d2([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
    ])
    var out_gpt_8 = gpt(tokens_8)
    assert_true(out_gpt_8.shape() == Shape(2, 8, V))


# ═════════════════════════════════════════════════════════════════════════════
# Case 2 — Residual identity
# ═════════════════════════════════════════════════════════════════════════════


def test_gpt_residual_identity() raises:
    """
    TEST.
    Zero c_proj.weight AND bias inside attn and mlp;
    TransformerBlock(x) == x exactly (both residual +s verified).

    WHY: if the residual stream were broken or if c_proj zeroed output
    were not exactly zero, the block output would drift from x.
    """
    comptime dtype = DType.float32
    var C = 8
    var block = TransformerBlock[dtype](n_embd=C, n_head=2)
    block.eval()
    var x = Tensor[dtype].rand(Shape(1, 4, C), init_seed=42)

    # Zero c_proj weights AND biases inside attn and mlp
    block.attn.c_proj.weight.fill(Scalar[dtype](0.0))
    block.attn.c_proj.bias.value().fill(Scalar[dtype](0.0))
    block.mlp.c_proj.weight.fill(Scalar[dtype](0.0))
    block.mlp.c_proj.bias.value().fill(Scalar[dtype](0.0))

    var out = block(x)
    # With zeroed c_proj, attn_out = 0, residual_1 = x + 0 = x,
    # mlp_out = 0, out = x + 0 = x.  Bitwise exact (adding 0.0).
    assert_true(out == x)


# ═════════════════════════════════════════════════════════════════════════════
# Case 3 — Numeric parity (pre-norm order)
# ═════════════════════════════════════════════════════════════════════════════


def test_gpt_numeric_parity() raises:
    """
    TEST.
    Block output equals a hand-written reference that calls the
    already-verified ln_1, attn, ln_2, mlp in exact order with the
    residual adds: ln_1 consumes x, ln_2 consumes h (= x + attn(ln_1(x))).

    ASSERTS: block(x) ≈ reference within atol=1e-5 (eval mode, no graph).
    """
    comptime dtype = DType.float32
    var C = 8
    var block = TransformerBlock[dtype](n_embd=C, n_head=2, init_seed=42)
    block.eval()
    var x = Tensor[dtype].rand(Shape(1, 5, C), init_seed=99)

    # Hand-written reference
    var h1 = block.ln_1(x)
    var attn_out = block.attn(h1)
    var res1 = x.__add__[track_grad=False, sync=True](attn_out)
    var h2 = block.ln_2(res1)
    var mlp_out = block.mlp(h2)
    var residual_ref = res1.__add__[track_grad=False, sync=True](mlp_out)

    # Module forward
    var got = block(x)

    assert_true(got.all_close[atol=Scalar[dtype](1e-5), rtol=Scalar[dtype](1e-5)](residual_ref))


# ═════════════════════════════════════════════════════════════════════════════
# Case 4 — Block causal no-lookahead
# ═════════════════════════════════════════════════════════════════════════════


def test_gpt_block_causal_no_lookahead() raises:
    """
    TEST.
    Future input positions must not affect earlier logits through
    the whole block (LayerNorm is per-position; this confirms no
    cross-position leakage was introduced by composition).

    METHOD: compute block(x) on original x; zero out positions 3,4;
    recompute; compare positions 0..2.  Identical to attention test's
    causal no-lookahead pattern.
    """
    comptime dtype = DType.float32
    var C = 8
    var block = TransformerBlock[dtype](n_embd=C, n_head=2, init_seed=42)
    block.eval()
    var x = Tensor[dtype].randn(Shape(1, 5, C), mean=0.0, std=0.1, init_seed=7)
    var out_full = block(x)

    # Zero out all inputs at positions > 2; rows 0..2 must not change.
    var x_cut = x.clone()
    for t in range(3, 5):
        for c in range(C):
            x_cut[0, t, c] = 0.0
    var out_cut = block(x_cut)
    for t in range(3):
        assert_true(
            out_full[i(0), i(t), s()].all_close[atol=Scalar[dtype](1e-4)](
                out_cut[i(0), i(t), s()]
            )
        )


# ═════════════════════════════════════════════════════════════════════════════
# Case 5 — GPTEmbedding: wte + wpe parity, position-sharing, raises, contiguity
# ═════════════════════════════════════════════════════════════════════════════


def test_gpt_embedding_parity() raises:
    """
    TEST.
    GPTEmbedding parity and properties:
      1. Numeric parity: out[b,t,:] == wte[tok] + wpe[t] for each (b,t).
      2. Position-sharing: same position row added regardless of batch
         item or token.
      3. T > n_ctx raises Error (in-process assert_raises).
      4. Contiguity of position_ids.
    """
    comptime dtype = DType.float32
    var V = 8
    var n_ctx = 6
    var C = 4
    var emb = GPTEmbedding[dtype](V, n_ctx, C, init_seed=42)
    emb.eval()

    # Set deterministic wte values: row v, col c → (v+1) + (c+1)*0.1
    for v in range(V):
        for c in range(C):
            emb.wte.weight[v, c] = Scalar[dtype](
                Float64(v + 1) + Float64(c + 1) * 0.1
            )

    # Set deterministic wpe values: row t, col c → (t+1)*0.2 + (c+1)*0.01
    for t in range(n_ctx):
        for c in range(C):
            emb.wpe.weight()[t, c] = Scalar[dtype](
                Float64(t + 1) * 0.2 + Float64(c + 1) * 0.01
            )

    # Tokens: (2, 5) covering distinct vocab ids
    var tokens = Tensor[DType.int64].d2([[1, 3, 5, 7, 2], [0, 2, 4, 6, 1]])
    var B = 2
    var T = 5

    # Forward
    var out = emb(tokens)  # (B, T, C)

    # 1. Numeric parity: build expected by direct wte+wpe row reads
    var expected = Tensor[dtype].zeros(Shape(B, T, C))
    var wte_ref = emb.wte.weight
    var wpe_ref = emb.wpe.weight()
    for b in range(B):
        for t in range(T):
            var tok = Int(tokens.get(b * T + t))
            for c in range(C):
                var wte_val = wte_ref.get(tok * C + c)
                var wpe_val = wpe_ref.get(t * C + c)
                expected.set(b * T * C + t * C + c, wte_val + wpe_val)
    assert_true(out.all_close[atol=Scalar[dtype](1e-5)](expected))

    # 2. Position-sharing: for same position t, different batch items b1, b2,
    #    the output difference equals the wte difference (wpe contribution
    #    is identical).
    for t in range(T):
        var tok0 = Int(tokens.get(0 * T + t))
        var tok1 = Int(tokens.get(1 * T + t))
        for c in range(C):
            var diff0 = out.get(0 * T * C + t * C + c) - wte_ref.get(
                tok0 * C + c
            )
            var diff1 = out.get(1 * T * C + t * C + c) - wte_ref.get(
                tok1 * C + c
            )
            assert_true(
                abs(diff0 - diff1) < Scalar[dtype](1e-5)
            )

    # 3. T > n_ctx raises Error — test position_ids directly (raises Error)
    with assert_raises():
        _ = PositionalEmbedding[dtype, DType.int64].position_ids(
            n_ctx, B, n_ctx + 1
        )

    # 4. Contiguity of position_ids
    assert_true(
        PositionalEmbedding[dtype, DType.int64]
        .position_ids(n_ctx, B, T)
        .is_contiguous()
    )


# ═════════════════════════════════════════════════════════════════════════════
# Case 6 — Tied weight-tying gradient
# ═════════════════════════════════════════════════════════════════════════════


def test_gpt_tied_gradient() raises:
    """
    TEST — after one backward step.
    Wte's grad equals the sum of the
    embedding-use and output-use contributions.

    Compare against an untied reference model seeded identically with
    its lm_head weight overwritten to equal wte^T (so forwards are
    identical); the tied wte grad should equal:
        untied_wte.grad  +  untied_lm_head.weight.grad.transpose([1,0])

    Also asserts wte appears exactly once in parameters()/named_parameters().
    """
    comptime dtype = DType.float32
    var V = 8
    var C = 8
    var n_ctx = 8
    var n_head = 2
    var n_layer = 2
    var B = 2
    var T = 4

    var tokens = Tensor[DType.int64].d2([[1, 3, 5, 7], [0, 2, 4, 6]])
    var targets = Tensor[DType.int64].d2([[3, 5, 7, 1], [2, 4, 6, 0]])

    # Build two GPTModels seeded identically
    var model_tied = GPTModel[dtype](
        V, n_ctx, C, n_head, n_layer, tie_weights=True, init_seed=42
    )
    var model_untied = GPTModel[dtype](
        V, n_ctx, C, n_head, n_layer, tie_weights=False, init_seed=42
    )
    model_tied.train()
    model_untied.train()

    # Overwrite untied lm_head weight to be wte^T so forwards are identical
    # lm_head.weight is (C, V); wte is (V, C).
    var wte_src = model_untied.wte_wpe.wte.weight
    for c in range(C):
        for v in range(V):
            model_untied.lm_head.value().weight[c, v] = wte_src.get(
                v * C + c
            )
    # lm_head bias is already zero (bias_zero=True default)

    # Forward + backward on tied model
    var logits_tied = model_tied(tokens)  # (B, T, V)
    var logits_btv_tied = logits_tied.permute([0, 2, 1])  # (B, V, T)
    var criterion_tied = CrossEntropyLoss[dtype](reduction="mean")
    criterion_tied.train()
    var loss_tied = criterion_tied(logits_btv_tied, targets)
    loss_tied.backward()

    # Forward + backward on untied model
    var logits_untied = model_untied(tokens)
    var logits_btv_untied = logits_untied.permute([0, 2, 1])
    var criterion_untied = CrossEntropyLoss[dtype](reduction="mean")
    criterion_untied.train()
    var loss_untied = criterion_untied(logits_btv_untied, targets)
    loss_untied.backward()

    # Grab grads
    var g_tied = model_tied.wte_wpe.wte.weight.grad()
    var g_emb = model_untied.wte_wpe.wte.weight.grad()
    var g_head_transposed = model_untied.lm_head.value().weight.grad().transpose(
        IntArray(1, 0)
    )
    var g_ref = g_emb + g_head_transposed

    # Tolerant comparison (accumulation order differs slightly)
    assert_true(
        g_tied.all_close[rtol=Scalar[dtype](1e-3), atol=Scalar[dtype](1e-3)](
            g_ref
        )
    )

    # wte appears exactly once in parameters() / named_parameters()
    var wte_count_params = 0
    var params = model_tied.parameters()
    var wte_ptr = model_tied.wte_wpe.wte.weight
    for p in range(len(params)):
        if params[p][].data_ptr() == wte_ptr.data_ptr():
            wte_count_params += 1
    assert_true(wte_count_params == 1)

    var wte_count_named = 0
    var named = model_tied.named_parameters("")
    for p in range(len(named)):
        if named[p].name == "wte_wpe.wte.weight":
            wte_count_named += 1
    assert_true(wte_count_named == 1)

    # No lm_head in tied model
    var has_lm_head = False
    for p in range(len(named)):
        if named[p].name == "lm_head.weight":
            has_lm_head = True
    assert_false(has_lm_head)


# ═════════════════════════════════════════════════════════════════════════════
# Case 7 — num_parameters
# ═════════════════════════════════════════════════════════════════════════════


def test_gpt_num_parameters() raises:
    """
    TEST.
    Pinned parameter counts at the reference pins (bias-free QKV
    default; `qkv_bias=True` restores the GPT-2 biased counts in
    parentheses):
      - MLP C=384           == 1,181,568
      - TransformerBlock C=384, H=8 == 1,773,312 (1,774,464 biased)
      - tied GPTModel  (V=50257, C=384, n_ctx=512, n_layer=8) == 33,682,560
        (33,691,776 biased)
      - untied GPTModel == 53,031,505 (53,040,721 biased)
      - tying removes V*C + V relative to untied.
    """
    comptime dtype = DType.float32
    var C = 384
    var H = 8
    var V = 50257
    var n_ctx = 512
    var n_layer = 8

    # MLP
    var mlp = MLP[dtype](n_embd=C)
    assert_true(mlp.num_parameters() == 1_181_568)

    # TransformerBlock (bias-free QKV default)
    var block = TransformerBlock[dtype](n_embd=C, n_head=H)
    assert_true(block.num_parameters() == 1_773_312)

    # Tied GPTModel
    var model_tied = GPTModel[dtype](
        V, n_ctx, C, H, n_layer, tie_weights=True
    )
    assert_true(model_tied.num_parameters() == 33_682_560)

    # Untied GPTModel
    var model_untied = GPTModel[dtype](
        V, n_ctx, C, H, n_layer, tie_weights=False
    )
    assert_true(model_untied.num_parameters() == 53_031_505)

    # Tying removes V*C + V
    var delta = V * C + V
    assert_true(model_untied.num_parameters() - model_tied.num_parameters() == delta)


# ═════════════════════════════════════════════════════════════════════════════
# Case 7b — qkv_bias flag
# ═════════════════════════════════════════════════════════════════════════════


def test_gpt_qkv_bias() raises:
    """
    TEST — the `qkv_bias` flag.
    Bias-free QKV by default, GPT-2 layout
    (biased QKV) on explicit opt-in.

    ASSERTS:
      - a biased block/model carries exactly 3*C more params per block
        than the default (the fused `c_attn` bias is `(3*C,)`).
      - `blk.attn.c_attn.bias` is absent from the default model's
        `named_parameters` and present under `qkv_bias=True`.
      - forward + backward run on the bias-free default and every
        parameter receives a gradient (the None-bias graph is complete).
    """
    comptime dtype = DType.float32
    var V = 16
    var C = 8
    var H = 2
    var n_ctx = 4
    var n_layer = 2

    # Block level: exactly one (3*C,) bias per block.
    var free = TransformerBlock[dtype](n_embd=C, n_head=H)
    var biased = TransformerBlock[dtype](
        n_embd=C, n_head=H, qkv_bias=True
    )
    assert_true(
        biased.num_parameters() - free.num_parameters() == 3 * C
    )

    # The bias key is absent/present in named_parameters.
    var named_free = free.named_parameters("blk.")
    var named_biased = biased.named_parameters("blk.")
    var found_free = False
    var found_biased = False
    for p in range(len(named_free)):
        if named_free[p].name == "blk.attn.c_attn.bias":
            found_free = True
    for p in range(len(named_biased)):
        if named_biased[p].name == "blk.attn.c_attn.bias":
            found_biased = True
    assert_false(found_free)
    assert_true(found_biased)

    # Model level: same per-block delta, plus a full forward+backward
    # smoke on the bias-free default.
    var m_free = GPTModel[dtype](V, n_ctx, C, H, n_layer, init_seed=7)
    var m_biased = GPTModel[dtype](
        V, n_ctx, C, H, n_layer, init_seed=7, qkv_bias=True
    )
    assert_true(
        m_biased.num_parameters() - m_free.num_parameters()
        == n_layer * 3 * C
    )
    var tokens = Tensor[DType.int64].d2([[0, 1, 2, 3], [4, 5, 6, 7]])
    var targets = Tensor[DType.int64].d2([[1, 2, 3, 0], [5, 6, 7, 4]])
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()
    var logits = m_free(tokens)
    var loss = criterion(logits.permute([0, 2, 1]), targets)
    loss.backward()
    var params = m_free.parameters()
    for p in range(len(params)):
        assert_true(params[p][].has_grad())


# ═════════════════════════════════════════════════════════════════════════════
# Case 10 — Composability
# ═════════════════════════════════════════════════════════════════════════════


def test_gpt_composability() raises:
    """
    TEST.
    Composability features:
      1. train/eval toggle propagates to all submodules.
      2. to_gpu/to_cpu round-trip preserves parameter count.
      3. named-parameter prefix plumbing works correctly.
    """
    comptime dtype = DType.float32
    var V = 16
    var C = 8
    var n_ctx = 8
    var n_head = 2
    var n_layer = 2

    # --- 1. train/eval toggle ---
    var model = GPTModel[dtype](V, n_ctx, C, n_head, n_layer, init_seed=42)
    assert_true(model.training)
    assert_true(model.h[0].training)

    model.eval()
    assert_false(model.training)
    assert_false(model.h[0].training)

    model.train()
    assert_true(model.training)
    assert_true(model.h[0].training)

    # --- 2. named-parameter prefix plumbing ---
    var named = model.named_parameters("gpt.")
    # Count should match len(parameters())
    var params = model.parameters()
    assert_true(len(named) == len(params))

    # Every name starts with "gpt."
    for p in range(len(named)):
        assert_true(named[p].name.startswith("gpt."))

    # Expected names present
    var has_wte_weight = False
    var has_wpe_weight = False
    var has_block_ln1 = False
    var has_ln_f = False
    for p in range(len(named)):
        if named[p].name == "gpt.wte_wpe.wte.weight":
            has_wte_weight = True
        if named[p].name == "gpt.wte_wpe.wpe.weight":
            has_wpe_weight = True
        if named[p].name == "gpt.h.0.ln_1.gamma":
            has_block_ln1 = True
        if named[p].name == "gpt.ln_f.gamma":
            has_ln_f = True
    assert_true(has_wte_weight)
    assert_true(has_wpe_weight)
    assert_true(has_block_ln1)
    assert_true(has_ln_f)

    # --- 3. to_gpu/to_cpu round-trip (guarded) ---
    comptime if has_accelerator():
        var g = model.to_gpu()
        var back = g.to_cpu()
        assert_true(back.num_parameters() == model.num_parameters())


# ═════════════════════════════════════════════════════════════════════════════
# Case 8 — Overfit one batch
# ═════════════════════════════════════════════════════════════════════════════
#
# NOTE (history): these two training gates lived in a separate
# `tests/test_gpt_training.mojo` while a zero-gradient bug was under
# investigation. Root cause turned out to be weight initialization, not the
# graph: a `"standard"` init name silently built an all-zero model whose
# `backward()` then faithfully propagated zeros. The fix was the single
# shared `Weights.initialize` vocabulary (`tenmo/weight_init.mojo`), and the
# regression fence is `tests/test_gpt_gradflow.mojo` (`./execute.sh gptflow`).
# With `gpt_train` green, the gates move back home so this file is the full
# 10-case gate its docstring promises.


def test_gpt_overfit_one_batch() raises:
    """
    TEST.
    End-to-end "does learning actually work?" check.  A tiny
    GPTModel (V=32, C=16, T=4, n_layer=2, n_head=2) is pushed over one
    (B, T) batch with SGD lr=0.5 for 60 steps.

    ASSERTS: lossN < loss0 * 0.5 AND lossN < 0.5.

    WHY: proves forward AND backward through every block, the embeddings,
    and the tied head work correctly end-to-end.
    """
    comptime dtype = DType.float32
    var B = 2
    var T = 4
    var C = 16
    var V = 32
    var n_head = 2
    var n_layer = 2

    var model = GPTModel[dtype](
        n_vocab=V, n_ctx=8, n_embd=C, n_head=n_head, n_layer=n_layer,
        tie_weights=True, init_seed=42
    )
    model.train()

    var tokens = Tensor[DType.int64].d2([[3, 17, 5, 22], [1, 4, 9, 30]])
    var targets = Tensor[DType.int64].d2([[17, 5, 22, 3], [4, 9, 30, 1]])

    var params = model.parameters()
    var optimizer = SGD[dtype](params, lr=0.5)
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()

    var loss0: Float64 = 0.0
    var lossN: Float64 = 0.0
    for step in range(60):
        var logits = model(tokens)  # (B, T, V)
        var logits_btv = logits.permute([0, 2, 1])  # (B, V, T)
        var loss = criterion(logits_btv, targets)

        var v = Float64(loss.item())
        if step == 0:
            loss0 = v
        lossN = v

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("  loss0=", loss0, " lossN=", lossN, " ratio=", lossN / loss0)
    assert_true(lossN < loss0 * 0.5)
    assert_true(lossN < 0.5)


# ═════════════════════════════════════════════════════════════════════════════
# Case 9 — Gradient wiring
# ═════════════════════════════════════════════════════════════════════════════


def test_gpt_gradient_wiring() raises:
    """
        TEST — one training step changes every parameter inside.
    ln_1/attn/ln_2/mlp/wte/wpe/ln_f (catches an off-by-one in the
    block-stacking loop).

    METHOD: snapshot each param via clone before a step; after
    forward+backward+step, assert every param tensor differs from its
    snapshot (at least one element changed beyond atol).
    """
    comptime dtype = DType.float32
    var V = 16
    var C = 8
    var n_ctx = 4  # T == n_ctx so all wpe rows get grads
    var n_head = 2
    var n_layer = 2
    var B = 2
    var T = 4

    var model = GPTModel[dtype](
        V, n_ctx, C, n_head, n_layer, tie_weights=True, init_seed=42
    )
    model.train()

    var tokens = Tensor[DType.int64].d2([[0, 1, 2, 3], [4, 5, 6, 7]])
    var targets = Tensor[DType.int64].d2([[1, 2, 3, 0], [5, 6, 7, 4]])

    var params = model.parameters()
    var snapshots = List[Tensor[dtype]]()
    for i in range(len(params)):
        snapshots.append(params[i][].clone())

    var optimizer = SGD[dtype](params, lr=0.5)
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()

    var logits = model(tokens)
    var logits_btv = logits.permute([0, 2, 1])
    var loss = criterion(logits_btv, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Verify at least one element changed in every parameter
    for i in range(len(params)):
        var snap = snapshots[i]
        var param = params[i][]
        var changed = False
        for j in range(snap.num_elements()):
            var diff = Float64(snap.get(j)) - Float64(param.get(j))
            if diff > 1e-6 or diff < -1e-6:
                changed = True
                break
        assert_true(changed, "param did not change after step")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll GPT stack tests passed!")
