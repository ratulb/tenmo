"""
Tests for bidirectional (BERT-style) attention on `SelfAttention`.

`SelfAttention` defaults to `causal=True` (GPT-style: query `i`
attends keys `j <= i`). Constructing with `causal=False` substitutes an
all-allow (T,T) mask so every query attends every key, and the optional
`pad_mask` argument to `forward_padded` (`(B,T)` bool, True = real token) zeroes attention
weight onto padded keys via a second `where`. This file is the stage gate
for that flag.

Unlike `test_attention.mojo` (which proves causality by showing the future
does NOT matter), these tests prove the complement:
  (a) default construction is still causal (no behavior change),
  (b) `causal=False` lets the future flow backward into early outputs,
  (c) `pad_mask` blocks padded keys (perturbing them changes nothing),
  (d) causal + padding compose,
  (e) gradients flow through the bidirectional + padding path (overfit).
"""

from std.testing import assert_true, assert_false, TestSuite
from std.sys import has_accelerator
from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape
from tenmo.shared.indexhelper import i, s
from tenmo.attention import SelfAttention
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.embedding import Embedding
from tenmo.net import Linear
from tenmo.optim import SGD


def _set_deterministic_weights_bidir[
    dtype: DType
](model: SelfAttention[dtype],):
    """Local copy of the deterministic-weight pattern.
    `test_attention.mojo` (helpers are module-private there):
    `c_attn.weight[r,c] = ((r+1)*10+(c+1))*0.01`, `c_attn.bias[r] =
    0.01*(r+1)`, `c_proj.weight[r,c] = (r+1)+0.05*(c+1)`,
    `c_proj.bias[r] = 0.02*(r+1)`. Lets every property below run on
    hand-reproducible numbers instead of random init."""
    var w = model.c_attn.weight
    for r in range(w.shape()[0]):
        for c in range(w.shape()[1]):
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


def test_bidir_default_is_causal() raises:
    """Default construction keeps `causal=True` and behaves causally.
    zeroing future positions 3..4 leaves outputs at 0..2 bit-comparable
    (atol=1e-4). Guards against the new flag changing existing callers."""
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=4, n_head=1, qkv_bias=True)
    assert_true(model.causal)
    _set_deterministic_weights_bidir(model)
    model.eval()
    var x = Tensor[dtype].randn(Shape(1, 5, 4), mean=0.0, std=0.1)
    var out_full = model(x)
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


def test_bidir_future_information_flows() raises:
    """Mirror image of the causality check: with `causal=False`.
    corrupting the future MUST change early outputs. If position 0 came
    out identical after destroying positions 3..4, the all-allow mask
    would be dead code and the encoder could not use right-side context."""
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=4, n_head=1, causal=False, qkv_bias=True)
    assert_false(model.causal)
    _set_deterministic_weights_bidir(model)
    model.eval()
    var x = Tensor[dtype].randn(Shape(1, 5, 4), mean=0.0, std=0.1)
    var out_full = model(x)
    var x_cut = x.clone()
    for t in range(3, 5):
        for c in range(4):
            x_cut[0, t, c] = 0.0
    var out_cut = model(x_cut)
    assert_false(
        out_full[i(0), i(0), s()].all_close[atol=1e-4](
            out_cut[i(0), i(0), s()]
        )
    )


def test_bidir_padding_mask_blocks_padded_keys() raises:
    """With `causal=False` and `pad_mask` marking position 3 padded.
    rewriting position 3's inputs must leave every real-position output
    unchanged (padded keys get exactly zero weight). Control: the same
    rewrite WITHOUT the mask must change position 0 (proves the test is
    sensitive — cf. `test_bidir_future_information_flows`)."""
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=4, n_head=1, causal=False, qkv_bias=True)
    _set_deterministic_weights_bidir(model)
    model.eval()
    var x = Tensor[dtype].randn(Shape(1, 4, 4), mean=0.0, std=0.1)
    var pad = Tensor[DType.bool].d2([[True, True, True, False]])
    var out_masked = model.forward_padded(x, pad)
    var x_pert = x.clone()
    for c in range(4):
        x_pert[0, 3, c] = 5.0
    var out_masked_pert = model.forward_padded(x_pert, pad)
    for t in range(3):
        assert_true(
            out_masked[i(0), i(t), s()].all_close[atol=1e-4](
                out_masked_pert[i(0), i(t), s()]
            )
        )
    # Control: no mask -> the perturbation must leak into position 0.
    var out_bare = model(x)
    var out_bare_pert = model(x_pert)
    assert_false(
        out_bare[i(0), i(0), s()].all_close[atol=1e-4](
            out_bare_pert[i(0), i(0), s()]
        )
    )


def test_bidir_causal_and_padding_compose() raises:
    """`causal=True` (default) plus `pad_mask`: both masks apply.
    Corrupting a padded FUTURE position must leave position 0 unchanged
    (either mask alone would suffice; this guards the combination path —
    the second `where` must not disturb the first)."""
    comptime dtype = DType.float32
    var model = SelfAttention[dtype](n_embd=4, n_head=1, qkv_bias=True)
    _set_deterministic_weights_bidir(model)
    model.eval()
    var x = Tensor[dtype].randn(Shape(1, 4, 4), mean=0.0, std=0.1)
    var pad = Tensor[DType.bool].d2([[True, True, True, False]])
    var out = model.forward_padded(x, pad)
    var x_pert = x.clone()
    for c in range(4):
        x_pert[0, 3, c] = 5.0
    var out_pert = model.forward_padded(x_pert, pad)
    assert_true(
        out[i(0), i(0), s()].all_close[atol=1e-4](
            out_pert[i(0), i(0), s()]
        )
    )


def test_bidir_overfit_one_batch() raises:
    """Backward flows through the bidirectional + padding path.
    Embedding -> Attention(causal=False) -> Linear head stack overfits one
    fixed batch (all-positions-real pad mask) under SGD, loss dropping to
    under half its initial value. Mirrors `test_attn_overfit_one_batch`
    (the LM-shifted-targets version) with next-position targets so the
    bidirectional model has something to learn."""
    comptime dtype = DType.float32
    var B = 2
    var T = 4
    var C = 16
    var V = 32
    var H = 4
    var model_out = SelfAttention[dtype](
        n_embd=C, n_head=H, causal=False, qkv_bias=True
    )
    var wte = Embedding[dtype](num_embeddings=V, embedding_dim=C)
    var head = Linear[dtype](C, V)
    model_out.train()
    var tokens = Tensor[DType.int64].d2([[3, 17, 5, 22], [1, 4, 9, 30]])
    var targets = Tensor[DType.int64].d2([[17, 5, 22, 3], [4, 9, 30, 1]])
    var pad = Tensor[DType.bool].d2(
        [[True, True, True, True], [True, True, True, True]]
    )
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
        var attn = model_out.forward_padded(x, pad)  # (B, T, C)
        var logits = head(attn)  # (B, T, V)
        var logits_btv = logits.permute([0, 2, 1])  # (B, V, T)
        var loss = criterion(logits_btv, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        var v = Float64(loss.item())
        if step == 0:
            loss0 = v
        lossN = v
    assert_true(lossN < loss0 * 0.5)


def test_bidir_gpu_roundtrip() raises:
    """`causal=False` survives a GPU round-trip with the parameter count.
    intact (288 for C=8, h=2: c_attn 8x24+24=216, c_proj 8x8+8=72).
    Compiles to a no-op where `has_accelerator()` is false."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var model = SelfAttention[dtype](
            n_embd=8, n_head=2, causal=False, qkv_bias=True
        )
        var g = model.to_gpu()
        var back = g.to_cpu()
        assert_false(back.causal)
        assert_true(back.num_parameters() == 288)


def main() raises:
    """Run every `test_bidir_*` function in this module."""
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll bidirectional attention tests passed!")
