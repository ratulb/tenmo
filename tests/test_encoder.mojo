"""
Tests for the BERT-style encoder stack (`tenmo/encoder.mojo`).

Covers `make_padding_mask`, `EncoderBlock` (bidirectional by
construction), `BertEmbeddings` (token + segment + position), and the
`BertMLMHead` / `BertClassifierHead` modules — every new library piece
except the attention flag itself (gated
by `tests/test_bidirectional_attention.mojo`). Conventions mirror
`test_attention.mojo`: property checks over hand-computation, plus
overfit smokes proving backward reaches every parameter.
"""

from std.testing import assert_true, assert_false, TestSuite
from std.sys import has_accelerator
from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape
from tenmo.shared.indexhelper import i, s
from tenmo.encoder import (
    EncoderBlock,
    BertEmbeddings,
    BertMLMHead,
    BertClassifierHead,
    BertForMLM,
    BertForSequenceClassification,
    make_padding_mask,
)
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.optim import SGD


def test_enc_make_padding_mask_values() raises:
    """`make_padding_mask` returns True exactly on `[0.
    Length)` per
    row: lengths `[3, 1]`, `T = 4` gives row 0 `[T,T,T,F]`, row 1
    `[T,F,F,F]`. Compared with `eq(...).all_true()` (bool-tensor
    equality idiom from `test_all_true_any_true.mojo`)."""
    var lengths = Tensor[DType.int64].d1([3, 1])
    var mask = make_padding_mask(lengths, 4)
    var expected = Tensor[DType.bool].d2(
        [[True, True, True, False], [True, False, False, False]]
    )
    assert_true(mask.eq(expected).all_true())


def test_enc_make_padding_mask_clamps() raises:
    """Lengths past `T` clamp instead of overrunning.
    Lengths `[9]`,
    `T = 2` gives `[T,T]` (truncation itself is the data builder's job;
    the mask stays total)."""
    var lengths = Tensor[DType.int64].d1([9])
    var mask = make_padding_mask(lengths, 2)
    var expected = Tensor[DType.bool].d2([[True, True]])
    assert_true(mask.eq(expected).all_true())


def test_enc_block_shape_and_bidirectional() raises:
    """`EncoderBlock` preserves `(B,T.
    C)` and is bidirectional by
    construction (`attn.causal == False`): corrupting future positions
    changes the position-0 output (the mirror of the causal
    no-lookahead property)."""
    comptime dtype = DType.float32
    var model = EncoderBlock[dtype](n_embd=8, n_head=2)
    assert_false(model.attn.causal)
    model.eval()
    var x = Tensor[dtype].randn(Shape(2, 5, 8), mean=0.0, std=0.1)
    var out = model(x)
    assert_true(out.shape()[0] == 2)
    assert_true(out.shape()[1] == 5)
    assert_true(out.shape()[2] == 8)
    var x_cut = x.clone()
    for t in range(3, 5):
        for c in range(8):
            x_cut[0, t, c] = 0.0
    var out_cut = model(x_cut)
    assert_false(
        out[i(0), i(0), s()].all_close[atol=1e-4](out_cut[i(0), i(0), s()])
    )


def test_enc_block_padding() raises:
    """`forward_padded` blocks padded keys through a full block.
    rewriting the padded position leaves real-position outputs
    unchanged, while the dense path is sensitive to the same rewrite
    (control)."""
    comptime dtype = DType.float32
    var model = EncoderBlock[dtype](n_embd=8, n_head=2)
    model.eval()
    var x = Tensor[dtype].randn(Shape(1, 4, 8), mean=0.0, std=0.1)
    var pad = Tensor[DType.bool].d2([[True, True, True, False]])
    var out_masked = model.forward_padded(x, pad)
    var x_pert = x.clone()
    for c in range(8):
        x_pert[0, 3, c] = 5.0
    var out_masked_pert = model.forward_padded(x_pert, pad)
    for t in range(3):
        assert_true(
            out_masked[i(0), i(t), s()].all_close[atol=1e-4](
                out_masked_pert[i(0), i(t), s()]
            )
        )
    var out_bare = model(x)
    var out_bare_pert = model(x_pert)
    assert_false(
        out_bare[i(0), i(0), s()].all_close[atol=1e-4](
            out_bare_pert[i(0), i(0), s()]
        )
    )


def test_enc_embeddings_shapes_and_segments() raises:
    """`BertEmbeddings` maps `(B,T)` ids to `(B,T.
    C)`; the segment table
    is live: all-zero vs all-one segment ids give different outputs
    (proves the `tte` path is wired, not dead code)."""
    comptime dtype = DType.float32
    var model = BertEmbeddings[dtype](
        n_vocab=32, n_ctx=16, n_embd=8, padding_idx=0
    )
    model.eval()
    var ids = Tensor[DType.int64].d2([[1, 5, 9, 3], [2, 7, 4, 6]])
    var out = model(ids)
    assert_true(out.shape()[0] == 2)
    assert_true(out.shape()[1] == 4)
    assert_true(out.shape()[2] == 8)
    var ones = Tensor[DType.int64].d2([[1, 1, 1, 1], [1, 1, 1, 1]])
    var out_seg = model.forward_segments(ids, ones)
    assert_false(
        out[i(0), s(), s()].all_close[atol=1e-4](out_seg[i(0), s(), s()])
    )


def test_enc_mlm_overfit() raises:
    """Backward reaches every MLM-path parameter.
    Embeddings + one
    encoder block + MLM head overfit fixed random ids (labels = inputs,
    all positions scored) under SGD — loss must more than halve. This
    is the REAL stage gate: the full pretraining-shaped graph trains."""
    comptime dtype = DType.float32
    var V = 16
    var C = 8
    var T = 4
    var B = 2
    var emb = BertEmbeddings[dtype](
        n_vocab=V, n_ctx=8, n_embd=C, padding_idx=0, init_seed=7
    )
    var block = EncoderBlock[dtype](n_embd=C, n_head=2, init_seed=7)
    var head = BertMLMHead[dtype](n_embd=C, n_vocab=V, init_seed=7)
    emb.train()
    block.train()
    head.train()
    var ids = Tensor[DType.int64].d2([[3, 7, 1, 12], [5, 2, 9, 4]])
    var params = emb.parameters()
    var bparams = block.parameters()
    for p in range(len(bparams)):
        params.append(bparams[p])
    var hparams = head.parameters()
    for p in range(len(hparams)):
        params.append(hparams[p])
    var optimizer = SGD[dtype](params, lr=0.5)
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()
    var loss0: Float64 = 0.0
    var lossN: Float64 = 0.0
    for step in range(30):
        var x = emb(ids)  # (B, T, C)
        var h = block(x)  # (B, T, C)
        var logits = head(h)  # (B, T, V)
        var lbt = logits.permute([0, 2, 1])  # (B, V, T)
        var loss = criterion(lbt, ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        var v = Float64(loss.item())
        if step == 0:
            loss0 = v
        lossN = v
    assert_true(lossN < loss0 * 0.5)


def test_enc_classifier_overfit() raises:
    """The classifier head learns from the `[CLS]` (first-token) slice.
    fixed random `(B,T,C)` inputs with fixed binary labels overfit under
    SGD — loss must more than halve. Proves the slice/reshape/linear
    path carries gradient into the encoder outputs."""
    comptime dtype = DType.float32
    var B = 4
    var T = 5
    var C = 8
    var head = BertClassifierHead[dtype](n_embd=C, n_labels=2)
    head.train()
    var x = Tensor[dtype].randn(Shape(B, T, C), mean=0.0, std=1.0)
    var labels = Tensor[DType.int64].d1([0, 1, 1, 0])
    var params = head.parameters()
    var optimizer = SGD[dtype](params, lr=0.5)
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()
    var loss0: Float64 = 0.0
    var lossN: Float64 = 0.0
    for step in range(40):
        var logits = head(x)  # (B, 2)
        var loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        var v = Float64(loss.item())
        if step == 0:
            loss0 = v
        lossN = v
    assert_true(lossN < loss0 * 0.5)


def test_enc_gpu_roundtrip() raises:
    """Encoder blocks survive a GPU round-trip with the parameter count.
    intact. C=8, h=2: attention (8x24+24 + 8x8+8 = 288) + MLP
    (8x32+32 + 32x8+8 = 552) + norms (2 x 2x8 = 32) = 872.
    Compiles to a no-op where `has_accelerator()` is false."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var model = EncoderBlock[dtype](n_embd=8, n_head=2)
        assert_true(model.num_parameters() == 872)
        var g = model.to_gpu()
        var back = g.to_cpu()
        assert_false(back.attn.causal)
        assert_true(back.num_parameters() == 872)


def test_enc_wrappers_shapes_and_overfit() raises:
    """`BertForMLM` and `BertForSequenceClassification` shrink.
    to the
    toy scale (V=32, C=8, 1 layer, seeded): MLM gives `(B,T,V)`, the
    classifier `(B,2)`; one classifier backward leaves NONZERO grads
    in the head, a block, and the embedding table (direct proof the
    graph spans the wrapper end to end), and 40 small SGD steps
    (lr=0.1) decrease the loss. No halving gate here by design: the
    single-`[CLS]`-vector bottleneck supervises too sparsely to halve
    reliably at toy scale (and lr=0.5 oscillates instead of
    descending) — optimization hardness, not graph damage, as the
    nonzero grads prove."""
    comptime dtype = DType.float32
    var B = 4
    var T = 6
    var ids = Tensor[DType.int64].d2(
        [[1, 5, 9, 3, 2, 7], [2, 7, 4, 6, 1, 3], [3, 1, 8, 2, 5, 4],
         [4, 6, 2, 7, 3, 1]]
    )
    var mlm = BertForMLM[dtype](
        n_vocab=32, n_ctx=8, n_embd=8, n_head=2, n_layer=1,
        dropout_p=0.0, init_seed=7,
    )
    mlm.train()
    var logits = mlm(ids)
    assert_true(logits.shape()[0] == B)
    assert_true(logits.shape()[1] == T)
    assert_true(logits.shape()[2] == 32)

    var clf = BertForSequenceClassification[dtype](
        n_vocab=32, n_ctx=8, n_embd=8, n_head=2, n_layer=1,
        dropout_p=0.0, init_seed=7,
    )
    clf.train()
    var s2 = clf(ids)
    assert_true(s2.shape()[0] == B and s2.shape()[1] == 2)

    var labels = Tensor[DType.int64].d1([0, 1, 1, 0])
    var params = clf.parameters()
    var optimizer = SGD[dtype](params, lr=0.1)
    var criterion = CrossEntropyLoss[dtype](reduction="mean")
    criterion.train()
    var out = clf(ids)
    var loss = criterion(out, labels)
    optimizer.zero_grad()
    loss.backward()
    # Spot-check the three spans: first param (embedding table),
    # a middle one (encoder block), the second-to-last (classifier
    # weight). NOTE the last param (the head bias) is skipped by
    # design: for mean-reduced CE the bias grad is mathematically
    # ~zero (per-sample softmax grads sum to zero over classes), so a
    # nonzero assertion there would fail on correct code.
    var g0 = params[0][].grad().sum().item()
    var gm = params[len(params) // 2][].grad().sum().item()
    var g1 = params[len(params) - 2][].grad().sum().item()
    assert_true(g0 != 0.0)
    assert_true(gm != 0.0)
    assert_true(g1 != 0.0)
    var loss0 = Float64(loss.item())
    for _ in range(40):
        var o = clf(ids)
        var l = criterion(o, labels)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    var lossN = Float64(criterion(clf(ids), labels).item())
    assert_true(lossN < loss0)


def main() raises:
    """Run every `test_enc_*` function in this module."""
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll encoder tests passed!")
