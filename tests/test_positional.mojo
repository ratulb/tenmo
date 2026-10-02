from std.testing import assert_true, assert_false, TestSuite
from std.sys import has_accelerator
from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape
from tenmo.embedding import Embedding
from tenmo.positional import PositionalEmbedding
from tenmo.net import Sequential, Module, Layer
from tenmo.shared.indexhelper import i, s


# =============================================================================
# Tests for PositionalEmbedding — the wpe layer.
# Prefix: test_pos_ on all test names.
# =============================================================================


def test_pos_init_shape() raises:
    comptime dtype = DType.float32
    var pe = PositionalEmbedding[dtype](n_ctx=512, n_embd=384)
    assert_true(pe.weight().shape() == Shape(512, 384))
    assert_true(pe.n_ctx == 512)
    assert_true(pe.n_embd == 384)
    assert_true(pe.weight().requires_grad)
    assert_true(pe.training)


def test_pos_num_parameters() raises:
    comptime dtype = DType.float32
    var pe = PositionalEmbedding[dtype](n_ctx=512, n_embd=384)
    assert_true(pe.num_parameters() == 512 * 384)


def test_pos_fwd_single_below_ctx() raises:
    comptime dtype = DType.float32
    var pe = PositionalEmbedding[dtype](
        n_ctx=10, n_embd=4, init_method="zero"
    )
    pe.weight().fill(2.0, i(2), s())  # position 2 vector = all 2.0
    var out = pe([2])  # (1, 4)
    assert_true(out.shape() == Shape(1, 4))
    assert_true(out[i(0), s()].all_close(Tensor[dtype].d1([2.0, 2.0, 2.0, 2.0])))


def test_pos_fwd_flags_grad_in_train() raises:
    comptime dtype = DType.float32
    var pe = PositionalEmbedding[dtype](n_ctx=5, n_embd=3, init_method="zero")
    pe.train()
    var out = pe([1])
    assert_true(out.requires_grad)


def test_pos_fwd_no_grad_in_eval() raises:
    comptime dtype = DType.float32
    var pe = PositionalEmbedding[dtype](n_ctx=5, n_embd=3, init_method="zero")
    pe.eval()
    var out = pe([1])
    assert_false(out.requires_grad)


def test_pos_position_ids_broadcast_values() raises:
    comptime dtype = DType.float32
    # Verify the (B, T) position ids are 0..T-1 in every batch row by feeding
    # the (B, T) id tensor into wpe and confirming out[b, t, :] == weight[t, :]
    # for all b, t (Gather consumes a contiguous id tensor -> (B,T,embd)).
    var pe = PositionalEmbedding[dtype](n_ctx=8, n_embd=3, init_method="zero")
    for p in range(8):
        pe.weight().fill(Scalar[dtype](Float64(p)), i(p), s())
    var ids = PositionalEmbedding[dtype].position_ids(n_ctx=8, B=3, T=4)
    assert_true(ids.shape() == Shape(3, 4))
    var out = pe(ids)  # (3, 4, 3)
    assert_true(out.shape() == Shape(3, 4, 3))
    for b in range(3):
        for t in range(4):
            assert_true(
                out[i(b), i(t), s()].all_close(
                    Tensor[dtype].full(Shape(3), Scalar[dtype](t))
                )
            )


def test_pos_position_ids_single_row() raises:
    comptime dtype = DType.float32
    var pe = PositionalEmbedding[dtype](n_ctx=8, n_embd=3, init_method="zero")
    for p in range(8):
        pe.weight().fill(Scalar[dtype](Float64(p)), i(p), s())
    var ids = PositionalEmbedding[dtype].position_ids(n_ctx=8, B=1, T=4)
    assert_true(ids.shape() == Shape(1, 4))
    var out = pe(ids)  # (1, 4, 3)
    assert_true(out.shape() == Shape(1, 4, 3))
    for b in range(1):
        for t in range(4):
            assert_true(
                out[i(b), i(t), s()].all_close(
                    Tensor[dtype].full(Shape(3), Scalar[dtype](t))
                )
            )


def test_pos_token_plus_position_parity() raises:
    comptime dtype = DType.float32
    # Known wte and wpe values -> sum must be computed elementwise.
    var wte = Embedding[dtype](num_embeddings=4, embedding_dim=5, init_method="zero")
    var wpe = PositionalEmbedding[dtype](n_ctx=4, n_embd=5, init_method="zero")
    for v in range(4):
        wte.weight.fill(Scalar[dtype](10.0), i(v), s())
    for p in range(4):
        wpe.weight().fill(Scalar[dtype](1.0), i(p), s())
    var tokens = Tensor[DType.int64].d1([0, 1, 2, 3])
    var x_tok = wte(tokens)  # (4, 5) rows all 10.0
    var x_pos = wpe(tokens)  # (4, 5) rows all 1.0
    var x = x_tok + x_pos
    for tk in range(4):
        assert_true(
            x[i(tk), s()].all_close(
                Tensor[dtype].full(Shape(5), Scalar[dtype](11.0))
            )
        )


def test_pos_bwd_grad_into_looked_up_rows() raises:
    comptime dtype = DType.float32
    var pe = PositionalEmbedding[dtype](n_ctx=5, n_embd=3, init_method="zero")
    var result = pe([2, 2, 4])
    var loss = result.sum()
    loss.backward()
    var grad = pe.weight().grad()
    # Row 2 contributes 2 x [1,1,1]; row 4 contributes 1 x [1,1,1]; rest 0.
    assert_true(
        grad[i(2), s()].all_close(Tensor[dtype].d1([2.0, 2.0, 2.0]))
    )
    assert_true(
        grad[i(4), s()].all_close(Tensor[dtype].d1([1.0, 1.0, 1.0]))
    )
    assert_true(grad[i(0), s()].all_close(Tensor[dtype].zeros(Shape(3))))
    assert_true(grad[i(1), s()].all_close(Tensor[dtype].zeros(Shape(3))))


def test_pos_layertrait_composability() raises:
    comptime dtype = DType.float32
    var pe = PositionalEmbedding[dtype](n_ctx=6, n_embd=3, init_method="zero")
    var params = pe.parameters()
    assert_true(len(params) == 1)
    var named = pe.named_parameters("wpe.")
    assert_true(len(named) == 1)
    # zero_grad must not raise and leaves grad at zero.
    pe.train()
    var out = pe([0])
    var loss = out.sum()
    loss.backward()
    pe.zero_grad()
    var grad = pe.weight().grad()
    assert_true(grad[i(0), s()].all_close(Tensor[dtype].zeros(Shape(3))))


def test_pos_sequential_integration() raises:
    """PositionalEmbedding drops into legacy Sequential via the Layer Variant."""
    comptime dtype = DType.float32
    var model = Sequential[dtype]()
    var pe = PositionalEmbedding[dtype](n_ctx=8, n_embd=4, init_method="zero")
    pe.weight().fill(1.0, i(2), s())  # row 2 = all 1.0
    model.append(Module[dtype](Layer[dtype](pe)))
    model.eval()

    # ── forward through the Sequential chain ──
    var out = model(Tensor[dtype].d1([2.0, 2.0]))
    assert_true(out.shape() == Shape(2, 4))
    assert_true(out[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0])))

    # ── parameter plumbing ──
    assert_true(model.num_parameters() == 8 * 4)
    var params = model.parameters()
    assert_true(len(params) == 1)
    var named = model.named_parameters("m.")
    assert_true(len(named) == 1)

    # ── train/eval toggle flips PositionalEmbedding grad tracking ──
    var m0 = model.modules[0]
    m0.train()
    assert_true(m0.layer[PositionalEmbedding[dtype]].training)
    m0.eval()
    assert_false(m0.layer[PositionalEmbedding[dtype]].training)

    # ── device round-trip returns an intact Sequential ──
    if has_accelerator():
        var model_gpu = model.to_gpu()
        var model_back = model_gpu.to_cpu()
        assert_true(model_back.num_parameters() == 8 * 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
