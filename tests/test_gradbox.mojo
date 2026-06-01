from tenmo.buffers import Buffer
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.common_utils import i, s
from tenmo.ndbuffer import NDBuffer
from tenmo.shapes import Shape
from tenmo.gradbox import Gradbox
from tenmo.unsqueeze import Unsqueeze
from tenmo.tensor import Tensor
from tenmo.intarray import IntArray


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_gradbox_unsqueeze_basic() raises:
    comptime dtype = DType.float32
    var _tmp0 = Gradbox[dtype].arange(6)
    var g = _tmp0.reshape(Shape(2, 3))

    var g1 = g.unsqueeze([0])
    var g2 = g.unsqueeze([-1])
    var g3 = g.unsqueeze([0, 2])

    assert_true(g1.shape() == Shape(1, 2, 3))
    assert_true(g2.shape() == Shape(2, 3, 1))
    assert_true(g3.shape() == Shape(1, 2, 1, 3))

    assert_true(g.flatten() == g1.flatten())
    assert_true(g.flatten() == g2.flatten())
    assert_true(g.flatten() == g3.flatten())


def test_gradbox_squeeze_basic() raises:
    comptime dtype = DType.float32
    var _tmp0 = Gradbox[dtype].arange(6)
    var g = _tmp0.reshape(Shape(1, 2, 3, 1))

    var g1 = g.squeeze()
    var g2 = g.squeeze([0])
    var g3 = g.squeeze([-1])
    var g4 = g.squeeze([0, -1])

    assert_true(g1.shape() == Shape(2, 3))
    assert_true(g2.shape() == Shape(2, 3, 1))
    assert_true(g3.shape() == Shape(1, 2, 3))
    assert_true(g4.shape() == Shape(2, 3))

    assert_true(g.flatten() == g1.flatten())
    assert_true(g.flatten() == g2.flatten())
    assert_true(g.flatten() == g3.flatten())
    assert_true(g.flatten() == g4.flatten())


def test_gradbox_squeeze_unsqueeze_symmetry() raises:
    comptime dtype = DType.float32
    var _tmp0 = Gradbox[dtype].arange(12)
    var g = _tmp0.reshape(Shape(2, 3, 2))

    var u = g.unsqueeze([0, 3])
    var s = u.squeeze()

    assert_true(g.shape() == s.shape())
    assert_true(g.flatten() == s.flatten())


def test_gradbox_unsqueeze_negative_axes() raises:
    comptime dtype = DType.float32
    var _tmp0 = Gradbox[dtype].arange(4)
    var g = _tmp0.reshape(Shape(2, 2))

    var g1 = g.unsqueeze([-1])
    var g2 = g.unsqueeze([-2])
    var g3 = g.unsqueeze([-3])  # should prepend

    assert_true(g1.shape() == Shape(2, 2, 1))
    assert_true(g2.shape() == Shape(2, 1, 2))
    assert_true(g3.shape() == Shape(1, 2, 2))

    assert_true(g.flatten() == g1.flatten())
    assert_true(g.flatten() == g2.flatten())
    assert_true(g.flatten() == g3.flatten())


def test_gradbox_squeeze_unsqueeze_multiple_axes() raises:
    comptime dtype = DType.float32
    var _tmp0 = Gradbox[dtype].arange(8)
    var g = _tmp0.reshape(Shape(1, 2, 1, 4, 1))

    var s = g.squeeze([0, 2, 4])
    assert_true(s.shape() == Shape(2, 4))
    var u = s.unsqueeze([0, 2, 4])
    assert_true(u.shape() == g.shape())

    assert_true(g.flatten() == s.flatten())
    assert_true(s.flatten() == u.flatten())


def test_gradbox_permute_basic() raises:
    comptime dtype = DType.float32
    g1 = Gradbox[dtype](Shape(3, 4))
    g1.buffer.fill(Scalar[dtype](1.0))
    p = g1.permute(IntArray([1, 0]))
    assert_true(p.shape() == Shape(4, 3))
    assert_true(g1.numels() == p.numels())


def test_gradbox_permute_3d() raises:
    comptime dtype = DType.float32
    g1 = Gradbox[dtype](Shape(3, 4, 5))
    g1.buffer.fill(Scalar[dtype](2.0))
    p = g1.permute(IntArray([2, 0, 1]))
    assert_true(p.shape() == Shape(5, 3, 4))
    assert_true(g1.numels() == p.numels())


def test_gradbox_permute_inverse() raises:
    comptime dtype = DType.float32
    g1 = Gradbox[dtype](Shape(2, 3, 4))
    g1.buffer.fill(Scalar[dtype](3.0))
    p = g1.permute(IntArray([1, 2, 0]))
    inv = p.permute(IntArray([2, 0, 1]))
    assert_true(inv.shape() == g1.shape())
    assert_true(inv.all_close(g1))


def test_gradbox_permute_identity() raises:
    comptime dtype = DType.float32
    g1 = Gradbox[dtype](Shape(3, 4, 5))
    g1.buffer.fill(Scalar[dtype](5.0))
    p = g1.permute(IntArray([0, 1, 2]))
    assert_true(p.shape() == g1.shape())
    assert_true(p.all_close(g1))


def test_gradbox_permute_singleton_dims() raises:
    comptime dtype = DType.float32
    g1 = Gradbox[dtype](Shape(1, 4, 1))
    g1.buffer.fill(Scalar[dtype](7.0))
    p = g1.permute(IntArray([2, 1, 0]))
    assert_true(p.shape() == Shape(1, 4, 1))
    assert_true(p.numels() == g1.numels())
    assert_true(p.all_close(g1))


def test_gradbox_permute_high_rank() raises:
    comptime dtype = DType.float32
    g1 = Gradbox[dtype](Shape(2, 3, 4, 5))
    g1.buffer.fill(Scalar[dtype](9.0))
    p = g1.permute(IntArray([3, 2, 1, 0]))
    assert_true(p.shape() == Shape(5, 4, 3, 2))
    assert_true(g1.numels() == p.numels())


def test_gradbox_squeeze_noop() raises:
    var g = Gradbox[DType.float32].zeros(Shape.of(2, 3), share=False)
    var s = g.squeeze()  # no size-1 axes -> should be same shape
    assert_true(s.shape() == Shape.of(2, 3))
    # values unchanged
    assert_true(s.all_close(g.as_tensor(requires_grad=False)))


def test_gradbox_squeeze_remove_all_singletons() raises:
    comptime dtype = DType.float32
    var g = Gradbox[DType.float32](
        NDBuffer[DType.float32].full(
            Shape.of(1, 3, 1), Scalar[DType.float32](1.0)
        ),
        share=False,
    )
    var s = g.squeeze()  # should remove axes 0 and 2 -> shape (3,)
    assert_true(s.shape() == Shape.of(3))
    # round-trip via tensor to validate ordering & values
    assert_true(
        s.as_tensor(requires_grad=False).all_close(Tensor[dtype].d1([1, 1, 1]))
    )


def test_gradbox_squeeze_with_axes_list() raises:
    comptime dtype = DType.float32
    var g = Gradbox[DType.float32](
        NDBuffer[DType.float32].full(
            Shape.of(2, 1, 3), Scalar[DType.float32](2.0)
        ),
        share=False,
    )
    # specify axis 1 to squeeze -> shape becomes (2,3)
    var s = g.squeeze(IntArray([1]))
    assert_true(s.shape() == Shape.of(2, 3))
    # values preserved
    assert_true(
        s.as_tensor(requires_grad=False).all_close(
            Tensor[dtype].d2([[2, 2, 2], [2, 2, 2]])
        )
    )


def test_gradbox_squeeze_preserves_value_semantics() raises:
    # create non-shared gradbox and squeeze; result must not be shared (share=False semantics)
    comptime dtype = DType.float32
    var g = Gradbox[DType.float32].full(
        Shape.of(1, 2, 1), Scalar[DType.float32](3.0), share=False
    )
    assert_true(g.is_shared() == False)
    var s = g.squeeze()
    assert_true(s.is_shared() == False)
    assert_true(s.shape() == Shape.of(2))
    assert_true(
        s.as_tensor(requires_grad=False).all_close(Tensor[dtype].d1([3, 3]))
    )


def test_gradbox_squeeze_after_broadcast_and_sum_over_broadcasted_axes() raises:
    comptime dtype = DType.float32
    var base = Gradbox[DType.float32](
        NDBuffer[DType.float32].full(
            Shape.of(1, 3, 1), Scalar[DType.float32](1.0)
        ),
        share=False,
    )
    # broadcast to (2,3,4)
    var broadcasted = base.broadcast_to(Shape.of(2, 3, 4), share=False)
    # sum over broadcasted axes back to (1,3,1) -- uses NDBuffer.sum_over_broadcasted_axes
    var collapsed = Gradbox[DType.float32].sum_over_broadcasted_axes(
        broadcasted, Shape.of(1, 3, 1)
    )
    # then squeeze to (3,)
    var s = collapsed.squeeze()
    assert_true(s.shape() == Shape.of(3))
    # expected values: each element should equal number of elements contributed along collapsed axes
    # broadcasted values were all 1.0; summed down to (1,3,1) from (2,3,4) -> multiplier = 2*4 = 8
    assert_true(
        s.as_tensor(requires_grad=False).all_close(
            Tensor[dtype].d1([8.0, 8.0, 8.0]).float()
        )
    )


def test_gradbox_squeeze_integration_with_unsqueeze_backward() raises:
    # This verifies behaviour of Unsqueeze.backward: gradient of the unsqueezed tensor
    # should be reduced (squeezed) back to original shape. Behaviourally this uses Gradbox.squeeze().
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )  # shape (2,2)
    # Unsqueeze axes [0, 2] -> new rank = 4, shape (1, 2, 1, 2)
    var u = Unsqueeze.forward[track_grad=True](
        a, IntArray([0, 2]), requires_grad=None
    )
    # do a simple op and backward
    var out = u.sum()
    out.backward()
    # after backward, a.grad() should be ones of original shape because sum reduces to ones
    assert_true(a.grad().all_close(Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]])))


def test_gradbox_squeeze_chain_of_ops() raises:
    # chain: Tensor -> Unsqueeze -> some ops -> backward uses Gradbox.squeeze internally
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0]]], requires_grad=True
    )  # shape (1,1,2)
    var u = Unsqueeze.forward[track_grad=True](
        a, IntArray([0, 2]), requires_grad=None
    )
    var v = u * 3.0
    var loss = v.sum()
    loss.backward()
    # grad should be 3.0 in the original shape (1,1,2)
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[3.0, 3.0]]])))


# ── __getitem__ [Idx] tests (the broken GPU path) ──────────────────────


def test_gradbox_getitem_shared_cpu() raises:
    """CPU: basic slice on a Gradbox created via shared constructor."""
    comptime dtype = DType.float32
    var ndb = NDBuffer[dtype].arange(12).reshape(Shape(3, 4))
    var g = Gradbox[dtype](ndb^, share=True)  # explicitly shared
    var row = g[i(1), s()]  # shape (4,), returns unshared (share=False)
    assert_true(row.shape() == Shape(4))
    # Use List[Int] accessor to read values (no shared check)
    assert_true(row[[0]] == 4.0 and row[[3]] == 7.0)
    # Integer index → rank-0 scalar
    var elem = g[i(1), i(2)]
    assert_true(elem.item() == 6.0)


def test_gradbox_getitem_multi_axis_cpu() raises:
    """CPU: multi-axis slice with mixed int/slice indices."""
    comptime dtype = DType.float32
    var ndb = NDBuffer[dtype].arange(24).reshape(Shape(2, 3, 4))
    var g = Gradbox[dtype](ndb^, share=True)
    var slice1 = g[i(0), s(), i(1)]
    assert_true(slice1.shape() == Shape(3))
    assert_true(slice1[[0]] == 1.0 and slice1[[2]] == 9.0)
    var elem = g[i(0), i(1), s()]
    assert_true(elem.shape() == Shape(4))
    assert_true(elem[[0]] == 4.0 and elem[[3]] == 7.0)


def test_gradbox_getitem_detach_share_true_cpu() raises:
    """CPU: detach(share=True) then slice."""
    comptime dtype = DType.float32
    var ndb = NDBuffer[dtype].full(Shape(3, 4), Scalar[dtype](7.0))
    var g = Gradbox[dtype](ndb^, share=True)
    var detached = g.detach(share=True)
    var sliced = detached[i(1), s()]
    assert_true(sliced.shape() == Shape(4))
    assert_true(sliced[[0]] == 7.0 and sliced[[3]] == 7.0)


# ── GPU __getitem__ tests (guarded) ────────────────────────────────────


def test_gpu_gradbox_getitem_device_state_propagation() raises:
    """GPU: slice propagates device_state so to_cpu reads correct data."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # Fill CPU Gradbox → GPU
        var g = Gradbox[dtype].full(Shape(3, 4), 42.0)
        g = g.to_gpu()
        # Slice via __getitem__ (the broken path)
        var sliced = g[i(1), s()]
        # Bring back and verify
        sliced = sliced.to_cpu()
        assert_true(sliced.shape() == Shape(4))
        assert_true(sliced[i(0)].item() == 42.0)
        assert_true(sliced[i(1)].item() == 42.0)
        assert_true(sliced[i(2)].item() == 42.0)
        assert_true(sliced[i(3)].item() == 42.0)


def test_gpu_gradbox_getitem_after_detach() raises:
    """GPU: detach(share=False) → slice (matching embedding test pattern)."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var g = Gradbox[dtype].full(Shape(3, 4), 42.0, share=False)
        g = g.to_gpu()
        # detach(share=False) returns unshared at Gradbox level,
        # but NDBuffer.is_shared() returns True on GPU (DeviceBuffer always ref-counted)
        var detached = g.detach(share=False)
        var sliced = detached[i(1), s()]
        sliced = sliced.to_cpu()
        assert_true(sliced[i(0)].item() == 42.0)


def test_gpu_gradbox_getitem_chained() raises:
    """GPU: chained slices on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var g = Gradbox[dtype].arange(24).reshape(Shape(2, 3, 4))
        g = g.to_gpu()
        # First slice: grab batch 0, all rows, all cols
        var s1 = g[i(0), s(), s()]
        s1 = s1.to_cpu()
        assert_true(s1.shape() == Shape(3, 4))
        assert_true(s1[i(0), i(0)].item() == 0.0)  # g[0,0,0]
        assert_true(s1[i(2), i(3)].item() == 11.0)  # g[0,2,3]


# ── original tests follow ──────────────────────────────────────────────


def test_gradbox_reverse_division() raises:
    comptime dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    result = 2 / gradbox
    assert_true(
        result.buffer.data_buffer()
        == Buffer[dtype]([2.0, 1.0, 0.6666667, 0.5, 0.4, 0.33333334])
    )


def test_gradbox_reverse_subtract() raises:
    comptime dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    result = 2 - gradbox
    assert_true(
        result.buffer.data_buffer() == Buffer[dtype]([1, 0, -1, -2, -3, -4])
    )


def test_gradbox_reshape() raises:
    comptime dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    reshaped = gradbox.reshape(Shape(3, 2))
    assert_true(reshaped[[2, 1]] == 6 and reshaped[[1, 1]] == 4)
    reshaped.zero_grad()
    assert_true(reshaped[[2, 1]] == 0 and reshaped[[1, 1]] == 0)
    assert_true(gradbox[[1, 2]] == 6 and gradbox[[0, 1]] == 2)


def test_gradbox_inplace_add() raises:
    comptime dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)

    buffer2 = Buffer[dtype]([11, 12, 13, 14, 15, 16])
    ndb2 = NDBuffer[dtype](buffer2^, Shape(2, 3))
    gradbox2 = Gradbox[dtype](ndb2^)

    gradbox += gradbox2
    assert_true(
        gradbox.buffer.buffer == Buffer[dtype]([12, 14, 16, 18, 20, 22])
    )
    assert_true(
        gradbox.buffer.buffer == Buffer[dtype]([12, 14, 16, 18, 20, 22])
    )


def test_gradbox_is_shared() raises:
    comptime dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    assert_true(
        gradbox.buffer.is_shared(),
        "Gradbox buffer is shared - assertion failed",
    )


def test_seed_gradbox() raises:
    comptime dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    assert_true(gradbox.buffer.buffer == Buffer[dtype]([1, 2, 3, 4, 5, 6]))
    gradbox.seed_grad(42)
    assert_true(
        gradbox.buffer.buffer == Buffer[dtype]([42, 42, 42, 42, 42, 42])
    )
