from buffers import Buffer
from testing import assert_true
from ndbuffer import NDBuffer
from shapes import Shape
from gradbox import Gradbox
from unsqueeze import Unsqueeze
from tenmo import Tensor
from intlist import IntList

fn main() raises:
    run = 1
    for _ in range(run):
        test_gradbox_is_shared()
        test_seed_gradbox()
        test_gradbox_inplace_add()
        test_gradbox_reshape()
        test_gradbox_reverse_subtract()
        test_gradbox_reverse_division()
        test_gradbox_squeeze_noop()
        test_gradbox_squeeze_remove_all_singletons()
        test_gradbox_squeeze_with_axes_list()
        test_gradbox_squeeze_preserves_value_semantics()
        test_gradbox_squeeze_after_broadcast_and_sum_over_broadcasted_axes()
        test_gradbox_squeeze_integration_with_unsqueeze_backward()
        test_gradbox_squeeze_chain_of_ops()
        test_gradbox_permute_basic()
        test_gradbox_permute_3d()
        test_gradbox_permute_inverse()
        test_gradbox_permute_identity()
        test_gradbox_permute_singleton_dims()
        test_gradbox_permute_high_rank()


fn test_gradbox_permute_basic() raises:
    alias dtype = DType.float32
    print("Running test_gradbox_permute_basic")
    g1 = Gradbox[dtype](Shape(3, 4))
    g1.buffer.fill(Scalar[dtype](1.0))
    p = g1.permute(IntList([1, 0]))
    assert_true(p.shape() == Shape(4, 3))
    assert_true(g1.numels() == p.numels())
    print("✓ Passed test_gradbox_permute_basic")

fn test_gradbox_permute_3d() raises:
    alias dtype = DType.float32
    print("Running test_gradbox_permute_3d")
    g1 = Gradbox[dtype](Shape(3, 4, 5))
    g1.buffer.fill(Scalar[dtype](2.0))
    p = g1.permute(IntList([2, 0, 1]))
    assert_true(p.shape() == Shape(5, 3, 4))
    assert_true(g1.numels() == p.numels())
    print("✓ Passed test_gradbox_permute_3d")

fn test_gradbox_permute_inverse() raises:
    alias dtype = DType.float32
    print("Running test_gradbox_permute_inverse")
    g1 = Gradbox[dtype](Shape(2, 3, 4))
    g1.buffer.fill(Scalar[dtype](3.0))
    p = g1.permute(IntList([1, 2, 0]))
    inv = p.permute(IntList([2, 0, 1]))
    assert_true(inv.shape() == g1.shape())
    assert_true(inv.all_close(g1))
    print("✓ Passed test_gradbox_permute_inverse")

fn test_gradbox_permute_identity() raises:
    alias dtype = DType.float32
    print("Running test_gradbox_permute_identity")
    g1 = Gradbox[dtype](Shape(3, 4, 5))
    g1.buffer.fill(Scalar[dtype](5.0))
    p = g1.permute(IntList([0, 1, 2]))
    assert_true(p.shape() == g1.shape())
    assert_true(p.all_close(g1))
    print("✓ Passed test_gradbox_permute_identity")

fn test_gradbox_permute_singleton_dims() raises:
    alias dtype = DType.float32
    print("Running test_gradbox_permute_singleton_dims")
    g1 = Gradbox[dtype](Shape(1, 4, 1))
    g1.buffer.fill(Scalar[dtype](7.0))
    p = g1.permute(IntList([2, 1, 0]))
    assert_true(p.shape() == Shape(1, 4, 1))
    assert_true(p.numels() == g1.numels())
    assert_true(p.all_close(g1))
    print("✓ Passed test_gradbox_permute_singleton_dims")

fn test_gradbox_permute_high_rank() raises:
    alias dtype = DType.float32
    print("Running test_gradbox_permute_high_rank")
    g1 = Gradbox[dtype](Shape(2, 3, 4, 5))
    g1.buffer.fill(Scalar[dtype](9.0))
    p = g1.permute(IntList([3, 2, 1, 0]))
    assert_true(p.shape() == Shape(5, 4, 3, 2))
    assert_true(g1.numels() == p.numels())
    print("✓ Passed test_gradbox_permute_high_rank")


fn test_gradbox_squeeze_noop() raises:
    print("test_gradbox_squeeze_noop")
    var g = Gradbox[DType.float32].zeros(Shape.of(2, 3), share=False)
    var s = g.squeeze()  # no size-1 axes -> should be same shape
    assert_true(s.shape() == Shape.of(2, 3))
    # values unchanged
    assert_true(s.all_close(g.as_tensor(requires_grad=False)))

fn test_gradbox_squeeze_remove_all_singletons() raises:
    print("test_gradbox_squeeze_remove_all_singletons")
    var g = Gradbox[DType.float32](
        NDBuffer[DType.float32].full(Shape.of(1, 3, 1), Scalar[DType.float32](1.0)), share=False
    )
    var s = g.squeeze()  # should remove axes 0 and 2 -> shape (3,)
    assert_true(s.shape() == Shape.of(3))
    # round-trip via tensor to validate ordering & values
    assert_true(s.as_tensor(requires_grad=False).all_close(Tensor.d1([1, 1, 1])))

fn test_gradbox_squeeze_with_axes_list() raises:
    print("test_gradbox_squeeze_with_axes_list")
    var g = Gradbox[DType.float32](
        NDBuffer[DType.float32].full(Shape.of(2, 1, 3), Scalar[DType.float32](2.0)), share=False
    )
    # specify axis 1 to squeeze -> shape becomes (2,3)
    var s = g.squeeze(IntList([1]))
    assert_true(s.shape() == Shape.of(2, 3))
    # values preserved
    assert_true(s.as_tensor(requires_grad=False).all_close(Tensor.d2([[2, 2, 2], [2, 2, 2]])))

fn test_gradbox_squeeze_preserves_value_semantics() raises:
    print("test_gradbox_squeeze_preserves_value_semantics")
    # create non-shared gradbox and squeeze; result must not be shared (share=False semantics)
    var g = Gradbox[DType.float32].full(Shape.of(1, 2, 1), Scalar[DType.float32](3.0), share=False)
    assert_true(g.shared() == False)
    var s = g.squeeze()
    assert_true(s.shared() == False)
    assert_true(s.shape() == Shape.of(2))
    assert_true(s.as_tensor(requires_grad=False).all_close(Tensor.d1([3, 3])))

fn test_gradbox_squeeze_after_broadcast_and_sum_over_broadcasted_axes() raises:
    print("test_gradbox_squeeze_after_broadcast_and_sum_over_broadcasted_axes")
    var base = Gradbox[DType.float32](
        NDBuffer[DType.float32].full(Shape.of(1, 3, 1), Scalar[DType.float32](1.0)), share=False
    )
    # broadcast to (2,3,4)
    var broadcasted = base.broadcast_to(Shape.of(2, 3, 4), share=False)
    # sum over broadcasted axes back to (1,3,1) -- uses NDBuffer.sum_over_broadcasted_axes
    var collapsed = Gradbox[DType.float32].sum_over_broadcasted_axes(broadcasted, Shape.of(1, 3, 1))
    # then squeeze to (3,)
    var s = collapsed.squeeze()
    assert_true(s.shape() == Shape.of(3))
    # expected values: each element should equal number of elements contributed along collapsed axes
    # broadcasted values were all 1.0; summed down to (1,3,1) from (2,3,4) -> multiplier = 2*4 = 8
    assert_true(s.as_tensor(requires_grad=False).all_close(Tensor.d1([8.0, 8.0, 8.0]).float()))

fn test_gradbox_squeeze_integration_with_unsqueeze_backward() raises:
    print("test_gradbox_squeeze_integration_with_unsqueeze_backward")
    # This verifies behaviour of Unsqueeze.backward: gradient of the unsqueezed tensor
    # should be reduced (squeezed) back to original shape. Behaviourally this uses Gradbox.squeeze().
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # shape (2,2)
    # Unsqueeze axes [0, 2] -> new rank = 4, shape (1, 2, 1, 2)
    var u = Unsqueeze.forward[track_grad=True](a, IntList([0, 2]), requires_grad=None)
    # do a simple op and backward
    var out = u.sum()
    out.backward()
    # after backward, a.grad() should be ones of original shape because sum reduces to ones
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))

fn test_gradbox_squeeze_chain_of_ops() raises:
    print("test_gradbox_squeeze_chain_of_ops")
    # chain: Tensor -> Unsqueeze -> some ops -> backward uses Gradbox.squeeze internally
    var a = Tensor.d3([[[1.0, 2.0]]], requires_grad=True)  # shape (1,1,2)
    var u = Unsqueeze.forward[track_grad=True](a, IntList([0, 2]), requires_grad=None)
    var v = u * 3.0
    var loss = v.sum()
    loss.backward()
    # grad should be 3.0 in the original shape (1,1,2)
    assert_true(a.grad().all_close(Tensor.d3([[[3.0, 3.0]]])))


fn test_gradbox_reverse_division() raises:
    print("test_gradbox_reverse_division")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    result = 2 / gradbox
    assert_true(
        result.buffer.data()
        == Buffer[dtype]([2.0, 1.0, 0.6666667, 0.5, 0.4, 0.33333334])
    )


fn test_gradbox_reverse_subtract() raises:
    print("test_gradbox_reverse_subtract")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    result = 2 - gradbox
    assert_true(result.buffer.data() == Buffer[dtype]([1, 0, -1, -2, -3, -4]))


fn test_gradbox_reshape() raises:
    print("test_gradbox_reshape")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    reshaped = gradbox.reshape(Shape(3, 2))
    assert_true(reshaped[[2, 1]] == 6 and reshaped[[1, 1]] == 4)
    reshaped.zero_grad()
    assert_true(reshaped[[2, 1]] == 0 and reshaped[[1, 1]] == 0)
    assert_true(gradbox[[1, 2]] == 6 and gradbox[[0, 1]] == 2)


fn test_gradbox_inplace_add() raises:
    print("test_gradbox_inplace_add")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)

    buffer2 = Buffer[dtype]([11, 12, 13, 14, 15, 16])
    ndb2 = NDBuffer[dtype](buffer2^, Shape(2, 3))
    gradbox2 = Gradbox[dtype](ndb2^)

    gradbox += gradbox2
    assert_true(
        gradbox.buffer.data() == Buffer[dtype]([12, 14, 16, 18, 20, 22])
    )
    assert_true(
        gradbox.buffer.shared_buffer.value()[]
        == Buffer[dtype]([12, 14, 16, 18, 20, 22])
    )
    assert_true(gradbox.buffer.buffer == None)


fn test_gradbox_is_shared() raises:
    print("test_gradbox_is_shared")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    assert_true(
        gradbox.buffer.shared(), "Gradbox buffer is shared - assertion failed"
    )


fn test_seed_gradbox() raises:
    print("test_seed_gradbox")
    alias dtype = DType.float32
    buffer = Buffer[dtype]([1, 2, 3, 4, 5, 6])
    ndb = NDBuffer[dtype](buffer^, Shape(2, 3))
    gradbox = Gradbox[dtype](ndb^)
    assert_true(
        gradbox.buffer.shared_buffer.value()[]
        == Buffer[dtype]([1, 2, 3, 4, 5, 6])
    )
    gradbox.seed_grad(42)
    assert_true(
        gradbox.buffer.shared_buffer.value()[]
        == Buffer[dtype]([42, 42, 42, 42, 42, 42])
    )
