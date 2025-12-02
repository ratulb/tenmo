from tenmo import Tensor
from shapes import Shape
from argminmax import Argmin, Argmax


fn main() raises:
    test_tensor_argmax_keepdims()
    print("\n=== Running all Argmin tests ===")
    test_tensor_argmin_1d()
    test_tensor_argmin_2d_basic()
    test_tensor_argmin_2d_keepdims()
    test_tensor_argmin_3d_axis_and_neg()
    test_tensor_argmin_keepdims_true_false_3d()
    print("✓ All Argmin tests passed\n")
    print("\n=== Running all Argmax tests ===")
    test_tensor_argmax_1d()
    test_tensor_argmax_2d_basic()
    test_tensor_argmax_2d_keepdims()
    test_tensor_argmax_3d_axis_and_neg()
    test_tensor_argmax_keepdims_true_false_3d()
    print("✓ All Argmax tests passed\n")

    print("passes")


from testing import assert_true


fn test_tensor_argmax_keepdims() raises:
    print("test_tensor_argmax_keepdims")
    alias dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 5.0, 3.0], [7.0, 2.0, 8.0]])
    var a1 = Argmax[dtype].argmax(t, axis=1, keepdims=False)
    var a2 = t.argmax(axis=1, keepdims=True)
    assert_true(a1.shape() == Shape(2))
    assert_true(a2.shape() == Shape(2, 1))
    assert_true(a1 == Tensor[DType.int32].d1([1, 2]))
    assert_true(a2 == Tensor.d2([[1], [2]]).to_dtype[DType.int32]())
    print("✓ Passed argmax keepdims test")


# ==========================================================
# Argmax Tests
# ==========================================================


fn test_tensor_argmax_1d() raises:
    print("test_tensor_argmax_1d")
    alias dtype = DType.float32
    var t = Tensor[dtype].d1([0.5, 2.3, 1.1, 2.3])
    var a = t.argmax(axis=0)
    assert_true(a == Tensor[DType.int32].scalar(1))
    print("✓ Passed 1D argmax test")


fn test_tensor_argmax_2d_basic() raises:
    print("test_tensor_argmax_2d_basic")
    alias dtype = DType.float32
    var t = Tensor[dtype].d2(
        [[1.0, 3.0, 2.0], [4.0, 0.5, 7.0], [2.5, 5.5, 1.0]]
    )
    var a0 = t.argmax(axis=0)
    var a1 = t.argmax(axis=1)
    assert_true(a0 == Tensor[DType.int32].d1([1, 2, 1]))
    assert_true(a1 == Tensor[DType.int32].d1([1, 2, 1]))
    print("✓ Passed 2D argmax basic test")


fn test_tensor_argmax_2d_keepdims() raises:
    print("test_tensor_argmax_2d_keepdims")
    alias dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 5.0, 3.0], [7.0, 2.0, 8.0]])
    var a1 = Argmax[dtype].argmax(t, axis=1, keepdims=False)
    var a2 = t.argmax(axis=1, keepdims=True)
    assert_true(a1.shape() == Shape(2))
    assert_true(a2.shape() == Shape(2, 1))
    assert_true(a1 == Tensor[DType.int32].d1([1, 2]))
    assert_true(a2 == Tensor[DType.int32].d2([[1], [2]]))
    print("✓ Passed argmax keepdims test")


fn test_tensor_argmax_3d_axis_and_neg() raises:
    print("test_tensor_argmax_3d_axis_and_neg")
    alias dtype = DType.float32
    var t = Tensor[dtype].arange(0, 24).reshape(Shape(2, 3, 4))
    var a0 = t.argmax(axis=0)
    var a1 = t.argmax(axis=1)
    var a2 = t.argmax(axis=2)
    var aneg1 = t.argmax(axis=-1)
    assert_true(a0.shape() == Shape(3, 4))
    assert_true(a1.shape() == Shape(2, 4))
    assert_true(a2.shape() == Shape(2, 3))
    assert_true(a2 == aneg1)
    print("✓ Passed 3D argmax positive/negative axes test")


fn test_tensor_argmax_keepdims_true_false_3d() raises:
    print("test_tensor_argmax_keepdims_true_false_3d")
    alias dtype = DType.float32
    var t = Tensor[dtype].arange(0, 24).reshape(Shape(2, 3, 4))
    var a_no = t.argmax(axis=1, keepdims=False)
    var a_yes = t.argmax(axis=1, keepdims=True)
    assert_true(a_no.shape() == Shape(2, 4))
    assert_true(a_yes.shape() == Shape(2, 1, 4))
    print("✓ Passed argmax keepdims true/false shape test")


# ==========================================================
# Argmin Tests
# ==========================================================


fn test_tensor_argmin_1d() raises:
    print("test_tensor_argmin_1d")
    alias dtype = DType.float32
    var t = Tensor[dtype].d1([5.0, -1.0, 3.0, -1.0])
    var a = t.argmin(axis=0)
    assert_true(a == Tensor[DType.int32].scalar(1))
    print("✓ Passed 1D argmin test")


fn test_tensor_argmin_2d_basic() raises:
    print("test_tensor_argmin_2d_basic")
    alias dtype = DType.float32
    var t = Tensor[dtype].d2(
        [[3.0, 1.0, 4.0], [2.0, 0.5, 6.0], [7.0, 5.5, 1.0]]
    )
    var a0 = t.argmin(axis=0)
    var a1 = t.argmin(axis=1)
    assert_true(a0 == Tensor[DType.int32].d1([1, 1, 2]))
    assert_true(a1 == Tensor[DType.int32].d1([1, 1, 2]))
    print("✓ Passed 2D argmin basic test")


fn test_tensor_argmin_2d_keepdims() raises:
    print("test_tensor_argmin_2d_keepdims")
    alias dtype = DType.float32
    var t = Tensor[dtype].d2([[5.0, 2.0, 3.0], [1.0, 9.0, 0.5]])
    var a1 = Argmin[dtype].argmin(t, axis=1, keepdims=False)
    var a2 = t.argmin(axis=1, keepdims=True)
    assert_true(a1.shape() == Shape(2))
    assert_true(a2.shape() == Shape(2, 1))
    assert_true(a1 == Tensor[DType.int32].d1([1, 2]))
    assert_true(a2 == Tensor[DType.int32].d2([[1], [2]]))
    print("✓ Passed argmin keepdims test")


fn test_tensor_argmin_3d_axis_and_neg() raises:
    print("test_tensor_argmin_3d_axis_and_neg")
    alias dtype = DType.float32
    var t = Tensor[dtype].arange(10, 34).reshape(Shape(2, 3, 4))
    var a0 = t.argmin(axis=0)
    var a1 = t.argmin(axis=1)
    var a2 = t.argmin(axis=2)
    var aneg1 = t.argmin(axis=-1)
    assert_true(a0.shape() == Shape(3, 4))
    assert_true(a1.shape() == Shape(2, 4))
    assert_true(a2.shape() == Shape(2, 3))
    assert_true(a2 == aneg1)
    print("✓ Passed 3D argmin positive/negative axes test")


fn test_tensor_argmin_keepdims_true_false_3d() raises:
    print("test_tensor_argmin_keepdims_true_false_3d")
    alias dtype = DType.float32
    var t = Tensor[dtype].arange(10, 34).reshape(Shape(2, 3, 4))
    var a_no = t.argmin(axis=1, keepdims=False)
    var a_yes = t.argmin(axis=1, keepdims=True)
    assert_true(a_no.shape() == Shape(2, 4))
    assert_true(a_yes.shape() == Shape(2, 1, 4))
    print("✓ Passed argmin keepdims true/false shape test")
