from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.argminmax import Argmin, Argmax
from std.sys import has_accelerator


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll argminmax tests passed!")


from std.testing import assert_true, TestSuite


fn test_tensor_argmax_keepdims() raises:
    print("test_tensor_argmax_keepdims")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 5.0, 3.0], [7.0, 2.0, 8.0]])
    var a1 = Argmax[dtype].argmax(t, axis=1, keepdims=False)
    var a2 = t.argmax(axis=1, keepdims=True)
    assert_true(a1.shape() == Shape(2))
    assert_true(a2.shape() == Shape(2, 1))
    assert_true(a1 == Tensor[DType.int32].d1([1, 2]))
    assert_true(a2 == Tensor[dtype].d2([[1], [2]]).to_dtype[DType.int32]())
    print("Passed argmax keepdims test")


# ==========================================================
# Argmax Tests
# ==========================================================


fn test_tensor_argmax_1d() raises:
    print("test_tensor_argmax_1d")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([0.5, 2.3, 1.1, 2.3])
    var a = t.argmax(axis=0)
    assert_true(a == Tensor[DType.int32].scalar(1))
    print("Passed 1D argmax test")


fn test_tensor_argmax_2d_basic() raises:
    print("test_tensor_argmax_2d_basic")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2(
        [[1.0, 3.0, 2.0], [4.0, 0.5, 7.0], [2.5, 5.5, 1.0]]
    )
    var a0 = t.argmax(axis=0)
    var a1 = t.argmax(axis=1)
    assert_true(a0 == Tensor[DType.int32].d1([1, 2, 1]))
    assert_true(a1 == Tensor[DType.int32].d1([1, 2, 1]))
    print("Passed 2D argmax basic test")


fn test_tensor_argmax_2d_keepdims() raises:
    print("test_tensor_argmax_2d_keepdims")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 5.0, 3.0], [7.0, 2.0, 8.0]])
    var a1 = Argmax[dtype].argmax(t, axis=1, keepdims=False)
    var a2 = t.argmax(axis=1, keepdims=True)
    assert_true(a1.shape() == Shape(2))
    assert_true(a2.shape() == Shape(2, 1))
    assert_true(a1 == Tensor[DType.int32].d1([1, 2]))
    assert_true(a2 == Tensor[DType.int32].d2([[1], [2]]))
    print("Passed argmax keepdims test")


fn test_tensor_argmax_3d_axis_and_neg() raises:
    print("test_tensor_argmax_3d_axis_and_neg")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0, 24)
    t = a.reshape(Shape(2, 3, 4))
    var a0 = t.argmax(axis=0)
    var a1 = t.argmax(axis=1)
    var a2 = t.argmax(axis=2)
    var aneg1 = t.argmax(axis=-1)
    assert_true(a0.shape() == Shape(3, 4))
    assert_true(a1.shape() == Shape(2, 4))
    assert_true(a2.shape() == Shape(2, 3))
    assert_true(a2 == aneg1)
    print("Passed 3D argmax positive/negative axes test")


fn test_tensor_argmax_keepdims_true_false_3d() raises:
    print("test_tensor_argmax_keepdims_true_false_3d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0, 24)
    t = a.reshape(Shape(2, 3, 4))
    var a_no = t.argmax(axis=1, keepdims=False)
    var a_yes = t.argmax(axis=1, keepdims=True)
    assert_true(a_no.shape() == Shape(2, 4))
    assert_true(a_yes.shape() == Shape(2, 1, 4))
    print("Passed argmax keepdims true/false shape test")


# ==========================================================
# Argmin Tests
# ==========================================================


fn test_tensor_argmin_1d() raises:
    print("test_tensor_argmin_1d")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([5.0, -1.0, 3.0, -1.0])
    var a = t.argmin(axis=0)
    assert_true(a == Tensor[DType.int32].scalar(1))
    print("Passed 1D argmin test")


fn test_tensor_argmin_2d_basic() raises:
    print("test_tensor_argmin_2d_basic")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2(
        [[3.0, 1.0, 4.0], [2.0, 0.5, 6.0], [7.0, 5.5, 1.0]]
    )
    var a0 = t.argmin(axis=0)
    var a1 = t.argmin(axis=1)
    a0.print()
    assert_true(a0 == Tensor[DType.int32].d1([1, 1, 2]))
    assert_true(a1 == Tensor[DType.int32].d1([1, 1, 2]))
    print("Passed 2D argmin basic test")


fn test_tensor_argmin_2d_keepdims() raises:
    print("test_tensor_argmin_2d_keepdims")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[5.0, 2.0, 3.0], [1.0, 9.0, 0.5]])
    var a1 = Argmin[dtype].argmin(t, axis=1, keepdims=False)
    var a2 = t.argmin(axis=1, keepdims=True)
    assert_true(a1.shape() == Shape(2))
    assert_true(a2.shape() == Shape(2, 1))
    assert_true(a1 == Tensor[DType.int32].d1([1, 2]))
    assert_true(a2 == Tensor[DType.int32].d2([[1], [2]]))
    print("Passed argmin keepdims test")


fn test_tensor_argmin_3d_axis_and_neg() raises:
    print("test_tensor_argmin_3d_axis_and_neg")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(10, 34)
    t = a.reshape(Shape(2, 3, 4))
    var a0 = t.argmin(axis=0)
    var a1 = t.argmin(axis=1)
    var a2 = t.argmin(axis=2)
    var aneg1 = t.argmin(axis=-1)
    assert_true(a0.shape() == Shape(3, 4))
    assert_true(a1.shape() == Shape(2, 4))
    assert_true(a2.shape() == Shape(2, 3))
    assert_true(a2 == aneg1)
    print("Passed 3D argmin positive/negative axes test")


fn test_tensor_argmin_keepdims_true_false_3d() raises:
    print("test_tensor_argmin_keepdims_true_false_3d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(10, 34)
    t = a.reshape(Shape(2, 3, 4))
    var a_no = t.argmin(axis=1, keepdims=False)
    var a_yes = t.argmin(axis=1, keepdims=True)
    assert_true(a_no.shape() == Shape(2, 4))
    assert_true(a_yes.shape() == Shape(2, 1, 4))
    print("Passed argmin keepdims true/false shape test")


# ══════════════════════════════════════════════════════════════════════════════
# ARGMIN/ARGMAX EXHAUSTIVE TESTS
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# 1D Tests - CPU
# ──────────────────────────────────────────────────────────────────────────────


fn test_argmin_1d_basic_cpu() raises:
    print("test_argmin_1d_basic_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([5.0, 2.0, 8.0, 1.0, 9.0])
    var result = t.argmin(axis=0).reshape(Shape(1))
    assert_true(result == Tensor[DType.int32].d1([3]))
    print("✓ Passed")


fn test_argmax_1d_basic_cpu() raises:
    print("test_argmax_1d_basic_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([5.0, 2.0, 8.0, 1.0, 9.0])
    var result = t.argmax(axis=0).reshape(Shape(1))
    assert_true(result == Tensor[DType.int32].d1([4]))
    print("✓ Passed")


fn test_argmin_1d_negative_values_cpu() raises:
    print("test_argmin_1d_negative_values_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([-5.0, -2.0, -8.0, -1.0, -9.0])
    var result = t.argmin(axis=0).reshape(Shape(1))
    assert_true(result == Tensor[DType.int32].d1([4]))
    print("✓ Passed")


fn test_argmax_1d_negative_values_cpu() raises:
    print("test_argmax_1d_negative_values_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([-5.0, -2.0, -8.0, -1.0, -9.0])
    var result = t.argmax(axis=0).reshape(Shape(1))
    assert_true(result == Tensor[DType.int32].d1([3]))
    print("✓ Passed")


fn test_argmin_1d_keepdims_cpu() raises:
    print("test_argmin_1d_keepdims_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([5.0, 2.0, 8.0, 1.0, 9.0])
    var result = t.argmin(axis=0, keepdims=True)
    assert_true(result.shape() == Shape(1))
    assert_true(result == Tensor[DType.int32].d1([3]))
    print("✓ Passed")


# ──────────────────────────────────────────────────────────────────────────────
# 2D Tests - CPU
# ──────────────────────────────────────────────────────────────────────────────


fn test_argmin_2d_axis0_cpu() raises:
    print("test_argmin_2d_axis0_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2(
        [[3.0, 1.0, 4.0], [2.0, 0.5, 6.0], [7.0, 5.5, 1.0]]
    )
    var result = t.argmin(axis=0)
    assert_true(result == Tensor[DType.int32].d1([1, 1, 2]))
    print("✓ Passed")


fn test_argmin_2d_axis1_cpu() raises:
    print("test_argmin_2d_axis1_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2(
        [[3.0, 1.0, 4.0], [2.0, 0.5, 6.0], [7.0, 5.5, 1.0]]
    )
    var result = t.argmin(axis=1)
    assert_true(result == Tensor[DType.int32].d1([1, 1, 2]))
    print("✓ Passed")


fn test_argmax_2d_axis0_cpu() raises:
    print("test_argmax_2d_axis0_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2(
        [[3.0, 1.0, 4.0], [2.0, 0.5, 6.0], [7.0, 5.5, 1.0]]
    )
    var result = t.argmax(axis=0)
    assert_true(result == Tensor[DType.int32].d1([2, 2, 1]))
    print("✓ Passed")


fn test_argmax_2d_axis1_cpu() raises:
    print("test_argmax_2d_axis1_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2(
        [[3.0, 1.0, 4.0], [2.0, 0.5, 6.0], [7.0, 5.5, 1.0]]
    )
    var result = t.argmax(axis=1)
    assert_true(result == Tensor[DType.int32].d1([2, 2, 0]))
    print("✓ Passed")


fn test_argmin_2d_keepdims_axis0_cpu() raises:
    print("test_argmin_2d_keepdims_axis0_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[3.0, 1.0, 4.0], [2.0, 0.5, 6.0]])
    var result = t.argmin(axis=0, keepdims=True)
    assert_true(result.shape() == Shape(1, 3))
    assert_true(result == Tensor[DType.int32].d2([[1, 1, 0]]))
    print("✓ Passed")


fn test_argmax_2d_keepdims_axis1_cpu() raises:
    print("test_argmax_2d_keepdims_axis1_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[3.0, 1.0, 4.0], [2.0, 0.5, 6.0]])
    var result = t.argmax(axis=1, keepdims=True)
    assert_true(result.shape() == Shape(2, 1))
    assert_true(result == Tensor[DType.int32].d2([[2], [2]]))
    print("✓ Passed")


fn test_argmin_2d_negative_axis_cpu() raises:
    print("test_argmin_2d_negative_axis_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[3.0, 1.0, 4.0], [2.0, 0.5, 6.0]])
    var result = t.argmin(axis=-1)
    assert_true(result == Tensor[DType.int32].d1([1, 1]))
    print("✓ Passed")


fn test_argmax_2d_single_row_cpu() raises:
    print("test_argmax_2d_single_row_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[3.0, 1.0, 4.0, 2.0]])
    var result = t.argmax(axis=1)
    assert_true(result == Tensor[DType.int32].d1([2]))
    print("✓ Passed")


fn test_argmin_2d_single_column_cpu() raises:
    print("test_argmin_2d_single_column_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[3.0], [1.0], [4.0], [2.0]])
    var result = t.argmin(axis=0)
    assert_true(result == Tensor[DType.int32].d1([1]))
    print("✓ Passed")


# ──────────────────────────────────────────────────────────────────────────────
# 3D Tests - CPU
# ──────────────────────────────────────────────────────────────────────────────


fn test_argmin_3d_axis0_cpu() raises:
    print("test_argmin_3d_axis0_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
    )
    var result = t.argmin(axis=0)
    assert_true(result == Tensor[DType.int32].d2([[1, 1], [1, 1]]))
    print("✓ Passed")


fn test_argmin_3d_axis1_cpu() raises:
    print("test_argmin_3d_axis1_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
    )
    var result = t.argmin(axis=1)
    assert_true(result == Tensor[DType.int32].d2([[0, 0], [0, 0]]))
    print("✓ Passed")


fn test_argmin_3d_axis2_cpu() raises:
    print("test_argmin_3d_axis2_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
    )
    var result = t.argmin(axis=2)
    assert_true(result == Tensor[DType.int32].d2([[0, 0], [0, 0]]))
    print("✓ Passed")


fn test_argmax_3d_axis0_cpu() raises:
    print("test_argmax_3d_axis0_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
    )
    var result = t.argmax(axis=0)
    assert_true(result == Tensor[DType.int32].d2([[0, 0], [0, 0]]))
    print("✓ Passed")


fn test_argmax_3d_axis1_cpu() raises:
    print("test_argmax_3d_axis1_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
    )
    var result = t.argmax(axis=1)
    assert_true(result == Tensor[DType.int32].d2([[1, 1], [1, 1]]))
    print("✓ Passed")


fn test_argmax_3d_axis2_cpu() raises:
    print("test_argmax_3d_axis2_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
    )
    var result = t.argmax(axis=2)
    assert_true(result == Tensor[DType.int32].d2([[1, 1], [1, 1]]))
    print("✓ Passed")


fn test_argmin_3d_keepdims_cpu() raises:
    print("test_argmin_3d_keepdims_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
    )
    var result = t.argmin(axis=1, keepdims=True)
    assert_true(result.shape() == Shape(2, 1, 2))
    print("✓ Passed")


# ──────────────────────────────────────────────────────────────────────────────
# 4D Tests - CPU
# ──────────────────────────────────────────────────────────────────────────────


fn test_argmin_4d_axis0_cpu() raises:
    print("test_argmin_4d_axis0_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d4(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]],
        ]
    )
    var result = t.argmin(axis=0)
    assert_true(result.shape() == Shape(2, 2, 2))
    assert_true(
        result == Tensor[DType.int32].d3([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    )
    print("✓ Passed")


fn test_argmax_4d_axis3_cpu() raises:
    print("test_argmax_4d_axis3_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d4(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]],
        ]
    )
    var result = t.argmax(axis=3)
    assert_true(result.shape() == Shape(2, 2, 2))
    assert_true(
        result == Tensor[DType.int32].d3([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    )
    print("✓ Passed")


fn test_argmin_4d_negative_axis_cpu() raises:
    print("test_argmin_4d_negative_axis_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d4(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]],
        ]
    )
    var result = t.argmin(axis=-2)  # axis=2
    assert_true(result.shape() == Shape(2, 2, 2))
    print("✓ Passed")


# ──────────────────────────────────────────────────────────────────────────────
# Edge Cases - CPU
# ──────────────────────────────────────────────────────────────────────────────


fn test_argmin_all_same_values_cpu() raises:
    print("test_argmin_all_same_values_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    var result = t.argmin(axis=1)
    # Should return first occurrence (index 0)
    assert_true(result == Tensor[DType.int32].d1([0, 0]))
    print("✓ Passed")


fn test_argmax_all_same_values_cpu() raises:
    print("test_argmax_all_same_values_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    var result = t.argmax(axis=1)
    # Should return first occurrence (index 0)
    assert_true(result == Tensor[DType.int32].d1([0, 0]))
    print("✓ Passed")


fn test_argmin_single_element_cpu() raises:
    print("test_argmin_single_element_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[5.0]])
    var result = t.argmin(axis=0)
    assert_true(result == Tensor[DType.int32].d1([0]))
    print("✓ Passed")


fn test_argmax_with_zeros_cpu() raises:
    print("test_argmax_with_zeros_cpu")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[0.0, -1.0, 2.0], [0.0, 0.0, 0.0]])
    var result = t.argmax(axis=1)
    assert_true(result == Tensor[DType.int32].d1([2, 0]))
    print("✓ Passed")


# ──────────────────────────────────────────────────────────────────────────────
# GPU Tests - 1D
# ──────────────────────────────────────────────────────────────────────────────


fn test_argmin_1d_basic_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_1d_basic_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d1([5.0, 2.0, 8.0, 1.0, 9.0])
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=0)
        var result_cpu = result.to_cpu().reshape(Shape(1))
        assert_true(result_cpu == Tensor[DType.int32].d1([3]))
        print("✓ Passed")


fn test_argmax_1d_basic_gpu() raises:
    comptime if has_accelerator():
        print("test_argmax_1d_basic_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d1([5.0, 2.0, 8.0, 1.0, 9.0])
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmax(axis=0)
        var result_cpu = result.to_cpu().reshape(Shape(1))
        assert_true(result_cpu == Tensor[DType.int32].d1([4]))
        print("✓ Passed")


fn test_argmin_1d_negative_values_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_1d_negative_values_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d1([-5.0, -2.0, -8.0, -1.0, -9.0])
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=0)
        var result_cpu = result.to_cpu().reshape(Shape(1))
        assert_true(result_cpu == Tensor[DType.int32].d1([4]))
        print("✓ Passed")


fn test_argmax_1d_negative_values_gpu() raises:
    comptime if has_accelerator():
        print("test_argmax_1d_negative_values_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d1([-5.0, -2.0, -8.0, -1.0, -9.0])
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmax(axis=0)
        var result_cpu = result.to_cpu().reshape(Shape(1))
        assert_true(result_cpu == Tensor[DType.int32].d1([3]))
        print("✓ Passed")


# ──────────────────────────────────────────────────────────────────────────────
# GPU Tests - 2D
# ──────────────────────────────────────────────────────────────────────────────


fn test_argmin_2d_axis0_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_2d_axis0_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d2(
            [[3.0, 1.0, 4.0], [2.0, 0.5, 6.0], [7.0, 5.5, 1.0]]
        )
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=0)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu == Tensor[DType.int32].d1([1, 1, 2]))
        print("✓ Passed")


fn test_argmin_2d_axis1_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_2d_axis1_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d2(
            [[3.0, 1.0, 4.0], [2.0, 0.5, 6.0], [7.0, 5.5, 1.0]]
        )
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=1)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu == Tensor[DType.int32].d1([1, 1, 2]))
        print("✓ Passed")


fn test_argmax_2d_axis0_gpu() raises:
    comptime if has_accelerator():
        print("test_argmax_2d_axis0_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d2(
            [[3.0, 1.0, 4.0], [2.0, 0.5, 6.0], [7.0, 5.5, 1.0]]
        )
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmax(axis=0)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu == Tensor[DType.int32].d1([2, 2, 1]))
        print("✓ Passed")


fn test_argmax_2d_axis1_gpu() raises:
    comptime if has_accelerator():
        print("test_argmax_2d_axis1_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d2(
            [[3.0, 1.0, 4.0], [2.0, 0.5, 6.0], [7.0, 5.5, 1.0]]
        )
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmax(axis=1)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu == Tensor[DType.int32].d1([2, 2, 0]))
        print("✓ Passed")


fn test_argmin_2d_keepdims_axis0_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_2d_keepdims_axis0_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d2([[3.0, 1.0, 4.0], [2.0, 0.5, 6.0]])
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=0, keepdims=True)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu.shape() == Shape(1, 3))
        assert_true(result_cpu == Tensor[DType.int32].d2([[1, 1, 0]]))
        print("✓ Passed")


fn test_argmax_2d_keepdims_axis1_gpu() raises:
    comptime if has_accelerator():
        print("test_argmax_2d_keepdims_axis1_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d2([[3.0, 1.0, 4.0], [2.0, 0.5, 6.0]])
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmax(axis=1, keepdims=True)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu.shape() == Shape(2, 1))
        assert_true(result_cpu == Tensor[DType.int32].d2([[2], [2]]))
        print("✓ Passed")


fn test_argmin_2d_negative_axis_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_2d_negative_axis_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d2([[3.0, 1.0, 4.0], [2.0, 0.5, 6.0]])
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=-1)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu == Tensor[DType.int32].d1([1, 1]))
        print("✓ Passed")


# ──────────────────────────────────────────────────────────────────────────────
# GPU Tests - 3D
# ──────────────────────────────────────────────────────────────────────────────


fn test_argmin_3d_axis0_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_3d_axis0_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
        )
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=0)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu == Tensor[DType.int32].d2([[1, 1], [1, 1]]))
        print("✓ Passed")


fn test_argmin_3d_axis1_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_3d_axis1_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
        )
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=1)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu == Tensor[DType.int32].d2([[0, 0], [0, 0]]))
        print("✓ Passed")


fn test_argmin_3d_axis2_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_3d_axis2_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
        )
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=2)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu == Tensor[DType.int32].d2([[0, 0], [0, 0]]))
        print("✓ Passed")


fn test_argmax_3d_axis0_gpu() raises:
    comptime if has_accelerator():
        print("test_argmax_3d_axis0_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
        )
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmax(axis=0)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu == Tensor[DType.int32].d2([[0, 0], [0, 0]]))
        print("✓ Passed")


fn test_argmax_3d_axis1_gpu() raises:
    comptime if has_accelerator():
        print("test_argmax_3d_axis1_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
        )
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmax(axis=1)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu == Tensor[DType.int32].d2([[1, 1], [1, 1]]))
        print("✓ Passed")


fn test_argmax_3d_axis2_gpu() raises:
    comptime if has_accelerator():
        print("test_argmax_3d_axis2_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
        )
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmax(axis=2)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu == Tensor[DType.int32].d2([[1, 1], [1, 1]]))
        print("✓ Passed")


fn test_argmin_3d_keepdims_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_3d_keepdims_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[0.5, 1.5], [2.5, 3.5]]]
        )
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=1, keepdims=True)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu.shape() == Shape(2, 1, 2))
        print("✓ Passed")


# ──────────────────────────────────────────────────────────────────────────────
# GPU Tests - 4D
# ──────────────────────────────────────────────────────────────────────────────


fn test_argmin_4d_axis0_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_4d_axis0_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d4(
            [
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                [[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]],
            ]
        )
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=0)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu.shape() == Shape(2, 2, 2))
