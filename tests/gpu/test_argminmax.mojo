from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.argminmax import Argmin, Argmax
from std.sys import has_accelerator


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll argminmax tests passed!")


from std.testing import assert_true, TestSuite


def test_argmin_1d_basic_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_1d_basic_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d1([5.0, 2.0, 8.0, 1.0, 9.0])
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=0)
        var _tmp0 = result.to_cpu()
        var result_cpu = _tmp0.reshape(Shape(1))
        assert_true(result_cpu == Tensor[DType.int32].d1([3]))
        print("✓ Passed")


def test_argmax_1d_basic_gpu() raises:
    comptime if has_accelerator():
        print("test_argmax_1d_basic_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d1([5.0, 2.0, 8.0, 1.0, 9.0])
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmax(axis=0)
        var _tmp0 = result.to_cpu()
        var result_cpu = _tmp0.reshape(Shape(1))
        assert_true(result_cpu == Tensor[DType.int32].d1([4]))
        print("✓ Passed")


def test_argmin_1d_negative_values_gpu() raises:
    comptime if has_accelerator():
        print("test_argmin_1d_negative_values_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d1([-5.0, -2.0, -8.0, -1.0, -9.0])
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmin(axis=0)
        var _tmp0 = result.to_cpu()
        var result_cpu = _tmp0.reshape(Shape(1))
        assert_true(result_cpu == Tensor[DType.int32].d1([4]))
        print("✓ Passed")


def test_argmax_1d_negative_values_gpu() raises:
    comptime if has_accelerator():
        print("test_argmax_1d_negative_values_gpu")
        comptime dtype = DType.float32
        var t = Tensor[dtype].d1([-5.0, -2.0, -8.0, -1.0, -9.0])
        var t_gpu = t.to_gpu()
        var result = t_gpu.argmax(axis=0)
        var _tmp0 = result.to_cpu()
        var result_cpu = _tmp0.reshape(Shape(1))
        assert_true(result_cpu == Tensor[DType.int32].d1([3]))
        print("✓ Passed")


# ──────────────────────────────────────────────────────────────────────────────
# GPU Tests - 2D
# ──────────────────────────────────────────────────────────────────────────────


def test_argmin_2d_axis0_gpu() raises:
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


def test_argmin_2d_axis1_gpu() raises:
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


def test_argmax_2d_axis0_gpu() raises:
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


def test_argmax_2d_axis1_gpu() raises:
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


def test_argmin_2d_keepdims_axis0_gpu() raises:
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


def test_argmax_2d_keepdims_axis1_gpu() raises:
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


def test_argmin_2d_negative_axis_gpu() raises:
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


def test_argmin_3d_axis0_gpu() raises:
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


def test_argmin_3d_axis1_gpu() raises:
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


def test_argmin_3d_axis2_gpu() raises:
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


def test_argmax_3d_axis0_gpu() raises:
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


def test_argmax_3d_axis1_gpu() raises:
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


def test_argmax_3d_axis2_gpu() raises:
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


def test_argmin_3d_keepdims_gpu() raises:
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


def test_argmin_4d_axis0_gpu() raises:
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
