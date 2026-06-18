from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.shapes import Shape


def test_alltrue_gpu_1d_all_true() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d1([True, True, True, True]).to_gpu()
        assert_true(a.all_true())


def test_alltrue_gpu_1d_all_false() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d1([False, False, False]).to_gpu()
        assert_true(not a.all_true())


def test_alltrue_gpu_1d_mixed() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d1([True, False, True]).to_gpu()
        assert_true(not a.all_true())


def test_alltrue_gpu_1d_single_true() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d1([True]).to_gpu()
        assert_true(a.all_true())


def test_alltrue_gpu_1d_single_false() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d1([False]).to_gpu()
        assert_true(not a.all_true())


def test_alltrue_gpu_1d_last_false() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d1([True, True, True, False]).to_gpu()
        assert_true(not a.all_true())


def test_alltrue_gpu_2d_all_true() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d2([[True, True], [True, True]]).to_gpu()
        assert_true(a.all_true())


def test_alltrue_gpu_2d_mixed() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d2([[True, True], [True, False]]).to_gpu()
        assert_true(not a.all_true())


def test_alltrue_gpu_3d_all_true() raises:
    comptime if has_accelerator():
        var a = (
            Tensor[DType.bool]
            .d3([[[True, True], [True, True]], [[True, True], [True, True]]])
            .to_gpu()
        )
        assert_true(a.all_true())


def test_alltrue_gpu_3d_one_false() raises:
    comptime if has_accelerator():
        var a = (
            Tensor[DType.bool]
            .d3([[[True, True], [True, True]], [[True, True], [True, False]]])
            .to_gpu()
        )
        assert_true(not a.all_true())


def test_alltrue_gpu_from_comparison() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([5.0, 6.0, 7.0]).to_gpu()
        var mask = a > Tensor[dtype].full(Shape(3), 4.0).to_gpu()
        assert_true(mask.all_true())


def test_alltrue_gpu_from_comparison_false() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([3.0, 6.0, 7.0]).to_gpu()
        var mask = a > Tensor[dtype].full(Shape(3), 4.0).to_gpu()
        assert_true(not mask.all_true())


def test_alltrue_gpu_large_all_true() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].full(Shape(10000), True).to_gpu()
        assert_true(a.all_true())


def test_alltrue_gpu_large_one_false() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].full(Shape(10000), True)
        a[[5000]] = False
        var a_gpu = a.to_gpu()
        assert_true(not a_gpu.all_true())


# ═════════════════════════════════════════════════════════════════════════════
# GPU any_true Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_anytrue_gpu_1d_all_true() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d1([True, True, True]).to_gpu()
        assert_true(a.any_true())


def test_anytrue_gpu_1d_all_false() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d1([False, False, False]).to_gpu()
        assert_true(not a.any_true())


def test_anytrue_gpu_1d_mixed() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d1([False, True, False]).to_gpu()
        assert_true(a.any_true())


def test_anytrue_gpu_1d_single_true() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d1([True]).to_gpu()
        assert_true(a.any_true())


def test_anytrue_gpu_1d_single_false() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d1([False]).to_gpu()
        assert_true(not a.any_true())


def test_anytrue_gpu_1d_only_last_true() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d1([False, False, False, True]).to_gpu()
        assert_true(a.any_true())


def test_anytrue_gpu_2d_all_false() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d2([[False, False], [False, False]]).to_gpu()
        assert_true(not a.any_true())


def test_anytrue_gpu_2d_one_true() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].d2([[False, False], [False, True]]).to_gpu()
        assert_true(a.any_true())


def test_anytrue_gpu_3d_all_false() raises:
    comptime if has_accelerator():
        var a = (
            Tensor[DType.bool]
            .d3(
                [
                    [[False, False], [False, False]],
                    [[False, False], [False, False]],
                ]
            )
            .to_gpu()
        )
        assert_true(not a.any_true())


def test_anytrue_gpu_3d_one_true() raises:
    comptime if has_accelerator():
        var a = (
            Tensor[DType.bool]
            .d3(
                [
                    [[False, False], [False, False]],
                    [[False, False], [False, True]],
                ]
            )
            .to_gpu()
        )
        assert_true(a.any_true())


def test_anytrue_gpu_from_comparison() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 5.0]).to_gpu()
        var mask = a > Tensor[dtype].full(Shape(3), 4.0).to_gpu()
        assert_true(mask.any_true())


def test_anytrue_gpu_from_comparison_none() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var mask = a > Tensor[dtype].full(Shape(3), 4.0).to_gpu()
        assert_true(not mask.any_true())


def test_anytrue_gpu_large_all_false() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].full(Shape(10000), False).to_gpu()
        assert_true(not a.any_true())


def test_anytrue_gpu_large_one_true() raises:
    comptime if has_accelerator():
        var a = Tensor[DType.bool].full(Shape(10000), False)
        a[[5000]] = True
        var a_gpu = a.to_gpu()
        assert_true(a_gpu.any_true())


# ═════════════════════════════════════════════════════════════════════════════
# CPU/GPU Parity Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_alltrue_parity_all_true() raises:
    comptime if has_accelerator():
        var a_cpu = Tensor[DType.bool].d2([[True, True], [True, True]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.all_true() == a_gpu.all_true())


def test_alltrue_parity_mixed() raises:
    comptime if has_accelerator():
        var a_cpu = Tensor[DType.bool].d2([[True, False], [True, True]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.all_true() == a_gpu.all_true())


def test_anytrue_parity_all_false() raises:
    comptime if has_accelerator():
        var a_cpu = Tensor[DType.bool].d2([[False, False], [False, False]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.any_true() == a_gpu.any_true())


def test_anytrue_parity_one_true() raises:
    comptime if has_accelerator():
        var a_cpu = Tensor[DType.bool].d2([[False, False], [False, True]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.any_true() == a_gpu.any_true())


def test_alltrue_parity_from_comparison() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([5.0, 6.0, 7.0])
        var a_gpu = a_cpu.to_gpu()
        var threshold_cpu = Tensor[dtype].full(Shape(3), 4.0)
        var threshold_gpu = threshold_cpu.to_gpu()
        assert_true(
            (a_cpu > threshold_cpu).all_true()
            == (a_gpu > threshold_gpu).all_true()
        )


def test_anytrue_parity_from_comparison() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 5.0])
        var a_gpu = a_cpu.to_gpu()
        var threshold_cpu = Tensor[dtype].full(Shape(3), 4.0)
        var threshold_gpu = threshold_cpu.to_gpu()
        assert_true(
            (a_cpu > threshold_cpu).any_true()
            == (a_gpu > threshold_gpu).any_true()
        )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll all true any true tests passed!")
