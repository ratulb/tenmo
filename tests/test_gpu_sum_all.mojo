from std.testing import assert_true, TestSuite
from tenmo.tensor import Tensor
from std.sys import has_accelerator
from tenmo.shared.shapes import Shape

# GPU coverage for Tensor.sum_all: forward only — sum_all returns a host
# Scalar and is not differentiable. The differentiating full reduction is
# Tensor.sum(), covered in tests/test_gpu_sum_mean.mojo
# (test_gpu_sum_full_reduction_scalar_grad).


def test_gpu_sum_all_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_all_matches_cpu")
        var a = Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0]])
        assert_true(a.to_gpu().sum_all() == 10.0)


def test_gpu_sum_all_dtypes() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_all_dtypes")
        var f64 = Tensor[DType.float64].d2([[1.5, 2.5], [3.0, 4.0]])
        assert_true(f64.to_gpu().sum_all() == 11.0)
        var i32 = Tensor[DType.int32].d2([[1, 2], [3, 4]])
        assert_true(i32.to_gpu().sum_all() == 10)


def test_gpu_sum_all_noncontiguous() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_all_noncontiguous")
        var a = Tensor[DType.float32].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var t = a.transpose(0, 1)
        assert_true(t.to_gpu().sum_all() == 21.0)


def test_gpu_sum_all_rank_variants() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_all_rank_variants")
        var v1 = Tensor[DType.float32].d1([1.0, 2.0, 3.0])
        assert_true(v1.to_gpu().sum_all() == 6.0)
        var v4 = Tensor[DType.float32].rand(2, 3, 4, 5)
        var cpu_sum = v4.sum_all()
        assert_true(v4.to_gpu().sum_all() == cpu_sum)


def test_gpu_sum_all_large_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_gpu_sum_all_large_matches_cpu")
        var a = Tensor[DType.float32].randn(64, 128)
        var expected = a.sum_all()
        var got = a.to_gpu().sum_all()
        var tol = 1e-3 * abs(expected) + 1e-3
        assert_true(abs(got - expected) <= tol)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
