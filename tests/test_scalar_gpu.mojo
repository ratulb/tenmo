from tenmo import NDBuffer, Shape, IntArray
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.device import GPU
from tenmo.mnemonics import (
    Add, Subtract, ReverseSubtract, Multiply, Divide, ReverseDivide, MAX, MIN, POW,
)


# =============================================================================
# A. Out-of-place scalar_ops — contiguous input (flat SIMD kernel)
# =============================================================================

  # 1D, size=8
def test_oop_1d_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_oop_1d_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[Subtract](Scalar[dtype](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_oop_1d_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_oop_1d_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_oop_1d_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[Divide](Scalar[dtype](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_oop_1d_reversedivide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[ReverseDivide](Scalar[dtype](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[dtype](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_oop_1d_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[MAX](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_oop_1d_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[MIN](Scalar[dtype](8.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.scalar_ops[Subtract](Scalar[dtype](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.scalar_ops[Divide](Scalar[dtype](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_reversedivide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.scalar_ops[ReverseDivide](Scalar[dtype](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[dtype](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.scalar_ops[MAX](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.scalar_ops[MIN](Scalar[dtype](8.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_oop_3d_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_oop_3d_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.scalar_ops[Subtract](Scalar[dtype](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_oop_3d_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_oop_3d_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_oop_3d_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.scalar_ops[Divide](Scalar[dtype](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_oop_3d_reversedivide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.scalar_ops[ReverseDivide](Scalar[dtype](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[dtype](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_oop_3d_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.scalar_ops[MAX](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_oop_3d_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.scalar_ops[MIN](Scalar[dtype](8.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_oop_4d_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_oop_4d_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.scalar_ops[Subtract](Scalar[dtype](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_oop_4d_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_oop_4d_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_oop_4d_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.scalar_ops[Divide](Scalar[dtype](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_oop_4d_reversedivide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.scalar_ops[ReverseDivide](Scalar[dtype](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[dtype](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_oop_4d_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.scalar_ops[MAX](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_oop_4d_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.scalar_ops[MIN](Scalar[dtype](8.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

# =============================================================================
# A5. Out-of-place contiguous — float64
# =============================================================================

  # 1D f64, size=8
def test_oop_1d_add_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[Add](Scalar[DType.float64](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[DType.float64](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_oop_1d_subtract_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[Subtract](Scalar[DType.float64](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[DType.float64](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_oop_1d_reversesubtract_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[ReverseSubtract](Scalar[DType.float64](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[DType.float64](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_oop_1d_multiply_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[Multiply](Scalar[DType.float64](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[DType.float64](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_oop_1d_divide_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[Divide](Scalar[DType.float64](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[DType.float64](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_oop_1d_reversedivide_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[ReverseDivide](Scalar[DType.float64](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[DType.float64](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_oop_1d_max_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[MAX](Scalar[DType.float64](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[DType.float64](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_oop_1d_min_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[MIN](Scalar[DType.float64](8.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[DType.float64](8.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

# =============================================================================
# B. Out-of-place scalar_ops — strided input (strided kernel)
# =============================================================================


def test_oop_2d_add_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_subtract_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Subtract](Scalar[dtype](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_reversesubtract_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_multiply_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_divide_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Divide](Scalar[dtype](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_reversedivide_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[ReverseDivide](Scalar[dtype](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[dtype](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_max_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[MAX](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_min_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[MIN](Scalar[dtype](8.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_add_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_subtract_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Subtract](Scalar[dtype](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_reversesubtract_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_multiply_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_divide_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Divide](Scalar[dtype](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_reversedivide_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[ReverseDivide](Scalar[dtype](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[dtype](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_max_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[MAX](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_min_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[MIN](Scalar[dtype](8.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_4d_add_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_4d_subtract_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Subtract](Scalar[dtype](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_4d_reversesubtract_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_4d_multiply_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_4d_divide_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Divide](Scalar[dtype](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_4d_reversedivide_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[ReverseDivide](Scalar[dtype](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[dtype](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_4d_max_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[MAX](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_4d_min_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[MIN](Scalar[dtype](8.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_add_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous().scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_subtract_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous().scalar_ops[Subtract](Scalar[dtype](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_reversesubtract_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous().scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_multiply_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous().scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_divide_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous().scalar_ops[Divide](Scalar[dtype](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_reversedivide_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous().scalar_ops[ReverseDivide](Scalar[dtype](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[dtype](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_max_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous().scalar_ops[MAX](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_3d_min_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous().scalar_ops[MIN](Scalar[dtype](8.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_add_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous().scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_subtract_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous().scalar_ops[Subtract](Scalar[dtype](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_reversesubtract_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous().scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_multiply_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous().scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_divide_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous().scalar_ops[Divide](Scalar[dtype](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_reversedivide_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous().scalar_ops[ReverseDivide](Scalar[dtype](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[dtype](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_max_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous().scalar_ops[MAX](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

def test_oop_2d_min_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous().scalar_ops[MIN](Scalar[dtype](8.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

# =============================================================================
# C. POW — dedicated dtype-specific kernel
# =============================================================================

  # pow Scalar[dtype](2)
def test_oop_1d_pow_2_f32_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[POW](Scalar[DType.float32](2))
        var result = a.to_gpu(gpu).scalar_ops[POW](Scalar[DType.float32](2), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))
  # pow Scalar[dtype](2)
def test_oop_1d_pow_2_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[POW](Scalar[DType.float64](2))
        var result = a.to_gpu(gpu).scalar_ops[POW](Scalar[DType.float64](2), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))
  # pow Scalar[dtype](3)
def test_oop_1d_pow_3_f32_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[POW](Scalar[DType.float32](3))
        var result = a.to_gpu(gpu).scalar_ops[POW](Scalar[DType.float32](3), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))
  # pow Scalar[dtype](3)
def test_oop_1d_pow_3_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[POW](Scalar[DType.float64](3))
        var result = a.to_gpu(gpu).scalar_ops[POW](Scalar[DType.float64](3), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))
  # pow Scalar[dtype](0.5)
def test_oop_1d_pow_0_5_f32_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[POW](Scalar[DType.float32](0.5))
        var result = a.to_gpu(gpu).scalar_ops[POW](Scalar[DType.float32](0.5), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))
  # pow Scalar[dtype](0.5)
def test_oop_1d_pow_0_5_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[POW](Scalar[DType.float64](0.5))
        var result = a.to_gpu(gpu).scalar_ops[POW](Scalar[DType.float64](0.5), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))
  # pow Scalar[dtype](0)
def test_oop_1d_pow_0_f32_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[POW](Scalar[DType.float32](0))
        var result = a.to_gpu(gpu).scalar_ops[POW](Scalar[DType.float32](0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))
  # pow Scalar[dtype](0)
def test_oop_1d_pow_0_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[POW](Scalar[DType.float64](0))
        var result = a.to_gpu(gpu).scalar_ops[POW](Scalar[DType.float64](0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))
  # pow Scalar[dtype](-1)
def test_oop_1d_pow_neg1_f32_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[POW](Scalar[DType.float32](-1))
        var result = a.to_gpu(gpu).scalar_ops[POW](Scalar[DType.float32](-1), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))
  # pow Scalar[dtype](-1)
def test_oop_1d_pow_neg1_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[POW](Scalar[DType.float64](-1))
        var result = a.to_gpu(gpu).scalar_ops[POW](Scalar[DType.float64](-1), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))

# =============================================================================
# D. In-place inplace_scalar_ops — contiguous input (flat SIMD kernel)
# =============================================================================

  # 1D, size=8
def test_ip_1d_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_ip_1d_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_ip_1d_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_ip_1d_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_ip_1d_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_ip_1d_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[MAX](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1D, size=8
def test_ip_1d_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[MIN](Scalar[dtype](8.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_ip_2d_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_ip_2d_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_ip_2d_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_ip_2d_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_ip_2d_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_ip_2d_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[MAX](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_ip_2d_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[MIN](Scalar[dtype](8.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_ip_3d_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_ip_3d_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_ip_3d_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_ip_3d_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_ip_3d_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_ip_3d_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[MAX](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 3D (3,4,5)
def test_ip_3d_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[MIN](Scalar[dtype](8.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_ip_4d_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_ip_4d_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_ip_4d_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_ip_4d_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_ip_4d_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_ip_4d_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[MAX](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 4D (3,2,4,5)
def test_ip_4d_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[MIN](Scalar[dtype](8.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# =============================================================================
# D5. In-place contiguous — float64
# =============================================================================

  # 1D f64, size=8
def test_ip_1d_add_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[DType.float64](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[DType.float64](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_ip_1d_subtract_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[Subtract](Scalar[DType.float64](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[DType.float64](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_ip_1d_reversesubtract_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[ReverseSubtract](Scalar[DType.float64](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[ReverseSubtract](Scalar[DType.float64](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_ip_1d_multiply_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[DType.float64](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[DType.float64](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_ip_1d_divide_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[Divide](Scalar[DType.float64](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[DType.float64](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_ip_1d_max_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[MAX](Scalar[DType.float64](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[DType.float64](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1D f64, size=8
def test_ip_1d_min_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[MIN](Scalar[DType.float64](8.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[DType.float64](8.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# =============================================================================
# E. In-place inplace_scalar_ops — strided input (strided kernel)
# =============================================================================


def test_ip_2d_add_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_subtract_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_reversesubtract_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_multiply_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_divide_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_max_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[MAX](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_min_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[MIN](Scalar[dtype](8.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_add_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_subtract_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_reversesubtract_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_multiply_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_divide_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_max_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[MAX](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_min_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[MIN](Scalar[dtype](8.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_4d_add_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_4d_subtract_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_4d_reversesubtract_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_4d_multiply_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_4d_divide_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_4d_max_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[MAX](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_4d_min_transposed_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 121).reshape(Shape(3, 2, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[MIN](Scalar[dtype](8.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_add_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_subtract_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_reversesubtract_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous()
        expected.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_multiply_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_divide_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_max_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous()
        expected.inplace_scalar_ops[MAX](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_3d_min_permuted_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.permute(IntArray(2, 0, 1))
        var expected = a.contiguous()
        expected.inplace_scalar_ops[MIN](Scalar[dtype](8.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_add_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_subtract_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_reversesubtract_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous()
        expected.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_multiply_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_divide_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_max_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous()
        expected.inplace_scalar_ops[MAX](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ip_2d_min_sliced_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 41).reshape(Shape(5, 8))
        var a = a_base[1:4, 2:6]
        var expected = a.contiguous()
        expected.inplace_scalar_ops[MIN](Scalar[dtype](8.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# =============================================================================
# F. Edge cases — out-of-place
# =============================================================================

  # tail size 7
def test_oop_tail7_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_oop_tail7_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.scalar_ops[Subtract](Scalar[dtype](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_oop_tail7_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_oop_tail7_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_oop_tail7_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.scalar_ops[Divide](Scalar[dtype](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_oop_tail7_reversedivide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.scalar_ops[ReverseDivide](Scalar[dtype](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[dtype](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_oop_tail7_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.scalar_ops[MAX](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_oop_tail7_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.scalar_ops[MIN](Scalar[dtype](8.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_oop_1elem_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_oop_1elem_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.scalar_ops[Subtract](Scalar[dtype](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_oop_1elem_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_oop_1elem_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_oop_1elem_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.scalar_ops[Divide](Scalar[dtype](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_oop_1elem_reversedivide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.scalar_ops[ReverseDivide](Scalar[dtype](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[dtype](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_oop_1elem_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.scalar_ops[MAX](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_oop_1elem_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.scalar_ops[MIN](Scalar[dtype](8.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 100 elements
def test_oop_size100_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 101)
        var expected = a.scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 100 elements
def test_oop_size100_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 101)
        var expected = a.scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1000 elements
def test_oop_size1000_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 1001)
        var expected = a.scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 1000 elements
def test_oop_size1000_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 1001)
        var expected = a.scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 7777 elements
def test_oop_size7777_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 7778)
        var expected = a.scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 7777 elements
def test_oop_size7777_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 7778)
        var expected = a.scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # identity op
def test_oop_identity_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.scalar_ops[Add](Scalar[dtype](0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # identity op
def test_oop_identity_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.scalar_ops[Multiply](Scalar[dtype](1.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](1.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # negative scalar
def test_oop_neg_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[Add](Scalar[dtype](-3.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](-3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # negative scalar
def test_oop_neg_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[Multiply](Scalar[dtype](-2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](-2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # negative scalar
def test_oop_neg_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[MAX](Scalar[dtype](-3.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](-3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # negative scalar
def test_oop_neg_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.scalar_ops[MIN](Scalar[dtype](-3.0))
        var result = a.to_gpu(gpu).scalar_ops[MIN](Scalar[dtype](-3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))

# =============================================================================
# F8. f64 strided — 2D transposed out-of-place
# =============================================================================


def test_oop_2d_add_transposed_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Add](Scalar[DType.float64](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[DType.float64](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-10](expected))

def test_oop_2d_multiply_transposed_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Multiply](Scalar[DType.float64](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[DType.float64](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-10](expected))

# =============================================================================
# G. Edge cases — in-place
# =============================================================================

  # tail size 7
def test_ip_tail7_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_ip_tail7_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.copy()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_ip_tail7_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.copy()
        expected.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_ip_tail7_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_ip_tail7_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.copy()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_ip_tail7_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.copy()
        expected.inplace_scalar_ops[MAX](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # tail size 7
def test_ip_tail7_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8)
        var expected = a.copy()
        expected.inplace_scalar_ops[MIN](Scalar[dtype](8.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_ip_1elem_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_ip_1elem_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.copy()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_ip_1elem_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.copy()
        expected.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_ip_1elem_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_ip_1elem_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.copy()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_ip_1elem_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.copy()
        expected.inplace_scalar_ops[MAX](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # single element
def test_ip_1elem_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 2)
        var expected = a.copy()
        expected.inplace_scalar_ops[MIN](Scalar[dtype](8.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[dtype](8.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 100 elements
def test_ip_size100_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 101)
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 100 elements
def test_ip_size100_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 101)
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1000 elements
def test_ip_size1000_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 1001)
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 1000 elements
def test_ip_size1000_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 1001)
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 7777 elements
def test_ip_size7777_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 7778)
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # 7777 elements
def test_ip_size7777_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 7778)
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # identity op
def test_ip_identity_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # identity op
def test_ip_identity_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 26).reshape(Shape(5, 5))
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](1.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](1.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # negative scalar
def test_ip_neg_add_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](-3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](-3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # negative scalar
def test_ip_neg_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](-2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](-2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # negative scalar
def test_ip_neg_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[MAX](Scalar[dtype](-3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MAX](Scalar[dtype](-3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))
  # negative scalar
def test_ip_neg_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[MIN](Scalar[dtype](-3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[MIN](Scalar[dtype](-3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# =============================================================================
# G8. f64 strided — 2D transposed in-place
# =============================================================================


def test_ip_2d_add_transposed_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Add](Scalar[DType.float64](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[DType.float64](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-10](expected))

def test_ip_2d_multiply_transposed_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Multiply](Scalar[DType.float64](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[DType.float64](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-10](expected))

# =============================================================================
# H. f64 strided 3D transposed
# =============================================================================


def test_oop_3d_add_transposed_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Add](Scalar[DType.float64](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[DType.float64](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-10](expected))

def test_ip_3d_add_transposed_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Add](Scalar[DType.float64](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[DType.float64](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-10](expected))

def test_oop_3d_multiply_transposed_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous().scalar_ops[Multiply](Scalar[DType.float64](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[DType.float64](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-10](expected))

def test_ip_3d_multiply_transposed_f64_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 61).reshape(Shape(3, 4, 5))
        var a = a_base.transpose()
        var expected = a.contiguous()
        expected.inplace_scalar_ops[Multiply](Scalar[DType.float64](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[DType.float64](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-10](expected))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
