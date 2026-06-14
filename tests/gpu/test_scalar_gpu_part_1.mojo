from tenmo import NDBuffer, Shape, IntArray
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.device import GPU
from tenmo.mnemonics import (
    Add, Subtract, ReverseSubtract, Multiply, Divide, ReverseDivide, MAX, MIN, POW,
)
from tenmo.tensor import Tensor
from tenmo.shapes import Shape




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
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(5, 5))
        var expected = a.scalar_ops[Add](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_subtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(5, 5))
        var expected = a.scalar_ops[Subtract](Scalar[dtype](3.0))
        var result = a.to_gpu(gpu).scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_reversesubtract_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(5, 5))
        var expected = a.scalar_ops[ReverseSubtract](Scalar[dtype](10.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseSubtract](Scalar[dtype](10.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(5, 5))
        var expected = a.scalar_ops[Multiply](Scalar[dtype](2.0))
        var result = a.to_gpu(gpu).scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_divide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(5, 5))
        var expected = a.scalar_ops[Divide](Scalar[dtype](4.0))
        var result = a.to_gpu(gpu).scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_reversedivide_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(5, 5))
        var expected = a.scalar_ops[ReverseDivide](Scalar[dtype](20.0))
        var result = a.to_gpu(gpu).scalar_ops[ReverseDivide](Scalar[dtype](20.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_max_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(5, 5))
        var expected = a.scalar_ops[MAX](Scalar[dtype](5.0))
        var result = a.to_gpu(gpu).scalar_ops[MAX](Scalar[dtype](5.0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # 2D (5,5)
def test_oop_2d_min_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(5, 5))
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


# =============================================================================
# A. Out-of-place scalar_ops — contiguous input (flat SIMD kernel)
# =============================================================================

  # 1D, size=8
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



# ═══════════════════════════════════════════════════════════════════════════════
# GPU retain_graph tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_retain_graph_gpu_add() raises:
    comptime dtype = DType.float32
    comptime if has_accelerator():
        var a = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var mid_gpu = a_gpu + a_gpu
        var out_gpu = mid_gpu.sum()
        out_gpu.backward(retain_graph=False)
        assert_true(
            mid_gpu.grad().to_cpu().all_close(Tensor[dtype].zeros(Shape(4))),
            "GPU Add: retain_graph=False should zero intermediate grad",
        )
        assert_true(
            a.grad().all_close(Tensor[dtype].full(Shape(4), 2.0)),
            "GPU Add: leaf grad correct with retain_graph=False",
        )

        var a2 = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
        var a2_gpu = a2.to_gpu()
        var mid2_gpu = a2_gpu + a2_gpu
        var out2_gpu = mid2_gpu.sum()
        out2_gpu.backward(retain_graph=True)
        assert_true(
            mid2_gpu.grad().to_cpu().all_close(Tensor[dtype].full(Shape(4), 1.0)),
            "GPU Add: retain_graph=True should preserve intermediate grad",
        )
        assert_true(
            a2.grad().all_close(Tensor[dtype].full(Shape(4), 2.0)),
            "GPU Add: leaf grad correct with retain_graph=True",
        )


def test_retain_graph_gpu_matmul() raises:
    comptime dtype = DType.float32
    comptime if has_accelerator():
        var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
        var b = Tensor[dtype].full(Shape(3, 2), 2.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var mid_gpu = a_gpu.matmul(b_gpu)
        var out_gpu = mid_gpu.sum()
        out_gpu.backward(retain_graph=False)
        assert_true(
            mid_gpu.grad().to_cpu().all_close(Tensor[dtype].zeros(Shape(2, 2))),
            "GPU Matmul: retain_graph=False should zero intermediate grad",
        )

        var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
        var b2 = Tensor[dtype].full(Shape(3, 2), 2.0, requires_grad=True)
        var a2_gpu = a2.to_gpu()
        var b2_gpu = b2.to_gpu()
        var mid2_gpu = a2_gpu.matmul(b2_gpu)
        var out2_gpu = mid2_gpu.sum()
        out2_gpu.backward(retain_graph=True)
        assert_true(
            mid2_gpu.grad().to_cpu().all_close(Tensor[dtype].full(Shape(2, 2), 1.0)),
            "GPU Matmul: retain_graph=True should preserve intermediate grad",
        )


def test_retain_graph_gpu_sum() raises:
    comptime dtype = DType.float32
    comptime if has_accelerator():
        var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var mid_gpu = a_gpu.sum(axes=[0])
        var out_gpu = mid_gpu.sum()
        out_gpu.backward(retain_graph=False)
        assert_true(
            mid_gpu.grad().to_cpu().all_close(Tensor[dtype].zeros(Shape(3))),
            "GPU Sum: retain_graph=False should zero intermediate grad",
        )

        var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
        var a2_gpu = a2.to_gpu()
        var mid2_gpu = a2_gpu.sum(axes=[0])
        var out2_gpu = mid2_gpu.sum()
        out2_gpu.backward(retain_graph=True)
        assert_true(
            mid2_gpu.grad().to_cpu().all_close(Tensor[dtype].full(Shape(3), 1.0)),
            "GPU Sum: retain_graph=True should preserve intermediate grad",
        )


def test_retain_graph_gpu_view_zero_grad_always() raises:
    comptime dtype = DType.float32
    comptime if has_accelerator():
        var a = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var mid_gpu = a_gpu.into_view()
        var out_gpu = mid_gpu.sum()
        out_gpu.backward(retain_graph=False)
        assert_true(
            mid_gpu.grad().to_cpu().all_close(Tensor[dtype].zeros(Shape(2, 3))),
            "GPU View: retain_graph=False should zero intermediate grad (view)",
        )

        var a2 = Tensor[dtype].full(Shape(2, 3), 1.0, requires_grad=True)
        var a2_gpu = a2.to_gpu()
        var mid2_gpu = a2_gpu.into_view()
        var out2_gpu = mid2_gpu.sum()
        out2_gpu.backward(retain_graph=True)
        assert_true(
            mid2_gpu.grad().to_cpu().all_close(Tensor[dtype].zeros(Shape(2, 3))),
            "GPU View: retain_graph=True should ALSO zero intermediate grad (view)",
        )
        assert_true(
            a2.grad().all_close(Tensor[dtype].full(Shape(2, 3), 1.0)),
            "GPU View: leaf grad correct",
        )


def test_retain_graph_gpu_complex_graph() raises:
    comptime dtype = DType.float32
    comptime if has_accelerator():
        var a = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
        var b = Tensor[dtype].full(Shape(4), 3.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var c_gpu = a_gpu + b_gpu
        var d_gpu = a_gpu * b_gpu
        var mid_gpu = c_gpu + d_gpu
        var out_gpu = mid_gpu.sum()
        out_gpu.backward(retain_graph=True)

        assert_true(
            c_gpu.grad().to_cpu().all_close(Tensor[dtype].full(Shape(4), 1.0)),
            "GPU Complex: c grad preserved with retain_graph=True",
        )
        assert_true(
            d_gpu.grad().to_cpu().all_close(Tensor[dtype].full(Shape(4), 1.0)),
            "GPU Complex: d grad preserved with retain_graph=True",
        )
        assert_true(
            mid_gpu.grad().to_cpu().all_close(Tensor[dtype].full(Shape(4), 1.0)),
            "GPU Complex: mid grad preserved with retain_graph=True",
        )
        assert_true(
            a.grad().all_close(Tensor[dtype].full(Shape(4), 4.0)),
            "GPU Complex: a leaf grad correct",
        )
        assert_true(
            b.grad().all_close(Tensor[dtype].full(Shape(4), 3.0)),
            "GPU Complex: b leaf grad correct",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
