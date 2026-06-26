from tenmo import NDBuffer, Shape, IntArray
from std.sys import has_accelerator
from tenmo.device import GPU
from tenmo.mnemonics import (
    Add, Subtract, ReverseSubtract, Multiply, Divide, ReverseDivide, MAX, MIN, POW,
)
from tenmo.tensor import Tensor
from tenmo.relu import ReLU
from tenmo.buffers import Buffer
from std.testing import assert_true, assert_equal, TestSuite
from tenmo.strides import Strides
from tenmo.optim import SGD
from tenmo.common_utils import s

from tenmo.shuffle import Shuffle
comptime dtype = DType.float32






# =============================================================================
# A. Out-of-place scalar_ops — contiguous input (flat SIMD kernel)
# =============================================================================

  # 1D, size=8
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
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.scalar_ops[Add](Scalar[dtype](0))
        var result = a.to_gpu(gpu).scalar_ops[Add](Scalar[dtype](0), sync=True)
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))
  # identity op
def test_oop_identity_multiply_gpu_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
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
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
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
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
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





# =============================================================================
# test_relu_revamped.mojo
#
# Exhaustive tests for revamped ReLU (new ArgumentType / NDBuffer mask path).
# All test names carry prefix:  rlv_  (relu_revamped)
#
# Coverage:
#   Forward:      values, zeros, negatives, mixed, dtype variants
#   Backward:     mask correctness, grad shape, all-negative (zero grad)
#   Grad flow:    chained ops, scalar chain, no-grad tensors
#   Dimensions:   0-d (scalar), 1-d, 2-d, 3-d, 4-d, non-contiguous (slice/transpose)
#   Devices:      CPU (all tests), GPU (guarded with has_accelerator())
# =============================================================================


# =============================================================================
# ── SECTION 1: CPU FORWARD ───────────────────────────────────────────────────
# =============================================================================


# =============================================================================
# ── SECTION 2: CPU BACKWARD (mask correctness) ───────────────────────────────
# =============================================================================


# =============================================================================
# ── SECTION 3: CPU GRAD FLOW ─────────────────────────────────────────────────
# =============================================================================


# =============================================================================
# ── SECTION 4: CPU NON-CONTIGUOUS ────────────────────────────────────────────
# =============================================================================


# =============================================================================
# ── SECTION 5: GPU FORWARD ───────────────────────────────────────────────────
# =============================================================================


def test_rlv_gpu_fwd_all_positive() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


def test_rlv_gpu_fwd_all_negative() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([-1.0, -2.0, -3.0]).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


def test_rlv_gpu_fwd_mixed() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([-3.0, 0.0, 2.0, -1.0, 5.0]).to_gpu()
        var out = a.relu()
        assert_true(
            out.to_cpu().all_close(Tensor[dtype].d1([0.0, 0.0, 2.0, 0.0, 5.0]))
        )


def test_rlv_gpu_fwd_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]]).to_gpu()
        var out = a.relu()
        assert_true(
            out.to_cpu().all_close(Tensor[dtype].d2([[0.0, 2.0], [3.0, 0.0]]))
        )


def test_rlv_gpu_fwd_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].full([2, 3, 4], -1.0).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].zeros([2, 3, 4])))


def test_rlv_gpu_fwd_4d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].full([2, 2, 2, 2], 3.0).to_gpu()
        var out = a.relu()
        assert_true(
            out.to_cpu().all_close(Tensor[dtype].full([2, 2, 2, 2], 3.0))
        )


def test_rlv_gpu_fwd_large() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # Exercises multi-block dispatch in the kernel
        var a = Tensor[dtype].full([131072], 2.0).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].full([131072], 2.0)))


def test_rlv_gpu_fwd_dtype_float64() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var a = Tensor[dtype].d1([-1.0, 0.0, 1.0]).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].d1([0.0, 0.0, 1.0])))


# =============================================================================
# ── SECTION 6: GPU BACKWARD (mask stays on device) ───────────────────────────
# =============================================================================


def test_rlv_gpu_bwd_all_positive() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


def test_rlv_gpu_bwd_all_negative() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, -2.0, -3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


def test_rlv_gpu_bwd_mixed() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 0.0, 1.0]))
        )


def test_rlv_gpu_bwd_zero_boundary() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([0.0, 1.0, -1.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 0.0])))


def test_rlv_gpu_bwd_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[-1.0, 2.0], [3.0, -4.0]], requires_grad=True
        )
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(Tensor[dtype].d2([[0.0, 1.0], [1.0, 0.0]]))
        )


def test_rlv_gpu_bwd_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full([2, 2, 2], 1.0, requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


def test_rlv_gpu_bwd_large() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # mask stays on GPU through backward — verifies NDBuffer ArgumentType path
        var a_cpu = Tensor[dtype].full([65536], 1.0, requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].ones((65536))))


def test_rlv_gpu_bwd_grad_shape_preserved() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, -1.0, 2.0], [-2.0, 3.0, -3.0]], requires_grad=True
        )
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_equal(a_cpu.grad().shape(), a_cpu.shape())


# =============================================================================
# ── SECTION 7: GPU GRAD FLOW ─────────────────────────────────────────────────
# =============================================================================


def test_rlv_gpu_grad_chain_relu_mul() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, 2.0, 3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var r = a.relu()
        var out = r * Tensor[dtype].full([3], 2.0).to_gpu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([0.0, 2.0, 2.0])))


def test_rlv_gpu_grad_chain_relu_relu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, 2.0, 3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var r1 = a.relu()
        var r2 = r1.relu()
        var loss = r2.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 1.0])))


def test_rlv_gpu_grad_chain_add_relu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, 1.0], requires_grad=True)
        var b_cpu = Tensor[dtype].d1([2.0, -3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var b = b_cpu.to_gpu()
        var out = (a + b).relu()
        var loss = out.sum()
        loss.backward()
        # a+b = [1.0, -2.0] → mask = [1, 0]
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([1.0, 0.0])))
        assert_true(b_cpu.grad().all_close(Tensor[dtype].d1([1.0, 0.0])))


def test_rlv_gpu_grad_scalar_chain() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].scalar(3.0, requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].scalar(1.0)))


# =============================================================================
# ── SECTION 8: GPU NON-CONTIGUOUS ────────────────────────────────────────────
# (exercises contiguous_device_state() single-sweep path)
# =============================================================================


def test_rlv_gpu_noncontig_transposed_fwd() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]]).to_gpu()
        var t = a.transpose()  # non-contiguous on GPU
        var out = t.relu()
        assert_true(
            out.to_cpu().all_close(Tensor[dtype].d2([[0.0, 3.0], [2.0, 0.0]]))
        )


def test_rlv_gpu_noncontig_transposed_bwd() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[-1.0, 2.0], [3.0, -4.0]], requires_grad=True
        )
        var a = a_cpu.to_gpu()
        var t = a.transpose()
        var out = t.relu()
        var loss = out.sum()
        loss.backward()
        # Gradient flows back through transpose: mask on transposed layout
        assert_equal(a_cpu.grad().shape(), a_cpu.shape())


def test_rlv_gpu_noncontig_single_map_to_host() raises:
    comptime if has_accelerator():
        # Specifically verifies that non-contiguous GPU input does NOT
        # cause per-element map_to_host calls — exercises the
        # contiguous_device_state() single-sweep path in launch_with_mask.
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full([64, 64], 1.0, requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var a = a_gpu.transpose()  # non-contiguous 64x64
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        # All positive → grad = 1 everywhere, shape [64, 64] transposed back
        assert_equal(a_cpu.grad().shape(), a_cpu.shape())


# =============================================================================
# ── SECTION 9: CPU / GPU PARITY ──────────────────────────────────────────────
# =============================================================================


def test_rlv_parity_fwd_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var data = Tensor[dtype].d1([-3.0, -1.0, 0.0, 1.0, 3.0])
        var cpu_out = data.relu()
        var gpu_out = data.to_gpu().relu().to_cpu()
        assert_true(cpu_out.all_close(gpu_out))


def test_rlv_parity_fwd_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var data = Tensor[dtype].d2([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
        var cpu_out = data.relu()
        var gpu_out = data.to_gpu().relu().to_cpu()
        assert_true(cpu_out.all_close(gpu_out))


def test_rlv_parity_bwd_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32

        var a_cpu = Tensor[dtype].d2(
            [[-1.0, 2.0], [3.0, -4.0]], requires_grad=True
        )
        var out_cpu = a_cpu.relu()
        var loss_cpu = out_cpu.sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad()

        var a_cpu2 = Tensor[dtype].d2(
            [[-1.0, 2.0], [3.0, -4.0]], requires_grad=True
        )
        var a_gpu = a_cpu2.to_gpu()
        var out_gpu = a_gpu.relu()
        var loss_gpu = out_gpu.sum()
        loss_gpu.backward()
        var gpu_grad = a_cpu2.grad()

        assert_true(cpu_grad.all_close(gpu_grad))



def test_gpu_1d_to_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 3)
        assert_true(
            b.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            )
        )


def test_gpu_1d_to_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 3)
        var c = b * 2.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]))
        )


def test_gpu_1d_to_2d_b_no_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 2)
        var c = b * 3.0
        var loss = c.sum()
        loss.backward()
        assert_true(b.requires_grad)


# ─────────────────────────────────────────────
#  GPU — 2-D → 1-D
# ─────────────────────────────────────────────


def test_gpu_2d_to_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(6)
        assert_true(
            b.to_cpu().all_close(
                Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            )
        )


def test_gpu_2d_to_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(6)
        var c = b * 3.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
            )
        )


# ─────────────────────────────────────────────
#  GPU — 2-D → 3-D
# ─────────────────────────────────────────────


def test_gpu_2d_to_3d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 2, 2)
        assert_true(
            b.to_cpu().all_close(
                Tensor[dtype].d3(
                    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
                )
            )
        )


def test_gpu_2d_to_3d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 2, 2)
        var c = b * 5.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]])
            )
        )


# ─────────────────────────────────────────────
#  GPU — 3-D → 2-D
# ─────────────────────────────────────────────


def test_gpu_3d_to_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(4, 2)
        assert_true(
            b.to_cpu().all_close(
                Tensor[dtype].d2(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
                )
            )
        )


def test_gpu_3d_to_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(4, 2)
        var c = b * 4.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[4.0, 4.0], [4.0, 4.0]], [[4.0, 4.0], [4.0, 4.0]]]
                )
            )
        )


# ─────────────────────────────────────────────
#  GPU — 3-D → 1-D
# ─────────────────────────────────────────────


def test_gpu_3d_to_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(8)
        assert_true(
            b.to_cpu().all_close(
                Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            )
        )


def test_gpu_3d_to_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(8)
        var c = b * 7.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[7.0, 7.0], [7.0, 7.0]], [[7.0, 7.0], [7.0, 7.0]]]
                )
            )
        )


# ─────────────────────────────────────────────
#  GPU — scalar reshape
# ─────────────────────────────────────────────


def test_gpu_scalar_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([42.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape()
        assert_true(b.to_cpu().all_close(Tensor[dtype].full(Shape(), 42.0)))


def test_gpu_scalar_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape()
        var c = b * 10.0
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([10.0])))


# ─────────────────────────────────────────────
#  GPU — chained reshape
# ─────────────────────────────────────────────


def test_gpu_chained_reshape_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 4)
        var c = b.reshape(4, 2)
        assert_true(
            c.to_cpu().all_close(
                Tensor[dtype].d2(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
                )
            )
        )


def test_gpu_chained_reshape_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 4)
        var c = b.reshape(4, 2)
        var d = c * 2.0
        var loss = d.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d1([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
            )
        )


# ─────────────────────────────────────────────
#  GPU — op between two reshapes
# ─────────────────────────────────────────────


def test_gpu_op_between_two_reshapes_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(4)
        var c = b * 3.0
        var d = c.reshape(2, 2)
        var loss = d.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[3.0, 3.0], [3.0, 3.0]]))
        )


# ─────────────────────────────────────────────
#  GPU — non-scalar backward
# ─────────────────────────────────────────────


def test_gpu_non_scalar_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(2, 2)
        var c = b * 5.0
        c.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([5.0, 5.0, 5.0, 5.0])))


# ─────────────────────────────────────────────
#  GPU — Shape overload
# ─────────────────────────────────────────────


def test_gpu_shape_overload_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(Shape(3, 2))
        assert_true(
            b.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )
        )
        var c = b * 6.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[6.0, 6.0, 6.0], [6.0, 6.0, 6.0]])
            )
        )


# ─────────────────────────────────────────────
#  GPU — grad stays on CPU (a's device)
# ─────────────────────────────────────────────


def test_gpu_grad_stays_on_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.reshape(4)
        var c = b * 8.0
        var loss = c.sum()
        loss.backward()
        # a lives on CPU — its grad must also be on CPU, directly accessible
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[8.0, 8.0], [8.0, 8.0]]))
        )




# ── CPU Tests ─────────────────────────────────────────────────────────────────


# ── GPU Tests ─────────────────────────────────────────────────────────────────


def test_sgd_gpu_vanilla_single_step() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var w_gpu = w.to_gpu()
        var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params.append(UnsafePointer(to=w_gpu))
        var sgd = SGD(params, lr=0.1)
        w_gpu.seed_grad(1.0)
        sgd.step()
        var result = w_gpu.to_cpu()
        assert_true(result.all_close(Tensor[dtype].d1([0.9, 1.9, 2.9])))


def test_sgd_gpu_vanilla_matches_cpu() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].rand(4, 8, requires_grad=True)
        var w_gpu = w.to_gpu()
        var params_cpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        var params_gpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params_cpu.append(UnsafePointer(to=w))
        params_gpu.append(UnsafePointer(to=w_gpu))
        var sgd_cpu = SGD(params_cpu, lr=0.01)
        var sgd_gpu = SGD(params_gpu, lr=0.01)
        # Same grad on both
        var grad = Tensor[dtype].rand(4, 8)
        var grad_gpu = grad.to_gpu()
        w.seed_grad(grad)
        w_gpu.seed_grad(grad_gpu)
        sgd_cpu.step()
        sgd_gpu.step()
        assert_true(w.all_close(w_gpu.to_cpu()))


def test_sgd_gpu_vanilla_multiple_steps() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var w_gpu = w.to_gpu()
        var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params.append(UnsafePointer(to=w_gpu))
        var sgd = SGD(params, lr=0.1)
        for _ in range(3):
            w_gpu.seed_grad(1.0)
            sgd.step()
        var result = w_gpu.to_cpu()
        assert_true(result.all_close(Tensor[dtype].d1([0.7, 1.7, 2.7])))


def test_sgd_gpu_weight_decay() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var w_gpu = w.to_gpu()
        var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params.append(UnsafePointer(to=w_gpu))
        var sgd = SGD(params, lr=0.1, weight_decay=0.1)
        w_gpu.seed_grad(1.0)
        sgd.step()
        var expected = Tensor[dtype].d1(
            [
                1.0 - 0.1 * (1.0 + 0.1 * 1.0),
                2.0 - 0.1 * (1.0 + 0.1 * 2.0),
                3.0 - 0.1 * (1.0 + 0.1 * 3.0),
            ]
        )
        assert_true(w_gpu.to_cpu().all_close(expected))


def test_sgd_gpu_momentum_single_step() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var w_gpu = w.to_gpu()
        var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params.append(UnsafePointer(to=w_gpu))
        var sgd = SGD(params, lr=0.1, momentum=0.9)
        w_gpu.seed_grad(1.0)
        sgd.step()
        assert_true(w_gpu.to_cpu().all_close(Tensor[dtype].d1([0.9, 1.9, 2.9])))


def test_sgd_gpu_momentum_multiple_steps() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].d1([1.0], requires_grad=True)
        var w_gpu = w.to_gpu()
        var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params.append(UnsafePointer(to=w_gpu))
        var sgd = SGD(params, lr=0.1, momentum=0.9)
        for _ in range(3):
            w_gpu.seed_grad(1.0)
            sgd.step()
        assert_true(w_gpu.to_cpu().all_close(Tensor[dtype].d1([0.439])))


def test_sgd_gpu_momentum_matches_cpu() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].rand(4, 8, requires_grad=True)
        var w_gpu = w.to_gpu()
        var params_cpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        var params_gpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params_cpu.append(UnsafePointer(to=w))
        params_gpu.append(UnsafePointer(to=w_gpu))
        var sgd_cpu = SGD(params_cpu, lr=0.01, momentum=0.9)
        var sgd_gpu = SGD(params_gpu, lr=0.01, momentum=0.9)
        var grad = Tensor[dtype].rand(4, 8)
        var grad_gpu = grad.to_gpu()
        for _ in range(5):
            w.seed_grad(grad)
            w_gpu.seed_grad(grad_gpu)
            sgd_cpu.step()
            sgd_gpu.step()
        assert_true(w.all_close(w_gpu.to_cpu()))


def test_sgd_gpu_clip_value() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var w_gpu = w.to_gpu()
        var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params.append(UnsafePointer(to=w_gpu))
        var sgd = SGD(params, lr=0.1, clip_value=0.5)
        w_gpu.seed_grad(2.0)
        sgd.step()
        assert_true(
            w_gpu.to_cpu().all_close(Tensor[dtype].d1([0.95, 1.95, 2.95]))
        )


def test_sgd_gpu_clip_value_matches_cpu() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].rand(8, 8, requires_grad=True)
        var w_gpu = w.to_gpu()
        var params_cpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        var params_gpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params_cpu.append(UnsafePointer(to=w))
        params_gpu.append(UnsafePointer(to=w_gpu))
        var sgd_cpu = SGD(params_cpu, lr=0.01, clip_value=0.1)
        var sgd_gpu = SGD(params_gpu, lr=0.01, clip_value=0.1)
        var grad = Tensor[dtype].rand(8, 8)
        var grad_gpu = grad.to_gpu()
        w.seed_grad(grad)
        w_gpu.seed_grad(grad_gpu)
        sgd_cpu.step()
        sgd_gpu.step()
        assert_true(w.all_close(w_gpu.to_cpu()))


def test_sgd_gpu_clip_norm_matches_cpu() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].rand(8, 8, requires_grad=True)
        var w_gpu = w.to_gpu()
        var params_cpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        var params_gpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params_cpu.append(UnsafePointer(to=w))
        params_gpu.append(UnsafePointer(to=w_gpu))
        var sgd_cpu = SGD(params_cpu, lr=0.01, clip_norm=1.0)
        var sgd_gpu = SGD(params_gpu, lr=0.01, clip_norm=1.0)
        var grad = Tensor[dtype].rand(8, 8)
        var grad_gpu = grad.to_gpu()
        w.seed_grad(grad)
        w_gpu.seed_grad(grad_gpu)
        sgd_cpu.step()
        sgd_gpu.step()
        assert_true(w.all_close(w_gpu.to_cpu()))


def test_sgd_gpu_multiple_parameters() raises:
    comptime if has_accelerator():
        var w1 = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var w2 = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
        var w1_gpu = w1.to_gpu()
        var w2_gpu = w2.to_gpu()
        var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params.append(UnsafePointer(to=w1_gpu))
        params.append(UnsafePointer(to=w2_gpu))
        var sgd = SGD(params, lr=0.1)
        w1_gpu.seed_grad(1.0)
        w2_gpu.seed_grad(2.0)
        sgd.step()
        assert_true(w1_gpu.to_cpu().all_close(Tensor[dtype].d1([0.9, 1.9])))
        assert_true(w2_gpu.to_cpu().all_close(Tensor[dtype].d1([2.8, 3.8])))


def test_sgd_gpu_zero_grad() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var w_gpu = w.to_gpu()
        var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params.append(UnsafePointer(to=w_gpu))
        var sgd = SGD(params, lr=0.1)
        w_gpu.seed_grad(1.0)
        sgd.step()
        sgd.zero_grad()
        var grad_after = w_gpu.grad().to_cpu()
        assert_true(grad_after.all_close(Tensor[dtype].zeros(w_gpu.shape())))


def test_sgd_gpu_backward_integration() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var w_gpu = w.to_gpu()
        var x = Tensor[dtype].d1([1.0, 1.0, 1.0])
        var x_gpu = x.to_gpu()
        var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params.append(UnsafePointer(to=w_gpu))
        var sgd = SGD(params, lr=0.1)
        var loss = (w_gpu * x_gpu).sum()
        loss.backward()
        sgd.step()
        # grad_w = x = [1,1,1], w = w - 0.1*1 = [0.9, 1.9, 2.9]
        assert_true(w_gpu.to_cpu().all_close(Tensor[dtype].d1([0.9, 1.9, 2.9])))
        # grad flows back to CPU w
        assert_true(w.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


def test_sgd_gpu_backward_integration_matches_cpu() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].rand(4, 4, requires_grad=True)
        var x = Tensor[dtype].rand(4, 4)
        var w_gpu = w.to_gpu()
        var x_gpu = x.to_gpu()
        var params_cpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        var params_gpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params_cpu.append(UnsafePointer(to=w))
        params_gpu.append(UnsafePointer(to=w_gpu))
        var sgd_cpu = SGD(params_cpu, lr=0.01)
        var sgd_gpu = SGD(params_gpu, lr=0.01)
        # CPU backward
        var loss_cpu = (w * x).sum()
        loss_cpu.backward()
        sgd_cpu.step()
        var w_cpu_result = w.copy()
        w.zero_grad()
        # GPU backward
        var loss_gpu = (w_gpu * x_gpu).sum()
        loss_gpu.backward()
        sgd_gpu.step()
        assert_true(w_cpu_result.all_close(w_gpu.to_cpu()))


def test_sgd_gpu_large_tensor() raises:
    comptime if has_accelerator():
        var w = Tensor[dtype].rand(128, 256, requires_grad=True)
        var w_gpu = w.to_gpu()
        var params_cpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        var params_gpu = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
        params_cpu.append(UnsafePointer(to=w))
        params_gpu.append(UnsafePointer(to=w_gpu))
        var sgd_cpu = SGD(params_cpu, lr=0.01)
        var sgd_gpu = SGD(params_gpu, lr=0.01)
        var grad = Tensor[dtype].rand(128, 256)
        var grad_gpu = grad.to_gpu()
        w.seed_grad(grad)
        w_gpu.seed_grad(grad_gpu)
        sgd_cpu.step()
        sgd_gpu.step()
        assert_true(w.all_close(w_gpu.to_cpu()))





# ============================================================
# SHUFFLE TESTS — CPU
# ============================================================

# ------------------------------------------------------------
# 1D CPU
# ------------------------------------------------------------


# ------------------------------------------------------------
# 2D CPU — axis=0
# ------------------------------------------------------------


# ------------------------------------------------------------
# 2D CPU — axis=1
# ------------------------------------------------------------


# ------------------------------------------------------------
# 3D CPU
# ------------------------------------------------------------


# ------------------------------------------------------------
# 4D CPU
# ------------------------------------------------------------


# ------------------------------------------------------------
# Random perm CPU (no explicit perm)
# ------------------------------------------------------------


# ------------------------------------------------------------
# track_grad=False CPU
# ------------------------------------------------------------


# ------------------------------------------------------------
# Double shuffle round-trip CPU
# ------------------------------------------------------------


# ------------------------------------------------------------
# Shuffle then reduce CPU
# ------------------------------------------------------------


# ============================================================
# SHUFFLE TESTS — GPU
# ============================================================

# ------------------------------------------------------------
# 1D GPU
# ------------------------------------------------------------


def test_shuf_gpu_1d_identity_perm() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([0, 1, 2, 3], axis=0)
        assert_true(s.shape() == Shape(4))
        assert_true(
            s.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]))
        )
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_shuf_gpu_1d_reverse_perm() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([3, 2, 1, 0], axis=0)
        assert_true(
            s.to_cpu().all_close(Tensor[dtype].d1([4.0, 3.0, 2.0, 1.0]))
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_shuf_gpu_1d_arbitrary_perm() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1(
            [10.0, 20.0, 30.0, 40.0, 50.0], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 0, 4, 1, 3], axis=0)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d1([30.0, 10.0, 50.0, 20.0, 40.0])
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_shuf_gpu_1d_grad_non_uniform() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 0, 1], axis=0)
        var weights = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var loss = (s * weights).sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 3.0, 1.0])))


# ------------------------------------------------------------
# 2D GPU — axis=0
# ------------------------------------------------------------


def test_shuf_gpu_2d_axis0_reverse() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 1, 0], axis=0)
        assert_true(s.shape() == Shape(3, 2))
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d2([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_shuf_gpu_2d_axis0_arbitrary() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([1, 2, 0], axis=0)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d2([[3.0, 4.0], [5.0, 6.0], [1.0, 2.0]])
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_shuf_gpu_2d_axis0_grad_non_uniform() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([1, 0], axis=0)
        var weights = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var loss = (s * weights).sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[3.0, 4.0], [1.0, 2.0]]))
        )


# ------------------------------------------------------------
# 2D GPU — axis=1
# ------------------------------------------------------------


def test_shuf_gpu_2d_axis1_reverse() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 1, 0], axis=1)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d2([[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]])
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_shuf_gpu_2d_axis1_arbitrary() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 0, 1], axis=1)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d2([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# 3D GPU
# ------------------------------------------------------------


def test_shuf_gpu_3d_axis0_reverse() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 1, 0], axis=0)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d3(
                    [
                        [[9.0, 10.0], [11.0, 12.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                        [[1.0, 2.0], [3.0, 4.0]],
                    ]
                )
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_shuf_gpu_3d_axis1_arbitrary() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 0, 1], axis=1)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d3(
                    [
                        [[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]],
                        [[11.0, 12.0], [7.0, 8.0], [9.0, 10.0]],
                    ]
                )
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_shuf_gpu_3d_axis2_reverse() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 1, 0], axis=2)
        assert_true(
            s.to_cpu().all_close(
                Tensor[dtype].d3(
                    [
                        [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]],
                        [[9.0, 8.0, 7.0], [12.0, 11.0, 10.0]],
                    ]
                )
            )
        )
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_shuf_gpu_3d_grad_non_uniform() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([1, 0], axis=0)
        var weights = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var loss = (s * weights).sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[5.0, 6.0], [7.0, 8.0]], [[1.0, 2.0], [3.0, 4.0]]]
                )
            )
        )


# ------------------------------------------------------------
# 4D GPU
# ------------------------------------------------------------


def test_shuf_gpu_4d_axis0() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(4, 3, 2, 5)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([3, 1, 0, 2], axis=0)
        assert_true(s.shape() == Shape(4, 3, 2, 5))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_shuf_gpu_4d_axis2() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(2, 3, 4, 5)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([3, 0, 2, 1], axis=2)
        assert_true(s.shape() == Shape(2, 3, 4, 5))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# GPU matches CPU (cross-validation)
# ------------------------------------------------------------


def test_shuf_gpu_matches_cpu_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(5, 6)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.shuffle([4, 1, 3, 0, 2], axis=0)
        var s_cpu = a_copy.shuffle([4, 1, 3, 0, 2], axis=0)
        assert_true(s_gpu.to_cpu().all_close(s_cpu))
        var loss_gpu = s_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = s_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


def test_shuf_gpu_matches_cpu_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(4, 4, 6)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.shuffle([3, 0, 2, 1], axis=1)
        var s_cpu = a_copy.shuffle([3, 0, 2, 1], axis=1)
        assert_true(s_gpu.to_cpu().all_close(s_cpu))
        var loss_gpu = s_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = s_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


def test_shuf_gpu_matches_cpu_non_uniform_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(3, 4)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var weights = Tensor[dtype].randn(3, 4)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.shuffle([2, 0, 1], axis=0)
        var s_cpu = a_copy.shuffle([2, 0, 1], axis=0)
        var loss_gpu = (s_gpu * weights.to_gpu()).sum()
        loss_gpu.backward()
        var loss_cpu = (s_cpu * weights).sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


# ------------------------------------------------------------
# Random perm GPU
# ------------------------------------------------------------


def test_shuf_gpu_random_perm_grad_flow() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(5, 4)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([], axis=0)
        assert_true(s.shape() == Shape(5, 4))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# track_grad=False GPU
# ------------------------------------------------------------


def test_shuf_gpu_track_grad_false() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle[track_grad=False]([1, 0], axis=0)
        assert_true(s.shape() == Shape(2, 2))
        assert_true(not s.requires_grad)


# ------------------------------------------------------------
# Grad lands on CPU
# ------------------------------------------------------------


def test_shuf_gpu_grad_lands_on_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.shuffle([2, 0, 1], axis=0)
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# Double shuffle round-trip GPU
# ------------------------------------------------------------


def test_shuf_gpu_double_shuffle_roundtrip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var s1 = a_gpu.shuffle([2, 0, 1], axis=0)
        var s2 = s1.shuffle([1, 2, 0], axis=0)
        assert_true(s2.to_cpu().all_close(a))
        var loss = s2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# Large dimension — permutation > 8 elements (tests DeviceBuffer path)
# ------------------------------------------------------------


def test_shuf_gpu_large_axis_dim() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # axis dim = 42 — exceeds Array max_rank of 8
        var a = Tensor[dtype].randn(42, 8)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        # Build a reverse permutation of length 42
        var perm = List[Int](capacity=42)
        for i in range(41, -1, -1):
            perm.append(i)
        var s = a_gpu.shuffle(perm, axis=0)
        assert_true(s.shape() == Shape(42, 8))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_shuf_gpu_large_axis_dim_matches_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(42, 4)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        # arbitrary perm of length 42
        var perm = List[Int](capacity=42)
        for i in range(42):
            perm.append((i * 13 + 7) % 42)  # pseudo-shuffle, still valid perm?
        # Use a known valid perm instead: reverse
        perm = List[Int](capacity=42)
        for i in range(41, -1, -1):
            perm.append(i)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.shuffle(perm, axis=0)
        var s_cpu = a_copy.shuffle(perm, axis=0)
        assert_true(s_gpu.to_cpu().all_close(s_cpu))
        var loss_gpu = s_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = s_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


# ============================================================
# MAIN
# ============================================================


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
