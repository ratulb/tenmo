from tenmo import NDBuffer, Shape, Buffer, Strides, IntArray
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.common_utils import Epsilon
from tenmo.device import GPU
from tenmo.mnemonics import (
    Multiply,
    Add,
    Subtract,
    Divide,
)

# =============================================================================
# Test suite: NDBuffer in-place arithmetic ops on GPU
#
# In-place ops (A op= B) mutate A directly instead of creating a new buffer.
#
# Coverage map:
#
#   BINARY IN-PLACE (A.inplace_ops[Op](B)):
#
#     PATH 1 — Both contiguous, same shape, no broadcasting
#              → test_path1_*
#     PATH 2 — A contiguous, B strided or broadcast-expanded
#              → test_path2_*
#     PATH 3 — A contiguous, B non-contiguous (transposed), no broadcast
#              → test_path3_*
#     PATH 4 — A non-contiguous (transposed), B contiguous, no broadcast
#              → test_path4_*
#     PATH 5 — Both non-contiguous (transposed or custom strides)
#              → test_path5_*
#     EDGE  — tail sizes, subchunk, large, single-element, non-pow2,
#             rank mismatch, zeros/ones broadcast
#              → test_edge_*
#
#   SCALAR IN-PLACE (A.inplace_scalar_ops[Op](scalar)):
#
#     CONTIGUOUS — 1D through 4D
#              → test_scalar_*
#     NON-CONTIGUOUS — transposed views
#              → test_scalar_*
#     EDGE — tail, single, large, identity ops
#              → test_edge_scalar_*
#
# Suffix _gpu_inplace on binary tests, _gpu_scalar_ip on scalar tests.
# All tests guarded with `comptime if has_accelerator()`.
# Ops: Add, Subtract, Multiply, Divide.
# Ranks: 1-D through 4-D where applicable.
# Division operands always > 0 to avoid divide-by-zero.
# Tolerance: atol=1e-5 (float32).
#
# Each test pattern:
#   1. Create CPU tensors a, b
#   2. Compute CPU expected: expected = a.copy(); expected.inplace_ops[Op](b.copy())
#   3. Send to GPU, do in-place op with sync=True
#   4. Read back to CPU, compare with all_close
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# BINARY IN-PLACE: PATH 1 — Both contiguous, same shape
# A and B are same shape, flat in memory. Pure linear indexing in the kernel.
# ─────────────────────────────────────────────────────────────────────────────

def test_path1_1d_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var b = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_1d_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(10, 18)
        var b = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_1d_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var b = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_1d_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(8, 16)
        var b = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_2d_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_2d_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(20, 44).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_2d_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_2d_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(24, 48).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_3d_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_3d_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(50, 98).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_3d_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_3d_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(49, 97).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_4d_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_4d_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(100, 196).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_4d_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path1_4d_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(97, 193).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))


# ─────────────────────────────────────────────────────────────────────────────
# BINARY IN-PLACE: PATH 2 — A contiguous, B strided or broadcast-expanded
# A is the receiver (always full broadcast_shape), B broadcasts to match A.
# A is linearly indexed; B uses stride decomposition (possibly stride-0 on
# broadcast dims). This is the most common real-world pattern (bias add, etc.).
# ─────────────────────────────────────────────────────────────────────────────

# 2D row broadcast: (M,N) op= (N,)

def test_path2_2d_row_broadcast_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 7)
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_2d_row_broadcast_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(10, 34).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 7)
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_2d_row_broadcast_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 7)
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_2d_row_broadcast_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(24, 48).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 7)
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# 3D bias add: (B,T,C) op= (C,)

def test_path2_3d_bias_add_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 7)
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_3d_bias_add_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(50, 98).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 7)
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_3d_bias_add_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 7)
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_3d_bias_add_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(49, 97).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 7)
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# 3D: (B,T,C) op= (T,C) — 2D broadcast into 3D

def test_path2_3d_2d_into_3d_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_3d_2d_into_3d_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(50, 98).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_3d_2d_into_3d_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_3d_2d_into_3d_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(49, 97).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# 4D channel broadcast: (B,C,H,W) op= (1,C,1,1)

def test_path2_4d_channel_broadcast_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 73).reshape(Shape(2, 4, 3, 3))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4, 1, 1))
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_4d_channel_broadcast_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(73, 145).reshape(Shape(2, 4, 3, 3))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4, 1, 1))
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_4d_channel_broadcast_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 73).reshape(Shape(2, 4, 3, 3))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4, 1, 1))
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_4d_channel_broadcast_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(73, 145).reshape(Shape(2, 4, 3, 3))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4, 1, 1))
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# 2D outer product: (N,M) op= (1,M)

def test_path2_2d_outer_product_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 21).reshape(Shape(5, 4))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4))
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_2d_outer_product_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(10, 30).reshape(Shape(5, 4))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4))
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_2d_outer_product_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 21).reshape(Shape(5, 4))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4))
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_2d_outer_product_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(10, 30).reshape(Shape(5, 4))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4))
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# 3D scalar broadcast: (B,T,C) op= (1,)

def test_path2_3d_scalar_broadcast_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(2, 3, 4))
        var b = NDBuffer[dtype].full(Shape(1), 2.0)
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_3d_scalar_broadcast_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(2, 3, 4))
        var b = NDBuffer[dtype].full(Shape(1), 3.0)
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_3d_scalar_broadcast_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(2, 3, 4))
        var b = NDBuffer[dtype].full(Shape(1), 2.0)
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_3d_scalar_broadcast_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(2, 3, 4))
        var b = NDBuffer[dtype].full(Shape(1), 4.0)
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# 4D: (B,T,H,C) op= (H,C) — 2D broadcast into 4D

def test_path2_4d_2d_into_4d_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 17).reshape(Shape(4, 4))
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_4d_2d_into_4d_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(97, 193).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 17).reshape(Shape(4, 4))
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_4d_2d_into_4d_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 17).reshape(Shape(4, 4))
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path2_4d_2d_into_4d_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(97, 193).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 17).reshape(Shape(4, 4))
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))


# ─────────────────────────────────────────────────────────────────────────────
# BINARY IN-PLACE: PATH 3 — A contiguous, B non-contiguous (transposed)
# A is linearly indexed. B uses stride decomposition.
# No broadcasting — same shape, but B is a transposed view.
# ─────────────────────────────────────────────────────────────────────────────

def test_path3_2d_a_contiguous_b_transposed_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var b = b_base.transpose()
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path3_2d_a_contiguous_b_transposed_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(20, 44).reshape(Shape(4, 6))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var b = b_base.transpose()
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path3_2d_a_contiguous_b_transposed_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var b = b_base.transpose()
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path3_2d_a_contiguous_b_transposed_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(24, 48).reshape(Shape(4, 6))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var b = b_base.transpose()
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path3_3d_a_contiguous_b_transposed_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var b = b_base.transpose(IntArray(-1, -2))
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path3_3d_a_contiguous_b_transposed_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var b = b_base.transpose(IntArray(-1, -2))
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))


# ─────────────────────────────────────────────────────────────────────────────
# BINARY IN-PLACE: PATH 4 — A non-contiguous (transposed), B contiguous
# A uses stride decomposition. B is linearly indexed.
# No broadcasting — same shape.
# ─────────────────────────────────────────────────────────────────────────────

def test_path4_2d_a_transposed_b_contiguous_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path4_2d_a_transposed_b_contiguous_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(20, 44).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path4_2d_a_transposed_b_contiguous_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path4_2d_a_transposed_b_contiguous_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(24, 48).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path4_3d_a_transposed_b_contiguous_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var a = a_base.transpose(IntArray(-1, -2))
        var b = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path4_3d_a_transposed_b_contiguous_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var a = a_base.transpose(IntArray(-1, -2))
        var b = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))


# ─────────────────────────────────────────────────────────────────────────────
# BINARY IN-PLACE: PATH 5 — Both non-contiguous
# Both A and B are strided (transposed views or custom strides).
# Universal fallback: both operands stride-decomposed.
# ─────────────────────────────────────────────────────────────────────────────

def test_path5_2d_both_transposed_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path5_2d_both_transposed_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(20, 44).reshape(Shape(6, 4))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path5_2d_both_transposed_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path5_2d_both_transposed_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(24, 48).reshape(Shape(6, 4))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path5_3d_both_transposed_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var b_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path5_3d_both_transposed_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var b_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path5_4d_both_transposed_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var b_base = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path5_4d_both_transposed_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(97, 193).reshape(Shape(2, 3, 4, 4))
        var b_base = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# PATH 5 with share() — custom non-default strides

def test_path5_2d_custom_strides_both_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var buf_a = Buffer[dtype].arange(1, 25)
        var buf_b = Buffer[dtype].arange(1, 25)
        var a_o = NDBuffer[dtype](buf_a, Shape(3, 3))
        var a = a_o.share(Shape(3, 3), Strides(1, 3), offset=0)
        var b_o = NDBuffer[dtype](buf_b, Shape(3, 3))
        var b = b_o.share(Shape(3, 3), Strides(1, 3), offset=0)
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_path5_2d_custom_strides_both_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var buf_a = Buffer[dtype].arange(1, 25)
        var buf_b = Buffer[dtype].arange(1, 25)
        var a_o = NDBuffer[dtype](buf_a, Shape(3, 3))
        var a = a_o.share(Shape(3, 3), Strides(1, 3), offset=0)
        var b_o = NDBuffer[dtype](buf_b, Shape(3, 3))
        var b = b_o.share(Shape(3, 3), Strides(1, 3), offset=0)
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))


# ─────────────────────────────────────────────────────────────────────────────
# BINARY IN-PLACE: EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

# Tail sizes that are not multiples of simd_width (8 for fp32) and smaller
# than CHUNK_SIZE (128). Exercise the scalar tail loop.

def test_edge_tail_size7_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype](1, 2, 3, 4, 5, 6, 7)
        var b = NDBuffer[dtype](7, 6, 5, 4, 3, 2, 1)
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_tail_size7_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype](1, 2, 3, 4, 5, 6, 7)
        var b = NDBuffer[dtype](1, 2, 3, 4, 5, 6, 7)
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_tail_size13_subtract_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(10, 23)
        var b = NDBuffer[dtype].arange(1, 14)
        var expected = a.copy()
        expected.inplace_ops[Subtract](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Subtract](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_tail_size17_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(17, 34)
        var b = NDBuffer[dtype].arange(1, 18)
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# Sub-chunk: 33 elements — less than CHUNK_SIZE=128, single thread block

def test_edge_subchunk_size33_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 34)
        var b = NDBuffer[dtype].arange(1, 34)
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_subchunk_size33_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 34)
        var b = NDBuffer[dtype].arange(1, 34)
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# Large tensor: 1000 elements — crosses multiple CHUNK boundaries

def test_edge_large_size1000_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 1001)
        var b = NDBuffer[dtype].arange(1, 1001)
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_large_size1000_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1000, 2000)
        var b = NDBuffer[dtype].arange(1, 1001)
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# Single element

def test_edge_single_element_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].full(Shape(1), 5.0)
        var b = NDBuffer[dtype].full(Shape(1), 3.0)
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_single_element_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].full(Shape(1), 10.0)
        var b = NDBuffer[dtype].full(Shape(1), 4.0)
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# Large 4D tensor: (4, 8, 16, 16) = 8192 elements — many CHUNK boundaries

def test_edge_large_4d_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8193).reshape(Shape(4, 8, 16, 16))
        var b = NDBuffer[dtype].arange(1, 8193).reshape(Shape(4, 8, 16, 16))
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_large_4d_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8193).reshape(Shape(4, 8, 16, 16))
        var b = NDBuffer[dtype].arange(1, 8193).reshape(Shape(4, 8, 16, 16))
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# Non-power-of-2 inner dim broadcast — forces odd coordinate decomposition

def test_edge_broadcast_nonpow2_inner_dim_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 106).reshape(Shape(3, 5, 7))
        var b = NDBuffer[dtype].arange(1, 8)
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_broadcast_nonpow2_inner_dim_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 106).reshape(Shape(3, 5, 7))
        var b = NDBuffer[dtype].arange(1, 8)
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# Rank mismatch: 4D op= 2D — (2,3,4,5) op= (4,5)

def test_edge_broadcast_rank_mismatch_4d_2d_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(2, 3, 4, 5))
        var b = NDBuffer[dtype].arange(1, 21).reshape(Shape(4, 5))
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_broadcast_rank_mismatch_4d_2d_divide_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(121, 241).reshape(Shape(2, 3, 4, 5))
        var b = NDBuffer[dtype].arange(1, 21).reshape(Shape(4, 5))
        var expected = a.copy()
        expected.inplace_ops[Divide](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Divide](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# All-ones broadcast multiply — identity op

def test_edge_broadcast_ones_multiply_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].full(Shape(1), 1.0)
        var expected = a.copy()
        expected.inplace_ops[Multiply](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Multiply](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# All-zeros broadcast add — identity op

def test_edge_broadcast_zeros_add_gpu_inplace() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].full(Shape(6), 0.0)
        var expected = a.copy()
        expected.inplace_ops[Add](b.copy())
        var a_gpu = a.to_gpu(gpu)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu, sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))


# =============================================================================
# SCALAR IN-PLACE OPS (A.inplace_scalar_ops[Op](scalar))
# =============================================================================

# ── Contiguous 1D ──

def test_scalar_1d_add_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_scalar_1d_subtract_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(10, 18)
        var expected = a.copy()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_scalar_1d_multiply_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_scalar_1d_divide_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(8, 16)
        var expected = a.copy()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](4.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](4.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# ── Contiguous 2D ──

def test_scalar_2d_add_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](7.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](7.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_scalar_2d_subtract_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(20, 44).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_scalar_ops[Subtract](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Subtract](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_scalar_2d_multiply_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_scalar_2d_divide_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(24, 48).reshape(Shape(4, 6))
        var expected = a.copy()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# ── Contiguous 3D ──

def test_scalar_3d_add_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_scalar_3d_multiply_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](1.5))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](1.5), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# ── Contiguous 4D ──

def test_scalar_4d_add_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_scalar_4d_multiply_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# ── Non-contiguous (transposed) scalar ──

def test_scalar_2d_transposed_add_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_scalar_2d_transposed_multiply_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_scalar_3d_transposed_add_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var a = a_base.transpose()
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](5.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](5.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_scalar_3d_transposed_multiply_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var a = a_base.transpose()
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# ── Scalar edge cases ──

def test_edge_scalar_tail_size7_add_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype](1, 2, 3, 4, 5, 6, 7)
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](10.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](10.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_scalar_tail_size7_multiply_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype](1, 2, 3, 4, 5, 6, 7)
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_scalar_single_element_add_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].full(Shape(1), 5.0)
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](3.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](3.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_scalar_single_element_divide_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].full(Shape(1), 10.0)
        var expected = a.copy()
        expected.inplace_scalar_ops[Divide](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Divide](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_scalar_large_size1000_add_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 1001)
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](1.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](1.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_edge_scalar_large_size1000_multiply_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 1001)
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](2.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](2.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# Identity: add 0

def test_edge_scalar_add_zero_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(2, 3, 4))
        var expected = a.copy()
        expected.inplace_scalar_ops[Add](Scalar[dtype](0.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Add](Scalar[dtype](0.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

# Identity: multiply by 1

def test_edge_scalar_multiply_one_gpu_scalar_ip() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(2, 3, 4))
        var expected = a.copy()
        expected.inplace_scalar_ops[Multiply](Scalar[dtype](1.0))
        var a_gpu = a.to_gpu(gpu)
        a_gpu.inplace_scalar_ops[Multiply](Scalar[dtype](1.0), sync=True)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](expected))

def test_ndb_inter_gpu_copy_and_opeartion() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype](1, 2, 3, 4)
        var a_gpu = a.to_gpu(gpu)
        var b = NDBuffer[dtype](1, 2, 3, 4)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu)
        a.inplace_ops[Add](b)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](a))
        if gpu[].number_of_devices() > 1:
            var gpu_other = GPU(1)
            var a_in_other_gpu = a.to_gpu(gpu_other)
            var a_other = a_in_other_gpu.scalar_ops[Multiply](42)
            assert_true((a_gpu.to_device(gpu_other.into())[1].scalar_ops[Multiply](42)) == a_other, "Inter gpu ndn operation failed")
            print("Inter gpu ndb op passed")

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
