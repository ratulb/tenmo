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
# Test suite: NDBuffer out-of-place arithmetic ops on GPU
#
# Coverage map (C = A op B, result is a new buffer):
#
#   PATH 1 — both contiguous, same shape, no broadcasting
#             → test_path1_*
#   PATH 2 — both contiguous, shapes differ (broadcast expansion)
#             → test_path2_*
#   PATH 3 — A contiguous and fills broadcast_shape, B non-contiguous
#             → test_path3_*
#   PATH 4 — B contiguous and fills broadcast_shape, A non-contiguous
#             → test_path4_*
#   PATH 5 — both non-contiguous (general strided fallback)
#             → test_path5_*
#   EDGE    — tail-loop sizes, single-element, large tensors
#             → test_edge_*
#
# Suffix _gpu_arith on every name to avoid collision with other test files.
# All tests are guarded with `comptime if has_accelerator()`.
# Ops: Add, Subtract, Multiply, Divide.
# Ranks: 1-D through 4-D where applicable.
# Division operands are always > 0 to avoid divide-by-zero.
# Tolerance: atol=1e-5 (float32).
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# PATH 1: Both contiguous, same shape — pure linear indexing, no broadcast.
# Exercises the fast SIMD path end-to-end across all four ops and ranks.
# ─────────────────────────────────────────────────────────────────────────────

def test_path1_1d_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)          # [1..8]   shape (8,)
        var b = NDBuffer[dtype].arange(1, 9)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path1_1d_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(10, 18)        # [10..17] shape (8,)
        var b = NDBuffer[dtype].arange(1, 9)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path1_1d_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 9)
        var b = NDBuffer[dtype].arange(1, 9)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path1_1d_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(8, 16)         # [8..15]
        var b = NDBuffer[dtype].arange(1, 9)          # [1..8]  all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

def test_path1_2d_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))   # (4,6)
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path1_2d_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(20, 44).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path1_2d_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path1_2d_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(24, 48).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))   # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

def test_path1_3d_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        # shape (2, 4, 6) = 48 elements
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path1_3d_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(50, 98).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path1_3d_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path1_3d_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(49, 97).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))  # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

def test_path1_4d_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        # shape (2, 3, 4, 4) = 96 elements
        var a = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path1_4d_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(100, 196).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path1_4d_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path1_4d_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(97, 193).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))   # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))


# ─────────────────────────────────────────────────────────────────────────────
# PATH 2: Both contiguous, shapes differ — broadcast expansion required.
# These are the most common real-world patterns: bias add, channel scale,
# outer product, scalar broadcast.
# Both tensors are flat in memory; the kernel uses Strides.default().
# ─────────────────────────────────────────────────────────────────────────────

# 2-D: (M, N) op (N,)  — row-vector broadcast

def test_path2_2d_row_broadcast_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))   # (4,6)
        var b = NDBuffer[dtype].arange(1, 7)                          # (6,)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path2_2d_row_broadcast_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(10, 34).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 7)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path2_2d_row_broadcast_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 7)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path2_2d_row_broadcast_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(24, 48).reshape(Shape(4, 6))
        var b = NDBuffer[dtype].arange(1, 7)                          # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

# 3-D: (B, T, C) op (C,)  — bias add (transformer inner loop)

def test_path2_3d_bias_add_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))  # (2,4,6)
        var b = NDBuffer[dtype].arange(1, 7)                            # (6,)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path2_3d_bias_add_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(50, 98).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 7)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path2_3d_bias_add_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 7)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path2_3d_bias_add_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(49, 97).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 7)                            # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

# 3-D: (B, T, C) op (T, C)  — 2-D broadcast into 3-D

def test_path2_3d_2d_into_3d_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path2_3d_2d_into_3d_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(50, 98).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path2_3d_2d_into_3d_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path2_3d_2d_into_3d_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(49, 97).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))     # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

# 4-D: (B, C, H, W) op (1, C, 1, 1)  — per-channel scale (BN/LN pattern)

def test_path2_4d_channel_broadcast_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        # (2, 4, 3, 3) = 72 elements
        var a = NDBuffer[dtype].arange(1, 73).reshape(Shape(2, 4, 3, 3))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4, 1, 1))  # (1,4,1,1)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path2_4d_channel_broadcast_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(73, 145).reshape(Shape(2, 4, 3, 3))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4, 1, 1))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path2_4d_channel_broadcast_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 73).reshape(Shape(2, 4, 3, 3))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4, 1, 1))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path2_4d_channel_broadcast_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(73, 145).reshape(Shape(2, 4, 3, 3))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4, 1, 1))  # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

# 2-D: (N, 1) op (1, M)  — outer-product broadcast pattern

def test_path2_2d_outer_product_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 6).reshape(Shape(5, 1))   # (5,1)
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4))   # (1,4)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path2_2d_outer_product_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(10, 15).reshape(Shape(5, 1))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path2_2d_outer_product_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 6).reshape(Shape(5, 1))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path2_2d_outer_product_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(10, 15).reshape(Shape(5, 1))
        var b = NDBuffer[dtype].arange(1, 5).reshape(Shape(1, 4))   # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

# 3-D: (B, T, C) op (1,)  — scalar broadcast (single-element tensor)

def test_path2_3d_scalar_broadcast_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(2, 3, 4))
        var b = NDBuffer[dtype].full(Shape(1), 2.0)                   # scalar 2.0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path2_3d_scalar_broadcast_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(2, 3, 4))
        var b = NDBuffer[dtype].full(Shape(1), 3.0)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path2_3d_scalar_broadcast_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(2, 3, 4))
        var b = NDBuffer[dtype].full(Shape(1), 2.0)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path2_3d_scalar_broadcast_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(2, 3, 4))
        var b = NDBuffer[dtype].full(Shape(1), 4.0)                   # > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

# 4-D: (B, T, H, C) op (H, C)  — 2-D broadcast into 4-D

def test_path2_4d_2d_into_4d_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        # (2, 3, 4, 4) = 96 elements
        var a = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 17).reshape(Shape(4, 4))        # (4,4)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path2_4d_2d_into_4d_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var b = NDBuffer[dtype].arange(1, 17).reshape(Shape(4, 4))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))


# ─────────────────────────────────────────────────────────────────────────────
# PATH 3: A contiguous and fills broadcast_shape; B is non-contiguous.
# A is read linearly. B is accessed via broadcast strides derived from B.strides.
# Created by transposing B so it is non-contiguous, then checking result matches CPU.
# ─────────────────────────────────────────────────────────────────────────────

def test_path3_2d_a_contiguous_b_transposed_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        # A: (4,6) contiguous.  B: (6,4) transposed → shape (4,6) non-contiguous.
        # broadcast_shape = (4,6) = A_shape → A fills it, B is non-contiguous.
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var b = b_base.transpose()   # shape (4,6), non-contiguous
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path3_2d_a_contiguous_b_transposed_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(20, 44).reshape(Shape(4, 6))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var b = b_base.transpose()
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path3_2d_a_contiguous_b_transposed_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var b = b_base.transpose()
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path3_2d_a_contiguous_b_transposed_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(24, 48).reshape(Shape(4, 6))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var b = b_base.transpose()                                    # all elements > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

def test_path3_3d_a_contiguous_b_transposed_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        # A: (2,4,6) contiguous.
        # B: arange reshaped to (2,6,4) then transposed last two dims → (2,4,6) non-contiguous.
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var b = b_base.transpose(IntArray(-1, -2))  # non-contiguous (2,4,6)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path3_3d_a_contiguous_b_transposed_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var b = b_base.transpose(IntArray(-1, -2))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))


# ─────────────────────────────────────────────────────────────────────────────
# PATH 4: B contiguous and fills broadcast_shape; A is non-contiguous.
# B is read linearly. A is accessed via broadcast strides derived from A.strides.
# Created by transposing A so it is non-contiguous.
# ─────────────────────────────────────────────────────────────────────────────

def test_path4_2d_a_transposed_b_contiguous_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        # A: (6,4) transposed → shape (4,6) non-contiguous.
        # B: (4,6) contiguous fills broadcast_shape (4,6).
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()   # shape (4,6), non-contiguous
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path4_2d_a_transposed_b_contiguous_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(20, 44).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path4_2d_a_transposed_b_contiguous_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path4_2d_a_transposed_b_contiguous_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(24, 48).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = NDBuffer[dtype].arange(1, 25).reshape(Shape(4, 6))   # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

def test_path4_3d_a_transposed_b_contiguous_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var a = a_base.transpose(IntArray(-1, -2))   # non-contiguous (2,4,6)
        var b = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path4_3d_a_transposed_b_contiguous_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var a = a_base.transpose(IntArray(-1, -2))
        var b = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))


# ─────────────────────────────────────────────────────────────────────────────
# PATH 5: Both non-contiguous — general strided fallback.
# Both A and B are transposed views, so both are non-contiguous.
# Exercises the both_strided kernel with A.strides and B.strides.
# ─────────────────────────────────────────────────────────────────────────────

def test_path5_2d_both_transposed_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()   # (4,6) non-contiguous
        var b = b_base.transpose()   # (4,6) non-contiguous
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path5_2d_both_transposed_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(20, 44).reshape(Shape(6, 4))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_path5_2d_both_transposed_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path5_2d_both_transposed_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(24, 48).reshape(Shape(6, 4))
        var b_base = NDBuffer[dtype].arange(1, 25).reshape(Shape(6, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()                                      # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

def test_path5_3d_both_transposed_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var b_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var a = a_base.transpose()   # (2,4,6) non-contiguous
        var b = b_base.transpose()
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path5_3d_both_transposed_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var b_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_path5_4d_both_transposed_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        # (2,3,4,4) transposed → (2,3,4,4) with swapped last two dims
        var a_base = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var b_base = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path5_4d_both_transposed_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a_base = NDBuffer[dtype].arange(97, 193).reshape(Shape(2, 3, 4, 4))
        var b_base = NDBuffer[dtype].arange(1, 97).reshape(Shape(2, 3, 4, 4))
        var a = a_base.transpose()
        var b = b_base.transpose()                                     # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

# PATH 5 with share() — custom non-default strides (not just a transpose)

def test_path5_2d_custom_strides_both_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        # share() with custom strides: non-contiguous views into a larger buffer.
        var buf_a = Buffer[dtype].arange(1, 25)
        var buf_b = Buffer[dtype].arange(1, 25)
        # shape (3,3), strides (1,3), offset 0 — column-major-ish, non-contiguous
        var a_o = NDBuffer[dtype](buf_a, Shape(3, 3))
        var a = a_o.share(Shape(3, 3), Strides(1, 3), offset=0)
        var b_o = NDBuffer[dtype](buf_b, Shape(3, 3))
        var b = b_o.share(Shape(3, 3), Strides(1, 3), offset=0)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_path5_2d_custom_strides_both_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var buf_a = Buffer[dtype].arange(1, 25)
        var buf_b = Buffer[dtype].arange(1, 25)
        var a_o = NDBuffer[dtype](buf_a, Shape(3, 3))
        var a = a_o.share(Shape(3, 3), Strides(1, 3), offset=0)
        var b_o = NDBuffer[dtype](buf_b, Shape(3, 3))
        var b = b_o.share(Shape(3, 3), Strides(1, 3), offset=0)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

# Tail-loop: sizes that are not a multiple of simd_width (8 for fp32),
# and smaller than one CHUNK_SIZE (128 for fp32 with simd_width=8).
# These force the scalar tail-loop path inside the kernel.

def test_edge_tail_size7_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype](1, 2, 3, 4, 5, 6, 7)           # 7 elements
        var b = NDBuffer[dtype](7, 6, 5, 4, 3, 2, 1)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_edge_tail_size7_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype](1, 2, 3, 4, 5, 6, 7)
        var b = NDBuffer[dtype](1, 2, 3, 4, 5, 6, 7)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

def test_edge_tail_size13_subtract_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(10, 23)                  # 13 elements
        var b = NDBuffer[dtype].arange(1, 14)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Subtract](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a - b))

def test_edge_tail_size17_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(17, 34)                  # 17 elements
        var b = NDBuffer[dtype].arange(1, 18)                   # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

# Sub-CHUNK size: 33 elements — less than CHUNK_SIZE=128, covers the
# case where only one thread block processes any data.

def test_edge_subchunk_size33_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 34)                   # 33 elements
        var b = NDBuffer[dtype].arange(1, 34)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_edge_subchunk_size33_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 34)
        var b = NDBuffer[dtype].arange(1, 34)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

# Large tensor: 1000 elements — not a multiple of simd_width,
# crosses multiple CHUNK boundaries, exercises grid-stride loop.

def test_edge_large_size1000_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 1001)                 # 1000 elements
        var b = NDBuffer[dtype].arange(1, 1001)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_edge_large_size1000_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1000, 2000)              # 1000 elements, all > 0
        var b = NDBuffer[dtype].arange(1, 1001)                 # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

# Single element — minimum possible tensor.

def test_edge_single_element_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].full(Shape(1), 5.0)
        var b = NDBuffer[dtype].full(Shape(1), 3.0)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_edge_single_element_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].full(Shape(1), 10.0)
        var b = NDBuffer[dtype].full(Shape(1), 4.0)             # > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

# Large 4D tensor crossing many CHUNK boundaries (exercises the grid-stride
# while-loop across multiple iterations).

def test_edge_large_4d_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        # (4, 8, 16, 16) = 8192 elements — well above the launch_config thresholds
        var a = NDBuffer[dtype].arange(1, 8193).reshape(Shape(4, 8, 16, 16))
        var b = NDBuffer[dtype].arange(1, 8193).reshape(Shape(4, 8, 16, 16))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_edge_large_4d_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 8193).reshape(Shape(4, 8, 16, 16))
        var b = NDBuffer[dtype].arange(1, 8193).reshape(Shape(4, 8, 16, 16))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

# PATH 2 broadcast with non-power-of-two inner dim — stresses the
# coordinate decomposition modulo arithmetic on odd shapes.

def test_edge_broadcast_nonpow2_inner_dim_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        # (3, 5, 7) + (7,) — inner dim 7 is neither power-of-2 nor multiple of simd_width
        var a = NDBuffer[dtype].arange(1, 106).reshape(Shape(3, 5, 7))
        var b = NDBuffer[dtype].arange(1, 8)                    # (7,)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_edge_broadcast_nonpow2_inner_dim_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 106).reshape(Shape(3, 5, 7))
        var b = NDBuffer[dtype].arange(1, 8)
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

# PATH 2 broadcast with rank mismatch > 1: (2,3,4,5) + (4,5) — 4-D into 2-D

def test_edge_broadcast_rank_mismatch_4d_2d_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 121).reshape(Shape(2, 3, 4, 5))  # 120 elements
        var b = NDBuffer[dtype].arange(1, 21).reshape(Shape(4, 5))
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_edge_broadcast_rank_mismatch_4d_2d_divide_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(121, 241).reshape(Shape(2, 3, 4, 5))
        var b = NDBuffer[dtype].arange(1, 21).reshape(Shape(4, 5))          # all > 0
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Divide](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a / b))

# PATH 2 with all-ones broadcast tensor (common for masking / scaling by 1).

def test_edge_broadcast_ones_multiply_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].full(Shape(1), 1.0)             # scalar 1 → no-op multiply
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Multiply](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a * b))

# PATH 2 with all-zeros broadcast tensor (zero-add identity).

def test_edge_broadcast_zeros_add_gpu_arith() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
        var b = NDBuffer[dtype].full(Shape(6), 0.0)             # zero bias → identity add
        var c_gpu = a.to_gpu(gpu).arithmetic_ops[Add](b.to_gpu(gpu), sync=True)
        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
