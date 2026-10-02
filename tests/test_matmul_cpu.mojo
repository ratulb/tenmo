"""
Correctness tests for Tenmo's CPU matmul (matmul_cpu.mojo).
2D and ND with
broadcasting.

Every test validates actual VALUES against naive_matmul_2d / naive_matmul_
broadcast below -- deliberately simple triple-loop implementations with no
tiling, no SIMD, no prefetch, no parallelism -- not just output shape.

Tile-size dispatch coverage: matmul_cpu.mojo selects TILE_M in {32,64,128},
TILE_N in {32,64}, TILE_P in {64,128,256} based on runtime m/n/p, then
dispatches to one of 18 compile-time-specialized kernel instantiations.
MmCpu2d and MmCpuNd now share those kernels through the MatmulCpu trait
(matmul_simd_tile / matmul_scalar_tile) but still own separate loop
frameworks, so each dispatcher is independently swept here.

m/n/p bucket-representative values below are deliberately NOT exact
multiples of their tile size, so every dispatch test also exercises
partial/remainder-tile handling, not just the clean-multiple case -- this
is exactly the class of bug the "FIX Issue 1/2" comments in matmul_cpu.mojo
document as having occurred before.

ASSUMPTIONS VERIFIED AGAINST REAL TENMO:
  1. matmul_cpu.mojo exports NO top-level `matmul`; the public dispatcher
     is Tensor.matmul -> Matmul.forward (tenmo/matmul.mojo). A thin
     helper below routes to A.matmul[track_grad=False](B).
   2. Tensor.transpose() confirmed -- default reverses all axes, producing
      a non-contiguous view via NDBuffer.transpose (always-view).
  3. Tensor[dtype].zeros(Shape(out_dims)) confirmed -- Shape has a
     List[Int] constructor (used by Tensor.zeros(List[Int]) at
     tensor.mojo:1486); the variadic Tensor.zeros(m, p) is also confirmed.
"""

from std.testing import assert_true, assert_equal, TestSuite
from tenmo.tensor import Tensor
from tenmo.ndbuffer import Shape


def matmul(A: Tensor[dtype], B: Tensor[dtype]) raises -> Tensor[dtype]:
    """Thin route to the public dispatcher: Tensor.matmul -> Matmul.forward.
    Inputs are requires_grad=False leaves, so track_grad=False forward-only."""
    return A.matmul[track_grad=False](B)

comptime dtype = DType.float32
comptime TOL_RTOL = Scalar[dtype](1e-4)
comptime TOL_ATOL = Scalar[dtype](1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Naive reference implementations -- ground truth. Deliberately simple:
# one scalar multiply-add at a time, no blocking, no vectorization.
# ═══════════════════════════════════════════════════════════════════════

def naive_matmul_2d(A: Tensor[dtype], B: Tensor[dtype]) raises -> Tensor[dtype]:
    """Plain triple-loop 2D matmul."""
    var m = A.shape()[0]
    var n = A.shape()[1]
    var p = B.shape()[1]
    var C = Tensor[dtype].zeros(m, p)

    var A_strides = A.buffer.strides
    var A_offset = A.buffer.offset
    var A_data = A.buffer.data_ptr()
    var B_strides = B.buffer.strides
    var B_offset = B.buffer.offset
    var B_data = B.buffer.data_ptr()
    var C_data = C.buffer.data_ptr()

    for i in range(m):
        for j in range(p):
            var acc: Scalar[dtype] = 0
            for k in range(n):
                var a = A_data[unsafe_offset=i * A_strides[0] + k * A_strides[1] + A_offset]
                var b = B_data[unsafe_offset=k * B_strides[0] + j * B_strides[1] + B_offset]
                acc += a * b
            C_data[unsafe_offset=i * p + j] = acc
    return C^


def naive_matmul_broadcast(A: Tensor[dtype], B: Tensor[dtype]) raises -> Tensor[dtype]:
    """Broadcast-aware naive matmul. Independently re-derives the standard.
    broadcasting rule (align trailing dims; a batch dim of size 1 always
    maps to index 0) -- this is the same well-known algorithm the real
    MmCpuNd uses, since there isn't another correct way to do it, but the
    actual compute here is unoptimized and structurally independent of
    the tiled/SIMD kernel, so it still validates the kernel's arithmetic
    and boundary handling."""
    var A_rank = A.shape().rank()
    var B_rank = B.shape().rank()
    var m = A.shape()[A_rank - 2]
    var n = A.shape()[A_rank - 1]
    var p = B.shape()[B_rank - 1]

    var A_batch_rank = A_rank - 2
    var B_batch_rank = B_rank - 2
    var batch_rank = max(A_batch_rank, B_batch_rank)

    var batch_shape = List[Int]()
    for d in range(batch_rank):
        var A_d = d - (batch_rank - A_batch_rank)
        var B_d = d - (batch_rank - B_batch_rank)
        var a_dim = A.shape()[A_d] if A_d >= 0 else 1
        var b_dim = B.shape()[B_d] if B_d >= 0 else 1
        assert_true(
            a_dim == b_dim or a_dim == 1 or b_dim == 1,
            "naive_matmul_broadcast: incompatible batch shapes",
        )
        batch_shape.append(max(a_dim, b_dim))

    var total_batch = 1
    for d in range(len(batch_shape)):
        total_batch *= batch_shape[d]

    var out_dims = List[Int]()
    for d in range(len(batch_shape)):
        out_dims.append(batch_shape[d])
    out_dims.append(m)
    out_dims.append(p)
    var C = Tensor[dtype].zeros(Shape(out_dims))

    var A_strides = A.buffer.strides
    var A_offset0 = A.buffer.offset
    var A_data = A.buffer.data_ptr()
    var B_strides = B.buffer.strides
    var B_offset0 = B.buffer.offset
    var B_data = B.buffer.data_ptr()
    var C_data = C.buffer.data_ptr()

    for batch in range(total_batch):
        var remaining = batch
        var divisor = 1
        var A_off = A_offset0
        var B_off = B_offset0
        for d in range(batch_rank - 1, -1, -1):
            var coord = (remaining // divisor) % batch_shape[d]
            var A_d = d - (batch_rank - A_batch_rank)
            if A_d >= 0 and A.shape()[A_d] > 1:
                A_off += coord * A_strides[A_d]
            var B_d = d - (batch_rank - B_batch_rank)
            if B_d >= 0 and B.shape()[B_d] > 1:
                B_off += coord * B_strides[B_d]
            divisor *= batch_shape[d]

        var C_off = batch * m * p
        for i in range(m):
            for j in range(p):
                var acc: Scalar[dtype] = 0
                for k in range(n):
                    var a = A_data[
                        unsafe_offset=A_off + i * A_strides[A_rank - 2] + k * A_strides[A_rank - 1]
                    ]
                    var b = B_data[
                        unsafe_offset=B_off + k * B_strides[B_rank - 2] + j * B_strides[B_rank - 1]
                    ]
                    acc += a * b
                C_data[unsafe_offset=C_off + i * p + j] = acc
    return C^


def assert_tensors_close(
    actual: Tensor[dtype],
    expected: Tensor[dtype],
    rtol: Scalar[dtype] = TOL_RTOL,
    atol: Scalar[dtype] = TOL_ATOL,
) raises:
    """Elementwise closeness check. Tiled accumulation legitimately changes.
    floating-point rounding versus naive summation order (see matmul_cpu.
    mojo review) -- exact equality is the wrong check here.

    Assumes both tensors are contiguous and freshly allocated (true for
    matmul's own output and for both naive references, which are always
    Tensor.zeros()-allocated), so flat-index iteration is valid."""
    assert_equal(actual.shape().rank(), expected.shape().rank())
    for d in range(actual.shape().rank()):
        assert_equal(actual.shape()[d], expected.shape()[d])

    var a_ptr = actual.buffer.data_ptr()
    var a_off = actual.buffer.offset
    var e_ptr = expected.buffer.data_ptr()
    var e_off = expected.buffer.offset

    var total = 1
    for d in range(actual.shape().rank()):
        total *= actual.shape()[d]

    for idx in range(total):
        var a_val = a_ptr[unsafe_offset=a_off + idx]
        var e_val = e_ptr[unsafe_offset=e_off + idx]
        var diff = abs(a_val - e_val)
        var tol = atol + rtol * abs(e_val)
        assert_true(
            diff <= tol,
            "matmul mismatch at flat index "
            + String(idx)
            + ": got "
            + String(a_val)
            + " expected "
            + String(e_val),
        )


# ═══════════════════════════════════════════════════════════════════════
# 2D — all 18 tile-size dispatch combinations.
#
# Bucket-representative values (deliberately non-multiples of tile size,
# to also exercise partial/remainder-tile handling):
#   TILE_M=32  -> m=45   (<=64;         45 = 32+13)
#   TILE_M=64  -> m=100  (64<m<=256;   100 = 64+36)
#   TILE_M=128 -> m=260  (m>256;       260 = 128+128+4)
#   TILE_N=32  -> n=45   (<=64)
#   TILE_N=64  -> n=100  (>64)
#   TILE_P=64  -> p=45   (<=64;  exercises full-unroll + vector-tail + scalar-tail)
#   TILE_P=128 -> p=100  (64<p<=256; single j_tile, mostly full-unroll blocks)
#   TILE_P=256 -> p=260  (>256; spans two j_tiles, exercises j_tile boundary)
# ═══════════════════════════════════════════════════════════════════════

def test_2d_tilem32_tilen32_tilep64() raises:
    """2D matmul, TILE_M=32, TILE_N=32, TILE_P=64 (m=45, n=45, p=45)."""
    var A = Tensor[dtype].randn(45, 45)
    var B = Tensor[dtype].randn(45, 45)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem32_tilen32_tilep128() raises:
    """2D matmul, TILE_M=32, TILE_N=32, TILE_P=128 (m=45, n=45, p=100)."""
    var A = Tensor[dtype].randn(45, 45)
    var B = Tensor[dtype].randn(45, 100)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem32_tilen32_tilep256() raises:
    """2D matmul, TILE_M=32, TILE_N=32, TILE_P=256 (m=45, n=45, p=260)."""
    var A = Tensor[dtype].randn(45, 45)
    var B = Tensor[dtype].randn(45, 260)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem32_tilen64_tilep64() raises:
    """2D matmul, TILE_M=32, TILE_N=64, TILE_P=64 (m=45, n=100, p=45)."""
    var A = Tensor[dtype].randn(45, 100)
    var B = Tensor[dtype].randn(100, 45)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem32_tilen64_tilep128() raises:
    """2D matmul, TILE_M=32, TILE_N=64, TILE_P=128 (m=45, n=100, p=100)."""
    var A = Tensor[dtype].randn(45, 100)
    var B = Tensor[dtype].randn(100, 100)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem32_tilen64_tilep256() raises:
    """2D matmul, TILE_M=32, TILE_N=64, TILE_P=256 (m=45, n=100, p=260)."""
    var A = Tensor[dtype].randn(45, 100)
    var B = Tensor[dtype].randn(100, 260)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem64_tilen32_tilep64() raises:
    """2D matmul, TILE_M=64, TILE_N=32, TILE_P=64 (m=100, n=45, p=45)."""
    var A = Tensor[dtype].randn(100, 45)
    var B = Tensor[dtype].randn(45, 45)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem64_tilen32_tilep128() raises:
    """2D matmul, TILE_M=64, TILE_N=32, TILE_P=128 (m=100, n=45, p=100)."""
    var A = Tensor[dtype].randn(100, 45)
    var B = Tensor[dtype].randn(45, 100)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem64_tilen32_tilep256() raises:
    """2D matmul, TILE_M=64, TILE_N=32, TILE_P=256 (m=100, n=45, p=260)."""
    var A = Tensor[dtype].randn(100, 45)
    var B = Tensor[dtype].randn(45, 260)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem64_tilen64_tilep64() raises:
    """2D matmul, TILE_M=64, TILE_N=64, TILE_P=64 (m=100, n=100, p=45)."""
    var A = Tensor[dtype].randn(100, 100)
    var B = Tensor[dtype].randn(100, 45)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem64_tilen64_tilep128() raises:
    """2D matmul, TILE_M=64, TILE_N=64, TILE_P=128 (m=100, n=100, p=100)."""
    var A = Tensor[dtype].randn(100, 100)
    var B = Tensor[dtype].randn(100, 100)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem64_tilen64_tilep256() raises:
    """2D matmul, TILE_M=64, TILE_N=64, TILE_P=256 (m=100, n=100, p=260)."""
    var A = Tensor[dtype].randn(100, 100)
    var B = Tensor[dtype].randn(100, 260)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem128_tilen32_tilep64() raises:
    """2D matmul, TILE_M=128, TILE_N=32, TILE_P=64 (m=260, n=45, p=45)."""
    var A = Tensor[dtype].randn(260, 45)
    var B = Tensor[dtype].randn(45, 45)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem128_tilen32_tilep128() raises:
    """2D matmul, TILE_M=128, TILE_N=32, TILE_P=128 (m=260, n=45, p=100)."""
    var A = Tensor[dtype].randn(260, 45)
    var B = Tensor[dtype].randn(45, 100)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem128_tilen32_tilep256() raises:
    """2D matmul, TILE_M=128, TILE_N=32, TILE_P=256 (m=260, n=45, p=260)."""
    var A = Tensor[dtype].randn(260, 45)
    var B = Tensor[dtype].randn(45, 260)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem128_tilen64_tilep64() raises:
    """2D matmul, TILE_M=128, TILE_N=64, TILE_P=64 (m=260, n=100, p=45)."""
    var A = Tensor[dtype].randn(260, 100)
    var B = Tensor[dtype].randn(100, 45)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem128_tilen64_tilep128() raises:
    """2D matmul, TILE_M=128, TILE_N=64, TILE_P=128 (m=260, n=100, p=100)."""
    var A = Tensor[dtype].randn(260, 100)
    var B = Tensor[dtype].randn(100, 100)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_2d_tilem128_tilen64_tilep256() raises:
    """2D matmul, TILE_M=128, TILE_N=64, TILE_P=256 (m=260, n=100, p=260)."""
    var A = Tensor[dtype].randn(260, 100)
    var B = Tensor[dtype].randn(100, 260)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


# ═══════════════════════════════════════════════════════════════════════
# ND — same 18 tile-size dispatch combinations, batched (batch=3, no
# broadcasting -- separate tests below cover broadcasting semantics in
# isolation). This is a SEPARATE 18-combination sweep specifically
# because MmCpuNd reimplements the kernel inline rather than calling
# MmCpu2d -- a bug fixed in one is not automatically fixed in the other.
# ═══════════════════════════════════════════════════════════════════════

def test_nd_tilem32_tilen32_tilep64() raises:
    """ND matmul, TILE_M=32, TILE_N=32, TILE_P=64, batch=3 (m=45, n=45, p=45)."""
    var A = Tensor[dtype].randn(3, 45, 45)
    var B = Tensor[dtype].randn(3, 45, 45)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem32_tilen32_tilep128() raises:
    """ND matmul, TILE_M=32, TILE_N=32, TILE_P=128, batch=3 (m=45, n=45, p=100)."""
    var A = Tensor[dtype].randn(3, 45, 45)
    var B = Tensor[dtype].randn(3, 45, 100)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem32_tilen32_tilep256() raises:
    """ND matmul, TILE_M=32, TILE_N=32, TILE_P=256, batch=3 (m=45, n=45, p=260)."""
    var A = Tensor[dtype].randn(3, 45, 45)
    var B = Tensor[dtype].randn(3, 45, 260)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem32_tilen64_tilep64() raises:
    """ND matmul, TILE_M=32, TILE_N=64, TILE_P=64, batch=3 (m=45, n=100, p=45)."""
    var A = Tensor[dtype].randn(3, 45, 100)
    var B = Tensor[dtype].randn(3, 100, 45)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem32_tilen64_tilep128() raises:
    """ND matmul, TILE_M=32, TILE_N=64, TILE_P=128, batch=3 (m=45, n=100, p=100)."""
    var A = Tensor[dtype].randn(3, 45, 100)
    var B = Tensor[dtype].randn(3, 100, 100)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem32_tilen64_tilep256() raises:
    """ND matmul, TILE_M=32, TILE_N=64, TILE_P=256, batch=3 (m=45, n=100, p=260)."""
    var A = Tensor[dtype].randn(3, 45, 100)
    var B = Tensor[dtype].randn(3, 100, 260)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem64_tilen32_tilep64() raises:
    """ND matmul, TILE_M=64, TILE_N=32, TILE_P=64, batch=3 (m=100, n=45, p=45)."""
    var A = Tensor[dtype].randn(3, 100, 45)
    var B = Tensor[dtype].randn(3, 45, 45)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem64_tilen32_tilep128() raises:
    """ND matmul, TILE_M=64, TILE_N=32, TILE_P=128, batch=3 (m=100, n=45, p=100)."""
    var A = Tensor[dtype].randn(3, 100, 45)
    var B = Tensor[dtype].randn(3, 45, 100)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem64_tilen32_tilep256() raises:
    """ND matmul, TILE_M=64, TILE_N=32, TILE_P=256, batch=3 (m=100, n=45, p=260)."""
    var A = Tensor[dtype].randn(3, 100, 45)
    var B = Tensor[dtype].randn(3, 45, 260)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem64_tilen64_tilep64() raises:
    """ND matmul, TILE_M=64, TILE_N=64, TILE_P=64, batch=3 (m=100, n=100, p=45)."""
    var A = Tensor[dtype].randn(3, 100, 100)
    var B = Tensor[dtype].randn(3, 100, 45)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem64_tilen64_tilep128() raises:
    """ND matmul, TILE_M=64, TILE_N=64, TILE_P=128, batch=3 (m=100, n=100, p=100)."""
    var A = Tensor[dtype].randn(3, 100, 100)
    var B = Tensor[dtype].randn(3, 100, 100)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem64_tilen64_tilep256() raises:
    """ND matmul, TILE_M=64, TILE_N=64, TILE_P=256, batch=3 (m=100, n=100, p=260)."""
    var A = Tensor[dtype].randn(3, 100, 100)
    var B = Tensor[dtype].randn(3, 100, 260)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem128_tilen32_tilep64() raises:
    """ND matmul, TILE_M=128, TILE_N=32, TILE_P=64, batch=3 (m=260, n=45, p=45)."""
    var A = Tensor[dtype].randn(3, 260, 45)
    var B = Tensor[dtype].randn(3, 45, 45)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem128_tilen32_tilep128() raises:
    """ND matmul, TILE_M=128, TILE_N=32, TILE_P=128, batch=3 (m=260, n=45, p=100)."""
    var A = Tensor[dtype].randn(3, 260, 45)
    var B = Tensor[dtype].randn(3, 45, 100)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem128_tilen32_tilep256() raises:
    """ND matmul, TILE_M=128, TILE_N=32, TILE_P=256, batch=3 (m=260, n=45, p=260)."""
    var A = Tensor[dtype].randn(3, 260, 45)
    var B = Tensor[dtype].randn(3, 45, 260)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem128_tilen64_tilep64() raises:
    """ND matmul, TILE_M=128, TILE_N=64, TILE_P=64, batch=3 (m=260, n=100, p=45)."""
    var A = Tensor[dtype].randn(3, 260, 100)
    var B = Tensor[dtype].randn(3, 100, 45)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem128_tilen64_tilep128() raises:
    """ND matmul, TILE_M=128, TILE_N=64, TILE_P=128, batch=3 (m=260, n=100, p=100)."""
    var A = Tensor[dtype].randn(3, 260, 100)
    var B = Tensor[dtype].randn(3, 100, 100)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_nd_tilem128_tilen64_tilep256() raises:
    """ND matmul, TILE_M=128, TILE_N=64, TILE_P=256, batch=3 (m=260, n=100, p=260)."""
    var A = Tensor[dtype].randn(3, 260, 100)
    var B = Tensor[dtype].randn(3, 100, 260)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


# ═══════════════════════════════════════════════════════════════════════
# Broadcasting semantics — isolated from tile-size concerns, modest sizes.
# ═══════════════════════════════════════════════════════════════════════

def test_3d_times_2d() raises:
    """(1, 2, 4) @ (4, 9) -> (1, 2, 9): 2D operand broadcasts as batch=1."""
    var A = Tensor[dtype].randn(1, 2, 4)
    var B = Tensor[dtype].randn(4, 9)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_2d_times_3d() raises:
    """(4, 9) @ (3, 9, 5) -> (3, 4, 5): 2D operand on the LEFT broadcasts."""
    var A = Tensor[dtype].randn(4, 9)
    var B = Tensor[dtype].randn(3, 9, 5)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_batch_broadcast_A_singleton() raises:
    """(1, 4, 6) @ (5, 6, 3) -> (5, 4, 3): A's batch dim broadcasts."""
    var A = Tensor[dtype].randn(1, 4, 6)
    var B = Tensor[dtype].randn(5, 6, 3)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_batch_broadcast_B_singleton() raises:
    """(5, 4, 6) @ (1, 6, 3) -> (5, 4, 3): B's batch dim broadcasts."""
    var A = Tensor[dtype].randn(5, 4, 6)
    var B = Tensor[dtype].randn(1, 6, 3)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_batch_dims_equal_no_broadcast() raises:
    """(4, 4, 6) @ (4, 6, 3) -> (4, 4, 3): matching batch dims, no broadcast."""
    var A = Tensor[dtype].randn(4, 4, 6)
    var B = Tensor[dtype].randn(4, 6, 3)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


def test_4d_multi_axis_broadcast() raises:
    """(3, 1, 4, 6) @ (1, 5, 6, 2) -> (3, 5, 4.
    2): both operands broadcast
    on different batch axes simultaneously."""
    var A = Tensor[dtype].randn(3, 1, 4, 6)
    var B = Tensor[dtype].randn(1, 5, 6, 2)
    assert_tensors_close(matmul(A, B), naive_matmul_broadcast(A, B))


# NOTE: a test asserting that mismatched, non-broadcastable batch shapes
# raise/fail is deliberately omitted. matmul_cpu.mojo's panic() calls
# abort() -- a hard process abort, not a catchable Error -- so it cannot
# be exercised safely inside this test process without crashing the whole
# suite. If you want to verify that path, do it via a subprocess harness
# that spawns the mismatched-shape call and checks for a nonzero exit code.


# ═══════════════════════════════════════════════════════════════════════
# Edge cases and named regressions.
# ═══════════════════════════════════════════════════════════════════════

def test_tall_narrow_matrix_tile_p_regression() raises:
    """Direct regression guard for the documented 'FIX Issue 1': a tall.
    narrow matrix (large m, tiny p) must select a small TILE_P, not a
    large one sized for wide matrices. m=300 (TILE_M=128), p=5 (well
    under one SIMD vector width -- forces scalar-tail-only for every
    column, the exact shape the original bug mishandled)."""
    var A = Tensor[dtype].randn(300, 50)
    var B = Tensor[dtype].randn(50, 5)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_wide_short_matrix() raises:
    """Small m.
    Large n and p -- the inverse pathological shape from the
    tall-narrow regression above."""
    var A = Tensor[dtype].randn(3, 200)
    var B = Tensor[dtype].randn(200, 200)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_simd_scalar_tail_only() raises:
    """P[p]=3.
    Under simdwidth (8 for float32/AVX2) -- every column falls to
    the pure scalar tail; the SIMD paths never fire at all."""
    var A = Tensor[dtype].randn(10, 10)
    var B = Tensor[dtype].randn(10, 3)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_simd_single_vector_tail_no_full_unroll() raises:
    """P[p]=20: at or above simdwidth (8) but under simd_unroll (32) -- hits.
    the single-SIMD-vector tail tier without ever reaching the 4-wide
    unrolled block."""
    var A = Tensor[dtype].randn(10, 10)
    var B = Tensor[dtype].randn(10, 20)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_single_element_matmul() raises:
    """1x1 @ 1x1 -- minimal possible shape."""
    var A = Tensor[dtype].randn(1, 1)
    var B = Tensor[dtype].randn(1, 1)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_row_vector_times_matrix() raises:
    """M[m]=1: a single row times a full matrix."""
    var A = Tensor[dtype].randn(1, 50)
    var B = Tensor[dtype].randn(50, 50)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_matrix_times_column_vector() raises:
    """P[p]=1: a full matrix times a single column."""
    var A = Tensor[dtype].randn(50, 50)
    var B = Tensor[dtype].randn(50, 1)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_non_contiguous_A() raises:
    """Transposed A forces Path 1b (A non-contiguous, B contiguous)."""
    var A_base = Tensor[dtype].randn(30, 40)
    var A = A_base.transpose()  # non-contiguous view (Path 1b)
    var B = Tensor[dtype].randn(30, 25)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def test_non_contiguous_B() raises:
    """Transposed B forces Path 2 (scalar fallback) -- worth extra.
    scrutiny given how common A @ B.T is in attention/backward code."""
    var A = Tensor[dtype].randn(30, 25)
    var B_base = Tensor[dtype].randn(40, 25)
    var B = B_base.transpose()  # non-contiguous view (Path 2)
    assert_tensors_close(matmul(A, B), naive_matmul_2d(A, B))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
