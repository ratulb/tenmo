from tenmo import NDBuffer, Shape, Tensor

# from debug import MM
from std.testing import assert_true, assert_equal
from tenmo.ndbuffer import MmCpu2d

comptime dtype = DType.float32


def rref(A: NDBuffer[dtype], B: NDBuffer[dtype]) -> NDBuffer[dtype]:
    return A.matmul_2d_good(B)


def rref_naive(A: NDBuffer[dtype], B: NDBuffer[dtype]) -> NDBuffer[dtype]:
    ref A_shape = A.shape
    ref B_shape = B.shape
    var m = A_shape[0]
    var n = A_shape[1]
    var p = B_shape[1]

    var C = NDBuffer[dtype].zeros(Shape(m, p))
    for i in range(m):
        for k in range(n):
            for j in range(p):
                C[i, j] += A[i, k] * B[k, j]
    return C


def trial(A: NDBuffer[dtype], B: NDBuffer[dtype]) -> NDBuffer[dtype]:
    return MmCpu2d.matmul_2d_cpu(A, B)


def check(M: Int, K: Int, N: Int, label: String) raises:
    var A = Tensor[dtype].randn(M, K)
    var B = Tensor[dtype].randn(K, N)
    # var r = rref(A.buffer, B.buffer)
    var r = rref_naive(A.buffer, B.buffer)
    var t = trial(A.buffer, B.buffer)
    assert_equal(r.shape, t.shape, "shape mismatch: " + label)
    assert_true(r.all_close[atol=1e-4](t), "value mismatch: " + label)
    print("  PASS:", label)


def test_tiny() raises:
    check(1, 1, 1, "1x1 @ 1x1")
    check(1, 4, 1, "1x4 @ 4x1")
    check(1, 4, 8, "1x4 @ 4x8")
    check(8, 4, 1, "8x4 @ 4x1")


def test_small() raises:
    check(2, 3, 4, "(2,3) @ (3,4)")
    check(4, 5, 6, "(4,5) @ (5,6)")
    check(7, 2, 3, "(7,2) @ (2,3)")


def test_medium() raises:
    check(16, 12, 8, "(16,12) @ (12,8)")
    check(32, 16, 32, "(32,16) @ (16,32)")
    check(64, 32, 64, "(64,32) @ (32,64)")


def test_tile_boundaries() raises:
    # TILE_M=64, TILE_N=64, TILE_P=128, UNROLL=4, simdwidth=8
    # Test exact multiples and boundaries
    check(64, 64, 128, "(64,64) @ (64,128) — exact tiles")
    check(65, 65, 129, "(65,65) @ (65,129) — +1 past tile")
    check(127, 63, 255, "(127,63) @ (63,255) — odd sizes")
    check(32, 128, 64, "(32,128) @ (128,64)")
    check(50, 33, 77, "(50,33) @ (33,77) — prime-adjacent")


def test_simd_tails() raises:
    # simdwidth=8, simd_unroll=32 → test just past these
    check(8, 8, 9, "(8,8) @ (8,9) — j_tail 1 past simd")
    check(8, 8, 33, "(8,8) @ (8,33) — j_tail 1 past unroll")
    check(8, 8, 39, "(8,8) @ (8,39) — k_tail, j_tail")
    check(8, 4, 31, "(8,4) @ (4,31)")


def test_non_contiguous_B() raises:
    # Create B with stride > 1 in last dim by using a wider buffer + view
    var K = 8
    var N = 6
    var pad = 2  # extra columns
    var M = 10

    var A = Tensor[dtype].randn(M, K)
    var B_wide = Tensor[dtype].randn(K, N + pad)
    # View into B_wide with shape (K, N) but strides (N+pad, 1):
    # is_contiguous() will see stride0 = N+pad ≠ N → False
    var B_strided = B_wide.buffer.share(
        Shape(K, N), B_wide.buffer.strides, B_wide.buffer.offset
    )

    # Reference: contiguous copy of the N columns
    var B_contig = Tensor[dtype].zeros(K, N)
    for i in range(K):
        for j in range(N):
            B_contig[i, j] = B_wide[i, j]
    var r = rref(A.buffer, B_contig.buffer)
    var t = trial(A.buffer, B_strided)

    assert_equal(r.shape, t.shape, "shape mismatch: non-contig B")
    assert_true(r.all_close[atol=1e-4](t), "value mismatch: non-contig B")
    print("  PASS: non-contiguous B (padded view)")


def test_all_ones() raises:
    # All-ones: result[i,j] = K
    var M = 7
    var K = 5
    var N = 9
    var A = Tensor[dtype].ones(M, K)
    var B = Tensor[dtype].ones(K, N)
    var t = trial(A.buffer, B.buffer)
    var expected = Tensor[dtype].full(Shape(M, N), Float32(K))
    assert_true(
        t.all_close[atol=1e-6](expected.buffer),
        "all-ones mismatch",
    )
    print("  PASS: all-ones")


def test_identity() raises:
    # A = identity: result = B
    var M = 5
    var _ = 5
    var N = 7
    var A = Tensor[dtype].eye(M)
    var B = Tensor[dtype].randn(M, N)
    var t = trial(A.buffer, B.buffer)
    assert_true(
        t.all_close[atol=1e-6](B.buffer),
        "identity mismatch",
    )
    print("  PASS: identity")


def test_non_contiguous_A() raises:
    # Non-contiguous A → a_also_contiguous=False path
    var M = 8
    var K = 6
    var N = 10
    var pad = 3
    var A_wide = Tensor[dtype].randn(M, K + pad)
    var A_strided = A_wide.buffer.share(
        Shape(M, K), A_wide.buffer.strides, A_wide.buffer.offset
    )
    var B = Tensor[dtype].randn(K, N)
    var B_strided = B.buffer.share(
        B.buffer.shape, B.buffer.strides, B.buffer.offset
    )
    # Reference: contiguous copy of A
    var A_contig = Tensor[dtype].zeros(M, K)
    for i in range(M):
        for j in range(K):
            A_contig[i, j] = A_wide[i, j]
    var r = rref(A_contig.buffer, B_strided)
    var t = trial(A_strided, B_strided)
    assert_true(
        r.all_close[atol=1e-4](t),
        "non-contiguous A mismatch",
    )
    print("  PASS: non-contiguous A (a_also_contiguous=False)")


def test_large_n() raises:
    # Large n → multiple k_tiles → exercises k_tile==0 fast path + prefetch
    check(16, 256, 32, "(16,256) @ (256,32) — multiple k_tiles")
    check(8, 512, 16, "(8,512) @ (512,16) — 8 k_tiles")


def test_square_large() raises:
    # Square matrices that span many tiles
    check(128, 128, 128, "(128,128) @ (128,128)")
    check(256, 64, 256, "(256,64) @ (64,256)")
    check(64, 256, 64, "(64,256) @ (256,64)")


def test_tall_skinny() raises:
    # Tall-skinny: many rows, small inner dim
    check(1024, 8, 16, "(1024,8) @ (8,16) — tall + skinny")


def test_wide_short() raises:
    # Wide-short: large p, small m and n
    check(4, 8, 1024, "(4,8) @ (8,1024) — wide output")


def test_k_tile_exact_multiple() raises:
    # n is exact multiple of TILE_N=64
    check(8, 128, 16, "(8,128) @ (128,16) — exact k_tile boundary")
    check(13, 192, 27, "(13,192) @ (192,27) — 3 × TILE_N")


def test_both_non_contiguous() raises:
    # Both A and B non-contiguous → neither fast lane applies
    var M = 6
    var K = 5
    var N = 8
    var pad = 2
    var A_wide = Tensor[dtype].randn(M, K + pad)
    var A_strided = A_wide.buffer.share(
        Shape(M, K), A_wide.buffer.strides, A_wide.buffer.offset
    )
    var B_wide = Tensor[dtype].randn(K, N + pad)
    var B_strided = B_wide.buffer.share(
        Shape(K, N), B_wide.buffer.strides, B_wide.buffer.offset
    )
    var A_ref = Tensor[dtype].zeros(M, K)
    for i in range(M):
        for j in range(K):
            A_ref[i, j] = A_wide[i, j]
    var B_ref = Tensor[dtype].zeros(K, N)
    for i in range(K):
        for j in range(N):
            B_ref[i, j] = B_wide[i, j]
    var r = rref(A_ref.buffer, B_ref.buffer)
    var t = trial(A_strided, B_strided)
    assert_true(
        r.all_close[atol=1e-4](t),
        "both non-contiguous mismatch",
    )
    print("  PASS: both A and B non-contiguous")


def test_single_k_tile_large_p() raises:
    # Single k_tile (n < TILE_N), large p → exercises j_tiling
    check(16, 32, 256, "(16,32) @ (32,256) — single k_tile, many j_tiles")


def test_matches_tensor_matmul() raises:
    # Compare against Tensor.matmul() for end-to-end consistency
    var M = 23
    var K = 17
    var N = 31
    var A = Tensor[dtype].randn(M, K)
    var B = Tensor[dtype].randn(K, N)
    var t = trial(A.buffer, B.buffer)
    var r = rref(A.buffer, B.buffer)
    assert_true(r.all_close[atol=1e-4](t), "ndbuffer reference mismatch")
    print("  PASS: matches NDBuffer reference")


def main() raises:
    print("─── tiny ───")
    test_tiny()
    print("─── small ───")
    test_small()
    print("─── medium ───")
    test_medium()
    print("─── tile boundaries ───")
    test_tile_boundaries()
    print("─── SIMD tails ───")
    test_simd_tails()
    print("─── non-contiguous B ───")
    test_non_contiguous_B()
    print("─── non-contiguous A (a_also_contiguous=False) ───")
    test_non_contiguous_A()
    print("─── both non-contiguous ───")
    test_both_non_contiguous()
    print("─── large n (multiple k_tiles, prefetch) ───")
    test_large_n()
    print("─── square large ───")
    test_square_large()
    print("─── tall-skinny ───")
    test_tall_skinny()
    print("─── wide-short ───")
    test_wide_short()
    print("─── k_tile exact multiples ───")
    test_k_tile_exact_multiple()
    print("─── single k_tile + many j_tiles ───")
    test_single_k_tile_large_p()
    print("─── all-ones ───")
    test_all_ones()
    print("─── identity ───")
    test_identity()
    print("─── matmul reference ───")
    test_matches_tensor_matmul()
    print("\nAll matmul_2d_cpu tests passed!")
