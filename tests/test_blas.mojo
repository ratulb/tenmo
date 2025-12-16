from tenmo import Tensor
from shapes import Shape
from testing import assert_true
from time import perf_counter_ns
from blashandle import BLASHandle
from common_utils import now

# void cblas_sgemm(
#    int order,          // 1. CblasRowMajor (101)
#    int transA,         // 2. CblasNoTrans (111) or CblasTrans (112)
#    int transB,         // 3. CblasNoTrans (111) or CblasTrans (112)
#    int M,              // 4. Rows of op(A) and C
#    int N,              // 5. Columns of op(B) and C
#    int K,              // 6. Inner dimension
#    float alpha,        // 7. FLOAT - scaling for A*B (CRITICAL!)
#    const float *A,     // 8. Pointer to A
#    int lda,            // 9. Leading dimension of A (AFTER alpha!)
#    const float *B,     // 10. Pointer to B
#    int ldb,            // 11. Leading dimension of B
#    float beta,         // 12. FLOAT - scaling for C (CRITICAL!)
#    float *C,           // 13. Pointer to output C
#    int ldc             // 14. Leading dimension of C
# );


fn benchmark_cifar_sizes() raises:
    print("CIFAR-10 FC Layer Sizes:")
    print("=" * 50)
    alias dtype = DType.float32
    blas = BLASHandle[dtype]()
    # First layer: batch @ weights (128, 3072) @ (3072, 256)
    var start = now()
    var x1 = Tensor[DType.float32].rand(128, 3072, requires_grad=True)
    var w1 = Tensor[DType.float32].rand(3072, 256, requires_grad=True)
    for _ in range(100):
        var _c1 = blas.matmul(x1, w1)  # BLAS version
    print("Layer 1 BLAS (100 iters):", now() - start, "sec")

    start = now()
    for _ in range(100):
        var _d1 = x1.matmul(w1)  # Your version
    print("Layer 1 Normal (100 iters):", now() - start, "sec")
    print()

    # Second layer: (128, 256) @ (256, 128)
    start = now()
    var x2 = Tensor[DType.float32].rand(128, 256, requires_grad=True)
    var w2 = Tensor[DType.float32].rand(256, 128, requires_grad=True)
    for _ in range(100):
        var _c2 = blas.matmul(x2, w2)
    print("Layer 2 BLAS (100 iters):", now() - start, "sec")

    start = now()
    for _ in range(100):
        var _d2 = x2.matmul(w2)
    print("Layer 2 Normal (100 iters):", now() - start, "sec")
    print()

    # Large batch: (512, 3072) @ (3072, 256)
    start = now()
    var x3 = Tensor[DType.float32].rand(512, 3072, requires_grad=True)
    var w3 = Tensor[DType.float32].rand(3072, 256, requires_grad=True)
    for _ in range(100):
        var _c3 = blas.matmul(x3, w3)
    print("Large batch BLAS (100 iters):", now() - start, "sec")

    start = now()
    for _ in range(100):
        var _d3 = x3.matmul(w3)
    print("Large batch Normal (100 iters):", now() - start, "sec")


fn main() raises:
    # benchmark_cifar_sizes()
    run_all_blas_tests()
    pass


# ============================================================================
# Basic Matmul Tests
# ============================================================================


fn test_blas_matmul_simple_f32() raises:
    """Test simple 2x2 matrix multiplication."""
    print("test_blas_matmul_simple_f32")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])

    var C = blas.matmul(A, B)

    # Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    #         = [[19, 22], [43, 50]]
    var expected = Tensor[dtype].d2([[19.0, 22.0], [43.0, 50.0]])
    assert_true(C.all_close[atol=1e-5](expected))


fn test_blas_matmul_simple_f64() raises:
    """Test simple matrix multiplication with float64."""
    print("test_blas_matmul_simple_f64")
    alias dtype = DType.float64
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])

    var C = blas.matmul(A, B)

    var expected = Tensor[dtype].d2([[19.0, 22.0], [43.0, 50.0]])
    assert_true(C.all_close[atol=1e-10](expected))


fn test_blas_matmul_identity() raises:
    """Test multiplication with identity matrix."""
    print("test_blas_matmul_identity")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var I = Tensor[dtype].d2(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    var C = blas.matmul(A, I)

    # A @ I should equal A
    assert_true(C.all_close[atol=1e-5](A))


fn test_blas_matmul_rectangular() raises:
    """Test rectangular matrix multiplication."""
    print("test_blas_matmul_rectangular")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    # (2, 3) @ (3, 4) -> (2, 4)
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    )

    var C = blas.matmul(A, B)

    assert_true(C.shape()[0] == 2)
    assert_true(C.shape()[1] == 4)

    # Verify first element: 1*1 + 2*5 + 3*9 = 1 + 10 + 27 = 38
    assert_true(abs(C[0, 0] - 38.0) < 1e-5)


# ============================================================================
# Transpose Tests
# ============================================================================
fn test_blas_matmul_transpose_a() raises:
    """Test matrix multiplication with transpose_A=True."""

    print("test_blas_matmul_transpose_a")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    # A is (3, 2), A^T is (2, 3)
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var B = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)

    # A^T @ B: (2, 3) @ (3, 2) -> (2, 2)
    var C = blas.matmul(A, B, transpose_A=True)

    assert_true(C.shape()[0] == 2)
    assert_true(C.shape()[1] == 2)

    # Verify: A^T @ B
    # A^T = [[1, 3, 5], [2, 4, 6]]
    # C[0,0] = 1*1 + 3*3 + 5*5 = 1 + 9 + 25 = 35
    assert_true(abs(C[0, 0] - 35.0) < 1e-5)


fn test_blas_matmul_transpose_b() raises:
    """Test matrix multiplication with transpose_B=True."""
    print("test_blas_matmul_transpose_b")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    var B = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)

    # A @ B^T: (2, 3) @ (3, 2) -> (2, 2)
    var C = blas.matmul(A, B, transpose_B=True)

    assert_true(C.shape()[0] == 2)
    assert_true(C.shape()[1] == 2)

    # B^T = [[1, 4], [2, 5], [3, 6]]
    # C[0,0] = 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
    assert_true(abs(C[0, 0] - 14.0) < 1e-5)


fn test_blas_matmul_transpose_both() raises:
    """Test matrix multiplication with both transposes."""
    print("test_blas_matmul_transpose_both")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])  # (2, 2)

    # A^T @ B^T
    var C = blas.matmul(A, B, transpose_A=True, transpose_B=True)

    # A^T = [[1, 3], [2, 4]]
    # B^T = [[5, 7], [6, 8]]
    # C[0,0] = 1*5 + 3*6 = 5 + 18 = 23
    assert_true(abs(C[0, 0] - 23.0) < 1e-5)


# ============================================================================
# Large Matrix Tests
# ============================================================================


fn test_blas_matmul_large_square() raises:
    """Test large square matrix multiplication."""
    print("test_blas_matmul_large_square")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var N = 100
    var A = Tensor[dtype].ones(N, N)
    var B = Tensor[dtype].ones(N, N)

    var C = blas.matmul(A, B)

    # All elements should be N (sum of N ones)
    assert_true(C.shape()[0] == N)
    assert_true(C.shape()[1] == N)
    assert_true(abs(C[0, 0] - Float32(N)) < 1e-4)
    assert_true(abs(C[N - 1, N - 1] - Float32(N)) < 1e-4)


fn test_blas_matmul_large_rectangular() raises:
    """Test large rectangular matrix multiplication."""
    print("test_blas_matmul_large_rectangular")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    # Simulate neural network layer: (batch, in_features) @ (in_features, out_features)
    var batch_size = 128
    var in_features = 784
    var out_features = 256

    var X = Tensor[dtype].rand(batch_size, in_features)
    var W = Tensor[dtype].rand(in_features, out_features)

    var Y = blas.matmul(X, W)

    assert_true(Y.shape()[0] == batch_size)
    assert_true(Y.shape()[1] == out_features)


# ============================================================================
# Gradient Tests
# ============================================================================


fn test_blas_matmul_backward_simple() raises:
    """Test backward pass for simple matmul."""
    print("test_blas_matmul_backward_simple")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var C = blas.matmul(A, B)
    var loss = C.sum()
    loss.backward()

    # dL/dA = grad_output @ B^T
    # grad_output is all ones (2, 2)
    # B^T = [[5, 7], [6, 8]]
    # dL/dA[0,0] = 1*5 + 1*6 = 11
    var expected_grad_A = Tensor[dtype].d2([[11.0, 15.0], [11.0, 15.0]])
    assert_true(A.grad().all_close[atol=1e-4](expected_grad_A))

    # dL/dB = A^T @ grad_output
    # A^T = [[1, 3], [2, 4]]
    # dL/dB[0,0] = 1*1 + 3*1 = 4
    var expected_grad_B = Tensor[dtype].d2([[4.0, 4.0], [6.0, 6.0]])
    assert_true(B.grad().all_close[atol=1e-4](expected_grad_B))


fn test_blas_matmul_backward_single_requires_grad() raises:
    """Test backward when only one tensor requires grad."""
    print("test_blas_matmul_backward_single_requires_grad")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=False)

    var C = blas.matmul(A, B)
    assert_true(C.requires_grad)

    var loss = C.sum()
    loss.backward()

    # A should have gradients
    assert_true(A.grad().shape()[0] == 2)
    assert_true(A.grad().shape()[1] == 2)


fn test_blas_matmul_backward_chain() raises:
    """Test backward through chain of matmuls."""
    print("test_blas_matmul_backward_chain")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    var C = Tensor[dtype].d2([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)

    # Chain: A @ B @ C
    var AB = blas.matmul(A, B)
    var ABC = blas.matmul(AB, C)

    var loss = ABC.sum()
    loss.backward()

    # All should have gradients
    assert_true(A.grad().shape()[0] == 2)
    assert_true(B.grad().shape()[0] == 2)
    assert_true(C.grad().shape()[0] == 2)


# ============================================================================
# Correctness vs Native Implementation
# ============================================================================


fn test_blas_vs_native_random() raises:
    """Compare BLAS result with native matmul."""
    print("test_blas_vs_native_random")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].rand(50, 60)
    var B = Tensor[dtype].rand(60, 70)

    var C_blas = blas.matmul(A, B)
    var C_native = A.matmul(B)

    # Should produce same results (within floating point tolerance)
    assert_true(C_blas.all_close[atol=1e-4](C_native))


fn test_blas_vs_native_transpose_a() raises:
    """Compare BLAS transpose_A with native implementation."""
    print("test_blas_vs_native_transpose_a")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].rand(60, 50)  # Will be transposed to (50, 60)
    var B = Tensor[dtype].rand(60, 70)

    var C_blas = blas.matmul(A, B, transpose_A=True)
    A_t = A.transpose()
    var C_native = A_t.matmul(B)

    assert_true(C_blas.all_close[atol=1e-4](C_native))


fn test_blas_vs_native_transpose_b() raises:
    """Compare BLAS transpose_B with native implementation."""
    print("test_blas_vs_native_transpose_b")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].rand(50, 60)
    var B = Tensor[dtype].rand(70, 60)  # Will be transposed to (60, 70)

    var C_blas = blas.matmul(A, B, transpose_B=True)
    B_t = B.transpose()
    var C_native = A.matmul(B_t)

    assert_true(C_blas.all_close[atol=1e-4](C_native))


# ============================================================================
# Performance Tests
# ============================================================================


fn test_blas_performance_small() raises:
    """Benchmark small matrices."""
    print("test_blas_performance_small")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].rand(10, 10)
    var B = Tensor[dtype].rand(10, 10)

    var start = perf_counter_ns()
    for _ in range(1000):
        var _C = blas.matmul(A, B)
    var time_blas = (perf_counter_ns() - start) / 1e9

    start = perf_counter_ns()
    for _ in range(1000):
        var _C = A.matmul(B)
    var time_native = (perf_counter_ns() - start) / 1e9

    print("  BLAS (1000 iters):", time_blas, "sec")
    print("  Native (1000 iters):", time_native, "sec")
    print("  Speedup:", time_native / time_blas, "x")


fn test_blas_performance_medium() raises:
    """Benchmark medium matrices (typical NN layer)."""
    print("test_blas_performance_medium")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].rand(128, 784)  # Batch of 128, 784 features
    var B = Tensor[dtype].rand(784, 256)  # FC layer weights

    var start = perf_counter_ns()
    for _ in range(100):
        var _C = blas.matmul(A, B)
    var time_blas = (perf_counter_ns() - start) / 1e9

    start = perf_counter_ns()
    for _ in range(100):
        var _C = A.matmul(B)
    var time_native = (perf_counter_ns() - start) / 1e9

    print("  BLAS (100 iters):", time_blas, "sec")
    print("  Native (100 iters):", time_native, "sec")
    print("  Speedup:", time_native / time_blas, "x")


fn test_blas_performance_large() raises:
    """Benchmark large matrices."""
    print("test_blas_performance_large")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].rand(512, 1024)
    var B = Tensor[dtype].rand(1024, 512)

    var start = perf_counter_ns()
    for _ in range(10):
        var _C = blas.matmul(A, B)
    var time_blas = (perf_counter_ns() - start) / 1e9

    start = perf_counter_ns()
    for _ in range(10):
        var _C = A.matmul(B)
    var time_native = (perf_counter_ns() - start) / 1e9

    print("  BLAS (10 iters):", time_blas, "sec")
    print("  Native (10 iters):", time_native, "sec")
    print("  Speedup:", time_native / time_blas, "x")


# ============================================================================
# Edge Cases
# ============================================================================


fn test_blas_matmul_vector_like() raises:
    """Test with very thin matrices (vector-like)."""
    print("test_blas_matmul_vector_like")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    # (100, 1) @ (1, 100) -> (100, 100) outer product
    var A = Tensor[dtype].ones(100, 1)
    var B = Tensor[dtype].ones(1, 100)

    var C = blas.matmul(A, B)

    assert_true(C.shape()[0] == 100)
    assert_true(C.shape()[1] == 100)
    # All elements should be 1
    assert_true(abs(C[0, 0] - 1.0) < 1e-5)
    assert_true(abs(C[99, 99] - 1.0) < 1e-5)


fn test_blas_matmul_non_contiguous() raises:
    """Test with non-contiguous input (should be made contiguous)."""
    print("test_blas_matmul_non_contiguous")
    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    var A = Tensor[dtype].rand(10, 10)
    var A_T = A.transpose()  # Non-contiguous
    var B = Tensor[dtype].rand(10, 10)

    assert_true(not A_T.is_contiguous())

    # Should still work (internally makes contiguous)
    var C = blas.matmul(A_T, B, transpose_A=True)

    assert_true(C.shape()[0] == 10)
    assert_true(C.shape()[1] == 10)


fn run_all_blas_tests() raises:
    # Basic tests
    print("--- Basic Matmul Tests ---")
    test_blas_matmul_simple_f32()
    test_blas_matmul_simple_f64()
    test_blas_matmul_identity()
    test_blas_matmul_rectangular()
    print()

    # Transpose tests
    print("--- Transpose Tests ---")
    test_blas_matmul_transpose_a()
    test_blas_matmul_transpose_b()
    test_blas_matmul_transpose_both()
    print()

    # Large matrix tests
    print("--- Large Matrix Tests ---")
    test_blas_matmul_large_square()
    test_blas_matmul_large_rectangular()
    print()

    # Gradient tests
    print("--- Gradient Tests ---")
    test_blas_matmul_backward_simple()
    test_blas_matmul_backward_single_requires_grad()
    test_blas_matmul_backward_chain()
    print()

    # Correctness tests
    print("--- Correctness vs Native ---")
    test_blas_vs_native_random()
    test_blas_vs_native_transpose_a()
    test_blas_vs_native_transpose_b()
    print()

    # Performance tests
    print("--- Performance Benchmarks ---")
    test_blas_performance_small()
    test_blas_performance_medium()
    test_blas_performance_large()
    print()

    # Edge cases
    print("--- Edge Cases ---")
    test_blas_matmul_vector_like()
    test_blas_matmul_non_contiguous()
    print()

    print("=== All BLAS Tests Passed! ===\n")
