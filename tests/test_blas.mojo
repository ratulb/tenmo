from tenmo import Tensor
from shapes import Shape
from time import perf_counter_ns
from blashandle import BLASHandle
from common_utils import now
from testing import assert_true


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
    #test_case_4_comprehensive()
    test_case_2()
    test_blas_case_1_A_B()
    test_blas_case_2_Atranspose_B()
    test_blas_case_3_A_Btranspose()
    test_blas_case_4_Atranspose_Btranspose()
    test_blas_matmul_edge_cases()
    test_blas_gradient_accuracy()


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
    var blas_handle = blas.lite_handle()

    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var C = blas_handle.matmul(A, B)
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
    _ = blas^

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
    _ = blas^

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

    _ = blas^

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

    _ = blas^

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

    _ = blas^

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




# ============================================================================
# COMPREHENSIVE BLAS MATMUL TESTS
# ============================================================================

fn test_blas_case_4_Atranspose_Btranspose() raises:
    """Test Case 4: C = A^T @ B^T."""
    print("\n" + "=" * 80)
    print("TEST CASE 4: C = A^T @ B^T")
    print("=" * 80)

    alias dtype = DType.float32

    # Dimensions: A(3,2)^T @ B(4,3)^T → A^T(2,3) @ B^T(3,4) → C(2,4)
    var A = Tensor[dtype].d2(
        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]],
        requires_grad=True
    )  # 3x2

    var B = Tensor[dtype].d2(
        [[7.0, 8.0, 9.0],
         [10.0, 11.0, 12.0],
         [13.0, 14.0, 15.0],
         [16.0, 17.0, 18.0]],
        requires_grad=True
    )  # 4x3

    # === NATIVE MOJO ===
    print("\n1. NATIVE MOJO:")
    var A_T = A.transpose()  # 2x3
    var B_T = B.transpose()  # 3x4
    var C_native = A_T.matmul(B_T)  # (2x3) @ (3x4) = 2x4
    var loss_native = C_native.sum()
    loss_native.backward()

    print("Forward result shape:", C_native.shape())
    print("\ngrad_A native:")
    A.grad().print()
    print("\ngrad_B native:")
    B.grad().print()

    var native_A_grad = A.grad().copy()
    var native_B_grad = B.grad().copy()

    # Reset gradients
    A.zero_grad()
    B.zero_grad()

    # === BLAS ===
    print("\n2. BLAS:")
    var blas = BLASHandle[dtype]()
    var C_blas = blas.matmul(A, B, transpose_A=True, transpose_B=True)
    var loss_blas = C_blas.sum()
    loss_blas.backward()

    print("Forward result shape:", C_blas.shape())
    print("\ngrad_A BLAS:")
    A.grad().print()
    print("\ngrad_B BLAS:")
    B.grad().print()

    # === VALIDATION ===
    print("\n3. VALIDATION:")
    assert_true(C_native.all_close(C_blas), "Forward results differ!")

    var blas_A_grad = A.grad().copy()
    var blas_B_grad = B.grad().copy()

    assert_true(native_A_grad.all_close(blas_A_grad), "grad_A differs!")
    assert_true(native_B_grad.all_close(blas_B_grad), "grad_B differs!")

    print("✓ All checks passed!")

    _ = blas^

fn test_case_2() raises:
    """Test Case 2: C = A^T @ B (The problematic one)."""
    print("\n" + "=" * 80)
    print("TEST CASE 2: C = A^T @ B")
    print("=" * 80)

    alias dtype = DType.float32

    # Same matrices as before
    var A = Tensor[dtype].d2(
        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]],
        requires_grad=True
    )

    var B = Tensor[dtype].d2(
        [[7.0, 8.0, 9.0],
         [10.0, 11.0, 12.0],
         [13.0, 14.0, 15.0]],
        requires_grad=True
    )

    # BLAS forward
    var blas = BLASHandle[dtype]()
    var C = blas.matmul(A, B, transpose_A=True, transpose_B=False)
    var loss = C.sum()
    loss.backward()

    print("BLAS Results:")
    print("grad_A:")
    A.grad().print()
    print("\ngrad_B:")
    B.grad().print()

    # Expected from PyTorch
    var expected_grad_A = Tensor[dtype].d2(
        [[24.0, 24.0],
         [33.0, 33.0],
         [42.0, 42.0]]
    )

    var expected_grad_B = Tensor[dtype].d2(
        [[3.0, 3.0, 3.0],
         [7.0, 7.0, 7.0],
         [11.0, 11.0, 11.0]]
    )

    print("\nValidation:")
    if A.grad().all_close(expected_grad_A):
        print("✓ grad_A matches PyTorch!")
    else:
        print("✗ grad_A DOES NOT match PyTorch!")
        print("Expected:")
        expected_grad_A.print()

    if B.grad().all_close(expected_grad_B):
        print("✓ grad_B matches PyTorch!")
    else:
        print("✗ grad_B DOES NOT match PyTorch!")
        print("Expected:")
        expected_grad_B.print()

    _ = blas^

fn test_blas_case_1_A_B() raises:
    """Test Case 1: C = A @ B (no transposes)."""
    print("\n" + "=" * 80)
    print("TEST CASE 1: C = A @ B")
    print("=" * 80)

    alias dtype = DType.float32

    # Create tensors: A(3,2), B(2,4) → C(3,4)
    var A = Tensor[dtype].d2(
        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]],
        requires_grad=True
    )

    var B = Tensor[dtype].d2(
        [[7.0, 8.0, 9.0, 10.0],
         [11.0, 12.0, 13.0, 14.0]],
        requires_grad=True
    )

    # === NATIVE MOJO ===
    print("\n1. NATIVE MOJO:")
    var C_native = A.matmul(B)
    var loss_native = C_native.sum()
    loss_native.backward()

    print("Forward result shape:", C_native.shape())
    print("grad_A native:")
    print("grad_B native:")

    var native_A_grad = A.grad().copy()
    var native_B_grad = B.grad().copy()


    # Reset gradients
    A.zero_grad()
    B.zero_grad()

    # === BLAS ===
    print("\n2. BLAS:")
    var blas = BLASHandle[dtype]()
    if not blas.is_initialized():
        print("ERROR: BLAS not initialized")
        return

    var C_blas = blas.matmul(A, B, transpose_A=False, transpose_B=False)
    var loss_blas = C_blas.sum()
    loss_blas.backward()


    # === VALIDATION ===
    print("\n3. VALIDATION:")
    # Forward check
    assert_true(C_native.all_close(C_blas), "Forward results differ!")

    # Backward check
    var blas_A_grad = A.grad().copy()
    var blas_B_grad = B.grad().copy()

    assert_true(native_A_grad.all_close(blas_A_grad), "grad_A differs!")
    assert_true(native_B_grad.all_close(blas_B_grad), "grad_B differs!")
    _ = blas^
    print("✓ All checks passed!")


fn test_blas_case_2_Atranspose_B() raises:
    """Test Case 2: C = A^T @ B."""
    print("\n" + "=" * 80)
    print("TEST CASE 2: C = A^T @ B")
    print("=" * 80)

    alias dtype = DType.float32

    A = Tensor[dtype].d2(
        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]],
        requires_grad=True
    )  # 3x2 (m=3, k=2)

    B = Tensor[dtype].d2(
        [[7.0, 8.0, 9.0],
         [10.0, 11.0, 12.0],
         [13.0, 14.0, 15.0]],
        requires_grad=True
    )  # 3x3 (m=3, n=3)

    print("\nFinal dimensions:")
    print("A shape:", A.shape(), "(3x2) → A^T is 2x3")
    print("B shape:", B.shape(), "(3x3)")
    print("A^T @ B → (2x3) @ (3x3) = 2x3")

    # === NATIVE MOJO ===
    var A_T = A.transpose()  # 2x3
    var C_native = A_T.matmul(B)  # (2x3) @ (3x3) = 2x3
    var loss_native = C_native.sum()
    loss_native.backward()

    print("\nNative forward result (2x3):")
    C_native.print()
    print("\ngrad_A native:")
    A.grad().print()
    print("\ngrad_B native:")
    B.grad().print()

    var native_A_grad = A.grad().copy()
    var native_B_grad = B.grad().copy()

    # Reset gradients
    A.zero_grad()
    B.zero_grad()

    # === BLAS ===
    print("\n2. BLAS:")
    var blas = BLASHandle[dtype]()
    var C_blas = blas.matmul(A, B, transpose_A=True, transpose_B=False)
    var loss_blas = C_blas.sum()
    loss_blas.backward()

    print("BLAS forward result:")
    C_blas.print()
    print("\ngrad_A BLAS:")
    A.grad().print()
    print("\ngrad_B BLAS:")
    B.grad().print()

    # === VALIDATION ===
    print("\n3. VALIDATION:")
    assert_true(C_native.all_close(C_blas), "Forward results differ!")

    var blas_A_grad = A.grad().copy()
    var blas_B_grad = B.grad().copy()

    assert_true(native_A_grad.all_close(blas_A_grad), "grad_A differs!")
    assert_true(native_B_grad.all_close(blas_B_grad), "grad_B differs!")

    _ = blas^
    print("✓ All checks passed!")


fn test_blas_case_3_A_Btranspose() raises:
    """Test Case 3: C = A @ B^T."""
    print("\n" + "=" * 80)
    print("TEST CASE 3: C = A @ B^T")
    print("=" * 80)

    alias dtype = DType.float32

    # Dimensions: A(3,2) @ B(4,2)^T → A(3,2) @ B^T(2,4) → C(3,4)
    var A = Tensor[dtype].d2(
        [[1.0, 2.0],
         [3.0, 4.0],
         [5.0, 6.0]],
        requires_grad=True
    )  # 3x2

    var B = Tensor[dtype].d2(
        [[7.0, 8.0],
         [9.0, 10.0],
         [11.0, 12.0],
         [13.0, 14.0]],
        requires_grad=True
    )  # 4x2

    # === NATIVE MOJO ===
    print("\n1. NATIVE MOJO:")
    var B_T = B.transpose()  # 2x4
    var C_native = A.matmul(B_T)  # (3x2) @ (2x4) = 3x4
    var loss_native = C_native.sum()
    loss_native.backward()


    var native_A_grad = A.grad().copy()
    var native_B_grad = B.grad().copy()

    # Reset gradients
    A.zero_grad()
    B.zero_grad()

    # === BLAS ===
    print("\n2. BLAS:")
    var blas = BLASHandle[dtype]()
    var C_blas = blas.matmul(A, B, transpose_A=False, transpose_B=True)
    var loss_blas = C_blas.sum()
    loss_blas.backward()


    # === VALIDATION ===
    print("\n3. VALIDATION:")
    assert_true(C_native.all_close(C_blas), "Forward results differ!")

    var blas_A_grad = A.grad().copy()
    var blas_B_grad = B.grad().copy()

    assert_true(native_A_grad.all_close(blas_A_grad), "grad_A differs!")
    assert_true(native_B_grad.all_close(blas_B_grad), "grad_B differs!")
    _ = blas^
    print("✓ All checks passed!")


fn test_case_4_comprehensive() raises:
    """
    Test Case 4: C = A^T @ B^T.
    Compare: Native Mojo matmul, BLAS matmul, and PyTorch.
    """
    alias dtype = DType.float32

    print("=" * 80)
    print("TEST CASE 4: C = A^T @ B^T")
    print("=" * 80)

    # Create test matrices
    # A: (3, 2), B: (2, 3)
    # A^T: (2, 3), B^T: (3, 2)
    # C = A^T @ B^T: (2, 3) @ (3, 2) = (2, 2)

    print("\nInput matrices:")
    print("A (3x2):")
    print("  [[1, 2],")
    print("   [3, 4],")
    print("   [5, 6]]")
    print("\nB (2x3):")
    print("  [[7, 8, 9],")
    print("   [10, 11, 12]]")

    # ========================================
    # 1. NATIVE MOJO MATMUL
    # ========================================
    print("\n" + "=" * 80)
    print("1. NATIVE MOJO MATMUL")
    print("=" * 80)

    var A_native = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var B_native = Tensor[dtype].d2(
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], requires_grad=True
    )

    # Forward: C = A^T @ B^T
    var A_T_native = A_native.transpose()
    var B_T_native = B_native.transpose()
    var C_native = A_T_native.matmul(B_T_native)  # Use native matmul

    print("\nForward result C (2x2):")
    C_native.print()

    # Backward
    var loss_native = C_native.sum()
    loss_native.backward()

    print("\nGradients:")
    print("grad_A (3x2):")
    A_native.grad().print()
    print("\ngrad_B (2x3):")
    B_native.grad().print()

    # ========================================
    # 2. BLAS MATMUL
    # ========================================
    print("\n" + "=" * 80)
    print("2. BLAS MATMUL")
    print("=" * 80)

    var blas = BLASHandle[dtype]()
    if not blas.is_initialized():
        print("ERROR: BLAS not initialized")
        return
    var A_blas = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var B_blas = Tensor[dtype].d2(
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], requires_grad=True
    )

    # Forward: C = A^T @ B^T using BLAS
    var C_blas = blas.matmul(A_blas, B_blas, transpose_A=True, transpose_B=True)

    print("\nForward result C (2x2):")
    C_blas.print()

    # Backward
    var loss_blas = C_blas.sum()
    loss_blas.backward()

    print("\nGradients:")
    print("grad_A (3x2):")
    A_blas.grad().print()
    print("\ngrad_B (2x3):")
    B_blas.grad().print()


# ============================================================================
# DIMENSION EDGE CASE TESTS
# ============================================================================

fn test_blas_matmul_edge_cases() raises:
    """Test edge cases: different sizes, non-square, etc."""
    print("\n" + "=" * 80)
    print("BLAS MATMUL EDGE CASES")
    print("=" * 80)

    alias dtype = DType.float32
    var blas = BLASHandle[dtype]()

    # Test 1: Tall skinny matrices
    print("\n1. Tall skinny matrices (10x3 @ 3x5):")
    var A_tall = Tensor[dtype].randn(Shape([10, 3]), requires_grad=True)
    var B_skinny = Tensor[dtype].randn(Shape([3, 5]), requires_grad=True)

    var C_native = A_tall.matmul(B_skinny)
    var C_blas = blas.matmul(A_tall, B_skinny)
    assert_true(C_native.all_close(C_blas), "Tall skinny failed")
    print("✓ Tall skinny passed")

    # Test 2: Short wide matrices
    print("\n2. Short wide matrices (3x10 @ 10x4):")
    var A_wide = Tensor[dtype].randn(Shape([3, 10]), requires_grad=True)
    var B_wide = Tensor[dtype].randn(Shape([10, 4]), requires_grad=True)

    C_native = A_wide.matmul(B_wide)
    C_blas = blas.matmul(A_wide, B_wide)
    assert_true(C_native.all_close(C_blas), "Short wide failed")
    print("✓ Short wide passed")

    # Test 3: Vector @ Matrix (1x3 @ 3x4 → 1x4)
    print("\n3. Vector @ Matrix:")
    var vec = Tensor[dtype].randn(Shape([1, 3]), requires_grad=True)
    var mat = Tensor[dtype].randn(Shape([3, 4]), requires_grad=True)

    C_native = vec.matmul(mat)
    C_blas = blas.matmul(vec, mat)
    assert_true(C_native.all_close(C_blas), "Vector @ Matrix failed")
    print("✓ Vector @ Matrix passed")

    # Test 4: Matrix @ Vector (3x4 @ 4x1 → 3x1)
    print("\n4. Matrix @ Vector:")
    mat = Tensor[dtype].randn(Shape([3, 4]), requires_grad=True)
    vec = Tensor[dtype].randn(Shape([4, 1]), requires_grad=True)

    C_native = mat.matmul(vec)
    C_blas = blas.matmul(mat, vec)
    assert_true(C_native.all_close(C_blas), "Matrix @ Vector failed")
    print("✓ Matrix @ Vector passed")

    print("\n✓ All edge cases passed!")


# ============================================================================
# GRADIENT ACCURACY TESTS (Compare with finite differences)
# ============================================================================

fn test_blas_gradient_accuracy() raises:
    """Test gradient accuracy using finite differences."""
    print("\n" + "=" * 80)
    print("BLAS GRADIENT ACCURACY TEST (Finite Differences)")
    print("=" * 80)

    alias dtype = DType.float32
    var eps = Scalar[dtype](0.0001)

    # Simple test case: C = A @ B, A(2,2), B(2,2)
    var A = Tensor[dtype].d2(
        [[1.0, 2.0],
         [3.0, 4.0]],
        requires_grad=True
    )

    var B = Tensor[dtype].d2(
        [[5.0, 6.0],
         [7.0, 8.0]],
        requires_grad=True
    )

    var blas = BLASHandle[dtype]()

    # Forward with BLAS
    var C = blas.matmul(A, B)
    var loss = C.sum()
    loss.backward()

    var grad_A_blas = A.grad().copy()

    # Finite difference for grad_A[0,0]
    A.zero_grad()
    B.zero_grad()

    # Perturb A[0,0] and compute finite difference
    var A_plus = A.copy()
    A_plus[0,0] = A_plus[0,0] + eps
    var C_plus = blas.matmul(A_plus, B)
    var loss_plus = C_plus.sum()

    var A_minus = A.copy()
    A_minus[0,0] = A_minus[0,0] - eps
    var C_minus = blas.matmul(A_minus, B)
    var loss_minus = C_minus.sum()

    var grad_A_fd = (loss_plus - loss_minus) / (2 * eps)
    var grad_A_blas_00 = grad_A_blas[0,0]

    print("\nFinite difference check for grad_A[0,0]:")
    print("\nBLAS gradient:\n", grad_A_blas_00)

    print("\nFinite difference:\n")
    grad_A_fd.print()
    print("\nDifference:\n")
    (grad_A_blas_00 - grad_A_fd).print()

    # Should be close
    assert_true(((grad_A_blas_00 - grad_A_fd).__abs__() < 0.014).all_true(),
                "Gradient accuracy test failed!")

    print("\n✓ Gradient accuracy test passed!")

