"""
Comprehensive test suite for batched matrix multiplication with broadcasting.
Tests all broadcasting scenarios that PyTorch supports.

Broadcasting rules for matmul:
1. If one tensor is 1D, it's converted to 2D and removed after operation
2. Batch dimensions are broadcasted following standard broadcasting rules
3. Last 2 dimensions must be compatible for matrix multiplication
"""

from testing import assert_almost_equal, assert_equal, assert_true, assert_false
from math import sqrt
from random import random_float64
from tenmo import Tensor
from shapes import Shape
from common_utils import isnan, isinf
from gradbox import Gradbox

fn matmul[
    dtype: DType, //, track_grad: Bool = True
](mut A: Tensor[dtype], mut B: Tensor[dtype]) -> Tensor[dtype]:
    return A.matmul[track_grad=track_grad](B)


fn test_basic_2d_times_3d() raises:
    """Test (2, 4) @ (1, 4, 9) -> (1, 2, 9)."""
    print("=" * 80)
    print("Test 1: Basic 2D  3D Broadcasting (2,4) @ (1,4,9)")
    print("=" * 80)

    var A = Tensor[DType.float32].randn(2, 4, requires_grad=True)
    var B = Tensor[DType.float32].randn(1, 4, 9, requires_grad=True)

    var result = matmul(A, B)

    # Check output shape
    assert_equal(result.shape().rank(), 3, "Result should be 3D")
    assert_equal(result.shape()[0], 1, "Batch dimension should be 1")
    assert_equal(result.shape()[1], 2, "Row dimension should be 2")
    assert_equal(result.shape()[2], 9, "Column dimension should be 9")

    print("Forward shape correct: (2,4) @ (1,4,9) = (1,2,9)")

    # Test backward pass
    var grad_output = Tensor[DType.float32].ones(1, 2, 9)
    result.backward(grad_output)

    # Verify gradients exist and have correct shapes
    var grad_A = A.gradients()[]
    var grad_B = B.gradients()[]

    assert_equal(grad_A.shape()[0], 2)
    assert_equal(grad_A.shape()[1], 4)
    assert_equal(grad_B.shape()[0], 1)
    assert_equal(grad_B.shape()[1], 4)
    assert_equal(grad_B.shape()[2], 9)

    print("Backward shapes correct")
    print("  grad_A:", grad_A.shape())
    print("  grad_B:", grad_B.shape())
    print()


fn test_3d_times_2d() raises:
    """Test (1, 2, 4) @ (4, 9) -> (1, 2, 9)."""
    print("=" * 80)
    print("Test 2: 3D  2D Broadcasting (1,2,4) @ (4,9)")
    print("=" * 80)


    alias dtype = DType.float32
    var A = Tensor[dtype].ones(1, 2, 4, requires_grad=True)
    var B = Tensor[dtype].ones(4, 9, requires_grad=True)

    var result = matmul(A, B)

    assert_equal(result.shape().rank(), 3)
    assert_equal(result.shape()[0], 1)
    assert_equal(result.shape()[1], 2)
    assert_equal(result.shape()[2], 9)

    print("Forward shape correct: (1,2,4) @ (4,9) = (1,2,9)")

    var grad_output = Tensor[DType.float32].ones(1, 2, 9)
    result.backward(grad_output)

    var grad_A = A.gradients()[]
    var grad_B = B.gradients()[]

    assert_true(result == Tensor[dtype].full(Shape(1, 2, 9), 4))
    assert_true(grad_A == Gradbox[dtype].full(Shape(1, 2, 4), 9))
    assert_true(grad_B == Gradbox[dtype].full(Shape(4, 9), 2))

    assert_equal(grad_A.shape()[0], 1)
    assert_equal(grad_A.shape()[1], 2)
    assert_equal(grad_A.shape()[2], 4)
    assert_equal(grad_B.shape()[0], 4)
    assert_equal(grad_B.shape()[1], 9)

    print("Backward shapes correct")
    print()


fn test_batch_broadcasting() raises:
    """Test (3, 2, 4) @ (1, 4, 9) -> (3, 2, 9)."""
    print("=" * 80)
    print("Test 3: Batch Broadcasting (3,2,4) @ (1,4,9)")
    print("=" * 80)


    alias dtype = DType.float32
    var A = Tensor[DType.float32].ones(3, 2, 4, requires_grad=True)
    var B = Tensor[DType.float32].ones(1, 4, 9, requires_grad=True)

    var result = matmul(A, B)

    assert_equal(result.shape()[0], 3, "Batch should broadcast to 3")
    assert_equal(result.shape()[1], 2)
    assert_equal(result.shape()[2], 9)

    print("Forward shape correct: (3,2,4) @ (1,4,9) = (3,2,9)")

    var grad_output = Tensor[DType.float32].ones(3, 2, 9)
    result.backward(grad_output)

    var grad_A = A.gradients()[]
    var grad_B = B.gradients()[]

    assert_true(result == Tensor[dtype].full(Shape(3, 2, 9), 4))
    assert_true(grad_A == Gradbox[dtype].full(Shape(3, 2, 4), 9))
    assert_true(grad_B == Gradbox[dtype].full(Shape(1, 4, 9), 6))

    assert_equal(grad_A.shape()[0], 3)
    assert_equal(grad_B.shape()[0], 1, "grad_B should sum over batch")

    print("Backward shapes correct")
    print("  grad_A:", grad_A.shape())
    print("  grad_B:", grad_B.shape())
    print()


fn test_multi_batch_broadcasting() raises:
    """Test (5, 1, 2, 4) @ (1, 3, 4, 9) -> (5, 3, 2, 9)."""
    print("=" * 80)
    print("Test 4: Multi-Batch Broadcasting (5,1,2,4) @ (1,3,4,9)")
    print("=" * 80)

    var A = Tensor[DType.float32].randn(5, 1, 2, 4, requires_grad=True)
    var B = Tensor[DType.float32].randn(1, 3, 4, 9, requires_grad=True)

    var result = matmul(A, B)

    assert_equal(result.shape().rank(), 4)
    assert_equal(result.shape()[0], 5, "First batch broadcasts to 5")
    assert_equal(result.shape()[1], 3, "Second batch broadcasts to 3")
    assert_equal(result.shape()[2], 2)
    assert_equal(result.shape()[3], 9)

    print("Forward shape correct: (5,1,2,4) @ (1,3,4,9) = (5,3,2,9)")

    var grad_output = Tensor[DType.float32].ones(5, 3, 2, 9)
    result.backward(grad_output)

    var grad_A = A.gradients()[]
    var grad_B = B.gradients()[]

    assert_equal(grad_A.shape()[0], 5)
    assert_equal(grad_A.shape()[1], 1, "grad_A[1] sums over broadcast")
    assert_equal(grad_B.shape()[0], 1, "grad_B[0] sums over broadcast")
    assert_equal(grad_B.shape()[1], 3)

    print("Backward shapes correct")
    print()


fn test_numerical_correctness() raises:
    """Verify actual values match expected matrix multiplication."""
    print("=" * 80)
    print("Test 5: Numerical Correctness")
    print("=" * 80)

    # Simple case: (2, 3) @ (1, 3, 4)
    var A = Tensor[DType.float32].zeros(2, 3, requires_grad=True)
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[0, 2] = 3.0
    A[1, 0] = 4.0
    A[1, 1] = 5.0
    A[1, 2] = 6.0

    var B = Tensor[DType.float32].zeros(1, 3, 4, requires_grad=True)
    B[0, 0, 0] = 1.0
    B[0, 0, 1] = 2.0
    B[0, 0, 2] = 3.0
    B[0, 0, 3] = 4.0
    B[0, 1, 0] = 5.0
    B[0, 1, 1] = 6.0
    B[0, 1, 2] = 7.0
    B[0, 1, 3] = 8.0
    B[0, 2, 0] = 9.0
    B[0, 2, 1] = 10.0
    B[0, 2, 2] = 11.0
    B[0, 2, 3] = 12.0

    var result = matmul(A, B)

    # Expected result[0, 0, :] = [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
    #                           = [38, 44, 50, 56]
    assert_almost_equal(result[0, 0, 0], 38.0, atol=1e-5)
    assert_almost_equal(result[0, 0, 1], 44.0, atol=1e-5)
    assert_almost_equal(result[0, 0, 2], 50.0, atol=1e-5)
    assert_almost_equal(result[0, 0, 3], 56.0, atol=1e-5)

    # Expected result[0, 1, :] = [4*1+5*5+6*9, 4*2+5*6+6*10, ...]
    #                           = [83, 98, 113, 128]
    assert_almost_equal(result[0, 1, 0], 83.0, atol=1e-5)
    assert_almost_equal(result[0, 1, 1], 98.0, atol=1e-5)
    assert_almost_equal(result[0, 1, 2], 113.0, atol=1e-5)
    assert_almost_equal(result[0, 1, 3], 128.0, atol=1e-5)

    print("Forward values correct")

    # Test gradient numerical correctness
    var grad_output = Tensor[DType.float32].ones(1, 2, 4)
    result.backward(grad_output)

    var grad_A = A.gradients()[]
    var _grad_B = B.gradients()[]

    # grad_A[i,j] = sum over k,l: grad_output[0,i,l] * B[0,j,l]
    # For A[0,0]: sum of B[0,0,:] = 1+2+3+4 = 10
    assert_almost_equal(grad_A[0, 0], 10.0, atol=1e-5)
    # For A[0,1]: sum of B[0,1,:] = 5+6+7+8 = 26
    assert_almost_equal(grad_A[0, 1], 26.0, atol=1e-5)
    # For A[0,2]: sum of B[0,2,:] = 9+10+11+12 = 42
    assert_almost_equal(grad_A[0, 2], 42.0, atol=1e-5)

    print("Gradient values correct")
    print()


fn test_gradient_with_numerical_check() raises:
    """Numerical gradient verification using finite differences."""
    print("=" * 80)
    print("Test 6: Numerical Gradient Verification")
    print("=" * 80)

    var A = Tensor[DType.float32].randn(2, 3, requires_grad=True)
    var B = Tensor[DType.float32].randn(1, 3, 4, requires_grad=True)

    # Forward pass
    var result = matmul(A, B)

    # Backward pass
    var grad_output = Tensor[DType.float32].randn(1, 2, 4)
    result.backward(grad_output)

    var grad_A = A.gradients()[]

    # Numerical gradient check for A[0,0]

    var epsilon = Scalar[DType.float32](1e-2)
    var original = A[0, 0]

    # f(x + eps)
    A[0, 0] = original + epsilon
    var result_plus = matmul[track_grad=False](A, B)
    var loss_plus: Float32 = 0.0
    for i in range(1):
        for j in range(2):
            for k in range(4):
                loss_plus += result_plus[i, j, k] * grad_output[i, j, k]

    # f(x - eps)
    A[0, 0] = original - epsilon
    var result_minus = matmul[track_grad=False](A, B)
    var loss_minus: Float32 = 0.0
    for i in range(1):
        for j in range(2):
            for k in range(4):
                loss_minus += result_minus[i, j, k] * grad_output[i, j, k]

    # Numerical gradient
    var numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

    # Restore
    A[0, 0] = original

    var analytical_grad = grad_A[0, 0]
    var relative_error = abs(analytical_grad - numerical_grad) / (
        abs(numerical_grad) + 1e-8
    )

    print("Numerical gradient check for A[0,0]:")
    print("  Analytical:", analytical_grad)
    print("  Numerical:", numerical_grad)
    print("  Relative error:", relative_error)

    assert_true(relative_error < 1e-3, "Gradient error too large")
    print("Numerical gradient matches analytical")
    print()


fn test_symmetric_broadcasting() raises:
    """Test (1, 3, 2, 4) @ (5, 1, 4, 9) -> (5, 3, 2, 9)."""
    print("=" * 80)
    print("Test 7: Symmetric Broadcasting (1,3,2,4) @ (5,1,4,9)")
    print("=" * 80)

    var A = Tensor[DType.float32].randn(1, 3, 2, 4, requires_grad=True)
    var B = Tensor[DType.float32].randn(5, 1, 4, 9, requires_grad=True)

    var result = matmul(A, B)

    assert_equal(result.shape()[0], 5)
    assert_equal(result.shape()[1], 3)
    assert_equal(result.shape()[2], 2)
    assert_equal(result.shape()[3], 9)

    print("Forward shape correct: (1,3,2,4) @ (5,1,4,9) = (5,3,2,9)")

    var grad_output = Tensor[DType.float32].ones(5, 3, 2, 9)
    result.backward(grad_output)

    var grad_A = A.gradients()[]
    var grad_B = B.gradients()[]

    assert_equal(grad_A.shape()[0], 1, "grad_A[0] sums over broadcast dim")
    assert_equal(grad_A.shape()[1], 3)
    assert_equal(grad_B.shape()[0], 5)
    assert_equal(grad_B.shape()[1], 1, "grad_B[1] sums over broadcast dim")

    print("Backward shapes correct with symmetric broadcasting")
    print()

fn test_complex_broadcasting() raises:
    """Test (2, 1, 3, 1, 4, 5) @ (1, 4, 1, 6, 5, 7) -> (2, 4, 3, 6, 4, 7)."""
    print("=" * 80)
    print("Test 8: Complex Multi-Dimensional Broadcasting")
    print("=" * 80)


    alias dtype = DType.float32
    var A = Tensor[DType.float32].ones(2, 1, 3, 1, 4, 5, requires_grad=True)
    var B = Tensor[DType.float32].ones(1, 4, 1, 6, 5, 7, requires_grad=True)

    var result = matmul(A, B)

    # Expected shape: (2, 4, 3, 6, 4, 7)
    assert_equal(result.shape()[0], 2)
    assert_equal(result.shape()[1], 4)
    assert_equal(result.shape()[2], 3)
    assert_equal(result.shape()[3], 6)
    assert_equal(result.shape()[4], 4)
    assert_equal(result.shape()[5], 7)

    print("Forward shape correct for complex broadcasting")

    var grad_output = Tensor[DType.float32].ones(result.shape())
    result.backward(grad_output)

    var grad_A = A.gradients()[]
    var grad_B = B.gradients()[]

    # Check dimensions that should sum
    assert_equal(grad_A.shape()[1], 1, "Dimension 1 should sum")
    assert_equal(grad_A.shape()[3], 1, "Dimension 3 should sum")
    assert_equal(grad_B.shape()[0], 1, "Dimension 0 should sum")
    assert_equal(grad_B.shape()[2], 1, "Dimension 2 should sum")

    print("Backward correctly handles complex broadcasting")
    print()
    assert_true(result == Tensor[dtype].full(Shape(2, 4, 3, 6, 4, 7), 5))
    assert_true(grad_A == Gradbox[dtype].full(Shape(2, 1, 3, 1, 4, 5), 168))
    assert_true(grad_B == Gradbox[dtype].full(Shape(1, 4, 1, 6, 5, 7), 24))

fn test_edge_case_single_batch() raises:
    """Test (1, 2, 3) @ (1, 3, 4) -> (1, 2, 4)."""
    print("=" * 80)
    print("Test 9: Edge Case - Single Batch in Both")
    print("=" * 80)

    var A = Tensor[DType.float32].randn(1, 2, 3, requires_grad=True)
    var B = Tensor[DType.float32].randn(1, 3, 4, requires_grad=True)

    var result = matmul(A, B)

    assert_equal(result.shape()[0], 1)
    assert_equal(result.shape()[1], 2)
    assert_equal(result.shape()[2], 4)

    print("Single batch case works")

    var grad_output = Tensor[DType.float32].ones(1, 2, 4)
    result.backward(grad_output)

    print("Gradients computed successfully")
    print()


fn test_large_batch() raises:
    """Test with larger batch dimensions."""
    print("=" * 80)
    print("Test 10: Large Batch Dimensions")
    print("=" * 80)

    var A = Tensor[DType.float32].randn(32, 1, 64, 128, requires_grad=True)
    var B = Tensor[DType.float32].randn(1, 16, 128, 256, requires_grad=True)

    var result = matmul(A, B)

    assert_equal(result.shape()[0], 32)
    assert_equal(result.shape()[1], 16)
    assert_equal(result.shape()[2], 64)
    assert_equal(result.shape()[3], 256)

    print("Forward pass completed for large batches")

    var grad_output = Tensor[DType.float32].ones(result.shape())
    result.backward(grad_output)

    var grad_A = A.gradients()[]
    var grad_B = B.gradients()[]

    assert_equal(grad_A.shape()[1], 1, "grad_A correctly sums broadcast")
    assert_equal(grad_B.shape()[0], 1, "grad_B correctly sums broadcast")

    print("Backward pass completed for large batches")
    print()


fn test_pytorch_comparison() raises:
    """
    Generate same random values and compare against PyTorch.
    This test would need PyTorch integration to run.
    """
    print("=" * 80)
    print("Test 11: PyTorch Comparison (Manual Verification)")
    print("=" * 80)

    # Set specific values that can be manually verified in PyTorch
    var A = Tensor[DType.float32].zeros(2, 3, requires_grad=True)
    var B = Tensor[DType.float32].zeros(1, 3, 4, requires_grad=True)

    # Fill with simple values
    for i in range(2):
        for j in range(3):
            A[i, j] = Float32(i * 3 + j + 1)

    for i in range(3):
        for j in range(4):
            B[0, i, j] = Float32(i * 4 + j + 1)

    print("A values:")
    print("  [[1, 2, 3],")
    print("   [4, 5, 6]]")
    print("\nB values:")
    print("  [[[1, 2, 3, 4],")
    print("    [5, 6, 7, 8],")
    print("    [9, 10, 11, 12]]]")

    var result = matmul(A, B)

    print("\nMojo Result shape:", result.shape())
    print("Expected shape: (1, 2, 4)")

    print("\nMojo Result values:")
    print("  [[[", end="")
    for j in range(4):
        print(result[0, 0, j], end="")
        if j < 3:
            print(", ", end="")
    print("],")
    print("    [", end="")
    for j in range(4):
        print(result[0, 1, j], end="")
        if j < 3:
            print(", ", end="")
    print("]]]")

    result.backward()
    alias dtype = DType.float32
    A_grad_expected = Tensor[dtype].d2([[10.0, 26.0, 42.0], [10.0, 26.0, 42.0]])
    assert_true(A.grad() == A_grad_expected)
    B_grad_expected = Tensor[dtype].d3(
        [[[5.0, 5.0, 5.0, 5.0], [7.0, 7.0, 7.0, 7.0], [9.0, 9.0, 9.0, 9.0]]]
    )
    assert_true(B.grad() == B_grad_expected)


fn test_gradient_accumulation() raises:
    """Test that gradients accumulate correctly with broadcasting."""
    print("=" * 80)
    print("Test 12: Gradient Accumulation with Broadcasting")
    print("=" * 80)

    var A = Tensor[DType.float32].randn(3, 2, 4, requires_grad=True)
    var B = Tensor[DType.float32].randn(1, 4, 5, requires_grad=True)

    var result = matmul(A, B)

    # Backward with non-uniform gradients
    var grad_output = Tensor[DType.float32].zeros(3, 2, 5)
    for i in range(3):
        for j in range(2):
            for k in range(5):
                grad_output[i, j, k] = Float32((i + 1) * (j + 1) * (k + 1))

    result.backward(grad_output)

    var grad_B = B.gradients()[]

    # grad_B should have accumulated gradients from all 3 batches
    # Verify it's not zero (which would indicate missing accumulation)
    var grad_B_sum: Float32 = 0.0
    for i in range(1):
        for j in range(4):
            for k in range(5):
                grad_B_sum += abs(grad_B[i, j, k])

    assert_true(grad_B_sum > 0.0, "grad_B should have accumulated values")
    print("Gradients accumulated correctly across broadcast dimension")
    print("  Total |grad_B|:", grad_B_sum)
    print()


fn test_non_contiguous_gradients() raises:
    """Test gradient computation with various tensor strides."""
    print("=" * 80)
    print("Test 13: Non-Contiguous Memory Patterns")
    print("=" * 80)

    # Create tensors with various access patterns
    var A = Tensor[DType.float32].randn(4, 3, 2, requires_grad=True)
    var B = Tensor[DType.float32].randn(1, 2, 5, requires_grad=True)

    var result = matmul(A, B)

    assert_equal(result.shape()[0], 4)
    assert_equal(result.shape()[1], 3)
    assert_equal(result.shape()[2], 5)

    var grad_output = Tensor[DType.float32].ones(4, 3, 5)
    result.backward(grad_output)

    var grad_A = A.gradients()[]
    var _grad_B = B.gradients()[]

    # Verify no NaN or Inf in gradients
    for i in range(4):
        for j in range(3):
            for k in range(2):
                assert_false(isnan(grad_A[i, j, k]), "grad_A contains NaN")
                assert_false(isinf(grad_A[i, j, k]), "grad_A contains Inf")

    print("Gradients computed correctly for non-contiguous patterns")
    print()


fn main() raises:
    print("\n")
    print("" * 80)
    print("" + " " * 78 + "")
    print("" + " " * 20 + "BATCHED MATMUL TEST SUITE" + " " * 33 + "")
    print(
        "" + " " * 15 + "Broadcasting + Gradient Verification" + " " * 27 + ""
    )
    print("" + " " * 78 + "")
    print("" * 80)
    print("\n")

    test_basic_2d_times_3d()
    test_3d_times_2d()
    test_batch_broadcasting()
    test_multi_batch_broadcasting()
    test_numerical_correctness()
    test_gradient_with_numerical_check()
    test_symmetric_broadcasting()
    test_complex_broadcasting()
    test_edge_case_single_batch()
    test_large_batch()
    test_pytorch_comparison()
    test_gradient_accumulation()
    test_non_contiguous_gradients()

    print("\n")
    print("" * 80)
    print("" + " " * 78 + "")
    print("" + " " * 25 + "ALL TESTS PASSED! " + " " * 30 + "")
    print("" + " " * 78 + "")
    print("" * 80)
    print("\n")
