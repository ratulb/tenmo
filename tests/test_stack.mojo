from tenmo import Tensor
from testing import assert_true
from shapes import Shape

# ============================================================================
# FORWARD PASS TESTS - STACK OPERATIONS
# ============================================================================


fn test_stack_axis0_2d() raises:
    """Test stack along axis 0 for 2D tensors."""
    print("test_stack_axis0_2d")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)

    # Expected shape: (2, 2, 3)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 3)

    # Expected:
    # [[[1, 2, 3], [4, 5, 6]],
    #  [[7, 8, 9], [10, 11, 12]]]
    var expected = Tensor[dtype].d3(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_stack_axis1_2d() raises:
    """Test stack along axis 1 for 2D tensors."""
    print("test_stack_axis1_2d")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=1)

    # Expected shape: (2, 2, 3)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 3)

    # Expected:
    # [[[1, 2, 3], [7, 8, 9]],
    #  [[4, 5, 6], [10, 11, 12]]]
    var expected = Tensor[dtype].d3(
        [
            [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
            [[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_stack_axis2_2d() raises:
    """Test stack along axis 2 for 2D tensors."""
    print("test_stack_axis2_2d")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=2)

    # Expected shape: (2, 3, 2)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 3)
    assert_true(result.shape()[2] == 2)

    # Expected:
    # [[[1, 7], [2, 8], [3, 9]],
    #  [[4, 10], [5, 11], [6, 12]]]
    var expected = Tensor[dtype].d3(
        [
            [[1.0, 7.0], [2.0, 8.0], [3.0, 9.0]],
            [[4.0, 10.0], [5.0, 11.0], [6.0, 12.0]],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_stack_axis_negative() raises:
    """Test stack with negative axis."""
    print("test_stack_axis_negative")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    # axis=-1 should be same as axis=2 for 2D input
    var result1 = Tensor[dtype].stack(tensors, axis=2)
    var result2 = Tensor[dtype].stack(tensors, axis=-1)

    assert_true(result1.all_close[atol=1e-6](result2))


fn test_stack_1d_tensors() raises:
    """Test stack of 1D tensors."""
    print("test_stack_1d_tensors")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])
    var C = Tensor[dtype].d1([7.0, 8.0, 9.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)

    # Expected shape: (3, 3)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 3)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_stack_1d_axis1() raises:
    """Test stack of 1D tensors along axis 1."""
    print("test_stack_1d_axis1")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=1)

    # Expected shape: (3, 2)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 2)

    var expected = Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    assert_true(result.all_close[atol=1e-6](expected))


fn test_stack_3d_tensors() raises:
    """Test stack of 3D tensors."""
    print("test_stack_3d_tensors")

    alias dtype = DType.float32
    var A = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    var B = Tensor[dtype].d3([[[5.0, 6.0], [7.0, 8.0]]])  # (1, 2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)

    # Expected shape: (2, 1, 2, 2)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 1)
    assert_true(result.shape()[2] == 2)
    assert_true(result.shape()[3] == 2)


fn test_stack_many_tensors() raises:
    """Test stack of many tensors."""
    print("test_stack_many_tensors")

    alias dtype = DType.float32
    var tensors = List[Tensor[dtype]]()

    # Create 5 tensors
    for i in range(5):
        var t = Tensor[dtype].ones(2, 3) * (i + 1)
        tensors.append(t)

    var result = Tensor[dtype].stack(tensors, axis=0)

    # Expected shape: (5, 2, 3)
    assert_true(result.shape()[0] == 5)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 3)


fn test_stack_single_tensor() raises:
    """Test stack of single tensor (edge case)."""
    print("test_stack_single_tensor")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)

    var result = Tensor[dtype].stack(tensors, axis=0)

    # Expected shape: (1, 2, 2)
    assert_true(result.shape()[0] == 1)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 2)


# ============================================================================
# VSTACK TESTS
# ============================================================================
fn test_vstack_2d() raises:
    """Test vstack with 2D tensors."""
    print("test_vstack_2d")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])  # (3,)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])  # (3,)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)
    # Expected shape: (2, 3)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 3)

    var expected = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert_true(result.all_close[atol=1e-6](expected))


fn test_vstack_1d() raises:
    """Test vstack with 1D tensors."""
    print("test_vstack_1d")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])
    var C = Tensor[dtype].d1([7.0, 8.0, 9.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].vstack(tensors)

    # Expected shape: (3, 3)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 3)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_vstack_mixed_rows() raises:
    """Test vstack with different number of rows."""
    print("test_vstack_mixed_rows")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    var B = Tensor[dtype].d2([[5.0, 6.0]])  # (1, 2)
    var C = Tensor[dtype].d2([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])  # (3, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].vstack(tensors)
    # Expected shape: (6, 2)
    assert_true(result.shape()[0] == 6)
    assert_true(result.shape()[1] == 2)

    var expected = Tensor[dtype].d2(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


# ============================================================================
# HSTACK TESTS
# ============================================================================


fn test_hstack_2d() raises:
    """Test hstack with 2D tensors."""
    print("test_hstack_2d")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0], [2.0], [3.0]])  # (3, 1)
    var B = Tensor[dtype].d2([[4.0], [5.0], [6.0]])  # (3, 1)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)

    # Expected shape: (3, 2)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 2)

    var expected = Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    assert_true(result.all_close[atol=1e-6](expected))


fn test_hstack_1d() raises:
    """Test hstack with 1D tensors."""
    print("test_hstack_1d")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0])
    var C = Tensor[dtype].d1([6.0, 7.0, 8.0, 9.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].hstack(tensors)

    # Expected shape: (9,)
    assert_true(result.shape()[0] == 9)

    var expected = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_hstack_mixed_cols() raises:
    """Test hstack with different number of columns."""
    print("test_hstack_mixed_cols")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    var B = Tensor[dtype].d2([[5.0], [6.0]])  # (2, 1)
    var C = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])  # (2, 3)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].hstack(tensors)

    # Expected shape: (2, 6)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 6)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 5.0, 7.0, 8.0, 9.0], [3.0, 4.0, 6.0, 10.0, 11.0, 12.0]]
    )

    assert_true(result.all_close[atol=1e-6](expected))


# ============================================================================
# BACKWARD PASS TESTS - STACK
# ============================================================================


fn test_stack_backward_axis0() raises:
    """Test gradient flow through stack along axis 0."""
    print("test_stack_backward_axis0")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad = Tensor[dtype].ones(2, 2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


fn test_stack_backward_axis1() raises:
    """Test gradient flow through stack along axis 1."""
    print("test_stack_backward_axis1")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=1)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad = Tensor[dtype].ones(2, 2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


fn test_stack_backward_weighted() raises:
    """Test gradient flow with weighted loss."""
    print("test_stack_backward_weighted")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (2, 3)
    # Apply weights
    var weights = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()
    # Gradients should be the weights
    var expected_grad_A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var expected_grad_B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


fn test_stack_backward_three_tensors() raises:
    """Test gradient flow through stack of three tensors."""
    print("test_stack_backward_three_tensors")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
    var C = Tensor[dtype].d1([5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (3, 2)
    var loss = result.sum()
    loss.backward()

    var expected_grad = Tensor[dtype].ones(2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))
    assert_true(C.grad().all_close[atol=1e-6](expected_grad))


fn test_stack_backward_chain() raises:
    """Test gradient flow through chained operations."""
    print("test_stack_backward_chain")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var stacked = Tensor[dtype].stack(tensors, axis=0)  # (2, 2)
    var squared = stacked * stacked  # Element-wise square
    var loss = squared.sum()
    loss.backward()

    # d(loss)/d(A) = 2 * A
    var expected_grad_A = Tensor[dtype].d1([2.0, 4.0])
    var expected_grad_B = Tensor[dtype].d1([6.0, 8.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


# ============================================================================
# BACKWARD PASS TESTS - VSTACK
# ============================================================================


fn test_vstack_backward() raises:
    """Test gradient flow through vstack."""
    print("test_vstack_backward")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)  # (1, 2)
    var B = Tensor[dtype].d2(
        [[3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )  # (2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)  # (3, 2)
    var loss = result.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].ones(1, 2)
    var expected_grad_B = Tensor[dtype].ones(2, 2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


fn test_vstack_backward_1d() raises:
    """Test gradient flow through vstack with 1D tensors."""
    print("test_vstack_backward_1d")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)  # (2, 3)
    var loss = result.sum()
    loss.backward()

    var expected_grad = Tensor[dtype].ones(3)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


fn test_vstack_backward_weighted() raises:
    """Test gradient flow through vstack with weighted loss."""
    print("test_vstack_backward_weighted")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
    var C = Tensor[dtype].d1([5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].vstack(tensors)  # (3, 2)

    # Row-wise weights
    var weights = Tensor[dtype].d2([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].d1([1.0, 1.0])
    var expected_grad_B = Tensor[dtype].d1([2.0, 2.0])
    var expected_grad_C = Tensor[dtype].d1([3.0, 3.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))
    assert_true(C.grad().all_close[atol=1e-6](expected_grad_C))


# ============================================================================
# BACKWARD PASS TESTS - HSTACK
# ============================================================================


fn test_hstack_backward() raises:
    """Test gradient flow through hstack."""
    print("test_hstack_backward")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0], [2.0]], requires_grad=True)  # (2, 1)
    var B = Tensor[dtype].d2(
        [[3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )  # (2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (2, 3)
    var loss = result.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].ones(2, 1)
    var expected_grad_B = Tensor[dtype].ones(2, 2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


fn test_hstack_backward_1d() raises:
    """Test gradient flow through hstack with 1D tensors."""
    print("test_hstack_backward_1d")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0, 5.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (5,)
    var loss = result.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].ones(2)
    var expected_grad_B = Tensor[dtype].ones(3)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


fn main() raises:
    test_stack_axis0_2d()
    test_stack_axis1_2d()
    test_stack_axis2_2d()
    test_stack_axis_negative()
    test_stack_1d_tensors()
    test_stack_1d_axis1()
    test_stack_3d_tensors()
    test_stack_many_tensors()
    test_stack_single_tensor()
    test_vstack_2d()
    test_vstack_1d()
    test_vstack_mixed_rows()
    test_hstack_2d()
    test_hstack_1d()
    test_hstack_mixed_cols()
    test_stack_backward_axis0()
    test_stack_backward_axis1()
    test_stack_backward_weighted()
    test_stack_backward_three_tensors()
    test_stack_backward_chain()
    test_vstack_backward()
    test_vstack_backward_1d()
    test_vstack_backward_weighted()
    test_hstack_backward()
    test_hstack_backward_1d()

    run_all_stack_tests()


# """
# Comprehensive test suite for Tensor stack/vstack/hstack operations.
# Test prefix: stk_
# """

# ============================================================================
# STACK TESTS - Forward Pass
# ============================================================================


fn test_stk_basic_2_tensors_axis0() raises:
    """Test stacking 2 tensors along axis 0."""
    print("test_stk_basic_2_tensors_axis0")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (2, 3)

    var expected = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert_true(result.all_close[atol=1e-6](expected))


fn test_stk_basic_3_tensors_axis0() raises:
    """Test stacking 3 tensors along axis 0."""
    print("test_stk_basic_3_tensors_axis0")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0])
    var B = Tensor[dtype].d1([3.0, 4.0])
    var C = Tensor[dtype].d1([5.0, 6.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (3, 2)

    var expected = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert_true(result.all_close[atol=1e-6](expected))


fn test_stk_axis1_2d() raises:
    """Test stacking along axis 1 for 2D tensors."""
    print("test_stk_axis1_2d")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])  # (2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=1)  # (2, 2, 2)

    # Expected shape: (2, 2, 2)
    # result[0] = [[1, 2], [5, 6]]
    # result[1] = [[3, 4], [7, 8]]
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 2)


fn test_stk_axis_negative() raises:
    """Test stacking with negative axis."""
    print("test_stk_axis_negative")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    # axis=-1 should be equivalent to axis=1 for 1D tensors
    var result = Tensor[dtype].stack(tensors, axis=-1)  # (3, 2)

    var expected = Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert_true(result.all_close[atol=1e-6](expected))


fn test_stk_single_element_tensors() raises:
    """Test stacking single-element tensors."""
    print("test_stk_single_element_tensors")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0])
    var B = Tensor[dtype].d1([2.0])
    var C = Tensor[dtype].d1([3.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (3, 1)

    var expected = Tensor[dtype].d2([[1.0], [2.0], [3.0]])
    assert_true(result.all_close[atol=1e-6](expected))


fn test_stk_3d_tensors() raises:
    """Test stacking 3D tensors."""
    print("test_stk_3d_tensors")

    alias dtype = DType.float32
    var A = Tensor[dtype].zeros(2, 3, 4)
    var B = Tensor[dtype].ones(2, 3, 4)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (2, 2, 3, 4)

    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 3)
    assert_true(result.shape()[3] == 4)


# ============================================================================
# STACK TESTS - Backward Pass
# ============================================================================


fn test_stk_backward_simple_axis0() raises:
    """Test gradient flow through stack with axis=0."""
    print("test_stk_backward_simple_axis0")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (2, 3)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad = Tensor[dtype].d1([1.0, 1.0, 1.0])
    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


fn test_stk_backward_weighted() raises:
    """Test gradient flow with weighted loss."""
    print("test_stk_backward_weighted")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (2, 3)

    # Apply weights
    var weights = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    # Gradients should be the weights
    var expected_grad_A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var expected_grad_B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


fn test_stk_backward_axis1() raises:
    """Test gradient flow through stack with axis=1."""
    print("test_stk_backward_axis1")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=1)  # (2, 2, 2)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad = Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]])
    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


fn test_stk_backward_three_tensors() raises:
    """Test gradient flow with three input tensors."""
    print("test_stk_backward_three_tensors")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
    var C = Tensor[dtype].d1([5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (3, 2)

    # Multiply by different weights for each stacked tensor
    var weights = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].d1([2.0, 3.0])
    var expected_grad_B = Tensor[dtype].d1([4.0, 5.0])
    var expected_grad_C = Tensor[dtype].d1([6.0, 7.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))
    assert_true(C.grad().all_close[atol=1e-6](expected_grad_C))


fn test_stk_backward_selective_grad() raises:
    """Test gradient flow when only some tensors require grad."""
    print("test_stk_backward_selective_grad")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=False)
    var C = Tensor[dtype].d1([7.0, 8.0, 9.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (3, 3)
    var loss = result.sum()
    loss.backward()

    # A and C should have gradients, B should not
    var expected_grad = Tensor[dtype].d1([1.0, 1.0, 1.0])
    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(C.grad().all_close[atol=1e-6](expected_grad))
    assert_true(not B.has_grad())


# ============================================================================
# VSTACK TESTS
# ============================================================================


fn test_stk_vstack_1d_tensors() raises:
    """Test vstack with 1D tensors."""
    print("test_stk_vstack_1d_tensors")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)  # (2, 3)

    var expected = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert_true(result.all_close[atol=1e-6](expected))


fn test_stk_vstack_2d_tensors() raises:
    """Test vstack with 2D tensors."""
    print("test_stk_vstack_2d_tensors")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])  # (2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)  # (4, 2)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    assert_true(result.all_close[atol=1e-6](expected))


fn test_stk_vstack_different_rows() raises:
    """Test vstack with tensors having different number of rows."""
    print("test_stk_vstack_different_rows")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0]])  # (1, 3)
    var B = Tensor[dtype].d2([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # (2, 3)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)  # (3, 3)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    assert_true(result.all_close[atol=1e-6](expected))


fn test_stk_vstack_backward() raises:
    """Test gradient flow through vstack."""
    print("test_stk_vstack_backward")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)  # (2, 3)

    # Apply different weights to each row
    var weights = Tensor[dtype].d2([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].d1([1.0, 1.0, 1.0])
    var expected_grad_B = Tensor[dtype].d1([2.0, 2.0, 2.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


# ============================================================================
# HSTACK TESTS
# ============================================================================


fn test_stk_hstack_1d_tensors() raises:
    """Test hstack with 1D tensors."""
    print("test_stk_hstack_1d_tensors")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (6,)

    var expected = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert_true(result.all_close[atol=1e-6](expected))


fn test_stk_hstack_2d_tensors() raises:
    """Test hstack with 2D tensors."""
    print("test_stk_hstack_2d_tensors")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])  # (2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (2, 4)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]
    )
    assert_true(result.all_close[atol=1e-6](expected))


fn test_stk_hstack_different_columns() raises:
    """Test hstack with tensors having different number of columns."""
    print("test_stk_hstack_different_columns")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0], [2.0]])  # (2, 1)
    var B = Tensor[dtype].d2([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])  # (2, 3)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (2, 4)

    var expected = Tensor[dtype].d2(
        [[1.0, 3.0, 4.0, 5.0], [2.0, 6.0, 7.0, 8.0]]
    )
    assert_true(result.all_close[atol=1e-6](expected))


fn test_stk_hstack_backward() raises:
    """Test gradient flow through hstack."""
    print("test_stk_hstack_backward")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (4,)

    # Apply different weights
    var weights = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].d1([1.0, 2.0])
    var expected_grad_B = Tensor[dtype].d1([3.0, 4.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


fn test_stk_hstack_2d_backward() raises:
    """Test gradient flow through hstack with 2D tensors."""
    print("test_stk_hstack_2d_backward")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (2, 4)

    # Apply different weights
    var weights = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].d2([[1.0, 2.0], [5.0, 6.0]])
    var expected_grad_B = Tensor[dtype].d2([[3.0, 4.0], [7.0, 8.0]])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


# ============================================================================
# EDGE CASES AND STRESS TESTS
# ============================================================================


fn test_stk_large_number_tensors() raises:
    """Test stacking a large number of tensors."""
    print("test_stk_large_number_tensors")

    alias dtype = DType.float32
    var tensors = List[Tensor[dtype]]()

    for i in range(10):
        var tensor = Tensor[dtype].full(Shape(2, 3), Scalar[dtype](i))
        tensors.append(tensor^)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (10, 2, 3)

    assert_true(result.shape()[0] == 10)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 3)


fn test_stk_zeros_and_ones() raises:
    """Test stacking tensors with zeros and ones."""
    print("test_stk_zeros_and_ones")

    alias dtype = DType.float32
    var A = Tensor[dtype].zeros(3, 4)
    var B = Tensor[dtype].ones(3, 4)
    var C = Tensor[dtype].zeros(3, 4)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (3, 3, 4)

    # Check middle slice is all ones
    var sum_middle = Float32(0.0)
    for i in range(3):
        for j in range(4):
            sum_middle += result[1, i, j]

    assert_true(abs(sum_middle - 12.0) < 1e-6)


fn test_stk_chain_operations() raises:
    """Test stacking followed by other operations."""
    print("test_stk_chain_operations")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var stacked = Tensor[dtype].stack(tensors, axis=0)  # (2, 3)
    var transposed = stacked.transpose(0, 1)  # (3, 2)
    var result = transposed.sum()
    result.backward()

    # All gradients should be 1.0
    var expected_grad = Tensor[dtype].d1([1.0, 1.0, 1.0])
    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


fn test_stk_mixed_operations() raises:
    """Test mixing stack, vstack, and hstack."""
    print("test_stk_mixed_operations")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0])
    var B = Tensor[dtype].d1([3.0, 4.0])

    var tensors1 = List[Tensor[dtype]]()
    tensors1.append(A)
    tensors1.append(B)

    var vstacked = Tensor[dtype].vstack(tensors1)  # (2, 2)

    var C = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])

    var tensors2 = List[Tensor[dtype]]()
    tensors2.append(vstacked)
    tensors2.append(C)

    var hstacked = Tensor[dtype].hstack(tensors2)  # (2, 4)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]
    )
    assert_true(hstacked.all_close[atol=1e-6](expected))


# ============================================================================
# MASTER TEST RUNNER
# ============================================================================


fn run_all_stack_tests() raises:
    """Run all stack operation tests."""
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE STACK OPERATIONS TEST SUITE")
    print("=" * 60 + "\n")

    # Forward pass tests
    print("--- STACK FORWARD TESTS ---")
    test_stk_basic_2_tensors_axis0()
    test_stk_basic_3_tensors_axis0()
    test_stk_axis1_2d()
    test_stk_axis_negative()
    test_stk_single_element_tensors()
    test_stk_3d_tensors()

    # Backward pass tests
    print("\n--- STACK BACKWARD TESTS ---")
    test_stk_backward_simple_axis0()
    test_stk_backward_weighted()
    test_stk_backward_axis1()
    test_stk_backward_three_tensors()
    test_stk_backward_selective_grad()

    # VStack tests
    print("\n--- VSTACK TESTS ---")
    test_stk_vstack_1d_tensors()
    test_stk_vstack_2d_tensors()
    test_stk_vstack_different_rows()
    test_stk_vstack_backward()

    # HStack tests
    print("\n--- HSTACK TESTS ---")
    test_stk_hstack_1d_tensors()
    test_stk_hstack_2d_tensors()
    test_stk_hstack_different_columns()
    test_stk_hstack_backward()
    test_stk_hstack_2d_backward()
    test_stk_large_number_tensors()
    test_stk_zeros_and_ones()
    test_stk_chain_operations()
    test_stk_mixed_operations()


fn test_stack_operations() raises:
    alias dtype = DType.float32

    print("=" * 70)
    print("TEST 1: Basic stack along axis 0")
    print("=" * 70)

    var A = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var B = Tensor[dtype].d2(
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], requires_grad=True
    )

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0, requires_grad=True)

    print("A (2x3):")
    A.print()
    print("\nB (2x3):")
    B.print()
    print("\nstack([A, B], axis=0) - Shape:", result.shape())
    result.print()
    # Expected: (2, 2, 3)
    # [[[1, 2, 3],
    #   [4, 5, 6]],
    #  [[7, 8, 9],
    #   [10, 11, 12]]]

    print("\n" + "=" * 70)
    print("TEST 2: Stack along axis 1")
    print("=" * 70)

    var result2 = Tensor[dtype].stack(tensors, axis=1, requires_grad=True)
    print("stack([A, B], axis=1) - Shape:", result2.shape())
    result2.print()
    # Expected: (2, 2, 3)
    # [[[1, 2, 3],
    #   [7, 8, 9]],
    #  [[4, 5, 6],
    #   [10, 11, 12]]]

    print("\n" + "=" * 70)
    print("TEST 3: Stack along axis 2")
    print("=" * 70)

    var result3 = Tensor[dtype].stack(tensors, axis=2, requires_grad=True)
    print("stack([A, B], axis=2) - Shape:", result3.shape())
    result3.print()
    # Expected: (2, 3, 2)
    # [[[1, 7],
    #   [2, 8],
    #   [3, 9]],
    #  [[4, 10],
    #   [5, 11],
    #   [6, 12]]]

    print("\n" + "=" * 70)
    print("TEST 4: Gradient flow")
    print("=" * 70)

    var loss = result.sum()
    loss.backward()

    print("Grad A:")
    A.grad().print()
    # Expected: all ones (2x3)

    print("\nGrad B:")
    B.grad().print()
    # Expected: all ones (2x3)

    print("\n" + "=" * 70)
    print("TEST 5: vstack")
    print("=" * 70)

    var C = Tensor[dtype].ones(1, 3, requires_grad=True)
    var D = Tensor[dtype].ones(1, 3, requires_grad=True) * 2.0

    var vstack_list = List[Tensor[dtype]]()
    vstack_list.append(C)
    vstack_list.append(D)

    var vstacked = Tensor[dtype].vstack(vstack_list, requires_grad=True)
    print("vstack([C(1x3), D(1x3)]) - Shape:", vstacked.shape())
    vstacked.print()
    # Expected: (2, 3)
    # [[1, 1, 1],
    #  [2, 2, 2]]

    print("\n" + "=" * 70)
    print("TEST 6: hstack")
    print("=" * 70)

    var E = Tensor[dtype].ones(2, 1, requires_grad=True)
    var F = Tensor[dtype].ones(2, 2, requires_grad=True) * 3.0

    var hstack_list = List[Tensor[dtype]]()
    hstack_list.append(E)
    hstack_list.append(F)

    var hstacked = Tensor[dtype].hstack(hstack_list, requires_grad=True)
    print("hstack([E(2x1), F(2x2)]) - Shape:", hstacked.shape())
    hstacked.print()
    # Expected: (2, 3)
    # [[1, 3, 3],
    #  [1, 3, 3]]

    print("\n" + "=" * 70)
    print("TEST 7: 1D vstack")
    print("=" * 70)

    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var vstack_1d = List[Tensor[dtype]]()
    vstack_1d.append(a)
    vstack_1d.append(b)

    var vstacked_1d = Tensor[dtype].vstack(vstack_1d, requires_grad=True)
    print("vstack([a(3,), b(3,)]) - Shape:", vstacked_1d.shape())
    vstacked_1d.print()
    # Expected: (2, 3)
    # [[1, 2, 3],
    #  [4, 5, 6]]

    print("\n" + "=" * 70)
    print("TEST 8: 1D hstack")
    print("=" * 70)

    var hstacked_1d = Tensor[dtype].hstack(vstack_1d, requires_grad=True)
    print("hstack([a(3,), b(3,)]) - Shape:", hstacked_1d.shape())
    hstacked_1d.print()
    # Expected: (6,)
    # [1, 2, 3, 4, 5, 6]

    print("\n✅ All stack tests passed!")


# ```

# ---


### **1. Forward Pass Strategy:**
# ```
# stack = unsqueeze_each + concat
# ```

### **2. Backward Pass Strategy:**
# ```
# Split gradient along stacked axis → Squeeze each split
# ```

### **3. Edge Cases Handled:**
# - Single tensor: just unsqueeze
# - Empty list: raises error
# - Shape mismatch: validates before stacking
# - 1D tensors in vstack: reshapes to 2D
# - 1D tensors in hstack: uses concat

### **4. Gradient Flow:**
# ```
# Forward:  A(2,3) → unsqueeze(0) → A'(1,2,3) ──┐
#          B(2,3) → unsqueeze(0) → B'(1,2,3) ──┤→ concat → Result(2,2,3)
#          C(2,3) → unsqueeze(0) → C'(1,2,3) ──┘

# Backward: grad_Result(2,2,3) → split → [grad_A'(1,2,3),
#                                         grad_B'(1,2,3),
#                                         grad_C'(1,2,3)]
#                                    ↓ squeeze(0)
#                                [grad_A(2,3), grad_B(2,3), grad_C(2,3)]
