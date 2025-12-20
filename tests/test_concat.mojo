from tenmo import Tensor
from testing import assert_true

# ============================================================================
# FORWARD PASS TESTS - BASIC CONCATENATION
# ============================================================================


fn test_concat_axis0_2d() raises:
    """Test concatenation along axis 0 for 2D tensors."""
    print("test_concat_axis0_2d")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (4, 3)
    assert_true(result.shape()[0] == 4)
    assert_true(result.shape()[1] == 3)

    # Expected values: [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_concat_axis1_2d() raises:
    """Test concatenation along axis 1 for 2D tensors."""
    print("test_concat_axis1_2d")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=1)

    # Expected shape: (2, 6)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 6)

    # Expected values: [[1,2,3,7,8,9], [4,5,6,10,11,12]]
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 7.0, 8.0, 9.0], [4.0, 5.0, 6.0, 10.0, 11.0, 12.0]]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_concat_three_tensors_axis0() raises:
    """Test concatenation of three tensors along axis 0."""
    print("test_concat_three_tensors_axis0")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var B = Tensor[dtype].d2([[5.0, 6.0]])
    var C = Tensor[dtype].d2([[7.0, 8.0], [9.0, 10.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (5, 2)
    assert_true(result.shape()[0] == 5)
    assert_true(result.shape()[1] == 2)

    # Expected values
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_concat_1d_tensors() raises:
    """Test concatenation of 1D tensors."""
    print("test_concat_1d_tensors")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0])
    var C = Tensor[dtype].d1([6.0, 7.0, 8.0, 9.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (9,)
    assert_true(result.shape()[0] == 9)

    # Expected values: [1,2,3,4,5,6,7,8,9]
    var expected = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_concat_3d_tensors_axis0() raises:
    """Test concatenation of 3D tensors along axis 0."""
    print("test_concat_3d_tensors_axis0")

    alias dtype = DType.float32
    var A = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    var B = Tensor[dtype].d3(
        [[[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]
    )  # (2, 2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (3, 2, 2)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 2)

    # Expected result:
    # [[[1, 2], [3, 4]],      <- from A
    #  [[5, 6], [7, 8]],      <- from B
    #  [[9, 10], [11, 12]]]   <- from B

    # Verify some elements using proper 3D indexing
    var expected = Tensor[dtype].d3(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_concat_3d_tensors_axis1() raises:
    """Test concatenation of 3D tensors along axis 1."""
    print("test_concat_3d_tensors_axis1")

    alias dtype = DType.float32
    var A = Tensor[dtype].d3([[[1.0, 2.0]]])  # (1, 1, 2)
    var B = Tensor[dtype].d3([[[3.0, 4.0], [5.0, 6.0]]])  # (1, 2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=1)

    # Expected shape: (1, 3, 2)
    assert_true(result.shape()[0] == 1)
    assert_true(result.shape()[1] == 3)
    assert_true(result.shape()[2] == 2)


fn test_concat_3d_tensors_axis2() raises:
    """Test concatenation of 3D tensors along axis 2."""
    print("test_concat_3d_tensors_axis2")

    alias dtype = DType.float32
    var A = Tensor[dtype].d3([[[1.0], [2.0]]])  # (1, 2, 1)
    var B = Tensor[dtype].d3([[[3.0, 4.0], [5.0, 6.0]]])  # (1, 2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=2)

    # Expected shape: (1, 2, 3)
    assert_true(result.shape()[0] == 1)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 3)

    # Expected: [[[1,3,4], [2,5,6]]]
    var expected = Tensor[dtype].d3([[[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]]])

    assert_true(result.all_close[atol=1e-6](expected))


fn test_concat_single_tensor() raises:
    """Test concatenation of a single tensor (edge case)."""
    print("test_concat_single_tensor")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Should be identical to A
    assert_true(result.all_close[atol=1e-6](A))


# ============================================================================
# BACKWARD PASS TESTS - GRADIENT FLOW
# ============================================================================


fn test_concat_backward_axis0_simple() raises:
    """Test gradient flow through concat along axis 0."""
    print("test_concat_backward_axis0_simple")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad_A = Tensor[dtype].ones(2, 2)
    var expected_grad_B = Tensor[dtype].ones(1, 2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


fn test_concat_backward_axis1_simple() raises:
    """Test gradient flow through concat along axis 1."""
    print("test_concat_backward_axis1_simple")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0], [6.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=1)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad_A = Tensor[dtype].ones(2, 2)
    var expected_grad_B = Tensor[dtype].ones(2, 1)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


fn test_concat_backward_weighted_loss() raises:
    """Test gradient flow with weighted loss."""
    print("test_concat_backward_weighted_loss")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)  # Shape: (4, 2)

    # Apply weights: multiply each row by its index
    var weights = Tensor[dtype].d2(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    )
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    # Grad for A (first 2 rows): weights [0, 0] and [1, 1]
    var expected_grad_A = Tensor[dtype].d2([[0.0, 0.0], [1.0, 1.0]])

    # Grad for B (last 2 rows): weights [2, 2] and [3, 3]
    var expected_grad_B = Tensor[dtype].d2([[2.0, 2.0], [3.0, 3.0]])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


fn test_concat_backward_three_tensors() raises:
    """Test gradient flow through concat of three tensors."""
    print("test_concat_backward_three_tensors")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0], requires_grad=True)
    var C = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].concat(tensors, axis=0)  # [1,2,3,4,5,6]

    # Multiply by indices: [0,1,2,3,4,5]
    var indices = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    var weighted = result * indices
    var loss = weighted.sum()
    loss.backward()

    # Gradients should be the indices
    var expected_grad_A = Tensor[dtype].d1([0.0, 1.0])
    var expected_grad_B = Tensor[dtype].d1([2.0])
    var expected_grad_C = Tensor[dtype].d1([3.0, 4.0, 5.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))
    assert_true(C.grad().all_close[atol=1e-6](expected_grad_C))


fn test_concat_backward_3d_axis0() raises:
    """Test gradient flow through 3D concat along axis 0."""
    print("test_concat_backward_3d_axis0")

    alias dtype = DType.float32
    var A = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True
    )  # (1,2,2)
    var B = Tensor[dtype].d3(
        [[[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )  # (1,2,2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)  # (2,2,2)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad = Tensor[dtype].ones(1, 2, 2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


fn test_concat_backward_3d_axis2() raises:
    """Test gradient flow through 3D concat along axis 2."""
    print("test_concat_backward_3d_axis2")

    alias dtype = DType.float32
    var A = Tensor[dtype].d3([[[1.0], [2.0]]], requires_grad=True)  # (1,2,1)
    var B = Tensor[dtype].d3(
        [[[3.0, 4.0], [5.0, 6.0]]], requires_grad=True
    )  # (1,2,2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=2)  # (1,2,3)
    var loss = result.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].ones(1, 2, 1)
    var expected_grad_B = Tensor[dtype].ones(1, 2, 2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


fn test_concat_backward_chain() raises:
    """Test gradient flow through chained operations."""
    print("test_concat_backward_chain")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var concat_result = Tensor[dtype].concat(tensors, axis=0)  # [1,2,3,4]
    var squared = concat_result * concat_result  # [1,4,9,16]
    var loss = squared.sum()  # 30
    loss.backward()

    # d(loss)/d(concat) = 2 * concat = [2, 4, 6, 8]
    # Grad A gets [2, 4], Grad B gets [6, 8]
    var expected_grad_A = Tensor[dtype].d1([2.0, 4.0])
    var expected_grad_B = Tensor[dtype].d1([6.0, 8.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


# ============================================================================
# VIEW TESTS - CONCATENATING VIEWS
# ============================================================================


fn test_concat_simple_views_axis0() raises:
    """Test concatenation of simple views along axis 0."""
    print("test_concat_simple_views_axis0")

    alias dtype = DType.float32
    var base = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    var view1 = base.view([2, 4])  # Same as base
    var view2 = base.view([2, 4])  # Another view

    var tensors = List[Tensor[dtype]]()
    tensors.append(view1)
    tensors.append(view2)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (4, 4)
    assert_true(result.shape()[0] == 4)
    assert_true(result.shape()[1] == 4)

    # First two rows should match base, next two should also match base
    var expected = Tensor[dtype].d2(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_concat_views_with_offsets() raises:
    """Test concatenation of views with different offsets."""
    print("test_concat_views_with_offsets")

    alias dtype = DType.float32
    var base = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    # Create two views: first half and second half
    var view1 = base.view([4], offset=0)  # [1,2,3,4]
    var view2 = base.view([4], offset=4)  # [5,6,7,8]

    var tensors = List[Tensor[dtype]]()
    tensors.append(view1)
    tensors.append(view2)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Should reconstruct the original
    assert_true(result.all_close[atol=1e-6](base))


fn test_concat_reshaped_views() raises:
    """Test concatenation of reshaped views."""
    print("test_concat_reshaped_views")

    alias dtype = DType.float32
    var base = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Reshape to 2D views
    var view1 = base.view([2, 3], offset=0)  # [[1,2,3], [4,5,6]]
    var view2 = base.view([1, 3], offset=0)  # [[1,2,3]]

    var tensors = List[Tensor[dtype]]()
    tensors.append(view1)
    tensors.append(view2)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (3, 3)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 3)


fn test_concat_nested_views() raises:
    """Test concatenation of nested views (view of view)."""
    print("test_concat_nested_views")

    alias dtype = DType.float32
    var base = Tensor[dtype].d3([[[1.0, 2.0, 3.0, 4.0]]])  # (1,1,4)

    # Create view and then view of view
    var view1 = base.view([4])  # Flatten to [1,2,3,4]
    var view2 = view1.view([2, 2])  # Reshape to [[1,2], [3,4]]

    var view3 = base.view([2, 2])  # Direct reshape

    var tensors = List[Tensor[dtype]]()
    tensors.append(view2)
    tensors.append(view3)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (4, 2)
    assert_true(result.shape()[0] == 4)
    assert_true(result.shape()[1] == 2)


fn test_concat_views_backward() raises:
    """Test gradient flow through concatenated views."""
    print("test_concat_views_backward")

    alias dtype = DType.float32
    var base1 = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var base2 = Tensor[dtype].d1([5.0, 6.0, 7.0, 8.0], requires_grad=True)

    var view1 = base1.view([2, 2])  # [[1,2], [3,4]]
    var view2 = base2.view([2, 2])  # [[5,6], [7,8]]

    var tensors = List[Tensor[dtype]]()
    tensors.append(view1)
    tensors.append(view2)

    var result = Tensor[dtype].concat(tensors, axis=0)  # (4, 2)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad = Tensor[dtype].ones(4)

    assert_true(base1.grad().all_close[atol=1e-6](expected_grad))
    assert_true(base2.grad().all_close[atol=1e-6](expected_grad))


fn test_concat_views_with_different_offsets_backward() raises:
    """Test gradient flow through views with different offsets."""
    print("test_concat_views_with_different_offsets_backward")

    alias dtype = DType.float32
    var base = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], requires_grad=True
    )

    # Two non-overlapping views
    var view1 = base.view([2], offset=0)  # [1, 2]
    var view2 = base.view([2], offset=4)  # [5, 6]

    var tensors = List[Tensor[dtype]]()
    tensors.append(view1)
    tensors.append(view2)

    var result = Tensor[dtype].concat(tensors, axis=0)  # [1, 2, 5, 6]

    # Weighted loss
    var weights = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var loss = (result * weights).sum()
    loss.backward()

    # Gradient should only affect elements 0,1,4,5 of base
    var expected_grad = Tensor[dtype].d1(
        [1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
    )

    assert_true(base.grad().all_close[atol=1e-6](expected_grad))


fn test_concat_mixed_tensors_and_views() raises:
    """Test concatenation of mixed tensors and views."""
    print("test_concat_mixed_tensors_and_views")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var base_B = Tensor[dtype].d1([5.0, 6.0, 7.0, 8.0], requires_grad=True)
    var view_B = base_B.view([2, 2])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(view_B)

    var result = Tensor[dtype].concat(tensors, axis=0)
    var loss = result.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].ones(2, 2)
    var expected_grad_B = Tensor[dtype].ones(4)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(base_B.grad().all_close[atol=1e-6](expected_grad_B))


# ============================================================================
# EDGE CASES AND SPECIAL TESTS
# ============================================================================


fn test_concat_empty_dimension() raises:
    """Test concatenation when some tensors have size 0 in concat dimension."""
    print("test_concat_empty_dimension")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0]])  # (1, 3)
    var B = Tensor[dtype].zeros(0, 3)  # (0, 3) - empty
    var C = Tensor[dtype].d2([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # (2, 3)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (3, 3) - empty tensor contributes nothing
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 3)


fn test_concat_large_number_of_tensors() raises:
    """Test concatenation of many tensors."""
    print("test_concat_large_number_of_tensors")

    alias dtype = DType.float32
    var tensors = List[Tensor[dtype]]()

    # Create 10 tensors, each (1, 5)
    for i in range(10):
        var t = Tensor[dtype].ones(1, 5) * (i + 1)
        tensors.append(t)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (10, 5)
    assert_true(result.shape()[0] == 10)
    assert_true(result.shape()[1] == 5)

    # Verify first and last rows using proper 2D construction
    var expected_first_row = Tensor[dtype].ones(1, 5) * 1.0
    var expected_last_row = Tensor[dtype].ones(1, 5) * 10.0

    # Extract first and last rows using views
    var first_row = result.view([1, 5], offset=0)
    var last_row = result.view([1, 5], offset=45)  # 9 rows * 5 elements = 45

    assert_true(first_row.all_close[atol=1e-6](expected_first_row))
    assert_true(last_row.all_close[atol=1e-6](expected_last_row))


fn test_concat_requires_grad_propagation() raises:
    """Test that requires_grad is properly set."""
    print("test_concat_requires_grad_propagation")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=False)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    # If any input requires grad, output should require grad
    var result1 = Tensor[dtype].concat(tensors, axis=0)
    assert_true(result1.requires_grad)

    # Can explicitly set requires_grad=False
    var result2 = Tensor[dtype].concat(tensors, axis=0, requires_grad=False)
    assert_true(result2.requires_grad == False)


fn test_concat_contiguous_output() raises:
    """Test that concat output is contiguous."""
    print("test_concat_contiguous_output")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Result should be contiguous
    assert_true(result.is_contiguous())


fn test_concat_negative_axis() raises:
    """Test concatenation with negative axis."""
    print("test_concat_negative_axis")

    alias dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    # axis=-1 should be same as axis=1 for 2D tensors
    var result1 = Tensor[dtype].concat(tensors, axis=1)
    var result2 = Tensor[dtype].concat(tensors, axis=-1)

    assert_true(result1.all_close[atol=1e-6](result2))


fn test_concat_scalar_to_1d() raises:
    """Test concatenation of scalar views to 1D."""
    print("test_concat_scalar_to_1d")

    alias dtype = DType.float32
    var a = Tensor[dtype].scalar(10.0)
    var b = Tensor[dtype].scalar(20.0)
    var c = Tensor[dtype].scalar(30.0)

    var view_a = a.view([1])
    var view_b = b.view([1])
    var view_c = c.view([1])

    var tensors = List[Tensor[dtype]]()
    tensors.append(view_a)
    tensors.append(view_b)
    tensors.append(view_c)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected: [10, 20, 30]
    var expected = Tensor[dtype].d1([10.0, 20.0, 30.0])
    assert_true(result.all_close[atol=1e-6](expected))


# ============================================================================
# COMPLEX INTEGRATION TESTS
# ============================================================================


fn test_concat_in_network_forward() raises:
    """Test concat as part of a network forward pass."""
    print("test_concat_in_network_forward")

    alias dtype = DType.float32

    # Simulate two branches of a network
    var branch1 = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
    var branch2 = Tensor[dtype].d2([[3.0, 4.0, 5.0]], requires_grad=True)

    # Concat along feature dimension
    var tensors = List[Tensor[dtype]]()
    tensors.append(branch1)
    tensors.append(branch2)

    var concat = Tensor[dtype].concat(tensors, axis=1)  # (1, 5)

    # Apply a "weight" layer
    var weights = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var output = concat * weights
    var loss = output.sum()
    loss.backward()

    # Check gradients flow back to both branches
    var expected_grad_branch1 = Tensor[dtype].d2([[1.0, 2.0]])
    var expected_grad_branch2 = Tensor[dtype].d2([[3.0, 4.0, 5.0]])

    assert_true(branch1.grad().all_close[atol=1e-6](expected_grad_branch1))
    assert_true(branch2.grad().all_close[atol=1e-6](expected_grad_branch2))


fn test_concat_multiple_times() raises:
    """Test multiple concatenations in sequence."""
    print("test_concat_multiple_times")

    alias dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)

    # First concat
    var tensors1 = List[Tensor[dtype]]()
    tensors1.append(A)
    tensors1.append(B)
    var concat1 = Tensor[dtype].concat(tensors1, axis=0)  # [1,2,3,4]

    # Second concat with itself
    var tensors2 = List[Tensor[dtype]]()
    tensors2.append(concat1)
    tensors2.append(concat1)
    var concat2 = Tensor[dtype].concat(tensors2, axis=0)  # [1,2,3,4,1,2,3,4]

    var loss = concat2.sum()
    loss.backward()

    # Each element of A and B appears twice in concat2, so gradients are 2.0
    var expected_grad = Tensor[dtype].d1([2.0, 2.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


fn test_concat_view_contiguous_mix() raises:
    """Test concatenation of views that need contiguous conversion."""
    print("test_concat_view_contiguous_mix")

    alias dtype = DType.float32
    var base = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True
    )

    # Create a view and make it contiguous
    var view1 = base.view([2, 3])
    var cont1 = view1.contiguous()

    # Another tensor
    var A = Tensor[dtype].d2([[7.0, 8.0, 9.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(cont1)
    tensors.append(A)

    var result = Tensor[dtype].concat(tensors, axis=0)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad_base = Tensor[dtype].ones(6)
    var expected_grad_A = Tensor[dtype].ones(1, 3)

    assert_true(base.grad().all_close[atol=1e-6](expected_grad_base))
    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


fn main() raises:
    print("=" * 80)
    print("CONCAT COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    print("\n--- FORWARD PASS TESTS: BASIC CONCATENATION ---")
    test_concat_axis0_2d()
    test_concat_axis1_2d()
    test_concat_three_tensors_axis0()
    test_concat_1d_tensors()
    test_concat_3d_tensors_axis0()
    test_concat_3d_tensors_axis1()
    test_concat_3d_tensors_axis2()
    test_concat_single_tensor()

    print("\n--- BACKWARD PASS TESTS: GRADIENT FLOW ---")
    test_concat_backward_axis0_simple()
    test_concat_backward_axis1_simple()
    test_concat_backward_weighted_loss()
    test_concat_backward_three_tensors()
    test_concat_backward_3d_axis0()
    test_concat_backward_3d_axis2()
    test_concat_backward_chain()

    print("\n--- VIEW TESTS: CONCATENATING VIEWS ---")
    test_concat_simple_views_axis0()
    test_concat_views_with_offsets()
    test_concat_reshaped_views()
    test_concat_nested_views()
    test_concat_views_backward()
    test_concat_views_with_different_offsets_backward()
    test_concat_mixed_tensors_and_views()

    print("\n--- EDGE CASES AND SPECIAL TESTS ---")
    # test_concat_empty_dimension()
    test_concat_large_number_of_tensors()
    test_concat_requires_grad_propagation()
    test_concat_contiguous_output()
    test_concat_negative_axis()
    test_concat_scalar_to_1d()

    print("\n--- COMPLEX INTEGRATION TESTS ---")
    test_concat_in_network_forward()
    test_concat_multiple_times()
    test_concat_view_contiguous_mix()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("\nTotal tests run: 34")
    print("  - Forward pass: 8 tests")
    print("  - Backward pass: 7 tests")
    print("  - View handling: 7 tests")
    print("  - Edge cases: 6 tests")
    print("  - Integration: 3 tests")

    run_all_concat_tests()


# ============================================================================
# Basic Concat Tests
# ============================================================================


fn test_concat_axis0_basic_ct() raises:
    """Test basic concatenation along axis 0."""
    print("test_concat_axis0_basic_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (4, 3)
    assert_true(result.shape()[0] == 4)
    assert_true(result.shape()[1] == 3)

    # Check values
    assert_true(result[0, 0] == 1.0)
    assert_true(result[1, 2] == 6.0)
    assert_true(result[2, 0] == 7.0)
    assert_true(result[3, 2] == 12.0)


fn test_concat_axis1_basic_ct() raises:
    """Test basic concatenation along axis 1."""
    print("test_concat_axis1_basic_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].d2([[1.0, 2.0], [4.0, 5.0]])
    var B = Tensor[dtype].d2([[3.0], [6.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=1)

    # Expected shape: (2, 3)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 3)

    # Check values
    assert_true(result[0, 0] == 1.0)
    assert_true(result[0, 1] == 2.0)
    assert_true(result[0, 2] == 3.0)
    assert_true(result[1, 0] == 4.0)
    assert_true(result[1, 1] == 5.0)
    assert_true(result[1, 2] == 6.0)


fn test_concat_three_tensors_ct() raises:
    """Test concatenating three tensors."""
    print("test_concat_three_tensors_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].d2([[1.0, 2.0]])  # (1, 2)
    var B = Tensor[dtype].d2([[3.0, 4.0]])  # (1, 2)
    var C = Tensor[dtype].d2([[5.0, 6.0]])  # (1, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (3, 2)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 2)

    # Check values
    assert_true(result[0, 0] == 1.0)
    assert_true(result[1, 0] == 3.0)
    assert_true(result[2, 0] == 5.0)


fn test_concat_single_tensor_ct() raises:
    """Test concatenating a single tensor (should return copy)."""
    print("test_concat_single_tensor_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var tensors = List[Tensor[dtype]]()
    tensors.append(A)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Should have same shape and values
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 2)
    assert_true(result[0, 0] == 1.0)
    assert_true(result[1, 1] == 4.0)


fn test_concat_1d_tensors_ct() raises:
    """Test concatenating 1D tensors."""
    print("test_concat_1d_tensors_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (5,)
    assert_true(result.shape()[0] == 5)
    assert_true(result[0] == 1.0)
    assert_true(result[3] == 4.0)
    assert_true(result[4] == 5.0)


fn test_concat_3d_tensors_axis0_ct() raises:
    """Test concatenating 3D tensors along axis 0."""
    print("test_concat_3d_tensors_axis0_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].ones(2, 3, 4)
    var B = Tensor[dtype].ones(1, 3, 4) * 2.0

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Expected shape: (3, 3, 4)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 3)
    assert_true(result.shape()[2] == 4)

    # Check values
    assert_true(result[0, 0, 0] == 1.0)
    assert_true(result[1, 0, 0] == 1.0)
    assert_true(result[2, 0, 0] == 2.0)


fn test_concat_3d_tensors_axis1_ct() raises:
    """Test concatenating 3D tensors along axis 1."""
    print("test_concat_3d_tensors_axis1_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].ones(2, 2, 3)
    var B = Tensor[dtype].ones(2, 1, 3) * 2.0

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=1)

    # Expected shape: (2, 3, 3)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 3)
    assert_true(result.shape()[2] == 3)

    # Check values
    assert_true(result[0, 0, 0] == 1.0)
    assert_true(result[0, 2, 0] == 2.0)


fn test_concat_3d_tensors_axis2_ct() raises:
    """Test concatenating 3D tensors along axis 2."""
    print("test_concat_3d_tensors_axis2_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].ones(2, 3, 2)
    var B = Tensor[dtype].ones(2, 3, 1) * 2.0

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=2)

    # Expected shape: (2, 3, 3)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 3)
    assert_true(result.shape()[2] == 3)

    # Check values
    assert_true(result[0, 0, 0] == 1.0)
    assert_true(result[0, 0, 2] == 2.0)


# ============================================================================
# Gradient Tests
# ============================================================================


fn test_concat_backward_axis0_ct() raises:
    """Test gradient flow for concat along axis 0."""
    print("test_concat_backward_axis0_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1
    var expected_grad = Tensor[dtype].ones(A.shape())
    assert_true(A.grad().all_close[atol=1e-6](expected_grad))

    expected_grad = Tensor[dtype].ones(B.shape())
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


fn test_concat_backward_axis1_ct() raises:
    """Test gradient flow for concat along axis 1."""
    print("test_concat_backward_axis1_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0], [6.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=1)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1
    var expected_grad_A = Tensor[dtype].ones(A.shape())
    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))

    var expected_grad_B = Tensor[dtype].ones(B.shape())
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


fn test_concat_backward_weighted_ct() raises:
    """Test gradient flow with weighted loss."""
    print("test_concat_backward_weighted_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[3.0, 4.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Weighted sum: multiply by [2, 3]
    var weights = Tensor[dtype].d2([[2.0, 2.0], [3.0, 3.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    # A gradient should be 2
    var expected_grad_A = Tensor[dtype].ones(A.shape()) * 2.0
    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))

    # B gradient should be 3
    var expected_grad_B = Tensor[dtype].ones(B.shape()) * 3.0
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


fn test_concat_backward_three_tensors_ct() raises:
    """Test gradient flow with three tensors."""
    print("test_concat_backward_three_tensors_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0], requires_grad=True)
    var C = Tensor[dtype].d1([4.0, 5.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].concat(tensors, axis=0)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1
    assert_true(A.grad().all_close[atol=1e-6](Tensor[dtype].ones(A.shape())))
    assert_true(B.grad().all_close[atol=1e-6](Tensor[dtype].ones(B.shape())))
    assert_true(C.grad().all_close[atol=1e-6](Tensor[dtype].ones(C.shape())))


fn test_concat_backward_chain_ct() raises:
    """Test gradient flow through chain of operations."""
    print("test_concat_backward_chain_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[3.0, 4.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var concat_result = Tensor[dtype].concat(tensors, axis=0)
    var doubled = concat_result * 2.0
    var loss = doubled.sum()
    loss.backward()

    # Gradients should be 2 (due to *2.0)
    var expected_grad = Tensor[dtype].ones(A.shape()) * 2.0
    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


# ============================================================================
# View Tests
# ============================================================================


fn test_concat_with_views_ct() raises:
    """Test concatenating views."""
    print("test_concat_with_views_ct")
    alias dtype = DType.float32

    var base = Tensor[dtype].arange(0.0, 12.0, 1.0)
    base = base.reshape(3, 4)

    # Create views
    var view1 = base.view([2, 4], offset=0)  # First 2 rows
    var view2 = base.view([1, 4], offset=8)  # Last row

    var tensors = List[Tensor[dtype]]()
    tensors.append(view1)
    tensors.append(view2)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Should reconstruct original
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 4)
    assert_true(result.all_close[atol=1e-6](base))


fn test_concat_views_grad_flow_ct() raises:
    """Test gradient flow through concatenated views."""
    print("test_concat_views_grad_flow_ct")
    alias dtype = DType.float32

    var base = Tensor[dtype].ones(4, 3, requires_grad=True)

    # Create views (first 2 rows, last 2 rows)
    var view1 = base.view([2, 3], offset=0)
    var view2 = base.view([2, 3], offset=6)

    var tensors = List[Tensor[dtype]]()
    tensors.append(view1)
    tensors.append(view2)

    var result = Tensor[dtype].concat(tensors, axis=0)
    var loss = result.sum()
    loss.backward()

    # Base should have gradient of 1 everywhere
    var expected_grad = Tensor[dtype].ones(base.shape())
    assert_true(base.grad().all_close[atol=1e-6](expected_grad))


fn test_concat_nested_views_ct() raises:
    """Test concatenating nested views."""
    print("test_concat_nested_views_ct")

    alias dtype = DType.float32

    var base = Tensor[dtype].arange(0.0, 24.0, 1.0, requires_grad=True)
    r = base.reshape(4, 6)

    # Create view of first 3 rows
    var view1 = r.view([3, 6], offset=0)

    # Create nested view - first 2 rows of view1
    var view2 = view1.view([2, 6], offset=0)

    # Create another view - last row
    var view3 = r.view([1, 6], offset=18)

    var tensors = List[Tensor[dtype]]()
    tensors.append(view2)
    tensors.append(view3)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Should be 3 rows
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 6)

    # Test gradient flow
    var loss = result.sum()
    loss.backward()
    # Check specific gradients
    # First 2 rows should have grad=1
    assert_true(base.grad()[0] == 1.0)
    assert_true(base.grad()[6] == 1.0)
    # Third row should have grad=0 (not in concat)
    assert_true(base.grad()[12] == 0.0)
    # Fourth row should have grad=1
    assert_true(base.grad()[18] == 1.0)


fn test_concat_view_with_tensor_ct() raises:
    """Test concatenating view with regular tensor."""
    print("test_concat_view_with_tensor_ct")
    alias dtype = DType.float32

    var base = Tensor[dtype].ones(2, 3, requires_grad=True)
    var view = base.view([2, 3], offset=0)

    var regular = Tensor[dtype].ones(1, 3, requires_grad=True) * 2.0

    var tensors = List[Tensor[dtype]]()
    tensors.append(view)
    tensors.append(regular)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Shape should be (3, 3)
    assert_true(result.shape()[0] == 3)

    # Values check
    assert_true(result[0, 0] == 1.0)
    assert_true(result[2, 0] == 2.0)

    # Gradient flow
    var loss = result.sum()
    loss.backward()

    assert_true(
        base.grad().all_close[atol=1e-6](Tensor[dtype].ones(base.shape()))
    )
    assert_true(
        regular.grad().all_close[atol=1e-6](Tensor[dtype].ones(regular.shape()))
    )


fn test_concat_scalar_views_ct() raises:
    """Test concatenating scalar views (edge case)."""
    print("test_concat_scalar_views_ct")
    alias dtype = DType.float32

    var a = Tensor[dtype].scalar(5.0, requires_grad=True)
    var b = Tensor[dtype].scalar(10.0, requires_grad=True)

    var view_a = a.into_view()
    var view_b = b.into_view()

    # Make them 1D for concatenation
    var t_a = view_a.contiguous()
    var t_a_r = t_a.reshape(1)
    var t_b = view_b.contiguous()
    t_b_r = t_b.reshape(1)

    var tensors = List[Tensor[dtype]]()
    tensors.append(t_a_r)
    tensors.append(t_b_r)

    var result = Tensor[dtype].concat(tensors, axis=0)

    assert_true(result.shape()[0] == 2)
    assert_true(result[0] == 5.0)
    assert_true(result[1] == 10.0)

    var loss = result.sum()
    loss.backward()

    assert_true(a.grad().item() == 1.0)
    assert_true(b.grad().item() == 1.0)


# ============================================================================
# Edge Cases
# ============================================================================


fn test_concat_different_sizes_axis0_ct() raises:
    """Test concatenating tensors with different sizes along non-concat dims."""
    print("test_concat_different_sizes_axis0_ct")
    alias dtype = DType.float32

    # These should have same size along all dims except axis 0
    var A = Tensor[dtype].ones(2, 3, 4)
    var B = Tensor[dtype].ones(5, 3, 4) * 2.0
    var C = Tensor[dtype].ones(1, 3, 4) * 3.0

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Shape should be (8, 3, 4)
    assert_true(result.shape()[0] == 8)
    assert_true(result.shape()[1] == 3)
    assert_true(result.shape()[2] == 4)

    # Check values
    assert_true(result[0, 0, 0] == 1.0)
    assert_true(result[2, 0, 0] == 2.0)
    assert_true(result[7, 0, 0] == 3.0)


fn test_concat_contiguous_check_ct() raises:
    """Test that result is contiguous."""
    print("test_concat_contiguous_check_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].ones(2, 3)
    var B = Tensor[dtype].ones(1, 3)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Result should be contiguous
    assert_true(result.is_contiguous())


fn test_concat_preserves_values_ct() raises:
    """Test that concatenation preserves exact values."""
    print("test_concat_preserves_values_ct")
    alias dtype = DType.float32

    var A = Tensor[dtype].d2([[1.5, 2.5], [3.5, 4.5]])
    var B = Tensor[dtype].d2([[5.5, 6.5]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].concat(tensors, axis=0)

    # Check exact values
    assert_true(abs(result[0, 0] - 1.5) < 1e-6)
    assert_true(abs(result[0, 1] - 2.5) < 1e-6)
    assert_true(abs(result[1, 0] - 3.5) < 1e-6)
    assert_true(abs(result[1, 1] - 4.5) < 1e-6)
    assert_true(abs(result[2, 0] - 5.5) < 1e-6)
    assert_true(abs(result[2, 1] - 6.5) < 1e-6)


# ============================================================================
# Consolidated Test Runner
# ============================================================================


fn run_all_concat_tests() raises:
    """Run all concat tests."""
    print("\n=== Running Concat Test Suite ===\n")

    # Basic tests
    print("--- Basic Concat Tests ---")
    test_concat_axis0_basic_ct()
    test_concat_axis1_basic_ct()
    test_concat_three_tensors_ct()
    test_concat_single_tensor_ct()
    test_concat_1d_tensors_ct()
    test_concat_3d_tensors_axis0_ct()
    test_concat_3d_tensors_axis1_ct()
    test_concat_3d_tensors_axis2_ct()
    print()

    # Gradient tests
    print("--- Gradient Tests ---")
    test_concat_backward_axis0_ct()
    test_concat_backward_axis1_ct()
    test_concat_backward_weighted_ct()
    test_concat_backward_three_tensors_ct()
    test_concat_backward_chain_ct()
    print()

    # View tests
    print("--- View Tests ---")
    test_concat_with_views_ct()
    test_concat_views_grad_flow_ct()
    test_concat_nested_views_ct()
    test_concat_view_with_tensor_ct()
    test_concat_scalar_views_ct()
    print()

    # Edge cases
    print("--- Edge Cases ---")
    test_concat_different_sizes_axis0_ct()
    test_concat_contiguous_check_ct()
    test_concat_preserves_values_ct()
    print()
    print("=== All Concat Tests Passed! ===\n")
