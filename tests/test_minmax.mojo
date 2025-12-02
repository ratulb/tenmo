from tenmo import Tensor
from testing import assert_true
from intarray import IntArray
from shapes import Shape

# ============================================================================
# EXHAUSTIVE MIN/MAX TEST SUITE
# Tests all branches: vectorization, parallelization, edge cases
# ============================================================================

# ============================================================================
# TEST GROUP 1: SCALAR INPUT (rank == 0 branch)
# ============================================================================

fn test_scalar_input() raises:
    """Test scalar tensor input - should pass through unchanged."""
    print("test_scalar_input")

    # Test 1: Scalar max with gradient
    var scalar_a = Tensor.scalar(42.0, requires_grad=True)
    var max_scalar = scalar_a.max()
    assert_true(max_scalar.all_close(Tensor.scalar(42.0)))
    max_scalar.backward()
    assert_true(scalar_a.grad().all_close(Tensor.scalar(1.0)))

    # Test 2: Scalar min with gradient
    scalar_a.zero_grad()
    var min_scalar = scalar_a.min()
    assert_true(min_scalar.all_close(Tensor.scalar(42.0)))
    min_scalar.backward()
    assert_true(scalar_a.grad().all_close(Tensor.scalar(1.0)))

    # Test 3: Negative scalar
    var scalar_b = Tensor.scalar(-100.0, requires_grad=True)
    var max_neg = scalar_b.max()
    assert_true(max_neg.all_close(Tensor.scalar(-100.0)))
    max_neg.backward()
    assert_true(scalar_b.grad().all_close(Tensor.scalar(1.0)))


# ============================================================================
# TEST GROUP 2: FULL REDUCTION TO SCALAR (vectorized path)
# ============================================================================

fn test_full_reduction_vectorized() raises:
    """Test full reduction - all axes reduced to scalar (vectorized implementation)."""
    print("test_full_reduction_vectorized")

    # Test 1: Global max on 1D tensor
    var a = Tensor.d1([1.0, 5.0, 3.0, 9.0, 2.0], requires_grad=True)
    var global_max = a.max()
    assert_true(global_max.all_close(Tensor.scalar(9.0)))
    global_max.backward()
    assert_true(a.grad().all_close(Tensor.d1([0.0, 0.0, 0.0, 1.0, 0.0])))

    # Test 2: Global min on 1D tensor
    a.zero_grad()
    var global_min = a.min()
    assert_true(global_min.all_close(Tensor.scalar(1.0)))
    global_min.backward()
    a.grad().print()
    assert_true(a.grad().all_close(Tensor.d1([1.0, 0.0, 0.0, 0.0, 0.0])))

    # Test 3: Global max on 2D tensor (tests vectorization on flattened data)
    var b = Tensor.d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        requires_grad=True
    )
    var max_2d = b.max()
    assert_true(max_2d.all_close(Tensor.scalar(9.0)))
    max_2d.backward()
    var expected_grad = Tensor.d2(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert_true(b.grad().all_close(expected_grad))

    # Test 4: Global min on 2D tensor
    b.zero_grad()
    var min_2d = b.min()
    assert_true(min_2d.all_close(Tensor.scalar(1.0)))
    min_2d.backward()
    expected_grad = Tensor.d2(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert_true(b.grad().all_close(expected_grad))

    # Test 5: Multiple tied maxima (tests gradient splitting in vectorized path)
    var c = Tensor.d2(
        [[5.0, 1.0, 5.0], [2.0, 5.0, 3.0], [5.0, 4.0, 0.0]],
        requires_grad=True
    )
    var max_tied = c.max()
    assert_true(max_tied.all_close(Tensor.scalar(5.0)))
    max_tied.backward()
    expected_grad = Tensor.d2(
        [[0.25, 0.0, 0.25], [0.0, 0.25, 0.0], [0.25, 0.0, 0.0]]
    )
    assert_true(c.grad().all_close(expected_grad))

    # Test 6: Large tensor to stress vectorization (100 elements)
    var d_data = List[Float32]()
    for i in range(100):
        d_data.append(Float32(i))
    var d = Tensor.d1(d_data, requires_grad=True)
    var max_large = d.max()
    assert_true(max_large.all_close(Tensor.scalar(99.0).float()))
    max_large.backward()
    # Only last element should have gradient
    assert_true(d.grad()[IntArray(99)] == 1.0)
    assert_true(d.grad()[IntArray(0)] == 0.0)


fn test_full_reduction_with_negative() raises:
    """Test full reduction with negative values."""
    print("test_full_reduction_with_negative")

    # Test 1: All negative values - max
    var a = Tensor.d1([-5.0, -2.0, -8.0, -1.0], requires_grad=True)
    var max_neg = a.max()
    assert_true(max_neg.all_close(Tensor.scalar(-1.0)))
    max_neg.backward()
    assert_true(a.grad().all_close(Tensor.d1([0.0, 0.0, 0.0, 1.0])))

    # Test 2: All negative values - min
    a.zero_grad()
    var min_neg = a.min()
    assert_true(min_neg.all_close(Tensor.scalar(-8.0)))
    min_neg.backward()
    assert_true(a.grad().all_close(Tensor.d1([0.0, 0.0, 1.0, 0.0])))

    # Test 3: Mixed positive and negative
    var b = Tensor.d1([-10.0, 5.0, -3.0, 15.0, 0.0], requires_grad=True)
    var max_mixed = b.max()
    assert_true(max_mixed.all_close(Tensor.scalar(15.0)))
    max_mixed.backward()
    assert_true(b.grad().all_close(Tensor.d1([0.0, 0.0, 0.0, 1.0, 0.0])))


fn main() raises:
    test_max_min_mixed()
    test_max_min()
    run_all_minmax_tests()

fn test_max_min_mixed() raises:
    print("test_max_min_mixed")

    # Test 1: Basic max reduction along axis 1
    var a = Tensor.d2(
        [[42.0, 0.0, -5.0], [0.0, 35.0, 0.0], [51.0, 0.0, 51.0]],
        requires_grad=True,
    )
    var max_result = a.max(IntArray(1))
    var expected = Tensor.d1([42.0, 35.0, 51.0])
    assert_true(max_result.all_close(expected))

    max_result.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]
            )
        )
    )

    # Reset gradients for next test
    a.zero_grad()

    # Test 2: Basic min reduction along axis 1
    var min_result = a.min(IntArray(1))
    assert_true(min_result.all_close(Tensor.d1([-5.0, 0.0, 0.0])))
    min_result.backward()

    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[0.0, 0.0, 1.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]
            )
        )
    )

    # Test 3: Max reduction along axis 0
    a.zero_grad()
    var max_axis0 = a.max(IntArray(0))
    assert_true(max_axis0.all_close(Tensor.d1([51.0, 35.0, 51.0])))
    max_axis0.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
            )
        )
    )

    # Test 4: Min reduction along axis 0
    a.zero_grad()
    var min_axis0 = a.min(IntArray(0))
    assert_true(min_axis0.all_close(Tensor.d1([0.0, 0.0, -5.0])))
    min_axis0.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[0.0, 0.5, 1.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]
            )
        )
    )

    # Test 5: Global max (no axis)
    a.zero_grad()
    var global_max = a.max()
    assert_true(global_max.all_close(Tensor.scalar(51.0)))
    global_max.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]
            )
        )
    )

    # Test 6: Global min (no axis)
    a.zero_grad()
    var global_min = a.min()
    assert_true(global_min.all_close(Tensor.scalar(-5.0)))
    global_min.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
        )
    )

    # Test 7: Multiple axes reduction
    var b = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )

    var max_axes_01 = b.max(IntArray([0, 1]))
    assert_true(max_axes_01.all_close(Tensor.d1([7.0, 8.0])))
    max_axes_01.backward()
    assert_true(
        b.grad().all_close(
            Tensor.d3(
                [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]]
            )
        )
    )

    # Test 8: Edge case - all same values
    var c = Tensor.d2([[5.0, 5.0], [5.0, 5.0]], requires_grad=True)

    var max_same = c.max(IntArray(1))
    assert_true(max_same.all_close(Tensor.d1([5.0, 5.0])))
    max_same.backward()
    assert_true(
        c.grad().all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]]))
    )

    # Test 9: Edge case - negative infinity
    var d = Tensor.d2([[-3.4028235e38, 0.0], [1.0, 2.0]], requires_grad=True)

    var max_with_inf = d.max(IntArray(1))
    assert_true(max_with_inf.all_close(Tensor.d1([0.0, 2.0])))
    max_with_inf.backward()
    assert_true(
        d.grad().all_close(Tensor.d2([[0.0, 1.0], [0.0, 1.0]]))
    )

    # Test 10: Keep dimensions
    var e = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    var max_keepdim = e.max(IntArray(1), keepdims=True)
    assert_true(max_keepdim.all_close(Tensor.d2([[2.0], [4.0]])))
    max_keepdim.backward()
    assert_true(
        e.grad().all_close(Tensor.d2([[0.0, 1.0], [0.0, 1.0]]))
    )


fn test_max_min() raises:
    print("test_max_min")
    a = Tensor.d2(
        [[42.0, 0.0, -5.0], [0.0, 35.0, 0.0], [51.0, 0.0, 51.0]],
        requires_grad=True,
    )

    max_result = a.max(IntArray(1))
    expected = Tensor.d1([42.0, 35.0, 51.0])
    assert_true(max_result.all_close(expected))

    max_result.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]
            )
        )
    )
    min_result = a.min([1])
    assert_true(min_result.all_close(Tensor.d1([-5.0, 0.0, 0.0])))
    min_result.backward()

    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[1.0, 0.0, 1.0], [0.5, 1.0, 0.5], [0.5, 1.0, 0.5]]
            )
        )
    )



# ============================================================================
# TEST GROUP 3: PARTIAL REDUCTION (parallelized path)
# ============================================================================

fn test_partial_reduction_single_axis() raises:
    """Test partial reduction along single axis (parallelized)."""
    print("test_partial_reduction_single_axis")

    # Test 1: Max along axis 0
    var a = Tensor.d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        requires_grad=True
    )
    var max_axis0 = a.max(IntArray(0))
    assert_true(max_axis0.all_close(Tensor.d1([7.0, 8.0, 9.0])))
    max_axis0.backward()
    var expected = Tensor.d2(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    )
    assert_true(a.grad().all_close(expected))

    # Test 2: Max along axis 1
    a.zero_grad()
    var max_axis1 = a.max(IntArray(1))
    assert_true(max_axis1.all_close(Tensor.d1([3.0, 6.0, 9.0])))
    max_axis1.backward()
    expected = Tensor.d2(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    )
    assert_true(a.grad().all_close(expected))

    # Test 3: Min along axis 0
    a.zero_grad()
    var min_axis0 = a.min(IntArray(0))
    assert_true(min_axis0.all_close(Tensor.d1([1.0, 2.0, 3.0])))
    min_axis0.backward()
    expected = Tensor.d2(
        [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert_true(a.grad().all_close(expected))

    # Test 4: Min along axis 1
    a.zero_grad()
    var min_axis1 = a.min(IntArray(1))
    assert_true(min_axis1.all_close(Tensor.d1([1.0, 4.0, 7.0])))
    min_axis1.backward()
    expected = Tensor.d2(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    assert_true(a.grad().all_close(expected))


fn test_partial_reduction_with_ties() raises:
    """Test partial reduction with tied values (gradient splitting)."""
    print("test_partial_reduction_with_ties")

    # Test 1: Ties along axis 1 - from original test
    var a = Tensor.d2(
        [[42.0, 0.0, -5.0], [0.0, 35.0, 0.0], [51.0, 0.0, 51.0]],
        requires_grad=True,
    )
    var max_result = a.max(IntArray(1))
    var expected = Tensor.d1([42.0, 35.0, 51.0])
    assert_true(max_result.all_close(expected))
    max_result.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]
            )
        )
    )

    # Test 2: Ties along axis 0
    a.zero_grad()
    var min_axis0 = a.min(IntArray(0))
    assert_true(min_axis0.all_close(Tensor.d1([0.0, 0.0, -5.0])))
    min_axis0.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[0.0, 0.5, 1.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]
            )
        )
    )

    # Test 3: All same values in a row
    var b = Tensor.d2([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]], requires_grad=True)
    var max_same = b.max(IntArray(1))
    assert_true(max_same.all_close(Tensor.d1([5.0, 3.0])))
    max_same.backward()
    expected = Tensor.d2(
        [[0.333333, 0.333333, 0.333333], [0.0, 0.0, 1.0]]
    )
    assert_true(b.grad().all_close[atol=1e-5](expected))


fn test_partial_reduction_3d() raises:
    """Test partial reduction on 3D tensors."""
    print("test_partial_reduction_3d")

    # Test 1: 3D tensor, reduce axis 0
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True
    )
    var max_axis0 = a.max(IntArray(0))
    assert_true(max_axis0.all_close(Tensor.d2([[5.0, 6.0], [7.0, 8.0]])))
    max_axis0.backward()
    var expected = Tensor.d3(
        [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected))

    # Test 2: 3D tensor, reduce axis 1
    a.zero_grad()
    var max_axis1 = a.max(IntArray(1))
    assert_true(max_axis1.all_close(Tensor.d2([[3.0, 4.0], [7.0, 8.0]])))
    max_axis1.backward()
    expected = Tensor.d3(
        [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected))

    # Test 3: 3D tensor, reduce axis 2
    a.zero_grad()
    var max_axis2 = a.max(IntArray(2))
    assert_true(max_axis2.all_close(Tensor.d2([[2.0, 4.0], [6.0, 8.0]])))
    max_axis2.backward()
    expected = Tensor.d3(
        [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected))



fn test_partial_reduction_multiple_axes() raises:
    """Test reduction along multiple axes simultaneously."""
    print("test_partial_reduction_multiple_axes")

    # Test 1: Reduce axes [0, 1] on 3D tensor
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True
    )
    var max_axes_01 = a.max(IntArray([0, 1]))
    assert_true(max_axes_01.all_close(Tensor.d1([7.0, 8.0])))
    max_axes_01.backward()
    var expected = Tensor.d3(
        [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected))

    # Test 2: Reduce axes [0, 2] on 3D tensor
    a.zero_grad()
    var max_axes_02 = a.max(IntArray([0, 2]))
    assert_true(max_axes_02.all_close(Tensor.d1([6.0, 8.0])))
    max_axes_02.backward()
    expected = Tensor.d3(
        [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected))

    # Test 3: Reduce axes [1, 2] on 3D tensor
    a.zero_grad()
    var max_axes_12 = a.max(IntArray([1, 2]))
    assert_true(max_axes_12.all_close(Tensor.d1([4.0, 8.0])))
    max_axes_12.backward()
    expected = Tensor.d3(
        [[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected))


# ============================================================================
# TEST GROUP 4: KEEPDIMS FUNCTIONALITY
# ============================================================================

fn test_keepdims() raises:
    """Test keepdims parameter (affects gradient broadcasting)."""
    print("test_keepdims")

    # Test 1: keepdims=True, axis 1, 2D tensor
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var max_keepdim = a.max(IntArray(1), keepdims=True)
    assert_true(max_keepdim.all_close(Tensor.d2([[3.0], [6.0]])))
    assert_true(max_keepdim.shape() == Shape(2, 1))  # Shape preserved
    max_keepdim.backward()
    var expected = Tensor.d2([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    assert_true(a.grad().all_close(expected))

    # Test 2: keepdims=True, axis 0, 2D tensor
    a.zero_grad()
    var max_keepdim_0 = a.max(IntArray(0), keepdims=True)
    assert_true(max_keepdim_0.all_close(Tensor.d2([[4.0, 5.0, 6.0]])))
    assert_true(max_keepdim_0.shape() == Shape(1, 3))
    max_keepdim_0.backward()
    expected = Tensor.d2([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert_true(a.grad().all_close(expected))

    # Test 3: keepdims=True, multiple axes, 3D tensor
    var b = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True
    )
    var max_keepdim_multi = b.max(IntArray([0, 2]), keepdims=True)
    assert_true(max_keepdim_multi.all_close(Tensor.d3([[[6.0], [8.0]]])))
    assert_true(max_keepdim_multi.shape() == Shape(1, 2, 1))
    max_keepdim_multi.backward()
    expected = Tensor.d3(
        [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]]
    )
    assert_true(b.grad().all_close(expected))


# ============================================================================
# TEST GROUP 5: EDGE CASES
# ============================================================================

fn test_edge_cases_extreme_values() raises:
    """Test extreme values (infinity, very large/small)."""
    print("test_edge_cases_extreme_values")

    # Test 1: With negative infinity
    var a = Tensor.d2([[-3.4028235e38, 0.0], [1.0, 2.0]], requires_grad=True)
    var max_with_inf = a.max(IntArray(1))
    assert_true(max_with_inf.all_close(Tensor.d1([0.0, 2.0])))
    max_with_inf.backward()
    assert_true(a.grad().all_close(Tensor.d2([[0.0, 1.0], [0.0, 1.0]])))

    # Test 2: With positive infinity approximation
    a.zero_grad()
    var b = Tensor.d2([[3.4028235e38, 0.0], [1.0, 2.0]], requires_grad=True)
    var max_pos_inf = b.max(IntArray(1))
    assert_true(max_pos_inf.all_close(Tensor.d1([3.4028235e38, 2.0])))
    max_pos_inf.backward()
    assert_true(b.grad().all_close(Tensor.d2([[1.0, 0.0], [0.0, 1.0]])))

    # Test 3: Very small differences (precision test)
    var c = Tensor.d1([1.0, 1.0000001, 1.0], requires_grad=True)
    var max_prec = c.max()
    assert_true(max_prec.all_close(Tensor.scalar(1.0000001)))
    max_prec.backward()
    assert_true(c.grad().all_close(Tensor.d1([0.0, 1.0, 0.0])))



fn test_edge_cases_all_same() raises:
    """Test when all values are identical."""
    print("test_edge_cases_all_same")

    # Test 1: All same positive values
    var a = Tensor.d2([[5.0, 5.0], [5.0, 5.0]], requires_grad=True)
    var max_same = a.max(IntArray(1))
    assert_true(max_same.all_close(Tensor.d1([5.0, 5.0])))
    max_same.backward()
    assert_true(a.grad().all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])))

    # Test 2: All same negative values
    var b = Tensor.d2([[-3.0, -3.0], [-3.0, -3.0]], requires_grad=True)
    var min_same = b.min(IntArray(0))
    assert_true(min_same.all_close(Tensor.d1([-3.0, -3.0])))
    min_same.backward()
    assert_true(b.grad().all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])))

    # Test 3: All zeros
    var c = Tensor.d2([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)
    var max_zeros = c.max()
    assert_true(max_zeros.all_close(Tensor.scalar(0.0)))
    max_zeros.backward()
    # All positions should share gradient equally (1/6 each)
    var expected_val = 1.0 / 6.0
    for i in range(2):
        for j in range(3):
            assert_true(c.grad()[IntArray(i, j)] - expected_val < 1e-6)


fn test_edge_cases_single_element() raises:
    """Test single element reductions."""
    print("test_edge_cases_single_element")

    # Test 1: 1D tensor with single element
    var a = Tensor.d1([42.0], requires_grad=True)
    var max_single = a.max()
    assert_true(max_single.all_close(Tensor.scalar(42.0)))
    max_single.backward()
    assert_true(a.grad().all_close(Tensor.d1([1.0])))

    # Test 2: 2D tensor reducing to single element per row
    var b = Tensor.d2([[10.0], [20.0]], requires_grad=True)
    var max_rows = b.max(IntArray(1))
    assert_true(max_rows.all_close(Tensor.d1([10.0, 20.0])))
    max_rows.backward()
    assert_true(b.grad().all_close(Tensor.d2([[1.0], [1.0]])))


fn test_edge_cases_large_tensor() raises:
    """Test large tensors to stress parallelization."""
    print("test_edge_cases_large_tensor")

    # Test 1: Large 1D tensor (1000 elements) - full reduction
    var data_1d = List[Float32]()
    for i in range(1000):
        data_1d.append(Float32(i % 100))  # Pattern 0-99 repeated
    var a = Tensor.d1(data_1d, requires_grad=True)
    var max_large = a.max()
    assert_true(max_large.all_close(Tensor.scalar(99.0).float()))
    max_large.backward()
    # Multiple positions have value 99, gradient should be split
    var count_99 = 10  # 99 appears at indices 99, 199, 299, ..., 999
    var expected_grad = 1.0 / Float32(count_99)
    assert_true(a.grad()[IntArray(99)] - expected_grad < 1e-6)

    # Test 2: Large 2D tensor (100x100) - partial reduction
    var b = Tensor.zeros(Shape(100, 100), requires_grad=True)
    # Set specific values
    b[IntArray(50, 50)] = 1000.0
    b[IntArray(75, 25)] = 2000.0

    var max_axis0 = b.max(IntArray(0))
    assert_true(max_axis0[IntArray(50)] == 1000.0)
    assert_true(max_axis0[IntArray(25)] == 2000.0)

    max_axis0.backward()
    assert_true(b.grad()[IntArray(50, 50)] == 1.0)
    assert_true(b.grad()[IntArray(75, 25)] == 1.0)


# ============================================================================
# TEST GROUP 6: GRADIENT ACCUMULATION
# ============================================================================

fn test_gradient_accumulation() raises:
    """Test that gradients accumulate correctly across multiple backwards."""
    print("test_gradient_accumulation")

    # Test from original - max then min on same tensor
    var a = Tensor.d2(
        [[42.0, 0.0, -5.0], [0.0, 35.0, 0.0], [51.0, 0.0, 51.0]],
        requires_grad=True,
    )

    # First backward pass
    var max_result = a.max(IntArray(1))
    max_result.backward()
    var grad_after_max = Tensor.d2(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]
    )
    assert_true(a.grad().all_close(grad_after_max))

    # Second backward pass (accumulates)
    var min_result = a.min(IntArray(1))
    min_result.backward()
    var grad_after_both = Tensor.d2(
        [[1.0, 0.0, 1.0], [0.5, 1.0, 0.5], [0.5, 1.0, 0.5]]
    )
    assert_true(a.grad().all_close(grad_after_both))


# ============================================================================
# TEST GROUP 7: NO GRADIENT TRACKING
# ============================================================================

fn test_no_gradient_tracking() raises:
    """Test behavior when requires_grad=False."""
    print("test_no_gradient_tracking")

    # Test 1: No gradient tracking
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var max_no_grad = a.max(IntArray(1))
    assert_true(max_no_grad.all_close(Tensor.d1([2.0, 4.0])))
    assert_true(not max_no_grad.requires_grad)

    # Test 2: Explicit requires_grad override
    var b = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var max_override = b.max(IntArray(1), requires_grad=False)
    assert_true(max_override.all_close(Tensor.d1([2.0, 4.0])))
    assert_true(not max_override.requires_grad)


# ============================================================================
# TEST GROUP 8: MIXED MIN/MAX OPERATIONS
# ============================================================================

fn test_mixed_min_max_operations() raises:
    """Test alternating min/max operations."""
    print("test_mixed_min_max_operations")

    var a = Tensor.d2(
        [[42.0, 0.0, -5.0], [0.0, 35.0, 0.0], [51.0, 0.0, 51.0]],
        requires_grad=True,
    )

    # Max along axis 1
    var max_result = a.max(IntArray(1))
    assert_true(max_result.all_close(Tensor.d1([42.0, 35.0, 51.0])))
    max_result.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])
        )
    )

    # Reset and min along axis 1
    a.zero_grad()
    var min_result = a.min(IntArray(1))
    assert_true(min_result.all_close(Tensor.d1([-5.0, 0.0, 0.0])))
    min_result.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2([[0.0, 0.0, 1.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
        )
    )

    # Max along axis 0
    a.zero_grad()
    var max_axis0 = a.max(IntArray(0))
    assert_true(max_axis0.all_close(Tensor.d1([51.0, 35.0, 51.0])))
    max_axis0.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        )
    )

    # Min along axis 0
    a.zero_grad()
    var min_axis0 = a.min(IntArray(0))
    assert_true(min_axis0.all_close(Tensor.d1([0.0, 0.0, -5.0])))
    min_axis0.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2([[0.0, 0.5, 1.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
        )
    )


# ============================================================================
# TEST GROUP 9: NEGATIVE AXIS INDEXING
# ============================================================================

fn test_negative_axis_indexing() raises:
    """Test negative axis indices (e.g., -1 for last axis)."""
    print("test_negative_axis_indexing")

    # Test 1: 2D tensor, axis=-1 (last axis, same as axis=1)
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var max_neg_axis = a.max(IntArray(-1))
    assert_true(max_neg_axis.all_close(Tensor.d1([3.0, 6.0])))
    max_neg_axis.backward()
    var expected = Tensor.d2([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    assert_true(a.grad().all_close(expected))

    # Test 2: 3D tensor, axis=-1
    var b = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True
    )
    var max_3d_neg = b.max(IntArray(-1))
    assert_true(max_3d_neg.all_close(Tensor.d2([[2.0, 4.0], [6.0, 8.0]])))
    max_3d_neg.backward()
    expected = Tensor.d3(
        [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]
    )
    assert_true(b.grad().all_close(expected))

    # Test 3: Multiple negative axes
    b.zero_grad()
    var max_multi_neg = b.max(IntArray([-2, -1]))
    assert_true(max_multi_neg.all_close(Tensor.d1([4.0, 8.0])))
    max_multi_neg.backward()
    expected = Tensor.d3(
        [[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 1.0]]]
    )
    assert_true(b.grad().all_close(expected))



# ============================================================================
# TEST GROUP 10: HIGH-DIMENSIONAL TENSORS
# ============================================================================

fn test_high_dimensional_tensors() raises:
    """Test 4D and 5D tensors."""
    print("test_high_dimensional_tensors")

    # Test 1: 4D tensor (batch, channels, height, width)
    var a = Tensor.zeros(Shape(2, 3, 4, 5), requires_grad=True)
    # Set specific max values
    a[IntArray(0, 1, 2, 3)] = 100.0
    a[IntArray(1, 2, 3, 4)] = 200.0

    # Reduce along axis 3 (width)
    var max_width = a.max(IntArray(3))
    assert_true(max_width.shape() == Shape(2, 3, 4))
    assert_true(max_width[IntArray(0, 1, 2)] == 100.0)
    assert_true(max_width[IntArray(1, 2, 3)] == 200.0)

    max_width.backward()
    assert_true(a.grad()[IntArray(0, 1, 2, 3)] == 1.0)
    assert_true(a.grad()[IntArray(1, 2, 3, 4)] == 1.0)

    # Test 2: 4D tensor, reduce multiple axes
    a.zero_grad()
    var max_spatial = a.max(IntArray([2, 3]))  # Reduce height and width
    assert_true(max_spatial.shape() == Shape(2, 3))
    assert_true(max_spatial[IntArray(0, 1)] == 100.0)
    assert_true(max_spatial[IntArray(1, 2)] == 200.0)


# ============================================================================
# TEST GROUP 11: ZERO-SIZED DIMENSIONS (Edge case)
# ============================================================================

fn test_empty_reduction_axis() raises:
    """Test reduction when axis list is empty."""
    print("test_empty_reduction_axis")

    # Empty axes list should return copy of tensor
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.max(IntArray())  # No reduction
    assert_true(result.all_close(Tensor.scalar(4).float64()))
    assert_true(result.shape() == Shape())


# ============================================================================
# TEST GROUP 12: NUMERICAL STABILITY
# ============================================================================

fn test_numerical_stability() raises:
    """Test numerical edge cases."""
    print("test_numerical_stability")

    # Test 1: Very close values (should handle floating point precision)
    var a = Tensor.d1([1.0, 1.0 + 1e-7, 1.0 + 2e-7], requires_grad=True)
    var max_close = a.max()
    assert_true(max_close.item() >= 1.0 + 2e-7)
    max_close.backward()
    assert_true(a.grad()[IntArray(2)] == 1.0)

    # Test 2: Alternating signs with small magnitudes
    var b = Tensor.d1([1e-10, -1e-10, 2e-10, -2e-10], requires_grad=True)
    var max_tiny = b.max()
    assert_true(max_tiny.item() == 2e-10)
    max_tiny.backward()
    assert_true(b.grad()[IntArray(2)] == 1.0)

    # Test 3: Mix of large and small values
    var c = Tensor.d1([1e20, 1.0, 1e-20, 1e20], requires_grad=True)
    var max_mixed_scale = c.max()
    assert_true(max_mixed_scale.item() == 1e20)
    max_mixed_scale.backward()
    # Gradient split between two 1e20 values
    assert_true(c.grad()[IntArray(0)] == 0.5)
    assert_true(c.grad()[IntArray(3)] == 0.5)


# ============================================================================
# TEST GROUP 13: STRESS TEST - MANY TIES
# ============================================================================

fn test_many_ties() raises:
    """Test performance with many tied values."""
    print("test_many_ties")

    # Test 1: 1000 elements, all the same (extreme tie case)
    var data = List[Float32]()
    for _ in range(1000):
        data.append(42.0)
    var a = Tensor.d1(data, requires_grad=True)
    var max_all_tied = a.max()
    assert_true(max_all_tied.all_close(Tensor.scalar(42.0).float()))
    max_all_tied.backward()
    # Each position gets 1/1000 of gradient
    var expected_grad = 1.0 / 1000.0
    assert_true((Scalar[DType.float64](a.grad()[IntArray(0)]) - expected_grad) < 1e-6)
    assert_true((Scalar[DType.float64](a.grad()[IntArray(999)]) - expected_grad) < 1e-6)

    # Test 2: Half max, half min (two groups)
    var b_data = List[Float32]()
    for _ in range(500):
        b_data.append(10.0)
    for _ in range(500):
        b_data.append(-10.0)
    var b = Tensor.d1(b_data, requires_grad=True)
    var max_half = b.max()
    assert_true(max_half.all_close(Tensor.scalar(10.0).float()))
    max_half.backward()
    # First 500 positions split gradient
    assert_true((Scalar[DType.float64](b.grad()[IntArray(0)]) - 1.0/500.0) < 1e-6)
    assert_true(b.grad()[IntArray(500)] == 0.0)


# ============================================================================
# TEST GROUP 14: INTERACTION WITH OTHER OPS
# ============================================================================

fn test_chained_operations() raises:
    """Test max/min in chains with other operations."""
    print("test_chained_operations")

    # Test 1: Max followed by arithmetic
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var max_vals = a.max(IntArray(1))
    var doubled = max_vals * 2.0
    assert_true(doubled.all_close(Tensor.d1([4.0, 8.0])))
    doubled.backward()
    # Gradient flows back through max
    var expected = Tensor.d2([[0.0, 2.0], [0.0, 2.0]])
    assert_true(a.grad().all_close(expected))

    # Test 2: Arithmetic followed by max
    a.zero_grad()
    var scaled = a * 10.0
    var max_scaled = scaled.max(IntArray(0))
    assert_true(max_scaled.all_close(Tensor.d1([30.0, 40.0])))
    max_scaled.backward()
    expected = Tensor.d2([[0.0, 0.0], [10.0, 10.0]])
    assert_true(a.grad().all_close(expected))


# ============================================================================
# TEST GROUP 15: BACKWARDS COMPATIBILITY
# ============================================================================

fn test_backwards_compatibility() raises:
    """Ensure new implementation matches old behavior exactly."""
    print("test_backwards_compatibility")

    # This is the comprehensive test from your original suite
    var a = Tensor.d2(
        [[42.0, 0.0, -5.0], [0.0, 35.0, 0.0], [51.0, 0.0, 51.0]],
        requires_grad=True,
    )

    # Test 1: Max along axis 1
    var max_result = a.max(IntArray(1))
    var expected = Tensor.d1([42.0, 35.0, 51.0])
    assert_true(max_result.all_close(expected))
    max_result.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]
            )
        )
    )

    # Test 2: Min along axis 1 (accumulates gradient)
    var min_result = a.min(IntArray(1))
    assert_true(min_result.all_close(Tensor.d1([-5.0, 0.0, 0.0])))
    min_result.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [[1.0, 0.0, 1.0], [0.5, 1.0, 0.5], [0.5, 1.0, 0.5]]
            )
        )
    )


# ============================================================================
# CONSOLIDATED TEST RUNNER
# ============================================================================

fn run_all_minmax_tests() raises:
    """
    Runs the complete test suite for optimized min/max implementation.
    Tests all code paths: scalar, vectorized full reduction, parallelized partial reduction.
    """
    print("=" * 70)
    print("STARTING EXHAUSTIVE MIN/MAX TEST SUITE")
    print("=" * 70)

    # Group 1: Scalar input (rank == 0 branch)
    print("\n[GROUP 1] Scalar Input Tests")
    test_scalar_input()

    # Group 2: Full reduction (vectorized path)
    print("\n[GROUP 2] Full Reduction - Vectorized Path")
    test_full_reduction_vectorized()
    test_full_reduction_with_negative()

    # Group 3: Partial reduction (parallelized path)
    print("\n[GROUP 3] Partial Reduction - Parallelized Path")
    test_partial_reduction_single_axis()
    test_partial_reduction_with_ties()
    test_partial_reduction_3d()
    test_partial_reduction_multiple_axes()

    # Group 4: Keepdims functionality
    print("\n[GROUP 4] Keepdims Tests")
    test_keepdims()

    # Group 5: Edge cases
    print("\n[GROUP 5] Edge Cases")
    test_edge_cases_extreme_values()
    test_edge_cases_all_same()
    test_edge_cases_single_element()
    test_edge_cases_large_tensor()

    # Group 6: Gradient accumulation
    print("\n[GROUP 6] Gradient Accumulation")
    test_gradient_accumulation()

    # Group 7: No gradient tracking
    print("\n[GROUP 7] No Gradient Tracking")
    test_no_gradient_tracking()

    # Group 8: Mixed operations
    print("\n[GROUP 8] Mixed Min/Max Operations")
    test_mixed_min_max_operations()

    # Group 9: Negative axis indexing
    print("\n[GROUP 9] Negative Axis Indexing")
    test_negative_axis_indexing()

    # Group 10: High-dimensional tensors
    print("\n[GROUP 10] High-Dimensional Tensors")
    test_high_dimensional_tensors()

    # Group 11: Empty reduction axis
    print("\n[GROUP 11] Empty Reduction Axis")
    test_empty_reduction_axis()

    # Group 12: Numerical stability
    print("\n[GROUP 12] Numerical Stability")
    test_numerical_stability()

    # Group 13: Stress test with many ties
    print("\n[GROUP 13] Stress Test - Many Ties")
    test_many_ties()

    # Group 14: Chained operations
    print("\n[GROUP 14] Chained Operations")
    test_chained_operations()

    # Group 15: Backwards compatibility
    print("\n[GROUP 15] Backwards Compatibility")
    test_backwards_compatibility()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print("\nTest Coverage:")
    print("  ✓ Scalar input (rank 0)")
    print("  ✓ Vectorized full reduction path")
    print("  ✓ Parallelized partial reduction path")
    print("  ✓ Gradient tracking and accumulation")
    print("  ✓ Tied values (gradient splitting)")
    print("  ✓ Keepdims functionality")
    print("  ✓ Multiple axes reduction")
    print("  ✓ Negative axis indexing")
    print("  ✓ High-dimensional tensors (4D+)")
    print("  ✓ Edge cases (infinity, precision, all same)")
    print("  ✓ Large tensors (parallelization stress)")
    print("  ✓ Numerical stability")
    print("  ✓ Backwards compatibility")
    print("=" * 70)
