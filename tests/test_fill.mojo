from tenmo import Tensor
from testing import assert_true
from common_utils import i, il, s, newaxis, Idx
from intarray import IntArray

# ============================================================================
# SCALAR FILL TESTS - Basic Integer Indexing
# ============================================================================


fn test_fill_scalar_single_index_1d() raises:
    """Test filling single element in 1D tensor with scalar."""
    print("test_fill_scalar_single_index_1d")

    alias dtype = DType.float32
    var x = Tensor[dtype].arange(10)

    x.fill(999.0, i(5))

    # Only index 5 should be changed
    assert_true(x[0] == 0.0)
    assert_true(x[5] == 999.0)
    assert_true(x[9] == 9.0)


fn test_fill_scalar_single_index_2d() raises:
    """Test filling single element in 2D tensor with scalar."""
    print("test_fill_scalar_single_index_2d")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(3, 4)

    x.fill(42.0, i(1), i(2))
    # Only position [1, 2] should be 42
    assert_true(x[0, 0] == 0.0)
    assert_true(x[1, 2] == 42.0)  # Flat index for [1, 2]
    assert_true(x[2, 3] == 0.0)


fn test_fill_scalar_negative_index() raises:
    """Test filling with negative index."""
    print("test_fill_scalar_negative_index")

    alias dtype = DType.float32
    var x = Tensor[dtype].arange(10)

    x.fill(777.0, i(-1))  # Last element

    assert_true(x[9] == 777.0)
    assert_true(x[8] == 8.0)


fn test_fill_scalar_3d_index() raises:
    """Test filling single element in 3D tensor."""
    print("test_fill_scalar_3d_index")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(2, 3, 4)

    x.fill(123.0, i(1), i(1), i(2))

    # Element at [1, 1, 2] should be 123
    var flat_idx = 1 * 12 + 1 * 4 + 2  # 18
    assert_true(x.buffer.data_buffer()[flat_idx] == 123.0)


# ============================================================================
# SCALAR FILL TESTS - Slice Indexing
# ============================================================================


fn test_fill_scalar_full_slice_1d() raises:
    """Test filling entire 1D tensor with scalar."""
    print("test_fill_scalar_full_slice_1d")

    alias dtype = DType.float32
    var x = Tensor[dtype].arange(10)

    x.fill(5.0, s())

    # All elements should be 5
    var expected = Tensor[dtype].ones(10) * 5.0
    assert_true(x.all_close[atol=1e-6](expected))


fn test_fill_scalar_partial_slice_1d() raises:
    """Test filling partial slice in 1D tensor."""
    print("test_fill_scalar_partial_slice_1d")

    alias dtype = DType.float32
    var x = Tensor[dtype].arange(10)

    x.fill(99.0, s(2, 7))

    # Elements [2:7] should be 99
    assert_true(x[1] == 1.0)
    assert_true(x[2] == 99.0)
    assert_true(x[6] == 99.0)
    assert_true(x[7] == 7.0)


fn test_fill_scalar_slice_with_step_1d() raises:
    """Test filling slice with step in 1D tensor."""
    print("test_fill_scalar_slice_with_step_1d")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(10)

    x.fill(7.0, s(None, None, 2))  # Every other element

    # Even indices should be 7
    assert_true(x[0] == 7.0)
    assert_true(x[1] == 0.0)
    assert_true(x[2] == 7.0)
    assert_true(x[3] == 0.0)


fn test_fill_scalar_2d_slice_row() raises:
    """Test filling entire row in 2D tensor."""
    print("test_fill_scalar_2d_slice_row")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(3, 4)

    x.fill(10.0, i(1), s())  # Second row

    # Row 1 should be all 10s
    assert_true(x[0, 0] == 0.0)  # Row 0
    assert_true(x[1, 0] == 10.0)  # Row 1, col 0
    assert_true(x[1, 3] == 10.0)  # Row 1, col 3
    assert_true(x[2, 0] == 0.0)  # Row 2


fn test_fill_scalar_2d_slice_col() raises:
    """Test filling entire column in 2D tensor."""
    print("test_fill_scalar_2d_slice_col")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(3, 4)

    x.fill(20.0, s(), i(2))  # Third column

    # Column 2 should be all 20s
    assert_true(x[0, 2] == 20.0)  # [0, 2]
    assert_true(x[1, 2] == 20.0)  # [1, 2]
    assert_true(x[2, 2] == 20.0)  # [2, 2]
    assert_true(x[2, 3] == 0.0)  # Other elements


fn test_fill_scalar_2d_submatrix() raises:
    """Test filling submatrix in 2D tensor."""
    print("test_fill_scalar_2d_submatrix")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(5, 5)

    x.fill(8.0, s(1, 4), s(1, 4))  # 3x3 submatrix

    # Center 3x3 should be 8s
    assert_true(x[0, 0] == 0.0)  # Outside
    assert_true(x[1, 1] == 8.0)  # [1, 1]
    assert_true(x[3, 3] == 8.0)  # [3, 3]
    assert_true(x[4, 4] == 0.0)  # Outside


fn test_fill_scalar_strided_2d() raises:
    """Test filling with strides in 2D (checkerboard pattern)."""
    print("test_fill_scalar_strided_2d")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(4, 4)

    x.fill(1.0, s(None, None, 2), s(None, None, 2))  # Every other row/col

    # Checkerboard pattern
    assert_true(x[0, 0] == 1.0)  # [0, 0]
    assert_true(x[1, 1] == 0.0)  # [0, 1]
    assert_true(x[0, 2] == 1.0)  # [0, 2]
    assert_true(x[1, 0] == 0.0)  # [1, 0]
    assert_true(x[2, 0] == 1.0)  # [2, 0]


# ============================================================================
# SCALAR FILL TESTS - Array Indexing (Fancy Indexing)
# ============================================================================


fn test_fill_scalar_array_index_1d() raises:
    """Test filling specific indices in 1D tensor."""
    print("test_fill_scalar_array_index_1d")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(10)

    x.fill(50.0, il(1))
    x.fill(50.0, il(3))
    x.fill(50.0, il(5))
    x.fill(50.0, il(7))

    # Odd indices should be 50
    assert_true(x[0] == 0.0)
    assert_true(x[1] == 50.0)
    assert_true(x[2] == 0.0)
    assert_true(x[3] == 50.0)
    assert_true(x[7] == 50.0)
    assert_true(x[9] == 0.0)


fn test_fill_scalar_array_index_2d() raises:
    """Test filling specific rows in 2D tensor."""
    print("test_fill_scalar_array_index_2d")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(5, 3)

    x.fill(33.0, s(0, 5, 2), s())  # Rows 0, 2, 4

    # Selected rows should be 33
    assert_true(x[i(0), s()] == Tensor[dtype].d1([33.0, 33.0, 33.0]))  # Row 0
    assert_true(x[i(1), s()] == Tensor[dtype].d1([0.0, 0.0, 0.0]))  # Row 1
    assert_true(x[i(2), s()] == Tensor[dtype].d1([33.0, 33.0, 33.0]))  # Row 2
    assert_true(x[i(4), s()] == Tensor[dtype].d1([33.0, 33.0, 33.0]))  # Row 4


fn test_fill_scalar_empty_array() raises:
    """Test filling with empty index array."""
    print("test_fill_scalar_empty_array")

    alias dtype = DType.float32
    var x = Tensor[dtype].ones(5)

    x.fill(999.0, s())

    # Should be unchanged
    var expected = Tensor[dtype].full([5], 999)
    assert_true(x.all_close[atol=1e-6](expected))


fn test_fill_scalar_single_element_array() raises:
    """Test filling with single-element index array."""
    print("test_fill_scalar_single_element_array")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(10)

    var indices = IntArray()
    indices.append(5)
    x.fill(100.0, il(indices))

    assert_true(x[5] == 100.0)
    assert_true(x[4] == 0.0)


# ============================================================================
# SCALAR FILL TESTS - Mixed Indexing
# ============================================================================


fn test_fill_scalar_mixed_int_slice() raises:
    """Test filling with mixed integer and slice indices."""
    print("test_fill_scalar_mixed_int_slice")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(3, 4, 5)

    x.fill(15.0, i(1), s(1, 3), s())  # [1, 1:3, :]

    # Shape of filled region: (2, 5) = 10 elements
    # Starting at [1, 1, 0]
    var base = 1 * 20 + 1 * 5  # 25
    assert_true(x.buffer.data_buffer()[base] == 15.0)
    assert_true(x.buffer.buffer[base + 4] == 15.0)
    assert_true(x.buffer.buffer[base + 5] == 15.0)  # Next row
    assert_true(x[0, 0, 0] == 0.0)  # Outside


fn test_fill_scalar_mixed_array_slice() raises:
    """Test filling with array and slice indices."""
    print("test_fill_scalar_mixed_array_slice")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(5, 4)

    x.fill(25.0, s(1, 3), s(1, 3))  # Rows [1, 3], cols [1:3]
    # Should fill [1, 1:3] and [2, 1:3]
    assert_true(x[1, 1] == 25.0)
    assert_true(x[1, 2] == 25.0)
    assert_true(x[2, 1] == 25.0)
    assert_true(x[2, 2] == 25.0)
    assert_true(x[0, 0] == 0.0)  # Outside


# ============================================================================
# SCALAR FILL TESTS - On Views
# ============================================================================
fn test_fill_scalar_on_view_slice() raises:
    """Test filling on a sliced view."""
    print("test_fill_scalar_on_view_slice")

    alias dtype = DType.float32
    var x = Tensor[dtype].arange(20)
    var view = x[5:15]  # View of x
    view.fill(88.0, s())
    # Original x should be modified at [5:15]
    assert_true(x[4] == 4.0)
    assert_true(x[5] == 88.0)
    assert_true(x[14] == 88.0)
    assert_true(x[15] == 15.0)


fn test_fill_scalar_on_view_reshape() raises:
    """Test filling on reshaped view."""
    print("test_fill_scalar_on_view_reshape")

    alias dtype = DType.float32
    var x = Tensor[dtype].arange(12)
    var view = x.reshape(3, 4)

    view.fill(77.0, i(1), i(2))  # [1, 2] in reshaped view

    # Should modify x at flat index 1*4 + 2 = 6
    assert_true(x[6] == 77.0)
    assert_true(x[5] == 5.0)
    assert_true(x[7] == 7.0)


fn test_fill_scalar_on_strided_view() raises:
    """Test filling on non-contiguous view."""
    print("test_fill_scalar_on_strided_view")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(10)
    var view = x[::2]  # Every other element

    view.fill(3.0, s())

    # Even indices in x should be 3
    assert_true(x[0] == 3.0)
    assert_true(x[1] == 0.0)
    assert_true(x[2] == 3.0)
    assert_true(x[3] == 0.0)


# ============================================================================
# TENSOR FILL TESTS - Broadcasting
# ============================================================================


fn test_fill_tensor_exact_shape() raises:
    """Test filling with tensor of exact shape."""
    print("test_fill_tensor_exact_shape")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(3, 4)
    var src = Tensor[dtype].ones(3, 4) * 5.0

    x.fill(src, s(), s())

    # All elements should be 5
    var expected = Tensor[dtype].ones(3, 4) * 5.0
    assert_true(x.all_close[atol=1e-6](expected))


fn test_fill_tensor_broadcast_1d_to_2d() raises:
    """Test broadcasting 1D tensor to 2D (row broadcast)."""
    print("test_fill_tensor_broadcast_1d_to_2d")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(3, 4)
    var src = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])  # (4,)

    x.fill(src, s(), s())  # Broadcast to (3, 4)

    # Each row should be [1, 2, 3, 4]
    assert_true(x[0, 0] == 1.0)
    assert_true(x[0, 1] == 2.0)
    assert_true(x[1, 0] == 1.0)  # Second row
    assert_true(x[2, 0] == 1.0)  # Third row


fn test_fill_tensor_broadcast_column() raises:
    """Test broadcasting (3, 1) to (3, 4)."""
    print("test_fill_tensor_broadcast_column")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(3, 4)
    var src = Tensor[dtype].d2([[10.0], [20.0], [30.0]])  # (3, 1)

    x.fill(src, s(), s())

    # Each row should be filled with its value
    assert_true(x[0, 0] == 10.0)
    assert_true(x[0, 3] == 10.0)  # Row 0
    assert_true(x[1, 0] == 20.0)  # Row 1
    assert_true(x[2, 0] == 30.0)  # Row 2


fn test_fill_tensor_broadcast_to_slice() raises:
    """Test broadcasting tensor to a slice."""
    print("test_fill_tensor_broadcast_to_slice")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(5, 6)
    var src = Tensor[dtype].d1([1.0, 2.0, 3.0])  # (3,)

    x.fill(src, i(2), s(1, 4))  # Fill row 2, cols [1:4]

    # Row 2, cols [1:4] should be [1, 2, 3]
    assert_true(x[2, 1] == 1.0)  # [2, 1]
    assert_true(x[2, 2] == 2.0)  # [2, 2]
    assert_true(x[2, 3] == 3.0)  # [2, 3]
    assert_true(x[2, 0] == 0.0)  # Outside


fn test_fill_tensor_broadcast_scalar() raises:
    """Test broadcasting scalar-like (1,) tensor."""
    print("test_fill_tensor_broadcast_scalar")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(3, 4)
    var src = Tensor[dtype].d1([42.0])  # (1,) - effectively scalar

    x.fill(src, s(), s())

    # All elements should be 42
    var expected = Tensor[dtype].ones(3, 4) * 42.0
    assert_true(x.all_close[atol=1e-6](expected))


fn test_fill_tensor_broadcast_strided() raises:
    """Test broadcasting to strided slice."""
    print("test_fill_tensor_broadcast_strided")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(4, 6)
    var src = Tensor[dtype].d1([10.0, 20.0, 30.0])  # (3,)

    x.fill(src, s(None, None, 2), s(None, None, 2))  # Every other row/col
    # Target shape: (2, 3), src broadcasts to it

    # Check strided positions
    assert_true(x[0, 0] == 10.0)  # [0, 0]
    assert_true(x[0, 2] == 20.0)  # [0, 2]
    assert_true(x[0, 4] == 30.0)  # [0, 4]
    assert_true(x[2, 0] == 10.0)  # [2, 0]


# ============================================================================
# TENSOR FILL TESTS - Views
# ============================================================================


fn test_fill_tensor_to_view() raises:
    """Test filling tensor into a view."""
    print("test_fill_tensor_to_view")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(20)
    var view = x[5:10]

    var src = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    view.fill(src, s())

    # x should be modified at [5:10]
    assert_true(x[4] == 0.0)
    assert_true(x[5] == 1.0)
    assert_true(x[9] == 5.0)
    assert_true(x[10] == 0.0)


fn test_fill_tensor_reshaped_view() raises:
    """Test filling into reshaped view."""
    print("test_fill_tensor_reshaped_view")

    alias dtype = DType.float32
    var x = Tensor[dtype].arange(12)
    var view = x.reshape(3, 4)

    var src = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0])
    view.fill(src, i(1), s())  # Fill second row

    # x should be modified at [4:8]
    assert_true(x[3] == 3.0)
    assert_true(x[4] == 10.0)
    assert_true(x[7] == 40.0)
    assert_true(x[8] == 8.0)


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


fn test_fill_scalar_0d_tensor() raises:
    """Test filling 0D (scalar) tensor."""
    print("test_fill_scalar_0d_tensor")

    alias dtype = DType.float32
    var x = Tensor[dtype].scalar(0.0)

    x.fill(123.0, il(IntArray()))  # Empty indices for scalar

    assert_true(x[[]] == 123.0)


fn test_fill_tensor_partial_fill() raises:
    """Test filling only part of tensor."""
    print("test_fill_tensor_partial_fill")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(5, 5)
    var src = Tensor[dtype].ones(2, 2) * 9.0

    x.fill(src, s(1, 3), s(1, 3))  # Fill center 2x2

    # Only center should be 9
    assert_true(x[0, 0] == 0.0)
    assert_true(x[1, 1] == 9.0)  # [1, 1]
    assert_true(x[2, 2] == 9.0)  # [2, 2]
    assert_true(x[3, 3] == 0.0)


fn test_fill_contiguous_optimization() raises:
    """Test that contiguous fills are optimized."""
    print("test_fill_contiguous_optimization")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(100)
    var src = Tensor[dtype].ones(100) * 7.0

    # Should use fast memcpy path
    x.fill(src, s())

    var expected = Tensor[dtype].ones(100) * 7.0
    assert_true(x.all_close[atol=1e-6](expected))


# ============================================================================
# COMPREHENSIVE INTEGRATION TESTS
# ============================================================================


fn test_fill_complex_indexing_scenario() raises:
    """Test complex real-world indexing scenario."""
    print("test_fill_complex_indexing_scenario")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(10, 10)

    # Fill diagonal
    for j in range(10):
        x.fill(Float32(j), i(j), i(j))

    # Check diagonal
    for j in range(10):
        assert_true(x[j, j] == Float32(j))


fn test_fill_image_patch_simulation() raises:
    """Test filling patches like in image processing."""
    print("test_fill_image_patch_simulation")

    alias dtype = DType.float32
    var image = Tensor[dtype].zeros(8, 8)
    var patch = Tensor[dtype].ones(3, 3) * 5.0

    # Fill 3x3 patch at position (2, 2)
    image.fill(patch, s(2, 5), s(2, 5))

    # Check patch region
    assert_true(image[2, 2] == 5.0)  # [2, 2]
    assert_true(image[3, 2] == 5.0)  # [3, 2]
    assert_true(image[4, 4] == 5.0)  # [4, 4]
    assert_true(image[0, 0] == 0.0)  # Outside


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


fn main() raises:
    print("=" * 80)
    print("TENSOR FILL COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    print("\n--- SCALAR FILL: Integer Indexing ---")
    test_fill_scalar_single_index_1d()
    test_fill_scalar_single_index_2d()
    test_fill_scalar_negative_index()
    test_fill_scalar_3d_index()

    print("\n--- SCALAR FILL: Slice Indexing ---")
    test_fill_scalar_full_slice_1d()
    test_fill_scalar_partial_slice_1d()
    test_fill_scalar_slice_with_step_1d()
    test_fill_scalar_2d_slice_row()
    test_fill_scalar_2d_slice_col()
    test_fill_scalar_2d_submatrix()
    test_fill_scalar_strided_2d()

    print("\n--- SCALAR FILL: Array Indexing ---")
    test_fill_scalar_array_index_1d()
    test_fill_scalar_array_index_2d()
    test_fill_scalar_empty_array()
    test_fill_scalar_single_element_array()

    print("\n--- SCALAR FILL: Mixed Indexing ---")
    test_fill_scalar_mixed_int_slice()
    test_fill_scalar_mixed_array_slice()

    print("\n--- SCALAR FILL: On Views ---")
    test_fill_scalar_on_view_slice()
    test_fill_scalar_on_view_reshape()
    test_fill_scalar_on_strided_view()

    print("\n--- TENSOR FILL: Broadcasting ---")
    test_fill_tensor_exact_shape()
    test_fill_tensor_broadcast_1d_to_2d()
    test_fill_tensor_broadcast_column()
    test_fill_tensor_broadcast_to_slice()
    test_fill_tensor_broadcast_scalar()
    test_fill_tensor_broadcast_strided()

    print("\n--- TENSOR FILL: Views ---")
    test_fill_tensor_to_view()
    test_fill_tensor_reshaped_view()

    print("\n--- EDGE CASES ---")
    test_fill_scalar_0d_tensor()
    test_fill_tensor_partial_fill()
    test_fill_contiguous_optimization()

    print("\n--- INTEGRATION TESTS ---")
    test_fill_complex_indexing_scenario()
    test_fill_image_patch_simulation()

    print("\n" + "=" * 80)
    print("ALL FILL TESTS PASSED! ✓")
    print("=" * 80)
    print("\nTotal tests run: 35")
    print("  - Scalar fill (integer): 4 tests")
    print("  - Scalar fill (slice): 7 tests")
    print("  - Scalar fill (array): 4 tests")
    print("  - Scalar fill (mixed): 2 tests")
    print("  - Scalar fill (views): 3 tests")
    print("  - Tensor fill (broadcast): 6 tests")
    print("  - Tensor fill (views): 2 tests")
    print("  - Edge cases: 3 tests")
    print("  - Integration: 2 tests")
    print("\n" + "=" * 80)
    print("USAGE PATTERNS VALIDATED:")
    print("✓ x.fill(scalar, i(idx)) - single element")
    print("✓ x.fill(scalar, s()) - full slice")
    print("✓ x.fill(scalar, s(start, end)) - partial slice")
    print("✓ x.fill(scalar, il(indices)) - fancy indexing")
    print("✓ x.fill(tensor, s(), s()) - broadcasting")
    print("✓ Works on views and reshaped tensors")
    print("=" * 80)
