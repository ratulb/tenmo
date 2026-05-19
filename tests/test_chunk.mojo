from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from tenmo.common_utils import s, i

# ============================================================================
# 1D TENSOR chunking TESTS
# ============================================================================


def test_chunk_1d_full() raises:
    """Test chunking entire 1D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])

    var chunked = x.chunk(s())

    # Should be identical to original
    assert_true(chunked.shape()[0] == 5)
    assert_true(chunked.all_close[atol=1e-6](x))


def test_chunk_1d_start_only() raises:
    """Test 1D slice with start index only."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])

    var chunked = x.chunk(s(2, None, None))

    # Should be [3, 4, 5]
    assert_true(chunked.shape()[0] == 3)
    var expected = Tensor[dtype].d1([3.0, 4.0, 5.0])
    assert_true(chunked.all_close[atol=1e-6](expected))


def test_chunk_1d_end_only() raises:
    """Test 1D slice with end index only."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])

    # var chunked = x[:3]
    var chunked = x.chunk(s(3))

    # Should be [1, 2, 3]
    assert_true(chunked.shape()[0] == 3)
    var expected = Tensor[dtype].d1([1.0, 2.0, 3.0])
    assert_true(chunked.all_close[atol=1e-6](expected))


def test_chunk_1d_start_end() raises:
    """Test 1D slice with both start and end."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])

    var chunked = x.chunk(s(1, 4))

    # Should be [2, 3, 4]
    assert_true(chunked.shape()[0] == 3)
    var expected = Tensor[dtype].d1([2.0, 3.0, 4.0])
    assert_true(chunked.all_close[atol=1e-6](expected))


def test_chunk_1d_with_step() raises:
    """Test 1D slice with step."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    var chunked = x.chunk(s(None, None, 2))  # Every other element

    # Should be [1, 3, 5]
    assert_true(chunked.shape()[0] == 3)
    var expected = Tensor[dtype].d1([1.0, 3.0, 5.0])
    assert_true(chunked.all_close[atol=1e-6](expected))


def test_chunk_1d_negative_indices() raises:
    """Test 1D slice with negative indices."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])

    # var chunked = x[-3:-1]
    var chunked = x.chunk(s(-3, -1))

    # Should be [3, 4]
    assert_true(chunked.shape()[0] == 2)
    var expected = Tensor[dtype].d1([3.0, 4.0])
    assert_true(chunked.all_close[atol=1e-6](expected))


def test_chunk_1d_single_element() raises:
    """Test 1D slice getting single element range."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])

    var chunked = x.chunk(s(2, 3))

    # Should be [3]
    assert_true(chunked.shape()[0] == 1)
    assert_true(chunked[0] == 3.0)


# ============================================================================
# 2D TENSOR chunking TESTS
# ============================================================================


def test_chunk_2d_full() raises:
    """Test chunking entire 2D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    var chunked = x.chunk(s(), s())

    assert_true(chunked.shape()[0] == 3)
    assert_true(chunked.shape()[1] == 3)
    assert_true(chunked.all_close[atol=1e-6](x))


def test_chunk_2d_single_row() raises:
    """Test chunking single row from 2D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    var chunked = x.chunk(s(1, 2), s())

    # Should be [[4, 5, 6]]
    assert_true(chunked.shape()[0] == 1)
    assert_true(chunked.shape()[1] == 3)
    var expected = Tensor[dtype].d2([[4.0, 5.0, 6.0]])
    assert_true(chunked.all_close[atol=1e-6](expected))


def test_chunk_2d_single_column() raises:
    """Test chunking single column from 2D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    var chunked = x.chunk(s(), s(1, 2))

    # Should be [[2], [5], [8]]
    assert_true(chunked.shape()[0] == 3)
    assert_true(chunked.shape()[1] == 1)
    var expected = Tensor[dtype].d2([[2.0], [5.0], [8.0]])
    assert_true(chunked.all_close[atol=1e-6](expected))


def test_chunk_2d_submatrix() raises:
    """Test chunking submatrix from 2D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )

    var chunked = x.chunk(s(1, 3), s(1, 3))

    # Should be [[6, 7], [10, 11]]
    assert_true(chunked.shape()[0] == 2)
    assert_true(chunked.shape()[1] == 2)
    var expected = Tensor[dtype].d2([[6.0, 7.0], [10.0, 11.0]])
    assert_true(chunked.all_close[atol=1e-6](expected))


def test_chunk_2d_rows_subset() raises:
    """Test chunking subset of rows."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    )

    var chunked = x.chunk(s(1, 3), s())

    # Should be [[4, 5, 6], [7, 8, 9]]
    assert_true(chunked.shape()[0] == 2)
    assert_true(chunked.shape()[1] == 3)
    var expected = Tensor[dtype].d2([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert_true(chunked.all_close[atol=1e-6](expected))


def test_chunk_2d_cols_subset() raises:
    """Test chunking subset of columns."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    )

    var chunked = x.chunk(s(), s(1, 3))

    # Should be [[2, 3], [6, 7], [10, 11]]
    assert_true(chunked.shape()[0] == 3)
    assert_true(chunked.shape()[1] == 2)
    var expected = Tensor[dtype].d2([[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]])
    assert_true(chunked.all_close[atol=1e-6](expected))


def test_chunk_2d_with_step() raises:
    """Test 2D chunking with step."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )

    var chunked = x.chunk(
        s(None, None, 2), s(None, None, 2)
    )  # Every other row and column

    # Should be [[1, 3], [9, 11]]
    assert_true(chunked.shape()[0] == 2)
    assert_true(chunked.shape()[1] == 2)
    var expected = Tensor[dtype].d2([[1.0, 3.0], [9.0, 11.0]])
    assert_true(chunked.all_close[atol=1e-6](expected))


# ============================================================================
# 3D TENSOR chunking TESTS
# ============================================================================


def test_chunk_3d_full() raises:
    """Test chunking entire 3D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].ones(2, 3, 4)

    var chunked = x.chunk(s(), s(), s())

    assert_true(chunked.shape()[0] == 2)
    assert_true(chunked.shape()[1] == 3)
    assert_true(chunked.shape()[2] == 4)
    assert_true(chunked.all_close[atol=1e-6](x))


def test_chunk_3d_single_depth() raises:
    """Test chunking single depth slice from 3D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].arange(2 * 3 * 4)
    for i in range(24):
        x[i] = Float32(i) + 1.0
    x = x.reshape(2, 3, 4)
    var chunked = x.chunk(s(0, 1), s(), s())

    assert_true(chunked.shape()[0] == 1)
    assert_true(chunked.shape()[1] == 3)
    assert_true(chunked.shape()[2] == 4)


def test_chunk_3d_subvolume() raises:
    """Test chunking subvolume from 3D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].arange(4 * 4 * 4)
    for i in range(64):
        x[i] = Float32(i)

    x = x.reshape(4, 4, 4)
    var chunked = x.chunk(s(1, 3), s(1, 3), s(1, 3))

    # Should be 2x2x2 cube
    assert_true(chunked.shape()[0] == 2)
    assert_true(chunked.shape()[1] == 2)
    assert_true(chunked.shape()[2] == 2)


def test_chunk_3d_single_channel() raises:
    """Test chunking single channel from 3D tensor (like CNN)."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].arange(3 * 28 * 28)  # 3 channels, 28x28 spatial
    x = x.reshape(3, 28, 28)
    var chunked = x.chunk(s(1, 2), s(), s())  # Get channel 1

    assert_true(chunked.shape()[0] == 1)
    assert_true(chunked.shape()[1] == 28)
    assert_true(chunked.shape()[2] == 28)


def test_chunk_3d_spatial_region() raises:
    """Test chunking spatial region from 3D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].arange(3 * 28 * 28)
    x = x.reshape(3, 28, 28)

    var chunked = x.chunk(
        s(), s(10, 20), s(10, 20)
    )  # All channels, 10x10 spatial region

    assert_true(chunked.shape()[0] == 3)
    assert_true(chunked.shape()[1] == 10)
    assert_true(chunked.shape()[2] == 10)


# ============================================================================
# 4D TENSOR chunking TESTS (CNN BATCH)
# ============================================================================


def test_chunk_4d_full() raises:
    """Test chunking entire 4D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].ones(2, 3, 28, 28)

    var chunked = x.chunk(s(), s(), s(), s())

    assert_true(chunked.shape()[0] == 2)
    assert_true(chunked.shape()[1] == 3)
    assert_true(chunked.shape()[2] == 28)
    assert_true(chunked.shape()[3] == 28)


def test_chunk_4d_single_batch() raises:
    """Test chunking single batch element from 4D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].rand(8, 3, 32, 32)  # Batch of 8 images

    var chunked = x.chunk(s(0, 1), s(), s(), s())  # First image

    assert_true(chunked.shape()[0] == 1)
    assert_true(chunked.shape()[1] == 3)
    assert_true(chunked.shape()[2] == 32)
    assert_true(chunked.shape()[3] == 32)


def test_chunk_4d_batch_subset() raises:
    """Test chunking subset of batch."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].rand(16, 3, 32, 32)

    var chunked = x.chunk(s(4, 8), s(), s(), s())  # Mini-batch of 4 images

    assert_true(chunked.shape()[0] == 4)
    assert_true(chunked.shape()[1] == 3)
    assert_true(chunked.shape()[2] == 32)
    assert_true(chunked.shape()[3] == 32)


def test_chunk_4d_channel_subset() raises:
    """Test chunking subset of channels from 4D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].rand(2, 64, 14, 14)  # 64 channels

    var chunked = x.chunk(s(), s(0, 32), s(), s())  # First 32 channels

    assert_true(chunked.shape()[0] == 2)
    assert_true(chunked.shape()[1] == 32)
    assert_true(chunked.shape()[2] == 14)
    assert_true(chunked.shape()[3] == 14)


def test_chunk_4d_spatial_crop() raises:
    """Test spatial cropping from 4D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].rand(4, 3, 224, 224)  # Large images

    var chunked = x.chunk(
        s(), s(), s(50, 150), s(50, 150)
    )  # Crop center 100x100

    assert_true(chunked.shape()[0] == 4)
    assert_true(chunked.shape()[1] == 3)
    assert_true(chunked.shape()[2] == 100)
    assert_true(chunked.shape()[3] == 100)


# ============================================================================
# CHAINED chunking TESTS
# ============================================================================


def test_chained_chunking_2d() raises:
    """Test chaining multiple chunking operations."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )

    # Chain: first get rows 1-3, then get cols 1-3
    var chunk1 = x.chunk(s(1, 3), s())
    var chunk2 = chunk1.chunk(s(), s(1, 3))
    # Should be [[6, 7], [10, 11]]
    assert_true(chunk2.shape()[0] == 2)
    assert_true(chunk2.shape()[1] == 2)
    var expected = Tensor[dtype].d2([[6.0, 7.0], [10.0, 11.0]])
    assert_true(chunk2.all_close[atol=1e-6](expected))


def test_chained_chunking_3d() raises:
    """Test chaining slices on 3D tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].rand(4, 4, 4)
    for i in range(64):
        x.buffer.data_buffer()[i] = Float32(i)

    var chunked1 = x.chunk(s(1, 3), s(), s())  # Get 2 depth slices
    var chunked2 = chunked1.chunk(s(), s(1, 3), s())  # Get 2 rows
    var chunked3 = chunked2.chunk(s(), s(), s(1, 3))  # Get 2 cols

    assert_true(chunked3.shape()[0] == 2)
    assert_true(chunked3.shape()[1] == 2)
    assert_true(chunked3.shape()[2] == 2)


# ============================================================================
# IMMUTABILITY TESTS (KEY FOR 'self' vs 'mut self')
# ============================================================================


def test_immutable_multiple_slices() raises:
    """Test that we can create multiple slices from same immutable tensor."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])

    # With 'self' (not 'mut self'), we can create multiple views
    var slice1 = x.chunk(s(0, 2))
    var slice2 = x.chunk(s(2, 4))
    var slice3 = x.chunk(s(1, 3))

    # All slices should be valid
    assert_true(slice1.shape()[0] == 2)
    assert_true(slice2.shape()[0] == 2)
    assert_true(slice3.shape()[0] == 2)

    assert_true(slice1[0] == 1.0)
    assert_true(slice2[0] == 3.0)
    assert_true(slice3[0] == 2.0)


def test_immutable_chunk_from_const() raises:
    """Test chunking from a const-reference like scenario."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])

    var chunked = x.chunk(s(1, 4))

    assert_true(chunked.shape()[0] == 3)
    var expected = Tensor[dtype].d1([2.0, 3.0, 4.0])
    assert_true(chunked.all_close[atol=1e-6](expected))


def test_immutable_overlapping_slices() raises:
    """Test creating overlapping slices (only possible with 'self')."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    # Create overlapping views
    var top_half = x.chunk(s(0, 2), s())
    var bottom_half = x.chunk(s(1, 3), s())
    var middle_row = x.chunk(s(1, 2), s())

    # All should be valid and independent
    assert_true(top_half.shape()[0] == 2)
    assert_true(bottom_half.shape()[0] == 2)
    assert_true(middle_row.shape()[0] == 1)

    # Middle row is in both top and bottom halves
    assert_true(middle_row[0, 1] == 5.0)
    assert_true(top_half[1, 1] == 5.0)  # Row 1, col 1
    assert_true(bottom_half[0, 1] == 5.0)  # Row 0 of bottom_half, col 1


# ============================================================================
# GRADIENT FLOW TESTS
# ============================================================================


def test_chunk_backward_1d() raises:
    """Test gradient flow through 1D slice."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)

    var chunked = x.chunk(s(1, 4))  # [2, 3, 4]
    var loss = chunked.sum()
    loss.backward()

    # Gradient should be [0, 0, 0, 0, 0] - chunk does allocation
    var expected_grad = Tensor[dtype].d1([0.0, 0.0, 0.0, 0.0, 0.0])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_chunk_backward_2d() raises:
    """Test gradient flow through 2D slice."""

    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
    )

    var chunked = x.chunk(s(0, 2), s(1, 3))  # [[2, 3], [5, 6]]
    var loss = chunked.sum()
    loss.backward()

    # Gradient should have 1.0 only in chunked region
    var expected_grad = Tensor[dtype].d2(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
