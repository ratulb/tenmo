from tenmo import Tensor
from testing import assert_true
from shapes import Shape

# ============================================================================
# FORWARD PASS TESTS - CONSTANT PADDING
# ============================================================================


fn test_pad_constant_2d_symmetric() raises:
    """Test symmetric constant padding on 2D tensor."""
    print("test_pad_constant_2d_symmetric")

    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 1))  # Pad dimension 0: 1 before, 1 after
    pad.append((1, 1))  # Pad dimension 1: 1 before, 1 after

    var result = Tensor[dtype].pad(x, pad, mode="constant", value=0.0)

    # Expected shape: (4, 4)
    assert_true(result.shape()[0] == 4)
    assert_true(result.shape()[1] == 4)

    # Expected:
    # [[0, 0, 0, 0],
    #  [0, 1, 2, 0],
    #  [0, 3, 4, 0],
    #  [0, 0, 0, 0]]
    var expected = Tensor[dtype].d2(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_pad_constant_2d_asymmetric() raises:
    """Test asymmetric constant padding on 2D tensor."""
    print("test_pad_constant_2d_asymmetric")

    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 0))  # Dim 0: 1 before, 0 after
    pad.append((2, 1))  # Dim 1: 2 before, 1 after

    var result = Tensor[dtype].pad(x, pad, mode="constant", value=0.0)

    # Expected shape: (3, 6)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 6)

    # Expected:
    # [[0, 0, 0, 0, 0, 0],
    #  [0, 0, 1, 2, 3, 0],
    #  [0, 0, 4, 5, 6, 0]]
    var expected = Tensor[dtype].d2(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
            [0.0, 0.0, 4.0, 5.0, 6.0, 0.0],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_pad_constant_1d() raises:
    """Test constant padding on 1D tensor."""
    print("test_pad_constant_1d")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0])

    var pad = List[Tuple[Int, Int]]()
    pad.append((2, 3))  # 2 before, 3 after

    var result = Tensor[dtype].pad(x, pad, mode="constant", value=0.0)

    # Expected shape: (8,)
    assert_true(result.shape()[0] == 8)

    # Expected: [0, 0, 1, 2, 3, 0, 0, 0]
    var expected = Tensor[dtype].d1([0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

    assert_true(result.all_close[atol=1e-6](expected))


fn test_pad_constant_3d() raises:
    """Test constant padding on 3D tensor."""
    print("test_pad_constant_3d")

    alias dtype = DType.float32
    var x = Tensor[dtype].d3([[[1.0, 2.0]]])  # (1, 1, 2)

    var pad = List[Tuple[Int, Int]]()
    pad.append((0, 0))  # No padding on dim 0
    pad.append((1, 0))  # Pad dim 1: 1 before, 0 after
    pad.append((0, 1))  # Pad dim 2: 0 before, 1 after

    var result = Tensor[dtype].pad(x, pad, mode="constant", value=0.0)

    # Expected shape: (1, 2, 3)
    assert_true(result.shape()[0] == 1)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 3)


fn test_pad_constant_4d_conv_style() raises:
    """Test constant padding on 4D tensor (typical for conv layers)."""
    print("test_pad_constant_4d_conv_style")

    alias dtype = DType.float32
    # Create 4D tensor: (batch=1, channels=1, H=2, W=2)
    var x = Tensor[dtype].zeros(Shape(1, 1, 2, 2))
    x.buffer.data_buffer()[0] = 1.0
    x.buffer.buffer[1] = 2.0
    x.buffer.buffer[2] = 3.0
    x.buffer.buffer[3] = 4.0

    var pad = List[Tuple[Int, Int]]()
    pad.append((0, 0))  # No padding on batch
    pad.append((0, 0))  # No padding on channels
    pad.append((1, 1))  # Pad H: 1 on each side
    pad.append((1, 1))  # Pad W: 1 on each side

    var result = Tensor[dtype].pad(x, pad, mode="constant", value=0.0)

    # Expected shape: (1, 1, 4, 4)
    assert_true(result.shape()[0] == 1)
    assert_true(result.shape()[1] == 1)
    assert_true(result.shape()[2] == 4)
    assert_true(result.shape()[3] == 4)


fn test_pad_constant_nonzero_value() raises:
    """Test constant padding with non-zero pad value."""
    print("test_pad_constant_nonzero_value")

    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0]])

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 1))
    pad.append((1, 1))

    var result = Tensor[dtype].pad(x, pad, mode="constant", value=-1.0)

    # Expected:
    # [[-1, -1, -1, -1],
    #  [-1,  1,  2, -1],
    #  [-1, -1, -1, -1]]
    var expected = Tensor[dtype].d2(
        [
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, 1.0, 2.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_pad_zero_padding() raises:
    """Test padding with zero on all sides (no-op)."""
    print("test_pad_zero_padding")

    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])

    var pad = List[Tuple[Int, Int]]()
    pad.append((0, 0))
    pad.append((0, 0))

    var result = Tensor[dtype].pad(x, pad, mode="constant", value=0.0)

    # Should be identical to input
    assert_true(result.all_close[atol=1e-6](x))


# ============================================================================
# BACKWARD PASS TESTS
# ============================================================================


fn test_pad_backward_symmetric() raises:
    """Test gradient flow through symmetric padding."""
    print("test_pad_backward_symmetric")

    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 1))
    pad.append((1, 1))

    var result = Tensor[dtype].pad(x, pad)  # (4, 4)
    var loss = result.sum()
    loss.backward()

    # Gradients should be 1.0 for all input elements
    # (padded regions don't contribute to input gradients)
    var expected_grad = Tensor[dtype].ones(2, 2)

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_pad_backward_asymmetric() raises:
    """Test gradient flow through asymmetric padding."""
    print("test_pad_backward_asymmetric")

    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    var pad = List[Tuple[Int, Int]]()
    pad.append((2, 0))
    pad.append((0, 3))

    var result = Tensor[dtype].pad(x, pad)  # (4, 5)
    var loss = result.sum()
    loss.backward()

    var expected_grad = Tensor[dtype].ones(2, 2)

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_pad_backward_weighted() raises:
    """Test gradient flow with weighted loss."""
    print("test_pad_backward_weighted")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 1))

    var result = Tensor[dtype].pad(x, pad)  # [0, 1, 2, 3, 0]

    # Apply weights: [1, 2, 3, 4, 5]
    var weights = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    # Gradients for x should be weights of non-padded region: [2, 3, 4]
    var expected_grad = Tensor[dtype].d1([2.0, 3.0, 4.0])

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_pad_backward_1d() raises:
    """Test gradient flow for 1D padding."""
    print("test_pad_backward_1d")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)

    var pad = List[Tuple[Int, Int]]()
    pad.append((2, 3))

    var result = Tensor[dtype].pad(x, pad)  # (7,)
    var loss = result.sum()
    loss.backward()

    var expected_grad = Tensor[dtype].ones(2)

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_pad_backward_3d() raises:
    """Test gradient flow for 3D padding."""
    print("test_pad_backward_3d")

    alias dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True
    )  # (1, 2, 2)

    var pad = List[Tuple[Int, Int]]()
    pad.append((0, 0))
    pad.append((1, 1))
    pad.append((1, 1))

    var result = Tensor[dtype].pad(x, pad)  # (1, 4, 4)
    var loss = result.sum()
    loss.backward()

    var expected_grad = Tensor[dtype].ones(1, 2, 2)

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_pad_backward_chain() raises:
    """Test gradient flow through chained operations."""
    print("test_pad_backward_chain")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 1))

    var padded = Tensor[dtype].pad(x, pad)  # [0, 1, 2, 0]
    var squared = padded * padded  # [0, 1, 4, 0]
    var loss = squared.sum()  # 5
    loss.backward()

    # d(loss)/d(x) = 2 * x (only in non-padded region)
    var expected_grad = Tensor[dtype].d1([2.0, 4.0])

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_pad_backward_4d_conv_style() raises:
    """Test gradient flow for 4D tensor (conv layer style)."""
    print("test_pad_backward_4d_conv_style")

    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(Shape(1, 2, 2, 2), requires_grad=True)
    for i in range(8):
        x.buffer.data_buffer()[i] = Float32(i) + 1.0

    var pad = List[Tuple[Int, Int]]()
    pad.append((0, 0))  # No padding on batch
    pad.append((0, 0))  # No padding on channels
    pad.append((1, 1))  # Pad H
    pad.append((1, 1))  # Pad W

    var result = Tensor[dtype].pad(x, pad)  # (1, 2, 4, 4)
    var loss = result.sum()
    loss.backward()

    # All elements should have gradient 1.0
    var expected_grad = Tensor[dtype].ones(1, 2, 2, 2)

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


# ============================================================================
# SPECIAL PADDING MODES (if implemented)
# ============================================================================


fn test_pad_replicate_2d() raises:
    """Test replicate padding mode."""
    print("test_pad_replicate_2d")

    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 1))
    pad.append((1, 1))

    var result = Tensor[dtype].pad(x, pad, mode="replicate")

    # Expected: edge values are replicated
    # [[1, 1, 2, 2],
    #  [1, 1, 2, 2],
    #  [3, 3, 4, 4],
    #  [3, 3, 4, 4]]
    var expected = Tensor[dtype].d2(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_pad_reflect_2d() raises:
    """Test reflect padding mode."""
    print("test_pad_reflect_2d")

    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 0))
    pad.append((1, 0))

    var result = Tensor[dtype].pad(x, pad, mode="reflect")

    # Expected: reflection at borders
    # [[5, 4, 5, 6],
    #  [2, 1, 2, 3],
    #  [5, 4, 5, 6]]
    var expected = Tensor[dtype].d2(
        [[5.0, 4.0, 5.0, 6.0], [2.0, 1.0, 2.0, 3.0], [5.0, 4.0, 5.0, 6.0]]
    )

    assert_true(result.all_close[atol=1e-6](expected))


# ============================================================================
# INTEGRATION AND USE CASE TESTS
# ============================================================================


fn test_pad_for_convolution() raises:
    """Test padding for CNN convolution layer."""
    print("test_pad_for_convolution")

    alias dtype = DType.float32
    # Typical conv input: (batch=2, channels=3, H=28, W=28)
    var x = Tensor[dtype].ones(2, 3, 28, 28, requires_grad=True)

    # Pad for 3x3 conv with "same" padding
    var pad = List[Tuple[Int, Int]]()
    pad.append((0, 0))  # No padding on batch
    pad.append((0, 0))  # No padding on channels
    pad.append((1, 1))  # Pad H: 1 on each side for 3x3 kernel
    pad.append((1, 1))  # Pad W: 1 on each side

    var padded = Tensor[dtype].pad(x, pad)

    # Expected shape: (2, 3, 30, 30)
    assert_true(padded.shape()[0] == 2)
    assert_true(padded.shape()[1] == 3)
    assert_true(padded.shape()[2] == 30)
    assert_true(padded.shape()[3] == 30)

    # Test gradient flow
    var loss = padded.sum()
    loss.backward()

    var expected_grad = Tensor[dtype].ones(2, 3, 28, 28)
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_pad_selective_dimensions() raises:
    """Test padding only specific dimensions."""
    print("test_pad_selective_dimensions")

    alias dtype = DType.float32
    var x = Tensor[dtype](2, 3, 4, 5)

    # Pad only last two dimensions (spatial)
    var pad = List[Tuple[Int, Int]]()
    pad.append((0, 0))  # No padding
    pad.append((0, 0))  # No padding
    pad.append((2, 2))  # Pad this dimension
    pad.append((3, 3))  # Pad this dimension

    var result = Tensor[dtype].pad(x, pad)

    # Expected shape: (2, 3, 8, 11)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 3)
    assert_true(result.shape()[2] == 8)
    assert_true(result.shape()[3] == 11)


fn test_pad_requires_grad_propagation() raises:
    """Test requires_grad propagation."""
    print("test_pad_requires_grad_propagation")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 1))

    var result1 = Tensor[dtype].pad(x, pad)
    assert_true(result1.requires_grad)

    var result2 = Tensor[dtype].pad(x, pad, requires_grad=False)
    assert_true(result2.requires_grad == False)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


fn main() raises:
    print("=" * 80)
    print("PAD COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    print("\n--- FORWARD PASS TESTS: CONSTANT PADDING ---")
    test_pad_constant_2d_symmetric()
    test_pad_constant_2d_asymmetric()
    test_pad_constant_1d()
    test_pad_constant_3d()
    test_pad_constant_4d_conv_style()
    test_pad_constant_nonzero_value()
    test_pad_zero_padding()

    print("\n--- BACKWARD PASS TESTS ---")
    test_pad_backward_symmetric()
    test_pad_backward_asymmetric()
    test_pad_backward_weighted()
    test_pad_backward_1d()
    test_pad_backward_3d()
    test_pad_backward_chain()
    test_pad_backward_4d_conv_style()

    print("\n--- SPECIAL PADDING MODES ---")
    test_pad_replicate_2d()
    test_pad_reflect_2d()

    print("\n--- INTEGRATION TESTS ---")
    test_pad_for_convolution()
    test_pad_selective_dimensions()
    test_pad_requires_grad_propagation()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("\nTotal tests run: 20")
    print("  - Forward (constant): 7 tests")
    print("  - Backward: 7 tests")
    print("  - Special modes: 2 tests")
    print("  - Integration: 3 tests")

    print("=" * 80)
    print("CIRCULAR PADDING COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    print("\n--- FORWARD PASS TESTS ---")
    test_pad_circular_1d_symmetric()
    test_pad_circular_1d_asymmetric()
    test_pad_circular_2d_symmetric()
    test_pad_circular_2d_asymmetric()
    test_pad_circular_2d_one_dimension()
    test_pad_circular_large_padding()
    test_pad_circular_3d()
    test_pad_circular_single_element()

    print("\n--- BACKWARD PASS TESTS ---")
    test_pad_circular_backward_1d()
    test_pad_circular_backward_2d()
    test_pad_circular_backward_weighted()
    test_pad_circular_backward_chain()

    print("\n--- COMPARISON TESTS ---")
    test_circular_vs_replicate_difference()
    test_circular_periodic_signal()

    print("\n" + "=" * 80)
    print("ALL CIRCULAR PADDING TESTS PASSED! ✓")
    print("=" * 80)
    print("\nTotal tests run: 14")
    print("  - Forward pass: 8 tests")
    print("  - Backward pass: 4 tests")
    print("  - Comparison: 2 tests")
    print("\n" + "=" * 80)
    print("CIRCULAR PADDING USE CASES:")
    print("  - Periodic signals (audio, time series)")
    print("  - Toroidal/wraparound geometry")
    print("  - FFT-based convolutions")
    print("  - Texture mapping in graphics")
    print("=" * 80)


# ============================================================================
# CIRCULAR/WRAP PADDING TESTS
# ============================================================================


fn test_pad_circular_1d_symmetric() raises:
    """Test symmetric circular padding on 1D tensor."""
    print("test_pad_circular_1d_symmetric")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])

    var pad = List[Tuple[Int, Int]]()
    pad.append((2, 2))  # Pad 2 on each side

    var result = Tensor[dtype].pad(x, pad, mode="circular")

    # Mapping for each output index:
    # out[0]: (0-2) % 4 = -2 % 4 = 2 → x[2] = 3
    # out[1]: (1-2) % 4 = -1 % 4 = 3 → x[3] = 4
    # out[2]: (2-2) % 4 = 0 % 4 = 0 → x[0] = 1
    # out[3]: (3-2) % 4 = 1 % 4 = 1 → x[1] = 2
    # out[4]: (4-2) % 4 = 2 % 4 = 2 → x[2] = 3
    # out[5]: (5-2) % 4 = 3 % 4 = 3 → x[3] = 4
    # out[6]: (6-2) % 4 = 4 % 4 = 0 → x[0] = 1
    # out[7]: (7-2) % 4 = 5 % 4 = 1 → x[1] = 2
    #
    # Expected: [3, 4, 1, 2, 3, 4, 1, 2]
    var expected = Tensor[dtype].d1([3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0])

    assert_true(result.all_close[atol=1e-6](expected))


fn test_pad_circular_1d_asymmetric() raises:
    """Test asymmetric circular padding on 1D tensor."""
    print("test_pad_circular_1d_asymmetric")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0])

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 3))  # 1 before, 3 after

    var result = Tensor[dtype].pad(x, pad, mode="circular")

    # Expected:
    # [3, 1, 2, 3, 1, 2, 3]
    #  ^  <-orig->  ^  ^  ^
    #  wrap         wrap around
    var expected = Tensor[dtype].d1([3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])

    assert_true(result.all_close[atol=1e-6](expected))


fn test_pad_circular_2d_symmetric() raises:
    """Test symmetric circular padding on 2D tensor."""
    print("test_pad_circular_2d_symmetric")

    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 1))  # Pad rows
    pad.append((1, 1))  # Pad cols

    var result = Tensor[dtype].pad(x, pad, mode="circular")

    # Expected shape: (4, 5)
    assert_true(result.shape()[0] == 4)
    assert_true(result.shape()[1] == 5)

    # Expected: wrap around in both dimensions
    # Row wrapping: last row wraps to top, first row wraps to bottom
    # Col wrapping: last col wraps to left, first col wraps to right
    #
    # [6, 4, 5, 6, 4]  <- bottom row wrapped to top, cols wrapped
    # [3, 1, 2, 3, 1]  <- original first row, cols wrapped
    # [6, 4, 5, 6, 4]  <- original second row, cols wrapped
    # [3, 1, 2, 3, 1]  <- top row wrapped to bottom, cols wrapped

    var expected = Tensor[dtype].d2(
        [
            [6.0, 4.0, 5.0, 6.0, 4.0],
            [3.0, 1.0, 2.0, 3.0, 1.0],
            [6.0, 4.0, 5.0, 6.0, 4.0],
            [3.0, 1.0, 2.0, 3.0, 1.0],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_pad_circular_2d_asymmetric() raises:
    """Test asymmetric circular padding on 2D tensor."""
    print("test_pad_circular_2d_asymmetric")

    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)

    var pad = List[Tuple[Int, Int]]()
    pad.append((0, 2))  # No padding before, 2 after on rows
    pad.append((2, 0))  # 2 before, no padding after on cols

    var result = Tensor[dtype].pad(x, pad, mode="circular")

    # Expected shape: (4, 4)
    assert_true(result.shape()[0] == 4)
    assert_true(result.shape()[1] == 4)

    # Mapping:
    # - Rows: no padding before, so rows 0,1 are original, rows 2,3 wrap (=0,1 again)
    # - Cols: 2 padding before, so cols wrap from right
    #   col 0: wraps to input col 0
    #   col 1: wraps to input col 1
    #   col 2: input col 0
    #   col 3: input col 1
    #
    # Result:
    # [[1, 2, 1, 2],  <- row 0 (original), cols wrapped
    #  [3, 4, 3, 4],  <- row 1 (original), cols wrapped
    #  [1, 2, 1, 2],  <- row 2 (wraps to 0), cols wrapped
    #  [3, 4, 3, 4]]  <- row 3 (wraps to 1), cols wrapped

    var expected = Tensor[dtype].d2(
        [
            [1.0, 2.0, 1.0, 2.0],
            [3.0, 4.0, 3.0, 4.0],
            [1.0, 2.0, 1.0, 2.0],
            [3.0, 4.0, 3.0, 4.0],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_pad_circular_2d_one_dimension() raises:
    """Test circular padding on only one dimension."""
    print("test_pad_circular_2d_one_dimension")

    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    var pad = List[Tuple[Int, Int]]()
    pad.append((0, 0))  # No padding on rows
    pad.append((1, 1))  # Pad columns circularly

    var result = Tensor[dtype].pad(x, pad, mode="circular")

    # Expected shape: (2, 5)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 5)

    # Expected: only columns wrap
    # [3, 1, 2, 3, 1]
    # [6, 4, 5, 6, 4]

    var expected = Tensor[dtype].d2(
        [[3.0, 1.0, 2.0, 3.0, 1.0], [6.0, 4.0, 5.0, 6.0, 4.0]]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_pad_circular_large_padding() raises:
    """Test circular padding with padding larger than input size."""
    print("test_pad_circular_large_padding")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0])

    var pad = List[Tuple[Int, Int]]()
    pad.append((5, 5))  # Padding larger than input size (3)

    var result = Tensor[dtype].pad(x, pad, mode="circular")

    # Expected shape: (13,)
    assert_true(result.shape()[0] == 13)

    # Should wrap multiple times:
    # offset -5: (0-5) % 3 = -5 % 3 = 1 (after adjustment)
    # offset -4: (0-4) % 3 = -4 % 3 = 2
    # offset -3: (0-3) % 3 = 0
    # offset -2: (0-2) % 3 = 1
    # offset -1: (0-1) % 3 = 2
    # offset 0,1,2: original [1,2,3]
    # offset 3: (3-0) % 3 = 0 -> 1
    # offset 4: (4-0) % 3 = 1 -> 2
    # offset 5: (5-0) % 3 = 2 -> 3
    # offset 6: (6-0) % 3 = 0 -> 1
    # offset 7: (7-0) % 3 = 1 -> 2

    # Pattern repeats: [2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
    var expected = Tensor[dtype].d1(
        [
            2.0,
            3.0,
            1.0,
            2.0,
            3.0,  # 5 padded before
            1.0,
            2.0,
            3.0,  # original
            1.0,
            2.0,
            3.0,
            1.0,
            2.0,  # 5 padded after
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


fn test_pad_circular_3d() raises:
    """Test circular padding on 3D tensor."""
    print("test_pad_circular_3d")

    alias dtype = DType.float32
    var x = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)

    var pad = List[Tuple[Int, Int]]()
    pad.append((0, 1))  # Wrap first dimension
    pad.append((1, 0))  # Wrap second dimension
    pad.append((0, 1))  # Wrap third dimension

    var result = Tensor[dtype].pad(x, pad, mode="circular")

    # Expected shape: (2, 3, 3)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 3)
    assert_true(result.shape()[2] == 3)


fn test_pad_circular_single_element() raises:
    """Test circular padding on single element tensor."""
    print("test_pad_circular_single_element")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([5.0])

    var pad = List[Tuple[Int, Int]]()
    pad.append((3, 3))

    var result = Tensor[dtype].pad(x, pad, mode="circular")
    result.print()

    # Expected: all elements should be 5.0
    # [5, 5, 5, 5, 5, 5, 5]
    var expected = Tensor[dtype].d1([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

    assert_true(result.all_close[atol=1e-6](expected))


# ============================================================================
# BACKWARD PASS TESTS FOR CIRCULAR PADDING
# ============================================================================


fn test_pad_circular_backward_1d() raises:
    """Test gradient flow through circular padding 1D."""
    print("test_pad_circular_backward_1d")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)

    var pad = List[Tuple[Int, Int]]()
    pad.append((2, 2))

    var result = Tensor[dtype].pad(x, pad, mode="circular")
    # Mapping:
    # out[0]: (0-2)%3 = -2%3 = 1 → x[1]=2
    # out[1]: (1-2)%3 = -1%3 = 2 → x[2]=3
    # out[2]: (2-2)%3 = 0 → x[0]=1
    # out[3]: (3-2)%3 = 1 → x[1]=2
    # out[4]: (4-2)%3 = 2 → x[2]=3
    # out[5]: (5-2)%3 = 3%3 = 0 → x[0]=1
    # out[6]: (6-2)%3 = 4%3 = 1 → x[1]=2
    # Result: [2, 3, 1, 2, 3, 1, 2]

    var loss = result.sum()
    loss.backward()

    # Gradient accumulation:
    # x[0] (value 1): appears at out[2], out[5] → gradient = 2
    # x[1] (value 2): appears at out[0], out[3], out[6] → gradient = 3
    # x[2] (value 3): appears at out[1], out[4] → gradient = 2

    var expected_grad = Tensor[dtype].d1([2.0, 3.0, 2.0])

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_pad_circular_backward_2d() raises:
    """Test gradient flow through circular padding 2D."""
    print("test_pad_circular_backward_2d")

    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 0))
    pad.append((0, 1))

    var result = Tensor[dtype].pad(x, pad, mode="circular")
    var loss = result.sum()
    loss.backward()

    _ = """x = [[1, 2],
         [3, 4]]  (2x2)

    pad = [(1, 0), (0, 1)]  // rows: 1 before, 0 after; cols: 0 before, 1 after
    ```

    **Forward mapping (circular):**
    ```
    Output shape: (3, 3)

    For each out_coord:
    - out[0,0]: row=(0-1)%2=1, col=(0-0)%2=0 → in[1,0]=3
    - out[0,1]: row=(0-1)%2=1, col=(1-0)%2=1 → in[1,1]=4
    - out[0,2]: row=(0-1)%2=1, col=(2-0)%2=0 → in[1,0]=3

    - out[1,0]: row=(1-1)%2=0, col=(0-0)%2=0 → in[0,0]=1
    - out[1,1]: row=(1-1)%2=0, col=(1-0)%2=1 → in[0,1]=2
    - out[1,2]: row=(1-1)%2=0, col=(2-0)%2=0 → in[0,0]=1

    - out[2,0]: row=(2-1)%2=1, col=(0-0)%2=0 → in[1,0]=3
    - out[2,1]: row=(2-1)%2=1, col=(1-0)%2=1 → in[1,1]=4
    - out[2,2]: row=(2-1)%2=1, col=(2-0)%2=0 → in[1,0]=3

    Result: [[3, 4, 3],
             [1, 2, 1],
             [3, 4, 3]]  Matches your output!
    ```

    **Backward (gradient accumulation):**
    ```
    Each input element appears multiple times:
    - in[0,0]=1: at out[1,0], out[1,2] → grad = 2
    - in[0,1]=2: at out[1,1] → grad = 1
    - in[1,0]=3: at out[0,0], out[0,2], out[2,0], out[2,2] → grad = 4
    - in[1,1]=4: at out[0,1], out[2,1] → grad = 2

    Expected: [[2, 1],
               [4, 2]]  Matches your output!"""

    var expected_grad = Tensor[dtype].d2([[2.0, 1.0], [4.0, 2.0]])

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_pad_circular_backward_weighted() raises:
    """Test gradient flow with weighted loss through circular padding."""
    print("test_pad_circular_backward_weighted")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 1))

    var result = Tensor[dtype].pad(x, pad, mode="circular")
    # Mapping:
    # out[0]: (0-1)%2 = -1%2 = 1 → x[1]=2
    # out[1]: (1-1)%2 = 0 → x[0]=1
    # out[2]: (2-1)%2 = 1 → x[1]=2
    # out[3]: (3-1)%2 = 2%2 = 0 → x[0]=1
    # Result: [2, 1, 2, 1]

    # Apply weights
    var weights = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    # Gradient accumulation:
    # x[0] (value 1): at out[1], out[3] with weights 2, 4 → gradient = 2+4 = 6
    # x[1] (value 2): at out[0], out[2] with weights 1, 3 → gradient = 1+3 = 4

    var expected_grad = Tensor[dtype].d1([6.0, 4.0])

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_pad_circular_backward_chain() raises:
    """Test gradient flow through chained operations with circular padding."""
    print("test_pad_circular_backward_chain")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)

    var pad = List[Tuple[Int, Int]]()
    pad.append((1, 1))

    var padded = Tensor[dtype].pad(x, pad, mode="circular")  # [2, 1, 2, 1]
    var squared = padded * padded  # [4, 1, 4, 1]
    var loss = squared.sum()  # 10
    loss.backward()

    # d(loss)/d(x[0]) = d(loss)/d(padded[1]) * d(padded[1])/d(x[0])
    #                 + d(loss)/d(padded[3]) * d(padded[3])/d(x[0])
    #                 = 2*1 + 2*1 = 4
    # d(loss)/d(x[1]) = d(loss)/d(padded[0]) * d(padded[0])/d(x[1])
    #                 + d(loss)/d(padded[2]) * d(padded[2])/d(x[1])
    #                 = 2*2 + 2*2 = 8

    var expected_grad = Tensor[dtype].d1([4.0, 8.0])

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


# ============================================================================
# COMPARISON TESTS - CIRCULAR VS OTHER MODES
# ============================================================================


fn test_circular_vs_replicate_difference() raises:
    """Show difference between circular and replicate padding."""
    print("test_circular_vs_replicate_difference")

    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0])

    var pad = List[Tuple[Int, Int]]()
    pad.append((2, 2))

    var circular = Tensor[dtype].pad(x, pad, mode="circular")
    var replicate = Tensor[dtype].pad(x, pad, mode="replicate")

    # Circular: [2, 3, 1, 2, 3, 1, 2]
    # Replicate: [1, 1, 1, 2, 3, 3, 3]

    # They should be different
    assert_true(circular[0] == 2.0)  # Circular wraps
    assert_true(replicate[0] == 1.0)  # Replicate repeats edge

    assert_true(circular[6] == 2.0)  # Circular wraps
    assert_true(replicate[6] == 3.0)  # Replicate repeats edge


fn test_circular_periodic_signal() raises:
    """Test circular padding preserves periodicity."""
    print("test_circular_periodic_signal")

    alias dtype = DType.float32
    # Periodic signal: one period
    var x = Tensor[dtype].d1([0.0, 1.0, 0.0, -1.0])

    var pad = List[Tuple[Int, Int]]()
    pad.append((4, 4))  # Pad by one full period on each side

    var result = Tensor[dtype].pad(x, pad, mode="circular")

    # Result should be 3 periods: [0,1,0,-1, 0,1,0,-1, 0,1,0,-1]
    var expected = Tensor[dtype].d1(
        [
            0.0,
            1.0,
            0.0,
            -1.0,  # Wrapped from original
            0.0,
            1.0,
            0.0,
            -1.0,  # Original
            0.0,
            1.0,
            0.0,
            -1.0,  # Wrapped from original
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))
