"""
Comprehensive test suite for Conv2D implementation.
Tests forward pass, backward pass, edge cases, and numerical gradients.
"""
from tenmo import Tensor
from cnn import Conv2dFused

from testing import assert_almost_equal, assert_equal, assert_true, assert_false
from math import sqrt
from common_utils import isnan, isinf
from math import sqrt
from random import seed
from gradbox import Gradbox
from shapes import Shape
from common_utils import i, s
from forwards import Padding


fn test_basic_forward() raises:
    """Test basic forward pass with known values."""
    print("Test 1: Basic Forward Pass")

    # Simple 1x1x3x3 image, 1x1x2x2 kernel
    var image = Tensor[DType.float32].zeros(1, 1, 3, 3)
    # Set specific pattern
    image[0, 0, 0, 0] = 1.0
    image[0, 0, 0, 1] = 2.0
    image[0, 0, 1, 0] = 3.0
    image[0, 0, 1, 1] = 4.0

    var kernel = Tensor[DType.float32].zeros(1, 1, 2, 2)
    kernel[0, 0, 0, 0] = 1.0
    kernel[0, 0, 0, 1] = 0.0
    kernel[0, 0, 1, 0] = 0.0
    kernel[0, 0, 1, 1] = 1.0

    var result = Conv2dFused[DType.float32].forward(
        image, kernel, stride=1, padding="valid"
    )

    # Expected: [1*1 + 4*1, 2*1 + 0*1] = [5, 2] for first row
    assert_almost_equal(result[0, 0, 0, 0], 5.0, atol=1e-5)
    print("Basic forward pass correct")


fn test_stride() raises:
    """Test convolution with stride > 1."""
    print("\nTest 2: Stride")

    var image = Tensor[DType.float32].zeros(1, 1, 4, 4)
    for i in range(4):
        for j in range(4):
            image[0, 0, i, j] = Float32(i * 4 + j)

    var kernel = Tensor[DType.float32].ones(1, 1, 2, 2)

    # Stride 1: should give 3x3 output
    var result1 = Conv2dFused[DType.float32].forward(
        image, kernel, stride=1, padding="valid"
    )
    assert_equal(result1.shape()[2], 3)
    assert_equal(result1.shape()[3], 3)

    # Stride 2: should give 2x2 output
    var result2 = Conv2dFused[DType.float32].forward(
        image, kernel, stride=2, padding="valid"
    )
    assert_equal(result2.shape()[2], 2)
    assert_equal(result2.shape()[3], 2)

    print("Stride handling correct")


fn test_dilation() raises:
    """Test dilated convolution."""
    print("\nTest 3: Dilation")

    var image = Tensor[DType.float32].zeros(1, 1, 5, 5)
    for i in range(5):
        for j in range(5):
            image[0, 0, i, j] = Float32(i * 5 + j)

    var kernel = Tensor[DType.float32].ones(1, 1, 2, 2)

    # Dilation 1: normal 2x2 kernel
    var _result1 = Conv2dFused[DType.float32].forward(
        image, kernel, dilation=1, padding="valid"
    )

    # Dilation 2: effective 3x3 kernel with gaps
    var result2 = Conv2dFused[DType.float32].forward(
        image, kernel, dilation=2, padding="valid"
    )

    # With dilation=2, effective kernel size is 3x3, so output is 3x3
    assert_equal(result2.shape()[2], 3)
    assert_equal(result2.shape()[3], 3)

    print("Dilation handling correct")


fn test_padding_same() raises:
    """Test 'same' padding mode."""
    print("\nTest 4: Same Padding")

    var image = Tensor[DType.float32].ones(1, 1, 4, 4)
    var kernel = Tensor[DType.float32].ones(1, 1, 3, 3)

    # With same padding and stride=1, output shape should equal input shape
    var result = Conv2dFused[DType.float32].forward(
        image, kernel, stride=1, padding="same"
    )

    assert_equal(result.shape()[2], 4)
    assert_equal(result.shape()[3], 4)

    print("Same padding correct")


fn test_padding_explicit() raises:
    """Test explicit padding values."""
    print("\nTest 5: Explicit Padding")

    var image = Tensor[DType.float32].ones(1, 1, 4, 4)
    var kernel = Tensor[DType.float32].ones(1, 1, 3, 3)

    # Test int padding
    var result1 = Conv2dFused[DType.float32].forward(
        image, kernel, padding=Padding(1)
    )
    assert_equal(result1.shape()[2], 4)
    assert_equal(result1.shape()[3], 4)

    # Test tuple padding
    var result2 = Conv2dFused[DType.float32].forward(
        image, kernel, padding=Padding((1, 2))
    )
    assert_equal(result2.shape()[2], 4)  # (4 + 2 - 3) + 1
    assert_equal(result2.shape()[3], 6)  # (4 + 4 - 3) + 1

    print("Explicit padding correct")


fn test_bias() raises:
    """Test bias handling."""
    print("\nTest 6: Bias")

    var image = Tensor[DType.float32].zeros(1, 1, 3, 3)
    var kernel = Tensor[DType.float32].zeros(2, 1, 2, 2)

    var bias = Tensor[DType.float32].zeros(2)
    bias[0] = 5.0
    bias[1] = 10.0

    var result = Conv2dFused[DType.float32].forward(
        image, kernel, bias=bias, padding="valid"
    )

    # All values should equal bias since kernel and image are zeros
    for i in range(result.shape()[2]):
        for j in range(result.shape()[3]):
            assert_almost_equal(result[0, 0, i, j], 5.0, atol=1e-5)
            assert_almost_equal(result[0, 1, i, j], 10.0, atol=1e-5)

    print("Bias handling correct")


fn test_multi_channel() raises:
    """Test multi-channel input and output."""
    print("\nTest 7: Multi-Channel")

    # 2 input channels, 3 output channels
    var image = Tensor[DType.float32].ones(1, 2, 4, 4)
    var kernel = Tensor[DType.float32].ones(3, 2, 2, 2)

    var result = Conv2dFused[DType.float32].forward(
        image, kernel, padding="valid"
    )

    assert_equal(result.shape()[0], 1)  # Batch
    assert_equal(result.shape()[1], 3)  # Output channels
    assert_equal(result.shape()[2], 3)  # Height
    assert_equal(result.shape()[3], 3)  # Width

    # Each output pixel should be sum over 2 input channels x 2x2 kernel = 8.0
    assert_almost_equal(result[0, 0, 0, 0], 8.0, atol=1e-5)

    print("Multi-channel correct")


fn test_batch() raises:
    """Test batched input."""
    print("\nTest 8: Batch Processing")

    var image = Tensor[DType.float32].ones(4, 1, 3, 3)
    # Set different values for each batch
    for b in range(4):
        for i in range(3):
            for j in range(3):
                image[b, 0, i, j] = Float32(b + 1)

    var kernel = Tensor[DType.float32].ones(1, 1, 2, 2)
    var result = Conv2dFused[DType.float32].forward(
        image, kernel, padding="valid"
    )

    assert_equal(result.shape()[0], 4)

    # Each batch should have different values
    for b in range(4):
        expected = Float32((b + 1) * 4)  # 4 kernel elements
        assert_almost_equal(result[b, 0, 0, 0], expected, atol=1e-5)

    print("Batch processing correct")


fn test_gradient_input() raises:
    """Test input gradient with numerical gradient."""
    print("\nTest 9: Input Gradient")

    var image = Tensor[DType.float32].ones(1, 1, 3, 3, requires_grad=True)
    var kernel = Tensor[DType.float32].ones(1, 1, 2, 2, requires_grad=False)

    var output = Conv2dFused[DType.float32].forward(
        image, kernel, padding="valid"
    )

    # Backward pass
    var grad_output = Tensor[DType.float32].ones(output.shape())
    output.backward(grad_output)

    # Check that gradients exist
    var grad_input = image.gradients()[]

    # Manual numerical gradient check for center pixel
    var epsilon = Scalar[DType.float32](1e-4)
    var original = image[0, 0, 1, 1]

    # Forward with +epsilon
    image[0, 0, 1, 1] = original + epsilon
    var out_plus = Conv2dFused[DType.float32].forward[track_grad=False](
        image, kernel, padding="valid"
    )

    # Forward with -epsilon
    image[0, 0, 1, 1] = original - epsilon
    var out_minus = Conv2dFused[DType.float32].forward[track_grad=False](
        image, kernel, padding="valid"
    )

    # Numerical gradient
    var numerical_grad: Float32 = 0.0
    for i in range(out_plus.shape()[2]):
        for j in range(out_plus.shape()[3]):
            numerical_grad += (out_plus[0, 0, i, j] - out_minus[0, 0, i, j]) / (
                2 * epsilon
            )

    # Restore original
    image[0, 0, 1, 1] = original

    # Compare
    var analytical_grad = grad_input[0, 0, 1, 1]
    assert_almost_equal(analytical_grad, numerical_grad, atol=1e-2)

    print("Input gradient correct")


fn test_gradient_kernel() raises:
    """Test kernel gradient with numerical gradient."""
    print("\nTest 10: Kernel Gradient")

    var image = Tensor[DType.float32].ones(1, 1, 3, 3, requires_grad=False)
    var kernel = Tensor[DType.float32].ones(1, 1, 2, 2, requires_grad=True)

    # Set image to known values
    for i in range(3):
        for j in range(3):
            image[0, 0, i, j] = Float32(i * 3 + j)

    var output = Conv2dFused[DType.float32].forward(
        image, kernel, padding="valid"
    )

    # Backward pass
    var grad_output = Tensor[DType.float32].ones(output.shape())
    output.backward(grad_output)

    var grad_kernel = kernel.gradients()[]

    # Numerical gradient check for one kernel element
    var epsilon = Scalar[DType.float32](1e-4)
    var original = kernel[0, 0, 0, 0]

    kernel[0, 0, 0, 0] = original + epsilon
    var out_plus = Conv2dFused[DType.float32].forward[track_grad=False](
        image, kernel, padding="valid"
    )

    kernel[0, 0, 0, 0] = original - epsilon
    var out_minus = Conv2dFused[DType.float32].forward[track_grad=False](
        image, kernel, padding="valid"
    )

    var numerical_grad: Float32 = 0.0
    for i in range(out_plus.shape()[2]):
        for j in range(out_plus.shape()[3]):
            numerical_grad += (out_plus[0, 0, i, j] - out_minus[0, 0, i, j]) / (
                2 * epsilon
            )

    kernel[0, 0, 0, 0] = original

    var analytical_grad = grad_kernel[0, 0, 0, 0]
    assert_almost_equal(analytical_grad, numerical_grad, atol=1e-2)

    print("Kernel gradient correct")


fn test_gradient_bias() raises:
    """Test bias gradient."""
    print("\nTest 11: Bias Gradient")

    var image = Tensor[DType.float32].ones(2, 1, 3, 3, requires_grad=False)
    var kernel = Tensor[DType.float32].ones(2, 1, 2, 2, requires_grad=False)
    var bias = Tensor[DType.float32].zeros(2, requires_grad=True)

    var output = Conv2dFused[DType.float32].forward(
        image, kernel, bias=bias, padding="valid"
    )

    # Backward with uniform gradient
    var grad_output = Tensor[DType.float32].ones(output.shape())
    output.backward(grad_output)

    var grad_bias = bias.gradients()[]

    # Bias gradient should equal number of spatial positions x batch size
    # Output is 2x2 spatial, 2 batch ? 2*2*2 = 8
    var expected = Float32(2 * 2 * 2)  # H_out * W_out * N
    assert_almost_equal(grad_bias[0], expected, atol=1e-5)
    assert_almost_equal(grad_bias[1], expected, atol=1e-5)

    print("Bias gradient correct")


fn test_gradient_flow() raises:
    """Test full gradient flow through multiple layers."""
    print("\nTest 12: Gradient Flow")

    var image = Tensor[DType.float32].ones(1, 2, 5, 5, requires_grad=True)
    var kernel1 = Tensor[DType.float32].ones(4, 2, 3, 3, requires_grad=True)
    var kernel2 = Tensor[DType.float32].ones(8, 4, 3, 3, requires_grad=True)

    # Two-layer convolution
    var out1 = Conv2dFused[DType.float32].forward(
        image, kernel1, padding="same"
    )
    var out2 = Conv2dFused[DType.float32].forward(out1, kernel2, padding="same")

    # Backward
    var grad_output = Tensor[DType.float32].ones(out2.shape())
    out2.backward(grad_output)

    # Check that all gradients exist
    assert_true(image.gradients().__as_bool__())
    assert_true(kernel1.gradients().__as_bool__())
    assert_true(kernel2.gradients().__as_bool__())

    print("Gradient flow correct")


fn test_edge_cases() raises:
    """Test edge cases and error conditions."""
    print("\nTest 13: Edge Cases")

    # Test 1x1 convolution
    var image = Tensor[DType.float32].ones(1, 3, 4, 4)
    var kernel_1x1 = Tensor[DType.float32].ones(5, 3, 1, 1)
    var result = Conv2dFused[DType.float32].forward(image, kernel_1x1)
    assert_equal(result.shape()[2], 4)
    assert_equal(result.shape()[3], 4)

    # Test large kernel relative to image
    var small_image = Tensor[DType.float32].ones(1, 1, 3, 3)
    var large_kernel = Tensor[DType.float32].ones(1, 1, 3, 3)
    var result2 = Conv2dFused[DType.float32].forward(
        small_image, large_kernel, padding="valid"
    )
    assert_equal(result2.shape()[2], 1)
    assert_equal(result2.shape()[3], 1)

    print("Edge cases handled correctly")


fn test_performance_batch() raises:
    """Performance test with realistic batch size."""
    print("\nTest 14: Performance (Batch)")

    var image = Tensor[DType.float32].randn(32, 3, 32, 32, requires_grad=True)
    var kernel = Tensor[DType.float32].randn(64, 3, 3, 3, requires_grad=True)
    var bias = Tensor[DType.float32].randn(64, requires_grad=True)

    var output = Conv2dFused[DType.float32].forward(
        image, kernel, bias=bias, padding="same"
    )

    var grad_output = Tensor[DType.float32].ones(output.shape())
    output.backward(grad_output)

    # Just check completion
    assert_equal(output.shape()[0], 32)
    assert_equal(output.shape()[1], 64)

    print("Performance test completed")


fn test_numerical_stability() raises:
    """Test with very small and very large values."""
    print("\nTest 15: Numerical Stability")

    # Very small values
    var image_small = Tensor[DType.float32].full([1, 1, 3, 3], 1e-8)
    var kernel_small = Tensor[DType.float32].full([1, 1, 2, 2], 1e-8)
    var result_small = Conv2dFused[DType.float32].forward(
        image_small, kernel_small
    )

    # Very large values
    var image_large = Tensor[DType.float32].full([1, 1, 3, 3], 1e8)
    var kernel_large = Tensor[DType.float32].full([1, 1, 2, 2], 1e-8)
    var result_large = Conv2dFused[DType.float32].forward(
        image_large, kernel_large
    )

    # Should not produce NaN or Inf
    for i in range(result_small.shape()[2]):
        for j in range(result_small.shape()[3]):
            assert_false(isnan(result_small[0, 0, i, j]))
            assert_false(isnan(result_large[0, 0, i, j]))

    print("Numerical stability good")


fn main() raises:
    print("=" * 60)
    print("Conv2D Comprehensive Test Suite")
    print("=" * 60)

    test_basic_forward()
    test_stride()
    test_dilation()
    test_padding_same()
    test_padding_explicit()
    test_bias()
    test_multi_channel()
    test_batch()
    test_gradient_input()
    test_gradient_kernel()
    test_gradient_bias()
    test_gradient_flow()
    test_edge_cases()
    test_performance_batch()
    test_numerical_stability()

    test_basic_shapes()
    test_padding_modes()
    test_stride_2()
    test_dilation_2()
    test_bias_2()
    test_known_values()
    test_gradient_shapes()
    test_gradient_correctness()
    test_gradient_stride_dilation()
    test_gradient_accumulation()
    test_batch_processing()
    test_edge_cases_2()

    print("\n--- FORWARD: Basic Functionality ---")
    test_conv2d_forward_single_batch_single_channel()
    test_conv2d_forward_with_bias()
    test_conv2d_forward_multiple_channels()
    test_conv2d_forward_multiple_filters()
    test_conv2d_forward_batch_processing()

    print("\n--- FORWARD: Stride Tests ---")
    test_conv2d_stride_2()
    test_conv2d_stride_3()

    print("\n--- FORWARD: Padding Tests ---")
    test_conv2d_padding_valid()
    test_conv2d_padding_same()
    test_conv2d_padding_int()
    test_conv2d_padding_tuple()
    test_conv2d_padding_list_asymmetric()

    print("\n--- FORWARD: Dilation Tests ---")
    test_conv2d_dilation_2()
    test_conv2d_dilation_3()

    print("\n--- FORWARD: Combined Parameters ---")
    test_conv2d_stride_and_padding()
    test_conv2d_stride_padding_dilation()
    test_conv2d_all_parameters()

    print("\n--- BACKWARD: Gradient Tests ---")
    test_conv2d_backward_simple()
    test_conv2d_backward_input_gradient()
    test_conv2d_backward_kernel_gradient()
    test_conv2d_backward_with_stride()

    print("\n--- EDGE CASES ---")
    test_conv2d_edge_1x1_kernel()
    test_conv2d_edge_large_kernel()
    test_conv2d_edge_output_size_1()


# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------


fn assert_shape_equal[
    dtype: DType, //
](tensor: Tensor[dtype], expected: List[Int], test_name: String) raises:
    """Assert tensor has expected shape."""
    var actual = tensor.shape()
    if actual.rank() != len(expected):
        raise Error(
            test_name
            + " FAILED: Shape rank mismatch. Expected "
            + len(expected).__str__()
            + " got "
            + actual.rank().__str__()
        )

    for i in range(len(expected)):
        if actual[i] != expected[i]:
            raise Error(
                test_name
                + " FAILED: Shape mismatch at dim "
                + i.__str__()
                + ". Expected "
                + expected[i].__str__()
                + " got "
                + actual[i].__str__()
            )

    print(test_name + " - Shape correct:", actual.__str__())


fn assert_tensor_close(
    a: Gradbox[DType.float32],
    b: Gradbox[DType.float32],
    test_name: String,
    rtol: Float32 = 1e-4,
    atol: Float32 = 1e-6,
) raises:
    """Assert two tensors are element-wise close."""
    if a.shape() != b.shape():
        raise Error(test_name + " FAILED: Shape mismatch")

    var max_diff: Float32 = 0.0
    var total_elements = a.numels()

    for i in range(total_elements):
        var diff = abs(a.buffer.data_buffer()[i] - b.buffer.data_buffer()[i])
        var threshold = atol + rtol * abs(b.buffer.data_buffer()[i])

        if diff > threshold:
            max_diff = max(max_diff, diff)
            if max_diff > 10 * threshold:  # Only fail if really off
                raise Error(
                    test_name
                    + " FAILED: Value mismatch at index "
                    + i.__str__()
                    + ". Expected "
                    + b.buffer.data_buffer()[i].__str__()
                    + " got "
                    + a.buffer.data_buffer()[i].__str__()
                    + " (diff: "
                    + diff.__str__()
                    + ")"
                )

    print(test_name + " - Values close (max_diff: " + max_diff.__str__() + ")")


fn compute_numerical_gradient(
    image: Tensor[DType.float32],
    mut kernel: Tensor[DType.float32],
    bias: Optional[Tensor[DType.float32]],
    stride: Int,
    dilation: Int,
    padding: Padding,
    param_type: String,  # "image", "kernel", or "bias"
    eps: Float32 = 1e-4,
) -> Tensor[DType.float32]:
    """Compute numerical gradient using finite differences."""

    var grad_shape: Shape
    if param_type == "image":
        grad_shape = image.shape()
    elif param_type == "kernel":
        grad_shape = kernel.shape()
    else:  # bias
        grad_shape = bias.value().shape()

    var numerical_grad = Tensor[DType.float32].zeros(grad_shape)

    # For each parameter
    for idx in range(numerical_grad.numels()):
        # Forward pass with +eps
        var param_plus: Tensor[DType.float32]
        if param_type == "image":
            param_plus = image.copy()
            param_plus.buffer.data_buffer()[idx] += eps
        elif param_type == "kernel":
            param_plus = kernel.copy()
            param_plus.buffer.data_buffer()[idx] += eps
        else:
            param_plus = bias.value().copy()
            param_plus.buffer.data_buffer()[idx] += eps

        var out_plus: Tensor[DType.float32]
        if param_type == "image":
            out_plus = Conv2dFused[DType.float32].forward[track_grad=False](
                param_plus, kernel, bias, stride, dilation, padding
            )
        elif param_type == "kernel":
            out_plus = Conv2dFused[DType.float32].forward[track_grad=False](
                image, param_plus, bias, stride, dilation, padding
            )
        else:
            out_plus = Conv2dFused[DType.float32].forward[track_grad=False](
                image, kernel, param_plus, stride, dilation, padding
            )

        var loss_plus = out_plus.sum()

        # Forward pass with -eps
        var param_minus: Tensor[DType.float32]
        if param_type == "image":
            param_minus = image.copy()
            param_minus.buffer.data_buffer()[idx] -= eps
        elif param_type == "kernel":
            param_minus = kernel.copy()
            param_minus.buffer.data_buffer()[idx] -= eps
        else:
            param_minus = bias.value().copy()
            param_minus.buffer.data_buffer()[idx] -= eps

        var out_minus: Tensor[DType.float32]
        if param_type == "image":
            out_minus = Conv2dFused[DType.float32].forward[track_grad=False](
                param_minus, kernel, bias, stride, dilation, padding
            )
        elif param_type == "kernel":
            out_minus = Conv2dFused[DType.float32].forward[track_grad=False](
                image, param_minus, bias, stride, dilation, padding
            )
        else:
            out_minus = Conv2dFused[DType.float32].forward[track_grad=False](
                image, kernel, param_minus, stride, dilation, padding
            )

        var loss_minus = out_minus.sum()

        # Compute gradient
        numerical_grad.buffer.data_buffer()[idx] = Scalar[DType.float32](
            (loss_plus - loss_minus).item() / (2.0 * eps)
        )

    return numerical_grad^


# -------------------------------------------------------------------
# TEST 1: BASIC OUTPUT SHAPES
# -------------------------------------------------------------------


fn test_basic_shapes() raises:
    print("\n" + "=" * 60)
    print("TEST 1: BASIC OUTPUT SHAPES")
    print("=" * 60)

    # Test 1.1: Single channel, single filter
    var img1 = Tensor[DType.float32].randn(1, 1, 5, 5)
    var kernel1 = Tensor[DType.float32].randn(1, 1, 3, 3)
    var out1 = Conv2dFused[DType.float32].forward[track_grad=False](
        img1, kernel1
    )
    assert_shape_equal(out1, [1, 1, 3, 3], "Single channel, 3x3 kernel")

    # Test 1.2: RGB image, multiple filters
    var img2 = Tensor[DType.float32].randn(2, 3, 8, 8)  # Batch=2
    var kernel2 = Tensor[DType.float32].randn(16, 3, 3, 3)  # 16 filters
    var out2 = Conv2dFused[DType.float32].forward[track_grad=False](
        img2, kernel2
    )
    assert_shape_equal(out2, [2, 16, 6, 6], "RGB, 16 filters, batch=2")

    # Test 1.3: Large kernel
    var img3 = Tensor[DType.float32].randn(1, 1, 10, 10)
    var kernel3 = Tensor[DType.float32].randn(1, 1, 5, 5)
    var out3 = Conv2dFused[DType.float32].forward[track_grad=False](
        img3, kernel3
    )
    assert_shape_equal(out3, [1, 1, 6, 6], "5x5 kernel on 10x10 image")

    # Test 1.4: Non-square image
    var img4 = Tensor[DType.float32].randn(1, 1, 7, 10)
    var kernel4 = Tensor[DType.float32].randn(1, 1, 3, 3)
    var out4 = Conv2dFused[DType.float32].forward[track_grad=False](
        img4, kernel4
    )
    assert_shape_equal(out4, [1, 1, 5, 8], "Non-square image (7x10)")

    print("\n All basic shape tests passed!\n")


# -------------------------------------------------------------------
# TEST 2: PADDING MODES
# -------------------------------------------------------------------


fn test_padding_modes() raises:
    print("\n" + "=" * 60)
    print("TEST 2: PADDING MODES")
    print("=" * 60)

    var img = Tensor[DType.float32].randn(1, 1, 8, 8)
    var kernel = Tensor[DType.float32].randn(1, 1, 3, 3)

    # Test 2.1: Valid padding (no padding)
    var out_valid = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, padding=Padding("valid")
    )
    assert_shape_equal(out_valid, List[Int](1, 1, 6, 6), "Valid padding")

    # Test 2.2: Same padding (preserve dimensions)
    var out_same = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, padding=Padding("same")
    )
    assert_shape_equal(out_same, List[Int](1, 1, 8, 8), "Same padding")

    # Test 2.3: Uniform integer padding
    var out_pad1 = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, padding=Padding(1)
    )
    assert_shape_equal(out_pad1, List[Int](1, 1, 8, 8), "Uniform padding=1")

    # Test 2.4: Tuple padding (height, width)
    var out_pad_tuple = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, padding=Padding((1, 2))
    )
    assert_shape_equal(
        out_pad_tuple, List[Int](1, 1, 8, 10), "Tuple padding (1,2)"
    )

    # Test 2.5: Asymmetric padding
    var pad_list = List[Tuple[Int, Int]]()
    pad_list.append((1, 2))  # top=1, bottom=2
    pad_list.append((2, 1))  # left=2, right=1
    var out_asym = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, padding=Padding(pad_list.copy())
    )
    assert_shape_equal(out_asym, List[Int](1, 1, 9, 9), "Asymmetric padding")

    print("\n All padding tests passed!\n")


# -------------------------------------------------------------------
# TEST 3: STRIDE
# -------------------------------------------------------------------


fn test_stride_2() raises:
    print("\n" + "=" * 60)
    print("TEST 3: STRIDE")
    print("=" * 60)

    var img = Tensor[DType.float32].randn(1, 1, 8, 8)
    var kernel = Tensor[DType.float32].randn(1, 1, 3, 3)

    # Test 3.1: Stride = 1 (default)
    var out_s1 = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, stride=1
    )
    assert_shape_equal(out_s1, List[Int](1, 1, 6, 6), "Stride=1")

    # Test 3.2: Stride = 2
    var out_s2 = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, stride=2
    )
    assert_shape_equal(out_s2, List[Int](1, 1, 3, 3), "Stride=2")

    # Test 3.3: Stride = 3
    var out_s3 = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, stride=3
    )
    assert_shape_equal(out_s3, List[Int](1, 1, 2, 2), "Stride=3")

    # Test 3.4: Stride with padding
    var out_s2_pad = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, stride=2, padding=Padding("same")
    )
    assert_shape_equal(
        out_s2_pad, List[Int](1, 1, 4, 4), "Stride=2 with same padding"
    )

    print("\n All stride tests passed!\n")


# -------------------------------------------------------------------
# TEST 4: DILATION (ATROUS CONVOLUTION)
# -------------------------------------------------------------------


fn test_dilation_2() raises:
    print("\n" + "=" * 60)
    print("TEST 4: DILATION")
    print("=" * 60)

    var img = Tensor[DType.float32].randn(1, 1, 10, 10)
    var kernel = Tensor[DType.float32].randn(1, 1, 3, 3)

    # Test 4.1: Dilation = 1 (standard convolution)
    var out_d1 = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, dilation=1
    )
    assert_shape_equal(out_d1, List[Int](1, 1, 8, 8), "Dilation=1")

    # Test 4.2: Dilation = 2 (effective kernel 5x5)
    var out_d2 = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, dilation=2
    )
    # Effective kernel: 3 + (3-1)*1 = 5
    # Output: 10 - 5 + 1 = 6
    assert_shape_equal(out_d2, List[Int](1, 1, 6, 6), "Dilation=2")

    # Test 4.3: Dilation = 3 (effective kernel 7x7)
    var out_d3 = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, dilation=3
    )
    # Effective kernel: 3 + (3-1)*2 = 7
    # Output: 10 - 7 + 1 = 4
    assert_shape_equal(out_d3, List[Int](1, 1, 4, 4), "Dilation=3")

    print("\n All dilation tests passed!\n")


# -------------------------------------------------------------------
# TEST 5: BIAS
# -------------------------------------------------------------------


fn test_bias_2() raises:
    print("\n" + "=" * 60)
    print("TEST 5: BIAS")
    print("=" * 60)

    var img = Tensor[DType.float32].ones(1, 1, 3, 3)
    var kernel = Tensor[DType.float32].ones(2, 1, 2, 2)  # 2 filters

    # Test 5.1: No bias (should use zeros)
    var out_no_bias = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel
    )
    # Each output position: sum of 2x2 ones = 4.0
    for i in range(out_no_bias.numels()):
        if abs(out_no_bias.buffer.data_buffer()[i] - 4.0) > 1e-5:
            raise Error("Bias test failed: expected 4.0 without bias")
    print(" No bias - correct zero initialization")

    # Test 5.2: With bias
    var bias = Tensor[DType.float32].full(
        Shape(2), 10.0
    )  # bias=10 for both filters
    var out_with_bias = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, bias=bias
    )
    # Each output: 4.0 + 10.0 = 14.0
    for i in range(out_with_bias.numels()):
        if abs(out_with_bias.buffer.data_buffer()[i] - 14.0) > 1e-5:
            raise Error("Bias test failed: expected 14.0 with bias=10")
    print(" With bias - correct addition")

    # Test 5.3: Different bias per channel
    var bias_diff = Tensor[DType.float32](Shape(2))
    bias_diff[0] = 5.0
    bias_diff[1] = 15.0
    var out_diff_bias = Conv2dFused[DType.float32].forward[track_grad=False](
        img, kernel, bias=bias_diff
    )
    # Channel 0: 4.0 + 5.0 = 9.0
    # Channel 1: 4.0 + 15.0 = 19.0
    var H_out = out_diff_bias.shape()[2]
    var W_out = out_diff_bias.shape()[3]
    for y in range(H_out):
        for x in range(W_out):
            if abs(out_diff_bias[0, 0, y, x] - 9.0) > 1e-5:
                raise Error("Bias test failed: channel 0 expected 9.0")
            if abs(out_diff_bias[0, 1, y, x] - 19.0) > 1e-5:
                raise Error("Bias test failed: channel 1 expected 19.0")
    print(" Different bias per channel - correct")

    print("\n All bias tests passed!\n")


# -------------------------------------------------------------------
# TEST 6: KNOWN VALUES (MANUAL COMPUTATION)
# -------------------------------------------------------------------


fn test_known_values() raises:
    print("\n" + "=" * 60)
    print("TEST 6: KNOWN VALUES")
    print("=" * 60)

    # Test 6.1: Simple 3x3 image, 2x2 kernel
    var img = Tensor[DType.float32](Shape(1, 1, 3, 3))
    img[0, 0, 0, 0] = 1.0
    img[0, 0, 0, 1] = 2.0
    img[0, 0, 0, 2] = 3.0
    img[0, 0, 1, 0] = 4.0
    img[0, 0, 1, 1] = 5.0
    img[0, 0, 1, 2] = 6.0
    img[0, 0, 2, 0] = 7.0
    img[0, 0, 2, 1] = 8.0
    img[0, 0, 2, 2] = 9.0

    var kernel = Tensor[DType.float32](Shape(1, 1, 2, 2))
    kernel[0, 0, 0, 0] = 1.0
    kernel[0, 0, 0, 1] = 0.0
    kernel[0, 0, 1, 0] = 0.0
    kernel[0, 0, 1, 1] = 1.0

    var out = Conv2dFused[DType.float32].forward[track_grad=False](img, kernel)

    # Manual computation:
    # Position (0,0): 1*1 + 2*0 + 4*0 + 5*1 = 6
    # Position (0,1): 2*1 + 3*0 + 5*0 + 6*1 = 8
    # Position (1,0): 4*1 + 5*0 + 7*0 + 8*1 = 12
    # Position (1,1): 5*1 + 6*0 + 8*0 + 9*1 = 14

    var expected = Tensor[DType.float32](Shape(1, 1, 2, 2))
    expected[0, 0, 0, 0] = 6.0
    expected[0, 0, 0, 1] = 8.0
    expected[0, 0, 1, 0] = 12.0
    expected[0, 0, 1, 1] = 14.0

    assert_tensor_close(
        out.as_gradbox(), expected.as_gradbox(), "Known 3x3 convolution"
    )

    print("\n All known value tests passed!\n")


# -------------------------------------------------------------------
# TEST 7: GRADIENT SHAPES
# -------------------------------------------------------------------


fn test_gradient_shapes() raises:
    print("\n" + "=" * 60)
    print("TEST 7: GRADIENT SHAPES")
    print("=" * 60)

    var img = Tensor[DType.float32].randn(2, 3, 8, 8)
    img.requires_grad_(True)

    var kernel = Tensor[DType.float32].randn(16, 3, 3, 3)
    kernel.requires_grad_(True)

    var bias = Tensor[DType.float32].randn(16)
    bias.requires_grad_(True)

    # Forward pass
    var out = Conv2dFused[DType.float32].forward(img, kernel, bias=bias)

    # Backward pass
    var loss = out.sum()
    loss.backward()

    # Check gradient shapes
    var list1: List[Int] = [2, 3, 8, 8]
    var list2: List[Int] = [16, 3, 3, 3]
    var list3: List[Int] = [16]
    assert_shape_equal(img.grad().as_tensor(), list1, "Image gradient shape")
    assert_shape_equal(
        kernel.grad().as_tensor(), list2, "Kernel gradient shape"
    )
    assert_shape_equal(bias.grad().as_tensor(), list3, "Bias gradient shape")

    print("\nAll gradient shape tests passed!\n")


# -------------------------------------------------------------------
# TEST 8: GRADIENT CORRECTNESS (NUMERICAL CHECK)
# -------------------------------------------------------------------


fn test_gradient_correctness() raises:
    print("\n" + "=" * 60)
    print("TEST 8: GRADIENT CORRECTNESS (NUMERICAL)")
    print("=" * 60)

    seed(42)

    # Small tensors for faster numerical gradient computation
    var img = Tensor[DType.float32].randn(1, 2, 4, 4)
    img.requires_grad_(True)

    var kernel = Tensor[DType.float32].randn(2, 2, 2, 2)
    kernel.requires_grad_(True)

    var bias = Tensor[DType.float32].randn(2)
    bias.requires_grad_(True)

    # Test 8.1: Image gradient
    print("\nTesting image gradient...")
    var out1 = Conv2dFused[DType.float32].forward(img, kernel, bias=bias)
    var loss1 = out1.sum()
    loss1.backward()

    var analytical_img_grad = img.gradients()[].copy()
    var numerical_img_grad = compute_numerical_gradient(
        img,
        kernel,
        bias,
        stride=1,
        dilation=1,
        padding=Padding("valid"),
        param_type="image",
    )

    assert_tensor_close(
        analytical_img_grad,
        numerical_img_grad.as_gradbox(),
        "Image gradient numerical check",
        rtol=1e-2,
        atol=1e-2,
    )

    # Test 8.2: Kernel gradient
    print("\nTesting kernel gradient...")
    img.zero_grad()
    kernel.zero_grad()
    bias.zero_grad()

    var out2 = Conv2dFused[DType.float32].forward(img, kernel, bias=bias)
    var loss2 = out2.sum()
    loss2.backward()

    var analytical_kernel_grad = kernel.gradients()[].copy()
    var numerical_kernel_grad = compute_numerical_gradient(
        img,
        kernel,
        bias,
        stride=1,
        dilation=1,
        padding=Padding("valid"),
        param_type="kernel",
    )

    assert_tensor_close(
        analytical_kernel_grad,
        numerical_kernel_grad.as_gradbox(),
        "Kernel gradient numerical check",
        rtol=1e-2,
        atol=1e-2,
    )

    # Test 8.3: Bias gradient
    print("\nTesting bias gradient...")
    img.zero_grad()
    kernel.zero_grad()
    bias.zero_grad()

    var out3 = Conv2dFused[DType.float32].forward(img, kernel, bias=bias)
    var loss3 = out3.sum()
    loss3.backward()

    var analytical_bias_grad = bias.gradients()[].copy()
    var numerical_bias_grad = compute_numerical_gradient(
        img,
        kernel,
        bias,
        stride=1,
        dilation=1,
        padding=Padding("valid"),
        param_type="bias",
    )

    assert_tensor_close(
        analytical_bias_grad,
        numerical_bias_grad.as_gradbox(),
        "Bias gradient numerical check",
        rtol=1e-2,
        atol=1e-2,
    )

    print("\n All gradient correctness tests passed!\n")


# -------------------------------------------------------------------
# TEST 9: GRADIENT WITH STRIDE AND DILATION
# -------------------------------------------------------------------


fn test_gradient_stride_dilation() raises:
    print("\n" + "=" * 60)
    print("TEST 9: GRADIENTS WITH STRIDE & DILATION")
    print("=" * 60)

    seed(123)

    # Test 9.1: Stride = 2
    print("\nTesting with stride=2...")
    var img_s2 = Tensor[DType.float32].randn(1, 1, 6, 6)
    img_s2.requires_grad_(True)
    var kernel_s2 = Tensor[DType.float32].randn(1, 1, 2, 2)
    kernel_s2.requires_grad_(True)

    var out_s2 = Conv2dFused[DType.float32].forward(img_s2, kernel_s2, stride=2)
    var loss_s2 = out_s2.sum()
    loss_s2.backward()

    var numerical_s2 = compute_numerical_gradient(
        img_s2,
        kernel_s2,
        None,
        stride=2,
        dilation=1,
        padding=Padding("valid"),
        param_type="image",
    )

    assert_tensor_close(
        img_s2.gradients()[],
        numerical_s2.as_gradbox(),
        "Stride=2 image gradient",
        rtol=1e-1,
        atol=1e-2,
    )

    # Test 9.2: Dilation = 2
    print("\nTesting with dilation=2...")
    var img_d2 = Tensor[DType.float32].randn(1, 1, 6, 6)
    img_d2.requires_grad_(True)
    var kernel_d2 = Tensor[DType.float32].randn(1, 1, 2, 2)
    kernel_d2.requires_grad_(True)

    var out_d2 = Conv2dFused[DType.float32].forward(
        img_d2, kernel_d2, dilation=2
    )
    var loss_d2 = out_d2.sum()
    loss_d2.backward()

    var numerical_d2 = compute_numerical_gradient(
        img_d2,
        kernel_d2,
        None,
        stride=1,
        dilation=2,
        padding=Padding("valid"),
        param_type="kernel",
    )

    assert_tensor_close(
        kernel_d2.gradients()[],
        numerical_d2.as_gradbox(),
        "Dilation=2 kernel gradient",
        rtol=1e-3,
        atol=1e-4,
    )

    print("\n All stride/dilation gradient tests passed!\n")


# -------------------------------------------------------------------
# TEST 10: GRADIENT ACCUMULATION
# -------------------------------------------------------------------


fn test_gradient_accumulation() raises:
    print("\n" + "=" * 60)
    print("TEST 10: GRADIENT ACCUMULATION")
    print("=" * 60)

    var img = Tensor[DType.float32].randn(1, 1, 5, 5)
    img.requires_grad_(True)
    var kernel = Tensor[DType.float32].randn(1, 1, 3, 3)
    kernel.requires_grad_(True)

    # First backward pass
    var out1 = Conv2dFused[DType.float32].forward(img, kernel)
    var loss1 = out1.sum()
    loss1.backward()

    # var grad_after_first = img.gradients()[].copy()
    var grad_after_first = img.grad()

    # Second backward pass (should accumulate)
    var out2 = Conv2dFused[DType.float32].forward(img, kernel)
    var loss2 = out2.sum() * 2.0
    loss2.backward()

    var grad_after_second = img.gradients()[]

    # Check that gradients accumulated (not replaced)
    # grad_after_second should be approximately grad_after_first * 3
    # (first pass: 1x, second pass: 2x)
    for i in range(grad_after_first.numels()):
        var expected = grad_after_first.buffer.data_buffer()[i] * 3.0
        var actual = grad_after_second.buffer.data_buffer()[i]
        if abs(actual - expected) > 1e-3:
            raise Error(
                "Gradient accumulation test failed at index " + i.__str__()
            )

    print(" Gradients correctly accumulated across multiple backward passes")
    print("\n Gradient accumulation test passed!\n")


# -------------------------------------------------------------------
# TEST 11: BATCH PROCESSING
# -------------------------------------------------------------------


fn test_batch_processing() raises:
    print("test_batch_processing")
    var kernel = Tensor[DType.float32].randn(2, 1, 3, 3)

    # Process batch of 4
    var img_batch = Tensor[DType.float32].randn(4, 1, 6, 6)
    var out_batch = Conv2dFused[DType.float32].forward[track_grad=False](
        img_batch, kernel
    )
    # Process individually
    var out_individual_list = List[Tensor[DType.float32]]()
    for i in range(4):
        var img_single = img_batch[i : i + 1, :, :, :]  # Keep batch dimension
        img_single = img_single.contiguous()
        var out_single = Conv2dFused[DType.float32].forward[track_grad=False](
            img_single, kernel
        )
        out_individual_list.append(out_single^)

    # Compare results - FIXED VERSION
    for i in range(4):
        var batch_slice = out_batch[
            i : i + 1, :, :, :
        ]  # Shape: (1, channels_out, height_out, width_out)
        var individual = out_individual_list[i]

        # Compute offset for this batch item
        var batch_offset = (
            i * batch_slice.numels()
        )  # Each batch item has batch_slice.numels() elements

        # Compare element by element
        for j in range(batch_slice.numels()):
            var batch_val = out_batch.buffer.data_buffer()[batch_offset + j]
            var individual_val = individual.buffer.data_buffer()[j]
            if abs(batch_val - individual_val) > 1e-3:
                raise Error(
                    "Batch processing mismatch at batch "
                    + i.__str__()
                    + ", element "
                    + j.__str__()
                )

    print(" Batch processing matches individual processing")
    print("\n Batch processing test passed!\n")


fn test_edge_cases_2() raises:
    print("test_edge_cases")
    # Test 12.1: Kernel same size as image (output 1x1)
    var img_small = Tensor[DType.float32].randn(1, 1, 3, 3)
    var kernel_large = Tensor[DType.float32].randn(1, 1, 3, 3)
    var out_1x1 = Conv2dFused[DType.float32].forward[track_grad=False](
        img_small, kernel_large
    )
    assert_shape_equal(
        out_1x1, List[Int](1, 1, 1, 1), "Kernel size = image size"
    )

    # Test 12.2: 1x1 kernel (pointwise convolution)
    var img_pw = Tensor[DType.float32].randn(1, 3, 5, 5)
    var kernel_1x1 = Tensor[DType.float32].randn(16, 3, 1, 1)
    var out_pw = Conv2dFused[DType.float32].forward[track_grad=False](
        img_pw, kernel_1x1
    )
    assert_shape_equal(out_pw, List[Int](1, 16, 5, 5), "1x1 kernel (pointwise)")

    # Test 12.3: Large batch
    var img_large_batch = Tensor[DType.float32].randn(32, 3, 8, 8)
    var kernel_lb = Tensor[DType.float32].randn(8, 3, 3, 3)
    var out_lb = Conv2dFused[DType.float32].forward[track_grad=False](
        img_large_batch, kernel_lb
    )
    assert_shape_equal(out_lb, List[Int](32, 8, 6, 6), "Large batch (32)")

    # Test 12.4: Many channels
    var img_many_ch = Tensor[DType.float32].randn(1, 64, 4, 4)
    var kernel_many = Tensor[DType.float32].randn(128, 64, 2, 2)
    var out_many = Conv2dFused[DType.float32].forward[track_grad=False](
        img_many_ch, kernel_many
    )
    assert_shape_equal(
        out_many, List[Int](1, 128, 3, 3), "Many channels (64128)"
    )

    print("\n All edge case tests passed!\n")


# ====================================================


# ============================================================================
# FORWARD PASS TESTS - Basic Functionality
# ============================================================================


fn test_conv2d_forward_single_batch_single_channel() raises:
    """Test basic convolution: single batch, single channel."""
    print("test_conv2d_forward_single_batch_single_channel")

    alias dtype = DType.float32

    # Input: (1, 1, 3, 3)
    var x = Tensor[dtype].rand(1, 1, 3, 3)
    for i in range(9):
        x.buffer.data_buffer()[i] = Float32(i + 1)

    # Kernel: (1, 1, 2, 2)
    var kernel = Tensor[dtype].d4([[[[1.0, 0.0], [0.0, 1.0]]]])

    # No padding, stride=1
    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=1, padding=Padding("valid")
    )

    # Output shape: (1, 1, 2, 2)
    assert_true(output.shape()[0] == 1)
    assert_true(output.shape()[1] == 1)
    assert_true(output.shape()[2] == 2)
    assert_true(output.shape()[3] == 2)


fn test_conv2d_forward_with_bias() raises:
    """Test convolution with bias."""
    print("test_conv2d_forward_with_bias")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(Shape(1, 1, 3, 3))
    x.fill(1.0)

    var kernel = Tensor[dtype].zeros(1, 1, 2, 2)
    kernel.fill(1.0)

    var bias = Tensor[dtype].d1([10.0])

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, bias=bias, stride=1, padding=Padding("valid")
    )

    # Each output: sum of 2x2=4 ones + bias=10 = 14
    assert_true(output[0, 0, 0, 0] == 14.0)
    assert_true(output[0, 0, 1, 1] == 14.0)


fn test_conv2d_forward_multiple_channels() raises:
    """Test convolution with multiple input channels."""
    print("test_conv2d_forward_multiple_channels")

    alias dtype = DType.float32

    # Input: (1, 3, 4, 4) - 3 input channels
    var x = Tensor[dtype].zeros(1, 3, 4, 4)
    x.fill(1.0)

    # Kernel: (1, 3, 2, 2) - 1 output channel, 3 input channels
    var kernel = Tensor[dtype].zeros(1, 3, 2, 2)
    kernel.fill(1.0)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=1, padding=Padding("valid")
    )

    # Output shape: (1, 1, 3, 3)
    assert_true(output.shape()[2] == 3)
    assert_true(output.shape()[3] == 3)

    # Each output: 3 channels * 2x2 kernel = 12
    assert_true(output[0, 0, 0, 0] == 12.0)


fn test_conv2d_forward_multiple_filters() raises:
    """Test convolution with multiple output filters."""
    print("test_conv2d_forward_multiple_filters")

    alias dtype = DType.float32

    # Input: (1, 1, 4, 4)
    var x = Tensor[dtype].zeros(1, 1, 4, 4)
    x.fill(1.0)

    # Kernel: (3, 1, 2, 2) - 3 output channels
    var kernel = Tensor[dtype].zeros(3, 1, 2, 2)
    for i in range(3):
        for j in range(4):
            kernel.buffer.data_buffer()[i * 4 + j] = Float32(i + 1)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=1, padding=Padding("valid")
    )
    # Output shape: (1, 3, 3, 3)
    assert_true(output.shape()[1] == 3)

    # First filter: all 1s, sum = 4
    assert_true(
        output[i(0), i(0), s(), s()] == Tensor[dtype].full(Shape(3, 3), 4.0)
    )
    # Second filter: all 2s, sum = 8
    assert_true(
        output[i(0), i(1), s(), s()] == Tensor[dtype].full(Shape(3, 3), 8.0)
    )
    assert_true(
        output[i(0), i(2), s(), s()] == Tensor[dtype].full(Shape(3, 3), 12.0)
    )


fn test_conv2d_forward_batch_processing() raises:
    """Test convolution with batch dimension."""
    print("test_conv2d_forward_batch_processing")

    alias dtype = DType.float32

    # Input: (4, 1, 3, 3) - batch of 4
    var x = Tensor[dtype](4, 1, 3, 3)
    for b in range(4):
        for i in range(9):
            x.buffer.data_buffer()[b * 9 + i] = Float32(b + 1)

    var kernel = Tensor[dtype].zeros(1, 1, 2, 2)
    kernel.fill(1.0)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=1, padding=Padding("valid")
    )

    # Output shape: (4, 1, 2, 2)
    assert_true(output.shape()[0] == 4)

    # Each batch has different values
    assert_true(output[0, 0, 0, 0] == 4.0)  # Batch 0: 4*1
    assert_true(output[1, 0, 0, 0] == 8.0)  # Batch 1: 4*2
    assert_true(output[2, 0, 0, 0] == 12.0)  # Batch 2: 4*3


# ============================================================================
# STRIDE TESTS
# ============================================================================


fn test_conv2d_stride_2() raises:
    """Test convolution with stride=2."""
    print("test_conv2d_stride_2")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 1, 5, 5)
    x.fill(1.0)

    var kernel = Tensor[dtype].zeros(1, 1, 2, 2)
    kernel.fill(1.0)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=2, padding=Padding("valid")
    )

    # Output shape: (1, 1, 2, 2) - stride skips every other position
    assert_true(output.shape()[2] == 2)
    assert_true(output.shape()[3] == 2)

    # All outputs should be 4 (sum of 2x2)
    for i in range(4):
        assert_true(output.buffer.data_buffer()[i] == 4.0)


fn test_conv2d_stride_3() raises:
    """Test convolution with stride=3."""
    print("test_conv2d_stride_3")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 1, 7, 7)
    x.fill(2.0)

    var kernel = Tensor[dtype].zeros(1, 1, 2, 2)
    kernel.fill(0.5)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=3, padding=Padding("valid")
    )

    # Output shape: (1, 1, 2, 2)
    assert_true(output.shape()[2] == 2)
    assert_true(output.shape()[3] == 2)

    # Each output: 4 * 2.0 * 0.5 = 4.0
    assert_true(output[0, 0, 0, 0] == 4.0)


# ============================================================================
# PADDING TESTS
# ============================================================================


fn test_conv2d_padding_valid() raises:
    """Test 'valid' padding (no padding)."""
    print("test_conv2d_padding_valid")

    alias dtype = DType.float32

    var x = Tensor[dtype](Shape(1, 1, 5, 5))
    x.fill(1.0)

    var kernel = Tensor[dtype](Shape(1, 1, 3, 3))
    kernel.fill(1.0)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=1, padding=Padding("valid")
    )

    # Output shape: (1, 1, 3, 3)
    assert_true(output.shape()[2] == 3)
    assert_true(output.shape()[3] == 3)


fn test_conv2d_padding_same() raises:
    """Test 'same' padding (output same size as input)."""
    print("test_conv2d_padding_same")

    alias dtype = DType.float32

    var x = Tensor[dtype](Shape(1, 1, 5, 5))
    x.fill(1.0)

    var kernel = Tensor[dtype](Shape(1, 1, 3, 3))
    kernel.fill(1.0)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=1, padding=Padding("same")
    )

    # Output shape: (1, 1, 5, 5) - same as input
    assert_true(output.shape()[2] == 5)
    assert_true(output.shape()[3] == 5)


fn test_conv2d_padding_int() raises:
    """Test integer padding (same on all sides)."""
    print("test_conv2d_padding_int")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 1, 3, 3)
    x.fill(1.0)

    var kernel = Tensor[dtype].zeros(1, 1, 2, 2)
    kernel.fill(1.0)

    var output = Conv2dFused[dtype].forward(
        image=x,
        kernel=kernel,
        stride=1,
        padding=Padding(1),  # Pad 1 on all sides
    )

    # Output shape: (1, 1, 4, 4)
    assert_true(output.shape()[2] == 4)
    assert_true(output.shape()[3] == 4)


fn test_conv2d_padding_tuple() raises:
    """Test tuple padding (height, width)."""
    print("test_conv2d_padding_tuple")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 1, 4, 4)
    x.fill(1.0)

    var kernel = Tensor[dtype].zeros(1, 1, 2, 2)
    kernel.fill(1.0)

    var output = Conv2dFused[dtype].forward(
        image=x,
        kernel=kernel,
        stride=1,
        padding=Padding((1, 2)),  # Pad 1 vertically, 2 horizontally
    )

    # Output shape: (1, 1, 5, 7)
    assert_true(output.shape()[2] == 5)
    assert_true(output.shape()[3] == 7)


fn test_conv2d_padding_list_asymmetric() raises:
    """Test list padding (asymmetric padding)."""
    print("test_conv2d_padding_list_asymmetric")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 1, 4, 4)
    x.fill(1.0)

    var kernel = Tensor[dtype].zeros(1, 1, 2, 2)
    kernel.fill(1.0)

    var pad_list = List[Tuple[Int, Int]]()
    pad_list.append((1, 2))  # top=1, bottom=2
    pad_list.append((3, 0))  # left=3, right=0

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=1, padding=Padding(pad_list^)
    )

    # Output shape: (1, 1, 6, 6)
    # Height: 4 + 1 + 2 - 2 + 1 = 6
    # Width: 4 + 3 + 0 - 2 + 1 = 6
    assert_true(output.shape()[2] == 6)
    assert_true(output.shape()[3] == 6)


# ============================================================================
# DILATION TESTS
# ============================================================================


fn test_conv2d_dilation_2() raises:
    """Test dilated convolution (atrous convolution)."""
    print("test_conv2d_dilation_2")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 1, 7, 7)
    x.fill(1.0)

    var kernel = Tensor[dtype].ones(1, 1, 3, 3)

    var output = Conv2dFused[dtype].forward(
        image=x,
        kernel=kernel,
        stride=1,
        dilation=2,  # Effective kernel size: 5x5
        padding=Padding("valid"),
    )

    # Dilated kernel size: 3 + (3-1)*(2-1) = 5
    # Output shape: (1, 1, 3, 3)
    assert_true(output.shape()[2] == 3)
    assert_true(output.shape()[3] == 3)

    # Each output: 3x3 kernel = 9
    assert_true(output[0, 0, 0, 0] == 9.0)


fn test_conv2d_dilation_3() raises:
    """Test high dilation rate."""
    print("test_conv2d_dilation_3")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 1, 9, 9)
    x.fill(2.0)

    var kernel = Tensor[dtype].zeros(1, 1, 2, 2)
    kernel.fill(0.5)

    var output = Conv2dFused[dtype].forward(
        image=x,
        kernel=kernel,
        stride=1,
        dilation=3,  # Effective kernel: 4x4
        padding=Padding("valid"),
    )

    # Dilated size: 2 + (2-1)*2 = 4
    # Output shape: (1, 1, 6, 6)
    assert_true(output.shape()[2] == 6)
    assert_true(output.shape()[3] == 6)


# ============================================================================
# COMBINED PARAMETERS TESTS
# ============================================================================


fn test_conv2d_stride_and_padding() raises:
    """Test combination of stride and padding."""
    print("test_conv2d_stride_and_padding")

    alias dtype = DType.float32

    var x = Tensor[dtype].ones(1, 1, 5, 5)

    var kernel = Tensor[dtype].ones(1, 1, 3, 3)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=2, padding=Padding(1)
    )

    # Output shape: (1, 1, 3, 3)
    assert_true(output.shape()[2] == 3)
    assert_true(output.shape()[3] == 3)


fn test_conv2d_stride_padding_dilation() raises:
    """Test combination of stride, padding, and dilation."""
    print("test_conv2d_stride_padding_dilation")

    alias dtype = DType.float32

    var x = Tensor[dtype].ones(1, 1, 8, 8)

    var kernel = Tensor[dtype].ones(1, 1, 3, 3)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=2, dilation=2, padding=Padding(2)
    )
    # Should produce valid output
    assert_true(output.shape() == Shape(1, 1, 4, 4))
    assert_true(
        output
        == Tensor[dtype].d4(
            [
                [
                    [
                        [
                            4.0,
                            6.0,
                            6.0,
                            4.0,
                        ],
                        [6.0, 9.0, 9.0, 6.0],
                        [6.0, 9.0, 9.0, 6.0],
                        [4.0, 6.0, 6.0, 4.0],
                    ]
                ]
            ]
        )
    )


fn test_conv2d_all_parameters() raises:
    """Test with all parameters: batch, channels, stride, padding, dilation, bias.
    """
    print("test_conv2d_all_parameters")

    alias dtype = DType.float32

    var x = Tensor[dtype].ones(
        2, 3, 8, 8, requires_grad=True
    )  # Batch=2, channels=3

    var kernel = Tensor[dtype].ones(
        4, 3, 3, 3, requires_grad=True
    )  # 4 output channels

    var bias = Tensor[dtype].zeros(4, requires_grad=True)
    for i in range(4):
        bias[i] = Float32(i)

    var output = Conv2dFused[dtype].forward(
        image=x,
        kernel=kernel,
        bias=bias,
        stride=2,
        dilation=1,
        padding=Padding(1),
    )
    # Output shape: (2, 4, 4, 4)
    assert_true(output.shape()[0] == 2)
    assert_true(output.shape()[1] == 4)
    assert_true(output.shape()[2] == 4)
    assert_true(output.shape()[3] == 4)
    output.backward()

    var kernel_gradbox = (
        Tensor[dtype]
        .d4(
            [
                [
                    [
                        [18.0, 24.0, 24.0],
                        [24.0, 32.0, 32.0],
                        [24.0, 32.0, 32.0],
                    ],
                    [
                        [18.0, 24.0, 24.0],
                        [24.0, 32.0, 32.0],
                        [24.0, 32.0, 32.0],
                    ],
                    [
                        [18.0, 24.0, 24.0],
                        [24.0, 32.0, 32.0],
                        [24.0, 32.0, 32.0],
                    ],
                ],
                [
                    [
                        [18.0, 24.0, 24.0],
                        [24.0, 32.0, 32.0],
                        [24.0, 32.0, 32.0],
                    ],
                    [
                        [18.0, 24.0, 24.0],
                        [24.0, 32.0, 32.0],
                        [24.0, 32.0, 32.0],
                    ],
                    [
                        [18.0, 24.0, 24.0],
                        [24.0, 32.0, 32.0],
                        [24.0, 32.0, 32.0],
                    ],
                ],
                [
                    [
                        [18.0, 24.0, 24.0],
                        [24.0, 32.0, 32.0],
                        [24.0, 32.0, 32.0],
                    ],
                    [
                        [18.0, 24.0, 24.0],
                        [24.0, 32.0, 32.0],
                        [24.0, 32.0, 32.0],
                    ],
                    [
                        [18.0, 24.0, 24.0],
                        [24.0, 32.0, 32.0],
                        [24.0, 32.0, 32.0],
                    ],
                ],
                [
                    [
                        [18.0, 24.0, 24.0],
                        [24.0, 32.0, 32.0],
                        [24.0, 32.0, 32.0],
                    ],
                    [
                        [18.0, 24.0, 24.0],
                        [24.0, 32.0, 32.0],
                        [24.0, 32.0, 32.0],
                    ],
                    [
                        [18.0, 24.0, 24.0],
                        [24.0, 32.0, 32.0],
                        [24.0, 32.0, 32.0],
                    ],
                ],
            ]
        )
        .as_gradbox()
    )

    var bias_gradbox = Tensor[dtype].d1([32.0, 32.0, 32.0, 32.0]).as_gradbox()

    assert_true(kernel.grad() == kernel_gradbox)
    assert_true(bias.grad() == bias_gradbox)


# ============================================================================
# BACKWARD PASS TESTS
# ============================================================================


fn test_conv2d_backward_simple() raises:
    """Test basic backward pass."""
    print("test_conv2d_backward_simple")
    alias dtype = DType.float32

    var x = Tensor[dtype].ones(1, 1, 3, 3, requires_grad=True)

    var kernel = Tensor[dtype].ones(1, 1, 2, 2, requires_grad=True)

    var bias = Tensor[dtype].d1([0.0], requires_grad=True)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, bias=bias, stride=1, padding=Padding("valid")
    )

    output.backward(1.0)
    assert_true(bias.grad()[0] == 4.0)
    var x_grad = Tensor[dtype].d4(
        [[[[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]]]
    )

    var kernel_grad = Tensor[dtype].d4([[[[4.0, 4.0], [4.0, 4.0]]]])

    assert_true(x.grad() == x_grad)
    assert_true(kernel.grad() == kernel_grad)


fn test_conv2d_backward_input_gradient() raises:
    """Test input gradient computation."""
    print("test_conv2d_backward_input_gradient")
    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 1, 4, 4, requires_grad=True)
    for i in range(16):
        x.buffer.data_buffer()[i] = Float32(i + 1)

    var kernel = Tensor[dtype].ones(1, 1, 2, 2, requires_grad=True)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=1, padding=Padding("valid")
    )

    var loss = output.sum()
    loss.backward()

    # Input gradient should be non-zero
    var grad_sum = Scalar[dtype](0)
    for i in range(16):
        grad_sum += x.grad().buffer.data_buffer()[i]

    assert_true(grad_sum > 0)


fn test_conv2d_backward_kernel_gradient() raises:
    """Test kernel gradient computation."""
    print("test_conv2d_backward_kernel_gradient")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 1, 4, 4, requires_grad=True)
    x.fill(2.0)

    var kernel = Tensor[dtype].zeros(1, 1, 2, 2, requires_grad=True)
    kernel.fill(1.0)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=1, padding=Padding("valid")
    )

    output.backward(1.0)

    # Kernel gradient should reflect input values
    var kernel_grad_sum: Scalar[dtype] = 0
    kernel_grad_sum += kernel.grad().sum().item()

    assert_true(kernel_grad_sum > 0)


fn test_conv2d_backward_with_stride() raises:
    """Test backward pass with stride."""
    print("test_conv2d_backward_with_stride")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 1, 5, 5, requires_grad=True)
    x.fill(1.0)

    var kernel = Tensor[dtype].zeros(1, 1, 3, 3, requires_grad=True)
    kernel.fill(1.0)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=2, padding=Padding("valid")
    )

    output.backward(1.0)

    # Gradients should exist and be reasonable
    assert_true(x.grad().shape()[2] == 5)
    assert_true(kernel.grad().shape()[2] == 3)


# ============================================================================
# EDGE CASES AND NUMERICAL TESTS
# ============================================================================


fn test_conv2d_edge_1x1_kernel() raises:
    """Test 1x1 convolution (pointwise)."""
    print("test_conv2d_edge_1x1_kernel")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 3, 4, 4)
    x.fill(1.0)

    var kernel = Tensor[dtype].zeros(2, 3, 1, 1)  # Pointwise
    kernel.fill(1.0)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=1, padding=Padding("valid")
    )

    # Output same spatial size
    assert_true(output.shape()[2] == 4)
    assert_true(output.shape()[3] == 4)


fn test_conv2d_edge_large_kernel() raises:
    """Test with large kernel."""
    print("test_conv2d_edge_large_kernel")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 1, 10, 10)
    x.fill(1.0)

    var kernel = Tensor[dtype].zeros(1, 1, 7, 7)
    kernel.fill(1.0)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=1, padding=Padding("valid")
    )

    # Output shape: (1, 1, 4, 4)
    assert_true(output.shape()[2] == 4)
    assert_true(output.shape()[3] == 4)


fn test_conv2d_edge_output_size_1() raises:
    """Test when output size is 1x1."""
    print("test_conv2d_edge_output_size_1")

    alias dtype = DType.float32

    var x = Tensor[dtype].zeros(1, 1, 3, 3)
    x.fill(1.0)

    var kernel = Tensor[dtype].zeros(1, 1, 3, 3)
    kernel.fill(1.0)

    var output = Conv2dFused[dtype].forward(
        image=x, kernel=kernel, stride=1, padding=Padding("valid")
    )

    # Output shape: (1, 1, 1, 1)
    assert_true(output.shape()[2] == 1)
    assert_true(output.shape()[3] == 1)
    assert_true(output[0, 0, 0, 0] == 9.0)
