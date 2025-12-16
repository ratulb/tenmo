from tenmo import Tensor
from random import seed
from testing import assert_true
from net import Dropout

# ============================================================================
# FORWARD PASS TESTS
# ============================================================================

fn test_dropout_forward_training_mode() raises:
    """Test dropout forward pass in training mode - should zero out some elements."""
    print("test_dropout_forward_training_mode")

    seed(42)
    var dropout = Dropout[DType.float32](p=0.5)
    dropout.train()

    var x = Tensor[DType.float32](100)
    for i in range(100):
        x[i] = 1.0

    var out = dropout(x)

    # Count zeros and non-zeros
    var num_zeros = 0
    var num_nonzeros = 0
    for i in range(100):
        if out[i] == 0.0:
            num_zeros += 1
        else:
            num_nonzeros += 1

    # Should have some zeros and some non-zeros
    assert_true(num_zeros > 0)
    assert_true(num_nonzeros > 0)

    # Non-zero values should be scaled by 2.0 (= 1/(1-0.5))
    for i in range(100):
        if out[i] != 0.0:
            var expected = Tensor[DType.float32](1)
            expected[0] = 2.0
            var actual = Tensor[DType.float32](1)
            actual[0] = out[i]
            assert_true(actual.all_close[atol=1e-6](expected))


fn test_dropout_forward_eval_mode() raises:
    """Test dropout forward pass in eval mode - should pass through unchanged."""
    print("test_dropout_forward_eval_mode")

    var dropout = Dropout[DType.float32](p=0.5)
    dropout.eval()

    var x = Tensor[DType.float32](5, 4)
    for i in range(20):
        x.buffer.buffer[i] = Float32(i) + 1.0

    var out = dropout(x)

    # In eval mode, output should be identical to input
    assert_true(out.all_close[atol=1e-6](x))


fn test_dropout_forward_preserves_shape() raises:
    """Test that dropout preserves tensor shape."""
    print("test_dropout_forward_preserves_shape")

    var dropout = Dropout[DType.float32](p=0.3)
    dropout.train()

    # Test 1D
    var x1 = Tensor[DType.float32](10)
    var out1 = dropout(x1)
    assert_true(out1.shape() == x1.shape())

    # Test 2D
    var x2 = Tensor[DType.float32](5, 8)
    var out2 = dropout(x2)
    assert_true(out2.shape() == x2.shape())

    # Test 3D
    var x3 = Tensor[DType.float32](3, 4, 5)
    var out3 = dropout(x3)
    assert_true(out3.shape() == x3.shape())


fn test_dropout_forward_zero_probability() raises:
    """Test dropout with p=0 (no dropout) - all values kept."""
    print("test_dropout_forward_zero_probability")

    var dropout = Dropout[DType.float32](p=0.0)
    dropout.train()

    var x = Tensor[DType.float32](5, 5)
    for i in range(25):
        x.buffer.buffer[i] = Float32(i)

    var out = dropout(x)

    # With p=0, output should equal input (all kept, scale=1.0)
    assert_true(out.all_close[atol=1e-6](x))


fn test_dropout_forward_high_probability() raises:
    """Test dropout with p=0.9 - most values should be zero."""
    print("test_dropout_forward_high_probability")

    seed(456)
    var dropout = Dropout[DType.float32](p=0.9)
    dropout.train()

    var x = Tensor[DType.float32](100)
    for i in range(100):
        x[i] = 1.0

    var out = dropout(x)

    # Count zeros
    var num_zeros = 0
    for i in range(100):
        if out[i] == 0.0:
            num_zeros += 1

    # Should have high number of zeros (roughly 90%)
    assert_true(num_zeros > 75)

    # Non-zero values should be scaled by 10.0 (= 1/(1-0.9))
    for i in range(100):
        if out[i] != 0.0:
            var expected = Tensor[DType.float32](1)
            expected[0] = 10.0
            var actual = Tensor[DType.float32](1)
            actual[0] = out[i]
            assert_true(actual.all_close[atol=1e-5](expected))


fn test_dropout_forward_expected_value_preservation() raises:
    """Test that dropout maintains expected values (inverted dropout property)."""
    print("test_dropout_forward_expected_value_preservation")

    seed(123)
    var dropout = Dropout[DType.float32](p=0.5)
    dropout.train()

    # Large tensor to average out randomness
    var size = 10000
    var x = Tensor[DType.float32](size)
    for i in range(size):
        x[i] = 10.0

    var out = dropout(x)

    # Compute mean
    var sum_val = Scalar[DType.float32](0.0)
    for i in range(size):
        sum_val += out[i]
    var mean = sum_val / size

    # Expected value should be close to 10.0 (within statistical variance)
    var expected = Tensor[DType.float32](1)
    expected[0] = 10.0
    var actual = Tensor[DType.float32](1)
    actual[0] = mean
    assert_true(actual.all_close[atol=0.3](expected))


fn test_dropout_forward_mode_switching() raises:
    """Test switching between training and eval modes."""
    print("test_dropout_forward_mode_switching")

    seed(789)
    var dropout = Dropout[DType.float32](p=0.5)

    var x = Tensor[DType.float32](100)
    for i in range(100):
        x[i] = 5.0

    # Training mode - should have zeros
    dropout.train()
    var out_train = dropout(x)
    var has_zeros_train = False
    for i in range(100):
        if out_train[i] == 0.0:
            has_zeros_train = True
            break
    assert_true(has_zeros_train)

    # Eval mode - should be unchanged
    dropout.eval()
    var out_eval = dropout(x)
    assert_true(out_eval.all_close[atol=1e-6](x))

    # Back to training - should have zeros again
    dropout.train()
    var out_train2 = dropout(x)
    var has_zeros_train2 = False
    for i in range(100):
        if out_train2[i] == 0.0:
            has_zeros_train2 = True
            break
    assert_true(has_zeros_train2)


# ============================================================================
# BACKWARD PASS TESTS
# ============================================================================

fn test_dropout_backward_simple() raises:
    """Test dropout backward pass - gradients should flow through non-dropped elements."""
    print("test_dropout_backward_simple")

    seed(100)
    var dropout = Dropout[DType.float32](p=0.5)
    dropout.train()

    var x = Tensor[DType.float32](4, requires_grad=True)
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0

    var out = dropout(x)
    var loss = out.sum()
    loss.backward()

    # Gradients should be:
    # - 0.0 where elements were dropped
    # - scale (2.0) where elements were kept
    var grad = x.grad()
    for i in range(4):
        if out[i] == 0.0:
            # Element was dropped, gradient should be 0
            var expected = Tensor[DType.float32](1)
            expected[0] = 0.0
            var actual = Tensor[DType.float32](1)
            actual[0] = grad[i]
            assert_true(actual.all_close[atol=1e-6](expected))
        else:
            # Element was kept, gradient should be scale (2.0)
            var expected = Tensor[DType.float32](1)
            expected[0] = 2.0
            var actual = Tensor[DType.float32](1)
            actual[0] = grad[i]
            assert_true(actual.all_close[atol=1e-6](expected))


fn test_dropout_backward_with_upstream_gradient() raises:
    """Test dropout backward pass with non-uniform upstream gradients."""
    print("test_dropout_backward_with_upstream_gradient")

    seed(200)
    var dropout = Dropout[DType.float32](p=0.5)
    dropout.train()

    var x = Tensor[DType.float32](4, requires_grad=True)
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0

    var out = dropout(x)

    # Apply weighted sum (non-uniform upstream gradients)
    var weights = Tensor[DType.float32](4)
    weights[0] = 1.0
    weights[1] = 2.0
    weights[2] = 3.0
    weights[3] = 4.0

    var loss = (out * weights).sum()
    loss.backward()

    # Gradients should be:
    # grad_x[i] = weights[i] * dropout_mask[i]
    # where dropout_mask[i] = 0 (dropped) or scale (kept)
    var grad = x.grad()
    for i in range(4):
        if out[i] == 0.0:
            # Dropped: gradient should be 0
            var expected = Tensor[DType.float32](1)
            expected[0] = 0.0
            var actual = Tensor[DType.float32](1)
            actual[0] = grad[i]
            assert_true(actual.all_close[atol=1e-6](expected))
        else:
            # Kept: gradient should be weights[i] * scale
            var expected = Tensor[DType.float32](1)
            expected[0] = weights[i] * 2.0  # scale = 2.0
            var actual = Tensor[DType.float32](1)
            actual[0] = grad[i]
            assert_true(actual.all_close[atol=1e-6](expected))


fn test_dropout_backward_2d_tensor() raises:
    alias dtype = DType.float32
    """Test dropout backward pass on 2D tensor."""
    print("test_dropout_backward_2d_tensor")

    seed(300)
    var dropout = Dropout[DType.float32](p=0.3)
    dropout.train()

    var x = Tensor[DType.float32](3, 4, requires_grad=True)
    for i in range(12):
        x.buffer.buffer[i] = Float32(i) + 1.0

    var out = dropout(x)
    var loss = out.sum()
    loss.backward()

    # Check gradient consistency
    var grad = x.grad()
    var scale = Scalar[dtype](1.0 / (1.0 - 0.3))  # ~1.4286

    for i in range(12):
        if out.buffer.buffer[i] == 0.0:
            # Dropped element
            var expected = Tensor[DType.float32](1)
            expected[0] = 0.0
            var actual = Tensor[DType.float32](1)
            actual[0] = grad.buffer.buffer[i]
            assert_true(actual.all_close[atol=1e-6](expected))
        else:
            # Kept element: gradient should be scale
            var expected = Tensor[DType.float32](1)
            expected[0] = scale
            var actual = Tensor[DType.float32](1)
            actual[0] = grad.buffer.buffer[i]
            assert_true(actual.all_close[atol=1e-5](expected))


fn test_dropout_backward_eval_mode() raises:
    """Test dropout backward in eval mode - gradients should flow unchanged."""
    print("test_dropout_backward_eval_mode")

    var dropout = Dropout[DType.float32](p=0.5)
    dropout.eval()

    var x = Tensor[DType.float32](5, requires_grad=True)
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0
    x[4] = 5.0

    var out = dropout(x)
    var loss = out.sum()
    loss.backward()

    # In eval mode, gradients should be 1.0 for all elements
    var expected_grad = Tensor[DType.float32](5)
    for i in range(5):
        expected_grad[i] = 1.0

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_dropout_backward_chain_rule() raises:
    """Test dropout backward with chain rule (dropout in middle of computation)."""
    print("test_dropout_backward_chain_rule")

    seed(400)
    var dropout = Dropout[DType.float32](p=0.5)
    dropout.train()

    var x = Tensor[DType.float32](4, requires_grad=True)
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0

    # Apply operations before and after dropout
    var h = x * 2.0  # Multiply by 2
    var h_drop = dropout(h)
    var out = h_drop * 3.0  # Multiply by 3

    var loss = out.sum()
    loss.backward()

    # Chain rule: grad_x = 3.0 * dropout_mask * 2.0
    # where dropout_mask[i] = 0 (dropped) or scale (kept)
    var grad = x.grad()
    var scale = 2.0  # 1/(1-0.5)

    for i in range(4):
        if h_drop[i] == 0.0:
            # Dropped in dropout layer
            var expected = Tensor[DType.float32](1)
            expected[0] = 0.0
            var actual = Tensor[DType.float32](1)
            actual[0] = grad[i]
            assert_true(actual.all_close[atol=1e-6](expected))
        else:
            # Kept in dropout: grad = 3.0 * scale * 2.0
            var expected = Tensor[DType.float32](1)
            expected[0] = Float32(3.0 * scale * 2.0)
            var actual = Tensor[DType.float32](1)
            actual[0] = grad[i]
            assert_true(actual.all_close[atol=1e-5](expected))


fn test_dropout_backward_3d_tensor() raises:
    """Test dropout backward on 3D tensor."""
    print("test_dropout_backward_3d_tensor")

    seed(500)
    var dropout = Dropout[DType.float32](p=0.4)
    dropout.train()

    var x = Tensor[DType.float32](2, 2, 3, requires_grad=True)
    for i in range(12):
        x.buffer.buffer[i] = 1.0

    var out = dropout(x)
    var loss = out.sum()
    loss.backward()

    # Gradients should be 0 or scale (1/0.6 = 1.6667)
    var grad = x.grad()
    var scale = Float32(1.0 / (1.0 - 0.4))

    for i in range(12):
        if out.buffer.buffer[i] == 0.0:
            var expected = Tensor[DType.float32](1)
            expected[0] = 0.0
            var actual = Tensor[DType.float32](1)
            actual[0] = grad.buffer.buffer[i]
            assert_true(actual.all_close[atol=1e-6](expected))
        else:
            var expected = Tensor[DType.float32](1)
            expected[0] = scale
            var actual = Tensor[DType.float32](1)
            actual[0] = grad.buffer.buffer[i]
            assert_true(actual.all_close[atol=1e-5](expected))


fn test_dropout_backward_no_grad_input() raises:
    """Test dropout backward when input doesn't require grad."""
    print("test_dropout_backward_no_grad_input")

    var dropout = Dropout[DType.float32](p=0.5)
    dropout.train()

    var x = Tensor[DType.float32](4, requires_grad=False)
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0

    var out = dropout(x)

    # Output should not require grad if input doesn't
    assert_true(out.requires_grad == False)


fn test_dropout_backward_multiple_calls() raises:
    """Test dropout backward with multiple forward calls (different masks)."""
    print("test_dropout_backward_multiple_calls")

    seed(600)
    var dropout = Dropout[DType.float32](p=0.5)
    dropout.train()

    # First forward/backward
    var x1 = Tensor[DType.float32](4, requires_grad=True)
    for i in range(4):
        x1[i] = 1.0
    var out1 = dropout(x1)
    var loss1 = out1.sum()
    loss1.backward()
    var grad1 = x1.grad()

    # Second forward/backward (new mask)
    var x2 = Tensor[DType.float32](4, requires_grad=True)
    for i in range(4):
        x2[i] = 1.0
    var out2 = dropout(x2)
    var loss2 = out2.sum()
    loss2.backward()
    var grad2 = x2.grad()

    # Masks should be different (with high probability)
    var masks_different = False
    for i in range(4):
        if (grad1[i] == 0.0 and grad2[i] != 0.0) or (grad1[i] != 0.0 and grad2[i] == 0.0):
            masks_different = True
            break
    assert_true(masks_different)


# ============================================================================
# EDGE CASES AND SPECIAL TESTS
# ============================================================================

fn test_dropout_zero_probability_backward() raises:
    """Test dropout backward with p=0 - all gradients should flow through."""
    print("test_dropout_zero_probability_backward")

    var dropout = Dropout[DType.float32](p=0.0)
    dropout.train()

    var x = Tensor[DType.float32](5, requires_grad=True)
    for i in range(5):
        x[i] = Float32(i) + 1.0

    var out = dropout(x)
    var loss = out.sum()
    loss.backward()

    # With p=0, all gradients should be 1.0 (scale=1.0, no dropping)
    var expected_grad = Tensor[DType.float32](5)
    for i in range(5):
        expected_grad[i] = 1.0

    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_dropout_large_tensor_backward() raises:
    """Test dropout backward on large tensor - verify gradient statistics."""
    print("test_dropout_large_tensor_backward")

    seed(700)
    var dropout = Dropout[DType.float32](p=0.5)
    dropout.train()

    var size = 10000
    var x = Tensor[DType.float32](size, requires_grad=True)
    for i in range(size):
        x[i] = 1.0

    var out = dropout(x)
    var loss = out.sum()
    loss.backward()

    # Count gradient values
    var grad = x.grad()
    var num_zero_grads = 0
    var num_nonzero_grads = 0
    var scale =Float32(2.0)

    for i in range(size):
        if grad[i] == 0.0:
            num_zero_grads += 1
        else:
            num_nonzero_grads += 1
            # Non-zero gradients should equal scale
            var expected = Tensor[DType.float32](1)
            expected[0] = scale
            var actual = Tensor[DType.float32](1)
            actual[0] = grad[i]
            assert_true(actual.all_close[atol=1e-5](expected))

    # Roughly 50% should be zero, 50% should be scale
    var zero_ratio = Float64(num_zero_grads) / size
    assert_true(zero_ratio > 0.45 and zero_ratio < 0.55)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

fn main() raises:
    print("=" * 80)
    print("DROPOUT COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    print("\n--- FORWARD PASS TESTS ---")
    test_dropout_forward_training_mode()
    test_dropout_forward_eval_mode()
    test_dropout_forward_preserves_shape()
    test_dropout_forward_zero_probability()
    test_dropout_forward_high_probability()
    test_dropout_forward_expected_value_preservation()
    test_dropout_forward_mode_switching()

    print("\n--- BACKWARD PASS TESTS ---")
    test_dropout_backward_simple()
    test_dropout_backward_with_upstream_gradient()
    test_dropout_backward_2d_tensor()
    test_dropout_backward_eval_mode()
    test_dropout_backward_chain_rule()
    test_dropout_backward_3d_tensor()
    test_dropout_backward_no_grad_input()
    test_dropout_backward_multiple_calls()

    print("\n--- EDGE CASES ---")
    test_dropout_zero_probability_backward()
    test_dropout_large_tensor_backward()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
