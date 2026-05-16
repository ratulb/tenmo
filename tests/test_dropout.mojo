from tenmo.tensor import Tensor
from std.random import seed
from std.testing import assert_true, TestSuite
from tenmo.net import Dropout
from std.sys import has_accelerator
from std.random import seed
from tenmo.shapes import Shape


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
    var mean = sum_val / Float32(size)

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

    print()
    # Gradients should be:
    # - 0.0 where elements were dropped
    # - scale (2.0) where elements were kept
    var grad = x.grad()
    for i in range(4):
        if out[i] == 0.0:
            # Element was dropped, gradient should be 0
            var expected = Tensor[DType.float32].scalar(0)
            var actual = Tensor[DType.float32].scalar(grad[i])
            assert_true(actual.all_close[atol=1e-6](expected))
        else:
            # Element was kept, gradient should be scale (2.0)
            var expected = Tensor[DType.float32].scalar(2)
            var actual = Tensor[DType.float32].scalar(grad[i])
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
    comptime dtype = DType.float32
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
    var zero_ratio = Float64(num_zero_grads) / Float64(size)
    assert_true(zero_ratio > 0.45 and zero_ratio < 0.55)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll dropout tests passed!")


#=========



# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

fn count_zeros[dtype: DType](t: Tensor[dtype]) -> Int:
    var count = 0
    for i in range(t.numels()):
        if t.get(i) == 0.0:
            count += 1
    return count

fn count_nonzeros[dtype: DType](t: Tensor[dtype]) -> Int:
    return t.numels() - count_zeros(t)

fn all_nonzero_close[dtype: DType](
    t: Tensor[dtype], expected_val: Scalar[dtype], atol: Scalar[dtype]
) -> Bool:
    for i in range(t.numels()):
        var v = t.get(i)
        if v != 0.0:
            if abs(v - expected_val) > atol:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 1. FORWARD — CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_dropout2_fwd_cpu_eval_is_identity() raises:
    print("test_dropout2_fwd_cpu_eval_is_identity")
    comptime dtype = DType.float32
    var dropout = Dropout[dtype](p=0.5)
    dropout.eval()
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var out = dropout(x)
    assert_true(out.all_close(x))


fn test_dropout2_fwd_cpu_p0_is_identity() raises:
    print("test_dropout2_fwd_cpu_p0_is_identity")
    comptime dtype = DType.float32
    var dropout = Dropout[dtype](p=0.0)
    dropout.train()
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var out = dropout(x)
    assert_true(out.all_close(x))


fn test_dropout2_fwd_cpu_high_p_many_zeros_1d() raises:
    print("test_dropout2_fwd_cpu_high_p_many_zeros_1d")
    comptime dtype = DType.float32
    seed(100)
    var dropout = Dropout[dtype](p=0.9)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(200))
    var out = dropout(x)
    # ~90% zeros expected — check at least 75%
    assert_true(count_zeros(out) > 150)


fn test_dropout2_fwd_cpu_scale_correct_1d() raises:
    print("test_dropout2_fwd_cpu_scale_correct_1d")
    comptime dtype = DType.float32
    seed(200)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(200))
    var out = dropout(x)
    # Non-zero values must be exactly 2.0 (= 1 / (1 - 0.5))
    assert_true(all_nonzero_close(out, 2.0, 1e-5))


fn test_dropout2_fwd_cpu_scale_correct_2d() raises:
    print("test_dropout2_fwd_cpu_scale_correct_2d")
    comptime dtype = DType.float32
    seed(300)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(10, 10))
    var out = dropout(x)
    assert_true(all_nonzero_close(out, 2.0, 1e-5))


fn test_dropout2_fwd_cpu_scale_correct_3d() raises:
    print("test_dropout2_fwd_cpu_scale_correct_3d")
    comptime dtype = DType.float32
    seed(400)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(4, 4, 4))
    var out = dropout(x)
    assert_true(all_nonzero_close(out, 2.0, 1e-5))


fn test_dropout2_fwd_cpu_different_masks_per_call() raises:
    print("test_dropout2_fwd_cpu_different_masks_per_call")
    comptime dtype = DType.float32
    seed(500)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(100))
    var out1 = dropout(x)
    var out2 = dropout(x)
    # With 100 elements and p=0.5, masks should differ with overwhelming prob
    var same = True
    for i in range(100):
        if out1.get(i) != out2.get(i):
            same = False
            break
    assert_true(not same)


fn test_dropout2_fwd_cpu_output_shape_preserved_1d() raises:
    print("test_dropout2_fwd_cpu_output_shape_preserved_1d")
    comptime dtype = DType.float32
    seed(600)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(8))
    var out = dropout(x)
    assert_true(out.shape() == Shape(8))


fn test_dropout2_fwd_cpu_output_shape_preserved_2d() raises:
    print("test_dropout2_fwd_cpu_output_shape_preserved_2d")
    comptime dtype = DType.float32
    seed(700)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(4, 6))
    var out = dropout(x)
    assert_true(out.shape() == Shape(4, 6))


fn test_dropout2_fwd_cpu_output_shape_preserved_3d() raises:
    print("test_dropout2_fwd_cpu_output_shape_preserved_3d")
    comptime dtype = DType.float32
    seed(800)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(2, 3, 4))
    var out = dropout(x)
    assert_true(out.shape() == Shape(2, 3, 4))


# ─────────────────────────────────────────────────────────────────────────────
# 2. BACKWARD — CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_dropout2_bwd_cpu_grad_zero_where_dropped_1d() raises:
    print("test_dropout2_bwd_cpu_grad_zero_where_dropped_1d")
    comptime dtype = DType.float32
    seed(1000)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(64), requires_grad=True)
    var out = dropout(x)
    var loss = out.sum()
    loss.backward()
    # Where out==0, grad must be 0; where out!=0, grad must be scale=2.0
    for i in range(64):
        if out.get(i) == 0.0:
            assert_true(abs(x.grad().get(i)) < 1e-6)
        else:
            assert_true(abs(x.grad().get(i) - 2.0) < 1e-5)


fn test_dropout2_bwd_cpu_grad_zero_where_dropped_2d() raises:
    print("test_dropout2_bwd_cpu_grad_zero_where_dropped_2d")
    comptime dtype = DType.float32
    seed(1100)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(8, 8), requires_grad=True)
    var out = dropout(x)
    var loss = out.sum()
    loss.backward()
    for i in range(64):
        if out.get(i) == 0.0:
            assert_true(abs(x.grad().get(i)) < 1e-6)
        else:
            assert_true(abs(x.grad().get(i) - 2.0) < 1e-5)


fn test_dropout2_bwd_cpu_grad_zero_where_dropped_3d() raises:
    print("test_dropout2_bwd_cpu_grad_zero_where_dropped_3d")
    comptime dtype = DType.float32
    seed(1200)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(2, 4, 8), requires_grad=True)
    var out = dropout(x)
    var loss = out.sum()
    loss.backward()
    for i in range(x.numels()):
        if out.get(i) == 0.0:
            assert_true(abs(x.grad().get(i)) < 1e-6)
        else:
            assert_true(abs(x.grad().get(i) - 2.0) < 1e-5)


fn test_dropout2_bwd_cpu_no_grad_leaf_unaffected() raises:
    print("test_dropout2_bwd_cpu_no_grad_leaf_unaffected")
    comptime dtype = DType.float32
    seed(1300)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    # x has no requires_grad — dropout should return early, no ancestry
    var x = Tensor[dtype].ones(Shape(16))
    var out = dropout(x)
    assert_true(not out.requires_grad)


fn test_dropout2_bwd_cpu_high_p_grad_flow() raises:
    print("test_dropout2_bwd_cpu_high_p_grad_flow")
    comptime dtype = DType.float32
    seed(1400)
    var dropout = Dropout[dtype](p=0.9)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(200), requires_grad=True)
    var out = dropout(x)
    var loss = out.sum()
    loss.backward()
    # Non-zero grads must equal scale = 10.0
    for i in range(200):
        if out.get(i) != 0.0:
            assert_true(abs(x.grad().get(i) - 10.0) < 1e-4)
        else:
            assert_true(abs(x.grad().get(i)) < 1e-6)


fn test_dropout2_bwd_cpu_chained_with_linear_op() raises:
    # dropout -> * 3 -> sum -> backward
    print("test_dropout2_bwd_cpu_chained_with_linear_op")
    comptime dtype = DType.float32
    seed(1500)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(32), requires_grad=True)
    var out = dropout(x)
    var scaled = out * Tensor[dtype].full_like(out, 3.0)
    var loss = scaled.sum()
    loss.backward()
    # grad = mask * 3; mask is either 0 or scale(=2), so grad is 0 or 6
    for i in range(32):
        var g = x.grad().get(i)
        assert_true(abs(g) < 1e-6 or abs(g - 6.0) < 1e-5)


fn test_dropout2_bwd_cpu_eval_grad_is_ones() raises:
    # In eval mode dropout is identity — grad must be 1
    print("test_dropout2_bwd_cpu_eval_grad_is_ones")
    comptime dtype = DType.float32
    var dropout = Dropout[dtype](p=0.5)
    dropout.eval()
    var x = Tensor[dtype].ones(Shape(16), requires_grad=True)
    var out = dropout(x)
    var loss = out.sum()
    loss.backward()
    assert_true(x.grad().all_close[atol=1e-5](Tensor.ones_like(x)))


fn test_dropout2_bwd_cpu_multiple_calls_different_grads() raises:
    print("test_dropout2_bwd_cpu_multiple_calls_different_grads")
    comptime dtype = DType.float32
    seed(1600)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()

    var x1 = Tensor[dtype].ones(Shape(32), requires_grad=True)
    var out1 = dropout(x1)
    var loss1 = out1.sum()
    loss1.backward()

    var x2 = Tensor[dtype].ones(Shape(32), requires_grad=True)
    var out2 = dropout(x2)
    var loss2 = out2.sum()
    loss2.backward()

    var different = False
    for i in range(32):
        if x1.grad().get(i) != x2.grad().get(i):
            different = True
            break
    assert_true(different)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAD FLOW VERIFICATION — CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_dropout2_gradflow_cpu_mask_consistent_fwd_bwd() raises:
    # The mask used in forward must be the same one used in backward.
    # Verify: wherever out==0, grad==0; wherever out!=0, grad==scale.
    print("test_dropout2_gradflow_cpu_mask_consistent_fwd_bwd")
    comptime dtype = DType.float32
    seed(1700)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(128), requires_grad=True)
    var out = dropout(x)
    var loss = out.sum()
    loss.backward()
    var consistent = True
    for i in range(128):
        var dropped = out.get(i) == 0.0
        var grad_zero = abs(x.grad().get(i)) < 1e-6
        if dropped != grad_zero:
            consistent = False
            break
    assert_true(consistent)


fn test_dropout2_gradflow_cpu_sum_of_grads_matches_nonzero_count() raises:
    print("test_dropout2_gradflow_cpu_sum_of_grads_matches_nonzero_count")
    comptime dtype = DType.float32
    seed(1800)
    var dropout = Dropout[dtype](p=0.5)
    dropout.train()
    var x = Tensor[dtype].ones(Shape(128), requires_grad=True)
    var out = dropout(x)
    var loss = out.sum()
    loss.backward()
    # sum(grad) == count_nonzero(out) * scale
    var grad_sum = Scalar[dtype](0)
    for i in range(128):
        grad_sum += x.grad().get(i)
    var expected = Scalar[dtype](count_nonzeros(out)) * Scalar[dtype](2.0)
    assert_true(abs(grad_sum - expected) < 1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FORWARD — GPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_dropout2_fwd_gpu_eval_is_identity() raises:
    comptime if has_accelerator():
        print("test_dropout2_fwd_gpu_eval_is_identity")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.eval()
        var x_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        assert_true(out.to_cpu().all_close(x_cpu))


fn test_dropout2_fwd_gpu_output_shape_preserved_1d() raises:
    comptime if has_accelerator():
        print("test_dropout2_fwd_gpu_output_shape_preserved_1d")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(64)).to_gpu()
        var out = dropout(x)
        assert_true(out.shape() == Shape(64))


fn test_dropout2_fwd_gpu_output_shape_preserved_2d() raises:
    comptime if has_accelerator():
        print("test_dropout2_fwd_gpu_output_shape_preserved_2d")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(8, 8)).to_gpu()
        var out = dropout(x)
        assert_true(out.shape() == Shape(8, 8))


fn test_dropout2_fwd_gpu_output_shape_preserved_3d() raises:
    comptime if has_accelerator():
        print("test_dropout2_fwd_gpu_output_shape_preserved_3d")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(2, 4, 8)).to_gpu()
        var out = dropout(x)
        assert_true(out.shape() == Shape(2, 4, 8))


fn test_dropout2_fwd_gpu_high_p_many_zeros() raises:
    comptime if has_accelerator():
        print("test_dropout2_fwd_gpu_high_p_many_zeros")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.9)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(200)).to_gpu()
        var out = dropout(x).to_cpu()
        assert_true(count_zeros(out) > 150)


fn test_dropout2_fwd_gpu_scale_correct_1d() raises:
    comptime if has_accelerator():
        print("test_dropout2_fwd_gpu_scale_correct_1d")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(200)).to_gpu()
        var out = dropout(x).to_cpu()
        assert_true(all_nonzero_close(out, 2.0, 1e-5))


fn test_dropout2_fwd_gpu_scale_correct_2d() raises:
    comptime if has_accelerator():
        print("test_dropout2_fwd_gpu_scale_correct_2d")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(10, 10)).to_gpu()
        var out = dropout(x).to_cpu()
        assert_true(all_nonzero_close(out, 2.0, 1e-5))


fn test_dropout2_fwd_gpu_scale_correct_3d() raises:
    comptime if has_accelerator():
        print("test_dropout2_fwd_gpu_scale_correct_3d")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(2, 4, 8)).to_gpu()
        var out = dropout(x).to_cpu()
        assert_true(all_nonzero_close(out, 2.0, 1e-5))


fn test_dropout2_fwd_gpu_different_masks_per_call() raises:
    comptime if has_accelerator():
        print("test_dropout2_fwd_gpu_different_masks_per_call")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(100)).to_gpu()
        var out1 = dropout(x).to_cpu()
        var out2 = dropout(x).to_cpu()
        var same = True
        for i in range(100):
            if out1.get(i) != out2.get(i):
                same = False
                break
        assert_true(not same)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKWARD — GPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_dropout2_bwd_gpu_grad_zero_where_dropped_1d() raises:
    comptime if has_accelerator():
        print("test_dropout2_bwd_gpu_grad_zero_where_dropped_1d")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(64), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var out_cpu = out.to_cpu()
        var loss = out.sum()
        loss.backward()
        for i in range(64):
            if out_cpu.get(i) == 0.0:
                assert_true(abs(x_cpu.grad().get(i)) < 1e-6)
            else:
                assert_true(abs(x_cpu.grad().get(i) - 2.0) < 1e-5)


fn test_dropout2_bwd_gpu_grad_zero_where_dropped_2d() raises:
    comptime if has_accelerator():
        print("test_dropout2_bwd_gpu_grad_zero_where_dropped_2d")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(8, 8), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var out_cpu = out.to_cpu()
        var loss = out.sum()
        loss.backward()
        for i in range(64):
            if out_cpu.get(i) == 0.0:
                assert_true(abs(x_cpu.grad().get(i)) < 1e-6)
            else:
                assert_true(abs(x_cpu.grad().get(i) - 2.0) < 1e-5)


fn test_dropout2_bwd_gpu_grad_zero_where_dropped_3d() raises:
    comptime if has_accelerator():
        print("test_dropout2_bwd_gpu_grad_zero_where_dropped_3d")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(2, 4, 8), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var out_cpu = out.to_cpu()
        var loss = out.sum()
        loss.backward()
        for i in range(x_cpu.numels()):
            if out_cpu.get(i) == 0.0:
                assert_true(abs(x_cpu.grad().get(i)) < 1e-6)
            else:
                assert_true(abs(x_cpu.grad().get(i) - 2.0) < 1e-5)


fn test_dropout2_bwd_gpu_no_grad_leaf_unaffected() raises:
    comptime if has_accelerator():
        print("test_dropout2_bwd_gpu_no_grad_leaf_unaffected")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x = Tensor[dtype].ones(Shape(16)).to_gpu()
        var out = dropout(x)
        assert_true(not out.requires_grad)


fn test_dropout2_bwd_gpu_high_p_grad_flow() raises:
    comptime if has_accelerator():
        print("test_dropout2_bwd_gpu_high_p_grad_flow")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.9)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(200), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var out_cpu = out.to_cpu()
        var loss = out.sum()
        loss.backward()
        for i in range(200):
            if out_cpu.get(i) != 0.0:
                assert_true(abs(x_cpu.grad().get(i) - 10.0) < 1e-4)
            else:
                assert_true(abs(x_cpu.grad().get(i)) < 1e-6)


fn test_dropout2_bwd_gpu_chained_with_linear_op() raises:
    comptime if has_accelerator():
        print("test_dropout2_bwd_gpu_chained_with_linear_op")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(32), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var scaled = out * Tensor[dtype].full_like(out, 3.0)
        var loss = scaled.sum()
        loss.backward()
        for i in range(32):
            var g = x_cpu.grad().get(i)
            assert_true(abs(g) < 1e-6 or abs(g - 6.0) < 1e-5)


fn test_dropout2_bwd_gpu_eval_grad_is_ones() raises:
    comptime if has_accelerator():
        print("test_dropout2_bwd_gpu_eval_grad_is_ones")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.eval()
        var x_cpu = Tensor[dtype].ones(Shape(16), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var loss = out.sum()
        loss.backward()
        assert_true(x_cpu.grad().all_close[atol=1e-5](Tensor.ones_like(x_cpu)))


# ─────────────────────────────────────────────────────────────────────────────
# 6. GRAD FLOW VERIFICATION — GPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_dropout2_gradflow_gpu_mask_consistent_fwd_bwd() raises:
    comptime if has_accelerator():
        print("test_dropout2_gradflow_gpu_mask_consistent_fwd_bwd")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(128), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var out_cpu = out.to_cpu()
        var loss = out.sum()
        loss.backward()
        var consistent = True
        for i in range(128):
            var dropped = out_cpu.get(i) == 0.0
            var grad_zero = abs(x_cpu.grad().get(i)) < 1e-6
            if dropped != grad_zero:
                consistent = False
                break
        assert_true(consistent)


fn test_dropout2_gradflow_gpu_sum_of_grads_matches_nonzero_count() raises:
    comptime if has_accelerator():
        print("test_dropout2_gradflow_gpu_sum_of_grads_matches_nonzero_count")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.train()
        var x_cpu = Tensor[dtype].ones(Shape(128), requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var out = dropout(x_gpu)
        var out_cpu = out.to_cpu()
        var loss = out.sum()
        loss.backward()
        var grad_sum = Scalar[dtype](0)
        for i in range(128):
            grad_sum += x_cpu.grad().get(i)
        var expected = Scalar[dtype](count_nonzeros(out_cpu)) * Scalar[dtype](2.0)
        assert_true(abs(grad_sum - expected) < 1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CPU / GPU PARITY
# ─────────────────────────────────────────────────────────────────────────────

fn test_dropout2_parity_eval_cpu_gpu_match() raises:
    comptime if has_accelerator():
        print("test_dropout2_parity_eval_cpu_gpu_match")
        comptime dtype = DType.float32
        var dropout = Dropout[dtype](p=0.5)
        dropout.eval()
        var x_cpu = Tensor[dtype].arange(1.0, 17.0).reshape(4, 4)
        var cpu_out = dropout(x_cpu)
        var gpu_out = dropout(x_cpu.to_gpu()).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-5](gpu_out))


fn test_dropout2_parity_scale_value_matches() raises:
    comptime if has_accelerator():
        print("test_dropout2_parity_scale_value_matches")
        comptime dtype = DType.float32
        # Both CPU and GPU non-zero outputs must be exactly input * scale
        var dropout_cpu = Dropout[dtype](p=0.5)
        var dropout_gpu = Dropout[dtype](p=0.5)
        dropout_cpu.train()
        dropout_gpu.train()
        var x = Tensor[dtype].ones(Shape(200))
        var cpu_out = dropout_cpu(x)
        var gpu_out = dropout_gpu(x.to_gpu()).to_cpu()
        # Both should have scale=2.0 on non-zero elements
        assert_true(all_nonzero_close(cpu_out, 2.0, 1e-5))
        assert_true(all_nonzero_close(gpu_out, 2.0, 1e-5))


fn test_dropout2_parity_bwd_grad_scale_matches() raises:
    comptime if has_accelerator():
        print("test_dropout2_parity_bwd_grad_scale_matches")
        comptime dtype = DType.float32
        var dropout_cpu = Dropout[dtype](p=0.5)
        var dropout_gpu = Dropout[dtype](p=0.5)
        dropout_cpu.train()
        dropout_gpu.train()

        var x_cpu_leaf = Tensor[dtype].ones(Shape(64), requires_grad=True)
        var out_cpu = dropout_cpu(x_cpu_leaf)
        var loss_cpu = out_cpu.sum()
        loss_cpu.backward()

        var x_gpu_leaf = Tensor[dtype].ones(Shape(64), requires_grad=True)
        var out_gpu = dropout_gpu(x_gpu_leaf.to_gpu())
        var loss_gpu = out_gpu.sum()
        loss_gpu.backward()

        # Both: non-zero grads must be 2.0, zero grads must be 0.0
        for i in range(64):
            var gc = x_cpu_leaf.grad().get(i)
            var gg = x_gpu_leaf.grad().get(i)
            assert_true(abs(gc) < 1e-6 or abs(gc - 2.0) < 1e-5)
            assert_true(abs(gg) < 1e-6 or abs(gg - 2.0) < 1e-5)
