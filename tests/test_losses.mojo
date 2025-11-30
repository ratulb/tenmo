from tenmo import Tensor
from net import MSELoss, BCELoss, BCEWithLogitsLoss, Linear, Sigmoid
from common_utils import isnan, isinf
from testing import assert_true
from intarray import IntArray

# ============================================================================
# MSE Loss Tests
# ============================================================================

fn test_mse_loss_perfect_prediction() raises:
    print("test_mse_loss_perfect_prediction")
    alias dtype = DType.float32
    var pred = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var loss = pred.mse[track_grad=True](target)
    assert_true(abs(loss.item()) < 1e-6, "Perfect prediction should have zero loss")

fn test_mse_loss_simple_gradient() raises:
    print("test_mse_loss_simple_gradient")
    alias dtype = DType.float32
    var pred = Tensor[dtype].d2([[2.0, 4.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0, 2.0]])
    var loss = pred.mse[track_grad=True](target)
    loss.backward()
    # MSE gradient: 2*(pred - target) / N
    # For pred=[2,4], target=[1,2]: diff=[1,2]
    # grad = 2*[1,2] / 2 = [1, 2]
    var expected_grad = Tensor[dtype].d2([[1.0, 2.0]])
    assert_true(pred.grad().all_close(expected_grad), "MSE gradient mismatch")

fn test_mse_loss_batch_gradient() raises:
    print("test_mse_loss_batch_gradient")
    alias dtype = DType.float32
    var pred = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[0.0, 0.0], [0.0, 0.0]])
    var loss = pred.mse[track_grad=True](target)
    loss.backward()
    # Gradient: 2*(pred - target) / N = 2*pred / 4 = pred / 2
    var expected_grad = Tensor[dtype].d2([[0.5, 1.0], [1.5, 2.0]])
    assert_true(pred.grad().all_close[atol=1e-5](expected_grad), "Batch MSE gradient mismatch")

fn test_mse_loss_no_grad_mode() raises:
    print("test_mse_loss_no_grad_mode")
    alias dtype = DType.float32
    var pred = Tensor[dtype].d2([[2.0, 4.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0, 2.0]])
    var loss = pred.mse[track_grad=False](target)
    # Should not build graph
    assert_true(not loss.has_backward_fn(), "Loss should not have backward function in no_grad mode")

fn test_mse_loss_struct_train_mode() raises:
    print("test_mse_loss_struct_train_mode")
    alias dtype = DType.float32
    var criterion = MSELoss[dtype]()
    criterion.train()
    var pred = Tensor[dtype].d2([[2.0, 4.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0, 2.0]])
    var loss = criterion(pred, target)
    loss.backward()
    var expected_grad = Tensor[dtype].d2([[1.0, 2.0]])
    assert_true(pred.grad().all_close(expected_grad), "MSE struct train mode gradient mismatch")

fn test_mse_loss_struct_eval_mode() raises:
    print("test_mse_loss_struct_eval_mode")
    alias dtype = DType.float32
    var criterion = MSELoss[dtype]()
    criterion.eval()
    var pred = Tensor[dtype].d2([[2.0, 4.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0, 2.0]])
    var loss = criterion(pred, target)
    assert_true(not loss.has_backward_fn(), "Loss should not have backward function in eval mode")

# ============================================================================
# BCE Loss Tests
# ============================================================================

fn test_bce_loss_perfect_prediction() raises:
    print("test_bce_loss_perfect_prediction")
    alias dtype = DType.float32
    var pred = Tensor[dtype].d2([[0.9, 0.1], [0.2, 0.8]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]])
    var loss = pred.binary_cross_entropy[track_grad=True](target)
    # Should be small but not zero due to clipping
    assert_true(loss.item() < 0.5, "Near-perfect prediction should have low loss")

fn test_bce_loss_worst_prediction() raises:
    print("test_bce_loss_worst_prediction")
    alias dtype = DType.float32
    var pred = Tensor[dtype].d2([[0.1, 0.9]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0, 0.0]])
    var loss = pred.binary_cross_entropy[track_grad=True](target)
    # Should be high loss
    assert_true(loss.item() > 1.0, "Bad prediction should have high loss")

fn test_bce_loss_gradient_positive_target() raises:
    print("test_bce_loss_gradient_positive_target")
    alias dtype = DType.float32
    var pred = Tensor[dtype].d2([[0.6]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0]])
    var loss = pred.binary_cross_entropy[track_grad=True](target, epsilon=1e-7)
    loss.backward()
    # For y=1: BCE = -log(p), gradient = -1/p
    # At p=0.6: gradient ≈ -1/0.6 = -1.667
    # But averaged and negated through chain rule
    var grad = pred.grad()[IntArray(0, 0)]
    assert_true(grad < 0, "Gradient should be negative when pred < target")

fn test_bce_loss_gradient_negative_target() raises:
    print("test_bce_loss_gradient_negative_target")
    alias dtype = DType.float32
    var pred = Tensor[dtype].d2([[0.6]], requires_grad=True)
    var target = Tensor[dtype].d2([[0.0]])
    var loss = pred.binary_cross_entropy[track_grad=True](target, epsilon=1e-7)
    loss.backward()
    # For y=0: BCE = -log(1-p), gradient = 1/(1-p)
    # At p=0.6: gradient ≈ 1/0.4 = 2.5
    var grad = pred.grad()[IntArray(0, 0)]
    assert_true(grad > 0, "Gradient should be positive when pred > target")

fn test_bce_with_logits_positive_gradient() raises:
    print("test_bce_with_logits_positive_gradient")
    alias dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0]])
    var loss = logits.binary_cross_entropy_with_logits[track_grad=True](target)
    loss.backward()
    # High positive logit with target=1 should have small positive gradient
    # (pushing toward even higher confidence)
    var grad = logits.grad()[IntArray(0, 0)]
    assert_true(abs(grad) < 0.5, "Gradient should be small for confident correct prediction")

fn test_bce_with_logits_negative_gradient() raises:
    print("test_bce_with_logits_negative_gradient")
    alias dtype = DType.float32
    var logits = Tensor[dtype].d2([[-2.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0]])
    var loss = logits.binary_cross_entropy_with_logits[track_grad=True](target)
    loss.backward()
    # Negative logit with target=1 should have negative gradient
    # (pushing toward positive logits)
    var grad = logits.grad()[IntArray(0, 0)]
    assert_true(grad < 0, "Gradient should be negative, pushing logits up toward target")

fn test_bce_loss_boundary_values() raises:
    print("test_bce_loss_boundary_values")
    alias dtype = DType.float32
    # Test with values very close to 0 and 1
    var pred = Tensor[dtype].d2([[0.001, 0.999]], requires_grad=True)
    var target = Tensor[dtype].d2([[0.0, 1.0]])
    var loss = pred.binary_cross_entropy[track_grad=True](target)
    loss.backward()
    # Should handle boundary gracefully
    var grad0 = pred.grad()[IntArray(0, 0)]
    var grad1 = pred.grad()[IntArray(0, 1)]
    assert_true(not isnan(grad0) and not isnan(grad1), "Should handle boundary values")

fn test_bce_loss_clipping() raises:
    print("test_bce_loss_clipping")
    alias dtype = DType.float32
    # Test extreme values don't cause NaN
    var pred = Tensor[dtype].d2([[0.0, 1.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0, 0.0]])
    var loss = pred.binary_cross_entropy[track_grad=True](target, epsilon=1e-7)
    loss.backward()
    # Should not be NaN or Inf
    assert_true(not isnan(loss.item()), "Loss should not be NaN with clipping")
    assert_true(not isinf(loss.item()), "Loss should not be Inf with clipping")

fn test_bce_loss_no_grad_mode() raises:
    print("test_bce_loss_no_grad_mode")
    alias dtype = DType.float32
    var pred = Tensor[dtype].d2([[0.6, 0.4]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0, 0.0]])
    var loss = pred.binary_cross_entropy[track_grad=False](target)
    assert_true(not loss.has_backward_fn(), "BCE should not build graph in no_grad mode")

fn test_bce_loss_struct_train_eval() raises:
    print("test_bce_loss_struct_train_eval")
    alias dtype = DType.float32
    var criterion = BCELoss[dtype]()
    var pred = Tensor[dtype].d2([[0.7]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0]])

    # Train mode
    criterion.train()
    var train_loss = criterion(pred, target)
    assert_true(train_loss.has_backward_fn(), "BCE should build graph in train mode")

    # Eval mode
    criterion.eval()
    var eval_loss = criterion(pred, target)
    assert_true(not eval_loss.has_backward_fn(), "BCE should not build graph in eval mode")

# ============================================================================
# BCE with Logits Loss Tests
# ============================================================================

fn test_bce_with_logits_zero_logits() raises:
    print("test_bce_with_logits_zero_logits")
    alias dtype = DType.float32
    # Logits=0 -> sigmoid(0)=0.5
    var logits = Tensor[dtype].d2([[0.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[0.5]])
    var loss = logits.binary_cross_entropy_with_logits[track_grad=True](target)
    # Should be relatively small
    assert_true(loss.item() < 1.0, "Zero logits with 0.5 target should have moderate loss")



fn test_bce_with_logits_batch() raises:
    print("test_bce_with_logits_batch")
    alias dtype = DType.float32
    var logits = Tensor[dtype].d2([[1.0, -1.0], [2.0, -2.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
    var loss = logits.binary_cross_entropy_with_logits[track_grad=True](target)
    loss.backward()
    # All predictions align with targets, gradients should be relatively small
    assert_true(loss.item() < 1.0, "Well-aligned batch should have low loss")

fn test_bce_with_logits_no_grad_mode() raises:
    print("test_bce_with_logits_no_grad_mode")
    alias dtype = DType.float32
    var logits = Tensor[dtype].d2([[1.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0]])
    var loss = logits.binary_cross_entropy_with_logits[track_grad=False](target)
    assert_true(not loss.has_backward_fn(), "BCE with logits should not build graph in no_grad mode")

fn test_bce_with_logits_struct() raises:
    print("test_bce_with_logits_struct")
    alias dtype = DType.float32
    var criterion = BCEWithLogitsLoss[dtype]()
    var logits = Tensor[dtype].d2([[0.5]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0]])

    criterion.train()
    var train_loss = criterion(logits, target)
    train_loss.backward()
    assert_true(logits.has_grad(), "Should have gradients in train mode")

    logits.zero_grad()
    criterion.eval()
    var eval_loss = criterion(logits, target)
    assert_true(not eval_loss.has_backward_fn(), "Should not build graph in eval mode")

# ============================================================================
# Integration Tests - Loss Functions with Models
# ============================================================================

fn test_mse_loss_with_linear_layer() raises:
    print("test_mse_loss_with_linear_layer")
    alias dtype = DType.float32
    var layer = Linear[dtype](2, 1, xavier=True)
    layer.train()

    var X = Tensor[dtype].d2([[1.0, 2.0]])
    var y = Tensor[dtype].d2([[3.0]])

    var pred = layer(X)
    var criterion = MSELoss[dtype]()
    criterion.train()
    var loss = criterion(pred, y)
    loss.backward()

    # Check gradients exist
    assert_true(layer.weight.has_grad(), "Linear weight should have gradient")
    assert_true(layer.bias.has_grad(), "Linear bias should have gradient")

fn test_bce_loss_with_sigmoid_output() raises:
    print("test_bce_loss_with_sigmoid_output")
    alias dtype = DType.float32
    var layer = Linear[dtype](2, 1, xavier=True)
    var sigmoid = Sigmoid[dtype]()
    layer.train()
    sigmoid.train()

    var X = Tensor[dtype].d2([[1.0, 2.0]])
    var y = Tensor[dtype].d2([[1.0]])

    var logits = layer(X)
    var pred = sigmoid(logits)
    var criterion = BCELoss[dtype]()
    criterion.train()
    var loss = criterion(pred, y)
    loss.backward()

    assert_true(layer.weight.has_grad(), "Should backprop through sigmoid to linear")

fn test_bce_with_logits_end_to_end() raises:
    print("test_bce_with_logits_end_to_end")
    alias dtype = DType.float32
    var layer = Linear[dtype](2, 1, xavier=True)
    layer.train()

    var X = Tensor[dtype].d2([[1.0, 2.0]])
    var y = Tensor[dtype].d2([[1.0]])

    var logits = layer(X)
    var criterion = BCEWithLogitsLoss[dtype]()
    criterion.train()
    var loss = criterion(logits, y)
    loss.backward()

    assert_true(layer.weight.has_grad(), "BCE with logits should backprop to linear")

# ============================================================================
# Edge Cases
# ============================================================================

fn test_mse_loss_single_element() raises:
    print("test_mse_loss_single_element")
    alias dtype = DType.float32
    var pred = Tensor[dtype].d1([5.0], requires_grad=True)
    var target = Tensor[dtype].d1([3.0])
    var loss = pred.mse[track_grad=True](target)
    loss.backward()
    # (5-3)^2 = 4, gradient = 2*(5-3) = 4
    var expected_grad = Tensor[dtype].d1([4.0])
    assert_true(pred.grad().all_close(expected_grad), "Single element MSE gradient mismatch")


fn test_loss_with_zero_gradient() raises:
    print("test_loss_with_zero_gradient")
    alias dtype = DType.float32
    var pred = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
    var target = Tensor[dtype].d2([[1.0, 2.0]])
    var loss = pred.mse[track_grad=True](target)
    loss.backward()
    # Perfect prediction should give zero gradient
    var expected_grad = Tensor[dtype].d2([[0.0, 0.0]])
    assert_true(pred.grad().all_close[atol=1e-6](expected_grad), "Perfect prediction should have zero gradient")

# ============================================================================
# Master Test Runner
# ============================================================================

fn main() raises:
    run_all_loss_tests()

fn run_all_loss_tests() raises:
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE LOSS FUNCTION TESTS")
    print("="*80 + "\n")

    # MSE Loss Tests
    print("\n--- MSE Loss Tests ---")
    test_mse_loss_perfect_prediction()
    test_mse_loss_simple_gradient()
    test_mse_loss_batch_gradient()
    test_mse_loss_no_grad_mode()
    test_mse_loss_struct_train_mode()
    test_mse_loss_struct_eval_mode()

    # BCE Loss Tests
    print("\n--- BCE Loss Tests ---")
    test_bce_loss_perfect_prediction()
    test_bce_loss_worst_prediction()
    test_bce_loss_gradient_positive_target()
    test_bce_loss_gradient_negative_target()
    test_bce_loss_clipping()
    test_bce_loss_no_grad_mode()
    test_bce_loss_struct_train_eval()

    # BCE with Logits Tests
    print("\n--- BCE with Logits Loss Tests ---")
    test_bce_with_logits_zero_logits()
    test_bce_with_logits_positive_gradient()
    test_bce_with_logits_negative_gradient()
    test_bce_with_logits_batch()
    test_bce_with_logits_no_grad_mode()
    test_bce_with_logits_struct()

    # Integration Tests
    print("\n--- Integration Tests ---")
    test_mse_loss_with_linear_layer()
    test_bce_loss_with_sigmoid_output()
    test_bce_with_logits_end_to_end()

    # Edge Cases
    print("\n--- Edge Case Tests ---")
    test_mse_loss_single_element()
    test_bce_loss_boundary_values()
    test_loss_with_zero_gradient()

    print("\n" + "="*80)
    print("ALL LOSS FUNCTION TESTS PASSED! ✓")
    print("="*80 + "\n")
