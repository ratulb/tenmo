from tenmo import Tensor
from net import Tanh, Linear
from common_utils import isnan, isinf
from shapes import Shape
from testing import assert_true
from intarray import IntArray


# ============================================================================
# Tanh Activation Tests
# ============================================================================

fn test_tanh_forward_values() raises:
    print("test_tanh_forward_values")
    alias dtype = DType.float32
    var x = Tensor[dtype].d1([0.0, 1.0, -1.0, 2.0, -2.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    # tanh(0) = 0, tanh(1) ≈ 0.762, tanh(-1) ≈ -0.762
    # tanh(2) ≈ 0.964, tanh(-2) ≈ -0.964
    assert_true(abs(y[IntArray(0)]) < 1e-6, "tanh(0) should be 0")
    assert_true(abs(y[IntArray(1)] - 0.7616) < 0.001, "tanh(1) ≈ 0.762")
    assert_true(abs(y[IntArray(2)] + 0.7616) < 0.001, "tanh(-1) ≈ -0.762")
    assert_true(abs(y[IntArray(3)] - 0.9640) < 0.001, "tanh(2) ≈ 0.964")
    assert_true(abs(y[IntArray(4)] + 0.9640) < 0.001, "tanh(-2) ≈ -0.964")

fn test_tanh_moderate_range() raises:
    print("test_tanh_moderate_range")
    alias dtype = DType.float32
    # For |x| < 5, tanh should be strictly within (-1, 1)
    var x = Tensor[dtype].d1([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    for i in range(7):
        var val = y[IntArray(i)]
        assert_true(val > -1.0 and val < 1.0, "tanh should be strictly in (-1, 1)")

fn test_tanh_range_bounded() raises:
    print("test_tanh_range_bounded")
    alias dtype = DType.float32
    var x = Tensor[dtype].d1([-10.0, -5.0, 0.0, 5.0, 10.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    # tanh output should be in [-1, 1] (closed interval due to FP precision)
    # For extreme values, tanh saturates to exactly ±1.0 in float32
    for i in range(5):
        var val = y[IntArray(i)]
        assert_true(val >= -1.0 and val <= 1.0, "tanh output should be in [-1, 1]")
        assert_true(not isnan(val), "tanh should not produce NaN")
        assert_true(not isinf(val), "tanh should not produce Inf")

    # Test 1: Extreme values (will saturate to ±1.0)
    var x_extreme = Tensor[dtype].d1([-10.0, 10.0], requires_grad=False)
    var y_extreme = x_extreme.tanh[track_grad=False]()
    assert_true(abs(y_extreme[IntArray(0)] + 1.0) < 1e-6, "tanh(-10) should be ≈ -1.0")
    assert_true(abs(y_extreme[IntArray(1)] - 1.0) < 1e-6, "tanh(10) should be ≈ 1.0")

    # Test 2: Moderate values (strictly within (-1, 1))
    var x_moderate = Tensor[dtype].d1([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=False)
    var y_moderate = x_moderate.tanh[track_grad=False]()
    for i in range(5):
        var val = y_moderate[IntArray(i)]
        assert_true(val > -1.0 and val < 1.0, "tanh should be strictly in (-1, 1) for moderate inputs")
        assert_true(not isnan(val), "tanh should not produce NaN")
        assert_true(not isinf(val), "tanh should not produce Inf")


fn test_tanh_saturation() raises:
    print("test_tanh_saturation")
    alias dtype = DType.float32
    # For large |x|, tanh saturates to ±1.0 due to float32 precision
    var x = Tensor[dtype].d1([-10.0, -20.0, 10.0, 20.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    # Check negative saturation
    assert_true(abs(y[IntArray(0)] + 1.0) < 1e-5, "tanh should saturate to -1.0")
    assert_true(abs(y[IntArray(1)] + 1.0) < 1e-5, "tanh should saturate to -1.0")

    # Check positive saturation
    assert_true(abs(y[IntArray(2)] - 1.0) < 1e-5, "tanh should saturate to 1.0")
    assert_true(abs(y[IntArray(3)] - 1.0) < 1e-5, "tanh should saturate to 1.0")

fn test_tanh_no_overflow() raises:
    print("test_tanh_no_overflow")
    alias dtype = DType.float32
    # Test that extreme values don't cause NaN or Inf
    var x = Tensor[dtype].d1([-100.0, -50.0, 50.0, 100.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    for i in range(4):
        var val = y[IntArray(i)]
        assert_true(not isnan(val), "tanh should not produce NaN")
        assert_true(not isinf(val), "tanh should not produce Inf")
        assert_true(abs(val) <= 1.0, "tanh magnitude should not exceed 1.0")

fn test_tanh_gradient_at_zero() raises:
    print("test_tanh_gradient_at_zero")
    alias dtype = DType.float32
    var x = Tensor[dtype].d1([0.0], requires_grad=True)
    var y = x.tanh[track_grad=True]()
    var loss = y.sum()
    loss.backward()

    # d/dx tanh(x) = 1 - tanh²(x)
    # At x=0: tanh(0) = 0, so gradient = 1 - 0² = 1
    var expected_grad = Tensor[dtype].d1([1.0])
    assert_true(x.grad().all_close[atol=1e-5](expected_grad), "tanh gradient at 0 should be 1")

fn test_tanh_gradient_positive() raises:
    print("test_tanh_gradient_positive")
    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0], requires_grad=True)
    var y = x.tanh[track_grad=True]()
    var loss = y.sum()
    loss.backward()

    # At x=1: tanh(1) ≈ 0.7616
    # gradient = 1 - 0.7616² ≈ 0.42
    var grad = x.grad()[IntArray(0)]
    assert_true(abs(grad - 0.42) < 0.01, "tanh gradient at x=1 should be ≈ 0.42")

fn test_tanh_gradient_negative() raises:
    print("test_tanh_gradient_negative")
    alias dtype = DType.float32
    var x = Tensor[dtype].d1([-1.0], requires_grad=True)
    var y = x.tanh[track_grad=True]()
    var loss = y.sum()
    loss.backward()

    # tanh is symmetric, gradient should be same as at x=1
    var grad = x.grad()[IntArray(0)]
    assert_true(abs(grad - 0.42) < 0.01, "tanh gradient at x=-1 should be ≈ 0.42")

fn test_tanh_gradient_saturation() raises:
    print("test_tanh_gradient_saturation")
    alias dtype = DType.float32
    var x = Tensor[dtype].d1([5.0, -5.0], requires_grad=True)
    var y = x.tanh[track_grad=True]()
    var loss = y.sum()
    loss.backward()

    # At large |x|, tanh saturates, gradient → 0
    # tanh(5) ≈ 0.9999, gradient ≈ 1 - 0.9999² ≈ 0.0001
    var grad_pos = x.grad()[IntArray(0)]
    var grad_neg = x.grad()[IntArray(1)]
    assert_true(grad_pos < 0.01, "tanh gradient should be small at x=5")
    assert_true(grad_neg < 0.01, "tanh gradient should be small at x=-5")

fn test_tanh_gradient_batch() raises:
    print("test_tanh_gradient_batch")
    alias dtype = DType.float32
    var x = Tensor[dtype].d2([[0.0, 1.0], [-1.0, 2.0]], requires_grad=True)
    var y = x.tanh[track_grad=True]()
    var loss = y.sum()
    loss.backward()

    # Verify gradients are computed for all elements
    assert_true(x.has_grad(), "Should have gradients")
    # All gradients should be in (0, 1] range
    for i in range(2):
        for j in range(2):
            var grad = x.grad()[IntArray(i, j)]
            assert_true(grad > 0.0 and grad <= 1.0, "tanh gradient should be in (0, 1]")

fn test_tanh_chain_rule() raises:
    print("test_tanh_chain_rule")
    alias dtype = DType.float32
    var x = Tensor[dtype].d1([0.5], requires_grad=True)
    var y = x.tanh[track_grad=True]()
    var z = y * y  # z = tanh²(x)
    var loss = z.sum()
    loss.backward()

    # d/dx [tanh²(x)] = 2*tanh(x) * (1 - tanh²(x))
    # At x=0.5: tanh(0.5) ≈ 0.4621
    # gradient ≈ 2 * 0.4621 * (1 - 0.4621²) ≈ 0.727
    var grad = x.grad()[IntArray(0)]
    assert_true(abs(grad - 0.727) < 0.01, "Chain rule gradient mismatch")

fn test_tanh_no_grad_mode() raises:
    print("test_tanh_no_grad_mode")
    alias dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var y = x.tanh[track_grad=False]()

    assert_true(not y.has_backward_fn(), "Should not build graph in no_grad mode")
    assert_true(not y.requires_grad, "Output should not require grad")

fn test_tanh_layer_train_mode() raises:
    print("test_tanh_layer_train_mode")
    alias dtype = DType.float32
    var activation = Tanh[dtype]()
    activation.train()

    var x = Tensor[dtype].d1([1.0], requires_grad=True)
    var y = activation(x)
    var loss = y.sum()
    loss.backward()

    assert_true(x.has_grad(), "Should have gradients in train mode")

fn test_tanh_layer_eval_mode() raises:
    print("test_tanh_layer_eval_mode")
    alias dtype = DType.float32
    var activation = Tanh[dtype]()
    activation.eval()

    var x = Tensor[dtype].d1([1.0], requires_grad=True)
    var y = activation(x)

    assert_true(not y.has_backward_fn(), "Should not build graph in eval mode")

fn test_tanh_numerical_stability_positive() raises:
    print("test_tanh_numerical_stability_positive")
    alias dtype = DType.float32
    # Test large positive values
    var x = Tensor[dtype].d1([10.0, 20.0, 50.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    # Should approach 1, not overflow
    for i in range(3):
        var val = y[IntArray(i)]
        assert_true(not isnan(val), "Should not be NaN")
        assert_true(not isinf(val), "Should not be Inf")
        assert_true(val > 0.999 and val <= 1.0, "Should be close to 1")

fn test_tanh_numerical_stability_negative() raises:
    print("test_tanh_numerical_stability_negative")
    alias dtype = DType.float32
    # Test large negative values
    var x = Tensor[dtype].d1([-10.0, -20.0, -50.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    # Should approach -1, not overflow
    for i in range(3):
        var val = y[IntArray(i)]
        assert_true(not isnan(val), "Should not be NaN")
        assert_true(not isinf(val), "Should not be Inf")
        assert_true(val <= -0.999 and val >= -1.0, "Should be close to -1")



fn test_tanh_contiguous_vs_non_contiguous() raises:
    print("test_tanh_contiguous_vs_non_contiguous")

    alias dtype = DType.float32
    var x_contig = Tensor[dtype].d1([0.0, 1.0, -1.0], requires_grad=True)
    var y_contig = x_contig.tanh[track_grad=True]()

    # Create non-contiguous tensor (e.g., via slice or transpose)
    var x_large = Tensor[dtype].d2([[0.0, 99.0], [1.0, 99.0], [-1.0, 99.0]])
    var x_non_contig = x_large[:, slice(0,1)]  # Slice to get non-contiguous
    x_non_contig.requires_grad_(True)
    var y_non_contig = x_non_contig.tanh[track_grad=True]()

    # Both should give same results
    assert_true(y_contig.unsqueeze[track_grad=False](-1).all_close[atol=1e-5](y_non_contig),
                "Contiguous and non-contiguous should match")

fn test_tanh_with_linear_layer() raises:
    print("test_tanh_with_linear_layer")
    alias dtype = DType.float32
    var layer = Linear[dtype](2, 3, xavier=True)
    var activation = Tanh[dtype]()
    layer.train()
    activation.train()

    var x = Tensor[dtype].d2([[1.0, 2.0]])
    var linear_out = layer(x)
    var y = activation(linear_out)
    var loss = y.sum()
    loss.backward()


    assert_true(layer.weight.has_grad(), "Linear weight should have gradient")
    assert_true(layer.bias.has_grad(), "Linear bias should have gradient")

# ============================================================================
# Edge Cases
# ============================================================================

fn test_tanh_zero_tensor() raises:
    print("test_tanh_zero_tensor")
    alias dtype = DType.float32
    var x = Tensor[dtype].zeros(Shape(3, 3), requires_grad=True)
    var y = x.tanh[track_grad=True]()
    var loss = y.sum()
    loss.backward()

    # tanh(0) = 0, gradient = 1
    var expected_output = Tensor[dtype].zeros(Shape(3, 3))
    var expected_grad = Tensor[dtype].ones(Shape(3, 3))
    assert_true(y.all_close[atol=1e-6](expected_output), "tanh(0) should be 0")
    assert_true(x.grad().all_close[atol=1e-5](expected_grad), "gradient should be 1")

fn test_tanh_symmetry() raises:
    print("test_tanh_symmetry")
    alias dtype = DType.float32
    var x_pos = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=False)
    var x_neg = Tensor[dtype].d1([-1.0, -2.0, -3.0], requires_grad=False)

    var y_pos = x_pos.tanh[track_grad=False]()
    var y_neg = x_neg.tanh[track_grad=False]()

    # tanh(-x) = -tanh(x)
    for i in range(3):
        var pos_val = y_pos[IntArray(i)]
        var neg_val = y_neg[IntArray(i)]
        assert_true(abs(pos_val + neg_val) < 1e-5, "tanh should be symmetric")

# ============================================================================
# Master Test Runner
# ============================================================================

fn run_all_tanh_tests() raises:
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE TANH ACTIVATION TESTS")
    print("="*80 + "\n")

    # Forward pass tests
    print("\n--- Forward Pass Tests ---")
    test_tanh_forward_values()
    test_tanh_range_bounded()
    test_tanh_moderate_range()
    test_tanh_saturation()  # New
    test_tanh_no_overflow()  # New
    test_tanh_numerical_stability_positive()
    test_tanh_numerical_stability_negative()
    test_tanh_symmetry()

    # Gradient tests
    print("\n--- Gradient Tests ---")
    test_tanh_gradient_at_zero()
    test_tanh_gradient_positive()
    test_tanh_gradient_negative()
    test_tanh_gradient_saturation()
    test_tanh_gradient_batch()
    test_tanh_chain_rule()

    # Mode tests
    print("\n--- Train/Eval Mode Tests ---")
    test_tanh_no_grad_mode()
    test_tanh_layer_train_mode()
    test_tanh_layer_eval_mode()

    # Integration tests
    print("\n--- Integration Tests ---")
    test_tanh_with_linear_layer()
    test_tanh_contiguous_vs_non_contiguous()

    # Edge cases
    print("\n--- Edge Case Tests ---")
    test_tanh_zero_tensor()

    print("\n" + "="*80)
    print("ALL TANH ACTIVATION TESTS PASSED! ✓")
    print("="*80 + "\n")

fn main() raises:
    run_all_tanh_tests()
