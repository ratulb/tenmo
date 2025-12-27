from tenmo import Tensor
from testing import assert_true, assert_false
from common_utils import isnan, isinf
from math import log
# ============================================================================
# TESTS - Forward Pass
# ============================================================================

fn test_log_forward_basic() raises:
    """Test basic logarithm computation."""
    print("test_log_forward_basic")

    alias dtype = DType.float64
    var x = Tensor[dtype]([1.0, 2.0, 3.0, 4.0])
    var y = x.log()

    # log(1) = 0, log(2)  0.693, log(3)  1.099, log(4)  1.386
    var expected = Tensor[dtype]([0.0, 0.693147, 1.098612, 1.386294])

    assert_true(y.all_close[atol=1e-5](expected))



fn test_log_forward_with_epsilon() raises:
    """Test logarithm with very small values (epsilon handling)."""
    print("test_log_forward_with_epsilon")

    alias dtype = DType.float64
    var x = Tensor[dtype]([1e-15, 1e-13, 1.0, 10.0])
    var epsilon = Scalar[dtype](1e-12)
    var y = x.log(epsilon=epsilon)

    # First two values should be clamped to epsilon
    # log(1e-12) ˜ -27.63
    var expected_first = log(epsilon)

    # Should not crash and should produce finite values
    assert_true(not isnan(y[0]))
    assert_true(not isinf(y[0]))
    assert_true(abs(y[0] - expected_first) < 1e-6)



fn test_log_forward_zero_handling() raises:
    """Test that zero values are handled safely."""
    print("test_log_forward_zero_handling")

    alias dtype = DType.float64
    var x = Tensor[dtype]([0.0, 1.0, 2.0])
    var epsilon = Scalar[dtype](1e-10)
    var y = x.log(epsilon=epsilon)

    # log(0) should become log(epsilon)
    var expected_zero = log(epsilon)
    assert_true(abs(y[0] - expected_zero) < 1e-6)

    # Other values should be normal
    assert_true(abs(y[1] - 0.0) < 1e-6)  # log(1) = 0



fn test_log_forward_negative_values() raises:
    """Test that negative values are handled (clamped to epsilon)."""
    print("test_log_forward_negative_values")

    alias dtype = DType.float64
    var x = Tensor[dtype]([-1.0, -0.5, 1.0])
    var epsilon = Scalar[dtype](1e-10)
    var y = x.log(epsilon=epsilon)

    # Negative values should be clamped to epsilon
    var expected_neg = log(epsilon)
    assert_true(abs(y[0] - expected_neg) < 1e-6)
    assert_true(abs(y[1] - expected_neg) < 1e-6)



fn test_log_forward_2d() raises:
    """Test logarithm on 2D tensor."""
    print("test_log_forward_2d")

    alias dtype = DType.float64
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var y = x.log()

    var expected = Tensor[dtype].d2([
        [0.0, 0.693147],
        [1.098612, 1.386294]
    ])

    assert_true(y.all_close[atol=1e-5](expected))



fn test_log_forward_e() raises:
    """Test that log(e) = 1."""
    print("test_log_forward_e")

    alias dtype = DType.float64
    var e = Scalar[dtype](2.718281828459045)
    var x = Tensor[dtype].scalar(e)
    var y = x.log()
    assert_true(abs(y.item() - 1.0) < 1e-9)



# ============================================================================
# TESTS - Backward Pass
# ============================================================================

fn test_log_backward_basic() raises:
    """Test basic gradient computation: d/dx log(x) = 1/x."""
    print("test_log_backward_basic")

    alias dtype = DType.float64
    var x = Tensor[dtype]([2.0, 3.0, 4.0], requires_grad=True)
    var y = x.log()
    var loss = y.sum()
    loss.backward()

    # Gradient: d/dx log(x) = 1/x
    # Expected: [1/2, 1/3, 1/4] = [0.5, 0.333333, 0.25]
    var expected = Tensor[dtype]([0.5, 0.333333, 0.25])

    assert_true(x.grad().all_close[atol=1e-5](expected))



fn test_log_backward_weighted() raises:
    """Test gradient with weighted loss."""
    print("test_log_backward_weighted")

    alias dtype = DType.float64
    var x = Tensor[dtype]([1.0, 2.0, 4.0], requires_grad=True)
    var y = x.log()

    # Apply weights
    var weights = Tensor[dtype]([2.0, 3.0, 4.0])
    var weighted = y * weights
    var loss = weighted.sum()
    loss.backward()

    # Gradient: weight / x
    # Expected: [2/1, 3/2, 4/4] = [2.0, 1.5, 1.0]
    var expected = Tensor[dtype]([2.0, 1.5, 1.0])

    assert_true(x.grad().all_close[atol=1e-5](expected))



fn test_log_backward_chain_rule() raises:
    """Test gradient with chain rule: d/dx log(x^2)."""
    print("test_log_backward_chain_rule")

    alias dtype = DType.float64
    var x = Tensor[dtype]([2.0, 3.0, 4.0], requires_grad=True)
    var x_squared = x * x  # x^2
    var y = x_squared.log()  # log(x^2)
    var loss = y.sum()
    loss.backward()

    # d/dx log(x^2) = (1/x^2) * 2x = 2/x
    # Expected: [2/2, 2/3, 2/4] = [1.0, 0.666667, 0.5]
    var expected = Tensor[dtype]([1.0, 0.666667, 0.5])

    assert_true(x.grad().all_close[atol=1e-4](expected))



fn test_log_backward_with_epsilon() raises:
    """Test gradient computation with epsilon (small values)."""
    print("test_log_backward_with_epsilon")

    alias dtype = DType.float64
    var epsilon = Scalar[dtype](1e-10)
    var x = Tensor[dtype]([1e-15, 1.0, 2.0], requires_grad=True)
    var y = x.log(epsilon=epsilon)
    var loss = y.sum()
    loss.backward()

    # First value is clamped to epsilon, so gradient is 1/epsilon
    # Other gradients are normal: [1/1, 1/2]
    var expected_first = 1.0 / epsilon

    assert_true(abs(x.grad()[0] - expected_first) < 1e-3)
    assert_true(abs(x.grad()[1] - 1.0) < 1e-5)
    assert_true(abs(x.grad()[2] - 0.5) < 1e-5)



fn test_log_backward_zero_input() raises:
    """Test gradient with zero input (should use epsilon)."""
    print("test_log_backward_zero_input")

    alias dtype = DType.float64
    var epsilon = Scalar[dtype](1e-8)
    var x = Tensor[dtype]([0.0, 1.0, 2.0], requires_grad=True)
    var y = x.log(epsilon=epsilon)
    var loss = y.sum()
    loss.backward()

    # Zero is clamped to epsilon, gradient is 1/epsilon
    var expected_zero_grad = 1.0 / epsilon

    assert_true(abs(x.grad()[0] - expected_zero_grad) < 1.0)  # Should be very large
    assert_true(x.grad()[0] > 1e6)  # Should be large



fn test_log_backward_2d() raises:
    """Test gradient computation on 2D tensor."""
    print("test_log_backward_2d")

    alias dtype = DType.float64
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.log()
    var loss = y.sum()
    loss.backward()

    var expected = Tensor[dtype].d2([
        [1.0, 0.5],
        [0.333333, 0.25]
    ])

    assert_true(x.grad().all_close[atol=1e-5](expected))



fn test_log_backward_multiple_uses() raises:
    """Test gradient when log output is used multiple times."""
    print("test_log_backward_multiple_uses")

    alias dtype = DType.float64
    var x = Tensor[dtype]([2.0, 3.0], requires_grad=True)
    var y = x.log()

    # Use y twice
    var z1 = y * 2.0
    var z2 = y * 3.0
    var loss = z1.sum() + z2.sum()
    loss.backward()

    # Gradient accumulation: (2 + 3) / x = 5 / x
    var expected = Tensor[dtype]([2.5, 5.0/3.0])

    assert_true(x.grad().all_close[atol=1e-5](expected))



# ============================================================================
# NUMERICAL STABILITY TESTS
# ============================================================================

fn test_log_numerical_stability_large() raises:
    """Test numerical stability with large values."""
    print("test_log_numerical_stability_large")

    alias dtype = DType.float64
    var x = Tensor[dtype]([1e10, 1e20, 1e30])
    var y = x.log()

    # Should produce finite values
    for i in range(3):
        assert_true(not isnan(y[i]))
        assert_true(not isinf(y[i]))
        assert_true(y[i] > 0.0)  # log of large positive is positive




fn test_log_numerical_stability_small() raises:
    """Test numerical stability with very small values."""
    print("test_log_numerical_stability_small")

    alias dtype = DType.float64
    var epsilon = Scalar[dtype](1e-12)
    var x = Tensor[dtype]([1e-20, 1e-15, 1e-10])
    var y = x.log(epsilon=epsilon)

    # Should produce finite negative values
    for i in range(3):
        assert_true(not isnan(y[i]))
        assert_true(not isinf(y[i]))




fn test_log_gradient_stability() raises:
    """Test that gradients remain stable near zero."""
    print("test_log_gradient_stability")

    alias dtype = DType.float64
    var epsilon = Scalar[dtype](1e-10)
    var x = Tensor[dtype].d1([1e-12, 1e-8, 1.0], requires_grad=True)
    var y = x.log(epsilon=epsilon)
    var loss = y.sum()
    loss.backward()
    # All gradients should be finite
    for i in range(3):
        assert_false(isnan(x.grad()[i]))
        assert_false(isinf(x.grad()[i]))




# ============================================================================
# EDGE CASES
# ============================================================================

fn test_log_single_element() raises:
    """Test logarithm of single element tensor."""
    print("test_log_single_element")

    alias dtype = DType.float64
    var x = Tensor[dtype]([2.718281828459045], requires_grad=True)  # e
    var y = x.log()
    y.backward()

    assert_true(abs(y[0] - 1.0) < 1e-9)  # log(e) = 1
    assert_true(abs(x.grad()[0] - (1.0/2.718281828459045)) < 1e-9)  # 1/e



fn test_log_all_ones() raises:
    """Test logarithm of tensor with all ones."""
    print("test_log_all_ones")

    alias dtype = DType.float64
    var x = Tensor[dtype].ones(5, requires_grad=True)
    var y = x.log()
    var loss = y.sum()
    loss.backward()

    # log(1) = 0, gradient = 1/1 = 1
    assert_true(y.sum().item() < 1e-10)  # All zeros
    assert_true(x.grad().all_close[atol=1e-6](Tensor[dtype].ones(5)))



fn test_log_custom_epsilon_values() raises:
    """Test different epsilon values."""
    print("test_log_custom_epsilon_values")

    alias dtype = DType.float64
    var x = Tensor[dtype]([0.0, 1.0])

    # Test with different epsilon values
    var y1 = x.log(epsilon=1e-6)
    var y2 = x.log(epsilon=1e-10)
    var y3 = x.log(epsilon=1e-15)

    # Different epsilons should give different results for zero
    assert_true(abs(y1[0] - log(Scalar[dtype](1e-6))) < 1e-8)
    assert_true(abs(y2[0] - log(Scalar[dtype](1e-10))) < 1e-8)
    assert_true(abs(y3[0] - log(Scalar[dtype](1e-15))) < 1e-8)




# ============================================================================
# MASTER TEST RUNNER
# ============================================================================

fn run_all_log_tests() raises:
    """Run all logarithm tests."""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE LOGARITHM TEST SUITE")
    print("="*60 + "\n")

    # Forward pass tests
    print("--- LOG FORWARD TESTS ---")
    test_log_forward_basic()
    test_log_forward_with_epsilon()
    test_log_forward_zero_handling()
    test_log_forward_negative_values()
    test_log_forward_2d()
    test_log_forward_e()

    # Backward pass tests
    print("\n--- LOG BACKWARD TESTS ---")
    test_log_backward_basic()
    test_log_backward_weighted()
    test_log_backward_chain_rule()
    test_log_backward_with_epsilon()
    test_log_backward_zero_input()
    test_log_backward_2d()
    test_log_backward_multiple_uses()

    # Numerical stability tests
    print("\n--- NUMERICAL STABILITY TESTS ---")
    test_log_numerical_stability_large()
    test_log_numerical_stability_small()
    test_log_gradient_stability()

    # Edge cases
    print("\n--- EDGE CASE TESTS ---")
    test_log_single_element()
    test_log_all_ones()
    test_log_custom_epsilon_values()

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

fn main() raises:
    """Example usage and quick verification."""
    print("Example: Basic logarithm with epsilon")

    alias dtype = DType.float64

    # Example 1: Basic usage
    var x = Tensor[dtype]([1.0, 2.0, 3.0], requires_grad=True)
    var y = x.log()
    print("\nlog([1, 2, 3]):")
    y.print()

    # Example 2: With zero handling
    var x_with_zero = Tensor[dtype]([0.0, 1.0, 2.0], requires_grad=True)
    var y_with_zero = x_with_zero.log(epsilon=1e-10)
    print("\nlog([0, 1, 2]) with epsilon=1e-10:")
    y_with_zero.print()

    # Example 3: Gradient computation
    var loss = y.sum()
    loss.backward()
    print("\nGradient (should be [1, 0.5, 0.333]):")
    x.grad().print()

    # Run all tests
    print("\n")
    run_all_log_tests()
