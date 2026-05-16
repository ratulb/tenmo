from tenmo.tensor import Tensor
from std.testing import assert_true, assert_false, TestSuite
from tenmo.common_utils import isnan, isinf, Epsilon
from std.math import log
from std.sys import has_accelerator
from std.math import log
from tenmo.shapes import Shape

fn main() raises:
    print("Example: Basic logarithm with epsilon")

    comptime dtype = DType.float64

    # Example 1: Basic usage
    var x = Tensor[dtype]([1.0, 2.0, 3.0], requires_grad=True)
    var y = x.log()

    # Example 2: With zero handling
    # var x_with_zero = Tensor[dtype]([0.0, 1.0, 2.0], requires_grad=True)
    # var y_with_zero = x_with_zero.log(epsilon=1e-10)

    # Example 3: Gradient computation
    var loss = y.sum()
    loss.backward()
    x.grad().print()


    TestSuite.discover_tests[__functions_in_module()]().run()






    print("All logarithm tests passed!")

# ============================================================================
# TESTS - Forward Pass
# ============================================================================


fn test_log_forward_basic() raises:
    """Test basic logarithm computation."""
    print("test_log_forward_basic")

    comptime dtype = DType.float64
    var x = Tensor[dtype]([1.0, 2.0, 3.0, 4.0])
    var y = x.log()

    # log(1) = 0, log(2)  0.693, log(3)  1.099, log(4)  1.386
    var expected = Tensor[dtype]([0.0, 0.693147, 1.098612, 1.386294])

    assert_true(y.all_close[atol=1e-5](expected))


fn test_log_forward_with_epsilon() raises:
    """Test logarithm with very small values (epsilon handling)."""
    print("test_log_forward_with_epsilon")

    comptime dtype = DType.float64
    var x = Tensor[dtype]([1e-15, 1e-13, 1.0, 10.0])
    comptime epsilon = Scalar[dtype](1e-12)
    var y = x.log[epsilon=epsilon]()

    # First two values should be clamped to epsilon
    # log(1e-12)  -27.63
    var expected_first = log(epsilon)

    # Should not crash and should produce finite values
    assert_true(not isnan(y[0]))
    assert_true(not isinf(y[0]))
    assert_true(abs(y[0] - expected_first) < 1e-6)


fn test_log_forward_zero_handling() raises:
    """Test that zero values are handled safely."""
    print("test_log_forward_zero_handling")

    comptime dtype = DType.float64
    var x = Tensor[dtype]([0.0, 1.0, 2.0])
    comptime epsilon = Scalar[dtype](1e-10)
    var y = x.log[epsilon=epsilon]()
    # log(0) should become log(epsilon)
    var expected_zero = log(epsilon)
    assert_true(abs(y[0] - expected_zero) < 1e-6)

    # Other values should be normal
    assert_true(abs(y[1] - 0.0) < 1e-6)  # log(1) = 0


fn test_log_forward_negative_values() raises:
    """Test that negative values are handled (clamped to epsilon)."""
    print("test_log_forward_negative_values")

    comptime dtype = DType.float64
    var x = Tensor[dtype]([-1.0, -0.5, 1.0])
    comptime epsilon = Scalar[dtype](1e-10)
    var y = x.log[epsilon=epsilon]()

    # Negative values should be clamped to epsilon
    var expected_neg = log(epsilon)
    assert_true(abs(y[0] - expected_neg) < 1e-6)
    assert_true(abs(y[1] - expected_neg) < 1e-6)


fn test_log_forward_2d() raises:
    """Test logarithm on 2D tensor."""
    print("test_log_forward_2d")

    comptime dtype = DType.float64
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var y = x.log()

    var expected = Tensor[dtype].d2([[0.0, 0.693147], [1.098612, 1.386294]])

    assert_true(y.all_close[atol=1e-5](expected))


fn test_log_forward_e() raises:
    """Test that log(e) = 1."""
    print("test_log_forward_e")

    comptime dtype = DType.float64
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

    comptime dtype = DType.float64
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

    comptime dtype = DType.float64
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

    comptime dtype = DType.float64
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

    comptime dtype = DType.float64
    comptime epsilon = Scalar[dtype](1e-10)
    var x = Tensor[dtype]([1e-15, 1.0, 2.0], requires_grad=True)
    var y = x.log[epsilon=epsilon]()
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

    comptime dtype = DType.float64
    comptime epsilon = Scalar[dtype](1e-8)
    var x = Tensor[dtype]([0.0, 1.0, 2.0], requires_grad=True)
    var y = x.log[epsilon=epsilon]()
    var loss = y.sum()
    loss.backward()

    # Zero is clamped to epsilon, gradient is 1/epsilon
    var expected_zero_grad = 1.0 / epsilon

    assert_true(
        abs(x.grad()[0] - expected_zero_grad) < 1.0
    )  # Should be very large
    assert_true(x.grad()[0] > 1e6)  # Should be large


fn test_log_backward_2d() raises:
    """Test gradient computation on 2D tensor."""
    print("test_log_backward_2d")

    comptime dtype = DType.float64
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.log()
    var loss = y.sum()
    loss.backward()

    var expected = Tensor[dtype].d2([[1.0, 0.5], [0.333333, 0.25]])

    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_log_backward_multiple_uses() raises:
    """Test gradient when log output is used multiple times."""
    print("test_log_backward_multiple_uses")

    comptime dtype = DType.float64
    var x = Tensor[dtype]([2.0, 3.0], requires_grad=True)
    var y = x.log()

    # Use y twice
    var z1 = y * 2.0
    var z2 = y * 3.0
    var loss = z1.sum() + z2.sum()
    loss.backward()

    # Gradient accumulation: (2 + 3) / x = 5 / x
    var expected = Tensor[dtype]([2.5, 5.0 / 3.0])

    assert_true(x.grad().all_close[atol=1e-5](expected))


# ============================================================================
# NUMERICAL STABILITY TESTS
# ============================================================================


fn test_log_numerical_stability_large() raises:
    """Test numerical stability with large values."""
    print("test_log_numerical_stability_large")

    comptime dtype = DType.float64
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

    comptime dtype = DType.float64
    comptime epsilon = Scalar[dtype](1e-12)
    var x = Tensor[dtype]([1e-20, 1e-15, 1e-10])
    var y = x.log[epsilon=epsilon]()

    # Should produce finite negative values
    for i in range(3):
        assert_true(not isnan(y[i]))
        assert_true(not isinf(y[i]))


fn test_log_gradient_stability() raises:
    """Test that gradients remain stable near zero."""
    print("test_log_gradient_stability")

    comptime dtype = DType.float64
    comptime epsilon = Scalar[dtype](1e-10)
    var x = Tensor[dtype].d1([1e-12, 1e-8, 1.0], requires_grad=True)
    var y = x.log[epsilon=epsilon]()
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

    comptime dtype = DType.float64
    var x = Tensor[dtype]([2.718281828459045], requires_grad=True)  # e
    var y = x.log()
    y.backward()

    assert_true(abs(y[0] - 1.0) < 1e-9)  # log(e) = 1
    assert_true(abs(x.grad()[0] - (1.0 / 2.718281828459045)) < 1e-9)  # 1/e


fn test_log_all_ones() raises:
    """Test logarithm of tensor with all ones."""
    print("test_log_all_ones")

    comptime dtype = DType.float64
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

    comptime dtype = DType.float64
    var x = Tensor[dtype]([0.0, 1.0])

    # Test with different epsilon values
    var y1 = x.log[epsilon=1e-6]()
    var y2 = x.log[epsilon=1e-10]()
    var y3 = x.log[epsilon=1e-15]()

    # Different epsilons should give different results for zero
    assert_true(abs(y1[0] - log(Scalar[dtype](1e-6))) < 1e-8)
    assert_true(abs(y2[0] - log(Scalar[dtype](1e-10))) < 1e-8)
    assert_true(abs(y3[0] - log(Scalar[dtype](1e-15))) < 1e-8)


# ============================================================================
# MASTER TEST RUNNER
# ============================================================================


fn run_all_log_tests() raises:
    """Run all logarithm tests."""
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE LOGARITHM TEST SUITE")
    print("=" * 60 + "\n")

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

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================




# ── CPU Forward Tests ─────────────────────────────────────────────────────────


fn test_log_cpu_1d_basic_forward() raises:
    print("test_log_cpu_1d_basic_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.log()
    var expect = Tensor[dtype].d1(
        [log(Float32(1.0)), log(Float32(2.0)), log(Float32(3.0))]
    )
    assert_true(result.all_close(expect))


fn test_log_cpu_2d_forward() raises:
    print("test_log_cpu_2d_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var result = a.log()
    var expect = Tensor[dtype].d2(
        [
            [log(Float32(1.0)), log(Float32(2.0))],
            [log(Float32(3.0)), log(Float32(4.0))],
        ]
    )
    assert_true(result.all_close(expect))


fn test_log_cpu_3d_forward() raises:
    print("test_log_cpu_3d_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var result = a.log()
    var expect = Tensor[dtype].d3(
        [
            [
                [log(Float32(1.0)), log(Float32(2.0))],
                [log(Float32(3.0)), log(Float32(4.0))],
            ],
            [
                [log(Float32(5.0)), log(Float32(6.0))],
                [log(Float32(7.0)), log(Float32(8.0))],
            ],
        ]
    )
    assert_true(result.all_close(expect))


fn test_log_cpu_ones_forward() raises:
    print("test_log_cpu_ones_forward")
    comptime dtype = DType.float32
    # log(1) = 0
    var a = Tensor[dtype].ones(Shape(4))
    var result = a.log()
    assert_true(result.all_close(Tensor[dtype].zeros(Shape(4))))


fn test_log_cpu_epsilon_clamping() raises:
    print("test_log_cpu_epsilon_clamping")
    comptime dtype = DType.float32
    # Values <= 0 should be clamped to epsilon before log
    # With default epsilon=1e-12, log(1e-12) ≈ -27.631
    var a = Tensor[dtype].d1([0.0, -1.0, 1.0])
    var result = a.log()
    # log(1e-12) for first two, log(1.0)=0 for last
    var expected_clamped = log(Float32(1e-7))
    assert_true(result[[0]] == expected_clamped)
    assert_true(result[[1]] == expected_clamped)
    assert_true(result[[2]] == Float32(0.0))


fn test_log_cpu_custom_epsilon() raises:
    print("test_log_cpu_custom_epsilon")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([0.0, 1.0, 2.0])
    var result = a.log[epsilon = Scalar[dtype](1e-6)]()
    var expected_clamped = log(Float32(1e-6))
    assert_true(result[[0]] == expected_clamped)
    assert_true(result[[1]] == Float32(0.0))
    assert_true(result[[2]] == log(Float32(2.0)))


fn test_log_cpu_large_values() raises:
    print("test_log_cpu_large_values")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([100.0, 1000.0, 10000.0])
    var result = a.log()
    var expect = Tensor[dtype].d1(
        [log(Float32(100.0)), log(Float32(1000.0)), log(Float32(10000.0))]
    )
    assert_true(result.all_close(expect))


fn test_log_cpu_no_grad() raises:
    print("test_log_cpu_no_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=False)
    var result = a.log()
    assert_true(not result.requires_grad)


fn test_log_cpu_requires_grad_propagates() raises:
    print("test_log_cpu_requires_grad_propagates")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var result = a.log()
    assert_true(result.requires_grad)


fn test_log_cpu_suppress_grad() raises:
    print("test_log_cpu_suppress_grad")
    comptime dtype = DType.float32
    # requires_grad=True on input but suppressed via parameter
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var result = a.log(requires_grad=False)
    assert_true(not result.requires_grad)


# ── CPU Backward Tests ────────────────────────────────────────────────────────


fn test_log_cpu_1d_backward() raises:
    print("test_log_cpu_1d_backward")
    comptime dtype = DType.float32
    # d/dx log(x) = 1/x
    var a = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
    var result = a.log()
    var loss = result.sum()
    loss.backward()
    # grad = 1/x = [1.0, 0.5, 0.25]
    assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 0.5, 0.25])))


fn test_log_cpu_2d_backward() raises:
    print("test_log_cpu_2d_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [4.0, 8.0]], requires_grad=True)
    var result = a.log()
    var loss = result.sum()
    loss.backward()
    # grad = 1/x
    assert_true(
        a.grad().all_close(Tensor[dtype].d2([[1.0, 0.5], [0.25, 0.125]]))
    )


fn test_log_cpu_3d_backward() raises:
    print("test_log_cpu_3d_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [4.0, 8.0]], [[1.0, 4.0], [2.0, 8.0]]],
        requires_grad=True,
    )
    var result = a.log()
    var loss = result.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d3(
                [[[1.0, 0.5], [0.25, 0.125]], [[1.0, 0.25], [0.5, 0.125]]]
            )
        )
    )


fn test_log_cpu_backward_chain() raises:
    print("test_log_cpu_backward_chain")
    comptime dtype = DType.float32
    # Chain: log(x) * 2 → sum → backward
    # grad of x = 2/x
    var a = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
    var result = a.log() * 2.0
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 1.0, 0.5])))


fn test_log_cpu_backward_epsilon_clamping() raises:
    print("test_log_cpu_backward_epsilon_clamping")
    comptime dtype = DType.float32
    # For values <= epsilon, grad = 1/epsilon not 1/x
    # Default epsilon = 1e-12
    var a = Tensor[dtype].d1([0.0, 1.0, 2.0], requires_grad=True)
    var result = a.log()
    var loss = result.sum()
    loss.backward()
    # grad[0] = 1/epsilon = 1/1e-12
    # grad[1] = 1/1.0 = 1.0
    # grad[2] = 1/2.0 = 0.5
    assert_true(a.grad()[[0]] == Float32(1.0) / Float32(1e-7))
    assert_true(a.grad()[[1]] == Float32(1.0))
    assert_true(a.grad()[[2]] == Float32(0.5))


fn test_log_cpu_backward_custom_epsilon() raises:
    print("test_log_cpu_backward_custom_epsilon")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([0.0, 1.0, 4.0], requires_grad=True)
    var result = a.log[epsilon = Scalar[dtype](1e-6)]()
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad()[[0]] == Float32(1.0) / Float32(1e-6))
    assert_true(a.grad()[[1]] == Float32(1.0))
    assert_true(a.grad()[[2]] == Float32(0.25))


fn test_log_cpu_backward_chained_with_exp() raises:
    print("test_log_cpu_backward_chained_with_exp")
    comptime dtype = DType.float32
    # log(exp(x)) = x, so grad should be 1.0 everywhere
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var result = a.exp().log()
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(3))))


# ── GPU Forward Tests ─────────────────────────────────────────────────────────


fn test_log_gpu_1d_basic_forward() raises:
    comptime if has_accelerator():
        print("test_log_gpu_1d_basic_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.log()
        assert_true(result.is_on_gpu())
        var expect = Tensor[dtype].d1(
            [log(Float32(1.0)), log(Float32(2.0)), log(Float32(3.0))]
        )
        assert_true(result.to_cpu().all_close(expect))


fn test_log_gpu_2d_forward() raises:
    comptime if has_accelerator():
        print("test_log_gpu_2d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var result = a.log()
        assert_true(result.is_on_gpu())
        var expect = Tensor[dtype].d2(
            [
                [log(Float32(1.0)), log(Float32(2.0))],
                [log(Float32(3.0)), log(Float32(4.0))],
            ]
        )
        assert_true(result.to_cpu().all_close(expect))


fn test_log_gpu_3d_forward() raises:
    comptime if has_accelerator():
        print("test_log_gpu_3d_forward")
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var result = a.log()
        assert_true(result.is_on_gpu())
        var expect = Tensor[dtype].d3(
            [
                [
                    [log(Float32(1.0)), log(Float32(2.0))],
                    [log(Float32(3.0)), log(Float32(4.0))],
                ],
                [
                    [log(Float32(5.0)), log(Float32(6.0))],
                    [log(Float32(7.0)), log(Float32(8.0))],
                ],
            ]
        )
        assert_true(result.to_cpu().all_close(expect))


fn test_log_gpu_ones_forward() raises:
    comptime if has_accelerator():
        print("test_log_gpu_ones_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].ones(Shape(4)).to_gpu()
        var result = a.log()
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close(Tensor[dtype].zeros(Shape(4))))


fn test_log_gpu_epsilon_clamping() raises:
    comptime if has_accelerator():
        print("test_log_gpu_epsilon_clamping")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([0.0, -1.0, 1.0]).to_gpu()
        var result = a.log()
        var result_cpu = result.to_cpu()
        var expected_clamped = log(Float32(1e-7))
        assert_true(result_cpu[[0]] == expected_clamped)
        assert_true(result_cpu[[1]] == expected_clamped)
        assert_true(result_cpu[[2]] == Float32(0.0))


fn test_log_gpu_large_values() raises:
    comptime if has_accelerator():
        print("test_log_gpu_large_values")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([100.0, 1000.0, 10000.0]).to_gpu()
        var result = a.log()
        var expect = Tensor[dtype].d1(
            [log(Float32(100.0)), log(Float32(1000.0)), log(Float32(10000.0))]
        )
        assert_true(result.to_cpu().all_close(expect))


# ── GPU Backward Tests ────────────────────────────────────────────────────────
fn test_log_gpu_1d_backward() raises:
    comptime if has_accelerator():
        print("test_log_gpu_1d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 0.5, 0.25])))


fn test_log_gpu_2d_backward() raises:
    comptime if has_accelerator():
        print("test_log_gpu_2d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [4.0, 8.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log()
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 0.5], [0.25, 0.125]]))
        )


fn test_log_gpu_3d_backward() raises:
    comptime if has_accelerator():
        print("test_log_gpu_3d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [4.0, 8.0]], [[1.0, 4.0], [2.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.log()
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [
                        [[1.0, 0.5], [0.25, 0.125]],
                        [[1.0, 0.25], [0.5, 0.125]],
                    ]
                )
            )
        )


fn test_log_gpu_backward_chain() raises:
    comptime if has_accelerator():
        print("test_log_gpu_backward_chain")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log() * 2.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 1.0, 0.5])))


fn test_log_gpu_backward_epsilon_clamping() raises:
    comptime if has_accelerator():
        print("test_log_gpu_backward_epsilon_clamping")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([0.0, 1.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad()[[0]] == Float32(1.0) / Float32(1e-7))
        assert_true(a.grad()[[1]] == Float32(1.0))
        assert_true(a.grad()[[2]] == Float32(0.5))


fn test_log_gpu_backward_chained_with_exp() raises:
    comptime if has_accelerator():
        print("test_log_gpu_backward_chained_with_exp")
        comptime dtype = DType.float32
        # log(exp(x)) = x, grad should be 1.0 everywhere
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.exp().log()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(3))))


fn test_log_gpu_backward_custom_epsilon() raises:
    comptime if has_accelerator():
        print("test_log_gpu_backward_custom_epsilon")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([0.0, 1.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log[epsilon = Scalar[dtype](1e-6)]()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad()[[0]] == Float32(1.0) / Float32(1e-6))
        assert_true(a.grad()[[1]] == Float32(1.0))
        assert_true(a.grad()[[2]] == Float32(0.25))


# ── CPU/GPU Parity Tests ──────────────────────────────────────────────────────


fn test_log_parity_1d_forward() raises:
    comptime if has_accelerator():
        print("test_log_parity_1d_forward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
        var a_gpu = a_cpu.to_gpu()
        var result_cpu = a_cpu.log()
        var result_gpu = a_gpu.log()
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


fn test_log_parity_2d_forward() raises:
    comptime if has_accelerator():
        print("test_log_parity_2d_forward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a_cpu.to_gpu()
        var result_cpu = a_cpu.log()
        var result_gpu = a_gpu.log()
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


fn test_log_parity_1d_backward() raises:
    comptime if has_accelerator():
        print("test_log_parity_1d_backward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 4.0, 8.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.log().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.log().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


fn test_log_parity_2d_backward() raises:
    comptime if has_accelerator():
        print("test_log_parity_2d_backward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [4.0, 8.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.log().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.log().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


fn test_log_parity_epsilon_clamping() raises:
    comptime if has_accelerator():
        print("test_log_parity_epsilon_clamping")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([0.0, -1.0, 1.0, 2.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.log().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.log().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


fn test_log_parity_chain_exp() raises:
    comptime if has_accelerator():
        print("test_log_parity_chain_exp")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.exp().log().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.exp().log().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))

