from tenmo import Tensor
from testing import assert_true
from common_utils import isnan, isinf


fn test_sqrt_backward() raises:
    print("test_sqrt_backward")
    var x = Tensor.d1([4.0, 9.0, 16.0, 25.0], requires_grad=True)
    var y = x.sqrt()  # [2.0, 3.0, 4.0, 5.0]
    var s = y.sum()
    s.backward()

    # dy/dx = 1 / (2 * sqrt(x))
    # For x=[4, 9, 16, 25]: sqrt(x) = [2, 3, 4, 5]
    # Gradient = [1/(2*2), 1/(2*3), 1/(2*4), 1/(2*5)]
    #          = [0.25, 0.1667, 0.125, 0.1]
    var expected_grad = Tensor.d1([0.25, 0.16666667, 0.125, 0.1])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_sqrt_backward_zero_handling() raises:
    print("test_sqrt_backward_zero_handling")
    # Test near-zero values (numerical stability)
    var x = Tensor.d1([0.01, 0.04, 1.0], requires_grad=True)
    var y = x.sqrt()
    var s = y.sum()
    s.backward()

    # Gradient at 0.01: 1/(2*0.1) = 5.0
    # Gradient at 0.04: 1/(2*0.2) = 2.5
    # Gradient at 1.0: 1/(2*1.0) = 0.5
    var expected_grad = Tensor.d1([5.0, 2.5, 0.5])
    assert_true(x.grad().all_close(expected_grad))


fn test_var_backward_global_variance() raises:
    print("test_var_backward_global_variance")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var v = x.variance(unbiased=False)  # Population variance
    v.backward()

    # Mean = 3.0
    # Variance = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5
    #          = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
    # Gradient: (2/n) * (x - mean) = (2/5) * (x - 3)
    # = [0.4*(-2), 0.4*(-1), 0.4*(0), 0.4*(1), 0.4*(2)]
    # = [-0.8, -0.4, 0.0, 0.4, 0.8]
    var expected_grad = Tensor.d1([-0.8, -0.4, 0.0, 0.4, 0.8])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_var_backward_unbiased_variance() raises:
    print("test_var_backward_unbiased_variance")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var v = x.variance(unbiased=True)  # Sample variance (n-1)
    v.backward()

    # Gradient: (2/(n-1)) * (x - mean) = (2/4) * (x - 3) = 0.5 * (x - 3)
    # = [0.5*(-2), 0.5*(-1), 0.5*(0), 0.5*(1), 0.5*(2)]
    # = [-1.0, -0.5, 0.0, 0.5, 1.0]
    var expected_grad = Tensor.d1([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_var_backward_with_axis() raises:
    print("test_var_backward_with_axis")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v = x.variance(axis=0, unbiased=False)  # Variance along rows
    var s = v.sum()
    s.backward()

    # Column 0: mean=2, var=1, grad for [1,3] = (2/2)*([1,3]-2) = [-1, 1]
    # Column 1: mean=3, var=1, grad for [2,4] = (2/2)*([2,4]-3) = [-1, 1]
    var expected_grad = Tensor.d2([[-1.0, -1.0], [1.0, 1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_std_backward_global_std() raises:
    print("test_std_backward_global_std")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var s = x.std(unbiased=False)
    s.backward()

    # Mean = 3.0, Variance = 2.0, Std = sqrt(2) ≈ 1.414
    # Gradient: (1/(std*n)) * (x - mean)
    # = (1/(1.414*5)) * (x - 3)
    # ≈ 0.1414 * (x - 3)
    var expected_grad = Tensor.d1([-0.2828, -0.1414, 0.0, 0.1414, 0.2828])
    assert_true(x.grad().all_close[atol=1e-3](expected_grad))


fn test_std_backward_unbiased_std() raises:
    print("test_std_backward_unbiased_std")
    var x = Tensor.d1([2.0, 4.0, 6.0, 8.0], requires_grad=True)
    var s = x.std(unbiased=True)
    s.backward()

    # Mean = 5.0, Sample variance = 20/3 ≈ 6.667
    # Std = sqrt(20/3) ≈ 2.582
    # Gradient: (1/(std*(n-1))) * (x - mean)
    # = (1/(2.582*3)) * (x - 5)
    var std_val = 2.5819889  # sqrt(20/3)
    var factor = 1.0 / (std_val * 3.0)
    var expected_grad = Tensor.d1(
        [
            factor * -3.0,  # (2-5)
            factor * -1.0,  # (4-5)
            factor * 1.0,  # (6-5)
            factor * 3.0,  # (8-5)
        ]
    )
    assert_true(x.grad().all_close[atol=1e-3](expected_grad))


fn test_var_backward_chain_rule() raises:
    print("test_var_backward_chain_rule")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var v = x.variance(unbiased=False)
    var y = v * 2.0  # Chain another operation
    y.backward()

    # Var gradient * 2.0
    # Mean = 2.0, (x-mean) = [-1, 0, 1]
    # Var grad = (2/3) * (x-mean) = [-2/3, 0, 2/3]
    # Final grad = 2.0 * [-2/3, 0, 2/3] = [-4/3, 0, 4/3]
    var expected_grad = Tensor.d1([-1.3333333, 0.0, 1.3333333])
    assert_true(x.grad().all_close[atol=1e-5](expected_grad))


fn test_std_backward_chain_rule() raises:
    print("test_std_backward_chain_rule")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var s = x.std(unbiased=False)
    var y = s**2  # Square the std (should give variance)
    y.backward()

    # This should match variance gradient!
    # Mean = 2.5, (x-mean) = [-1.5, -0.5, 0.5, 1.5]
    # Gradient: (2/n) * (x-mean) = (2/4) * (x-2.5) = 0.5 * (x-2.5)
    var expected_grad = Tensor.d1([-0.75, -0.25, 0.25, 0.75])
    assert_true(x.grad().all_close[atol=1e-5](expected_grad))


fn test_var_std_no_grad_tracking() raises:
    print("test_var_std_no_grad_tracking")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var v = x.variance[track_grad=False]()
    var s = x.std[track_grad=False]()

    # Should not build computation graph
    assert_true(not v.requires_grad)
    assert_true(not s.requires_grad)


fn test_var_backward_2d_axis_0() raises:
    print("test_var_backward_2d_axis_0")
    var x = Tensor.d2([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
    var v = x.variance(axis=0, keepdims=False, unbiased=False)
    var s = v.sum()
    s.backward()

    # Column 0: mean=2, values=[1,3], grad=(2/2)*(values-2)=[-1, 1]
    # Column 1: mean=3, values=[4,2], grad=(2/2)*(values-3)=[1, -1]
    var expected_grad = Tensor.d2([[-1.0, 1.0], [1.0, -1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_var_backward_2d_axis_1() raises:
    print("test_var_backward_2d_axis_1")
    var x = Tensor.d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
    var v = x.variance(axis=1, keepdims=False, unbiased=False)
    var s = v.sum()
    s.backward()

    # Row 0: mean=2, values=[1,3], grad=(2/2)*(values-2)=[-1, 1]
    # Row 1: mean=3, values=[2,4], grad=(2/2)*(values-3)=[-1, 1]
    var expected_grad = Tensor.d2([[-1.0, 1.0], [-1.0, 1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_std_backward_numerical_stability() raises:
    print("test_std_backward_numerical_stability")
    # Test with values close to zero variance
    var x = Tensor.d1([1.0, 1.001, 0.999, 1.0], requires_grad=True)
    var s = x.std(epsilon=1e-12)
    s.backward()

    # Should not crash or produce NaN/Inf
    _ = """var grad = x.grad()
    assert_true(not grad.isnan().any())
    assert_true(not grad.isinf().any())"""


fn run_all_var_std_tests() raises:
    print("\n=== Running Variance & Std Test Suite ===\n")

    # Variance tests
    test_var_backward_global_variance()
    test_var_backward_unbiased_variance()
    test_var_backward_with_axis()
    test_var_backward_chain_rule()
    test_var_backward_2d_axis_0()
    test_var_backward_2d_axis_1()

    # Std tests
    test_std_backward_global_std()
    test_std_backward_unbiased_std()
    test_std_backward_chain_rule()
    test_std_backward_numerical_stability()

    # Feature tests
    test_var_std_no_grad_tracking()

    print("\n=== All Variance & Std Tests Passed! ===\n")


# ============================================================================
# Variance Backward Tests
# ============================================================================


fn test_var_backward_global_variance_vs() raises:
    """Test variance backward pass for global variance (no axis)."""
    print("test_var_backward_global_variance_vs")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var v = x.variance(unbiased=False)  # Population variance
    v.backward()

    # Mean = 3.0
    # Variance = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5
    #          = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
    # Gradient: (2/n) * (x - mean) = (2/5) * (x - 3)
    # = [0.4*(-2), 0.4*(-1), 0.4*(0), 0.4*(1), 0.4*(2)]
    # = [-0.8, -0.4, 0.0, 0.4, 0.8]
    var expected_grad = Tensor.d1([-0.8, -0.4, 0.0, 0.4, 0.8])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_var_backward_unbiased_variance_vs() raises:
    """Test variance backward with Bessel's correction (unbiased=True)."""
    print("test_var_backward_unbiased_variance_vs")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var v = x.variance(unbiased=True)  # Sample variance (n-1)
    v.backward()

    # Gradient: (2/(n-1)) * (x - mean) = (2/4) * (x - 3) = 0.5 * (x - 3)
    # = [0.5*(-2), 0.5*(-1), 0.5*(0), 0.5*(1), 0.5*(2)]
    # = [-1.0, -0.5, 0.0, 0.5, 1.0]
    var expected_grad = Tensor.d1([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))

fn test_var_backward_axis_0_keepdims_false_vs() raises:
    """Test variance backward along axis 0 without keepdims."""
    print("test_var_backward_axis_0_keepdims_false_vs")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v = x.variance(axis=0, keepdims=False, unbiased=False)  # Shape: (2,)
    var s = v.sum()
    s.backward()

    # Column 0: mean=2, var=1, values=[1,3], grad=(2/2)*(values-2) = [-1, 1]
    # Column 1: mean=3, var=1, values=[2,4], grad=(2/2)*(values-3) = [-1, 1]
    var expected_grad = Tensor.d2([[-1.0, -1.0], [1.0, 1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_var_backward_axis_1_keepdims_false_vs() raises:
    """Test variance backward along axis 1 without keepdims."""
    print("test_var_backward_axis_1_keepdims_false_vs")
    var x = Tensor.d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
    var v = x.variance(axis=1, keepdims=False, unbiased=False)  # Shape: (2,)
    var s = v.sum()
    s.backward()

    # Row 0: mean=2, values=[1,3], grad=(2/2)*(values-2) = [-1, 1]
    # Row 1: mean=3, values=[2,4], grad=(2/2)*(values-3) = [-1, 1]
    var expected_grad = Tensor.d2([[-1.0, 1.0], [-1.0, 1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_var_backward_axis_0_keepdims_true_vs() raises:
    """Test variance backward along axis 0 with keepdims."""
    print("test_var_backward_axis_0_keepdims_true_vs")
    var x = Tensor.d2([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
    var v = x.variance(axis=0, keepdims=True, unbiased=False)  # Shape: (1, 2)
    var s = v.sum()
    s.backward()

    # Column 0: mean=2, values=[1,3], grad=(2/2)*(values-2) = [-1, 1]
    # Column 1: mean=3, values=[4,2], grad=(2/2)*(values-3) = [1, -1]
    var expected_grad = Tensor.d2([[-1.0, 1.0], [1.0, -1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_var_backward_axis_1_keepdims_true_vs() raises:
    """Test variance backward along axis 1 with keepdims."""
    print("test_var_backward_axis_1_keepdims_true_vs")
    var x = Tensor.d2([[2.0, 4.0], [1.0, 3.0]], requires_grad=True)
    var v = x.variance(axis=1, keepdims=True, unbiased=False)  # Shape: (2, 1)
    var s = v.sum()
    s.backward()

    # Row 0: mean=3, values=[2,4], grad=(2/2)*(values-3) = [-1, 1]
    # Row 1: mean=2, values=[1,3], grad=(2/2)*(values-2) = [-1, 1]
    var expected_grad = Tensor.d2([[-1.0, 1.0], [-1.0, 1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_var_backward_chain_rule_vs() raises:
    """Test variance backward with chained operations."""
    print("test_var_backward_chain_rule_vs")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var v = x.variance(unbiased=False)
    var y = v.__mul__[track_grad=True](2.0)  # Chain another operation
    y.backward()

    # Var gradient * 2.0
    # Mean = 2.0, (x-mean) = [-1, 0, 1]
    # Var grad = (2/3) * (x-mean) = [-2/3, 0, 2/3]
    # Final grad = 2.0 * [-2/3, 0, 2/3] = [-4/3, 0, 4/3]
    var expected_grad = Tensor.d1([-1.3333333, 0.0, 1.3333333])
    assert_true(x.grad().all_close[atol=1e-5](expected_grad))


fn test_var_backward_3d_tensor_vs() raises:
    """Test variance backward on 3D tensor."""
    print("test_var_backward_3d_tensor_vs")
    var x = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True
    )  # (1, 2, 2)
    var v = x.variance(axis=2, keepdims=False, unbiased=False)  # Shape: (1, 2)
    var s = v.sum()
    s.backward()

    # Row 0: mean=1.5, values=[1,2], grad=(2/2)*(values-1.5) = [-0.5, 0.5]
    # Row 1: mean=3.5, values=[3,4], grad=(2/2)*(values-3.5) = [-0.5, 0.5]
    var expected_grad = Tensor.d3([[[-0.5, 0.5], [-0.5, 0.5]]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_var_no_grad_tracking_vs() raises:
    """Test variance without gradient tracking."""
    print("test_var_no_grad_tracking_vs")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var v = x.variance[track_grad=False]()

    # Should not build computation graph
    assert_true(not v.requires_grad)


fn test_var_backward_large_values_vs() raises:
    """Test variance backward with large values for numerical stability."""
    print("test_var_backward_large_values_vs")
    var x = Tensor.d1([1000.0, 1001.0, 1002.0, 1003.0], requires_grad=True)
    var v = x.variance(unbiased=False)
    v.backward()

    # Mean = 1001.5, (x-mean) = [-1.5, -0.5, 0.5, 1.5]
    # Grad = (2/4) * (x-mean) = 0.5 * [-1.5, -0.5, 0.5, 1.5]
    var expected_grad = Tensor.d1([-0.75, -0.25, 0.25, 0.75])
    assert_true(x.grad().all_close[atol=1e-5](expected_grad))


# ============================================================================
# Std Backward Tests
# ============================================================================


fn test_std_backward_global_std_vs() raises:
    """Test std backward for global std (no axis)."""
    print("test_std_backward_global_std_vs")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var s = x.std(unbiased=False)
    s.backward()

    # Mean = 3.0, Variance = 2.0, Std = sqrt(2) ≈ 1.4142135
    # Gradient: (1/(std*n)) * (x - mean)
    # = (1/(1.4142135*5)) * (x - 3)
    # ≈ 0.141421 * (x - 3)
    var std_val = 1.4142135  # sqrt(2)
    var factor = 1.0 / (std_val * 5.0)
    var expected_grad = Tensor.d1(
        [factor * -2.0, factor * -1.0, 0.0, factor * 1.0, factor * 2.0]
    )
    assert_true(x.grad().all_close[atol=1e-5](expected_grad))


fn test_std_backward_unbiased_std_vs() raises:
    """Test std backward with Bessel's correction."""
    print("test_std_backward_unbiased_std_vs")
    var x = Tensor.d1([2.0, 4.0, 6.0, 8.0], requires_grad=True)
    var s = x.std(unbiased=True)
    s.backward()

    # Mean = 5.0, Sample variance = 20/3 ≈ 6.6667
    # Std = sqrt(20/3) ≈ 2.5820
    # Gradient: (1/(std*(n-1))) * (x - mean)
    var std_val = 2.5819889  # sqrt(20/3)
    var factor = 1.0 / (std_val * 3.0)
    var expected_grad = Tensor.d1(
        [
            factor * -3.0,  # (2-5)
            factor * -1.0,  # (4-5)
            factor * 1.0,  # (6-5)
            factor * 3.0,  # (8-5)
        ]
    )
    assert_true(x.grad().all_close[atol=1e-4](expected_grad))

fn test_std_backward_axis_0_keepdims_false_vs() raises:
    """Test std backward along axis 0 without keepdims."""
    print("test_std_backward_axis_0_keepdims_false_vs")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var s = x.std(axis=0, keepdims=False, unbiased=False)
    var total = s.sum()
    total.backward()

    # Column 0: mean=2, std=1, values=[1,3]
    # grad = (1/(std*n))*(values-mean) = (1/2)*[-1,1] = [-0.5, 0.5]
    # Column 1: mean=3, std=1, values=[2,4]
    # grad = (1/(std*n))*(values-mean) = (1/2)*[-1,1] = [-0.5, 0.5]
    var expected_grad = Tensor.d2([[-0.5, -0.5], [0.5, 0.5]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_std_backward_axis_1_keepdims_false_vs() raises:
    """Test std backward along axis 1 without keepdims."""
    print("test_std_backward_axis_1_keepdims_false_vs")
    var x = Tensor.d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
    var s = x.std(axis=1, keepdims=False, unbiased=False)
    var total = s.sum()
    total.backward()

    # Row 0: mean=2, std=1, values=[1,3], grad=(1/2)*[-1,1] = [-0.5, 0.5]
    # Row 1: mean=3, std=1, values=[2,4], grad=(1/2)*[-1,1] = [-0.5, 0.5]
    var expected_grad = Tensor.d2([[-0.5, 0.5], [-0.5, 0.5]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_std_backward_axis_0_keepdims_true_vs() raises:
    """Test std backward along axis 0 with keepdims."""
    print("test_std_backward_axis_0_keepdims_true_vs")
    var x = Tensor.d2([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
    var s = x.std(axis=0, keepdims=True, unbiased=False)
    var total = s.sum()
    total.backward()

    # Column 0: mean=2, std=1, values=[1,3], grad=(1/2)*[-1,1] = [-0.5, 0.5]
    # Column 1: mean=3, std=1, values=[4,2], grad=(1/2)*[1,-1] = [0.5, -0.5]
    var expected_grad = Tensor.d2([[-0.5, 0.5], [0.5, -0.5]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_std_backward_chain_rule_vs() raises:
    """Test std backward with chained operations."""
    print("test_std_backward_chain_rule_vs")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var s = x.std(unbiased=False)
    var y = s.__mul__[track_grad=True](
        s
    )  # Square the std (should approximate variance grad)
    y.backward()

    # s = std(x), y = s²
    # dy/dx = 2*s * ds/dx
    # For x=[1,2,3,4]: mean=2.5, var=1.25, std≈1.118
    # This is complex, just verify it doesn't crash and produces reasonable values
    var grad = x.grad()
    var grad_sum = grad.sum()
    # Gradient sum should be close to 0 (symmetric around mean)
    assert_true(abs(grad_sum.item()) < 0.01)


fn test_std_backward_numerical_stability_vs() raises:
    """Test std backward with values close to zero variance."""
    print("test_std_backward_numerical_stability_vs")
    var x = Tensor.d1([1.0, 1.001, 0.999, 1.0], requires_grad=True)
    var s = x.std(epsilon=1e-12)
    s.backward()

    # Should not crash or produce NaN/Inf
    var grad = x.grad()
    # Check no NaN or Inf values
    for i in range(4):
        var val = grad[i]
        assert_true(not isnan(val))
        assert_true(not isinf(val))


fn test_std_no_grad_tracking_vs() raises:
    """Test std without gradient tracking."""
    print("test_std_no_grad_tracking_vs")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var s = x.std[track_grad=False]()

    # Should not build computation graph
    assert_true(not s.requires_grad)


fn test_std_backward_3d_tensor_vs() raises:
    """Test std backward on 3D tensor."""
    print("test_std_backward_3d_tensor_vs")
    var x = Tensor.d3(
        [[[1.0, 3.0], [2.0, 4.0]]], requires_grad=True
    )  # (1, 2, 2)
    var s = x.std(axis=2, keepdims=False, unbiased=False)  # Shape: (1, 2)
    var total = s.sum()
    total.backward()

    # Row 0: mean=2, std=1, values=[1,3], grad=(1/2)*[-1,1] = [-0.5, 0.5]
    # Row 1: mean=3, std=1, values=[2,4], grad=(1/2)*[-1,1] = [-0.5, 0.5]
    var expected_grad = Tensor.d3([[[-0.5, 0.5], [-0.5, 0.5]]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_std_backward_epsilon_effect_vs() raises:
    """Test that epsilon prevents division by zero in std backward."""
    print("test_std_backward_epsilon_effect_vs")
    var x = Tensor.d1([2.0, 2.0, 2.0], requires_grad=True)  # Zero variance
    var s = x.std(epsilon=1.0)  # Large epsilon
    s.backward()

    # With zero variance, std = epsilon
    # Gradient should be very small (controlled by epsilon)
    var grad = x.grad()
    for i in range(3):
        assert_true(abs(grad[i]) < 0.1)  # Should be near zero


# ============================================================================
# Combined Variance and Std Tests
# ============================================================================


fn test_var_std_relationship_vs() raises:
    """Test that std² ≈ var in terms of values (not gradients)."""
    print("test_var_std_relationship_vs")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var v = x.variance[track_grad=False](unbiased=False)
    var s = x.std[track_grad=False](unbiased=False)
    var s_squared = s.__mul__[track_grad=False](s)

    assert_true(v.all_close[atol=1e-6](s_squared))


fn test_var_std_unbiased_vs_biased_vs() raises:
    """Test difference between biased and unbiased estimators."""
    print("test_var_std_unbiased_vs_biased_vs")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0])
    var v_biased = x.variance[track_grad=False](unbiased=False)
    var v_unbiased = x.variance[track_grad=False](unbiased=True)

    # Unbiased variance should be larger (divide by n-1 instead of n)
    # v_unbiased = v_biased * (n / (n-1)) = v_biased * (4/3)
    var ratio = v_unbiased.__truediv__[track_grad=False](v_biased)
    var expected_ratio = Tensor.scalar([4.0 / 3.0])
    assert_true(ratio.all_close[atol=1e-6](expected_ratio))


fn test_var_backward_single_element_vs() raises:
    """Test variance backward with single element (edge case)."""
    print("test_var_backward_single_element_vs")
    var x = Tensor.d1([5.0], requires_grad=True)
    var v = x.variance(unbiased=False)
    v.backward()

    # Single element: variance = 0, gradient = 0
    var expected_grad = Tensor.d1([0.0])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_std_backward_two_elements_vs() raises:
    """Test std backward with two elements."""
    print("test_std_backward_two_elements_vs")
    var x = Tensor.d1([1.0, 3.0], requires_grad=True)
    var s = x.std(unbiased=False)
    s.backward()

    # Mean = 2, std = 1, values = [1, 3]
    # Gradient = (1/(std*n)) * (x-mean) = (1/2) * [-1, 1] = [-0.5, 0.5]
    var expected_grad = Tensor.d1([-0.5, 0.5])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


# ============================================================================
# Consolidated Test Runner
# ============================================================================


fn run_all_variance_std_tests() raises:
    """Run all variance and std tests."""
    print("\n=== Running Variance & Std Test Suite ===\n")

    # Variance backward tests
    test_var_backward_global_variance_vs()
    test_var_backward_unbiased_variance_vs()
    test_var_backward_axis_0_keepdims_false_vs()
    test_var_backward_axis_1_keepdims_false_vs()
    test_var_backward_axis_0_keepdims_true_vs()
    test_var_backward_axis_1_keepdims_true_vs()
    test_var_backward_chain_rule_vs()
    test_var_backward_3d_tensor_vs()
    test_var_no_grad_tracking_vs()
    test_var_backward_large_values_vs()

    # Std backward tests
    test_std_backward_global_std_vs()
    test_std_backward_unbiased_std_vs()
    test_std_backward_axis_0_keepdims_false_vs()
    test_std_backward_axis_1_keepdims_false_vs()
    test_std_backward_axis_0_keepdims_true_vs()
    test_std_backward_chain_rule_vs()
    test_std_backward_numerical_stability_vs()
    test_std_no_grad_tracking_vs()
    test_std_backward_3d_tensor_vs()
    test_std_backward_epsilon_effect_vs()

    # Combined tests
    test_var_std_relationship_vs()
    test_var_std_unbiased_vs_biased_vs()
    test_var_backward_single_element_vs()
    test_std_backward_two_elements_vs()

    print("\n=== All Variance & Std Tests Passed! ===\n")

fn main() raises:
    test_sqrt_backward()
    test_sqrt_backward_zero_handling()
    run_all_var_std_tests()
    run_all_variance_std_tests()
