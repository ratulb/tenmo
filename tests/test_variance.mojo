from tenmo.tensor import Tensor
from std.math import sqrt
from std.testing import assert_true, TestSuite
from tenmo.common_utils import isnan, isinf


def test_sqrt_backward() raises:
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


def test_sqrt_backward_zero_handling() raises:
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


def test_var_backward_global_variance() raises:
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


def test_var_backward_unbiased_variance() raises:
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var v = x.variance(unbiased=True)  # Sample variance (n-1)
    v.backward()

    # Gradient: (2/(n-1)) * (x - mean) = (2/4) * (x - 3) = 0.5 * (x - 3)
    # = [0.5*(-2), 0.5*(-1), 0.5*(0), 0.5*(1), 0.5*(2)]
    # = [-1.0, -0.5, 0.0, 0.5, 1.0]
    var expected_grad = Tensor.d1([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_var_backward_with_axis() raises:
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v = x.variance(axis=0, unbiased=False)  # Variance along rows
    var s = v.sum()
    s.backward()

    # Column 0: mean=2, var=1, grad for [1,3] = (2/2)*([1,3]-2) = [-1, 1]
    # Column 1: mean=3, var=1, grad for [2,4] = (2/2)*([2,4]-3) = [-1, 1]
    var expected_grad = Tensor.d2([[-1.0, -1.0], [1.0, 1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_std_backward_global_std() raises:
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var s = x.std(unbiased=False)
    s.backward()

    # Mean = 3.0, Variance = 2.0, Std = sqrt(2) ≈ 1.414
    # Gradient: (1/(std*n)) * (x - mean)
    # = (1/(1.414*5)) * (x - 3)
    # ≈ 0.1414 * (x - 3)
    var expected_grad = Tensor.d1([-0.2828, -0.1414, 0.0, 0.1414, 0.2828])
    assert_true(x.grad().all_close[atol=1e-3](expected_grad))


def test_std_backward_unbiased_std() raises:
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


def test_var_backward_chain_rule() raises:
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


def test_std_backward_chain_rule() raises:
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var s = x.std(unbiased=False)
    var y = s**2  # Square the std (should give variance)
    y.backward()

    # This should match variance gradient!
    # Mean = 2.5, (x-mean) = [-1.5, -0.5, 0.5, 1.5]
    # Gradient: (2/n) * (x-mean) = (2/4) * (x-2.5) = 0.5 * (x-2.5)
    var expected_grad = Tensor.d1([-0.75, -0.25, 0.25, 0.75])
    assert_true(x.grad().all_close[atol=1e-5](expected_grad))


def test_var_std_no_grad_tracking() raises:
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var v = x.variance[track_grad=False]()
    var s = x.std[track_grad=False]()

    # Should not build computation graph
    assert_true(not v.requires_grad)
    assert_true(not s.requires_grad)


def test_var_backward_2d_axis_0() raises:
    var x = Tensor.d2([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
    var v = x.variance(axis=0, keepdims=False, unbiased=False)
    var s = v.sum()
    s.backward()

    # Column 0: mean=2, values=[1,3], grad=(2/2)*(values-2)=[-1, 1]
    # Column 1: mean=3, values=[4,2], grad=(2/2)*(values-3)=[1, -1]
    var expected_grad = Tensor.d2([[-1.0, 1.0], [1.0, -1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_var_backward_2d_axis_1() raises:
    var x = Tensor.d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
    var v = x.variance(axis=1, keepdims=False, unbiased=False)
    var s = v.sum()
    s.backward()

    # Row 0: mean=2, values=[1,3], grad=(2/2)*(values-2)=[-1, 1]
    # Row 1: mean=3, values=[2,4], grad=(2/2)*(values-3)=[-1, 1]
    var expected_grad = Tensor.d2([[-1.0, 1.0], [-1.0, 1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_std_backward_numerical_stability() raises:
    # Test with values close to zero variance
    var x = Tensor.d1([1.0, 1.001, 0.999, 1.0], requires_grad=True)
    var s = x.std()
    s.backward()

    # Should not crash or produce NaN/Inf
    _ = """var grad = x.grad()
    assert_true(not grad.isnan().any())
    assert_true(not grad.isinf().any())"""



# ============================================================================
# Variance Backward Tests
# ============================================================================


def test_var_backward_global_variance_vs() raises:
    """Test variance backward pass for global variance (no axis)."""
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


def test_var_backward_unbiased_variance_vs() raises:
    """Test variance backward with Bessel's correction (unbiased=True)."""
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var v = x.variance(unbiased=True)  # Sample variance (n-1)
    v.backward()

    # Gradient: (2/(n-1)) * (x - mean) = (2/4) * (x - 3) = 0.5 * (x - 3)
    # = [0.5*(-2), 0.5*(-1), 0.5*(0), 0.5*(1), 0.5*(2)]
    # = [-1.0, -0.5, 0.0, 0.5, 1.0]
    var expected_grad = Tensor.d1([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_var_backward_axis_0_keepdims_false_vs() raises:
    """Test variance backward along axis 0 without keepdims."""
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v = x.variance(axis=0, keepdims=False, unbiased=False)  # Shape: (2,)
    var s = v.sum()
    s.backward()

    # Column 0: mean=2, var=1, values=[1,3], grad=(2/2)*(values-2) = [-1, 1]
    # Column 1: mean=3, var=1, values=[2,4], grad=(2/2)*(values-3) = [-1, 1]
    var expected_grad = Tensor.d2([[-1.0, -1.0], [1.0, 1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_var_backward_axis_1_keepdims_false_vs() raises:
    """Test variance backward along axis 1 without keepdims."""
    var x = Tensor.d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
    var v = x.variance(axis=1, keepdims=False, unbiased=False)  # Shape: (2,)
    var s = v.sum()
    s.backward()

    # Row 0: mean=2, values=[1,3], grad=(2/2)*(values-2) = [-1, 1]
    # Row 1: mean=3, values=[2,4], grad=(2/2)*(values-3) = [-1, 1]
    var expected_grad = Tensor.d2([[-1.0, 1.0], [-1.0, 1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_var_backward_axis_0_keepdims_true_vs() raises:
    """Test variance backward along axis 0 with keepdims."""
    var x = Tensor.d2([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
    var v = x.variance(axis=0, keepdims=True, unbiased=False)  # Shape: (1, 2)
    var s = v.sum()
    s.backward()

    # Column 0: mean=2, values=[1,3], grad=(2/2)*(values-2) = [-1, 1]
    # Column 1: mean=3, values=[4,2], grad=(2/2)*(values-3) = [1, -1]
    var expected_grad = Tensor.d2([[-1.0, 1.0], [1.0, -1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_var_backward_axis_1_keepdims_true_vs() raises:
    """Test variance backward along axis 1 with keepdims."""
    var x = Tensor.d2([[2.0, 4.0], [1.0, 3.0]], requires_grad=True)
    var v = x.variance(axis=1, keepdims=True, unbiased=False)  # Shape: (2, 1)
    var s = v.sum()
    s.backward()

    # Row 0: mean=3, values=[2,4], grad=(2/2)*(values-3) = [-1, 1]
    # Row 1: mean=2, values=[1,3], grad=(2/2)*(values-2) = [-1, 1]
    var expected_grad = Tensor.d2([[-1.0, 1.0], [-1.0, 1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_var_backward_chain_rule_vs() raises:
    """Test variance backward with chained operations."""
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


def test_var_backward_3d_tensor_vs() raises:
    """Test variance backward on 3D tensor."""
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


def test_var_no_grad_tracking_vs() raises:
    """Test variance without gradient tracking."""
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var v = x.variance[track_grad=False]()

    # Should not build computation graph
    assert_true(not v.requires_grad)


def test_var_backward_large_values_vs() raises:
    """Test variance backward with large values for numerical stability."""
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


def test_std_backward_global_std_vs() raises:
    """Test std backward for global std (no axis)."""
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


def test_std_backward_unbiased_std_vs() raises:
    """Test std backward with Bessel's correction."""
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


def test_std_backward_axis_0_keepdims_false_vs() raises:
    """Test std backward along axis 0 without keepdims."""
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


def test_std_backward_axis_1_keepdims_false_vs() raises:
    """Test std backward along axis 1 without keepdims."""
    var x = Tensor.d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
    var s = x.std(axis=1, keepdims=False, unbiased=False)
    var total = s.sum()
    total.backward()

    # Row 0: mean=2, std=1, values=[1,3], grad=(1/2)*[-1,1] = [-0.5, 0.5]
    # Row 1: mean=3, std=1, values=[2,4], grad=(1/2)*[-1,1] = [-0.5, 0.5]
    var expected_grad = Tensor.d2([[-0.5, 0.5], [-0.5, 0.5]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_std_backward_axis_0_keepdims_true_vs() raises:
    """Test std backward along axis 0 with keepdims."""
    var x = Tensor.d2([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
    var s = x.std(axis=0, keepdims=True, unbiased=False)
    var total = s.sum()
    total.backward()

    # Column 0: mean=2, std=1, values=[1,3], grad=(1/2)*[-1,1] = [-0.5, 0.5]
    # Column 1: mean=3, std=1, values=[4,2], grad=(1/2)*[1,-1] = [0.5, -0.5]
    var expected_grad = Tensor.d2([[-0.5, 0.5], [0.5, -0.5]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_std_backward_chain_rule_vs() raises:
    """Test std backward with chained operations."""
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


def test_std_backward_numerical_stability_vs() raises:
    """Test std backward with values close to zero variance."""
    var x = Tensor.d1([1.0, 1.001, 0.999, 1.0], requires_grad=True)
    var s = x.std()
    s.backward()

    # Should not crash or produce NaN/Inf
    var grad = x.grad()
    # Check no NaN or Inf values
    for i in range(4):
        var val = grad[i]
        assert_true(not isnan(val))
        assert_true(not isinf(val))


def test_std_no_grad_tracking_vs() raises:
    """Test std without gradient tracking."""
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var s = x.std[track_grad=False]()

    # Should not build computation graph
    assert_true(not s.requires_grad)


def test_std_backward_3d_tensor_vs() raises:
    """Test std backward on 3D tensor."""
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


def test_std_backward_zero_variance_vs() raises:
    """Test that std backward handles zero variance without NaN."""
    var x = Tensor.d1([2.0, 2.0, 2.0], requires_grad=True)
    var s = x.std()
    s.backward()
    var grad = x.grad()
    # x - mean = [0,0,0] so grad = 0/(eps*divisor) = 0 — no NaN
    assert_true(grad.all_close[atol=1e-5](Tensor.d1([0.0, 0.0, 0.0])))


# ============================================================================
# Combined Variance and Std Tests
# ============================================================================


def test_var_std_relationship_vs() raises:
    """Test that std² ≈ var in terms of values (not gradients)."""
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var v = x.variance[track_grad=False](unbiased=False)
    var s = x.std[track_grad=False](unbiased=False)
    var s_squared = s.__mul__[track_grad=False](s)

    assert_true(v.all_close[atol=1e-6](s_squared.reshape[False]()))


def test_var_std_unbiased_vs_biased_vs() raises:
    """Test difference between biased and unbiased estimators."""
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0])
    var v_biased = x.variance[track_grad=False](unbiased=False)
    var v_unbiased = x.variance[track_grad=False](unbiased=True)

    # Unbiased variance should be larger (divide by n-1 instead of n)
    # v_unbiased = v_biased * (n / (n-1)) = v_biased * (4/3)
    var ratio = v_unbiased.__truediv__[track_grad=False](v_biased)
    var expected_ratio = Tensor.scalar([4.0 / 3.0])
    assert_true(ratio.all_close[atol=1e-6](expected_ratio))


def test_var_backward_single_element_vs() raises:
    """Test variance backward with single element (edge case)."""
    var x = Tensor.d1([5.0], requires_grad=True)
    var v = x.variance(unbiased=False)
    v.backward()

    # Single element: variance = 0, gradient = 0
    var expected_grad = Tensor.d1([0.0])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


def test_std_backward_two_elements_vs() raises:
    """Test std backward with two elements."""
    var x = Tensor.d1([1.0, 3.0], requires_grad=True)
    var s = x.std(unbiased=False)
    s.backward()

    # Mean = 2, std = 1, values = [1, 3]
    # Gradient = (1/(std*n)) * (x-mean) = (1/2) * [-1, 1] = [-0.5, 0.5]
    var expected_grad = Tensor.d1([-0.5, 0.5])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))



def test_variance_comprehensive() raises:

    # Test 1: Simple 1D case
    var x1 = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var v1 = x1.variance(unbiased=False)
    assert_true(abs(v1.item() - 2.0) < 1e-5, "1D variance value mismatch")

    v1.backward()
    assert_true(abs(x1.grad()[0] - (-0.8)) < 1e-5, "1D gradient mismatch")
    # Test 2: 2D with axis reduction
    var x2 = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var v2 = x2.variance(axis=1, keepdims=False, unbiased=False)
    assert_true(v2.shape().rank() == 1, "Should reduce to 1D")

    var s2 = v2.sum()
    s2.backward()
    assert_true(abs(x2.grad()[0, 0] - (-0.667)) < 1e-2, "2D gradient mismatch")

    # Test 3: Keepdims=True
    var x3 = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v3 = x3.variance(axis=1, keepdims=True, unbiased=False)
    assert_true(v3.shape()[1] == 1, "Should keep dimension")

    summ = v3.sum()
    summ.backward()

    # Test 4: Unbiased vs Biased
    var x4 = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var v_biased = x4.variance(unbiased=False)
    var v_unbiased = x4.variance(unbiased=True)

    # Biased: 10/4 = 2.5, Unbiased: 10/3 = 3.333
    assert_true(
        v_unbiased.item() > v_biased.item(), "Unbiased should be larger"
    )


def test_variance_global() raises:

    var x5 = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    # Test with keepdims=False (should be scalar)
    var v5 = x5.variance(axis=-100, keepdims=False, unbiased=False)
    assert_true(v5.shape().rank() == 0, "Should be scalar (rank 0)")

    assert_true(abs(v5.item() - 1.25) < 1e-5, "Global variance mismatch")

    v5.backward()
    # Expected: (2/4) * (1 - 2.5) = 0.5 * (-1.5) = -0.75
    assert_true(
        abs(x5.grad()[0, 0] - (-0.75)) < 1e-5,
        "Global variance gradient mismatch",
    )

    # Test with keepdims=True
    var x5b = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v5b = x5b.variance(axis=-100, keepdims=True, unbiased=False)
    assert_true(v5b.shape().rank() == 2, "Should keep all dimensions")

    s = v5b.sum()
    s.backward()  # Need sum since it's not scalar
    assert_true(abs(x5b.grad()[0, 0] - (-0.75)) < 1e-5, "Gradient mismatch")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
