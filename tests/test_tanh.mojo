from tenmo.tensor import Tensor
from tenmo.net import Tanh, Linear
from tenmo.common_utils import isnan, isinf
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from std.math import tanh as scalar_tanh, abs as scalar_abs

comptime dtype = DType.float32
comptime tol = Float32(1e-4)


# ============================================================================
# Tanh Activation Tests
# ============================================================================


def test_tanh_forward_values() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([0.0, 1.0, -1.0, 2.0, -2.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    # tanh(0) = 0, tanh(1) ≈ 0.762, tanh(-1) ≈ -0.762
    # tanh(2) ≈ 0.964, tanh(-2) ≈ -0.964
    assert_true(abs(y[IntArray(0)]) < 1e-6, "tanh(0) should be 0")
    assert_true(abs(y[IntArray(1)] - 0.7616) < 0.001, "tanh(1) ≈ 0.762")
    assert_true(abs(y[IntArray(2)] + 0.7616) < 0.001, "tanh(-1) ≈ -0.762")
    assert_true(abs(y[IntArray(3)] - 0.9640) < 0.001, "tanh(2) ≈ 0.964")
    assert_true(abs(y[IntArray(4)] + 0.9640) < 0.001, "tanh(-2) ≈ -0.964")


def test_tanh_moderate_range() raises:
    comptime dtype = DType.float32
    # For |x| < 5, tanh should be strictly within (-1, 1)
    var x = Tensor[dtype].d1(
        [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], requires_grad=False
    )
    var y = x.tanh[track_grad=False]()

    for i in range(7):
        var val = y[IntArray(i)]
        assert_true(
            val > -1.0 and val < 1.0, "tanh should be strictly in (-1, 1)"
        )


def test_tanh_range_bounded() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([-10.0, -5.0, 0.0, 5.0, 10.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    # tanh output should be in [-1, 1] (closed interval due to FP precision)
    # For extreme values, tanh saturates to exactly ±1.0 in float32
    for i in range(5):
        var val = y[IntArray(i)]
        assert_true(
            val >= -1.0 and val <= 1.0, "tanh output should be in [-1, 1]"
        )
        assert_true(not isnan(val), "tanh should not produce NaN")
        assert_true(not isinf(val), "tanh should not produce Inf")

    # Test 1: Extreme values (will saturate to ±1.0)
    var x_extreme = Tensor[dtype].d1([-10.0, 10.0], requires_grad=False)
    var y_extreme = x_extreme.tanh[track_grad=False]()
    assert_true(
        abs(y_extreme[IntArray(0)] + 1.0) < 1e-6, "tanh(-10) should be ≈ -1.0"
    )
    assert_true(
        abs(y_extreme[IntArray(1)] - 1.0) < 1e-6, "tanh(10) should be ≈ 1.0"
    )

    # Test 2: Moderate values (strictly within (-1, 1))
    var x_moderate = Tensor[dtype].d1(
        [-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=False
    )
    var y_moderate = x_moderate.tanh[track_grad=False]()
    for i in range(5):
        var val = y_moderate[IntArray(i)]
        assert_true(
            val > -1.0 and val < 1.0,
            "tanh should be strictly in (-1, 1) for moderate inputs",
        )
        assert_true(not isnan(val), "tanh should not produce NaN")
        assert_true(not isinf(val), "tanh should not produce Inf")


def test_tanh_saturation() raises:
    comptime dtype = DType.float32
    # For large |x|, tanh saturates to ±1.0 due to float32 precision
    var x = Tensor[dtype].d1([-10.0, -20.0, 10.0, 20.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    # Check negative saturation
    assert_true(
        abs(y[IntArray(0)] + 1.0) < 1e-5, "tanh should saturate to -1.0"
    )
    assert_true(
        abs(y[IntArray(1)] + 1.0) < 1e-5, "tanh should saturate to -1.0"
    )

    # Check positive saturation
    assert_true(abs(y[IntArray(2)] - 1.0) < 1e-5, "tanh should saturate to 1.0")
    assert_true(abs(y[IntArray(3)] - 1.0) < 1e-5, "tanh should saturate to 1.0")


def test_tanh_no_overflow() raises:
    comptime dtype = DType.float32
    # Test that extreme values don't cause NaN or Inf
    var x = Tensor[dtype].d1([-100.0, -50.0, 50.0, 100.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    for i in range(4):
        var val = y[IntArray(i)]
        assert_true(not isnan(val), "tanh should not produce NaN")
        assert_true(not isinf(val), "tanh should not produce Inf")
        assert_true(abs(val) <= 1.0, "tanh magnitude should not exceed 1.0")


def test_tanh_gradient_at_zero() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([0.0], requires_grad=True)
    var y = x.tanh[track_grad=True]()
    var loss = y.sum()
    loss.backward()

    # d/dx tanh(x) = 1 - tanh²(x)
    # At x=0: tanh(0) = 0, so gradient = 1 - 0² = 1
    var expected_grad = Tensor[dtype].d1([1.0])
    assert_true(
        x.grad().all_close[atol=1e-5](expected_grad),
        "tanh gradient at 0 should be 1",
    )


def test_tanh_gradient_positive() raises:

    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0], requires_grad=True)
    var y = x.tanh[track_grad=True]()
    var loss = y.sum()
    loss.backward()

    # At x=1: tanh(1) ≈ 0.7616
    # gradient = 1 - 0.7616² ≈ 0.42
    var grad = x.grad()[IntArray(0)]
    assert_true(
        abs(grad - 0.42) < 0.01, "tanh gradient at x=1 should be ≈ 0.42"
    )


def test_tanh_gradient_negative() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([-1.0], requires_grad=True)
    var y = x.tanh[track_grad=True]()
    var loss = y.sum()
    loss.backward()

    # tanh is symmetric, gradient should be same as at x=1
    var grad = x.grad()[IntArray(0)]
    assert_true(
        abs(grad - 0.42) < 0.01, "tanh gradient at x=-1 should be ≈ 0.42"
    )


def test_tanh_gradient_saturation() raises:
    comptime dtype = DType.float32
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


def test_tanh_gradient_batch() raises:
    comptime dtype = DType.float32
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
            assert_true(
                grad > 0.0 and grad <= 1.0, "tanh gradient should be in (0, 1]"
            )


def test_tanh_chain_rule() raises:
    comptime dtype = DType.float32
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


def test_tanh_no_grad_mode() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var y = x.tanh[track_grad=False]()

    assert_true(not y.has_ancestry(), "Should not build graph in no_grad mode")
    assert_true(not y.requires_grad, "Output should not require grad")


def test_tanh_layer_train_mode() raises:
    comptime dtype = DType.float32
    var activation = Tanh[dtype]()
    activation.train()

    var x = Tensor[dtype].d1([1.0], requires_grad=True)
    var y = activation(x)
    var loss = y.sum()
    loss.backward()

    assert_true(x.has_grad(), "Should have gradients in train mode")


def test_tanh_layer_eval_mode() raises:
    comptime dtype = DType.float32
    var activation = Tanh[dtype]()
    activation.eval()

    var x = Tensor[dtype].d1([1.0], requires_grad=True)
    var y = activation(x)

    assert_true(not y.has_ancestry(), "Should not build graph in eval mode")


def test_tanh_numerical_stability_positive() raises:
    comptime dtype = DType.float32
    # Test large positive values
    var x = Tensor[dtype].d1([10.0, 20.0, 50.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    # Should approach 1, not overflow
    for i in range(3):
        var val = y[IntArray(i)]
        assert_true(not isnan(val), "Should not be NaN")
        assert_true(not isinf(val), "Should not be Inf")
        assert_true(val > 0.999 and val <= 1.0, "Should be close to 1")


def test_tanh_numerical_stability_negative() raises:
    comptime dtype = DType.float32
    # Test large negative values
    var x = Tensor[dtype].d1([-10.0, -20.0, -50.0], requires_grad=False)
    var y = x.tanh[track_grad=False]()

    # Should approach -1, not overflow
    for i in range(3):
        var val = y[IntArray(i)]
        assert_true(not isnan(val), "Should not be NaN")
        assert_true(not isinf(val), "Should not be Inf")
        assert_true(val <= -0.999 and val >= -1.0, "Should be close to -1")


def test_tanh_contiguous_vs_non_contiguous() raises:

    comptime dtype = DType.float32
    var x_contig = Tensor[dtype].d1([0.0, 1.0, -1.0], requires_grad=True)
    var y_contig = x_contig.tanh[track_grad=True]()

    # Create non-contiguous tensor (e.g., via slice or transpose)
    var x_large = Tensor[dtype].d2([[0.0, 99.0], [1.0, 99.0], [-1.0, 99.0]])
    var x_non_contig = x_large[:, slice(0, 1)]  # Slice to get non-contiguous
    x_non_contig.requires_grad_(True)
    var y_non_contig = x_non_contig.tanh[track_grad=True]()

    # Both should give same results
    assert_true(
        y_contig.unsqueeze[track_grad=False](-1).all_close[atol=1e-5](
            y_non_contig
        ),
        "Contiguous and non-contiguous should match",
    )


def test_tanh_with_linear_layer() raises:
    comptime dtype = DType.float32
    var layer = Linear[dtype](2, 3, init_method="xavier")
    var activation = Tanh[dtype]()
    layer.train()
    activation.train()

    var x = Tensor[dtype].d2([[1.0, 2.0]])
    var linear_out = layer(x)
    var y = activation(linear_out)
    var loss = y.sum()
    loss.backward()

    assert_true(layer.weight.has_grad(), "Linear weight should have gradient")
    assert_true(layer.bias.value().has_grad(), "Linear bias should have gradient")


# ============================================================================
# Edge Cases
# ============================================================================


def test_tanh_zero_tensor() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].zeros(Shape(3, 3), requires_grad=True)
    var y = x.tanh[track_grad=True]()
    var loss = y.sum()
    loss.backward()

    # tanh(0) = 0, gradient = 1
    var expected_output = Tensor[dtype].zeros(Shape(3, 3))
    var expected_grad = Tensor[dtype].ones(Shape(3, 3))
    assert_true(y.all_close[atol=1e-6](expected_output), "tanh(0) should be 0")
    assert_true(
        x.grad().all_close[atol=1e-5](expected_grad), "gradient should be 1"
    )


def test_tanh_symmetry() raises:
    comptime dtype = DType.float32
    var x_pos = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=False)
    var x_neg = Tensor[dtype].d1([-1.0, -2.0, -3.0], requires_grad=False)

    var y_pos = x_pos.tanh[track_grad=False]()
    var y_neg = x_neg.tanh[track_grad=False]()

    # tanh(-x) = -tanh(x)
    for i in range(3):
        var pos_val = y_pos[IntArray(i)]
        var neg_val = y_neg[IntArray(i)]
        assert_true(abs(pos_val + neg_val) < 1e-5, "tanh should be symmetric")



# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def tanh_close(a: Tensor[dtype], b: Tensor[dtype]) raises -> Bool:
    return a.all_close[atol=tol](b)


def tanh_expected_1d() -> Tensor[dtype]:
    """T[t]anh([0, 0.5, -0.5, 1, -1])."""
    return Tensor[dtype].d1(
        [
            scalar_tanh(Float32(0.0)),
            scalar_tanh(Float32(0.5)),
            scalar_tanh(Float32(-0.5)),
            scalar_tanh(Float32(1.0)),
            scalar_tanh(Float32(-1.0)),
        ]
    )


def tanh_grad_expected_1d() -> Tensor[dtype]:
    """1 - tanh^2([0, 0.5, -0.5, 1, -1])."""
    return Tensor[dtype].d1(
        [
            Float32(1) - scalar_tanh(Float32(0.0)) ** 2,
            Float32(1) - scalar_tanh(Float32(0.5)) ** 2,
            Float32(1) - scalar_tanh(Float32(-0.5)) ** 2,
            Float32(1) - scalar_tanh(Float32(1.0)) ** 2,
            Float32(1) - scalar_tanh(Float32(-1.0)) ** 2,
        ]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD — CPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_tanh_cpu_fwd_scalar_zero() raises:
    var t = Tensor[dtype].scalar(0.0)
    var out = t.tanh()
    assert_true(tanh_close(out, Tensor[dtype].scalar(0.0)))


def test_tanh_cpu_fwd_scalar_one() raises:
    var t = Tensor[dtype].scalar(1.0)
    var out = t.tanh()
    assert_true(
        tanh_close(out, Tensor[dtype].scalar(scalar_tanh(Float32(1.0))))
    )


def test_tanh_cpu_fwd_scalar_neg() raises:
    var t = Tensor[dtype].scalar(-1.0)
    var out = t.tanh()
    assert_true(
        tanh_close(out, Tensor[dtype].scalar(scalar_tanh(Float32(-1.0))))
    )


def test_tanh_cpu_fwd_1d_known() raises:
    var t = Tensor[dtype].d1([0.0, 0.5, -0.5, 1.0, -1.0])
    var out = t.tanh()
    assert_true(tanh_close(out, tanh_expected_1d()))


def test_tanh_cpu_fwd_1d_zeros() raises:
    var t = Tensor[dtype].zeros(Shape(8))
    var out = t.tanh()
    assert_true(tanh_close(out, Tensor[dtype].zeros(Shape(8))))


def test_tanh_cpu_fwd_2d_zeros() raises:
    var t = Tensor[dtype].zeros(Shape(3, 4))
    var out = t.tanh()
    assert_true(out.shape() == Shape(3, 4))
    assert_true(tanh_close(out, Tensor[dtype].zeros(Shape(3, 4))))


def test_tanh_cpu_fwd_2d_known() raises:
    var t = Tensor[dtype].d2([[0.0, 1.0], [-1.0, 0.5]])
    var out = t.tanh()
    var expected = Tensor[dtype].d2(
        [
            [scalar_tanh(Float32(0.0)), scalar_tanh(Float32(1.0))],
            [scalar_tanh(Float32(-1.0)), scalar_tanh(Float32(0.5))],
        ]
    )
    assert_true(tanh_close(out, expected))


def test_tanh_cpu_fwd_3d() raises:
    var t = Tensor[dtype].zeros(Shape(2, 3, 4))
    var out = t.tanh()
    assert_true(out.shape() == Shape(2, 3, 4))
    assert_true(tanh_close(out, Tensor[dtype].zeros(Shape(2, 3, 4))))


def test_tanh_cpu_fwd_4d() raises:
    var t = Tensor[dtype].zeros(Shape(2, 3, 4, 5))
    var out = t.tanh()
    assert_true(out.shape() == Shape(2, 3, 4, 5))
    assert_true(tanh_close(out, Tensor[dtype].zeros(Shape(2, 3, 4, 5))))


def test_tanh_cpu_fwd_range_clamping() raises:

    # tanh output must be in (-1, 1)
    var t = Tensor[dtype].d1([-10.0, -1.0, 0.0, 1.0, 10.0])
    var out = t.tanh()
    var data = out.data_ptr()
    for i in range(5):
        assert_true(data[i] >= Float32(-1.0) and data[i] <= Float32(1.0))


def test_tanh_cpu_fwd_large() raises:
    var t = Tensor[dtype].zeros(Shape(64, 128))
    var out = t.tanh()
    assert_true(out.shape() == Shape(64, 128))
    assert_true(tanh_close(out, Tensor[dtype].zeros(Shape(64, 128))))


def test_tanh_cpu_fwd_non_contiguous() raises:
    var t = Tensor[dtype].d2([[0.0, 1.0], [-1.0, 0.5]])
    var t_T = t.transpose()
    var out = t_T.tanh()
    var expected = Tensor[dtype].d2(
        [
            [scalar_tanh(Float32(0.0)), scalar_tanh(Float32(-1.0))],
            [scalar_tanh(Float32(1.0)), scalar_tanh(Float32(0.5))],
        ]
    )
    assert_true(tanh_close(out, expected))


def test_tanh_cpu_fwd_no_requires_grad() raises:
    var t = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var out = t.tanh[track_grad=False]()
    assert_true(not out.requires_grad)
    assert_true(not out.has_ancestry())


def test_tanh_cpu_fwd_requires_grad_propagates() raises:
    var t = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var out = t.tanh()
    assert_true(out.requires_grad)
    assert_true(out.has_ancestry())
    assert_true(out.has_ancestry())


def test_tanh_cpu_fwd_no_ancestry_without_grad() raises:
    var t = Tensor[dtype].d1([1.0, 2.0], requires_grad=False)
    var out = t.tanh()
    assert_true(not out.requires_grad)
    assert_true(not out.has_ancestry())


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD — CPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_tanh_cpu_bwd_scalar_zero() raises:
    var t = Tensor[dtype].scalar(0.0, requires_grad=True)
    var out = t.tanh()
    var loss = out.sum()
    loss.backward()
    # grad = 1 - tanh(0)^2 = 1
    assert_true(tanh_close(t.grad().as_tensor(), Tensor[dtype].scalar(1.0)))


def test_tanh_cpu_bwd_scalar_one() raises:
    var t = Tensor[dtype].scalar(1.0, requires_grad=True)
    var out = t.tanh()
    var loss = out.sum()
    loss.backward()
    var expected = Tensor[dtype].scalar(
        Float32(1) - scalar_tanh(Float32(1.0)) ** 2
    )
    assert_true(tanh_close(t.grad().as_tensor(), expected))


def test_tanh_cpu_bwd_1d_zeros() raises:
    var t = Tensor[dtype].zeros(Shape(4), requires_grad=True)
    var out = t.tanh()
    var loss = out.sum()
    loss.backward()
    # grad = 1 - tanh(0)^2 = 1 everywhere
    assert_true(tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(4))))


def test_tanh_cpu_bwd_1d_known() raises:
    var t = Tensor[dtype].d1([0.0, 0.5, -0.5, 1.0, -1.0], requires_grad=True)
    var out = t.tanh()
    var loss = out.sum()
    loss.backward()
    assert_true(tanh_close(t.grad().as_tensor(), tanh_grad_expected_1d()))


def test_tanh_cpu_bwd_2d_zeros() raises:
    var t = Tensor[dtype].zeros(Shape(3, 4), requires_grad=True)
    var out = t.tanh()
    var loss = out.sum()
    loss.backward()
    assert_true(
        tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(3, 4)))
    )


def test_tanh_cpu_bwd_3d() raises:
    var t = Tensor[dtype].zeros(Shape(2, 3, 4), requires_grad=True)
    var out = t.tanh()
    var loss = out.sum()
    loss.backward()
    assert_true(
        tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(2, 3, 4)))
    )


def test_tanh_cpu_bwd_4d() raises:
    var t = Tensor[dtype].zeros(Shape(2, 3, 4, 5), requires_grad=True)
    var out = t.tanh()
    var loss = out.sum()
    loss.backward()
    assert_true(
        tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(2, 3, 4, 5)))
    )


def test_tanh_cpu_bwd_chain_mul() raises:
    # y = tanh(2*x), dy/dx = 2 * (1 - tanh(2x)^2)
    var t = Tensor[dtype].d1([0.0, 1.0], requires_grad=True)
    var t2 = t * Scalar[dtype](2)
    var out = t2.tanh()
    var loss = out.sum()
    loss.backward()
    var expected = Tensor[dtype].d1(
        [
            Float32(2) * (Float32(1) - scalar_tanh(Float32(0.0)) ** 2),
            Float32(2) * (Float32(1) - scalar_tanh(Float32(2.0)) ** 2),
        ]
    )
    assert_true(tanh_close(t.grad().as_tensor(), expected))


def test_tanh_cpu_bwd_chain_add() raises:
    # y = tanh(x + 1), dy/dx = 1 - tanh(x+1)^2
    var t = Tensor[dtype].d1([0.0, 1.0], requires_grad=True)
    var t2 = t + Scalar[dtype](1)
    var out = t2.tanh()
    var loss = out.sum()
    loss.backward()
    var expected = Tensor[dtype].d1(
        [
            Float32(1) - scalar_tanh(Float32(1.0)) ** 2,
            Float32(1) - scalar_tanh(Float32(2.0)) ** 2,
        ]
    )
    assert_true(tanh_close(t.grad().as_tensor(), expected))


def test_tanh_cpu_bwd_double_tanh() raises:
    # y = tanh(tanh(x)), at x=0: tanh(0)=0, tanh(tanh(0))=0
    # dy/dx = (1-tanh(0)^2) * (1-tanh(tanh(0))^2) = 1*1 = 1
    var t = Tensor[dtype].scalar(0.0, requires_grad=True)
    var out = t.tanh().tanh()
    var loss = out.sum()
    loss.backward()
    assert_true(tanh_close(t.grad().as_tensor(), Tensor[dtype].scalar(1.0)))


def test_tanh_cpu_bwd_large() raises:
    var t = Tensor[dtype].zeros(Shape(32, 32), requires_grad=True)
    var out = t.tanh()
    var loss = out.sum()
    loss.backward()
    assert_true(
        tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(32, 32)))
    )


def test_tanh_cpu_bwd_grad_flow_two_paths() raises:
    # loss = tanh(x).sum() + tanh(x).sum() → grad = 2*(1-tanh(x)^2)
    var t = Tensor[dtype].zeros(Shape(3), requires_grad=True)
    var a = t.tanh()
    var b = t.tanh()
    var loss_a = a.sum()
    var loss_b = b.sum()
    var loss = loss_a + loss_b
    loss.backward()
    # grad = 2 * (1 - tanh(0)^2) = 2
    assert_true(
        tanh_close(
            t.grad().as_tensor(), Tensor[dtype].full(Shape(3), Float32(2.0))
        )
    )


def test_tanh_cpu_bwd_non_contiguous() raises:
    var t = Tensor[dtype].zeros(Shape(2, 3), requires_grad=True)
    var t_T = t.transpose()
    var out = t_T.tanh()
    var loss = out.sum()
    loss.backward()
    # grad flows back through transpose — all ones
    assert_true(
        tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(2, 3)))
    )


def test_tanh_cpu_bwd_with_matmul() raises:
    # loss = sum(tanh(A @ B))
    var A = Tensor[dtype].zeros(Shape(2, 3), requires_grad=True)
    var B = Tensor[dtype].rand(Shape(3, 4))
    var C = A.matmul(B)
    var out = C.tanh()
    var loss = out.sum()
    loss.backward()
    # At A=0: C=0, tanh(0)=0, grad_tanh=1, grad_C=ones(2,4)
    # grad_A = ones(2,4) @ B.T
    var grad_expected = Tensor[dtype].ones(Shape(2, 4)).matmul(B.transpose())
    assert_true(tanh_close(A.grad().as_tensor(), grad_expected))


# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD — GPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_tanh_gpu_fwd_scalar_zero() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].scalar(0.0).to_gpu()
        var out = t.tanh()
        assert_true(out.is_on_gpu())
        assert_true(tanh_close(out.to_cpu(), Tensor[dtype].scalar(0.0)))


def test_tanh_gpu_fwd_1d_zeros() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(8)).to_gpu()
        var out = t.tanh()
        assert_true(out.is_on_gpu())
        assert_true(tanh_close(out.to_cpu(), Tensor[dtype].zeros(Shape(8))))


def test_tanh_gpu_fwd_1d_known() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].d1([0.0, 0.5, -0.5, 1.0, -1.0])
        var out_gpu = t_cpu.to_gpu().tanh()
        assert_true(out_gpu.is_on_gpu())
        assert_true(tanh_close(out_gpu.to_cpu(), tanh_expected_1d()))


def test_tanh_gpu_fwd_2d_known() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].d2([[0.0, 1.0], [-1.0, 0.5]])
        var out_cpu = t_cpu.tanh()
        var out_gpu = t_cpu.to_gpu().tanh()
        assert_true(out_gpu.is_on_gpu())
        assert_true(tanh_close(out_gpu.to_cpu(), out_cpu))


def test_tanh_gpu_fwd_3d() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(Shape(2, 3, 4))
        var out_cpu = t_cpu.tanh()
        var out_gpu = t_cpu.to_gpu().tanh()
        assert_true(out_gpu.is_on_gpu())
        assert_true(tanh_close(out_gpu.to_cpu(), out_cpu))


def test_tanh_gpu_fwd_4d() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(Shape(2, 3, 4, 5))
        var out_cpu = t_cpu.tanh()
        var out_gpu = t_cpu.to_gpu().tanh()
        assert_true(out_gpu.is_on_gpu())
        assert_true(tanh_close(out_gpu.to_cpu(), out_cpu))


def test_tanh_gpu_fwd_large() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(Shape(64, 128))
        var out_cpu = t_cpu.tanh()
        var out_gpu = t_cpu.to_gpu().tanh()
        assert_true(out_gpu.is_on_gpu())
        assert_true(tanh_close(out_gpu.to_cpu(), out_cpu))


def test_tanh_gpu_fwd_matches_cpu() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(Shape(9, 20))
        assert_true(tanh_close(t_cpu.to_gpu().tanh().to_cpu(), t_cpu.tanh()))


def test_tanh_gpu_fwd_no_requires_grad() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].d1([1.0, 2.0]).to_gpu()
        var out = t.tanh[track_grad=False]()
        assert_true(not out.requires_grad)
        assert_true(not out.has_ancestry())


def test_tanh_gpu_fwd_requires_grad_propagates() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].d1([1.0, 2.0], requires_grad=True).to_gpu()
        var out = t.tanh()
        assert_true(out.is_on_gpu())
        assert_true(out.requires_grad)
        assert_true(out.has_ancestry())


def test_tanh_gpu_fwd_range_clamping() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].d1([-10.0, -1.0, 0.0, 1.0, 10.0]).to_gpu()
        var out = t.tanh().to_cpu()
        var data = out.data_ptr()
        for i in range(5):
            assert_true(data[i] >= Float32(-1.0) and data[i] <= Float32(1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD — GPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_tanh_gpu_bwd_zeros_1d() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(4), requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(
            tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(4)))
        )


def test_tanh_gpu_bwd_scalar_zero() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].scalar(0.0, requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(tanh_close(t.grad().as_tensor(), Tensor[dtype].scalar(1.0)))


def test_tanh_gpu_bwd_1d_known() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].d1(
            [0.0, 0.5, -0.5, 1.0, -1.0], requires_grad=True
        )
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(tanh_close(t.grad().as_tensor(), tanh_grad_expected_1d()))


def test_tanh_gpu_bwd_matches_cpu() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].rand(Shape(4, 5), requires_grad=True)

        # CPU backward
        var out_cpu = t.tanh()
        var loss_cpu = out_cpu.sum()
        loss_cpu.backward()
        var grad_cpu = t.grad().as_tensor().copy()
        t.zero_grad()

        # GPU backward
        var t_gpu = t.to_gpu()
        var out_gpu = t_gpu.tanh()
        var loss_gpu = out_gpu.sum()
        loss_gpu.backward()

        assert_true(tanh_close(t.grad().as_tensor(), grad_cpu))


def test_tanh_gpu_bwd_2d() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(3, 4), requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(
            tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(3, 4)))
        )


def test_tanh_gpu_bwd_3d() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(2, 3, 4), requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(
            tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(2, 3, 4)))
        )


def test_tanh_gpu_bwd_4d() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(2, 3, 4, 5), requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(
            tanh_close(
                t.grad().as_tensor(), Tensor[dtype].ones(Shape(2, 3, 4, 5))
            )
        )


def test_tanh_gpu_bwd_chain_mul() raises:
    comptime if has_accelerator():
        # y = tanh(2*x), dy/dx = 2*(1-tanh(2x)^2)
        var t = Tensor[dtype].d1([0.0, 1.0], requires_grad=True)
        var t_gpu = t.to_gpu()
        var t2 = t_gpu * Scalar[dtype](2)
        var out = t2.tanh()
        var loss = out.sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [
                Float32(2) * (Float32(1) - scalar_tanh(Float32(0.0)) ** 2),
                Float32(2) * (Float32(1) - scalar_tanh(Float32(2.0)) ** 2),
            ]
        )
        assert_true(tanh_close(t.grad().as_tensor(), expected))


def test_tanh_gpu_bwd_double_tanh() raises:
    comptime if has_accelerator():
        # y = tanh(tanh(x)), x=0 → grad=1
        var t = Tensor[dtype].scalar(0.0, requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh().tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(tanh_close(t.grad().as_tensor(), Tensor[dtype].scalar(1.0)))


def test_tanh_gpu_bwd_large() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(32, 32), requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(
            tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(32, 32)))
        )


def test_tanh_gpu_bwd_chain_add() raises:
    comptime if has_accelerator():
        # y = tanh(x + 1), dy/dx = 1 - tanh(x+1)^2
        var t = Tensor[dtype].d1([0.0, 1.0], requires_grad=True)
        var t_gpu = t.to_gpu()
        var t2 = t_gpu + Scalar[dtype](1)
        var out = t2.tanh()
        var loss = out.sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [
                Float32(1) - scalar_tanh(Float32(1.0)) ** 2,
                Float32(1) - scalar_tanh(Float32(2.0)) ** 2,
            ]
        )
        assert_true(tanh_close(t.grad().as_tensor(), expected))


def test_tanh_gpu_bwd_grad_flow_two_paths() raises:
    comptime if has_accelerator():
        # loss = tanh(x).sum() + tanh(x).sum() → grad = 2*(1-tanh(x)^2)
        var t = Tensor[dtype].zeros(Shape(3), requires_grad=True)
        var t_gpu = t.to_gpu()
        var a = t_gpu.tanh()
        var b = t_gpu.tanh()
        var loss_a = a.sum()
        var loss_b = b.sum()
        var loss = loss_a + loss_b
        loss.backward()
        assert_true(
            tanh_close(
                t.grad().as_tensor(), Tensor[dtype].full(Shape(3), Float32(2.0))
            )
        )


def test_tanh_gpu_bwd_scalar_one() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].scalar(1.0, requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        var expected = Tensor[dtype].scalar(
            Float32(1) - scalar_tanh(Float32(1.0)) ** 2
        )
        assert_true(tanh_close(t.grad().as_tensor(), expected))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
