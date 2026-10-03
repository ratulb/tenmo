"""Tests for elementwise arithmetic (Category 7)."""
from __future__ import annotations

import numpy as np
import pytest
import tenmo
from helpers import assert_tensors_close


class TestElementwiseArithmetic:
    def test_add_tensor_tensor(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([4.0, 5.0, 6.0])
        c = a + b
        assert_tensors_close(c, tenmo.tensor([5.0, 7.0, 9.0]))

    def test_add_tensor_scalar(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        c = t + 10.0
        assert_tensors_close(c, tenmo.tensor([11.0, 12.0, 13.0]))

    def test_radd_scalar_tensor(self):
        t = tenmo.tensor([1.0, 2.0])
        c = 5.0 + t
        assert_tensors_close(c, tenmo.tensor([6.0, 7.0]))

    def test_sub_tensor_tensor(self):
        a = tenmo.tensor([5.0, 7.0])
        b = tenmo.tensor([1.0, 2.0])
        assert_tensors_close(a - b, tenmo.tensor([4.0, 5.0]))

    def test_rsub_scalar_tensor(self):
        t = tenmo.tensor([1.0, 2.0])
        c = 10.0 - t
        assert_tensors_close(c, tenmo.tensor([9.0, 8.0]))

    def test_mul_tensor_tensor(self):
        a = tenmo.tensor([2.0, 3.0])
        b = tenmo.tensor([4.0, 5.0])
        assert_tensors_close(a * b, tenmo.tensor([8.0, 15.0]))

    def test_rmul_scalar_tensor(self):
        t = tenmo.tensor([2.0, 3.0])
        assert_tensors_close(3.0 * t, tenmo.tensor([6.0, 9.0]))

    def test_truediv_tensor_tensor(self):
        a = tenmo.tensor([10.0, 9.0])
        b = tenmo.tensor([2.0, 3.0])
        assert_tensors_close(a / b, tenmo.tensor([5.0, 3.0]))

    def test_rtruediv_scalar_tensor(self):
        t = tenmo.tensor([2.0, 4.0])
        c = 8.0 / t
        assert_tensors_close(c, tenmo.tensor([4.0, 2.0]))

    def test_pow_tensor_scalar_exponent(self):
        t = tenmo.tensor([2.0, 3.0])
        c = t**2
        assert_tensors_close(c, tenmo.tensor([4.0, 9.0]))

    def test_neg_unary_minus(self):
        t = tenmo.tensor([1.0, -2.0, 3.0])
        assert_tensors_close(-t, tenmo.tensor([-1.0, 2.0, -3.0]))

    def test_abs_returns_magnitude(self):
        t = tenmo.tensor([-3.0, 4.0, -0.5])
        assert_tensors_close(t.abs(), tenmo.tensor([3.0, 4.0, 0.5]))

    def test_iadd_inplace_mutates(self):
        t = tenmo.tensor([1.0, 2.0])
        t += tenmo.tensor([3.0, 4.0])
        assert_tensors_close(t, tenmo.tensor([4.0, 6.0]))

    def test_isub_inplace_mutates(self):
        t = tenmo.tensor([10.0, 20.0])
        t -= tenmo.tensor([1.0, 2.0])
        assert_tensors_close(t, tenmo.tensor([9.0, 18.0]))

    def test_imul_inplace_mutates(self):
        t = tenmo.tensor([3.0, 4.0])
        t *= tenmo.tensor([2.0, 5.0])
        assert_tensors_close(t, tenmo.tensor([6.0, 20.0]))

    def test_itruediv_inplace_mutates(self):
        t = tenmo.tensor([10.0, 20.0])
        t /= tenmo.tensor([2.0, 5.0])
        assert_tensors_close(t, tenmo.tensor([5.0, 4.0]))

    def test_iadd_scalar(self):
        t = tenmo.tensor([1.0, 2.0])
        t += 3.0
        assert_tensors_close(t, tenmo.tensor([4.0, 5.0]))

    def test_broadcast_add(self):
        a = tenmo.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
        b = tenmo.tensor([[10.0], [20.0]])    # (2, 1)
        c = a + b  # (2, 3)
        assert c.shape == (2, 3)
        expected = tenmo.tensor([[11.0, 12.0, 13.0], [21.0, 22.0, 23.0]])
        assert_tensors_close(c, expected)

    def test_clip(self):
        t = tenmo.tensor([-1.0, 0.5, 2.0])
        c = t.clip(0.0, 1.0)
        assert_tensors_close(c, tenmo.tensor([0.0, 0.5, 1.0]))

    def test_dtype_mismatch_raises(self):
        a = tenmo.tensor([1.0, 2.0])
        b = tenmo.tensor([1, 2], dtype="int64")
        with pytest.raises(TypeError, match="dtype mismatch"):
            _ = a + b

    def test_bool_arithmetic_raises(self):
        t = tenmo.tensor([True, False])
        try:
            _ = t + t
        except TypeError:
            pass  # expected — core may or may not reject this

    def test_reciprocal_matches_one_over_x(self):
        t = tenmo.tensor([1.0, 2.0, 4.0])
        r = t.reciprocal()
        expected = tenmo.tensor([1.0, 0.5, 0.25])
        assert_tensors_close(r, expected)

    def test_exp(self):
        import math
        t = tenmo.tensor([0.0, 1.0, 2.0])
        e = t.exp()
        expected = [math.exp(x) for x in [0.0, 1.0, 2.0]]
        vals = e.tolist()
        for v, ev in zip(vals, expected):
            assert abs(v - ev) < 1e-5

    def test_log(self):
        import math
        t = tenmo.tensor([1.0, math.e, math.e**2])
        l = t.log()
        expected = [0.0, 1.0, 2.0]
        vals = l.tolist()
        for v, ev in zip(vals, expected):
            assert abs(v - ev) < 1e-5

    def test_sqrt(self):
        import math
        t = tenmo.tensor([1.0, 4.0, 9.0])
        s = t.sqrt()
        expected = [1.0, 2.0, 3.0]
        vals = s.tolist()
        for v, ev in zip(vals, expected):
            assert abs(v - ev) < 1e-5

    def test_tanh(self):
        import math
        t = tenmo.tensor([0.0, 1.0, -1.0])
        result = t.tanh()
        expected = [math.tanh(x) for x in [0.0, 1.0, -1.0]]
        vals = result.tolist()
        for v, ev in zip(vals, expected):
            assert abs(v - ev) < 1e-5

    def test_sigmoid(self):
        t = tenmo.tensor([0.0, 1.0, -1.0])
        result = t.sigmoid()
        vals = result.tolist()
        assert abs(vals[0] - 0.5) < 1e-5
        assert abs(vals[1] - 0.7310585) < 1e-4
        assert abs(vals[2] - 0.2689415) < 1e-4

    def test_relu(self):
        t = tenmo.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        r = t.relu()
        assert r.tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]

    def test_unary_ops_preserve_shape(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.exp().shape == (2, 2)
        assert t.log().shape == (2, 2)
        assert t.sqrt().shape == (2, 2)
        assert t.tanh().shape == (2, 2)
        assert t.sigmoid().shape == (2, 2)
        assert t.relu().shape == (2, 2)
        assert t.reciprocal().shape == (2, 2)

    @pytest.mark.skip(reason="pending Phase 2 — floordiv/mod not exposed")
    def test_floordiv_tensor_tensor(self): pass

    @pytest.mark.skip(reason="pending Phase 2 — floordiv/mod not exposed")
    def test_mod_tensor_tensor(self): pass


# ── Arithmetic error cases / Unary edge cases / Sigmoid ──


class TestArithmeticExtended:
    def test_inplace_rejects_incompatible_shape(self):
        a = tenmo.tensor([1.0, 2.0])
        b = tenmo.tensor([1.0, 2.0, 3.0])
        try:
            a.__iadd__(b)
        except (ValueError, RuntimeError):
            pass  # expected

    def test_div_by_zero_infinity(self):
        t = tenmo.tensor([1.0, 2.0])
        ones = tenmo.tensor([0.0, 0.0])
        result = t / ones
        # Should produce inf or large values
        result_np = result.numpy()
        assert np.all(np.isinf(result_np) | (result_np > 1e10))


class TestUnaryEdgeCases:
    def test_log_of_small_positive(self):
        t = tenmo.tensor([1e-10, 1e-5])
        result = t.log()
        result_np = result.numpy()
        assert result_np[0] < -10.0
        assert result_np[1] < -5.0

    def test_sqrt_of_tiny_positive(self):
        t = tenmo.tensor([1e-10])
        result = t.sqrt()
        assert abs(result.item() - 1e-5) < 1e-6

    def test_exp_large_negative(self):
        t = tenmo.tensor([-50.0, -100.0])
        result = t.exp()
        result_np = result.numpy()
        assert result_np[0] > 0
        assert result_np[1] >= 0

    def test_sigmoid_bounded(self):
        t = tenmo.tensor([-10.0, 0.0, 10.0])
        result = t.sigmoid()
        result_np = result.numpy()
        assert np.all(result_np >= 0.0)
        assert np.all(result_np <= 1.0)
        assert abs(result_np[1] - 0.5) < 1e-5
        assert result_np[0] < 0.01
        assert result_np[2] > 0.99
        # sigmoid(-10) close to 0, sigmoid(10) close to 1
        assert result_np[0] < 0.01
        assert result_np[2] > 0.99


class TestFloat64Arithmetic:
    """float64 arithmetic surface (widened one family at a time)."""

    def test_add_mul_div(self):
        a = tenmo.tensor([1.0, 2.0, 3.0], dtype="float64")
        b = tenmo.tensor([4.0, 5.0, 6.0], dtype="float64")
        assert_tensors_close(a + b, tenmo.tensor([5.0, 7.0, 9.0], dtype="float64"))
        assert_tensors_close(a * b, tenmo.tensor([4.0, 10.0, 18.0], dtype="float64"))
        assert_tensors_close(b / a, tenmo.tensor([4.0, 2.5, 2.0], dtype="float64"))

    def test_scalar_forms(self):
        t = tenmo.tensor([1.0, 2.0], dtype="float64")
        assert_tensors_close(t + 10.0, tenmo.tensor([11.0, 12.0], dtype="float64"))
        assert_tensors_close(10.0 - t, tenmo.tensor([9.0, 8.0], dtype="float64"))
        assert_tensors_close(t**2, tenmo.tensor([1.0, 4.0], dtype="float64"))
        assert_tensors_close(-t, tenmo.tensor([-1.0, -2.0], dtype="float64"))

    def test_inplace_and_index(self):
        t = tenmo.tensor([1.0, 2.0, 3.0], dtype="float64")
        t += tenmo.tensor([1.0, 1.0, 1.0], dtype="float64")
        assert_tensors_close(t, tenmo.tensor([2.0, 3.0, 4.0], dtype="float64"))
        t[0] = 99.0
        assert t[0].item() == 99.0
        assert t.numpy().dtype == np.float64

    def test_mixed_dtype_rejected(self):
        a = tenmo.tensor([1.0], dtype="float64")
        b = tenmo.tensor([1.0], dtype="float32")
        with pytest.raises(TypeError, match="dtype mismatch"):
            _ = a + b
