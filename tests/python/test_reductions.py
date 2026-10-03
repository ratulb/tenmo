"""Tests for reduction operations (Category 12)."""
from __future__ import annotations

import numpy as np
import pytest
import tenmo
from helpers import assert_tensors_close


class TestReductions:
    def test_sum_full_reduction(self):
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0])
        assert t.sum().item() == 10.0

    def test_sum_along_axis(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = t.sum(axes=[1])
        assert_tensors_close(result, tenmo.tensor([3.0, 7.0]))

    def test_sum_along_axis0(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = t.sum(axes=[0])
        assert_tensors_close(result, tenmo.tensor([4.0, 6.0]))

    def test_sum_keepdims(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = t.sum(axes=[1], keepdims=True)
        assert result.shape == (2, 1)
        assert_tensors_close(result, tenmo.tensor([[3.0], [7.0]]))

    def test_mean_full_reduction(self):
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0])
        assert t.mean().item() == 2.5

    def test_mean_along_axis(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = t.mean(axes=[1])
        assert_tensors_close(result, tenmo.tensor([1.5, 3.5]))

    def test_max_full_reduction(self):
        t = tenmo.tensor([3.0, 1.0, 4.0, 1.0])
        assert t.max().item() == 4.0

    def test_max_along_axis(self):
        t = tenmo.tensor([[1.0, 5.0], [3.0, 2.0]])
        result = t.max(axes=[1])
        assert_tensors_close(result, tenmo.tensor([5.0, 3.0]))

    def test_min_full_reduction(self):
        t = tenmo.tensor([3.0, 1.0, 4.0])
        assert t.min().item() == 1.0

    def test_min_along_axis(self):
        t = tenmo.tensor([[5.0, 1.0], [3.0, 2.0]])
        result = t.min(axes=[1])
        assert_tensors_close(result, tenmo.tensor([1.0, 2.0]))

    def test_argmax_returns_correct_index(self):
        t = tenmo.tensor([1.0, 3.0, 2.0])
        assert t.argmax() == 1

    def test_argmax_along_axis(self):
        t = tenmo.tensor([[1.0, 5.0], [3.0, 2.0]])
        result = t.argmax(axis=1)
        assert result == [1, 0]

    def test_argmin_returns_correct_index(self):
        t = tenmo.tensor([3.0, 1.0, 2.0])
        assert t.argmin() == 1

    def test_argmin_along_axis(self):
        t = tenmo.tensor([[5.0, 1.0], [3.0, 2.0]])
        result = t.argmin(axis=1)
        assert result == [1, 1]  # both rows have min at index 1

    def test_sum_matches_numpy(self):
        np.random.seed(42)
        arr = np.random.randn(3, 4, 5).astype(np.float32)
        t = tenmo.Tensor.from_numpy(arr)
        np.testing.assert_allclose(
            t.sum(axes=[1]).numpy(), arr.sum(axis=1), atol=1e-5
        )

    def test_softmax_sums_to_one(self):
        t = tenmo.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        s = t.softmax(axes=[1])
        sums = s.sum(axes=[1])
        assert_tensors_close(sums, tenmo.tensor([1.0, 1.0]), atol=1e-5)

    def test_softmax_numerically_stable(self):
        """Uses subprocess to avoid JIT SIGILL from exp on large values."""
        import os, subprocess, sys
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(
            os.path.dirname(__file__), "..", "..", "python-binding"
        )
        result = subprocess.run(
            [sys.executable, "-c", """
import tenmo
t = tenmo.tensor([1000.0, 1001.0, 1002.0])
s = t.softmax()
total = s.sum().item()
assert abs(total - 1.0) < 1e-5
for v in s.tolist():
    assert v > 0
"""],
            capture_output=True, timeout=30, env=env,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr.decode()}"


# ── Previously skipped stubs now implemented ───────────────────


class TestReductionExtended:
    def test_prod_full_reduction(self):
        t = tenmo.tensor([2.0, 3.0, 4.0])
        result = t.product()
        assert abs(result.item() - 24.0) < 1e-5

    def test_prod_along_axis(self):
        t = tenmo.tensor([[2.0, 3.0], [4.0, 5.0]])
        result = t.product(axes=[1])
        np.testing.assert_allclose(result.numpy(), [6.0, 20.0], atol=1e-5)

    def test_std_matches_reference(self):
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = t.std()
        expected = np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)
        assert abs(result.item() - expected) < 1e-4

    def test_var_matches_reference(self):
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = t.variance()
        expected = np.var([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)
        assert abs(result.item() - expected) < 1e-4

    def test_argmax_tie_breaking(self):
        t = tenmo.tensor([1.0, 1.0])
        result = t.argmax()
        assert int(result) == 0

    def test_sum_along_multiple_axes(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = t.sum(axes=[0, 1])
        expected = np.sum(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=(0, 1))
        assert abs(result.item() - expected) < 1e-5
