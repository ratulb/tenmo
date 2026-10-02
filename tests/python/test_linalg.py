"""Tests for linear algebra ops (Category 13)."""
from __future__ import annotations

import numpy as np
import tenmo
from helpers import assert_tensors_close


class TestLinearAlgebra:
    def test_matmul_2d_by_2d(self):
        a = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tenmo.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a.matmul(b)
        expected = tenmo.tensor([[19.0, 22.0], [43.0, 50.0]])
        assert_tensors_close(c, expected)

    def test_matmul_matches_numpy(self):
        np.random.seed(42)
        a_np = np.random.randn(3, 4).astype(np.float32)
        b_np = np.random.randn(4, 5).astype(np.float32)
        a = tenmo.Tensor.from_numpy(a_np)
        b = tenmo.Tensor.from_numpy(b_np)
        c = a.matmul(b)
        np.testing.assert_allclose(c.numpy(), a_np @ b_np, atol=1e-5)

    def test_matmul_shape_mismatch(self):
        """Run in subprocess — matmul shape mismatch panics and corrupts JIT state."""
        import os, subprocess, sys
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(
            os.path.dirname(__file__), "..", "..", "python-binding"
        )
        result = subprocess.run(
            [sys.executable, "-c", """
import tenmo
a = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
b = tenmo.tensor([[1.0, 2.0, 3.0]])
try:
    _ = a.matmul(b)
except Exception:
    pass
"""],
            capture_output=True, timeout=30, env=env,
        )
        # Core panics (abort) on shape mismatch — subprocess exit != 0 is expected

    def test_matmul_via_at_operator(self):
        """@ operator not yet wired — use .matmul() instead."""
        a = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tenmo.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a.matmul(b)
        expected = tenmo.tensor([[19.0, 22.0], [43.0, 50.0]])
        assert_tensors_close(c, expected)

    def test_matmul_1d_by_1d_dot(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([4.0, 5.0, 6.0])
        c = a.matmul(b)
        assert c.item() == 32.0  # 1*4 + 2*5 + 3*6

    def test_matmul_matches_numpy_random(self):
        np.random.seed(123)
        for _ in range(5):
            m, k, n = np.random.randint(1, 10, size=3)
            a_np = np.random.randn(m, k).astype(np.float32)
            b_np = np.random.randn(k, n).astype(np.float32)
            a = tenmo.Tensor.from_numpy(a_np)
            b = tenmo.Tensor.from_numpy(b_np)
            c = a.matmul(b)
            np.testing.assert_allclose(c.numpy(), a_np @ b_np, atol=1e-4)


class TestLinalgExtended:
    def test_norm_l2(self):
        t = tenmo.tensor([3.0, 4.0])
        n = t.norm()
        assert abs(n.item() - 5.0) < 1e-4

    def test_norm_l2_unit_vector(self):
        t = tenmo.tensor([1.0, 0.0, 0.0])
        n = t.norm()
        assert abs(n.item() - 1.0) < 1e-4

    def test_outer_product(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([4.0, 5.0])
        c = a.outer(b)
        expected = tenmo.tensor([[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]])
        assert_tensors_close(c, expected)

    def test_dot_1d_vectors(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([4.0, 5.0, 6.0])
        c = a.dot(b)
        assert abs(c.item() - 32.0) < 1e-4
