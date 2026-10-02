"""Tests for module-level factory functions (full, rand, linspace, eye, randn, arange)
and instance methods (zeros_like, ones_like)."""

from __future__ import annotations

import numpy as np
import pytest

import tenmo


# ── full ────────────────────────────────────────────────────────────────

class TestFull:
    def test_basic(self) -> None:
        t = tenmo.full((2, 3), 42.0)
        assert t.shape == (2, 3)
        np.testing.assert_allclose(t.numpy(), np.full((2, 3), 42.0))

    def test_zeros(self) -> None:
        t = tenmo.full((4,), 0.0)
        np.testing.assert_allclose(t.numpy(), np.zeros(4))

    def test_ones(self) -> None:
        t = tenmo.full((3, 3), 1.0)
        np.testing.assert_allclose(t.numpy(), np.ones((3, 3)))

    def test_negative_value(self) -> None:
        t = tenmo.full((2, 2), -3.5)
        np.testing.assert_allclose(t.numpy(), np.full((2, 2), -3.5))

    def test_scalar_shape(self) -> None:
        t = tenmo.full((1,), 7.0)
        assert t.shape == (1,)
        np.testing.assert_allclose(t.numpy(), np.array([7.0]))

    def test_int64_dtype(self) -> None:
        t = tenmo.full((2, 2), 5.0, dtype="int64")
        assert t.numpy().dtype == np.int64
        np.testing.assert_array_equal(t.numpy(), np.full((2, 2), 5, dtype=np.int64))


# ── rand ────────────────────────────────────────────────────────────────

class TestRand:
    def test_default_range(self) -> None:
        t = tenmo.rand((100, 10))
        vals = t.numpy()
        assert vals.min() >= 0.0
        assert vals.max() < 1.0

    def test_custom_range(self) -> None:
        t = tenmo.rand((100,), low=-2.0, high=2.0)
        vals = t.numpy()
        assert vals.min() >= -2.0
        assert vals.max() < 2.0

    def test_shape(self) -> None:
        t = tenmo.rand((3, 4, 5))
        assert t.shape == (3, 4, 5)

    def test_uniform_distribution(self) -> None:
        t = tenmo.rand((10000,), low=0.0, high=1.0)
        mean = t.numpy().mean()
        assert 0.45 < mean < 0.55

    def test_single_element(self) -> None:
        t = tenmo.rand((1,))
        assert t.shape == (1,)

    def test_float64(self) -> None:
        t = tenmo.rand((10,), dtype="float64")
        assert t.numpy().dtype == np.float64


# ── linspace ────────────────────────────────────────────────────────────

class TestLinspace:
    def test_basic(self) -> None:
        t = tenmo.linspace(0.0, 1.0, 5)
        np.testing.assert_allclose(t.numpy(), np.linspace(0.0, 1.0, 5))

    def test_negative_range(self) -> None:
        t = tenmo.linspace(-1.0, 1.0, 11)
        np.testing.assert_allclose(t.numpy(), np.linspace(-1.0, 1.0, 11), atol=1e-6)

    def test_single_step(self) -> None:
        t = tenmo.linspace(0.0, 5.0, 1)
        np.testing.assert_allclose(t.numpy(), np.array([0.0]), atol=1e-6)

    def test_shape(self) -> None:
        t = tenmo.linspace(0.0, 10.0, 7)
        assert t.shape == (7,)

    def test_endpoint(self) -> None:
        t = tenmo.linspace(0.0, 1.0, 100)
        assert abs(t.numpy()[-1] - 1.0) < 1e-5
        assert abs(t.numpy()[0] - 0.0) < 1e-5


# ── eye ─────────────────────────────────────────────────────────────────

class TestEye:
    def test_basic(self) -> None:
        t = tenmo.eye(3)
        np.testing.assert_allclose(t.numpy(), np.eye(3))

    def test_1x1(self) -> None:
        t = tenmo.eye(1)
        np.testing.assert_allclose(t.numpy(), np.eye(1))

    def test_5x5(self) -> None:
        t = tenmo.eye(5)
        expected = np.eye(5)
        np.testing.assert_allclose(t.numpy(), expected)
        assert t.shape == (5, 5)

    def test_off_diagonal_zero(self) -> None:
        t = tenmo.eye(4)
        vals = t.numpy()
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert vals[i, j] == 0.0


# ── zeros_like / ones_like ─────────────────────────────────────────────

class TestZerosOnesLike:
    def test_zeros_like(self) -> None:
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        z = t.zeros_like()
        assert z.shape == (2, 3)
        np.testing.assert_allclose(z.numpy(), np.zeros((2, 3)))

    def test_ones_like(self) -> None:
        t = tenmo.tensor([1.0, 2.0, 3.0])
        o = t.ones_like()
        assert o.shape == (3,)
        np.testing.assert_allclose(o.numpy(), np.ones(3))

    def test_zeros_like_preserves_shape(self) -> None:
        t = tenmo.rand((5, 4, 3))
        z = t.zeros_like()
        assert z.shape == t.shape

    def test_ones_like_2d(self) -> None:
        t = tenmo.zeros((4, 4))
        o = t.ones_like()
        np.testing.assert_allclose(o.numpy(), np.ones((4, 4)))


# ── existing factories still work ──────────────────────────────────────

class TestExistingFactories:
    def test_zeros(self) -> None:
        t = tenmo.zeros((3, 3))
        np.testing.assert_allclose(t.numpy(), np.zeros((3, 3)))

    def test_ones(self) -> None:
        t = tenmo.ones((2, 4))
        np.testing.assert_allclose(t.numpy(), np.ones((2, 4)))

    def test_randn(self) -> None:
        # 10k samples: |mean| < 0.2 is a ~20-sigma bound (SE=0.01), so this
        # is deterministic in practice; n=100 failed ~5% of runs by chance.
        t = tenmo.randn((10000,))
        assert t.shape == (10000,)
        mean = abs(t.numpy().mean())
        assert mean < 0.2

    def test_arange(self) -> None:
        t = tenmo.arange(10, start=0, step=1)
        np.testing.assert_allclose(t.numpy(), np.arange(0, 10, 1))
