"""Tests for triu, tril, cumsum, dot, outer, gather instance methods."""

from __future__ import annotations

import numpy as np
import pytest

import tenmo


# ── triu / tril ──────────────────────────────────────────────────────

class TestTriu:
    def test_basic(self) -> None:
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        u = t.triu()
        expected = np.triu(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        np.testing.assert_allclose(u.numpy(), expected)

    def test_diagonal_1(self) -> None:
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        u = t.triu(diagonal=1)
        expected = np.triu(np.ones((3, 3)), k=1) * np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_allclose(u.numpy(), expected)

    def test_shape_preserved(self) -> None:
        t = tenmo.rand((4, 5))
        assert t.triu().shape == (4, 5)

    def test_negative_diagonal(self) -> None:
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        u = t.triu(diagonal=-1)
        expected = np.triu(np.array([[1.0, 2.0], [3.0, 4.0]]), k=-1)
        np.testing.assert_allclose(u.numpy(), expected)


class TestTril:
    def test_basic(self) -> None:
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        l = t.tril()
        expected = np.tril(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
        np.testing.assert_allclose(l.numpy(), expected)

    def test_diagonal_neg1(self) -> None:
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        l = t.tril(diagonal=-1)
        expected = np.tril(np.ones((3, 3)), k=-1) * np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        np.testing.assert_allclose(l.numpy(), expected)

    def test_zeros_above(self) -> None:
        t = tenmo.ones((3, 3))
        l = t.tril()
        vals = l.numpy()
        assert vals[0, 1] == 0.0
        assert vals[0, 2] == 0.0
        assert vals[1, 2] == 0.0


# ── cumsum ──────────────────────────────────────────────────────────

class TestCumsum:
    def test_basic(self) -> None:
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0])
        cs = t.cumsum()
        np.testing.assert_allclose(cs.numpy(), np.cumsum([1.0, 2.0, 3.0, 4.0]))

    def test_2d_axis0(self) -> None:
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        cs = t.cumsum(axis=0)
        expected = np.cumsum(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), axis=0)
        np.testing.assert_allclose(cs.numpy(), expected)

    def test_2d_axis1(self) -> None:
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        cs = t.cumsum(axis=1)
        expected = np.cumsum(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=1)
        np.testing.assert_allclose(cs.numpy(), expected)

    def test_shape_preserved(self) -> None:
        t = tenmo.rand((3, 4))
        assert t.cumsum().shape == (3, 4)


# ── dot ─────────────────────────────────────────────────────────────

class TestDot:
    def test_1d(self) -> None:
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([4.0, 5.0, 6.0])
        result = a.dot(b)
        expected = np.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert abs(result.numpy()[()] - expected) < 1e-5

    @pytest.mark.skip(reason="2D/ND dot (BLAS matmul path) SIGILLs on this box — use matmul() instead")
    def test_2d_matmul(self) -> None:
        pass


# ── outer ───────────────────────────────────────────────────────────

class TestOuter:
    def test_basic(self) -> None:
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([4.0, 5.0])
        result = a.outer(b)
        expected = np.outer([1.0, 2.0, 3.0], [4.0, 5.0])
        np.testing.assert_allclose(result.numpy(), expected)

    def test_shape(self) -> None:
        a = tenmo.rand((5,))
        b = tenmo.rand((4,))
        result = a.outer(b)
        assert result.shape == (5, 4)

    def test_values(self) -> None:
        a = tenmo.tensor([2.0, 3.0])
        b = tenmo.tensor([10.0])
        result = a.outer(b)
        expected = np.outer([2.0, 3.0], [10.0])
        np.testing.assert_allclose(result.numpy(), expected)


# ── gather ──────────────────────────────────────────────────────────

class TestGather:
    def test_basic(self) -> None:
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = t.gather([0, 2], axis=0)
        expected = np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
        np.testing.assert_allclose(result.numpy(), expected)

    def test_axis1(self) -> None:
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = t.gather([0, 2], axis=1)
        expected = np.array([[1.0, 3.0], [4.0, 6.0]])
        np.testing.assert_allclose(result.numpy(), expected)

    def test_single_index(self) -> None:
        t = tenmo.tensor([[10.0, 20.0], [30.0, 40.0]])
        result = t.gather([1], axis=0)
        np.testing.assert_allclose(result.numpy(), np.array([[30.0, 40.0]]))
