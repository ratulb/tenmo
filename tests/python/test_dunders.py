"""Tests for dunders and metadata (__len__, __matmul__, __invert__, allclose, seed_grad)."""

from __future__ import annotations

import numpy as np
import pytest

import tenmo


class TestLen:
    def test_basic(self) -> None:
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert len(t) == 3

    def test_1d(self) -> None:
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0])
        assert len(t) == 4

    def test_single(self) -> None:
        t = tenmo.tensor([42.0])
        assert len(t) == 1


class TestMatmulDunder:
    def test_2d(self) -> None:
        a = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tenmo.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = a @ b
        expected = np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([[5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)

    def test_matches_matmul(self) -> None:
        a = tenmo.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = tenmo.tensor([[3.0, 4.0], [5.0, 6.0]])
        assert np.allclose((a @ b).numpy(), a.matmul(b).numpy())


class TestInvert:
    def test_bool_invert(self) -> None:
        t = tenmo.tensor([True, False, True])
        result = ~t
        np.testing.assert_allclose(result.numpy(), [0.0, 1.0, 0.0])

    def test_invert_shape(self) -> None:
        t = tenmo.tensor([True, False, True, False])
        assert (~t).shape == t.shape


class TestAllclose:
    def test_identical(self) -> None:
        a = tenmo.tensor([1.0, 2.0, 3.0])
        assert a.allclose(a)

    def test_close(self) -> None:
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([1.000001, 2.000001, 3.000001])
        assert a.allclose(b, atol=1e-4)

    def test_different(self) -> None:
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([1.0, 2.0, 99.0])
        assert not a.allclose(b)


class TestSeedGrad:
    def test_basic(self) -> None:
        t = tenmo.tensor([1.0, 2.0, 3.0], requires_grad=True)
        t.seed_grad(5.0)
        grads = t.grad.numpy()
        np.testing.assert_allclose(grads, [5.0, 5.0, 5.0])

    def test_default(self) -> None:
        t = tenmo.tensor([10.0], requires_grad=True)
        t.seed_grad()
        grads = t.grad.numpy()
        np.testing.assert_allclose(grads, [1.0])
