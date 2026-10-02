"""Tests for loss methods (mse, bce, bce_logits)."""

from __future__ import annotations

import numpy as np
import pytest

import tenmo


class TestMSE:
    def test_perfect_match(self) -> None:
        a = tenmo.tensor([1.0, 2.0, 3.0])
        result = a.mse(a)
        assert abs(result.numpy()[()]) < 1e-6

    def test_known_value(self) -> None:
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([1.0, 0.0, 3.0])
        result = a.mse(b)
        expected = ((1 - 1) ** 2 + (2 - 0) ** 2 + (3 - 3) ** 2) / 3
        assert abs(result.numpy()[()] - expected) < 1e-5

    def test_2d(self) -> None:
        a = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = a.mse(b)
        assert abs(result.numpy()[()]) < 1e-6


class TestBCE:
    def test_perfect_pred(self) -> None:
        a = tenmo.tensor([0.99, 0.01, 0.99])
        b = tenmo.tensor([1.0, 0.0, 1.0])
        result = a.bce(b)
        assert result.numpy()[()] < 0.1

    def test_opposite(self) -> None:
        a = tenmo.tensor([0.99, 0.01])
        b = tenmo.tensor([1.0, 0.0])
        result = a.bce(b)
        val = result.numpy()[()]
        assert val < 0.1


class TestBCELogits:
    def test_zero_logits_correct(self) -> None:
        a = tenmo.tensor([0.0, 0.0])
        b = tenmo.tensor([0.0, 0.0])
        result = a.bce_logits(b)
        val = result.numpy()[()]
        assert val < 0.7
