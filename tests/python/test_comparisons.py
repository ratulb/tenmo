"""Tests for comparison operators (Category 8)."""
from __future__ import annotations

import numpy as np
import tenmo
from helpers import assert_tensors_close


class TestComparisonOperators:
    def test_eq_elementwise(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([1.0, 2.5, 3.0])
        result = a == b
        assert result.tolist() == [True, False, True]

    def test_ne_elementwise(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([1.0, 2.5, 3.0])
        result = a != b
        assert result.tolist() == [False, True, False]

    def test_lt_elementwise(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([2.0, 2.0, 1.0])
        result = a < b
        assert result.tolist() == [True, False, False]

    def test_le_elementwise(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([1.0, 3.0, 2.0])
        result = a <= b
        assert result.tolist() == [True, True, False]

    def test_gt_elementwise(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([2.0, 1.0, 3.0])
        result = a > b
        assert result.tolist() == [False, True, False]

    def test_ge_elementwise(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([1.0, 3.0, 2.0])
        result = a >= b
        assert result.tolist() == [True, False, True]

    def test_comparison_against_scalar(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        result = t > 1.5
        assert result.tolist() == [False, True, True]

    def test_comparison_broadcasts(self):
        a = tenmo.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        b = tenmo.tensor([[2.5, 2.5, 2.5], [0.5, 0.5, 0.5]])
        result = a < b
        assert result.shape == (2, 3)
        assert result.tolist() == [[True, True, False], [False, False, False]]
