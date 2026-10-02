"""Tests for the float64 rich surface, slice 2 (E4 continued).

Linalg, unary math, masked_fill, triu/tril/cumsum/dot/outer/gather,
and mse/bce on Float64Tensor. Companion to test_float64_rich.py
(slice 1: autograd, reductions, shape/view, to_dtype).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import tenmo
from helpers import assert_tensors_close

F64 = "float64"


def ft64(data, **kwargs):
    return tenmo.tensor(data, dtype=F64, **kwargs)


class TestFloat64UnaryMath:
    def test_exp_log_sqrt(self):
        t = ft64([0.0, 1.0, 4.0])
        assert_tensors_close(t.exp(), ft64([1.0, math.e, math.e**4]))
        assert_tensors_close(
            ft64([1.0, math.e]).log(), ft64([0.0, 1.0]))
        assert_tensors_close(t.sqrt(), ft64([0.0, 1.0, 2.0]))

    def test_tanh_sigmoid_relu(self):
        t = ft64([0.0, 1.0, -1.0])
        assert_tensors_close(
            t.tanh(), ft64([math.tanh(0.0), math.tanh(1.0), math.tanh(-1.0)]))
        s = ft64([0.0]).sigmoid()
        assert abs(s.item() - 0.5) < 1e-9
        assert ft64([-2.0, 0.0, 3.0]).relu().tolist() == [0.0, 0.0, 3.0]

    def test_reciprocal_clip_abs_softmax(self):
        t = ft64([1.0, 2.0, 4.0])
        assert_tensors_close(t.reciprocal(), ft64([1.0, 0.5, 0.25]))
        assert_tensors_close(
            ft64([-1.0, 0.5, 2.0]).clip(0.0, 1.0), ft64([0.0, 0.5, 1.0]))
        assert_tensors_close(ft64([-3.0, 4.0]).abs(), ft64([3.0, 4.0]))
        sm = ft64([1.0, 2.0, 3.0]).softmax()
        assert abs(sm.numpy().sum() - 1.0) < 1e-9

    def test_unary_grad(self):
        x = ft64([1.0, 2.0], requires_grad=True)
        y = x.exp().sum()
        y.backward()
        assert_tensors_close(x.grad, ft64([math.e, math.e**2]))


class TestFloat64Linalg:
    def test_matmul(self):
        a = ft64([[1.0, 2.0], [3.0, 4.0]])
        b = ft64([[5.0, 6.0], [7.0, 8.0]])
        assert_tensors_close(a.matmul(b), ft64([[19.0, 22.0], [43.0, 50.0]]))
        assert_tensors_close(a @ b, ft64([[19.0, 22.0], [43.0, 50.0]]))

    def test_matmul_grad(self):
        a = ft64([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = ft64([[1.0, 0.0], [0.0, 1.0]])
        c = a.matmul(b).sum()
        c.backward()
        assert_tensors_close(a.grad, ft64([[1.0, 1.0], [1.0, 1.0]]))

    def test_triu_tril(self):
        t = ft64([[1.0, 2.0], [3.0, 4.0]])
        assert t.triu().tolist() == [[1.0, 2.0], [0.0, 4.0]]
        assert t.tril().tolist() == [[1.0, 0.0], [3.0, 4.0]]

    def test_cumsum_dot_outer_gather(self):
        t = ft64([1.0, 2.0, 3.0])
        assert_tensors_close(t.cumsum(), ft64([1.0, 3.0, 6.0]))
        assert t.dot(ft64([1.0, 1.0, 1.0])).item() == pytest.approx(6.0)
        assert t.outer(ft64([1.0, 2.0])).shape == (3, 2)
        assert t.gather([0, 2]).tolist() == [1.0, 3.0]

    def test_masked_fill(self):
        t = ft64([1.0, 2.0, 3.0])
        mask = tenmo.tensor([True, False, True])
        assert t.masked_fill(mask, 0.0).tolist() == [0.0, 2.0, 0.0]


class TestFloat64Losses:
    def test_mse(self):
        a = ft64([1.0, 2.0, 3.0])
        b = ft64([1.0, 2.0, 3.0])
        assert abs(a.mse(b).item()) < 1e-12
        c = ft64([2.0, 3.0, 4.0])
        assert a.mse(c).item() == pytest.approx(1.0)

    def test_bce(self):
        p = ft64([0.9, 0.1])
        tgt = ft64([1.0, 0.0])
        assert p.bce(tgt).item() > 0.0
        assert p.bce(tgt).item() == pytest.approx(
            -(math.log(0.9) + math.log(0.9)) / 2, rel=1e-6)

    def test_bce_logits_shape(self):
        p = ft64([0.9, 0.1])
        tgt = ft64([1.0, 0.0])
        assert p.bce_logits(tgt).shape == p.bce(tgt).shape
