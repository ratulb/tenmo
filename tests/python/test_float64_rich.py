"""Tests for the float64 rich surface, slice 1 (E4 continued).

Autograd, reductions, shape/view ops, and to_dtype on Float64Tensor,
plus the facade autograd guard for dtypes without native support.
Slice 2 (matmul, unary math, masked_fill, triu/tril/cumsum/dot/outer,
gather, mse/bce) follows in its own commit with its tests.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import tenmo
from helpers import assert_tensors_close, assert_tensors_equal

F64 = "float64"


def ft64(data, **kwargs):
    return tenmo.tensor(data, dtype=F64, **kwargs)


class TestFloat64Autograd:
    def test_backward_populates_grad(self):
        x = ft64([2.0, 3.0], requires_grad=True)
        y = x * x
        y.backward()
        assert_tensors_close(x.grad, ft64([4.0, 6.0]))

    def test_requires_grad_settable_and_zero_grad(self):
        x = ft64([1.0, 2.0], requires_grad=True)
        assert x.requires_grad is True
        y = (x * 2.0).sum()
        y.backward()
        assert_tensors_close(x.grad, ft64([2.0, 2.0]))
        x.zero_grad()
        if x.grad is not None:
            assert_tensors_close(x.grad, ft64([0.0, 0.0]))

    def test_seed_grad(self):
        x = ft64([1.0, 2.0], requires_grad=True)
        x.seed_grad(3.0)
        assert_tensors_close(x.grad, ft64([3.0, 3.0]))

    def test_grad_through_sum_mean(self):
        x = ft64([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        s = x.sum()
        s.backward()
        assert_tensors_close(x.grad, ft64([[1.0, 1.0], [1.0, 1.0]]))

    def test_detach_cuts_graph(self):
        x = ft64([1.0, 2.0], requires_grad=True)
        d = (x * 2.0).detach()
        assert d.is_leaf() or d.grad is None


class TestFloat64Reductions:
    def test_sum_mean(self):
        t = ft64([[1.0, 2.0], [3.0, 4.0]])
        assert t.sum().item() == 10.0
        assert t.mean().item() == 2.5
        assert_tensors_close(t.sum(axes=[0]), ft64([4.0, 6.0]))
        assert t.sum(axes=[1], keepdims=True).shape == (2, 1)

    def test_max_min_argmax_argmin(self):
        t = ft64([3.0, 1.0, 2.0])
        assert t.max().item() == 3.0
        assert t.min().item() == 1.0
        assert t.argmax(axis=0) == 0
        assert t.argmin(axis=0) == 1

    def test_product_variance_std_norm(self):
        t = ft64([1.0, 2.0, 3.0, 4.0])
        assert t.product().item() == pytest.approx(24.0)
        assert t.variance().item() == pytest.approx(
            np.var([1.0, 2.0, 3.0, 4.0], ddof=1))
        assert t.std().item() == pytest.approx(
            np.std([1.0, 2.0, 3.0, 4.0], ddof=1))
        assert t.norm().item() == pytest.approx(math.sqrt(30.0))


class TestFloat64ShapeAndLinalg:
    def test_reshape_flatten_transpose(self):
        t = ft64([[1.0, 2.0], [3.0, 4.0]])
        assert t.reshape([4]).shape == (4,)
        assert t.flatten().shape == (4,)
        assert t.transpose().tolist() == [[1.0, 3.0], [2.0, 4.0]]
        assert t.permute([1, 0]).tolist() == [[1.0, 3.0], [2.0, 4.0]]

    def test_squeeze_unsqueeze_expand_contiguous(self):
        t = ft64([[[1.0, 2.0]]])
        assert t.squeeze().shape == (2,)
        assert ft64([1.0, 2.0]).unsqueeze(0).shape == (1, 2)
        assert ft64([[1.0, 2.0]]).expand([2, 2]).shape == (2, 2)
        assert t.contiguous().is_contiguous() is True
        assert t.detach().shape == t.shape
        assert t.device == "cpu"

    def test_zeros_ones_like(self):
        t = ft64([[1.0, 2.0], [3.0, 4.0]])
        assert_tensors_equal(t.zeros_like(), ft64([[0.0, 0.0], [0.0, 0.0]]))
        assert_tensors_equal(t.ones_like(), ft64([[1.0, 1.0], [1.0, 1.0]]))


class TestFloat64Dtype:
    def test_to_dtype_roundtrip(self):
        t = ft64([1.5, 2.5])
        f32 = t.to_dtype("float32")
        assert f32.numpy().dtype == np.float32
        back = f32.to_dtype("float64")
        assert back.numpy().dtype == np.float64
        assert_tensors_close(back, t, atol=1e-6)

    def test_numpy_dtype_is_float64(self):
        assert ft64([1.0]).numpy().dtype == np.float64


class TestAutogradGuard:
    def test_int_backward_raises_typeerror(self):
        t = tenmo.tensor([1, 2], dtype="int64")
        with pytest.raises(TypeError, match="autograd"):
            t.backward()

    def test_int_grad_raises_typeerror(self):
        t = tenmo.tensor([1, 2], dtype="int64")
        with pytest.raises(TypeError, match="autograd"):
            _ = t.grad

    def test_int_zero_grad_raises_typeerror(self):
        t = tenmo.tensor([1, 2], dtype="int64")
        with pytest.raises(TypeError, match="autograd"):
            t.zero_grad()
