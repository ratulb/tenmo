"""Tests for dtype conversion, device query, and detach (Batch 2E)."""
from __future__ import annotations

import numpy as np
import pytest
import tenmo


class TestDetach:
    def test_detach_shares_data(self):
        t = tenmo.tensor([1.0, 2.0, 3.0], requires_grad=True)
        d = t.detach()
        assert d.requires_grad is False
        assert d.tolist() == [1.0, 2.0, 3.0]

    def test_detach_is_not_leaf(self):
        t = tenmo.tensor([1.0, 2.0], requires_grad=True)
        d = t.detach()
        assert d.is_leaf() is False

    def test_detach_no_grad(self):
        t = tenmo.tensor([1.0, 2.0], requires_grad=True)
        d = t.detach()
        assert d.grad is None


class TestDevice:
    def test_device_returns_cpu(self):
        t = tenmo.tensor([1.0, 2.0])
        assert t.device == "cpu"

    def test_device_after_operations(self):
        t = tenmo.tensor([1.0, 2.0])
        s = t + 1.0
        assert s.device == "cpu"


class TestIsContiguous:
    def test_contiguous_tensor(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.is_contiguous() is True

    def test_transposed_is_not_contiguous(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        tp = t.transpose([1, 0])
        assert tp.is_contiguous() is False

    def test_reshaped_preserves_contiguous(self):
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        r = t.reshape((2, 3))
        assert r.is_contiguous() is True


class TestToDtype:
    def test_to_dtype_float32_to_int64(self):
        t = tenmo.tensor([1.5, 2.7, 3.0])
        i = t.to_dtype("int64")
        assert i.tolist() == [1, 2, 3]

    def test_to_dtype_int64_to_float32(self):
        t = tenmo.tensor([1, 2, 3], dtype="int64")
        f = t.to_dtype("float32")
        assert f.tolist() == [1.0, 2.0, 3.0]

    def test_to_dtype_float32_to_float64(self):
        t = tenmo.tensor([1.0, 2.0])
        f64 = t.to_dtype("float64")
        assert f64.numpy().dtype == np.float64

    def test_to_dtype_same_is_noop(self):
        t = tenmo.tensor([1.0, 2.0])
        f = t.to_dtype("float32")
        assert f.tolist() == [1.0, 2.0]

    def test_to_dtype_float_to_bool(self):
        t = tenmo.tensor([0.0, 1.0, 0.0, 3.14])
        b = t.to_dtype("bool")
        assert b.tolist() == [False, True, False, True]

    def test_float_alias(self):
        t = tenmo.tensor([1, 2, 3], dtype="int64")
        f = t.float()
        assert f.numpy().dtype == np.float32

    def test_int64_alias(self):
        t = tenmo.tensor([1.5, 2.7, 3.0])
        i = t.int64()
        assert i.tolist() == [1, 2, 3]

    def test_to_dtype_matches_numpy(self):
        t_np = np.array([1.5, 2.7, 3.9], dtype=np.float32)
        t = tenmo.tensor(t_np.tolist())
        result = t.to_dtype("int64").numpy()
        np.testing.assert_array_equal(result, t_np.astype(np.int64))


class TestDtypeExtended:
    def test_to_dtype_roundtrip_float32_to_int64_to_float32(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        r = t.to_dtype("int64").to_dtype("float32")
        np.testing.assert_allclose(r.numpy(), [1.0, 2.0, 3.0], atol=1e-5)

    def test_to_dtype_preserves_shape(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        r = t.to_dtype("int64")
        assert r.shape == (2, 2)

    def test_bool_to_float(self):
        t_bool = tenmo.tensor([True, False, True])
        t_float = t_bool.to_dtype("float32")
        assert t_float.tolist() == [1.0, 0.0, 1.0]

    def test_to_dtype_large_values(self):
        t = tenmo.tensor([1e10, -1e10])
        r = t.to_dtype("float64")
        assert abs(r.numpy()[0] - 1e10) < 1.0
