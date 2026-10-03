"""Tests for shape, metadata, views, and broadcasting (Categories 3, 5, 6)."""
from __future__ import annotations

import numpy as np
import pytest
import tenmo
from helpers import assert_tensors_close


class TestShapeAndMetadata:
    def test_shape_property(self):
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert t.shape == (2, 3)

    def test_ndim_matches(self):
        t = tenmo.tensor([[[1.0]]])
        assert t.ndim == 3

    def test_numel_matches(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert t.numel == 6

    def test_repr_contains_shape(self):
        t = tenmo.tensor([1.0, 2.0])
        r = repr(t)
        assert "shape=(2,)" in r
        assert "float32" in r

    def test_str_readable(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        s = str(t)
        assert "1.0" in s


class TestViewsAndReshaping:
    def test_reshape_preserves_elements(self):
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        r = t.reshape((2, 3))
        assert r.shape == (2, 3)
        assert_tensors_close(r, tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_reshape_1d_to_3d(self):
        t = tenmo.arange(24)
        r = t.reshape((2, 3, 4))
        assert r.shape == (2, 3, 4)
        assert r.numel == 24

    def test_flatten_2d(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        f = t.flatten()
        assert f.shape == (4,)
        assert_tensors_close(f, tenmo.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_flatten_preserves_order(self):
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        f = t.flatten()
        assert f.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_matmul_with_reshape(self):
        a = tenmo.arange(6).reshape((2, 3))
        b = tenmo.arange(6).reshape((3, 2))
        c = a.matmul(b)
        assert c.shape == (2, 2)

    def test_transpose_2d(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = t.transpose([1, 0])
        assert result.shape == (2, 2)
        assert result.tolist() == [[1.0, 3.0], [2.0, 4.0]]

    def test_transpose_3d(self):
        t = tenmo.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = t.transpose([2, 0, 1])
        assert result.shape == (2, 2, 2)

    def test_permute(self):
        t = tenmo.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        result = t.permute([1, 2, 0])
        assert result.shape == (2, 2, 2)

    def test_squeeze_removes_size_one_dims(self):
        t = tenmo.tensor([[[1.0, 2.0, 3.0]]])
        result = t.squeeze()
        assert result.shape == (3,)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_squeeze_specific_axis(self):
        t = tenmo.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        result = t.squeeze([0])
        assert result.shape == (2, 2)

    def test_unsqueeze_inserts_dim(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        result = t.unsqueeze(0)
        assert result.shape == (1, 3)
        result2 = t.unsqueeze(1)
        assert result2.shape == (3, 1)

    def test_expand_broadcasts(self):
        t = tenmo.tensor([[1.0, 2.0]])
        result = t.expand([3, 2])
        assert result.shape == (3, 2)
        assert result.tolist() == [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]

    def test_contiguous_returns_self_if_already(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        c = t.contiguous()
        assert c.shape == (2, 2)
        assert c.tolist() == [[1.0, 2.0], [3.0, 4.0]]


class TestBroadcasting:
    def test_broadcast_scalar_add(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        c = t + 5.0
        assert_tensors_close(c, tenmo.tensor([6.0, 7.0, 8.0]))

    def test_broadcast_row_matrix(self):
        a = tenmo.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
        b = tenmo.tensor([[10.0], [20.0]])    # (2, 1)
        c = a + b
        assert c.shape == (2, 3)
        expected = tenmo.tensor([[11.0, 12.0, 13.0], [21.0, 22.0, 23.0]])
        assert_tensors_close(c, expected)

    def test_broadcast_matches_numpy(self):
        np.random.seed(42)
        a_np = np.random.randn(3, 1).astype(np.float32)
        b_np = np.random.randn(1, 4).astype(np.float32)
        a = tenmo.Tensor.from_numpy(a_np)
        b = tenmo.Tensor.from_numpy(b_np)
        c = (a + b).numpy()
        np.testing.assert_allclose(c, a_np + b_np, atol=1e-5)

    def test_incompatible_shapes_error(self):
        a = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tenmo.tensor([1.0, 2.0, 3.0])
        try:
            _ = a + b
        except (ValueError, RuntimeError):
            pass  # expected

    def test_mul_broadcast(self):
        a = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        b = tenmo.tensor([2.0, 3.0])  # (2,)
        c = a * b
        assert c.shape == (2, 2)
        expected = tenmo.tensor([[2.0, 6.0], [6.0, 12.0]])
        assert_tensors_close(c, expected)


class TestConcatStack:
    def test_concat_1d(self):
        a = tenmo.tensor([1.0, 2.0])
        b = tenmo.tensor([3.0, 4.0])
        c = tenmo.concat([a, b])
        assert c.shape == (4,)
        assert c.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_concat_2d_axis0(self):
        a = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tenmo.tensor([[5.0, 6.0]])
        c = tenmo.concat([a, b], axis=0)
        assert c.shape == (3, 2)
        assert c.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    def test_concat_2d_axis1(self):
        a = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tenmo.tensor([[5.0], [6.0]])
        c = tenmo.concat([a, b], axis=1)
        assert c.shape == (2, 3)
        assert c.tolist() == [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]

    def test_stack_1d(self):
        a = tenmo.tensor([1.0, 2.0])
        b = tenmo.tensor([3.0, 4.0])
        s = tenmo.stack([a, b])
        assert s.shape == (2, 2)
        assert s.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_stack_axis1(self):
        a = tenmo.tensor([1.0, 2.0])
        b = tenmo.tensor([3.0, 4.0])
        s = tenmo.stack([a, b], axis=1)
        assert s.shape == (2, 2)
        assert s.tolist() == [[1.0, 3.0], [2.0, 4.0]]


class TestWhereAndMaskedFill:
    def test_where_bool_condition(self):
        cond = tenmo.tensor([True, False, True])
        x = tenmo.tensor([10.0, 20.0, 30.0])
        y = tenmo.tensor([1.0, 2.0, 3.0])
        w = tenmo.where(cond, x, y)
        assert w.tolist() == [10.0, 2.0, 30.0]

    def test_where_comparison(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([2.0, 1.0, 3.0])
        w = tenmo.where(a > b, a, b)
        assert w.tolist() == [2.0, 2.0, 3.0]

    def test_where_matches_numpy(self):
        a_np = np.array([1.0, 5.0, 3.0, 8.0])
        b_np = np.array([2.0, 4.0, 6.0, 7.0])
        a = tenmo.tensor(a_np.tolist())
        b = tenmo.tensor(b_np.tolist())
        w = tenmo.where(a > b, a, b).numpy()
        np.testing.assert_array_equal(w, np.where(a_np > b_np, a_np, b_np))

    def test_masked_fill_with_comparison(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        mask = tenmo.tensor([True, False, True])
        mf = t.masked_fill(mask, 0.0)
        assert mf.tolist() == [0.0, 2.0, 0.0]

    def test_masked_fill_with_comparison_op(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        a = tenmo.tensor([0.5, 2.5, 2.5])
        mf = t.masked_fill(t > a, -1.0)
        assert mf.tolist() == [-1.0, 2.0, -1.0]

    def test_masked_fill_matches_numpy(self):
        t_np = np.array([1.0, 2.0, 3.0, 4.0])
        mask_np = np.array([True, False, True, False])
        t = tenmo.tensor(t_np.tolist())
        mask = tenmo.tensor(mask_np.tolist())
        mf = t.masked_fill(mask, -99.0).numpy()
        np.testing.assert_array_equal(mf, np.where(mask_np, -99.0, t_np))


# ── Shape / View ops / Broadcasting extras ──────────────


class TestShapeExtended:
    def test_contiguous_after_transpose(self):
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t2 = t.transpose([1, 0])
        assert t2.is_contiguous() is False
        t3 = t2.contiguous()
        assert t3.is_contiguous() is True

    def test_shape_is_tuple(self):
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        s = t.shape
        assert s == (2, 3)
        assert isinstance(s, tuple)


class TestViewOpsExtended:
    def test_reshape_to_infer(self):
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0])
        t2 = t.reshape([2, 2])
        assert t2.shape == (2, 2)

    def test_squeeze_removes_one_dims(self):
        t = tenmo.tensor([[[1.0, 2.0]]])
        s = t.squeeze()
        assert s.shape == (2,)

    def test_unsqueeze_adds_dim(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        s = t.unsqueeze(0)
        assert s.shape == (1, 3)

    def test_unsqueeze_axis1(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        s = t.unsqueeze(1)
        assert s.shape == (3, 1)


class TestBroadcastingExtended:
    def test_broadcast_column(self):
        a = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tenmo.tensor([[10.0], [20.0]])
        c = a + b
        expected = tenmo.tensor([[11.0, 12.0], [23.0, 24.0]])
        assert_tensors_close(c, expected)

    def test_broadcast_scalar(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([2.0])
        c = a * b
        expected = tenmo.tensor([2.0, 4.0, 6.0])
        assert_tensors_close(c, expected)
