"""Tests for numpy interop (Category 19)."""
from __future__ import annotations

import numpy as np
import pytest
import tenmo


class TestNumpyInterop:
    def test_to_numpy_matches_values(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        arr = t.numpy()
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_to_numpy_matches_dtype(self):
        t = tenmo.tensor([1.0, 2.0], dtype="float32")
        arr = t.numpy()
        assert arr.dtype == np.float32

    def test_to_numpy_matches_shape(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        arr = t.numpy()
        assert arr.shape == (2, 2)

    def test_from_numpy_round_trip_preserves_values(self):
        original = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        t = tenmo.Tensor.from_numpy(original)
        np.testing.assert_allclose(t.numpy(), original)

    def test_from_numpy_all_supported_dtypes(self):
        dtypes = [
            (np.float32, [1.0, 2.0]),
            (np.float64, [1.0, 2.0]),
            (np.int64, [1, 2]),
            (np.int32, [1, 2]),
            (np.int8, [1, 2]),
            (np.uint8, [1, 2]),
            (np.bool_, [True, False]),
        ]
        for dt, data in dtypes:
            arr = np.array(data, dtype=dt)
            t = tenmo.Tensor.from_numpy(arr)
            assert t.numpy_dtype() == arr.dtype, f"roundtrip failed for {dt}"

    def test_2d_numpy_round_trip(self):
        original = np.random.randn(3, 4).astype(np.float32)
        t = tenmo.Tensor.from_numpy(original)
        np.testing.assert_allclose(t.numpy(), original)


class TestNumpyInteropExtended:
    def test_numpy_returns_float32_array(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        arr = t.numpy()
        assert arr.dtype == np.float32
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_from_numpy_large(self):
        a = np.random.randn(100, 50).astype(np.float32)
        t = tenmo.Tensor.from_numpy(a)
        np.testing.assert_allclose(t.numpy(), a, atol=1e-5)

    def test_numpy_strided_view_correct_values(self):
        # Transposed (non-contiguous) source takes the owned-copy leg.
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tp = t.transpose([1, 0])
        arr = tp.numpy()
        assert arr.shape == (3, 2)
        np.testing.assert_allclose(
            arr, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        )

    def test_numpy_offset_view_correct_values(self):
        # Dense-strided slice with nonzero storage offset.
        t = tenmo.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        s = t[2:]
        arr = s.numpy()
        np.testing.assert_allclose(arr, [2.0, 3.0, 4.0, 5.0])

    def test_numpy_is_a_copy(self):
        t = tenmo.tensor([1.0, 2.0])
        arr = t.numpy()
        arr[0] = 99.0
        assert t.tolist() == [1.0, 2.0]

    def test_numpy_all_dtypes(self):
        cases = [
            ("float16", np.float16, [1.0, 2.0]),
            ("float32", np.float32, [1.0, 2.0]),
            ("float64", np.float64, [1.0, 2.0]),
            ("int8", np.int8, [1, 2]),
            ("int16", np.int16, [1, 2]),
            ("int32", np.int32, [1, 2]),
            ("int64", np.int64, [1, 2]),
            ("uint8", np.uint8, [1, 2]),
            ("uint16", np.uint16, [1, 2]),
            ("uint32", np.uint32, [1, 2]),
            ("uint64", np.uint64, [1, 2]),
            ("bool", np.bool_, [True, False]),
        ]
        for dtype_name, np_dt, data in cases:
            t = tenmo.tensor(data, dtype=dtype_name)
            arr = t.numpy()
            assert arr.dtype == np_dt, dtype_name
            np.testing.assert_array_equal(arr, np.array(data, dtype=np_dt))

    def test_list_construction_ragged_raises(self):
        # np.asarray rejects ragged nesting; the Mojo boundary surfaces it
        # as a catchable Exception (previously a process-aborting panic).
        with pytest.raises(Exception, match="inhomogeneous"):
            tenmo.tensor([[1.0, 2.0], [3.0]])
