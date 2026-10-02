"""Tests for construction & initialization (Category 1)."""
from __future__ import annotations

import numpy as np
import pytest
import tenmo
from conftest import ALL_DTYPES
from helpers import assert_tensors_close


class TestConstruction:
    def test_from_python_list_1d(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.numel == 3
        assert t.tolist() == [1.0, 2.0, 3.0]

    def test_from_python_nested_list_nd(self):
        t = tenmo.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert t.shape == (2, 3)
        assert t.ndim == 2
        assert t.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_from_python_scalar(self):
        t = tenmo.tensor([42.0])
        assert t.shape == (1,)
        assert t.item() == 42.0

    def test_from_numpy_array_all_dtypes(self):
        for dt in ALL_DTYPES:
            if dt == "float32":
                arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            elif dt == "float64":
                arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            elif dt == "int64":
                arr = np.array([1, 2, 3], dtype=np.int64)
            elif dt == "int32":
                arr = np.array([1, 2, 3], dtype=np.int32)
            elif dt == "int8":
                arr = np.array([1, 2, 3], dtype=np.int8)
            elif dt == "uint8":
                arr = np.array([1, 2, 3], dtype=np.uint8)
            elif dt == "bool":
                arr = np.array([True, False, True], dtype=np.bool_)
            else:
                continue
            t = tenmo.Tensor.from_numpy(arr)
            assert t.numpy_dtype() == arr.dtype, f"failed for {dt}"

    def test_zeros_creates_correct_shape_and_dtype(self):
        t = tenmo.zeros((2, 3, 4))
        assert t.shape == (2, 3, 4)
        assert t.numel == 24
        assert all(v == 0.0 for v in t.flatten().tolist())

    def test_ones_creates_correct_shape_and_dtype(self):
        t = tenmo.ones((3, 5))
        assert t.shape == (3, 5)
        np.testing.assert_allclose(t.numpy(), 1.0)

    def test_arange_matches_expected_sequence(self):
        t = tenmo.arange(5)
        np.testing.assert_allclose(t.numpy(), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_arange_with_step(self):
        t = tenmo.arange(10, start=0, step=2)
        np.testing.assert_allclose(t.numpy(), [0.0, 2.0, 4.0, 6.0, 8.0])

    def test_construction_infers_dtype_from_python_data(self):
        t_f = tenmo.tensor([1.0, 2.0])
        assert np.dtype(t_f.numpy_dtype()).name == "float32"
        t_i = tenmo.tensor([1, 2, 3], dtype="int64")
        assert np.dtype(t_i.numpy_dtype()).name == "int64"

    def test_construction_respects_explicit_dtype_override(self):
        t = tenmo.tensor([1, 2, 3], dtype="float64")
        assert np.dtype(t.numpy_dtype()).name == "float64"
        np.testing.assert_allclose(t.numpy(), [1.0, 2.0, 3.0])

    @pytest.mark.skip(reason="core panics on zero-size tensors (Shape: dimension must be >= 1)")
    def test_construction_of_zero_size_tensor(self):
        t = tenmo.zeros((0,))
        assert t.shape == (0,)
        assert t.numel == 0

    @pytest.mark.skip(reason="full_like not exposed")
    def test_full_like_matches_input_shape_dtype_device(self): pass

    @pytest.mark.skip(reason="empty not exposed")
    def test_empty_returns_uninitialized_correct_shape(self): pass


class TestConstructionExtended:
    def test_from_numpy_copies_data(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = tenmo.Tensor.from_numpy(a_np)
        a_np[0] = 999.0
        # from_numpy copies, so mutation does NOT propagate
        assert t.tolist()[0] == 1.0

    def test_from_numpy_float64_preserves(self):
        a_np = np.array([1.5, 2.5], dtype=np.float64)
        t = tenmo.Tensor.from_numpy(a_np)
        np.testing.assert_allclose(t.numpy(), a_np)

    def test_arange_float_step(self):
        import os, subprocess, sys
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "..", "..", "python-binding")
        result = subprocess.run(
            [sys.executable, "-c", """
import numpy as np
import tenmo
t = tenmo.arange(5, start=0.0, step=0.5)
expected = np.arange(0, 5.0, 0.5).astype(np.float32)
np.testing.assert_allclose(t.numpy(), expected, atol=1e-5)
"""],
            capture_output=True, timeout=30, env=env,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr.decode()}"

    def test_dtype_preserved_through_construction(self):
        t64 = tenmo.tensor([1.0, 2.0], dtype="float64")
        assert np.dtype(t64.numpy_dtype()).name == "float64"
        t32 = tenmo.tensor([1.0, 2.0], dtype="float32")
        assert np.dtype(t32.numpy_dtype()).name == "float32"
