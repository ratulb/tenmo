"""Closure of the §32.18 "unimplemented" audit stubs.

Each test uses the exact stub name from `tests/test_python_bindings.txt` so the
audit matches by name. Stubs with an already-covered equivalent live in the
themed files; stubs whose surface does not exist were moved to the "blocked"
bucket in §32.18.
"""
from __future__ import annotations

import copy
import gc
import math
import sys

import numpy as np
import pytest

import tenmo


# ── §1 Construction ─────────────────────────────────────────────────


class TestConstructionClosure:
    def test_random_normal_statistical_sanity(self):
        vals = tenmo.randn((20000,)).numpy()
        assert abs(float(vals.mean())) < 0.05
        assert 0.95 < float(vals.std()) < 1.05
        assert float(vals.min()) > -6.0
        assert float(vals.max()) < 6.0


# ── §2 Dtype & Casting ─────────────────────────────────────────────


class TestDtypeClosure:
    DTYPES = [
        "float16", "float32", "float64",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64", "bool",
    ]

    def test_to_dtype_all_pairwise_conversions(self):
        pair = np.array([1, 100], dtype=np.float64)
        for src in self.DTYPES:
            s = tenmo.Tensor.from_numpy(pair.astype(np.dtype(src)))
            for dst in self.DTYPES:
                r = s.to_dtype(dst)
                assert np.dtype(r.numpy_dtype()).name == dst
                expected = pair.astype(np.dtype(src)).astype(np.dtype(dst)).tolist()
                assert r.tolist() == expected

    def test_to_dtype_widening_float_is_lossless(self):
        x = tenmo.tensor(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        w = x.to_dtype("float64")
        assert np.dtype(w.numpy_dtype()).name == "float64"
        expected = np.array([0.1, 0.2, 0.3], dtype=np.float32).astype(np.float64).tolist()
        assert w.tolist() == expected


# ── §3 Shape & Metadata ─────────────────────────────────────────────


class TestShapeClosure:
    def test_ndim_matches_number_of_dimensions(self):
        for shape in [(2,), (2, 3), (2, 3, 4)]:
            assert tenmo.zeros(shape).ndim == len(shape)

    def test_itemsize_matches_dtype_byte_width(self):
        for dt, width in [("float32", 4), ("float64", 8),
                          ("int64", 8), ("int32", 4),
                          ("uint8", 1), ("bool", 1)]:
            t = tenmo.Tensor.from_numpy(np.array([1], dtype=np.dtype(dt)))
            assert t.itemsize == width

    def test_nbytes_matches_numel_times_itemsize(self):
        t = tenmo.zeros((3, 4))
        assert t.numel * t.itemsize == 48
        assert t.nbytes == 48


# ── §5 Views & Reshaping ────────────────────────────────────────────


class TestViewsClosure:
    def test_ravel_matches_flatten_semantics(self):
        base = tenmo.tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
        r = base.ravel()
        assert r.shape == (6,)
        assert r.tolist() == base.flatten().tolist()

    def test_swapaxes_matches_expected_layout(self):
        t = tenmo.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        s = t.swapaxes(0, 2)
        assert s.shape == (4, 3, 2)
        np.testing.assert_allclose(s.numpy(), t.numpy().swapaxes(0, 2))


# ── §8 Comparison Operators ─────────────────────────────────────────


class TestComparisonsClosure:
    def test_comparison_broadcasts_correctly(self):
        a = tenmo.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert (a > 1.5).tolist() == [False, True, True]
        assert (a < 2.0).tolist() == [True, False, False]
        assert (a >= 2.0).tolist() == [False, True, True]
        assert (a <= 1.0).tolist() == [True, False, False]
        assert (a > 0.0).tolist() == [True, True, True]
        b = tenmo.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        assert (a < b).tolist() == [False, False, False]
        assert (a <= b).tolist() == [True, True, True]

    def test_allclose_respects_rtol_and_atol(self):
        x = tenmo.tensor(np.array([1.0, 2.0], dtype=np.float32))
        y = tenmo.tensor(np.array([1.001, 2.001], dtype=np.float32))
        assert x.allclose(y, rtol=0.1)
        assert not x.allclose(y, rtol=1e-6)
        z = tenmo.tensor(np.array([1.0, 2.0000005], dtype=np.float32))  # ~4.8e-7 above 2.0 in f32
        assert x.allclose(z, rtol=0.0, atol=1e-6)
        assert not x.allclose(z, rtol=0.0, atol=1e-9)


# ── §10 Unary Math Functions ────────────────────────────────────────


class TestUnaryMathClosure:
    def test_clamp_clips_values_to_bounds(self):
        t = tenmo.tensor(np.array([0.1, 1.0, 9.0], dtype=np.float32))
        c = t.clip(0.5, 1.5)
        assert c.tolist() == [0.5, 1.0, 1.5]


# ── §12 Reduction Operations ────────────────────────────────────────


class TestReductionsClosure:
    def test_var_respects_ddof_parameter(self):
        v = tenmo.tensor(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert abs(v.variance().item() - np.var([1, 2, 3, 4], ddof=1)) < 1e-4
        assert abs(v.variance(unbiased=False).item() - np.var([1, 2, 3, 4], ddof=0)) < 1e-4


# ── §13 Linear Algebra ──────────────────────────────────────────────


class TestLinalgClosure:
    def test_matmul_batched_nd(self):
        np.random.seed(0)
        a_np = np.random.randn(2, 3, 4).astype(np.float32)
        b_np = np.random.randn(2, 4, 5).astype(np.float32)
        c = tenmo.Tensor.from_numpy(a_np).matmul(tenmo.Tensor.from_numpy(b_np))
        assert c.shape == (2, 3, 5)
        np.testing.assert_allclose(c.numpy(), a_np @ b_np, atol=1e-3)


# ── §14 Autograd Core Mechanics ─────────────────────────────────────


class TestAutogradCoreClosure:
    def test_backward_reruns_without_retain_flag(self):
        # Documented behavior: backward is re-runnable without any retain
        # flag; the graph stays alive until the tensors are dropped, and
        # repeated calls accumulate into leaf grads.
        x = tenmo.tensor([1.0, 2.0])
        x.requires_grad_(True)
        y = x * 2.0
        y.sum().backward()
        assert x.grad.tolist() == [2.0, 2.0]
        y.sum().backward()
        assert x.grad.tolist() == [4.0, 4.0]


# ── §16 Autograd Graph Edge Cases ───────────────────────────────────


class TestAutogradGraphClosure:
    def test_backward_on_disconnected_subgraph_raises_or_noops_as_documented(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        a.requires_grad_(True)
        b = tenmo.tensor([4.0, 5.0, 6.0])
        b.requires_grad_(True)
        a.sum().backward()
        assert a.grad.tolist() == [1.0, 1.0, 1.0]
        assert b.grad.tolist() == [0.0, 0.0, 0.0]


# ── §17 Memory & Reference Counting ─────────────────────────────────


class TestMemoryClosure:
    def test_view_keeps_base_tensor_alive(self):
        base = tenmo.tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
        view = base.transpose([1, 0])
        del base
        gc.collect()
        np.testing.assert_allclose(
            view.numpy(), np.arange(6, dtype=np.float32).reshape(2, 3).T
        )


# ── §19 NumPy Interop ───────────────────────────────────────────────


class TestNumpyInteropClosure:
    @pytest.mark.skipif(sys.version_info < (3, 12), reason="__buffer__ requires Python 3.12+")
    def test_buffer_protocol_supports_memoryview(self):
        t = tenmo.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        mv = memoryview(t)
        assert mv.tolist() == [1.0, 2.0, 3.0]


# ── §20 Python Protocol Compliance ──────────────────────────────────


class TestPythonProtocolClosure:
    def test_hash_raises_or_is_explicitly_unsupported(self):
        t = tenmo.tensor([1.0])
        with pytest.raises(TypeError):
            hash(t)
        with pytest.raises(TypeError):
            {t}

    def test_deepcopy_produces_independent_tensor(self):
        x = tenmo.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        y = copy.deepcopy(x)
        assert y.tolist() == x.tolist()
        y += 20.0
        assert y.tolist() == [21.0, 22.0, 23.0]
        assert x.tolist() == [1.0, 2.0, 3.0]

    def test_context_manager_protocol_if_applicable(self):
        # "if applicable": tensors are not context managers.
        t = tenmo.tensor([1.0])
        with pytest.raises(AttributeError):
            t.__enter__()


# ── §22 Error Handling & Edge Cases ─────────────────────────────────


class TestErrorHandlingClosure:
    def test_overflow_behavior_matches_dtype_semantics(self):
        x = tenmo.tensor([1e38])
        r = x * 10.0
        assert math.isinf(float(r.item()))
        assert np.float32(1e38) * 10 == math.inf

    def test_underflow_behavior_matches_dtype_semantics(self):
        x = tenmo.tensor([1e-45])
        r = x * 0.1
        assert float(r.item()) == 0.0
        assert np.float32(1e-45) * np.float32(0.1) == 0.0