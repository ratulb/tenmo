"""Pytest configuration for tenmo Python binding tests.

Injects the python-binding directory into sys.path so that ``import tenmo``
and ``import _tenmo`` resolve to the local build, then exposes session-wide
fixtures used across the test suite.
"""
from __future__ import annotations

import os
import sys
import pytest

# ── Path injection ────────────────────────────────────────────────────
# The _tenmo.so shared library and tenmo.py wrapper live in python-binding/.
_binding_dir = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "python-binding")
)
if _binding_dir not in sys.path:
    sys.path.insert(0, _binding_dir)

# Verify the shared library is loadable (fail fast with a clear message)
try:
    import _tenmo  # noqa: F401
except ImportError as exc:
    raise ImportError(
        f"Cannot import _tenmo from {_binding_dir}. "
        "Run `scripts/run_python_tests.sh` or build manually first: "
        "pixi run mojo build python-binding/tenmo_bind.mojo -I . "
        "--emit shared-lib -o python-binding/_tenmo.so"
    ) from exc

import numpy as np
import tenmo  # noqa: E402  (after path injection)


# ── Constants ─────────────────────────────────────────────────────────

ALL_DTYPES = ["float32", "float64", "int64", "int32", "int8", "uint8", "bool"]

FLOAT_DTYPES = ["float32", "float64"]
INT_DTYPES = ["int64", "int32", "int8", "uint8"]

SHAPES_1D = [(0,), (1,), (5,), (100,)]
SHAPES_2D = [(1, 1), (2, 3), (5, 1), (1, 5)]
SHAPES_ND = [(2, 3, 4), (1, 1, 5), (2, 3, 4, 5)]


# ── Helpers exposed as fixtures ───────────────────────────────────────

@pytest.fixture(params=ALL_DTYPES, ids=ALL_DTYPES)
def dtype_name(request) -> str:
    """Parametrize over all supported dtype strings."""
    return request.param


@pytest.fixture(params=FLOAT_DTYPES, ids=FLOAT_DTYPES)
def float_dtype_name(request) -> str:
    """Parametrize over floating-point dtype strings only."""
    return request.param


@pytest.fixture(params=INT_DTYPES, ids=INT_DTYPES)
def int_dtype_name(request) -> str:
    """Parametrize over integer dtype strings only."""
    return request.param


# ── Pre-built tensor fixtures ─────────────────────────────────────────

@pytest.fixture
def f32_vec():
    """1-D float32 tensor [1.0, 2.0, 3.0]."""
    return tenmo.tensor([1.0, 2.0, 3.0])


@pytest.fixture
def f32_2d():
    """2x2 float32 tensor [[1.0, 2.0], [3.0, 4.0]]."""
    return tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def f32_3d():
    """2x2x2 float32 tensor."""
    return tenmo.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])


@pytest.fixture
def i64_vec():
    """1-D int64 tensor [0, 1, 2, 3, 4]."""
    return tenmo.tensor([0, 1, 2, 3, 4], dtype="int64")


@pytest.fixture
def bool_vec():
    """1-D bool tensor [True, False, True]."""
    return tenmo.tensor([True, False, True])


@pytest.fixture
def f32_grad():
    """1-D float32 tensor with requires_grad=True."""
    return tenmo.tensor([1.0, 2.0, 3.0], requires_grad=True)


@pytest.fixture
def f32_2d_grad():
    """2x2 float32 tensor with requires_grad=True."""
    return tenmo.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
