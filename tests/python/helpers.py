"""Shared test utilities for tenmo Python binding tests."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import numpy as np
import tenmo


# ── Tensor comparison ─────────────────────────────────────────────────

def assert_tensors_close(
    a: tenmo.Tensor,
    b: Union[tenmo.Tensor, float, int],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    msg: str = "",
) -> None:
    """Assert two tensors are elementwise close within tolerance.

    Handles Tensor-vs-Tensor, Tensor-vs-scalar, and Tensor-vs-numpy-array.
    Uses numpy's ``allclose`` under the hood.
    """
    if isinstance(b, tenmo.Tensor):
        a_np = a.numpy()
        b_np = b.numpy()
    elif isinstance(b, (float, int)):
        a_np = a.numpy()
        b_np = np.asarray(b, dtype=a_np.dtype)
    elif isinstance(b, np.ndarray):
        a_np = a.numpy()
        b_np = b
    else:
        raise TypeError(f"unsupported comparison type: {type(b)}")

    if a_np.shape != b_np.shape:
        raise AssertionError(
            f"shape mismatch: {a_np.shape} vs {b_np.shape} {msg}"
        )

    if not np.allclose(a_np, b_np, atol=atol, rtol=rtol):
        diff = np.abs(a_np - b_np)
        max_diff = float(diff.max())
        raise AssertionError(
            f"tensors not close (max diff={max_diff:.2e}, "
            f"atol={atol}, rtol={rtol}) {msg}\n"
            f"  a={a_np}\n  b={b_np}"
        )


def assert_tensors_equal(
    a: tenmo.Tensor,
    b: Union[tenmo.Tensor, float, int],
    *,
    msg: str = "",
) -> None:
    """Assert two tensors are exactly equal (no tolerance)."""
    if isinstance(b, tenmo.Tensor):
        a_np = a.numpy()
        b_np = b.numpy()
    else:
        a_np = a.numpy()
        b_np = np.asarray(b, dtype=a_np.dtype)

    if a_np.shape != b_np.shape:
        raise AssertionError(
            f"shape mismatch: {a_np.shape} vs {b_np.shape} {msg}"
        )

    if not np.array_equal(a_np, b_np):
        raise AssertionError(
            f"tensors not equal {msg}\n  a={a_np}\n  b={b_np}"
        )


# ── Gradient checking ────────────────────────────────────────────────

def numeric_grad_check(
    fn,
    x: tenmo.Tensor,
    *,
    eps: float = 1e-4,
    atol: float = 1e-3,
    msg: str = "",
) -> None:
    """Finite-difference gradient check against autograd.

    ``fn`` must be ``def (Tensor) -> Tensor`` and operate elementwise.
    Verifies that ``autograd.grad(fn(x))`` matches finite differences.
    """
    x_np = x.numpy()
    flat = x_np.flatten()
    grad_np = np.zeros_like(flat)

    for i in range(len(flat)):
        orig = flat[i]

        flat[i] = orig + eps
        plus = fn(tenmo.tensor(flat.reshape(x_np.shape))).numpy().sum()

        flat[i] = orig - eps
        minus = fn(tenmo.tensor(flat.reshape(x_np.shape))).numpy().sum()

        grad_np[i] = (plus - minus) / (2 * eps)
        flat[i] = orig  # restore

    expected = tenmo.tensor(grad_np.reshape(x_np.shape))

    x_plus = tenmo.tensor(x_np, requires_grad=True)
    result = fn(x_plus)
    result.backward()
    actual = x_plus.grad

    assert_tensors_close(actual, expected, atol=atol, msg=msg)


# ── Common tensor factories ──────────────────────────────────────────

def make_tensor(
    data: Union[list, np.ndarray],
    *,
    dtype: Optional[str] = None,
    requires_grad: bool = False,
) -> tenmo.Tensor:
    """Shorthand for tenmo.tensor with consistent defaults."""
    return tenmo.tensor(data, dtype=dtype, requires_grad=requires_grad)


def ones(shape: Sequence[int], **kwargs) -> tenmo.Tensor:
    """Create a ones tensor."""
    return tenmo.ones(shape, **kwargs)


def zeros(shape: Sequence[int], **kwargs) -> tenmo.Tensor:
    """Create a zeros tensor."""
    return tenmo.zeros(shape, **kwargs)


# ── Numerical stability helpers ──────────────────────────────────────

def safe_log(x: float, min_val: float = 1e-7) -> float:
    """Log with floor to avoid log(0)."""
    return math.log(max(x, min_val))


def softmax_ref(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Reference softmax implementation (for parity checks)."""
    e_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def cross_entropy_ref(
    logits: np.ndarray, target: np.ndarray
) -> float:
    """Reference cross-entropy loss (for parity checks)."""
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
    n = target.shape[0]
    return -np.sum(log_probs[np.arange(n), target]) / n
