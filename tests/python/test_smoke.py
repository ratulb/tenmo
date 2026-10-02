"""Smoke tests — verify the test framework, _tenmo.so import, and basic ops."""
from __future__ import annotations

import numpy as np
import tenmo
from conftest import ALL_DTYPES
from helpers import assert_tensors_close, make_tensor


# ── Import / construction smoke tests ────────────────────────────────

class TestSmokeImport:
    def test_tensor_from_list(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.numel == 3

    def test_tensor_from_nested_list(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.shape == (2, 2)
        assert t.ndim == 2

    def test_zeros(self):
        t = tenmo.zeros((2, 3))
        assert t.shape == (2, 3)
        assert_tensors_close(t, tenmo.zeros((2, 3)))

    def test_ones(self):
        t = tenmo.ones((3,))
        assert t.shape == (3,)
        assert_tensors_close(t, tenmo.ones((3,)))

    def test_arange(self):
        t = tenmo.arange(5)
        assert t.shape == (5,)
        assert_tensors_close(t, tenmo.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))

    def test_randn(self):
        t = tenmo.randn((100,))
        assert t.shape == (100,)
        assert abs(float(t.mean().item())) < 0.3

    def test_dtype_preservation(self):
        t32 = tenmo.tensor([1.0], dtype="float32")
        t64 = tenmo.tensor([1.0], dtype="float64")
        ti64 = tenmo.tensor([1], dtype="int64")
        assert np.dtype(t32.numpy_dtype()).name == "float32"
        assert np.dtype(t64.numpy_dtype()).name == "float64"
        assert np.dtype(ti64.numpy_dtype()).name == "int64"

    def test_item(self):
        t = tenmo.tensor([42.0])
        assert t.item() == 42.0

    def test_tolist(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.tolist() == [[1.0, 2.0], [3.0, 4.0]]


# ── Arithmetic smoke tests ───────────────────────────────────────────

class TestSmokeArithmetic:
    def test_add_tensor(self):
        a = tenmo.tensor([1.0, 2.0, 3.0])
        b = tenmo.tensor([4.0, 5.0, 6.0])
        c = a + b
        assert_tensors_close(c, tenmo.tensor([5.0, 7.0, 9.0]))

    def test_add_scalar(self):
        t = tenmo.tensor([1.0, 2.0])
        c = t + 3.0
        assert_tensors_close(c, tenmo.tensor([4.0, 5.0]))

    def test_mul_tensor(self):
        a = tenmo.tensor([2.0, 3.0])
        b = tenmo.tensor([4.0, 5.0])
        c = a * b
        assert_tensors_close(c, tenmo.tensor([8.0, 15.0]))

    def test_neg(self):
        t = tenmo.tensor([1.0, -2.0])
        c = -t
        assert_tensors_close(c, tenmo.tensor([-1.0, 2.0]))

    def test_matmul(self):
        a = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tenmo.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = a.matmul(b)
        # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        expected = tenmo.tensor([[19.0, 22.0], [43.0, 50.0]])
        assert_tensors_close(c, expected)


# ── Reduction smoke tests ────────────────────────────────────────────

class TestSmokeReductions:
    def test_sum(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        assert t.sum().item() == 6.0

    def test_mean(self):
        t = tenmo.tensor([1.0, 2.0, 3.0, 4.0])
        assert t.mean().item() == 2.5

    def test_max(self):
        t = tenmo.tensor([3.0, 1.0, 2.0])
        assert t.max().item() == 3.0

    def test_argmax(self):
        t = tenmo.tensor([1.0, 3.0, 2.0])
        assert t.argmax() == 1


# ── Autograd smoke tests ────────────────────────────────────────────

class TestSmokeAutograd:
    def test_basic_backward(self):
        x = tenmo.tensor([2.0, 3.0], requires_grad=True)
        y = (x * x).sum()
        y.backward()
        assert_tensors_close(x.grad, tenmo.tensor([4.0, 6.0]))

    def test_grad_zero(self):
        x = tenmo.tensor([1.0, 2.0], requires_grad=True)
        y = (x + x).sum()
        y.backward()
        x.zero_grad()
        assert x.grad is None or all(
            v == 0.0 for v in x.grad.tolist()
        )


# ── Layer / training smoke tests ─────────────────────────────────────

class TestSmokeLayers:
    def test_linear_forward(self):
        layer = tenmo.Linear(3, 2)
        x = tenmo.tensor([[1.0, 2.0, 3.0]])
        model = tenmo.Sequential([layer.into()])
        y = model(x)
        assert y.shape == (1, 2)

    def test_sequential_forward(self):
        model = tenmo.Sequential([
            tenmo.Linear(4, 8).into(),
            tenmo.ReLU().into(),
            tenmo.Linear(8, 2).into(),
        ])
        x = tenmo.tensor([[1.0, 2.0, 3.0, 4.0]])
        y = model(x)
        assert y.shape == (1, 2)

    def test_cross_entropy(self):
        logits = tenmo.tensor([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]])
        target = tenmo.tensor([0, 1], dtype="int64")
        loss_fn = tenmo.CrossEntropyLoss()
        loss = loss_fn(logits, target)
        assert loss.item() > 0


# ── Numpy interop smoke tests ────────────────────────────────────────

class TestSmokeNumpy:
    def test_to_numpy(self):
        t = tenmo.tensor([1.0, 2.0, 3.0])
        arr = t.numpy()
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_from_numpy(self):
        arr = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        t = tenmo.Tensor.from_numpy(arr)
        np.testing.assert_allclose(t.numpy(), arr)

    def test_numpy_roundtrip(self):
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = tenmo.Tensor.from_numpy(original)
        recovered = t.numpy()
        np.testing.assert_allclose(recovered, original)
