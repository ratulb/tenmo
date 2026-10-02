"""Tests for the extra loss classes: MSELoss, BCELoss, BCEWithLogitsLoss.

These cover the extra loss classes bound alongside them.
Numeric references are computed directly in numpy so the tests are
independent of the library's own Tensor.mse/bce methods.
"""

from __future__ import annotations

import numpy as np
import pytest

import tenmo


class TestMSELoss:
    def test_known_value(self) -> None:
        a = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tenmo.tensor([[1.0, 0.0], [3.0, 4.0]])
        result = tenmo.MSELoss()(a, b)
        expected = float(np.mean((np.array([[1, 2], [3, 4]]) - np.array([[1, 0], [3, 4]])) ** 2))
        assert abs(result.numpy()[()] - expected) < 1e-5

    def test_perfect_match(self) -> None:
        a = tenmo.tensor([1.0, 2.0, 3.0])
        result = tenmo.MSELoss()(a, a)
        assert abs(result.numpy()[()]) < 1e-6

    def test_grad_matches_closed_form(self) -> None:
        x = tenmo.tensor([0.5])
        x.requires_grad_(True)
        loss = tenmo.MSELoss()(x, tenmo.tensor([1.0]))
        loss.backward()
        grad = x.grad.numpy()[0]
        expected = 2.0 * (0.5 - 1.0)
        assert abs(grad - expected) < 1e-5

    def test_repr_and_noops(self) -> None:
        loss = tenmo.MSELoss()
        assert repr(loss) == "MSELoss()"
        assert loss.train() is loss
        assert loss.eval() is loss


class TestBCELoss:
    def test_known_value(self) -> None:
        pred = np.array([0.2, 0.8], dtype=np.float32)
        target = np.array([1.0, 0.0], dtype=np.float32)
        result = tenmo.BCELoss()(
            tenmo.tensor(pred), tenmo.tensor(target)
        )
        expected = -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
        assert abs(result.numpy()[()] - expected) < 1e-5

    def test_perfect_pred(self) -> None:
        result = tenmo.BCELoss()(
            tenmo.tensor([0.99, 0.01]), tenmo.tensor([1.0, 0.0])
        )
        assert result.numpy()[()] < 0.1

    def test_grad_matches_closed_form(self) -> None:
        p = tenmo.tensor([0.5])
        p.requires_grad_(True)
        loss = tenmo.BCELoss()(p, tenmo.tensor([1.0]))
        loss.backward()
        assert abs(p.grad.numpy()[0] - (-1.0 / 0.5)) < 1e-5

    def test_eval_mode_disables_tracking(self) -> None:
        p = tenmo.tensor([0.5])
        p.requires_grad_(True)
        loss = tenmo.BCELoss().eval()(p, tenmo.tensor([0.0]))
        assert not loss.requires_grad


class TestBCEWithLogitsLoss:
    def test_known_value_fused(self) -> None:
        logits = np.array([1.5, -0.5], dtype=np.float32)
        sig = 1.0 / (1.0 + np.exp(-logits))
        target = np.array([1.0, 0.0], dtype=np.float32)
        result = tenmo.BCEWithLogitsLoss()(
            tenmo.tensor(logits), tenmo.tensor(target)
        )
        expected = -np.mean(target * np.log(sig) + (1 - target) * np.log(1 - sig))
        assert abs(result.numpy()[()] - expected) < 1e-5

    def test_zero_logits_loss_positive(self) -> None:
        result = tenmo.BCEWithLogitsLoss()(
            tenmo.tensor([0.0, 0.0]), tenmo.tensor([0.0, 0.0])
        )
        assert abs(result.numpy()[()] - np.log(2.0)) < 1e-5

    def test_grad_matches_sigmoid_formula(self) -> None:
        logits = tenmo.tensor([0.0])
        logits.requires_grad_(True)
        loss = tenmo.BCEWithLogitsLoss()(logits, tenmo.tensor([1.0]))
        loss.backward()
        sig = 1.0 / (1.0 + np.exp(0.0))
        assert abs(logits.grad.numpy()[0] - (sig - 1.0)) < 1e-5

    def test_train_eval(self) -> None:
        loss = tenmo.BCEWithLogitsLoss()
        assert repr(loss) == "BCEWithLogitsLoss()"
        loss.eval()
        assert repr(loss).startswith("BCEWithLogitsLoss")
        loss.train()
        assert repr(loss).startswith("BCEWithLogitsLoss")