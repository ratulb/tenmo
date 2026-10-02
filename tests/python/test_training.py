"""Tests for training-loop building blocks (Category 23)."""
from __future__ import annotations

import numpy as np
import pytest
import tenmo


class TestTrainingBuildingBlocks:
    def test_linear_forward(self):
        layer = tenmo.Linear(3, 2)
        model = tenmo.Sequential([layer.into()])
        x = tenmo.tensor([[1.0, 2.0, 3.0]])
        y = model(x)
        assert y.shape == (1, 2)

    def test_relu_forward(self):
        model = tenmo.Sequential([
            tenmo.Linear(3, 2).into(),
            tenmo.ReLU().into(),
        ])
        x = tenmo.tensor([[1.0, -1.0, 2.0]])
        y = model(x)
        assert y.shape == (1, 2)

    def test_sigmoid_forward(self):
        model = tenmo.Sequential([
            tenmo.Linear(3, 2).into(),
            tenmo.Sigmoid().into(),
        ])
        x = tenmo.tensor([[1.0, 2.0, 3.0]])
        y = model(x)
        assert y.shape == (1, 2)

    def test_tanh_forward(self):
        model = tenmo.Sequential([
            tenmo.Linear(3, 2).into(),
            tenmo.Tanh().into(),
        ])
        x = tenmo.tensor([[1.0, 2.0, 3.0]])
        y = model(x)
        assert y.shape == (1, 2)

    def test_flatten_forward(self):
        model = tenmo.Sequential([
            tenmo.Linear(3, 2).into(),
            tenmo.Flatten().into(),
        ])
        x = tenmo.tensor([[1.0, 2.0, 3.0]])
        y = model(x)
        assert y.shape == (1, 2)

    def test_cross_entropy_loss(self):
        logits = tenmo.tensor([[2.0, 1.0, 0.1], [0.1, 2.0, 1.0]])
        target = tenmo.tensor([0, 1], dtype="int64")
        loss_fn = tenmo.CrossEntropyLoss()
        loss = loss_fn(logits, target)
        assert loss.item() > 0

    def test_cross_entropy_reproducible(self):
        logits = tenmo.tensor([[2.0, 1.0, 0.1]])
        target = tenmo.tensor([0], dtype="int64")
        loss_fn = tenmo.CrossEntropyLoss()
        loss1 = loss_fn(logits, target)
        loss2 = loss_fn(logits, target)
        assert abs(loss1.item() - loss2.item()) < 1e-6

    def test_sgd_step(self):
        model = tenmo.Sequential([
            tenmo.Linear(4, 2).into(),
        ])
        opt = tenmo.SGD(model, lr=0.1)
        x = tenmo.tensor([[1.0, 2.0, 3.0, 4.0]])
        target = tenmo.tensor([1], dtype="int64")
        loss_fn = tenmo.CrossEntropyLoss()

        opt.zero_grad()
        y = model(x)
        loss = loss_fn(y, target)
        loss.backward()
        opt.step()
        # No crash = success

    def test_sgd_get_lr(self):
        model = tenmo.Sequential([tenmo.Linear(2, 2).into()])
        opt = tenmo.SGD(model, lr=0.05)
        assert abs(opt.get_lr() - 0.05) < 1e-6

    def test_sgd_set_lr(self):
        model = tenmo.Sequential([tenmo.Linear(2, 2).into()])
        opt = tenmo.SGD(model, lr=0.05)
        opt.set_lr(0.01)
        assert abs(opt.get_lr() - 0.01) < 1e-6

    def test_accuracy(self):
        pred = tenmo.tensor([[0.1, 0.9], [0.8, 0.2]])
        # Binding forces target to float32 internally (converts to i64)
        target = tenmo.tensor([1.0, 0.0])
        acc = tenmo.accuracy(pred, target)
        assert acc == 1.0

    def test_accuracy_partial(self):
        pred = tenmo.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        target = tenmo.tensor([1.0, 0.0, 1.0, 0.0])
        acc = tenmo.accuracy(pred, target)
        assert acc == 1.0

    def test_train_epoch(self):
        np.random.seed(42)
        model = tenmo.Sequential([
            tenmo.Linear(4, 8).into(),
            tenmo.ReLU().into(),
            tenmo.Linear(8, 2).into(),
        ])
        opt = tenmo.SGD(model, lr=0.01)
        loss_fn = tenmo.CrossEntropyLoss()

        features = np.random.randn(32, 4).astype(np.float32)
        labels = np.random.randint(0, 2, size=(32,)).astype(np.int64)

        avg_loss, acc = tenmo.train_epoch(
            model, loss_fn, opt, features, labels,
            batch_size=8, shuffle=False,
        )
        assert avg_loss > 0
        assert 0.0 <= acc <= 1.0

    def test_eval_epoch(self):
        np.random.seed(42)
        model = tenmo.Sequential([
            tenmo.Linear(4, 2).into(),
        ])
        loss_fn = tenmo.CrossEntropyLoss()

        features = np.random.randn(16, 4).astype(np.float32)
        labels = np.random.randint(0, 2, size=(16,)).astype(np.int64)

        avg_loss, acc = tenmo.eval_epoch(
            model, loss_fn, features, labels, batch_size=8,
        )
        assert avg_loss > 0
        assert 0.0 <= acc <= 1.0

    @pytest.mark.skip(reason="pending Phase 2 — embedding not exposed")
    def test_embedding_lookup_returns_correct_rows(self): pass

    @pytest.mark.skip(reason="pending Phase 2 — multinomial not exposed")
    def test_multinomial_sampling_reproducible_with_fixed_seed(self): pass


class TestTrainingExtended:
    def test_accuracy_half_wrong(self):
        pred = tenmo.tensor([[0.9, 0.1], [0.8, 0.2]])
        target = tenmo.tensor([0.0, 1.0])
        acc = tenmo.accuracy(pred, target)
        # argmax = [0, 0], target = [0, 1] → 1 match / 2 = 0.5
        assert acc == 0.5

    def test_cross_entropy_with_one_hot_targets(self):
        logits = tenmo.tensor([[2.0, 1.0, 0.1]])
        target = tenmo.tensor([[1.0, 0.0, 0.0]])
        loss_fn = tenmo.CrossEntropyLoss()
        loss = loss_fn(logits, target)
        assert loss.item() > 0

    def test_sgd_get_set_lr(self):
        model = tenmo.Sequential([
            tenmo.Linear(2, 4).into(),
            tenmo.ReLU().into(),
            tenmo.Linear(4, 2).into(),
        ])
        opt = tenmo.SGD(model, lr=0.01)
        assert abs(opt.get_lr() - 0.01) < 1e-6
        opt.set_lr(0.05)
        assert abs(opt.get_lr() - 0.05) < 1e-6
