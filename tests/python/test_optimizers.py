"""Tests for the AdamW optimizer binding (G3a).

Mirrors the SGD surface: construction with kwargs, get/set lr, step,
zero_grad, and state_dict parity.
"""
from __future__ import annotations

import numpy as np
import tenmo


class TestAdamW:
    def _model(self):
        return tenmo.Sequential([tenmo.Linear(2, 1).into()])

    def test_default_lr(self):
        opt = tenmo.AdamW(self._model())
        assert abs(opt.get_lr() - 0.001) < 1e-9

    def test_get_set_lr(self):
        opt = tenmo.AdamW(self._model(), lr=0.01)
        assert abs(opt.get_lr() - 0.01) < 1e-9
        opt.set_lr(0.002)
        assert abs(opt.get_lr() - 0.002) < 1e-9

    def test_custom_kwargs_construct(self):
        opt = tenmo.AdamW(
            self._model(),
            lr=0.002,
            beta1=0.8,
            beta2=0.9,
            eps=1e-7,
            weight_decay=0.01,
            clip_norm=1.0,
            clip_value=0.5,
        )
        assert abs(opt.get_lr() - 0.002) < 1e-9

    def test_step_reduces_mse(self):
        model = self._model()
        opt = tenmo.AdamW(model, lr=0.05)
        loss_fn = tenmo.MSELoss()
        x = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        y = tenmo.tensor([[3.0], [7.0]])  # y = x0 + x1
        before = loss_fn(model(x), y).item()
        for _ in range(50):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        after = loss_fn(model(x), y).item()
        assert after < before

    def test_state_dict(self):
        model = self._model()
        opt = tenmo.AdamW(model)
        sd = opt.state_dict()
        assert sd["type"] == "AdamW"
        assert float(np.asarray(sd["step_count"])[0]) == 0.0
        assert abs(float(np.asarray(sd["lr"])[0]) - 0.001) < 1e-9
        n_params = len(sd["m_states"])
        assert n_params > 0
        assert len(sd["v_states"]) == n_params

        opt.zero_grad()
        out = model(tenmo.tensor([[1.0, 2.0]]))
        loss = tenmo.MSELoss()(out, tenmo.tensor([[3.0]]))
        loss.backward()
        opt.step()
        sd2 = opt.state_dict()
        assert float(np.asarray(sd2["step_count"])[0]) == 1.0
        assert len(sd2["m_states"]) == n_params

    def test_repr(self):
        opt = tenmo.AdamW(self._model(), lr=0.01)
        assert "AdamW" in repr(opt)
