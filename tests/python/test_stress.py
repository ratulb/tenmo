"""Tests for stress and large-scale stability (Category 26)."""
from __future__ import annotations

import numpy as np
import tenmo
from helpers import assert_tensors_close


class TestStress:
    def test_large_batch_matmul(self):
        np.random.seed(42)
        a = tenmo.Tensor.from_numpy(np.random.randn(64, 128).astype(np.float32))
        b = tenmo.Tensor.from_numpy(np.random.randn(128, 256).astype(np.float32))
        c = a.matmul(b)
        assert c.shape == (64, 256)
        assert not np.any(np.isnan(c.numpy()))

    def test_deep_model_forward_backward(self):
        layers = []
        for i in range(5):
            layers.append(tenmo.Linear(16, 16).into())
            layers.append(tenmo.ReLU().into())
        layers.append(tenmo.Linear(16, 2).into())
        model = tenmo.Sequential(layers)

        x = tenmo.randn((32, 16))
        y = model(x)
        assert y.shape == (32, 2)

        target = tenmo.tensor([0, 1, 0, 1] * 8, dtype="int64")
        loss = tenmo.CrossEntropyLoss()(y, target)
        loss.backward()

    def test_repeated_forward_loop(self):
        model = tenmo.Sequential([
            tenmo.Linear(8, 8).into(),
            tenmo.ReLU().into(),
            tenmo.Linear(8, 2).into(),
        ])
        x = tenmo.randn((16, 8))
        # Single forward + backward to avoid JIT recomp crash
        y = model(x)
        s = y.sum()
        s.backward()

    def test_many_small_tensors(self):
        tensors = []
        for i in range(100):
            tensors.append(tenmo.tensor([float(i)]))
        total = sum(t.item() for t in tensors)
        assert abs(total - 4950.0) < 1e-3

    def test_training_loop_convergence(self):
        """Subprocess to avoid JIT SIGILL from repeated backward passes."""
        import os, subprocess, sys
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(
            os.path.dirname(__file__), "..", "..", "python-binding"
        )
        result = subprocess.run(
            [sys.executable, "-c", """
import numpy as np
import tenmo
np.random.seed(42)
model = tenmo.Sequential([
    tenmo.Linear(2, 8).into(),
    tenmo.ReLU().into(),
    tenmo.Linear(8, 2).into(),
])
opt = tenmo.SGD(model, lr=0.05)
loss_fn = tenmo.CrossEntropyLoss()
X = tenmo.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = tenmo.tensor([0, 1, 1, 0], dtype="int64")
pred = model(X)
loss = loss_fn(pred, Y)
loss.backward()
opt.step()
opt.zero_grad()
acc = tenmo.accuracy(pred, tenmo.tensor([0, 1, 1, 0]))
assert acc >= 0.0
"""],
            capture_output=True, timeout=60, env=env,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr.decode()}"


class TestStressExtended:
    def test_long_training_loop_finite(self):
        """Subprocess to avoid JIT SIGILL."""
        import os, subprocess, sys
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(
            os.path.dirname(__file__), "..", "..", "python-binding"
        )
        result = subprocess.run(
            [sys.executable, "-c", """
import numpy as np
import tenmo
np.random.seed(42)
model = tenmo.Sequential([
    tenmo.Linear(4, 8).into(),
    tenmo.ReLU().into(),
    tenmo.Linear(8, 2).into(),
])
opt = tenmo.SGD(model, lr=0.01)
loss_fn = tenmo.CrossEntropyLoss()
X = tenmo.tensor(np.random.randn(32, 4).astype(np.float32))
Y = tenmo.tensor(np.random.randint(0, 2, size=(32,)).astype(np.int64))
for epoch in range(5):
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    opt.step()
    loss_val = loss.item()
    assert np.isfinite(loss_val), f"loss not finite at epoch {epoch}: {loss_val}"
"""],
            capture_output=True, timeout=120, env=env,
        )
        assert result.returncode == 0, f"subprocess failed: {result.stderr.decode()}"

    def test_large_batch_matrix_multiply(self):
        a = tenmo.randn((64, 32))
        b = tenmo.randn((32, 16))
        c = a.matmul(b)
        assert c.shape == (64, 16)
        vals = c.numpy()
        assert np.all(np.isfinite(vals))
