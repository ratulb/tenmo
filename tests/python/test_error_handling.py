"""Tests for error handling & edge cases (Category 22)."""
from __future__ import annotations

import numpy as np
import pytest
import tenmo


class TestErrorHandling:
    def test_dtype_mismatch_raises(self):
        a = tenmo.tensor([1.0, 2.0])
        b = tenmo.tensor([1, 2], dtype="int64")
        with pytest.raises(TypeError, match="dtype mismatch"):
            _ = a + b

    def test_ce_non_tensor_target_raises_loudly(self):
        """Per-flavor CE dispatch: a non-Tensor target must TypeError,
        never silently fall into the float path (E3)."""
        loss_fn = tenmo.CrossEntropyLoss()
        logits = tenmo.tensor([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(TypeError, match="must be a tenmo Tensor"):
            loss_fn(logits, [1, 0])
        with pytest.raises(TypeError, match="must be a tenmo Tensor"):
            loss_fn(logits, np.array([1, 0]))

    def test_ce_bool_target_raises_loudly(self):
        loss_fn = tenmo.CrossEntropyLoss()
        logits = tenmo.tensor([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(TypeError, match="must be int64 or float32"):
            loss_fn(logits, tenmo.tensor([True, False], dtype="bool"))

    def test_shape_mismatch_in_matmul_raises(self):
        """Run in subprocess — matmul shape mismatch panics and corrupts JIT state."""
        import os, subprocess, sys
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(
            os.path.dirname(__file__), "..", "..", "python-binding"
        )
        result = subprocess.run(
            [sys.executable, "-c", """
import tenmo
a = tenmo.tensor([[1.0, 2.0, 3.0]])
b = tenmo.tensor([[1.0, 2.0]])
try:
    _ = a.matmul(b)
except Exception:
    pass
"""],
            capture_output=True, timeout=30, env=env,
        )
        # Core panics (abort) on shape mismatch — subprocess exit != 0 is expected
        assert result.returncode != 0 or b"panic" in result.stderr or b"ABORT" in result.stdout

    def test_bool_arithmetic_does_not_crash(self):
        """Bool tensors — core may silently handle or raise; verify no SIGILL."""
        t = tenmo.tensor([True, False])
        try:
            _ = t + t
        except TypeError:
            pass  # acceptable

    def test_single_element_tensor(self):
        t = tenmo.tensor([42.0])
        assert t.item() == 42.0
        assert t + t == tenmo.tensor([84.0])

    def test_nan_propagates(self):
        t = tenmo.tensor([1.0, float("nan"), 3.0])
        result = t + 1.0
        vals = result.tolist()
        assert vals[0] == 2.0
        assert np.isnan(vals[1])
        assert vals[2] == 4.0

    def test_inplace_on_leaf_with_grad_raises(self):
        t = tenmo.tensor([1.0, 2.0], requires_grad=True)
        with pytest.raises(RuntimeError, match="leaf"):
            t += tenmo.tensor([3.0, 4.0])

    def test_repr_is_useful(self):
        t = tenmo.tensor([[1.0, 2.0], [3.0, 4.0]])
        r = repr(t)
        assert "Tensor" in r
        assert "shape=" in r

    @pytest.mark.skip(reason="core aborts on negative/zero shape; no Python-level validation hook")
    def test_construction_rejects_negative_shape(self): pass

    @pytest.mark.skip(reason="stress probe deferred — very large allocations not yet load-tested")
    def test_very_large_tensor_construction_does_not_crash(self): pass


class TestErrorHandlingExtended:
    def test_inf_propagation(self):
        import math
        t = tenmo.tensor([1.0, 2.0])
        inf_val = t * float("inf")
        vals = inf_val.tolist()
        assert math.isinf(vals[0])
        assert math.isinf(vals[1])

    def test_nan_comparison_with_nan(self):
        import math
        t = tenmo.tensor([1.0, float("nan"), 3.0])
        vals = t.tolist()
        assert vals[0] == 1.0
        assert math.isnan(vals[1])
        assert vals[2] == 3.0

    def test_medium_tensor_no_crash(self):
        t = tenmo.tensor([1.0] * 1000)
        result = t.sum()
        assert abs(result.item() - 1000.0) < 0.01
