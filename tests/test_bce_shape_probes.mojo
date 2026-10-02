"""Guard-probe harness for BCE shape validation.

Each `--probe-*` invocation performs exactly ONE shape-mismatched BCE
call; the forward shape guard panics (→ abort) with its diagnostic, and
the spawning test in tests/test_bce.mojo asserts non-zero exit plus the
exact message. Deliberately a MINIMAL file: the child JIT runs alongside
its resident parent, so its compile surface must stay small enough to
avoid OOM (see tests/test_mixed_seq_annotation_probes.mojo precedent).
"""

from std.sys import argv

from tenmo.tensor import Tensor
from tenmo.shared.panic import panic


def fire_logits_numels() raises:
    """BCEWithLogits: logits (4,) vs target (2,) — different numels would
    OOB-read the shorter buffer in the BceBuffer SIMD loops."""
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d1([2.0, -1.0, 0.5, 1.0])
    var target = Tensor[dtype].d1([1.0, 0.0])
    _ = Tensor[dtype].binary_cross_entropy_with_logits(logits, target)
    panic("BCE shape guard did not fire")


def fire_bce_shape() raises:
    """BCE.
    Pred (4,) vs target (2, 2) — same numels, different shape would
    silently compute with mispaired elements."""
    comptime dtype = DType.float32
    var pred = Tensor[dtype].d1([0.9, 0.2, 0.7, 0.4])
    var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
    _ = Tensor[dtype].binary_cross_entropy(pred, target)
    panic("BCE shape guard did not fire")


def main() raises:
    var args = argv()
    for i in range(1, len(args)):
        if args[i] == "--probe-logits-numels":
            fire_logits_numels()
            return
        elif args[i] == "--probe-bce-shape":
            fire_bce_shape()
            return
