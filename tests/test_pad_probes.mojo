"""Guard-probe harness for Pad input validation.

Each `--probe-*` invocation performs exactly ONE invalid pad call; the
forward validation guard panics (→ abort) with its diagnostic, and the
spawning test in tests/test_pad.mojo asserts non-zero exit plus the exact
message. Deliberately a MINIMAL file: the child JIT runs alongside its
resident parent, so its compile surface must stay small enough to avoid
OOM (see tests/test_mixed_seq_annotation_probes.mojo precedent).
"""

from std.sys import argv

from tenmo.tensor import Tensor
from tenmo.shared.panic import panic


def fire_negative_pad() raises:
    """Negative pad (crop semantics — unimplemented) must abort up front
    instead of flowing into undersized-shape math and OOB downstream."""
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var pad = List[Tuple[Int, Int]]()
    pad.append((-1, 0))
    pad.append((0, 0))
    _ = Tensor[dtype].pad(x, pad, mode="constant", value=0.0)
    panic("Pad validation guard did not fire")


def main() raises:
    var args = argv()
    for i in range(1, len(args)):
        if args[i] == "--probe-negative-pad":
            fire_negative_pad()
            return
