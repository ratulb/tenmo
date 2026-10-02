"""Guard-probe harness for Concate input validation.

Each `--probe-*` invocation performs exactly ONE invalid concat call; the
forward validation guard panics (→ abort) with its diagnostic, and the
spawning test in tests/test_concat.mojo asserts non-zero exit plus the
exact message. Deliberately a MINIMAL file: the child JIT runs alongside
its resident parent, so its compile surface must stay small enough to
avoid OOM (see tests/test_mixed_seq_annotation_probes.mojo precedent).
"""

from std.sys import argv

from tenmo.tensor import Tensor
from tenmo.shared.panic import panic


def fire_single_bad_axis() raises:
    """Single-tensor concat with an out-of-bounds axis must abort in axis
    validation (previously the early alias return skipped it)."""
    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)

    _ = Tensor[dtype].concat(tensors, axis=99)
    panic("Concate validation guard did not fire")


def main() raises:
    var args = argv()
    for i in range(1, len(args)):
        if args[i] == "--probe-single-bad-axis":
            fire_single_bad_axis()
            return
