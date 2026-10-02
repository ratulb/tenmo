"""Guard-probe harness for negative-dim normalization bounds.

Each `--probe-*` invocation performs exactly ONE out-of-range dim call; the
validation guard panics (→ abort) with its diagnostic, and the spawning
tests in tests/test_flatten.mojo / tests/test_shuffle.mojo assert non-zero
exit plus the exact message. Deliberately a MINIMAL file: the child JIT runs
alongside its resident parent, so its compile surface must stay small enough
to avoid OOM (see tests/test_mixed_seq_annotation_probes.mojo precedent).
"""

from std.sys import argv

from tenmo.tensor import Tensor
from tenmo.shared.panic import panic


def fire_flatten_bad_start() raises:
    """flatten(start_dim=5) on a rank-3 tensor must abort in start_dim
    validation (previously the raw value flowed into slice arithmetic)."""
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )

    _ = a.flatten(start_dim=5)
    panic("Flatten validation guard did not fire")


def fire_flatten_bad_end() raises:
    """Flatten(end_dim=-4) on a rank-3 tensor must abort in end_dim.
    validation after normalization (rank + -4 = -1, still out of range)."""
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )

    _ = a.flatten(start_dim=0, end_dim=-4)
    panic("Flatten validation guard did not fire")


def fire_shuffle_bad_axis() raises:
    """Shuffle(axis=-4) on a rank-2 tensor must abort in axis validation.
    (previously the raw value flowed into shape[axis] and the NDBuffer
    fast-path range(axis) arithmetic)."""
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])

    _ = a.shuffle([1, 0], axis=-4)
    panic("Shuffle validation guard did not fire")


def main() raises:
    var args = argv()
    for i in range(1, len(args)):
        if args[i] == "--probe-flatten-bad-start":
            fire_flatten_bad_start()
            return
        if args[i] == "--probe-flatten-bad-end":
            fire_flatten_bad_end()
            return
        if args[i] == "--probe-shuffle-bad-axis":
            fire_shuffle_bad_axis()
            return
