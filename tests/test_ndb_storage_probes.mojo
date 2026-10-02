"""Floor-panic probes for NDBuffer.storage_get/storage_set.

Each `--probe-*` invocation performs exactly ONE below-floor storage
access on an offset view; the bounds check panics (→ abort) with its
diagnostic, and the spawning test in tests/test_ndb.mojo asserts non-zero
exit plus the exact message. Deliberately a MINIMAL file: the child JIT
runs alongside its resident parent, so its compile surface must stay
small enough to avoid OOM.
"""

from std.sys import argv

from tenmo.ndbuffer import NDBuffer
from tenmo.shared.buffers import Buffer
from tenmo.shared.shapes import Shape
from tenmo.shared.panic import panic


def fire_get_floor() raises:
    """storage_get below min_storage_index must abort."""
    var ndb = NDBuffer[DType.float32](
        Buffer[DType.float32](1, 2, 3, 4, 5, 6), Shape(2, 3)
    )
    var shared = ndb.share(Shape(3, 1), offset=3)
    _ = shared.storage_get(2)
    panic("storage_get floor guard did not fire")


def fire_set_floor() raises:
    """Storage set below min_storage_index must abort."""
    var ndb = NDBuffer[DType.float32](
        Buffer[DType.float32](1, 2, 3, 4, 5, 6), Shape(2, 3)
    )
    var shared = ndb.share(Shape(3, 1), offset=3)
    shared.storage_set(2, 99)
    panic("storage_set floor guard did not fire")


def main() raises:
    var args = argv()
    for i in range(1, len(args)):
        if args[i] == "--probe-get-floor":
            fire_get_floor()
        elif args[i] == "--probe-set-floor":
            fire_set_floor()
