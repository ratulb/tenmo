"""Atomic global ID counter.

`IDGen.generate_id[global_name]()` fetches-and-increments a per-name counter
stored in a process-global `std.ffi._Global` slot. Each distinct
`global_name` gets an independent counter.

Moved here from `tenmo/common_utils.mojo` so `tenmo.gpu` (which must not import `common_utils`) can share it: tenmo
tensors draw from `_TENMO_ID_COUNTER` (default), `GPU` instances draw from
`_TENMO_GPU_ID_COUNTER`. Deliberately two counters — `tests/test_idgen.mojo`
asserts `Tensor` and `IDGen` share one slot, so GPU ids must not reuse it.
"""

from std.ffi import _Global
from std.atomic import Atomic

from .panic import panic


def _init_id_counter() -> Scalar[DType.uint64]:
    return Scalar[DType.uint64](0)


struct IDGen(RegisterPassable):
    @always_inline
    @staticmethod
    def generate_id[global_name: StringLiteral = "_TENMO_ID_COUNTER"]() -> UInt:
        try:
            comptime _COUNTER = _Global[global_name, _init_id_counter]
            var ptr = _COUNTER.get_or_create_ptr()
            _ = Atomic.fetch_add(ptr, Scalar[DType.uint64](1))
            return UInt(ptr[])
        except:
            panic("IDGen.generate_id: failed to access global ID counter")
            # Unreachable
            return 0
