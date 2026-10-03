"""Process-wide canonical GPU registry.

Every ordinary `GPU()` / `GPU(device_id)` construction routes through
`GPURegistry.get()`, which returns the process-wide canonical `RegisteredGPU`
for that device — creating it once on first use and caching it. All kernels
then share a single `DeviceContext` per device: one stream, one memory pool,
and one compiled-function cache. Without this, every `GPU()` construction
spins up a fresh `DeviceContext`, and each context re-compiles the same
kernels on first launch (`max.gpu.host.DeviceContext` keeps its compiled
functions in a per-context cache).

`GPU(force=True)` routes through `GPURegistry.create()` instead: a fresh,
caller-owned `DeviceContext` that is never registered and does not disturb the
canonical.

The registry is stored in a `std.ffi._Global` slot (the same mechanism the
`IDGen` counters use), so it is process-wide and lazily initialized. It is
keyed by device id (`Int64`), so it can never hold more entries than the
machine has physical devices. Single-threaded today: no locking around the
create-or-fetch critical section — add a mutex before introducing threads.

Purity: this module imports only `std` + `tenmo.shared` (never any
`tenmo.gpu` module and never upward), so `device.mojo` can depend on it
one-way without creating an import cycle.
"""

from std.ffi import _Global
from std.collections import Dict
from max.gpu.host import DeviceContext

from ..shared.idgen import IDGen


def _init_gpu_registry() -> Dict[Int64, RegisteredGPU]:
    return Dict[Int64, RegisteredGPU]()


struct RegisteredGPU(ImplicitlyCopyable):
    """The per-device GPU state that must be stable: the canonical
    `DeviceContext`, the physical device id, and the stable instance identity
    (`_id`, drawn once from the shared `_TENMO_GPU_ID_COUNTER`). Copies share
    the same underlying context (its C++ refcount is incremented), so the
    registry's copy and every GPU built from it stay alive together.
    """

    var device_context: DeviceContext
    var id: Int64
    var _id: UInt

    def __init__(out self, ctx: DeviceContext, id: Int64, _id: UInt):
        self.device_context = ctx.copy()
        self.id = id
        self._id = _id

    def __init__(out self, *, copy: Self):
        self.device_context = copy.device_context.copy()
        self.id = copy.id
        self._id = copy._id

    def __init__(out self, *, deinit move: Self):
        self.device_context = move.device_context^
        self.id = move.id
        self._id = move._id


struct GPURegistry(RegisterPassable):
    """Lazily-initialized process-global map: device id -> canonical GPU."""

    @staticmethod
    def get(device_id: Int) raises -> RegisteredGPU:
        """Return the canonical GPU state for `device_id`, creating it once."""
        comptime _REG = _Global["_TENMO_GPU_REGISTRY", _init_gpu_registry]
        var reg_ptr = _REG.get_or_create_ptr()
        ref reg = reg_ptr[]
        var key = Int64(device_id)
        if key in reg:
            return reg[key].copy()
        var ctx = DeviceContext(device_id)
        var canonical = RegisteredGPU(
            ctx^, key, IDGen.generate_id["_TENMO_GPU_ID_COUNTER"]()
        )
        reg[key] = canonical.copy()
        return canonical^

    @staticmethod
    def create(device_id: Int) raises -> RegisteredGPU:
        """Build a fresh, untracked GPU state (the `force=True` path)."""
        var ctx = DeviceContext(device_id)
        return RegisteredGPU(
            ctx^, Int64(device_id), IDGen.generate_id["_TENMO_GPU_ID_COUNTER"]()
        )
