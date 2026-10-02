"""Layer-0 GPU device types: `GPU` handle and `DeviceState` device storage.

This module is self-contained: it depends only on the stdlib and the
`tenmo.shared` package. There is **no** reference to `NDBuffer`, `Tensor`,
 or any tenmo-layer type. The NDBuffer bridge (`from_device_state` / `fill_device_state`)
lives in `tenmo/ndbuffer.mojo` keeping this layer pure.

`GPU.__eq__` uses a stable per-device instance identity: every ordinary
`GPU()` / `GPU(device_id)` construction returns a copy of the process-wide
canonical GPU for that device (`tenmo/gpu/registry.mojo`), so all kernels
share one `DeviceContext` (one stream, one memory pool, one compiled-function
cache). `GPU(force=True)` bypasses the registry for a fresh, caller-owned
context. Identity comes from the shared `IDGen` (`tenmo/shared/idgen.mojo`,
`_TENMO_GPU_ID_COUNTER` slot — Layer-0).
Delegating equality to the `DeviceContext` handle is not possible in the b2
stdlib (`DeviceContext` implements no `__eq__`, and address-of identity would
break the copy-preserving identity that `NDBuffer.__is__` relies on).
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.sys import simd_width_of
from std.sys.defines import get_defined_int
from std.utils import Variant

from ..shared.panic import panic
from .registry import GPURegistry, RegisteredGPU


comptime DeviceType = Variant[CPU, GPU]


@fieldwise_init
struct Device(Equatable, ImplicitlyCopyable, Writable):
    var kind: DeviceType

    def __init__(out self):
        self.kind = CPU()

    def __eq__(self, other: Self) -> Bool:
        if self.kind.isa[CPU]():
            if other.kind.isa[CPU]():
                return self.kind[CPU] == other.kind[CPU]
            else:
                return False
        else:
            if other.kind.isa[CPU]():
                return False
            else:
                var self_gpu = self.kind[GPU]
                var other_gpu = other.kind[GPU]
                return self_gpu == other_gpu

    def __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    def is_cpu(self) -> Bool:
        return self.kind.isa[CPU]()

    def is_gpu(self) -> Bool:
        return self.kind.isa[GPU]()

    def write_to[W: Writer](self, mut writer: W):
        if self.is_cpu():
            writer.write(self.kind[CPU])
        else:
            writer.write(self.kind[GPU])

    def gpu(self) -> GPU:
        if not self.is_gpu():
            panic("Device is not a gpu")
        return self.kind[GPU]


@fieldwise_init
struct CPU(Equatable, ImplicitlyCopyable, Writable):
    var id: Int

    def __init__(out self):
        self.id = get_defined_int["CPU", 0]()

    def __eq__(self, other: Self) -> Bool:
        return self.id == other.id

    def __ne__(self, other: Self) -> Bool:
        return not self == other

    def into(self) -> Device:
        return Device(self)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("CPU[" + String(self.id) + "]")


@fieldwise_init
struct GPU(Equatable, ImplicitlyCopyable, Writable):
    """Handle to the process-wide canonical DeviceContext for a device, or a
    fresh caller-owned one when constructed with `force=True`."""

    var device_context: DeviceContext
    var id: Int64
    var _id: UInt

    def __init__(out self, device_id: Int = 0, *, force: Bool = False) raises:
        var reg: RegisteredGPU
        if force:
            reg = GPURegistry.create(device_id)
        else:
            reg = GPURegistry.get(device_id)
        self.device_context = reg.device_context.copy()
        self.id = reg.id
        self._id = reg._id

    def __init__(out self, *, copy: Self):
        self.device_context = copy.device_context.copy()
        self.id = copy.id
        self._id = copy._id

    def __init__(out self, *, deinit move: Self):
        self.device_context = move.device_context^
        self.id = move.id
        self._id = move._id

    def write_to[W: Writer](self, mut writer: W):
        # Not printing _id
        writer.write("GPU[" + String(self.id) + "]")

    def __eq__(self, other: Self) -> Bool:
        return self._id == other._id

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def __enter__(mut self) -> Self:
        return self

    def __exit__(mut self):
        try:
            self.device_context.synchronize()
        except e:
            print(e)
            print("Error synchronizing GPU device context: ", String(e))

    def __getitem__(self) -> DeviceContext:
        return self.device_context.copy()

    def into(self) -> Device:
        return Device(self)


struct DeviceState[dtype: DType](
    Equatable & ImplicitlyCopyable & Sized
):
    """
        GPU device storage for a single (contiguous) buffer.

    DType.bool is stored internally as DType.uint8 since DeviceBuffer[DType.bool]
    is unsupported on GPU. All buffer operations cast accordingly.

    """

    # Internal storage dtype: bool → uint8, everything else → dtype
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    var buffer: DeviceBuffer[Self.datatype]
    var gpu: GPU

    def __init__(
        out self,
        size: Int,
        gpu: Optional[GPU] = None,
    ) raises:
        var device_ctx = gpu.or_else(GPU())
        var device_buffer = device_ctx[].enqueue_create_buffer[Self.datatype](
            size
        )
        self.buffer = device_buffer^
        self.gpu = device_ctx^

    def __init__[
        special: Bool
    ](
        out self,
        buffer: DeviceBuffer[
            Self.datatype
        ],  # accepts datatype (uint8 for bool)
        gpu: GPU,
    ) raises:
        self.buffer = buffer
        self.gpu = gpu

    def __init__(
        out self,
        buffer: DeviceBuffer[Self.dtype],
        gpu: GPU,
    ) raises:
        self.buffer = buffer.create_sub_buffer[Self.datatype](0, len(buffer))
        self.gpu = gpu

    def __init__(out self, *, copy: Self):
        self.gpu = copy.gpu.copy()
        self.buffer = copy.buffer.copy()

    def __init__(out self, *, deinit move: Self):
        self.buffer = move.buffer^
        self.gpu = move.gpu^

    def __eq__(self, other: Self) -> Bool:
        return self.gpu == other.gpu

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def __len__(self) -> Int:
        return len(self.buffer)

    @always_inline
    def sync(self) raises:
        self.gpu[].synchronize()

    def new(
        self,
        size: Int,
        value: Scalar[Self.dtype] = Scalar[Self.dtype](0),
        sync: Bool = False,
    ) raises -> DeviceState[Self.dtype]:
        var device_state = DeviceState[Self.dtype](size, self.gpu)

        comptime if Self.dtype == DType.bool:
            var storage_val = UInt8(1) if value.cast[DType.bool]() else UInt8(0)
            device_state.buffer.enqueue_fill(
                rebind[Scalar[Self.datatype]](storage_val)
            )
        else:
            device_state.buffer.enqueue_fill(
                rebind[Scalar[Self.datatype]](value)
            )
        if sync:
            self.sync()
        return device_state

    def fill(self, value: Scalar[Self.dtype], sync: Bool = False) raises:
        with self.buffer.map_to_host() as host_buffer:
            comptime if Self.dtype == DType.bool:
                var storage_val = UInt8(1) if value.cast[
                    DType.bool
                ]() else UInt8(0)
                host_buffer.enqueue_fill(
                    rebind[Scalar[Self.datatype]](storage_val)
                )
            else:
                host_buffer.enqueue_fill(rebind[Scalar[Self.datatype]](value))
        if sync:
            self.sync()

    def device_buffer(
        ref self,
    ) -> ref[self.buffer] DeviceBuffer[Self.datatype]:
        return self.buffer

    def get_gpu(
        ref self,
    ) -> ref[self.gpu] GPU:
        return self.gpu

    def __getitem__(self, index: Int) raises -> Scalar[Self.dtype]:
        with self.buffer.map_to_host() as host_buffer:
            comptime if Self.dtype == DType.bool:
                return Scalar[Self.dtype](
                    UInt8(
                        host_buffer[index].cast[DType.uint8]() == UInt8(1)
                    )
                )
            else:
                return host_buffer[index].cast[Self.dtype]()

    def __setitem__(self, index: Int, value: Scalar[Self.dtype]) raises:
        with self.buffer.map_to_host() as host_buffer:
            comptime if Self.dtype == DType.bool:
                host_buffer[index] = Scalar[Self.datatype](
                    UInt8(1) if value.cast[DType.bool]() else UInt8(0)
                )
            else:
                host_buffer[index] = value.cast[Self.datatype]()

    def load[
        simdwidth: Int = simd_width_of[Self.datatype]()
    ](self, addr: Int) raises -> SIMD[Self.datatype, simdwidth]:
        with self.buffer.map_to_host() as host_buffer:
            var device_ptr = host_buffer.unsafe_ptr()
            return device_ptr.unsafe_load[width=simdwidth](addr)

    def store[
        simdwidth: Int = simd_width_of[Self.datatype]()
    ](self, addr: Int, value: SIMD[Self.datatype, simdwidth]) raises:
        with self.buffer.map_to_host() as host_buffer:
            var device_ptr = host_buffer.unsafe_ptr()
            device_ptr.unsafe_store[width=simdwidth](addr, value)

    def all_true(self) -> Bool where Self.dtype == DType.bool:
        try:
            var length = len(self)
            if length == 0:
                return True
            with self.buffer.map_to_host() as host_buffer:
                for i in range(length):
                    if host_buffer[i] != Scalar[Self.datatype](1):
                        return False
            return True
        except e:
            print(e)
            return False

    def any_true(self) -> Bool where Self.dtype == DType.bool:
        try:
            var length = len(self)
            if length == 0:
                return False
            with self.buffer.map_to_host() as host_buffer:
                for i in range(length):
                    if host_buffer[i] == Scalar[Self.datatype](1):
                        return True
            return False
        except e:
            print(e)
            return False
