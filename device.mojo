from common_utils import panic, now
from gpu.host import DeviceContext, DeviceBuffer
from memory import ArcPointer, memcpy
from sys import has_accelerator
from utils import Variant
from ndbuffer import NDBuffer
from shapes import Shape

comptime DeviceType = Variant[CPU, GPU]


@fieldwise_init
struct Device(Equatable, ImplicitlyCopyable, Movable):
    comptime has_accelerator = has_accelerator()
    var kind: DeviceType

    fn __init__(out self):
        self.kind = CPU()

    fn __eq__(self, other: Self) -> Bool:
        if self.kind.isa[CPU]():
            if other.kind.isa[CPU]():
                return True
            else:
                return False
        else:
            if other.kind.isa[CPU]():
                return False
            else:
                var self_gpu = self.kind[GPU]
                var other_gpu = other.kind[GPU]
                return self_gpu == other_gpu

    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    fn into(self) -> Device:
        return Device(self)

    fn is_cpu(self) -> Bool:
        return self.kind.isa[CPU]()

    fn is_gpu(self) -> Bool:
        return self.kind.isa[GPU]()


@fieldwise_init
struct CPU(Equatable, ImplicitlyCopyable, Movable):
    fn __eq__(self, other: Self) -> Bool:
        return True

    fn __ne__(self, other: Self) -> Bool:
        return False

    fn into(self) -> Device:
        return Device(self)


@fieldwise_init
struct GPU(Equatable, ImplicitlyCopyable, Movable):
    """Essentially a shared DeviceContext."""

    var device_context: ArcPointer[DeviceContext]

    fn into(self) -> Device:
        return Device(self)

    fn __init__(out self, device_id: Int = 0) raises:
        self.device_context = ArcPointer(DeviceContext(device_id))

    fn __copyinit__(out self, existing: Self):
        self.device_context = existing.device_context.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.device_context = existing.device_context^

    fn __eq__(self, other: Self) -> Bool:
        return self.device_context.__is__(other.device_context)

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __enter__(mut self) -> Self:
        return self.copy()

    fn __exit__(mut self):
        print("Context exit")

    fn __getitem__(self) -> ArcPointer[DeviceContext]:
        return self.device_context.copy()

    fn handle(self) -> ArcPointer[DeviceContext]:
        return self.device_context.copy()

    fn __call__(self) -> DeviceContext:
        return self.device_context.copy()[]


@fieldwise_init
struct DeviceState[dtype: DType](Equatable & ImplicitlyCopyable & Movable):
    var buffer: DeviceBuffer[Self.dtype]
    var gpu: GPU

    fn __init__(
        out self,
        size: Int,
        gpu: Optional[GPU] = None,
    ) raises:
        var device_ctx = gpu.or_else(GPU())
        var device_buffer = device_ctx().enqueue_create_buffer[Self.dtype](size)
        self.buffer = device_buffer^
        self.gpu = device_ctx^

    fn __copyinit__(out self, existing: Self):
        self.buffer = existing.buffer.copy()
        self.gpu = existing.gpu.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.buffer = existing.buffer^
        self.gpu = existing.gpu^

    fn __eq__(self, other: Self) -> Bool:
        return self.gpu == other.gpu

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn fill(self, ref source: NDBuffer[Self.dtype]) raises:
        with self.buffer.map_to_host() as host_buffer:
            var device_ptr = host_buffer.unsafe_ptr()
            var src_ptr = source.data_ptr()

            if source.is_contiguous():
                var src_offset = source.offset
                # Take care of contiguous views with offset
                var src_ptr = source.data_ptr() + src_offset
                memcpy(dest=device_ptr, src=src_ptr, count=source.numels())
                # self.gpu().enqueue_copy(self.buffer, src_ptr)
            else:
                var next_index = 0
                # Iterate strided indices
                for index in source.index_iterator():
                    (device_ptr + next_index)[] = (src_ptr + index)[]
                    next_index += 1

    fn into(
        self, shape: Shape, *, copy: Bool = True
    ) raises -> NDBuffer[Self.dtype]:
        """Copy the DeviceState content to realize a filled NDBuffer.
        The NDBuffer is contiguous with 0 offset.
        """
        return NDBuffer[Self.dtype](self.buffer, shape, copy=copy)

    fn device_buffer(
        ref self,
    ) -> ref [self.buffer] DeviceBuffer[Self.dtype]:
        return self.buffer


fn main() raises:
    # var device = Device(CPU())
    print("passes", Device.has_accelerator)
