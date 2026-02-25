from common_utils import panic, copy
from gpu.host import DeviceContext, DeviceBuffer
from memory import ArcPointer
from sys import has_accelerator
from utils import Variant
from ndbuffer import NDBuffer

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
struct BufferDeviceState[dtype: DType](
    Equatable & ImplicitlyCopyable & Movable
):
    var buffer_state: DeviceBuffer[Self.dtype]
    var gpu: GPU
    var synched_back: Bool

    fn __init__(
        out self,
        ref nd_buffer: NDBuffer[Self.dtype],
        gpu: Optional[GPU] = None,
    ) raises:
        var buffer_gpu = gpu.or_else(GPU())
        var numels = nd_buffer.numels()
        var buffer_state = buffer_gpu().enqueue_create_buffer[Self.dtype](
            numels
        )
        self.buffer_state = buffer_state^
        self.gpu = buffer_gpu^
        self.synched_back = False
        self.to_gpu(nd_buffer)

    fn __copyinit__(out self, existing: Self):
        self.buffer_state = existing.buffer_state.copy()
        self.gpu = existing.gpu.copy()
        self.synched_back = existing.synched_back

    fn __moveinit__(out self, deinit existing: Self):
        self.buffer_state = existing.buffer_state^
        self.gpu = existing.gpu^
        self.synched_back = existing.synched_back

    fn __eq__(self, other: Self) -> Bool:
        return self.gpu == other.gpu

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn to_gpu(mut self, ref nd_buffer: NDBuffer[Self.dtype]) raises:
        if nd_buffer.is_contiguous():
            var offset = nd_buffer.offset
            var src_ptr = nd_buffer.data_ptr() + offset
            #self.gpu().enqueue_copy(self.buffer_state, src_ptr)
            with self.buffer_state.map_to_host() as host_buffer:
                dest_ptr = host_buffer.unsafe_ptr()
                copy(src_ptr, dest_ptr, nd_buffer.numels())
        else:
            with self.buffer_state.map_to_host() as host_buffer:
                var ptr = host_buffer.unsafe_ptr()
                ref data_buffer = nd_buffer.data_buffer()
                var offset = 0
                for index in nd_buffer.index_iterator():
                    (ptr + offset)[] = data_buffer[index]
                    offset += 1
        self.synched_back = False

    fn to_cpu(mut self, mut nd_buffer: NDBuffer[Self.dtype]) raises:
        if nd_buffer.is_contiguous():
            var offset = nd_buffer.offset
            var data_dest = nd_buffer.data_ptr() + offset
            self.gpu().enqueue_copy(data_dest, self.buffer_state)
        else:
            with self.buffer_state.map_to_host() as host_buffer:
                var ptr = host_buffer.unsafe_ptr()
                ref data_buffer = nd_buffer.data_buffer()
                var offset = 0
                for index in nd_buffer.index_iterator():
                    data_buffer[index] = (ptr + offset)[]
                    offset += 1
        self.synched_back = True

    fn device_buffer(
        ref self,
    ) -> ref [self.buffer_state] DeviceBuffer[Self.dtype]:
        return self.buffer_state


fn main() raises:
    # var device = Device(CPU())
    print("passes", Device.has_accelerator)
