from common_utils import panic, now
from gpu.host import DeviceContext, DeviceBuffer
from memory import ArcPointer, memcpy
from sys import has_accelerator, simd_width_of
from utils import Variant
from ndbuffer import NDBuffer
from shapes import Shape

comptime DeviceType = Variant[CPU, GPU]


@fieldwise_init
struct Device(Equatable, ImplicitlyCopyable, Movable, Writable):
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

    fn is_cpu(self) -> Bool:
        return self.kind.isa[CPU]()

    fn is_gpu(self) -> Bool:
        return self.kind.isa[GPU]()

    fn write_to[W: Writer](self, mut writer: W):
        if self.is_cpu():
            writer.write(self.kind[CPU])
        else:
            writer.write(self.kind[GPU])


@fieldwise_init
struct CPU(Equatable, ImplicitlyCopyable, Movable, Writable):
    fn __eq__(self, other: Self) -> Bool:
        return True

    fn __ne__(self, other: Self) -> Bool:
        return False

    fn into(self) -> Device:
        return Device(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("CPU")


@fieldwise_init
struct GPU(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Essentially a shared DeviceContext."""

    var device_context: ArcPointer[DeviceContext]
    var id: Int64

    fn into(self) -> Device:
        return Device(self)

    fn __init__(out self, device_id: Int = 0) raises:
        self.device_context = ArcPointer(DeviceContext(device_id))
        self.id = device_id

    fn __copyinit__(out self, existing: Self):
        self.device_context = existing.device_context.copy()
        self.id = existing.id

    fn __moveinit__(out self, deinit existing: Self):
        self.device_context = existing.device_context^
        self.id = existing.id

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("GPU[" + self.id.__str__() + "]")

    fn __eq__(self, other: Self) -> Bool:
        return (
            self.device_context.__is__(other.device_context)
            or self.id == other.id
        )

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __enter__(mut self) -> Self:
        return self

    fn __exit__(mut self):
        try:
            self.device_context[].synchronize()
        except e:
            print(e)
            print("Error synchronizing GPU device context: ", e.__str__())

    fn __getitem__(self) -> ArcPointer[DeviceContext]:
        return self.device_context.copy()

    fn handle(self) -> ArcPointer[DeviceContext]:
        return self.device_context.copy()

    fn __call__(self) -> DeviceContext:
        return self.device_context.copy()[]


@fieldwise_init
struct DeviceState[dtype: DType](
    Equatable & ImplicitlyCopyable & Movable & Sized
):
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
        self.gpu = existing.gpu.copy()
        self.buffer = existing.buffer.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.buffer = existing.buffer^
        self.gpu = existing.gpu^

    fn __eq__(self, other: Self) -> Bool:
        return self.gpu == other.gpu

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __len__(self) -> Int:
        return len(self.buffer)

    @always_inline
    fn sync(self) raises:
        self.gpu().synchronize()

    fn new(
        self,
        size: Int,
        value: Scalar[Self.dtype] = Scalar[Self.dtype](0),
        sync: Bool = True,
    ) raises -> DeviceState[Self.dtype]:
        var device_state = Self(size, self.gpu)
        device_state.buffer.enqueue_fill(value)
        if sync:
            self.sync()
        return device_state

    fn fill(self, value: Scalar[Self.dtype], sync: Bool = True) raises:
        with self.buffer.map_to_host() as host_buffer:
            host_buffer.enqueue_fill(value)
        if sync:
            self.sync()

    fn fill(self, ref source: NDBuffer[Self.dtype], sync: Bool = True) raises:
        """Fill the DeviceBuffer from the source NDBuffer."""

        if source.is_on_gpu():
            if source.is_contiguous():
                source.device_state.value().buffer.enqueue_copy_to(self.buffer)
            else:
                # Materialise to CPU first, then CPU→GPU
                var cpu_ndb = source.device_state.value().into(
                    source.shape, sync=False
                )
                self.fill(cpu_ndb)
            if sync:
                self.sync()
            return

        with self.buffer.map_to_host() as host_buffer:
            var device_ptr = host_buffer.unsafe_ptr()
            var src_ptr = source.data_ptr()

            if source.is_contiguous():
                var src_offset = source.offset
                # Take care of contiguous views with offset
                src_ptr = src_ptr + src_offset
                memcpy(dest=device_ptr, src=src_ptr, count=source.numels())
            else:
                var next_index = 0
                # Iterate strided indices
                for index in source.index_iterator():
                    (device_ptr + next_index)[] = (src_ptr + index)[]
                    next_index += 1
        if sync:
            self.sync()

    fn into(
        self, shape: Shape, *, copy: Bool = True, sync: Bool = True
    ) raises -> NDBuffer[Self.dtype]:
        """Copy the DeviceState content to realize a filled NDBuffer.
        The NDBuffer is contiguous with 0 offset.
        """
        var ndb = NDBuffer[Self.dtype](self.buffer, shape, copy=copy)
        if sync:
            self.sync()
        return ndb^

    fn device_buffer(
        ref self,
    ) -> ref [self.buffer] DeviceBuffer[Self.dtype]:
        return self.buffer

    fn get_gpu(
        ref self,
    ) -> ref [self.gpu] GPU:
        return self.gpu

    fn __getitem__(self, index: Int) raises -> Scalar[Self.dtype]:
        with self.buffer.map_to_host() as host_buffer:
            return host_buffer[index]

    fn __setitem__(self, index: Int, value: Scalar[Self.dtype]) raises:
        with self.buffer.map_to_host() as host_buffer:
            host_buffer[index] = value

    fn load[
        simdwidth: Int = simd_width_of[Self.dtype]()
    ](self, addr: Int) raises -> SIMD[Self.dtype, simdwidth]:
        with self.buffer.map_to_host() as host_buffer:
            var device_ptr = host_buffer.unsafe_ptr()
            return device_ptr.load[width=simdwidth](addr)

    fn store[
        simdwidth: Int = simd_width_of[Self.dtype]()
    ](self, addr: Int, value: SIMD[Self.dtype, simdwidth]) raises:
        with self.buffer.map_to_host() as host_buffer:
            var device_ptr = host_buffer.unsafe_ptr()
            device_ptr.store[width=simdwidth](addr, value)

    fn all_true(self: DeviceState[DType.bool]) -> Bool:
        try:
            var length = len(self)
            if length == 0:
                return True
            with self.buffer.map_to_host() as host_buffer:
                for i in range(length):
                    if not host_buffer[i]:
                        return False
            return True
        except e:
            print(e)
            return False


from tenmo import Tensor


fn main() raises:
    # var device = Device(CPU())
    comptime dtype = DType.float32
    var ctx1 = DeviceContext()
    var ctx2 = DeviceContext()
    print(ctx1.id() == ctx2.id())

    var device_buffer = ctx1.enqueue_create_buffer[dtype](32)
    var a = Tensor[dtype].arange(32)
    ctx2.enqueue_copy(device_buffer, a.data_ptr())
    print(device_buffer)
    with device_buffer.map_to_host() as host_buffer:
        print(host_buffer)
