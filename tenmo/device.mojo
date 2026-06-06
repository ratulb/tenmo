from .common_utils import panic, IDGen
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import memcpy
from std.sys import has_accelerator, simd_width_of
from std.utils import Variant
from .ndbuffer import NDBuffer
from .shapes import Shape
from .buffers import Buffer
from std.sys.defines import get_defined_int

comptime DeviceType = Variant[CPU, GPU]


@fieldwise_init
struct Device(Equatable, ImplicitlyCopyable, Movable, Writable):
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


@fieldwise_init
struct CPU(Equatable, ImplicitlyCopyable, Movable, Writable):
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
struct GPU(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Essentially a shared DeviceContext."""

    var device_context: DeviceContext
    var id: Int64
    var _id: UInt

    def into(self) -> Device:
        return Device(self)

    def __init__(out self, device_id: Int = 0) raises:
        self.device_context = DeviceContext(device_id)
        self.id = Int64(device_id)
        self._id = IDGen.generate_id()

    def __init__(out self, *, copy: Self):
        self.device_context = copy.device_context.copy()
        self.id = copy.id
        self._id = copy._id

    def __init__(out self, deinit existing: Self):
        self.device_context = existing.device_context^
        self.id = existing.id
        self._id = existing._id

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


struct DeviceState[dtype: DType](
    Equatable & ImplicitlyCopyable & Movable & Sized
):
    """
    GPU device state for NDBuffer.
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

    def __init__(out self, deinit existing: Self):
        self.buffer = existing.buffer^
        self.gpu = existing.gpu^

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

    @staticmethod
    def map_where(
        ref ndb: NDBuffer[Self.dtype],
        pred: def(Scalar[Self.dtype]) thin -> Bool,
        value: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> DeviceState[Self.dtype]:
        ref src_device_state = ndb.device_state.value()
        var dst_device_state = DeviceState[Self.dtype](len(ndb), ndb.get_gpu())
        with src_device_state.buffer.map_to_host() as src, dst_device_state.buffer.map_to_host() as dst:
            for index in ndb.index_iterator():
                dst[index] = value.cast[Self.datatype]() if pred(
                    src[index].cast[Self.dtype]()
                ) else src[index].cast[Self.datatype]()
        if sync:
            dst_device_state.sync()

        return dst_device_state

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

    def fill(self, ref source: NDBuffer[Self.dtype], sync: Bool = False) raises:
        """Fill the DeviceBuffer from the source NDBuffer."""
        if source.is_on_gpu():
            if source.is_contiguous():
                # Both buffers are datatype — direct copy is safe
                source.device_state.value().buffer.enqueue_copy_to(self.buffer)
            else:
                with self.buffer.map_to_host() as host_buffer:
                    var next_index = 0
                    for index in source.index_iterator():
                        comptime if Self.dtype == DType.bool:
                            var v = source.get(index).cast[DType.bool]()
                            host_buffer[next_index] = rebind[
                                Scalar[Self.datatype]
                            ](UInt8(1) if v else UInt8(0))
                        else:
                            host_buffer[next_index] = rebind[
                                Scalar[Self.datatype]
                            ](source.get(index))
                        next_index += 1
            if sync:
                self.sync()
            return

        with self.buffer.map_to_host() as host_buffer:
            var device_ptr = host_buffer.unsafe_ptr()
            var src_ptr = source.data_ptr()

            if source.is_contiguous():
                var src_offset = source.offset
                src_ptr = src_ptr + src_offset
                var numels = source.numels()

                comptime if Self.dtype == DType.bool:
                    for i in range(numels):
                        device_ptr[i] = rebind[Scalar[Self.datatype]](
                            UInt8(1) if (src_ptr + i)[].cast[
                                DType.bool
                            ]() else UInt8(0)
                        )
                else:
                    memcpy(
                        dest=device_ptr,
                        src=src_ptr.bitcast[Scalar[Self.datatype]](),
                        count=numels,
                    )
            else:
                var next_index = 0
                for index in source.index_iterator():
                    comptime if Self.dtype == DType.bool:
                        device_ptr[next_index] = rebind[Scalar[Self.datatype]](
                            UInt8(1) if (src_ptr + index)[].cast[
                                DType.bool
                            ]() else UInt8(0)
                        )
                    else:
                        device_ptr[next_index] = rebind[Scalar[Self.datatype]](
                            (src_ptr + index)[]
                        )
                    next_index += 1
        if sync:
            self.sync()

    def into(
        self, shape: Shape
    ) raises -> NDBuffer[Self.dtype]:
        """
        Copy DeviceState content to a contiguous CPU NDBuffer with 0 offset.
        bool: converts uint8 0/1 back to bool.
        """

        comptime if Self.dtype == DType.bool:
            var numels = len(self)
            var cpu_buf = Buffer[DType.bool](numels)
            with self.buffer.map_to_host() as host_buffer:
                for i in range(numels):
                    cpu_buf[i] = host_buffer[i].cast[DType.uint8]() == UInt8(1)
            var casted_to_dtype = cpu_buf.to_dtype[Self.dtype]()
            return NDBuffer[Self.dtype](casted_to_dtype^, shape)
        else:
            # Map host buffer → CPU Buffer copy
            var cpu_buf = Buffer[Self.dtype](len(self))
            with self.buffer.map_to_host() as host_buffer:
                var src_ptr = host_buffer.unsafe_ptr()
                var dst_ptr = cpu_buf.unsafe_ptr()
                memcpy(
                    dest=dst_ptr,
                    src=src_ptr.bitcast[Scalar[Self.dtype]](),
                    count=len(self),
                )
            return NDBuffer[Self.dtype](cpu_buf^, shape)

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
                    (host_buffer[index].cast[DType.uint8]() == UInt8(1))
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
            return device_ptr.load[width=simdwidth](addr)

    def store[
        simdwidth: Int = simd_width_of[Self.datatype]()
    ](self, addr: Int, value: SIMD[Self.datatype, simdwidth]) raises:
        with self.buffer.map_to_host() as host_buffer:
            var device_ptr = host_buffer.unsafe_ptr()
            device_ptr.store[width=simdwidth](addr, value)

    def all_true(self: DeviceState[DType.bool]) -> Bool:
        try:
            var length = len(self)
            if length == 0:
                return True
            with self.buffer.map_to_host() as host_buffer:
                for i in range(length):
                    if host_buffer[i] != UInt8(1):
                        return False
            return True
        except e:
            print(e)
            return False

    def any_true(self: DeviceState[DType.bool]) -> Bool:
        try:
            var length = len(self)
            if length == 0:
                return False
            with self.buffer.map_to_host() as host_buffer:
                for i in range(length):
                    if host_buffer[i] == UInt8(1):
                        return True
            return False
        except e:
            print(e)
            return False
