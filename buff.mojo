from memory import AddressSpace
from builtin.device_passable import DevicePassable
from gpu.host import DeviceBuffer, HostBuffer


struct Buffer[
    dtype: DType = DType.float32,
    address_space: AddressSpace = AddressSpace.GENERIC,
](ImplicitlyCopyable & Movable & Sized & Stringable & Writable & Representable):
    var data: UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]

    fn __init__(out self):
        self.data = alloc[Scalar[Self.dtype]](1)

    @staticmethod
    fn get_type_name() -> String:
        return String(
            "Buffer[dtype = ",
            Self.dtype,
            "]",
        )

    comptime device_type: AnyType = Self

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    fn __init__(out self, elem: Scalar[Self.dtype]):
        self.data = alloc[Scalar[Self.dtype]](1)
        self.data[0] = elem

    fn __init__(
        out self,
        device_buffer: DeviceBuffer[Self.dtype],
    ):
        self.data = (
            device_buffer.unsafe_ptr()
            .mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

    fn __init__(
        out self,
        host_buffer: HostBuffer[Self.dtype],
    ):
        self.data = (
            host_buffer.unsafe_ptr()
            .mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

    # ========================================
    # Copy/Move semantics
    # ========================================

    fn __copyinit__(out self, other: Self):
        """Copy buffer - increment refcount if shared."""
        self.data = other.data

    fn __moveinit__(out self, deinit other: Self):
        """Move buffer - no refcount change."""
        self.data = other.data

    fn __del__(deinit self):
        """Destroy buffer - handle both shared and unshared cases."""
        if not self.data.__bool__():
            return

        self.data.free()

    @always_inline
    fn __len__(self) -> Int:
        return 1

    @always_inline
    fn get(self) -> Scalar[Self.dtype]:
        return self.data.load[width=1](0)

    @always_inline
    fn set(self, scalar: Scalar[Self.dtype]):
        self.data.store[width=1](0, scalar)

    @always_inline
    fn __add__(
        self: Buffer[Self.dtype], other: Buffer[Self.dtype]
    ) -> Buffer[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Buffer → __add__(other) is for numeric data types only",
        ]()
        var out = Buffer[Self.dtype]()
        out.set(self.get() + other.get())
        return out^

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Buffer[")
        writer.write(self.get())
        writer.write(", dtype=", self.dtype, "]")

    @no_inline
    fn __repr__(self) -> String:
        return self.__str__()


fn main() raises:
    print("passes good")
