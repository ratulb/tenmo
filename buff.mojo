from memory import AddressSpace


struct Buffer[
    dtype: DType = DType.float32,
    address_space: AddressSpace = AddressSpace.GENERIC,
](
    ImplicitlyCopyable
    & Movable
    & Sized
    # & Stringable
    # & Writable
    # & Representable
):
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

    fn __init__(out self, elem: Scalar[Self.dtype]):
        self.data = alloc[Scalar[Self.dtype]](1)
        self.data[0] = elem

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


fn main() raises:
    print("passes good")
