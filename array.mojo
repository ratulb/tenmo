from common_utils import panic
from memory import memcpy, stack_allocation, AddressSpace
from builtin.device_passable import DevicePassable
from gpu.host import DeviceBuffer


@register_passable
struct Array(
    DevicePassable,
    ImplicitlyCopyable,
    Representable,
    Sized,
    Stringable,
    Writable,
):
    """A lightweight, register-passable growable array of integers.

    Optimized for tensor indexing operations with minimal overhead.
    Uses capacity-based growth to avoid frequent reallocations.
    """

    comptime device_type: AnyType = Self

    @staticmethod
    fn get_type_name() -> String:
        return String("Array")

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    var _data: UnsafePointer[Scalar[DType.int], MutExternalOrigin]
    var _size: Int  # Current number of elements
    var _capacity: Int  # Allocated capacity

    # ========== Construction ==========

    @always_inline("nodebug")
    fn __init__(out self):
        """Create empty array."""
        self._data = UnsafePointer[Scalar[DType.int], MutExternalOrigin]()
        self._size = 0
        self._capacity = 0

    @always_inline("nodebug")
    fn __init__(out self, *values: Int):
        """Create from variadic args: IntArray(1, 2, 3)."""
        var n = len(values)
        self._data = alloc[Scalar[DType.int]](n)
        self._size = n
        self._capacity = n
        for i in range(n):
            self._data[i] = values[i]

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        """Deep copy."""
        self._size = existing._size
        self._capacity = existing._capacity
        if existing._capacity > 0:
            self._data = alloc[Scalar[DType.int]](existing._capacity)
            memcpy(dest=self._data, src=existing._data, count=existing._size)
        else:
            self._data = {}

    @always_inline("nodebug")
    fn __del__(deinit self):
        """Free memory."""
        if self._data.__bool__():
            self._data.free()

    fn data_ptr[
        origin: Origin, address_space: AddressSpace, //
    ](ref [origin, address_space]self) -> UnsafePointer[
        Scalar[DType.int], origin, address_space=address_space
    ]:
        return (
            self._data.unsafe_mut_cast[origin.mut]()
            .unsafe_origin_cast[origin]()
            .address_space_cast[address_space]()
        )

    # ========== Static Constructors ==========

    @staticmethod
    @always_inline("nodebug")
    fn filled(size: Int, value: Int) -> Self:
        """Create array filled with value."""
        var result = Array.with_capacity(size)
        for _ in range(size):
            result.append(value)
        return result^

    @staticmethod
    @always_inline("nodebug")
    fn with_capacity(capacity: Int) -> Self:
        """Create with capacity but zero size."""
        var result = Self()
        if capacity > 0:
            result._data = alloc[Scalar[DType.int]](capacity)
            result._capacity = capacity
        return result^

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Number of elements."""
        return self._size

    @always_inline("nodebug")
    fn capacity(self) -> Int:
        """Allocated capacity."""
        return self._capacity

    @always_inline("nodebug")
    fn is_empty(self) -> Bool:
        """Check if empty."""
        return self._size == 0

    # ========== Access ==========

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        """Get element at index (supports negative indexing)."""
        var index = idx if idx >= 0 else idx + self._size
        if index < 0 or index >= self._size:
            panic("IntArray: index out of bounds")

        var result: Int = Int((self._data + index)[][0])
        return result

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, value: Int):
        """Set element at index (supports negative indexing)."""
        var index = idx if idx >= 0 else idx + self._size
        if index < 0 or index >= self._size:
            panic("IntArray: index out of bounds")
        # self._data[index] = value
        (self._data + index)[] = value

    # ========== Growth Operations ==========

    @always_inline("nodebug")
    fn reserve(mut self, required: Int):
        """Ensure capacity, reallocating if needed."""
        if required <= self._capacity:
            return

        # Grow by 1.5x or required, whichever is larger
        var new_cap = max(required, self._capacity * 3 // 2 + 1)
        var new_data = alloc[Scalar[DType.int]](new_cap)

        # Copy existing
        if self._size > 0:
            memcpy(dest=new_data, src=self._data, count=self._size)

        if self._data:
            self._data.free()
        self._data = new_data
        self._capacity = new_cap

    @always_inline
    fn append(mut self, value: Int):
        """Append element."""
        self.reserve(self._size + 1)
        self._data[self._size] = value
        self._size += 1

    @always_inline
    fn prepend(mut self, value: Int):
        """Prepend element."""
        self.reserve(self._size + 1)
        # Shift right
        for i in range(self._size, 0, -1):
            self._data[i] = self._data[i - 1]
        self._data[0] = value
        self._size += 1

    fn clear(mut self):
        """Clear all elements."""
        self._size = 0

    @no_inline
    fn __str__(self) -> String:
        """String representation."""
        if self._size == 0:
            return "[]"
        var result = String("[")
        for i in range(self._size):
            if i > 0:
                result += ", "
            result += String(self._data[i])
        result += "]"
        return result

    @no_inline
    fn __repr__(self) -> String:
        return self.__str__()

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())


fn main():
    a = Array(3, 4, 5)
    print(a)
