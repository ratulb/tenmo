from common_utils import panic
from builtin.device_passable import DevicePassable
from utils import StaticTuple
from intarray import IntArray
from shapes import Shape
from strides import Strides
from mnemonics import max_rank


@register_passable
struct Array(
    Defaultable,
    DevicePassable,
    ImplicitlyCopyable,
    Representable,
    Sized,
    Stringable,
    Writable,
):
    comptime device_type: AnyType = Self

    @staticmethod
    fn get_type_name() -> String:
        return String("Array[max_rank:" + max_rank.__str__() + "]")

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    var storage: StaticTuple[Int, max_rank]
    var size: Int
    # ========== Construction ==========

    @always_inline("nodebug")
    fn __init__(out self):
        self.storage = StaticTuple[Int, max_rank]()
        self.size = 0

    @always_inline("nodebug")
    fn __init__(out self, *values: Int):
        """Create from variadic args (1, 2, 3)."""
        self = Self()
        self.size = len(values)
        for i in range(len(self)):
            self.storage[i] = values[i]

    @always_inline("nodebug")
    fn __init__(out self, intarray: IntArray):
        """Create from IntArray(1, 2, 3)."""
        var length = len(intarray)
        if length > max_rank:
            panic(
                "Only dims upto",
                max_rank.__str__(),
                "is supported.",
                "IntArray size exceeds that",
            )

        self = Self()
        self.size = len(intarray)
        for i in range(len(self)):
            self.storage[i] = intarray[i]

    @always_inline("nodebug")
    fn __init__(out self, shape: Shape):
        """Create from Shape(1, 2, 3)."""
        var length = len(shape)
        if length > max_rank:
            panic(
                "Only dims upto",
                max_rank.__str__(),
                "is supported.",
                "Shape dim exceeds that",
            )

        self = Self()
        self.size = length
        for i in range(len(self)):
            self.storage[i] = shape[i]

    @always_inline("nodebug")
    fn __init__(out self, strides: Strides):
        """Create from Strides(1, 2, 3)."""
        var length = len(strides)
        if length > max_rank:
            panic(
                "Only dims upto",
                max_rank.__str__(),
                "is supported.",
                "Strides length exceeds that",
            )

        self = Self()
        self.size = length
        for i in range(len(self)):
            self.storage[i] = strides[i]

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        """Deep copy."""
        self.storage = existing.storage
        self.size = existing.size

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Number of elements."""
        return self.size

    @always_inline("nodebug")
    fn capacity(self) -> Int:
        """Allocated capacity."""
        return max_rank

    @always_inline("nodebug")
    fn is_empty(self) -> Bool:
        """Check if empty."""
        return self.size == 0

    fn __contains__(self, value: Int) -> Bool:
        for i in range(len(self)):
            if value == self[i]:
                return True
        return False

    # ========== Access ==========

    fn product(self) -> Int:
        var result = 1
        for i in range(len(self)):
            result *= self[i]
        return result

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        """Get element at index (supports negative indexing)."""
        var index = idx if idx >= 0 else idx + self.size
        debug_assert(
            index >= 0 and index < self.size,
            "Array -> __getitem__: index out of bounds",
        )
        return self.storage[index]

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, value: Int):
        """Set element at index (supports negative indexing)."""
        var index = idx if idx >= 0 else idx + self.size
        debug_assert(
            index >= 0 and index < self.size,
            "Array -> __setitem__: index out of bounds",
        )
        self.storage[index] = value

    @always_inline("nodebug")
    fn append(mut self, value: Int):
        """Append a value. Panics if max_rank capacity is exceeded."""
        if self.size >= max_rank:
            panic(
                "Array -> append: capacity exceeded.",
                "max_rank is",
                max_rank.__str__(),
            )
        self.storage[self.size] = value
        self.size += 1

    fn clear(mut self):
        """Clear all elements."""
        self.size = 0

    @no_inline
    fn __str__(self) -> String:
        """String representation."""
        if self.size == 0:
            return "[]"
        var result = String("[")
        for i in range(self.size):
            if i > 0:
                result += ", "
            result += String(self.storage[i])
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
    print(4 in a, 1 in a, a.product())
