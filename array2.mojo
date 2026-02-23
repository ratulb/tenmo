from common_utils import panic
from builtin.device_passable import DevicePassable
from utils import StaticTuple
from intarray import IntArray
from shapes import Shape
from strides import Strides


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
    comptime max_dim = 8
    comptime device_type: AnyType = Self

    @staticmethod
    fn get_type_name() -> String:
        return String("Array[" + Self.max_dim.__str__() + "]")

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    var storage: StaticTuple[Int, Self.max_dim]
    var size: Int
    # ========== Construction ==========

    @always_inline("nodebug")
    fn __init__(out self):
        self.storage = StaticTuple[Int, Self.max_dim]()
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
        self = Self()
        self.size = len(intarray)
        for i in range(len(self)):
            self.storage[i] = intarray[i]

    @always_inline("nodebug")
    fn __init__(out self, shape: Shape):
        """Create from Shape(1, 2, 3)."""
        self = Self()
        self.size = shape.num_elements()
        print("Shape num elems: ", shape.num_elements())
        for i in range(len(self)):
            self.storage[i] = shape[i]

    @always_inline("nodebug")
    fn __init__(out self, strides: Strides):
        """Create from Strides(1, 2, 3)."""
        self = Self()
        self.size = len(strides)
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
        return Self.max_dim

    @always_inline("nodebug")
    fn is_empty(self) -> Bool:
        """Check if empty."""
        return self.size == 0

    # ========== Access ==========

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        """Get element at index (supports negative indexing)."""
        var index = idx if idx >= 0 else idx + self.size
        if index < 0 or index >= self.size:
            panic("Array: index out of bounds")
        return self.storage[index]

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, value: Int):
        """Set element at index (supports negative indexing)."""
        var index = idx if idx >= 0 else idx + self.size
        if index < 0 or index >= self.size:
            panic("Array: index out of bounds")
        self.storage[index] = value

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
    print(a)
