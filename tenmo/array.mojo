from common_utils import panic
from std.builtin.device_passable import DevicePassable
from std.utils import StaticTuple
from intarray import IntArray
import mnemonics
from std.sys.defines import get_defined_int


comptime max_rank = get_defined_int["MAX_RANK", mnemonics.max_rank]()


struct Array(
    Defaultable,
    DevicePassable,
    Equatable,
    ImplicitlyCopyable,
    Iterable,
    RegisterPassable,
    Sized,
    Writable,
):
    comptime device_type: AnyType = Self

    @staticmethod
    fn get_type_name() -> String:
        return String("Array[max_rank:" + String(max_rank) + "]")

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
        self = Self()
        self.size = len(values)
        for i in range(len(self)):
            self.storage[i] = values[i]

    @always_inline("nodebug")
    fn __init__(out self, values: VariadicList[Int, _]):
        var n = len(values)
        if n > max_rank:
            panic("Array: VariadicList size exceeds max_rank", String(max_rank))
        self = Self()
        self.size = n
        for i in range(n):
            self.storage[i] = values[i]

    @always_inline("nodebug")
    fn __init__(out self, values: List[Int]):
        var n = len(values)
        if n > max_rank:
            panic("Array: List size exceeds max_rank", String(max_rank))
        self = Self()
        self.size = n
        for i in range(n):
            self.storage[i] = values[i]

    @always_inline("nodebug")
    fn __init__(out self, intarray: IntArray):
        var length = len(intarray)
        if length > max_rank:
            panic("Array: IntArray size exceeds max_rank", String(max_rank))
        self = Self()
        self.size = length
        for i in range(length):
            self.storage[i] = intarray[i]

    @always_inline("nodebug")
    fn __copyinit__(out self, copy: Self):
        self.storage = copy.storage
        self.size = copy.size

    # ========== Static Constructors ==========

    @staticmethod
    @always_inline("nodebug")
    fn filled(size: Int, value: Int) -> Self:
        """Create array of given size filled with value."""
        if size > max_rank:
            panic("Array.filled: size exceeds max_rank", String(max_rank))
        var result = Self()
        result.size = size
        for i in range(size):
            result.storage[i] = value
        return result

    @staticmethod
    @always_inline("nodebug")
    fn with_capacity(capacity: Int) -> Self:
        """No-op for Array — always has max_rank capacity. Returns empty Array.
        """
        return Self()

    # ========== Properties ==========

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return self.size

    @always_inline("nodebug")
    fn capacity(self) -> Int:
        return max_rank

    @always_inline("nodebug")
    fn is_empty(self) -> Bool:
        return self.size == 0

    # ========== Access ==========

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        var index = idx if idx >= 0 else idx + self.size
        debug_assert(
            index >= 0 and index < self.size,
            "Array -> __getitem__: index out of bounds",
        )
        return self.storage[index]

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, value: Int):
        var index = idx if idx >= 0 else idx + self.size
        debug_assert(
            index >= 0 and index < self.size,
            "Array -> __setitem__: index out of bounds",
        )
        self.storage[index] = value

    # ========== Growth / Mutation ==========

    @always_inline("nodebug")
    fn append(mut self, value: Int):
        if self.size >= max_rank:
            panic(
                "Array -> append: capacity exceeded. max_rank is",
                String(max_rank),
            )
        self.storage[self.size] = value
        self.size += 1

    @always_inline
    fn prepend(mut self, value: Int):
        """Prepend element — shifts existing elements right."""
        if self.size >= max_rank:
            panic(
                "Array -> prepend: capacity exceeded. max_rank is",
                String(max_rank),
            )
        for i in range(self.size, 0, -1):
            self.storage[i] = self.storage[i - 1]
        self.storage[0] = value
        self.size += 1

    @always_inline("nodebug")
    fn pop(mut self, index: Int = -1) -> Int:
        """Remove and return element at index."""
        if self.size < 1:
            panic("Array: cannot pop from empty array")
        var i = index if index >= 0 else index + self.size
        if i < 0 or i >= self.size:
            panic("Array: pop index out of bounds")
        var val = self.storage[i]
        for j in range(i, self.size - 1):
            self.storage[j] = self.storage[j + 1]
        self.size -= 1
        return val

    fn clear(mut self):
        self.size = 0

    @always_inline("nodebug")
    fn fill(mut self, value: Int):
        """Fill all existing elements with value (does not change size)."""
        for i in range(self.size):
            self.storage[i] = value

    # ========== Arithmetic ==========

    @always_inline("nodebug")
    fn __iadd__(mut self, other: Self):
        """Element-wise += with other Array."""
        if len(self) != len(other):
            panic("Array -> __iadd__: unequal lengths")
        for i in range(len(self)):
            self.storage[i] += other[i]

    @always_inline("nodebug")
    fn __iadd__(mut self, other: List[Int]):
        """Element-wise += with List[Int]."""
        if len(self) != len(other):
            panic("Array -> __iadd__(List): unequal lengths")
        for i in range(len(self)):
            self.storage[i] += other[i]

    @always_inline("nodebug")
    fn __add__(self, value: Int) -> Self:
        """Return new Array with value appended."""
        var result = self
        result.append(value)
        return result

    @always_inline("nodebug")
    fn __add__(self, other: Self) -> Self:
        """Return new Array concatenating self + other."""
        if self.size + other.size > max_rank:
            panic("Array -> __add__: result exceeds max_rank")
        var result = self
        for i in range(other.size):
            result.append(other.storage[i])
        return result

    @always_inline("nodebug")
    fn __add__(self, other: List[Int]) -> Self:
        return self.__add__(Self(other))

    @always_inline("nodebug")
    fn __radd__(self, value: Int) -> Self:
        return Self(value).__add__(self)

    @always_inline("nodebug")
    fn __radd__(self, other: List[Int]) -> Self:
        return Self(other).__add__(self)

    @always_inline("nodebug")
    fn fma(self, other: Array, offset: Int = 0) -> Int:
        """Dot product of self and other, plus offset.
        Typical use: strides.fma(indices, base_offset) -> flat index."""
        debug_assert(self.size == other.size, "Array -> fma: size mismatch")
        var result = offset
        for i in range(self.size):
            result += self.storage[i] * other.storage[i]
        return result

    @always_inline("nodebug")
    fn fma(self, other: IntArray, offset: Int = 0) -> Int:
        debug_assert(self.size == len(other), "Array -> fma: size mismatch")
        var result = offset
        for i in range(self.size):
            result += self.storage[i] * other[i]
        return result

    # ========== Search / Query ==========

    fn __contains__(self, value: Int) -> Bool:
        for i in range(self.size):
            if self.storage[i] == value:
                return True
        return False

    fn __eq__(self, other: Self) -> Bool:
        if self.size != other.size:
            return False
        for i in range(self.size):
            if self.storage[i] != other.storage[i]:
                return False
        return True

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __eq__(self, other: List[Int]) -> Bool:
        if self.size != len(other):
            return False
        for i in range(self.size):
            if self.storage[i] != other[i]:
                return False
        return True

    # ========== Conversions ==========

    @always_inline("nodebug")
    fn tolist(self) -> List[Int]:
        var result = List[Int](capacity=self.size)
        for i in range(self.size):
            result.append(self.storage[i])
        return result^

    @always_inline("nodebug")
    fn copy(self) -> Self:
        return self  # register-passable structs copy by value

    # ========== Math ==========

    fn product(self) -> Int:
        var result = 1
        for i in range(self.size):
            result *= self.storage[i]
        return result

    fn sum(self) -> Int:
        var s = 0
        for i in range(self.size):
            s += self.storage[i]
        return s

    # ========== Functional ==========

    @always_inline("nodebug")
    fn reverse(mut self):
        """Reverse in place."""
        for i in range(self.size // 2):
            var tmp = self.storage[i]
            self.storage[i] = self.storage[self.size - 1 - i]
            self.storage[self.size - 1 - i] = tmp

    # ========== Iteration ==========

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = ArrayIterator[iterable_origin, True]

    @always_inline("nodebug")
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return {0, Pointer(to=self)}

    @always_inline("nodebug")
    fn __reversed__(ref self) -> ArrayIterator[origin_of(self), False]:
        return ArrayIterator[forward=False](len(self), Pointer(to=self))

    # ========== String / IO ==========

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        if self.size == 0:
            writer.write("[]")
            return
        writer.write("[")
        for i in range(self.size):
            if i > 0:
                writer.write(", ")
            writer.write(self.storage[i])
        writer.write("]")

    @no_inline
    fn __str__(self) -> String:
        return String(self)

    @no_inline
    fn __repr__(self) -> String:
        return self.__str__()


# ========== Iterator ==========


@fieldwise_init
struct ArrayIterator[
    mut: Bool,
    //,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator, RegisterPassable, Sized):
    comptime Element = Int

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var index: Int
    var src: Pointer[Array, Self.origin]

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    fn __next__(mut self) raises StopIteration -> Int:
        comptime if Self.forward:
            if self.index >= len(self.src[]):
                raise StopIteration()
            self.index += 1
            return self.src[][self.index - 1]
        else:
            if self.index <= 0:
                raise StopIteration()
            self.index -= 1
            return self.src[][self.index]

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len: Int
        comptime if Self.forward:
            iter_len = len(self.src[]) - self.index
        else:
            iter_len = self.index
        return (iter_len, {iter_len})

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        comptime if Self.forward:
            return len(self.src[]) - self.index
        else:
            return self.index


fn main():
    a = Array(3, 4, 5)
    print(4 in a, 1 in a, a.product())
