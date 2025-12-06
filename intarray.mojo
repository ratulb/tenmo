from common_utils import panic
from memory import memcpy


@register_passable
struct IntArray(ImplicitlyCopyable, Representable, Sized, Stringable, Writable):
    """A lightweight, register-passable growable array of integers.

    Optimized for tensor indexing operations with minimal overhead.
    Uses capacity-based growth to avoid frequent reallocations.
    """

    var _data: UnsafePointer[Int, MutAnyOrigin]
    var _size: Int  # Current number of elements
    var _capacity: Int  # Allocated capacity

    # ========== Construction ==========

    @always_inline("nodebug")
    fn __init__(out self):
        """Create empty array."""
        self._data = UnsafePointer[Int, MutAnyOrigin]()
        self._size = 0
        self._capacity = 0

    @always_inline("nodebug")
    fn __init__(out self, *values: Int):
        """Create from variadic args: IntArray(1, 2, 3)."""
        var n = len(values)
        #self._data = UnsafePointer[Int].alloc(n)
        self._data = alloc[Int](n)
        self._size = n
        self._capacity = n
        for i in range(n):
            self._data[i] = values[i]

    @always_inline("nodebug")
    fn __init__(out self, values: VariadicList[Int]):
        """Create from VariadicList : IntArray(1, 2, 3)."""
        var n = len(values)
        self._data = alloc[Int](n)
        self._size = n
        self._capacity = n
        for i in range(n):
            self._data[i] = values[i]

    @always_inline("nodebug")
    fn __init__(out self, values: List[Int]):
        """Create from List[Int]."""
        var n = len(values)
        self._data = alloc[Int](n)
        self._size = n
        self._capacity = n
        memcpy(dest=self._data, src=values._data, count=n)

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        """Deep copy."""
        self._size = existing._size
        self._capacity = existing._capacity
        if existing._capacity > 0:
            self._data = alloc[Int](existing._capacity)
            memcpy(dest=self._data, src=existing._data, count=existing._size)
        else:
            self._data = UnsafePointer[Int, MutAnyOrigin]()

    @always_inline("nodebug")
    fn __del__(deinit self):
        """Free memory."""
        if self._data:
            self._data.free()

    # ========== Static Constructors ==========

    @staticmethod
    @always_inline("nodebug")
    fn filled(size: Int, value: Int) -> Self:
        """Create array filled with value."""
        var result = IntArray.with_capacity(size)
        for _ in range(size):
            result.append(value)
        return result^

    @staticmethod
    @always_inline("nodebug")
    fn range(start: Int, end: Int, step: Int = 1) -> Self:
        """Create from range."""
        if step == 0:
            panic("IntArray.range: step cannot be zero")

        var size: Int
        if step > 0:
            size = max(0, (end - start + step - 1) // step)
        else:
            size = max(0, (start - end - step - 1) // (-step))

        var result = IntArray.with_capacity(size)
        var val = start

        if step > 0:
            while val < end:
                result.append(val)
                val += step
        else:
            while val > end:
                result.append(val)
                val += step

        return result^

    @staticmethod
    @always_inline("nodebug")
    fn with_capacity(capacity: Int) -> Self:
        """Create with capacity but zero size."""
        var result = Self()
        if capacity > 0:
            result._data = alloc[Int](capacity)
            result._capacity = capacity
        return result^

    @staticmethod
    @always_inline
    fn invert_permutation(perm: Self) -> Self:
        """Invert a permutation."""
        var inverted = IntArray.filled(len(perm), 0)
        for i in range(len(perm)):
            inverted[perm[i]] = i
        return inverted^

    # ========== Properties ==========

    @always_inline("nodebug")
    fn size(self) -> Int:
        """Number of elements."""
        return self._size

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
        return self._data[index]

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, value: Int):
        """Set element at index (supports negative indexing)."""
        var index = idx if idx >= 0 else idx + self._size
        if index < 0 or index >= self._size:
            panic("IntArray: index out of bounds")
        self._data[index] = value

    @always_inline("nodebug")
    fn __getitem__(self, slice: Slice) -> Self:
        """Get slice as new array."""
        var step = slice.step.or_else(1)

        if step == 0:
            panic("IntArray: slice step cannot be zero")

        # Handle start and stop based on step direction
        var start: Int
        var stop: Int

        if step > 0:
            start = slice.start.or_else(0)
            stop = slice.end.or_else(self._size)
        else:  # step < 0
            start = slice.start.or_else(self._size - 1)
            stop = slice.end.or_else(
                -self._size - 1
            )  # Use a value that's clearly before 0

        # Normalize negative indices
        if start < 0:
            start += self._size
        if stop < 0:
            stop += self._size

        # Calculate size based on step direction
        var size: Int
        if step > 0:
            size = max(0, (stop - start + step - 1) // step)
        else:
            size = max(0, (start - stop - step - 1) // (-step))

        var result = IntArray.with_capacity(size)
        var src_idx = start

        if step > 0:
            while src_idx < stop and src_idx < self._size:
                if src_idx >= 0:
                    result.append(self._data[src_idx])
                src_idx += step
        else:
            while src_idx > stop and src_idx >= 0:
                if src_idx < self._size:
                    result.append(self._data[src_idx])
                src_idx += step

        return result^

    # ========== Growth Operations ==========

    @always_inline("nodebug")
    fn reserve(mut self, required: Int):
        """Ensure capacity, reallocating if needed."""
        if required <= self._capacity:
            return

        # Grow by 1.5x or required, whichever is larger
        var new_cap = max(required, self._capacity * 3 // 2 + 1)
        var new_data = alloc[Int](new_cap)

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

    @always_inline("nodebug")
    fn __radd__(self, value: Int) -> Self:
        return IntArray(value).__add__(self)

    @always_inline("nodebug")
    fn __add__(self, value: Int) -> Self:
        """Return new array with value added."""
        var result = IntArray.with_capacity(self._size + 1)
        memcpy(dest=result._data, src=self._data, count=self._size)
        result._data[self._size] = value
        result._size = self._size + 1
        return result^

    @always_inline("nodebug")
    fn __add__(self, other: IntArray) -> Self:
        """Return new array with values of other IntArray added."""
        var result = Self.with_capacity(self._size + other._size)
        memcpy(dest=result._data, src=self._data, count=self._size)
        memcpy(dest=result._data + self._size, src=other._data, count=other._size)
        result._size = self._size + other._size
        return result^

    @always_inline("nodebug")
    fn __add__(self, other: List[Int]) -> Self:
        return self.__add__(IntArray(other))

    @always_inline("nodebug")
    fn __radd__(self, other: List[Int]) -> Self:
        return IntArray(other).__add__(self)

    @always_inline("nodebug")
    fn pop(mut self, index: Int = -1) -> Int:
        """Remove and return element."""
        if self._size < 1:
            panic("IntArray: cannot pop from empty array")
        var i = index if index >= 0 else index + self._size
        if i < 0 or i >= self._size:
            panic("IntArray: pop index out of bounds")

        var val = self._data[i]
        # Shift left
        for j in range(i, self._size - 1):
            self._data[j] = self._data[j + 1]
        self._size -= 1
        return val

    fn clear(mut self):
        """Clear all elements."""
        self._size = 0

    @always_inline("nodebug")
    fn replace(self, idx: Int, value: Int) -> Self:
        """Return copy with element at idx replaced."""
        var result = self
        result[idx] = value
        return result^

    @always_inline("nodebug")
    fn replace(self, indices: Self, values: Self) -> Self:
        """Return copy with multiple replacements."""
        var n = len(self)
        var m = len(indices)
        if m != len(values):
            panic("IntArray -> replace: indices and values must be same length")

        # Validate indices: no out-of-bounds, no duplicates
        for i in range(m):
            var idx = indices[i]
            if idx < 0 or idx >= n:
                panic(
                    "IntArray -> replace: index out of bounds: " + String(idx)
                )

        var result = self
        # Apply replacements
        for i in range(m):
            result[indices[i]] = values[i]

        return result^

    @always_inline("nodebug")
    fn insert(self, at: Int, value: Int) -> Self:
        """Insert value at position, return new IntArray."""
        if at < 0 or at > len(self):
            panic("IntArray -> insert - index out of bounds: " + String(at))

        var result = IntArray.with_capacity(len(self) + 1)
        for i in range(len(self) + 1):
            if i == at:
                result.append(value)
            if i < len(self):
                result.append(self[i])
        return result^

    @always_inline("nodebug")
    fn insert(self, indices: Self, values: Self) -> Self:
        """Insert values at multiple sorted indices."""
        if len(indices) != len(values):
            panic("IntArray -> insert: indices and values must be same length")

        var n = len(self)
        var m = len(indices)

        if n == 0:
            if m == 0:
                return IntArray()
            # Ensure indices = [0, 1, ..., m-1]
            var is_valid = True
            for i in range(m):
                if indices[i] != i:
                    is_valid = False
                    break
            if not is_valid:
                panic(
                    "IntArray -> insert: invalid indices for empty array"
                    " insertion"
                )
            return values

        var final_size = n + m
        # Create dense result
        var result = IntArray.with_capacity(final_size)
        var red_cursor = 0
        var orig_cursor = 0

        for i in range(final_size):
            if red_cursor < m and indices[red_cursor] == i:
                result.append(values[red_cursor])
                red_cursor += 1
            else:
                if orig_cursor >= n:
                    panic(
                        "IntArray -> insert: ran out of source values too early"
                    )
                result.append(self[orig_cursor])
                orig_cursor += 1

        return result^

    fn sort(mut self, asc: Bool = True):
        """Insertion sort in place."""
        for i in range(1, len(self)):
            var elem = self[i]
            var j = i
            while j > 0 and (elem < self[j - 1] if asc else elem > self[j - 1]):
                self[j] = self[j - 1]
                j -= 1
            self[j] = elem

    fn sorted(self, asc: Bool = True) -> Self:
        """Return sorted copy."""
        var result = self.copy()
        result.sort(asc)
        return result^

    # ========== Operations ==========
    @always_inline("nodebug")
    fn indices_of(self, val: Int) -> Self:
        """Return indices where value equals val."""
        var result = IntArray.with_capacity(len(self))
        for i in range(len(self)):
            if self[i] == val:
                result.append(i)
        return result

    @always_inline("nodebug")
    fn fill(mut self, value: Int):
        """Fill with value."""
        for i in range(self._size):
            self._data[i] = value

    @always_inline
    fn __contains__(self, value: Int) -> Bool:
        """Check if contains value."""
        for i in range(self._size):
            if self._data[i] == value:
                return True
        return False

    fn __eq__(self, other: Self) -> Bool:
        """Check equality."""
        if self._size != other._size:
            return False
        for i in range(self._size):
            if self._data[i] != other._data[i]:
                return False
        return True

    fn __eq__(self, other: List[Int]) -> Bool:
        """Check equality with List."""
        if self._size != len(other):
            return False
        for i in range(self._size):
            if self._data[i] != other[i]:
                return False
        return True

    # ========== Conversions ==========

    @always_inline("nodebug")
    fn tolist(self) -> List[Int]:
        """Convert to List[Int]."""
        var result = List[Int](capacity=Int(self._size))
        for i in range(self._size):
            result.append(self._data[i])
        return result^

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

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    # ========== Math Operations ==========

    @always_inline("nodebug")
    fn product(self) -> Int:
        """Product of all elements."""
        if self._size == 0:
            return 1
        var prod = 1
        for i in range(self._size):
            prod *= self._data[i]
        return prod

    @always_inline("nodebug")
    fn sum(self) -> Int:
        """Sum of all elements."""
        var s = 0
        for i in range(self._size):
            s += self._data[i]
        return s


    # ========== Functional Operations ==========

    @always_inline("nodebug")
    fn reverse(mut self):
        """Reverse in place."""
        for i in range(self._size // 2):
            var temp = self._data[i]
            self._data[i] = self._data[self._size - 1 - i]
            self._data[self._size - 1 - i] = temp

    @always_inline("nodebug")
    fn reversed(self) -> Self:
        """Return reversed copy."""
        var result = IntArray.with_capacity(self._size)
        for i in range(self._size - 1, -1, -1):
            result.append(self._data[i])
        return result^

    @always_inline("nodebug")
    fn __iter__(ref self) -> IntArrayIterator[origin_of(self)]:
        """Iterate over elements of the IntArray, returning immutable references.

        Returns:
            An iterator of immutable references to the IntArray elements.
        """
        return IntArrayIterator(0, Pointer(to=self))

    @always_inline("nodebug")
    fn __reversed__(
        ref self,
    ) -> IntArrayIterator[origin_of(self), False]:
        return IntArrayIterator[forward=False](len(self), Pointer(to=self))

    @always_inline("nodebug")
    fn zip(
        ref self, ref other: Self
    ) -> ZipIterator[origin_of(self), origin_of(other)]:
        """Iterate over elements of two IntArrays, returning pair of immutable references.

        Returns:
            An iterator of immutable references to the IntArray element pairs.
        """
        return ZipIterator(0, Pointer(to=self), Pointer(to=other))

    @always_inline("nodebug")
    fn zip_reversed(
        ref self, ref other: Self
    ) -> ZipIterator[origin_of(self), origin_of(other), False]:
        """Iterate over element pairs of two IntArrays, returning immutable references in reverse order.

        Returns:
            An iterator of immutable references to the IntArray elements in reverse order.
        """
        return ZipIterator[forward=False](
            min(len(self), len(other)), Pointer(to=self), Pointer(to=other)
        )

@register_passable
struct IntArrayIterator[
    origin: Origin[False],
    forward: Bool = True,
](Sized & Copyable):
    var index: Int
    var src: Pointer[IntArray, origin]

    fn __init__(out self, idx: Int, src: Pointer[IntArray, origin]):
        self.src = src
        self.index = idx

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self.index = existing.index
        self.src = existing.src

    fn __iter__(self) -> Self:
        return self.copy()

    fn __next__(mut self) -> Int:
        @parameter
        if forward:
            self.index += 1
            return self.src[][self.index - 1]
        else:
            self.index -= 1
            return self.src[][self.index]

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src[]) - self.index
        else:
            return self.index


struct ZipIterator[
    origin_this: Origin[False],
    origin_that: Origin[False],
    forward: Bool = True,
](Sized & Copyable):
    var index: Int
    var offset: Int
    var src_this: Pointer[IntArray, origin_this]
    var src_that: Pointer[IntArray, origin_that]

    fn __init__(
        out self,
        idx: Int,
        src_this: Pointer[IntArray, origin_this],
        src_that: Pointer[IntArray, origin_that],
    ):
        self.src_this = src_this
        self.src_that = src_that
        self.index = idx
        self.offset = abs(len(src_this[]) - len(src_that[]))

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self.index = existing.index
        self.offset = existing.offset
        self.src_this = existing.src_this
        self.src_that = existing.src_that

    fn __iter__(self) -> Self:
        return self.copy()

    fn __next__(mut self) -> Tuple[Int, Int]:
        @parameter
        if forward:
            self.index += 1
            return (
                self.src_this[][self.index - 1],
                self.src_that[][self.index - 1],
            )
        else:
            self.index -= 1
            if len(self.src_this[]) > len(self.src_that[]):
                return (
                    self.src_this[][self.index + self.offset],
                    self.src_that[][self.index],
                )
            else:
                return (
                    self.src_this[][self.index],
                    self.src_that[][self.index + self.offset],
                )

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return min(len(self.src_this[]), len(self.src_that[])) - self.index
        else:
            return self.index

fn main():
    ia = IntArray()
    ia.append(99)
    ia2 = IntArray()
    print(ia == ia2, ia2 == IntArray())
    print(ia)
