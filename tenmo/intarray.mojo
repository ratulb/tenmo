from .common_utils import panic
from std.memory import memcpy, memmove


struct IntArray(
    ImplicitlyCopyable,
    Iterable,
    RegisterPassable,
    Sized,
    Writable,
):
    """A lightweight, register-passable growable array of integers.

    Optimized for tensor indexing operations with minimal overhead.
    Uses capacity-based growth to avoid frequent reallocations.

    This array type is designed for high-performance computing scenarios
    where minimal memory overhead and register-passing capabilities are important.
    It provides dynamic array functionality with efficient memory management.
    """

    var _data: Optional[UnsafePointer[Int, MutAnyOrigin]]
    var _size: Int  # Current number of elements
    var _capacity: Int  # Allocated capacity

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = IntArrayIterator[iterable_origin, True]

    @always_inline("nodebug")
    def __init__(out self):
        """Create an empty IntArray instance.

        Initializes an array with zero size and zero capacity.
        Memory will be allocated as elements are appended.
        """
        self._data = {}
        self._size = 0
        self._capacity = 0

    @always_inline("nodebug")
    def __init__(out self, *values: Int):
        """Create IntArray from variadic integer arguments.

        Args:
            *values: Variable number of integers to initialize the array with

        Example:
            IntArray(1, 2, 3) creates an array with elements [1, 2, 3]
        """
        var n = len(values)
        self._data = alloc[Int](n)
        self._size = n
        self._capacity = n
        var data = self._data.unsafe_value()
        for i in range(n):
            data[i] = values[i]

    @always_inline("nodebug")
    def __init__(out self, values: VariadicList[Int, _]):
        """Create IntArray from a VariadicList of integers.

        Args:
            values: VariadicList of integers to initialize the array with
        """
        var n = len(values)
        self._data = alloc[Int](n)
        self._size = n
        self._capacity = n
        var data = self._data.unsafe_value()
        for i in range(n):
            data[i] = values[i]

    @always_inline("nodebug")
    def __init__(out self, values: List[Int]):
        """Create IntArray from a Mojo List.

        Args:
            values: Mojo List of integers to initialize the array with

        Note:
            This creates a copy of the data from the input list.
        """
        var n = len(values)
        self._data = alloc[Int](n)
        self._size = n
        self._capacity = n
        memcpy(dest=self._data.unsafe_value(), src=values._data, count=n)

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        """Copy constructor.

        Owning arrays get a deep copy (alloc + memcpy).
        View arrays get a shallow copy — just the pointer, zero alloc.
        """
        self._size = copy._size
        self._capacity = copy._capacity
        if copy.owning():
            self._data = alloc[Int](copy._capacity)
            memcpy(
                dest=self._data.unsafe_value(),
                src=copy._data.unsafe_value(),
                count=copy._size,
            )
        else:
            self._data = copy._data

    @always_inline("nodebug")
    def __del__(deinit self):
        """Free memory only if owning. Views are no-op."""
        if self.owning() and self._data:
            self._data.unsafe_value().free()

    @staticmethod
    @always_inline("nodebug")
    def filled(size: Int, value: Int) -> Self:
        """Create an IntArray filled with a specific value.

        Args:
            size: Number of elements in the array
            value: Value to fill each element with

        Returns:
            IntArray instance with all elements set to the specified value

        Example:
            IntArray.filled(5, 42) creates [42, 42, 42, 42, 42]
        """
        var result = IntArray.with_capacity(size)
        for _ in range(size):
            result.append(value)
        return result^

    @staticmethod
    @always_inline("nodebug")
    def range(start: Int, end: Int, step: Int = 1) -> Self:
        """Create an IntArray from a range of values.

        Args:
            start: Starting value (inclusive)
            end: Ending value (exclusive)
            step: Increment between values (default: 1)

        Returns:
            IntArray instance containing the range of values

        Raises:
            Panic if step is zero

        Example:
            IntArray.range(0, 5) -> [0, 1, 2, 3, 4]
            IntArray.range(0, 10, 2) -> [0, 2, 4, 6, 8]
            IntArray.range(10, 0, -2) -> [10, 8, 6, 4, 2]
        """
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
    def with_capacity(capacity: Int) -> Self:
        """Create an IntArray with pre-allocated capacity but zero size.

        Args:
            capacity: Pre-allocated capacity for the array

        Returns:
            IntArray instance with specified capacity but zero length

        Note:
            This avoids frequent reallocations when you know approximately
            how many elements you'll need to store.
        """
        var result = Self()
        if capacity > 0:
            result._data = alloc[Int](capacity)
            result._capacity = capacity
        return result^

    @staticmethod
    @always_inline
    def invert_permutation(perm: Self) -> Self:
        """Invert a permutation array.

        Args:
            perm: IntArray representing a permutation

        Returns:
            IntArray representing the inverse permutation

        Example:
            For perm = [2, 0, 1], the inverse is [1, 2, 0]
        """
        var inverted = IntArray.filled(len(perm), 0)
        for i in range(len(perm)):
            inverted[perm[i]] = i
        return inverted^

    @always_inline("nodebug")
    def size(self) -> Int:
        """Get the number of elements in the array.

        Returns:
            The current number of elements stored in the array.
        """
        return self._size

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """Get the number of elements in the array.

        Returns:
            The current number of elements stored in the array.
            This enables len() function support.
        """
        return self._size

    @always_inline("nodebug")
    def capacity(self) -> Int:
        """Get the allocated capacity of the array.

        Returns:
            The total number of elements that can be stored without reallocation.
            Returns 0 for views.
        """
        return self._capacity if self._capacity >= 0 else 0

    @always_inline("nodebug")
    def is_empty(self) -> Bool:
        """Check if the array is empty.

        Returns:
            True if the array contains no elements, False otherwise.
        """
        return self._size == 0

    @always_inline("nodebug")
    def owning(self) -> Bool:
        """Check if this array owns its memory.

        `_capacity >= 0` = owning (empty or allocated).
        `_capacity == -1` = view (borrowed pointer, zero-alloc copy, no-op destroy).
        Empty owning arrays (`_size=0, _capacity=0`) can still grow.
        """
        return self._capacity >= 0

    @always_inline("nodebug")
    def materialized(self) -> Self:
        """Ensure the returned array is owning (deep copy if view, no-op if owning).

        Use this when you need to outlive the original owner.
        """
        if self.owning():
            return self
        var result = Self.with_capacity(self._size)
        if self._size > 0:
            memcpy(
                dest=result._data.unsafe_value(),
                src=self._data.unsafe_value(),
                count=self._size,
            )
            result._size = self._size
        return result^

    @always_inline("nodebug")
    def __getitem__(ref self, idx: Int) -> ref[self] Int:
        """Get element at index.

        Args:
            idx: Index of element to retrieve. Supports negative indexing (e.g., -1 for last element).

        Returns:
            Reference to the element at the specified index.

        Raises:
            Panic if index is out of bounds.
        """
        var index = idx if idx >= 0 else idx + self._size
        if index < 0 or index >= self._size:
            panic("IntArray: index out of bounds")
        return (self._data.unsafe_value() + index)[]

    @always_inline("nodebug")
    def __setitem__(mut self, idx: Int, value: Int):
        """Set element at index.

        Args:
            idx: Index of element to set. Supports negative indexing.
            value: New value to store at the specified index.

        Raises:
            Panic if index is out of bounds or array is a view.
        """
        if not self.owning():
            panic("IntArray: can't modify a view")
        var index = idx if idx >= 0 else idx + self._size
        if index < 0 or index >= self._size:
            panic("IntArray: index out of bounds")
        (self._data.unsafe_value() + index)[] = value

    @always_inline("nodebug")
    def __getitem__(self, slice: Slice) -> Self:
        """Slice into a view (step==1) or deep copy (step != 1).

        Contiguous slices return a view — zero-alloc, just a borrowed pointer.
        Strided slices deep-copy. Supports negative indices.
        """
        var step = slice.step.or_else(1)

        if step == 0:
            panic("IntArray: slice step cannot be zero")

        var start: Int
        var stop: Int

        if step > 0:
            start = slice.start.or_else(0)
            stop = slice.end.or_else(self._size)
        else:
            start = slice.start.or_else(self._size - 1)
            stop = slice.end.or_else(-self._size - 1)

        if start < 0:
            start += self._size
        if stop < 0:
            stop += self._size

        var size: Int
        if step > 0:
            size = max(0, (stop - start + step - 1) // step)
        else:
            size = max(0, (start - stop - step - 1) // (-step))

        # Contiguous forward slice → return a view
        if step == 1:
            if size == 0:
                return Self()
            var result = Self()
            if self._data:
                result._data = self._data.unsafe_value() + start
            result._size = size
            result._capacity = -1  # marks as view
            return result^

        # Strided or negative step → deep copy
        var result = IntArray.with_capacity(size)
        var src_idx = start
        var data = self._data.unsafe_value()

        if step > 0:
            while src_idx < stop and src_idx < self._size:
                if src_idx >= 0:
                    result.append(data[src_idx])
                src_idx += step
        else:
            while src_idx > stop and src_idx >= 0:
                if src_idx < self._size:
                    result.append(data[src_idx])
                src_idx += step

        return result^

    @always_inline("nodebug")
    def reserve(mut self, required: Int):
        """Ensure capacity, reallocating if needed.

        Args:
            required: Minimum required capacity

        Raises:
            Panic if called on a non-owning (view) array.

        Note:
            Growth strategy: new capacity = max(required, current_capacity * 1.5 + 1)
        """
        if not self.owning():
            panic("IntArray: can't reserve on a view")
        if required <= self._capacity:
            return

        # Grow by 1.5x or required, whichever is larger
        var new_cap = max(required, self._capacity * 3 // 2 + 1)
        var new_data = alloc[Int](new_cap)

        # Copy existing
        if self._size > 0:
            memcpy(
                dest=new_data, src=self._data.unsafe_value(), count=self._size
            )

        if self._data:
            self._data.unsafe_value().free()
            self._data = {}
        self._data = new_data
        self._capacity = new_cap

    @always_inline
    def append(mut self, *values: Int):
        """Append one or more elements to the end of the array.

        Args:
            *values: One or more integers to append to the array

        Note:
            Automatically reserves additional capacity if needed using exponential growth strategy.
        """
        self.reserve(self._size + len(values))
        var data = self._data.unsafe_value()
        for i in range(len(values)):
            data[self._size + i] = values[i]
        self._size += len(values)

    @always_inline
    def prepend(mut self, value: Int):
        """Prepend an element to the beginning of the array.

        Args:
            value: Integer value to prepend

        Note:
            Uses memmove for O(n) block shift.
        """
        self.reserve(self._size + 1)
        var data = self._data.unsafe_value()
        if self._size > 0:
            memmove(dest=data + 1, src=data, count=self._size)
        data[0] = value
        self._size += 1

    @always_inline("nodebug")
    def __iadd__(mut self, other: Self):
        """In-place addition with another IntArray.

        Args:
            other: IntArray to add element-wise to this array

        Raises:
            Panic if arrays have unequal lengths or array is a view.
        """
        if not self.owning() and self._size > 0:
            panic("IntArray: can't mutate a view")
        if len(self) != len(other):
            panic("IntArray -> __iadd__(other): ", "unequal lengths")
        for i in range(len(self)):
            self[i] += other[i]

    @always_inline("nodebug")
    def __iadd__(mut self, other: List[Int]):
        """In-place addition with a Mojo List.

        Args:
            other: Mojo List of integers to add element-wise to this array

        Raises:
            Panic if array and list have unequal lengths or array is a view.
        """
        if not self.owning() and self._size > 0:
            panic("IntArray: can't mutate a view")
        if len(self) != len(other):
            panic("IntArray -> __iadd__(other[List]): ", "unequal lengths")
        for i in range(len(self)):
            self[i] += other[i]

    @always_inline("nodebug")
    def __radd__(self, value: Int) -> Self:
        """Reverse addition with an integer.

        Args:
            value: Integer to prepend to the array

        Returns:
            New IntArray with value at the start followed by elements of this array

        Note:
            Enables expressions like 5 + int_array.
        """
        return IntArray(value).__add__(self)

    @always_inline("nodebug")
    def __add__(self, value: Int) -> Self:
        """Return new array with value appended.

        Args:
            value: Integer to append to the array

        Returns:
            New IntArray with value at the end

        Note:
            Does not modify the original array (immutable operation).
        """
        var result = IntArray.with_capacity(self._size + 1)
        if self._size > 0:
            memcpy(
                dest=result._data.unsafe_value(),
                src=self._data.unsafe_value(),
                count=self._size,
            )
        result._data.unsafe_value()[self._size] = value
        result._size = self._size + 1
        return result^

    @always_inline("nodebug")
    def __add__(self, other: IntArray) -> Self:
        """Return new array with values of other IntArray concatenated.

        Args:
            other: IntArray to concatenate with this array

        Returns:
            New IntArray containing elements of this array followed by elements of other

        Note:
            Does not modify either original array (immutable operation).
        """
        var result = Self.with_capacity(self._size + other._size)
        if self._size > 0:
            memcpy(
                dest=result._data.unsafe_value(),
                src=self._data.unsafe_value(),
                count=self._size,
            )
        if other._size > 0:
            memcpy(
                dest=result._data.unsafe_value() + self._size,
                src=other._data.unsafe_value(),
                count=other._size,
            )
        result._size = self._size + other._size
        return result^

    @always_inline("nodebug")
    def __add__(self, other: List[Int]) -> Self:
        """Return new array with values of other List[Int] concatenated.

        Args:
            other: List of integers to concatenate with this array

        Returns:
            New IntArray containing elements of this array followed by elements of other

        Note:
            Internally converts the List to IntArray before concatenation.
        """
        return self.__add__(IntArray(other))

    @always_inline("nodebug")
    def __radd__(self, other: List[Int]) -> Self:
        """Reverse addition with a List[Int].

        Args:
            other: List of integers to prepend to this array

        Returns:
            New IntArray containing elements of other followed by elements of this array
        """
        return IntArray(other).__add__(self)

    @always_inline("nodebug")
    def pop(mut self, index: Int = -1) -> Int:
        """Remove and return element at specified index.

        Args:
            index: Index of element to remove (default: -1 for last element)

        Raises:
            Panic if array is empty, index is out of bounds, or array is a view.
        """
        if not self.owning():
            panic("IntArray: can't pop from a view")
        if self._size < 1:
            panic("IntArray: cannot pop from empty array")
        var i = index if index >= 0 else index + self._size
        if i < 0 or i >= self._size:
            panic("IntArray: pop index out of bounds")

        var data = self._data.unsafe_value()
        var val = data[i]
        var shift_count = self._size - i - 1
        if shift_count > 0:
            memmove(dest=data + i, src=data + i + 1, count=shift_count)
        self._size -= 1
        return val

    def clear(mut self):
        """Remove all elements from the array.

        Raises:
            Panic if array is a view.

        Note:
            Does not free memory; only resets size to zero.
            The capacity remains unchanged for efficient reuse.
        """
        if not self.owning():
            panic("IntArray: can't clear a view")
        self._size = 0

    @always_inline("nodebug")
    def replace(self, idx: Int, value: Int) -> Self:
        """Return owning copy with element at idx replaced."""
        var result = self.materialized()
        result[idx] = value
        return result^

    @always_inline("nodebug")
    def replace(self, indices: Self, values: Self) -> Self:
        """Return owning copy with multiple replacements."""
        var n = len(self)
        var m = len(indices)
        if m != len(values):
            panic("IntArray -> replace: indices and values must be same length")

        for i in range(m):
            var idx = indices[i]
            if idx < 0 or idx >= n:
                panic(
                    "IntArray -> replace: index out of bounds: " + String(idx)
                )

        var result = self.materialized()
        for i in range(m):
            result[indices[i]] = values[i]

        return result^

    @always_inline("nodebug")
    def insert(self, at: Int, value: Int) -> Self:
        """Insert value at position, return new IntArray."""
        if at < 0 or at > len(self):
            panic("IntArray -> insert - index out of bounds: " + String(at))

        var n = len(self)
        var result = Self.with_capacity(n + 1)
        var res_data = result._data.unsafe_value()
        var src_data = self._data.unsafe_value()
        if at > 0:
            memcpy(dest=res_data, src=src_data, count=at)
        res_data[at] = value
        if n > at:
            memcpy(dest=res_data + at + 1, src=src_data + at, count=n - at)
        result._size = n + 1
        return result^

    @always_inline("nodebug")
    def insert(self, indices: Self, values: Self) -> Self:
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

    def sort(mut self, asc: Bool = True):
        """Insertion sort in place.

        Raises:
            Panic if array is a view (and non-empty).
        """
        if not self.owning() and self._size > 0:
            panic("IntArray: can't sort a view")
        if self._size < 2:
            return
        for i in range(1, self._size):
            var elem = self[i]
            var j = i
            while j > 0 and (elem < self[j - 1] if asc else elem > self[j - 1]):
                self[j] = self[j - 1]
                j -= 1
            self[j] = elem

    def sorted(self, asc: Bool = True) -> Self:
        """Return sorted copy."""
        var result = self.materialized()
        result.sort(asc)
        return result^

    @always_inline("nodebug")
    def indices_of(self, val: Int) -> Self:
        """Return indices where value equals val."""
        var result = IntArray.with_capacity(len(self))
        for i in range(len(self)):
            if self[i] == val:
                result.append(i)
        return result

    @always_inline("nodebug")
    def fill(mut self, value: Int):
        """Fill with value.

        Raises:
            Panic if array is a view (and non-empty).
        """
        if not self.owning() and self._size > 0:
            panic("IntArray: can't fill a view")
        if self._size == 0:
            return
        var data = self._data.unsafe_value()
        for i in range(self._size):
            data[i] = value

    @always_inline
    def __contains__(self, value: Int) -> Bool:
        """Check if contains value."""
        var data = self._data.unsafe_value()
        for i in range(self._size):
            if data[i] == value:
                return True
        return False

    def __eq__(self, other: Self) -> Bool:
        """Check equality."""
        if self._size != other._size:
            return False
        var data = self._data.unsafe_value()
        var other_data = other._data.unsafe_value()
        for i in range(self._size):
            if data[i] != other_data[i]:
                return False
        return True

    def __eq__(self, other: List[Int]) -> Bool:
        """Check equality with List."""
        if self._size != len(other):
            return False
        var data = self._data.unsafe_value()
        for i in range(self._size):
            if data[i] != other[i]:
                return False
        return True

    @always_inline("nodebug")
    def tolist(self) -> List[Int]:
        """Convert to List[Int]."""
        var result = List[Int](capacity=Int(self._size))
        var data = self._data.unsafe_value()
        for i in range(self._size):
            result.append(data[i])
        return result^

    @no_inline
    def __str__(self) -> String:
        """String representation."""
        if self._size == 0:
            return "[]"
        var result = String("[")
        var data = self._data.unsafe_value()
        for i in range(self._size):
            if i > 0:
                result += ", "
            result += String(data[i])
        result += "]"
        return result

    @no_inline
    def __repr__(self) -> String:
        """Get official string representation of the IntArray.

        Returns:
            String representation suitable for debugging.
        """
        return self.__str__()

    @no_inline
    def write_to[W: Writer](self, mut writer: W):
        """Write IntArray to a writer.

        Args:
            writer: Writer to write to
        """
        writer.write(self.__str__())

    @always_inline("nodebug")
    def product(self) -> Int:
        """Calculate the product of all elements.

        Returns:
            Product of all elements. Returns 1 for empty arrays (multiplicative identity).
        """
        if self._size == 0:
            return 1
        var prod = 1
        var data = self._data.unsafe_value()
        for i in range(self._size):
            prod *= data[i]
        return prod

    @always_inline("nodebug")
    def sum(self) -> Int:
        """Calculate the sum of all elements.

        Returns:
            Sum of all elements. Returns 0 for empty arrays.
        """
        var s = 0
        var data = self._data.unsafe_value()
        for i in range(self._size):
            s += data[i]
        return s

    @always_inline("nodebug")
    def reverse(mut self):
        """Reverse the array in place.

        Raises:
            Panic if array is a view (and non-empty).

        Note:
            Modifies the array in place, reversing the order of elements.
            Time complexity is O(n).
        """
        if not self.owning() and self._size > 0:
            panic("IntArray: can't reverse a view")
        if self._size == 0:
            return
        var data = self._data.unsafe_value()
        for i in range(self._size // 2):
            var temp = data[i]
            data[i] = data[self._size - 1 - i]
            data[self._size - 1 - i] = temp

    @always_inline("nodebug")
    def reversed(self) -> Self:
        """Return a reversed copy of the array.

        Returns:
            New IntArray with elements in reverse order

        Note:
            Does not modify the original array (immutable operation).
        """
        var result = IntArray.with_capacity(self._size)
        var data = self._data.unsafe_value()
        for i in range(self._size - 1, -1, -1):
            result.append(data[i])
        return result^

    @always_inline("nodebug")
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Get iterator over elements of the IntArray.

        Returns:
            Iterator over elements of the IntArray.
        """
        return {0, Pointer(to=self)}

    @always_inline("nodebug")
    def __reversed__(
        ref self,
    ) -> IntArrayIterator[origin_of(self), False]:
        """Get reverse iterator over elements of the IntArray.

        Returns:
            Reverse iterator over elements of the IntArray.
        """
        return IntArrayIterator[forward=False](len(self), Pointer(to=self))

    @always_inline("nodebug")
    def zip(
        ref self, ref other: Self
    ) -> ZipIterator[origin_of(self), origin_of(other)]:
        """Iterate over elements of two IntArrays, returning pair of immutable references.

        Returns:
            An iterator of immutable references to the IntArray element pairs.
        """
        return ZipIterator(0, Pointer(to=self), Pointer(to=other))

    @always_inline("nodebug")
    def zip_reversed(
        ref self, ref other: Self
    ) -> ZipIterator[origin_of(self), origin_of(other), False]:
        """Iterate over element pairs of two IntArrays, returning immutable references in reverse order.

        Returns:
            An iterator of immutable references to the IntArray elements in reverse order.
        """
        return ZipIterator[forward=False](
            min(len(self), len(other)), Pointer(to=self), Pointer(to=other)
        )


@fieldwise_init
struct IntArrayIterator[
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
    var src: Pointer[IntArray, Self.origin]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
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

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len: Int

        comptime if Self.forward:
            iter_len = len(self.src[]) - self.index
        else:
            iter_len = self.index

        return (iter_len, {iter_len})

    @always_inline
    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    def __len__(self) -> Int:
        comptime if Self.forward:
            return len(self.src[]) - self.index
        else:
            return self.index


struct ZipIterator[
    origin_this: ImmutOrigin,
    origin_that: ImmutOrigin,
    forward: Bool = True,
](Sized & Copyable):
    var index: Int
    var offset: Int
    var src_this: Pointer[IntArray, Self.origin_this]
    var src_that: Pointer[IntArray, Self.origin_that]

    def __init__(
        out self,
        idx: Int,
        src_this: Pointer[IntArray, Self.origin_this],
        src_that: Pointer[IntArray, Self.origin_that],
    ):
        self.src_this = src_this
        self.src_that = src_that
        self.index = idx
        self.offset = abs(len(src_this[]) - len(src_that[]))

    @always_inline("nodebug")
    def __copyinit__(out self, copy: Self):
        self.index = copy.index
        self.offset = copy.offset
        self.src_this = copy.src_this
        self.src_that = copy.src_that

    def __iter__(self) -> Self:
        return self.copy()

    def __next__(mut self) -> Tuple[Int, Int]:
        comptime if Self.forward:
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
    def __has_next__(self) -> Bool:
        return self.__len__() > 0

    def __len__(self) -> Int:
        comptime if Self.forward:
            return min(len(self.src_this[]), len(self.src_that[])) - self.index
        else:
            return self.index
