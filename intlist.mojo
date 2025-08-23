from memory import memcpy, memset, Pointer
from os import abort
from common_utils import log_debug, panic


@register_passable
struct IntList(Sized & Copyable & Stringable & Representable & Writable):
    """A memory-efficient, register-passable, dynamic array of Ints. Would abort on any erroneous condition.
    """

    alias Empty = IntList()
    var data: UnsafePointer[Int]
    var size: Int
    var capacity: Int

    fn __init__(out self):
        """Constructs an empty IntList."""
        self.data = UnsafePointer[Int]()
        self.capacity = 0
        self.size = 0

    @always_inline("nodebug")
    fn __init__(out self, *elems: Int):
        """Initialize a new `IntList` with variadic number of elements.
        Args:
            elems: Number of Ints to allocate space for.
        """
        self.data = UnsafePointer[Int].alloc(len(elems))
        self.size = len(elems)
        self.capacity = len(elems)
        for idx in range(len(elems)):
            (self.data + idx)[] = elems[idx]

    @always_inline("nodebug")
    fn __init__(out self, elems: VariadicList[Int]):
        """Initialize a new `IntList` with the with VariadicList[Int] - used primarily in Shape.
        Args:
            elems: Number of Ints to allocate space for.
        """
        self.data = UnsafePointer[Int].alloc(len(elems))
        self.size = len(elems)
        self.capacity = len(elems)
        for idx in range(len(elems)):
            (self.data + idx)[] = elems[idx]

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        """Initialize by copying an existing `IntList`.
        Args:
            existing: The source array to copy from.
        """
        self.size = existing.size
        self.capacity = existing.capacity
        self.data = UnsafePointer[Int].alloc(existing.capacity)
        memcpy(self.data, existing.data, existing.size)

    @staticmethod
    fn new(src: List[Int]) -> IntList:
        result = IntList.with_capacity(len(src))
        memcpy(result.data, src._data, len(src))
        result.size = len(src)
        return result

    @staticmethod
    fn range_list(n: Int) -> IntList:
        var out = IntList.with_capacity(n)
        for i in range(n):
            out.append(i)
        return out

    @staticmethod
    fn filled(length: Int, value: Int) -> IntList:
        return Self.with_capacity(length, Optional(value))

    @staticmethod
    fn with_capacity(capacity: Int, fill: Optional[Int] = None) -> IntList:
        array = Self()
        array.data = UnsafePointer[Int].alloc(capacity)
        array.capacity = capacity
        array.size = 0
        if fill:
            for idx in range(capacity):
                (array.data + idx)[] = fill.value()
            array.size = capacity
        return array

    fn product(self) -> Int:
        result = 1
        for each in self:
            result *= each
        return result

    fn sum(self) -> Int:
        result = 0
        for each in self:
            result += each
        return result

    fn has_duplicates(self) -> Bool:
        if len(self) <= 1:
            return False
        var sorted_list = self.sorted()

        for i in range(len(sorted_list) - 1):
            if sorted_list[i] == sorted_list[i + 1]:
                return True

        return False

    fn sort_and_deduplicate(mut self):
        if len(self) == 0:
            return
        self.sort()
        write_index = 1
        for read_index in range(1, len(self)):
            if self[read_index] != self[write_index - 1]:
                self[write_index] = self[read_index]
                write_index += 1
        self.size = write_index

    fn sorted(self, asc: Bool = True) -> IntList:
        copied = self.copy()
        copied.sort(asc)
        return copied

    fn sort(self, asc: Bool = True):
        for i in range(1, len(self)):
            elem = self[i]
            j = i
            while j > 0 and (elem < self[j - 1] if asc else elem > self[j - 1]):
                (self.data + j)[] = self[j - 1]
                j -= 1
            (self.data + j)[] = elem

    fn select(self, indices: IntList) -> IntList:
        n = len(self)
        result = IntList.with_capacity(len(indices))

        for i in indices:
            if i < 0 or i >= n:
                abort(
                    "Index out of bounds in IntList - select: "
                    + String(i)
                    + ", not in [0, "
                    + String(n)
                    + ")"
                )
            result.append(self[i])
        return result

    fn indices_of(self, val: Int) -> IntList:
        """Returns a new IntList containing all indices where self[i] == val."""
        result = IntList.with_capacity(self.size)

        for i in range(self.size):
            if (self.data + i)[] == val:
                result.append(i)
        return result

    fn count(self, elem: Int) -> Int:
        times = 0
        for each in self:
            if each == elem:
                times += 1
        return times

    fn any(self, cond: fn (Int) -> Bool) -> Bool:
        for elem in self:
            if cond(elem):
                return True
        return False

    @always_inline
    fn is_strictly_increasing_from_zero(self) -> Bool:
        for i in range(len(self)):
            if self[i] != i:
                print(
                    (
                        "IntList -> is_strictly_increasing_from_zero: non"
                        " increasing index and value: "
                    ),
                    i,
                    self[i],
                )
                return False
        return True

    fn insert(self, indices: IntList, values: IntList) -> IntList:
        if len(indices) != len(values):
            abort("IntList -> insert: indices and values must be same length")

        n = len(self)
        m = len(indices)
        if n == 0:
            if m == 0:
                return IntList()
            # Ensure indices = [0, 1, ..., m-1]
            if not indices.is_strictly_increasing_from_zero():
                abort(
                    "IntList -> insert: invalid idices for empty list insertion"
                )
            return values

        final_size = n + m
        # Step 3: Create dense result
        result = IntList.with_capacity(final_size)
        red_cursor = 0
        orig_cursor = 0
        for i in range(final_size):
            if red_cursor < m and indices[red_cursor] == i:
                result.append(values[red_cursor])
                red_cursor += 1
            else:
                if orig_cursor >= n:
                    abort(
                        "IntList -> insert: ran out of source values too early"
                    )
                result.append(self[orig_cursor])
                orig_cursor += 1
        return result

    fn insert(self, at: Int, value: Int) -> IntList:
        # Insert `value` at position `at` in `self`, return a new IntList
        # `at` could be start, middle or end
        if at < 0 or at > len(self):
            abort("IntList -> insert - index out of bounds: " + String(at))

        result = IntList.with_capacity(len(self) + 1)
        for i in range(result.capacity):
            if i == at:
                result.append(value)
            if i < len(self):
                result.append(self[i])

        return result

    @always_inline("nodebug")
    # fn __del__(owned self):
    fn free(deinit self):
        """Destroy the `IntList` and free its memory."""
        if self.data:
            log_debug("Calling IntList __del__")
            for i in range(len(self)):
                (self.data + i).destroy_pointee()
            self.data.free()

    fn __mul__(self: IntList, factor: Int) -> IntList:
        if factor < 1 or self.data.__as_bool__() == False:
            return IntList()
        result = IntList.with_capacity(len(self) * factor)
        for i in range(factor):
            result.copy_from(i * len(self), self, 0, len(self))
        return result

    fn __mul__(self: IntList, other: Self) -> IntList:
        if len(self) != len(other):
            panic(
                "IntList → __mul__(other): lengths are not equal → ",
                String(len(self)),
                "<=>",
                String(len(other)),
            )
        result = IntList.with_capacity(len(self))
        for i in range(len(self)):
            result[i] = self[i] * other[i]
        return result

    fn __getitem__(self, slice: Slice) -> Self:
        var start, end, step = slice.indices(len(self))
        var spread = range(start, end, step)

        if not len(spread):
            return Self()

        var result = Self.with_capacity(capacity=len(spread))
        for i in spread:
            result.append(self[i])

        return result^

    fn pop(mut self, index: Int = -1) -> Int:
        if len(self) < 1:
            abort("cannot pop from empty IntList")

        var i = index
        if i < 0:
            i += self.size
        if i < 0 or i >= len(self):
            abort("pop index out of bounds")

        val = (self.data + i).take_pointee()
        for j in range(i + 1, self.size):
            (self.data + j).move_pointee_into(self.data + j - 1)

        self.size -= 1
        return val

    fn __add__(self: IntList, other: IntList) -> IntList:
        if (
            self.data.__as_bool__() == False
            and other.data.__as_bool__() == False
        ):
            return IntList()
        if self.data.__as_bool__() == False:
            # _other = other
            # return _other
            return other
        if other.data.__as_bool__() == False:
            # _self = self
            # return _self
            return self

        result = IntList.with_capacity(len(self) + len(other))
        result.copy_from(0, self, 0, len(self))
        result.copy_from(len(self), other, 0, len(other))
        return result

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        """Access an element at the specified index.
        Args:
            idx: Zero-based index of the element to access.

        Returns:
            The integer value at the specified index.

        """
        index = idx if idx >= 0 else idx + self.__len__()
        if index < 0 or index >= len(self):
            abort("IntList __getitem__  → Out-of-bounds read: " + String(idx))

        return (self.data + index)[]

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, value: Int):
        """Set the value at the specified index.

        Args:
            idx: Zero-based index of the element to modify.
            value: The integer value to store at the specified index.

        """

        if idx < 0 or idx > len(self):
            abort("IntList __setitem__ -> Cannot skip indices")
        if idx == self.size:
            self.append(value)
        else:
            (self.data + idx)[] = value

    fn replace(self, idx: Int, value: Int) -> Self:
        result = self
        result[idx] = value
        return result

    @staticmethod
    fn invert_permutation(perm: IntList) -> Self:
        n = perm.size
        inverted = IntList.filled(n, 0)
        for i in range(n):
            inverted[perm[i]] = i

        return inverted

    fn replace(self: IntList, indices: IntList, values: IntList) -> Self:
        n = len(self)
        m = len(indices)

        if m != len(values):
            abort("IntList -> replace: indices and values must be same length")

        # Validate indices: no out-of-bounds, no duplicates
        for i in range(m):
            idx = indices[i]
            if idx < 0 or idx >= n:
                abort("IntList -> replace: index out of bounds: " + String(idx))

        result = self.copy()

        # Apply replacements
        for i in range(m):
            result[indices[i]] = values[i]

        return result

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the number of elements in the array.

        Returns:
            The number of elements in the array.
        """
        return self.size

    @always_inline("nodebug")
    fn len(self) -> Int:
        """Get the number of elements in the array.

        Returns:
            The number of elements in the array.
        """
        return self.size

    fn is_empty(self) -> Bool:
        return len(self) == 0

    fn __eq__(self: IntList, other: IntList) -> Bool:
        if len(self) != len(other):
            return False
        var index = 0
        for element in self:
            if element != other[index]:
                return False
            index += 1
        return True

    fn __str__(self) -> String:
        var s = String("[")
        for i in range(len(self)):
            s += String(self[i])
            if i < len(self) - 1:
                s += ", "
        s += "]"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    @always_inline("nodebug")
    fn copy_from(
        mut self: IntList,
        dst_offset: Int,
        source: IntList,
        src_offset: Int,
        size: Int,
    ):
        """Copy elements from another IntList with source offset.

        Args:
            dst_offset: Destination offset in this array.
            source: Source array to copy from.
            src_offset: Source offset in the source array.
            size: Number of elements to copy.
        """

        # --- Safety checks ---
        if dst_offset < 0 or src_offset < 0 or size < 0:
            abort("Negative offset/size not allowed")

        if src_offset + size > source.size:
            abort("Source range out of bounds")

        required_dst_size = dst_offset + size

        # Resize capacity if needed
        if required_dst_size > self.capacity:
            new_capacity = max(
                required_dst_size,
                self.capacity * 2 if self.capacity > 0 else required_dst_size,
            )
            self.resize(new_capacity)

        # Fill any gap between current size and dst_offset with zeros
        if dst_offset > self.size:
            for i in range(self.size, dst_offset):
                self.data[i] = 0

        # Perform the actual copy
        memcpy(
            self.data.offset(dst_offset), source.data.offset(src_offset), size
        )

        # Update size
        if required_dst_size > self.size:
            self.size = required_dst_size

    @always_inline("nodebug")
    fn copy_from(mut self, offset: Int, source: IntList, size: Int):
        """Copy elements from another `IntList`.

        Args:
            offset: Destination offset in this array.
            source: Source array to copy from.
            size: Number of elements to copy.
        """
        self.copy_from(offset, source, 0, size)

    fn prepend(mut self, value: Int):
        if self.size == self.capacity:
            new_capacity = max(1, self.capacity * 2)
            self.resize(new_capacity)
        # Shift all elements to the right by 1
        for i in range(self.size, 0, -1):
            self.data[i] = self.data[i - 1]
        # Insert the new value at the beginning
        self.data[0] = value
        self.size += 1

    fn append(mut self, value: Int):
        if self.size == self.capacity:
            new_capacity = max(1, self.capacity * 2)
            self.resize(new_capacity)
        self.data[self.size] = value
        self.size += 1

    fn resize(mut self, new_capacity: Int):
        self.reserve(new_capacity)

    fn reserve(mut self, new_capacity: Int):
        if new_capacity <= self.capacity:
            return

        new_data = UnsafePointer[Int].alloc(new_capacity)
        if self.size > 0:
            memcpy(new_data, self.data, self.size)
        if self.data:
            for i in range(len(self)):
                (self.data + i).destroy_pointee()
            self.data.free()
        self.data = new_data
        self.capacity = new_capacity

    @always_inline
    fn __contains__(self, value: Int) -> Bool:
        for i in range(len(self)):
            if self[i] == value:
                return True
        return False

    fn copy(self) -> Self:
        """Creates a deep copy of the given IntList.

        Returns:
            A copy of the value.
        """
        var copy = Self.with_capacity(capacity=len(self))
        for e in self:
            copy.append(e)
        return copy^

    fn reversed(self) -> Self:
        copied = self.copy()
        copied.reverse()
        return copied

    fn reverse(mut self):
        """Reverses the elements of the list."""

        var left = 0
        var right = len(self) - 1

        var length = len(self)
        var half_len = length // 2

        for _ in range(half_len):
            var left_ptr = self.data + left
            var right_ptr = self.data + right

            var tmp = left_ptr.take_pointee()
            right_ptr.move_pointee_into(left_ptr)
            right_ptr.init_pointee_move(tmp)

            left += 1
            right -= 1

    fn print(self, limit: Int = 5) -> None:
        total = len(self)
        print("IntList[", total, "] = ", end="")

        if total <= 2 * limit:
            # Print all elements
            for i in range(total):
                print(self[i], end=" ")
        else:
            # Print first `limit` elements
            for i in range(limit):
                print(self[i], end=" ")
            print("... ", end="")

            # Print last `limit` elements
            for i in range(total - limit, total):
                print(self[i], end=" ")

        print()

    fn __iter__(ref self) -> Iterator[__origin_of(self)]:
        """Iterate over elements of the IntList, returning immutable references.

        Returns:
            An iterator of immutable references to the IntList elements.
        """
        return Iterator(0, Pointer(to=self))

    fn __reversed__(
        ref self,
    ) -> Iterator[__origin_of(self), False]:
        return Iterator[forward=False](len(self), Pointer(to=self))

    fn zip(
        ref self, ref other: Self
    ) -> ZipIterator[__origin_of(self), __origin_of(other)]:
        """Iterate over elements of the IntList, returning immutable references.

        Returns:
            An iterator of immutable references to the IntList elements.
        """
        return ZipIterator(0, Pointer(to=self), Pointer(to=other))

    fn zip_reversed(
        ref self, ref other: Self
    ) -> ZipIterator[__origin_of(self), __origin_of(other), False]:
        """Iterate over elements of the IntList, returning immutable references reverse order.

        Returns:
            An iterator of immutable references to the IntList elements in reverse order.
        """
        return ZipIterator[forward=False](
            min(len(self), len(other)), Pointer(to=self), Pointer(to=other)
        )


struct Iterator[
    origin: Origin[False],
    forward: Bool = True,
](Sized & Copyable):
    var index: Int
    var src: Pointer[IntList, origin]

    fn __init__(out self, idx: Int, src: Pointer[IntList, origin]):
        self.src = src
        self.index = idx

    fn __iter__(self) -> Self:
        return self

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
    var src_this: Pointer[IntList, origin_this]
    var src_that: Pointer[IntList, origin_that]

    fn __init__(
        out self,
        idx: Int,
        src_this: Pointer[IntList, origin_this],
        src_that: Pointer[IntList, origin_that],
    ):
        self.src_this = src_this
        self.src_that = src_that
        self.index = idx
        self.offset = abs(len(src_this[]) - len(src_that[]))

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> (Int, Int):
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


fn main() raises:
    ll = IntList.new([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sliced = ll[2::3]
    print(sliced)


    test_negative_indices()
    test_any()
    inferred_dim = 10
    l = IntList(1, 2, 5, -1, 10)
    concrete_shape = [inferred_dim if dim == -1 else dim for dim in l]
    print(concrete_shape.__str__())


from testing import assert_true


fn test_any() raises:
    l = IntList(1, 3, 0, -1)

    fn check(e: Int) -> Bool:
        return e == -1

    assert_true(l.any(check), "Any assertion for -1 failed")


fn test_negative_indices() raises:
    l1 = IntList(1, 2, 3)
    assert_true(
        l1[-1] == 3 and l1[-2] == 2 and l1[-3] == 1,
        "IntList negative indices assertion failed",
    )
