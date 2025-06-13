#from memory import UnsafePointer, memcpy, memset_zero, Pointer
from memory import UnsafePointer, memcpy, Pointer
from os import abort


@register_passable
struct IntList(Sized & Copyable):
    """A memory-efficient, register-passable, dynamic array of Ints. Would abort on any erroneous condition.
    """

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
    fn with_capacity(capacity: Int) -> IntList:
        array = Self()
        array.data = UnsafePointer[Int].alloc(capacity)
        #memset_zero(array.data, capacity)
        array.capacity = capacity
        array.size = 0
        return array

    @staticmethod
    fn insert_axis(indices: IntList, index: Int, axis: Int) -> IntList:
        # Insert `index` at position `axis` in `indices`, return a new IntList

        if axis < 0 or axis > len(indices):
            abort("Axis out of bounds")

        result = IntList.with_capacity(len(indices) + 1)

        for i in range(len(indices) + 1):
            if i == axis:
                result.append(index)
            if i < len(indices):
                result.append(indices[i])

        return result

    @always_inline("nodebug")
    # fn __del__(owned self):
    fn free(owned self):
        """Destroy the `IntList` and free its memory."""
        if self.data:
            print("IntList __del__ is kicking in alright")
            self.data.free()

    fn __mul__(self: IntList, factor: Int) -> IntList:
        if factor < 1 or self.data.__as_bool__() == False:
            return IntList()
        result = IntList.with_capacity(len(self) * factor)
        for i in range(factor):
            result.copy_from(i * len(self), self, 0, len(self))
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
            _other = other
            return other
        if other.data.__as_bool__() == False:
            _self = self
            return _self

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

        if idx < 0 or idx >= len(self):
            abort("IntList __getitem__ -> Out-of-bounds read: " + String(idx))

        return (self.data + idx)[]

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

    # @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the number of elements in the array.

        Returns:
            The number of elements in the array.
        """
        return self.size

    fn __eq__(self: IntList, other: IntList) -> Bool:
        if len(self) != len(other):
            return False
        var index = 0
        for element in self:
            if element != other[index]:
                return False
            index += 1
        return True

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
        var copy = Self.with_capacity(capacity=self.capacity)
        for e in self:
            copy.append(e)
        return copy^

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


fn main() raises:
    il = IntList(1, 2, 3)
    il.print()
    il.reverse()
    il.print()
    for _ in range(len(il)):
        _ = il.pop()
    il.print()
    ll = IntList(1, 2, 3, 4, 5, 6)
    ll[1:4].print()
