from memory import UnsafePointer, memcpy, Pointer
from os import abort

alias FLEX_ARRAY_VALIDATION = True


@register_passable
struct DynArray[T: Copyable = Int](Sized & Copyable):
    """A memory-efficient, register-passable array of DType.

    `DynArray` provides a low-level implementation of a dynamically-sized T array
    with direct memory management. It supports both owned and non-owned (view) modes
    for efficient memory sharing without copying.

    This struct serves as the underlying storage mechanism for `Shape` and related
    data structures, optimized for high-performance tensor operations.
    """

    var data: UnsafePointer[T]
    var size: Int
    var capacity: Int

    fn __init__(out self):
        """Constructs an empty DynArray."""
        self.data = UnsafePointer[T]()
        self.capacity = 0
        self.size = 0

    @always_inline("nodebug")
    fn __init__(out self, *elems: T):
        """Initialize a new owned `DynArray` with the elements.

        Args:
            elems: Number of scalars to allocate space for.
        """
        self.data = UnsafePointer[T].alloc(len(elems))
        self.size = len(elems)
        self.capacity = len(elems)
        for idx in range(len(elems)):
            (self.data + idx)[] = elems[idx]

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        """Initialize by copying an existing `DynArray`.
        Args:
            existing: The source array to copy from.
        """
        self.size = existing.size
        self.capacity = existing.capacity
        self.data = UnsafePointer[T].alloc(existing.capacity)
        memcpy(self.data, existing.data, existing.size)

    @staticmethod
    fn with_capacity(capacity: Int) -> DynArray[T]:
        array = Self()
        array.data = UnsafePointer[T].alloc(capacity)
        array.capacity = capacity
        array.size = 0
        return array

    @always_inline("nodebug")
    # fn __del__(owned self):
    fn free(owned self):
        """Destroy the `DynArray` and free its memory if owned.

        Only frees memory for owned arrays (positive _size) to prevent
        double-free errors with views.
        """
        if self.data:
            print("__del__ is kicking in alright")
            self.data.free()

    fn __mul__[
        D: Copyable & Defaultable
    ](self: DynArray[D], factor: Int) -> DynArray[D]:
        if factor < 1 or self.data.__as_bool__() == False:
            return DynArray[D]()
        result = DynArray[D].with_capacity(len(self) * factor)
        for i in range(factor):
            result.copy_from(i * len(self), self, 0, len(self))
        return result

    fn __add__[
        D: Copyable & Defaultable
    ](self: DynArray[D], other: DynArray[D]) -> DynArray[D]:
        if (
            self.data.__as_bool__() == False
            and other.data.__as_bool__() == False
        ):
            return DynArray[D]()
        if self.data.__as_bool__() == False:
            _other = other
            return other
        if other.data.__as_bool__() == False:
            _self = self
            return _self

        result = DynArray[D].with_capacity(len(self) + len(other))
        result.copy_from(0, self, 0, len(self))
        result.copy_from(len(self), other, 0, len(other))
        return result

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> T:
        """Access an element at the specified index.

        Args:
            idx: Zero-based index of the element to access.

        Returns:
            The integer value at the specified index.

        """

        @parameter
        if FLEX_ARRAY_VALIDATION:
            if idx < 0 or idx >= len(self):
                abort("DynArray __getitem__ -> Out-of-bounds read")

        return (self.data + idx)[]

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, value: T):
        """Set the value at the specified index.

        Args:
            idx: Zero-based index of the element to modify.
            value: The integer value to store at the specified index.

        """

        @parameter
        if FLEX_ARRAY_VALIDATION:
            if idx < 0 or idx > len(self):
                abort("DynArray __setitem__ -> Cannot skip indices")
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

    fn __eq__[
        D: EqualityComparable & Copyable, //
    ](self: DynArray[D], other: DynArray[D]) -> Bool:
        if len(self) != len(other):
            return False
        var index = 0
        for element in self:
            if element != other[index]:
                return False
            index += 1
        return True

    @always_inline("nodebug")
    fn copy_from[
        D: Copyable & Defaultable
    ](
        mut self: DynArray[D],
        dst_offset: Int,
        source: DynArray[D],
        src_offset: Int,
        size: Int,
    ):
        """Copy elements from another DynArray with source offset.

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
            var zero: D = D()
            for i in range(self.size, dst_offset):
                self.data[i] = zero

        # Perform the actual copy
        memcpy(
            self.data.offset(dst_offset), source.data.offset(src_offset), size
        )

        # Update size
        if required_dst_size > self.size:
            self.size = required_dst_size

    @always_inline("nodebug")
    fn copy_from[
        D: Copyable & Defaultable
    ](mut self: DynArray[D], offset: Int, source: DynArray[D], size: Int):
        """Copy elements from another `DynArray`.

        Args:
            offset: Destination offset in this array.
            source: Source array to copy from.
            size: Number of elements to copy.
        """
        self.copy_from(offset, source, 0, size)

    fn append(mut self, value: T):
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

        new_data = UnsafePointer[T].alloc(new_capacity)
        if self.size > 0:
            memcpy(new_data, self.data, self.size)
        if self.data:
            for i in range(len(self)):
                (self.data + i).destroy_pointee()
            self.data.free()
        self.data = new_data
        self.capacity = new_capacity

    @always_inline
    fn __contains__[
        D: EqualityComparable & Copyable, //
    ](self: DynArray[D], value: D) -> Bool:
        for i in range(len(self)):
            if self[i] == value:
                return True
        return False

    fn copy(self) -> Self:
        """Creates a deep copy of the given DynArray.

        Returns:
            A copy of the value.
        """
        var copy = Self.with_capacity(capacity=self.capacity)
        for e in self:
            copy.append(e)
        return copy^

    fn print[
        D: Copyable & Stringable
    ](self: DynArray[D], limit: Int = 5) raises -> None:
        total = len(self)
        print("DynArray[", total, "] = ", end="")

        if total <= 2 * limit:
            # Print all elements
            for i in range(total):
                print(self[i].__str__(), end=" ")
        else:
            # Print first `limit` elements
            for i in range(limit):
                print(self[i].__str__(), end=" ")
            print("... ", end="")

            # Print last `limit` elements
            for i in range(total - limit, total):
                print(self[i].__str__(), end=" ")

        print()

    fn __iter__(ref self) -> _DynArrayIter[self.T, __origin_of(self)]:
        """Iterate over elements of the DynArray, returning immutable references.

        Returns:
            An iterator of immutable references to the DynArray elements.
        """
        return _DynArrayIter[T](0, Pointer(to=self))

    fn __reversed__(
        ref self,
    ) -> _DynArrayIter[self.T, __origin_of(self), False]:
        return _DynArrayIter[T, forward=False](len(self), Pointer(to=self))


struct _DynArrayIter[
    T: Copyable,
    origin: Origin[False],
    forward: Bool = True,
](Sized & Copyable):
    var index: Int
    var src: Pointer[DynArray[T], origin]

    fn __init__(out self, idx: Int, src: Pointer[DynArray[T], origin]):
        self.src = src
        self.index = idx

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> T:
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
    da = DynArray[Scalar[DType.float32]](1,2,3)
    da.print()



