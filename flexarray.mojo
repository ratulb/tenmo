from memory import UnsafePointer, memcpy
from testing import assert_true
from os import abort

alias FLEX_ARRAY_VALIDATION = True


@register_passable
struct FlexArray[dtype: DType = DType.int64](Sized & Copyable):
    """A memory-efficient, register-passable array of DType.

    `FlexArray` provides a low-level implementation of a dynamically-sized dtype array
    with direct memory management. It supports both owned and non-owned (view) modes
    for efficient memory sharing without copying.

    This struct serves as the underlying storage mechanism for `Shape` and related
    data structures, optimized for high-performance tensor operations.
    """

    var data: UnsafePointer[Scalar[dtype]]
    var size: Int
    var capacity: Int

    fn __init__(out self):
        self.data = UnsafePointer[Scalar[dtype]]()
        self.capacity = 0
        self.size = 0

    @always_inline("nodebug")
    fn __init__(out self, *elems: Scalar[dtype]):
        """Initialize a new owned `FlexArray` with the specified size.

        Args:
            elems: Number of scalars to allocate space for.
        """
        self.data = UnsafePointer[Scalar[dtype]].alloc(len(elems))
        self.size = len(elems)
        self.capacity = len(elems)
        for i in range(len(elems)):
            (self.data + i)[] = elems[i]

    @always_inline("nodebug")
    fn __init__(out self, elems: VariadicList[Int]):
        """Initialize a new owned `FlexArray` with the specified size.

        Args:
            elems: Number of scalars to allocate space for.
        """
        self.data = UnsafePointer[Scalar[self.dtype]].alloc(len(elems))
        self.size = len(elems)
        self.capacity = len(elems)
        for i in range(len(elems)):
            (self.data + i)[] = elems[i]

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        """Initialize by copying an existing `FlexArray`.
        Args:
            existing: The source array to copy from.
        """
        self.size = existing.size
        self.capacity = existing.capacity
        self.data = UnsafePointer[Scalar[dtype]].alloc(existing.capacity)
        memcpy(self.data, existing.data, existing.size)

    @staticmethod
    fn with_capacity(capacity: Int) -> FlexArray[dtype]:
        array = Self()
        array.data = UnsafePointer[Scalar[dtype]].alloc(capacity)
        array.capacity = capacity
        array.size = 0
        return array

    @always_inline("nodebug")
    fn __del__(owned self):
        """Destroy the `FlexArray` and free its memory if owned.

        Only frees memory for owned arrays (positive _size) to prevent
        double-free errors with views.
        """
        if self.data:
            print("__del__ is kicking in alright")
            self.data.free()

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Scalar[dtype]:
        """Access an element at the specified index.

        Args:
            idx: Zero-based index of the element to access.

        Returns:
            The integer value at the specified index.

        """

        @parameter
        if FLEX_ARRAY_VALIDATION:
            if idx < 0 or idx >= len(self):
                abort("FlexArray __getitem__ -> Out-of-bounds read")

        return self.data[idx]

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, value: Scalar[dtype]):
        """Set the value at the specified index.

        Args:
            idx: Zero-based index of the element to modify.
            value: The integer value to store at the specified index.

        """

        @parameter
        if FLEX_ARRAY_VALIDATION:
            if idx < 0 and idx > len(self):
                abort("FlexArray __setitem__ -> Cannot skip indices")
        if idx == self.size:
            self.append(value)
        else:
            self.data[idx] = value

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the number of elements in the array.

        Returns:
            The number of elements in the array.
        """
        return self.size

    @always_inline("nodebug")
    fn copy_from(mut self, offset: Int, source: Self, size: Int) raises:
        """Copy elements from another `FlexArray`.

        Args:
            offset: Destination offset in this array.
            source: Source array to copy from.
            size: Number of elements to copy.
        """
        self.copy_from(offset, source, 0, size)

    fn copy_from(
        mut self, dst_offset: Int, source: Self, src_offset: Int, size: Int
    ) raises:
        """Copy elements from another FlexArray with source offset.

        Args:
            dst_offset: Destination offset in this array.
            source: Source array to copy from.
            src_offset: Source offset in the source array.
            size: Number of elements to copy.
        """

        # --- Safety checks ---
        assert_true(
            dst_offset >= 0 and src_offset >= 0 and size >= 0,
            "Negative offset/size not allowed",
        )
        assert_true(
            src_offset + size <= source.size, "Source range out of bounds"
        )

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
            var zero: Scalar[dtype] = Scalar[dtype](0)
            for i in range(self.size, dst_offset):
                self.data[i] = zero

        # Perform the actual copy
        memcpy(
            self.data.offset(dst_offset), source.data.offset(src_offset), size
        )

        # Update size
        if required_dst_size > self.size:
            self.size = required_dst_size

    fn append(mut self, value: Scalar[dtype]):
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

        new_data = UnsafePointer[Scalar[dtype]].alloc(new_capacity)
        if self.size > 0:
            memcpy(new_data, self.data, self.size)
        if self.data:
            for i in range(len(self)):
                (self.data + i).destroy_pointee()
            self.data.free()
        self.data = new_data
        self.capacity = new_capacity

    @always_inline
    fn __contains__(self, value: Scalar[dtype]) -> Bool:
        for i in range(len(self)):
            if self[i] == value:
                return True
        return False

    fn print(self, limit: Int = 5) raises -> None:
        total = len(self)
        print("FlexArray[", total, "] = ", end="")

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


from testing import assert_false


fn test_flexarray() raises:
    fa = FlexArray()
    assert_false(fa.data, "Uninitialized data pointer assertion failed")
    fa.append(100)
    assert_true(
        fa.size == 1 and fa.capacity == 1, "size and capacity assertion failed"
    )
    assert_true(100 in fa, "contains assertion failed")
    copy1 = fa
    assert_true(
        100 in copy1, "Post __copyint__ destination contains assertion failed"
    )
    assert_true(100 in fa, "Post __copyint__ source contains assertion failed")
    fa2 = FlexArray()
    fa2.copy_from(0, fa, 1)
    assert_true(
        100 in fa2, "Post copy_from destination contains assertion failed"
    )
    fa3 = FlexArray(True, True, True, False)
    fa3.print()
    fa4 = FlexArray[DType.bool]()
    fa4.copy_from(4, fa3, 4)
    assert_true(
        len(fa4) == 8,
        "Copy from source - capacity initialization assertion failed",
    )
    fa4.print()
    fa5 = FlexArray[DType.bool].with_capacity(12)
    assert_true(len(fa5) == 0, "with_capacity len assertion failed")
    fa5.copy_from(4, fa3, 4)
    assert_true(len(fa5) == 8, "post with_capacity len assertion failed")


fn main() raises:
    test_flexarray()
