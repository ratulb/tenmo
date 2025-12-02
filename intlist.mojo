from common_utils import panic
from intarray import IntArray

# ============================================================
# INTLIST (Simplified - mostly delegates to IntArray)
# ============================================================


@register_passable
struct IntList(ImplicitlyCopyable, Representable, Sized, Stringable, Writable):
    """Backward compatibility wrapper around IntArray with some extra methods.
    """

    var data: IntArray

    @staticmethod
    @always_inline
    fn Empty() -> IntList:
        return IntList()

    @always_inline("nodebug")
    fn __init__(out self):
        self.data = IntArray()

    @always_inline("nodebug")
    fn __init__(out self, values: IntArray):
        self.data = values

    @always_inline("nodebug")
    fn __init__(out self, values: List[Int]):
        self.data = IntArray(values)

    @always_inline("nodebug")
    fn __init__(out self, *values: Int):
        self.data = IntArray(values)

    @always_inline("nodebug")
    fn __init__(out self, values: VariadicList[Int]):
        self.data = IntArray(values)

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self.data = existing.data

    @staticmethod
    @always_inline
    fn new(src: List[Int]) -> IntList:
        return IntList(src)

    @staticmethod
    @always_inline
    fn with_capacity(capacity: Int, fill: Optional[Int] = None) -> IntList:
        var arr = IntArray.with_capacity(capacity)
        if fill:
            for _ in range(capacity):
                arr.append(fill.value())
        return IntList(arr)

    @staticmethod
    @always_inline
    fn filled(length: Int, value: Int) -> IntList:
        return IntList(IntArray.filled(length, value))

    @staticmethod
    @always_inline
    fn range_list(n: Int) -> IntList:
        return IntList(IntArray.range(0, n))

    # ========== Core Operations ==========

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return len(self.data)

    @always_inline("nodebug")
    fn len(self) -> Int:
        return len(self.data)

    @always_inline("nodebug")
    fn size(self) -> Int:
        return len(self.data)

    @always_inline("nodebug")
    fn capacity(self) -> Int:
        return self.data.capacity()

    @always_inline("nodebug")
    fn is_empty(self) -> Bool:
        return self.data.is_empty()

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        return self.data[idx]

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, value: Int):
        self.data[idx] = value

    fn __getitem__(self, slice: Slice) -> Self:
        return IntList(self.data[slice])

    @always_inline("nodebug")
    fn __contains__(self, value: Int) -> Bool:
        return value in self.data

    fn __eq__(self, other: Self) -> Bool:
        return self.data == other.data

    fn __eq__(self, other: IntArray) -> Bool:
        return self.data == other

    fn __eq__(self, other: List[Int]) -> Bool:
        return self.data == other

    # ========== Growth ==========

    @always_inline
    fn append(mut self, value: Int):
        self.data.append(value)

    @always_inline
    fn prepend(mut self, value: Int):
        self.data.prepend(value)

    fn pop(mut self, index: Int = -1) -> Int:
        return self.data.pop(index)

    fn clear(mut self):
        self.data.clear()

    # ========== Conversions ==========

    fn tolist(self) -> List[Int]:
        return self.data.tolist()

    @always_inline
    fn intarray(self) -> IntArray:
        return self.data

    fn __str__(self) -> String:
        return self.data.__str__()

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    # ========== Math ==========

    fn product(self) -> Int:
        return self.data.product()

    fn sum(self) -> Int:
        return self.data.sum()

    # ========== List Operations ==========

    fn __add__(self, other: Self) -> Self:
        var result = IntArray.with_capacity(len(self) + len(other))
        for i in range(len(self)):
            result.append(self[i])
        for i in range(len(other)):
            result.append(other[i])
        return IntList(result)

    fn __add__(self, other: List[Int]) -> Self:
        return self.__add__(IntList(other))

    fn __radd__(self, other: List[Int]) -> Self:
        return IntList(other).__add__(self)

    fn __mul__(self, factor: Int) -> Self:
        """Repeat."""
        var result = IntArray.with_capacity(len(self) * factor)
        for _ in range(factor):
            for i in range(len(self)):
                result.append(self[i])
        return IntList(result)

    fn __rmul__(self, factor: Int) -> Self:
        return self.__mul__(factor)

    fn __mul__(self, other: Self) -> Self:
        """Element-wise multiply."""
        if len(self) != len(other):
            panic("IntList: cannot multiply different lengths")
        var result = IntArray.with_capacity(len(self))
        for i in range(len(self)):
            result.append(self[i] * other[i])
        return IntList(result)

    fn __mul__(self, other: IntArray) -> Self:
        """Element-wise multiply with IntArray."""
        if len(self) != len(other):
            panic("IntList: cannot multiply different lengths")
        var result = IntArray.with_capacity(len(self))
        for i in range(len(self)):
            result.append(self[i] * other[i])
        return IntList(result)

    # ========== Functional ==========

    fn reverse(mut self):
        self.data.reverse()

    fn reversed(self) -> Self:
        return IntList(self.data.reversed())

    fn count(self, elem: Int) -> Int:
        """Count occurrences."""
        var c = 0
        for i in range(len(self)):
            if self[i] == elem:
                c += 1
        return c

    fn has_duplicates(self) -> Bool:
        """Check for duplicates."""
        if len(self) <= 1:
            return False
        var sorted = self.sorted()
        for i in range(len(sorted) - 1):
            if sorted[i] == sorted[i + 1]:
                return True
        return False

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
        var result = IntList(self.data)
        result.sort(asc)
        return result^

    fn sort_and_deduplicate(mut self):
        """Sort and remove duplicates."""
        if len(self) <= 1:
            return
        self.sort()
        var write_idx = 1
        for read_idx in range(1, len(self)):
            if self[read_idx] != self[write_idx - 1]:
                self[write_idx] = self[read_idx]
                write_idx += 1
        self.data._size = write_idx

    fn swap(mut self, i: Int, j: Int):
        """Swap two elements."""
        var idx1 = i if i >= 0 else i + len(self)
        var idx2 = j if j >= 0 else j + len(self)
        if idx1 < 0 or idx1 >= len(self) or idx2 < 0 or idx2 >= len(self):
            panic("IntList: swap index out of bounds")
        if idx1 != idx2:
            var temp = self[idx1]
            self[idx1] = self[idx2]
            self[idx2] = temp

    fn replace(self, idx: Int, value: Int) -> Self:
        """Return copy with replaced value."""
        var result = IntList(self.data)
        result[idx] = value
        return result^

    fn replace(self, indices: Self, values: Self) -> Self:
        """Return copy with multiple replacements."""
        if len(indices) != len(values):
            panic("IntList: indices and values must be same length")
        var result = IntList(self.data)
        for i in range(len(indices)):
            var idx = indices[i]
            if idx < 0 or idx >= len(result):
                panic("IntList: index out of bounds in replace")
            result[idx] = values[i]
        return result^

    # ========== Advanced Operations ==========

    fn permute(self, axes: Self) -> Self:
        """Reorder according to axes."""
        if len(self) != len(axes):
            panic("IntList: permute axes length mismatch")
        var result = IntArray.with_capacity(len(axes))
        var seen = IntArray.with_capacity(len(axes))
        for i in range(len(axes)):
            var axis = axes[i]
            if axis < 0:
                axis += len(self)
            if axis < 0 or axis >= len(self):
                panic("IntList: permute axis out of bounds")
            if axis in seen:
                panic("IntList: duplicate axis in permute")
            seen.append(axis)
            result.append(self[axis])
        return IntList(result)

    fn select(self, indices: Self) -> Self:
        """Select elements by indices."""
        var result = IntArray.with_capacity(len(indices))
        for i in range(len(indices)):
            var idx = indices[i]
            if idx < 0 or idx >= len(self):
                panic("IntList: select index out of bounds")
            result.append(self[idx])
        return IntList(result)

    fn indices_of(self, val: Int) -> Self:
        """Return indices where value equals val."""
        var result = IntArray.with_capacity(len(self))
        for i in range(len(self)):
            if self[i] == val:
                result.append(i)
        return IntList(result)

    fn insert(self, at: Int, value: Int) -> Self:
        """Insert value at position."""
        if at < 0 or at > len(self):
            panic("IntList: insert index out of bounds")
        var result = IntArray.with_capacity(len(self) + 1)
        for i in range(at):
            result.append(self[i])
        result.append(value)
        for i in range(at, len(self)):
            result.append(self[i])
        return IntList(result)

    fn insert(self, indices: Self, values: Self) -> Self:
        """Insert values at multiple sorted indices."""
        if len(indices) != len(values):
            panic("IntList: indices and values length mismatch")

        if len(self) == 0:
            if len(indices) == 0:
                return IntList()
            if not indices.is_strictly_increasing_from_zero():
                panic("IntList: invalid indices for empty list")
            return IntList(values.data)

        var result = IntArray.with_capacity(len(self) + len(indices))
        var red_cursor = 0
        var orig_cursor = 0

        for i in range(len(self) + len(indices)):
            if red_cursor < len(indices) and indices[red_cursor] == i:
                result.append(values[red_cursor])
                red_cursor += 1
            else:
                if orig_cursor >= len(self):
                    panic("IntList: insert ran out of source values")
                result.append(self[orig_cursor])
                orig_cursor += 1

        return IntList(result)

    fn any(self, cond: fn (Int) -> Bool) -> Bool:
        """Check if any element satisfies condition."""
        for i in range(len(self)):
            if cond(self[i]):
                return True
        return False

    @always_inline
    fn is_strictly_increasing_from_zero(self) -> Bool:
        """Check if [0, 1, 2, ...]."""
        for i in range(len(self)):
            if self[i] != i:
                return False
        return True

    @staticmethod
    fn invert_permutation(perm: Self) -> Self:
        """Invert a permutation."""
        var inverted = IntArray.filled(len(perm), 0)
        for i in range(len(perm)):
            inverted[perm[i]] = i
        return IntList(inverted)

    fn print(self, limit: Int = 20):
        """Pretty print."""
        var total = len(self)
        print("IntList[", total, "] = ", end="")
        if total <= 2 * limit:
            for i in range(total):
                print(self[i], end=" ")
        else:
            for i in range(limit):
                print(self[i], end=" ")
            print("... ", end="")
            for i in range(total - limit, total):
                print(self[i], end=" ")
        print()

    fn __iter__(ref self) -> Iterator[origin_of(self)]:
        """Iterate over elements of the IntList, returning immutable references.

        Returns:
            An iterator of immutable references to the IntList elements.
        """
        return Iterator(0, Pointer(to=self))

    fn __reversed__(
        ref self,
    ) -> Iterator[origin_of(self), False]:
        return Iterator[forward=False](len(self), Pointer(to=self))

    fn zip(
        ref self, ref other: Self
    ) -> ZipIterator[origin_of(self), origin_of(other)]:
        """Iterate over elements of the IntList, returning immutable references.

        Returns:
            An iterator of immutable references to the IntList elements.
        """
        return ZipIterator(0, Pointer(to=self), Pointer(to=other))

    fn zip_reversed(
        ref self, ref other: Self
    ) -> ZipIterator[origin_of(self), origin_of(other), False]:
        """Iterate over elements of the IntList, returning immutable references reverse order.

        Returns:
            An iterator of immutable references to the IntList elements in reverse order.
        """
        return ZipIterator[forward=False](
            min(len(self), len(other)), Pointer(to=self), Pointer(to=other)
        )


# ============================================================
# ITERATORS
# ============================================================


@register_passable
struct Iterator[
    origin: Origin[False],
    forward: Bool = True,
](Sized & Copyable):
    var index: Int
    var src: Pointer[IntList, origin]

    fn __init__(out self, idx: Int, src: Pointer[IntList, origin]):
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
    il = IntList.range_list(4).reversed()
    ia = IntArray.range(0, 4).reversed()
    print(il, ia)

    print(IntList.invert_permutation(il))
    print(IntArray.invert_permutation(ia))

    pass
