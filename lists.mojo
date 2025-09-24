from memory import memcpy, Pointer
from common_utils import log_debug, panic


struct IntList(
    Sized & Copyable & Movable & Stringable & Representable & Writable
):
    var elems: List[Int]

    fn __init__(out self):
        self.elems = List[Int](capacity=0)

    @always_inline("nodebug")
    fn __init__(out self, src: List[Int]):
        self.elems = List[Int](capacity=len(src))
        for elem in src:
            self.append(elem)

    @always_inline("nodebug")
    fn __init__(out self, *elems: Int):
        self.elems = List[Int](capacity=len(elems))
        for elem in elems:
            self.append(elem)

    @always_inline("nodebug")
    fn __init__(out self, elems: VariadicList[Int]):
        self.elems = List[Int](capacity=len(elems))
        for elem in elems:
            self.append(elem)

    # @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self.elems = existing.elems[::]

    fn __moveinit__(out self, deinit existing: Self):
        self.elems = existing.elems^

    @staticmethod
    fn new(src: List[Int]) -> IntList:
        out = IntList(src)
        return out

    fn tolist(self) -> List[Int]:
        out = List[Int](capacity=len(self))
        memcpy(out._data, self.elems._data, len(self))
        return out

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
        out = Self()
        out.elems = List[Int](capacity=capacity)
        if fill:
            for i in range(capacity):
                out[i] = fill.value()
        return out

    fn product(self) -> Int:
        prod = 1
        for elem in self:
            prod *= elem
        return prod

    fn sum(self) -> Int:
        summ = 0
        for elem in self:
            summ += elem
        return summ

    fn has_duplicates(self) -> Bool:
        if len(self) <= 1:
            return False
        var sorted_list = self.sorted()

        for i in range(len(sorted_list) - 1):
            if sorted_list[i] == sorted_list[i + 1]:
                return True

        return False

    fn permute(self, axes: IntList) -> IntList:
        if not len(self) == len(axes):
            panic(
                "IntList -> permute: axes length",
                len(axes).__str__(),
                "does not match list's length",
                len(self).__str__(),
            )
        permuted = IntList.with_capacity(len(axes))
        seen = IntList.with_capacity(len(axes))
        for ax in axes:
            axis = ax + len(self) if ax < 0 else ax
            if axis < 0 or axis >= len(self):
                panic("IntList -> permute: index out of bound", ax.__str__())
            if axis in seen:
                panic("IntList -> permute: duplicate axis", ax.__str__())
            seen.append(axis)
            permuted.append(self[axis])

        seen.free()

        return permuted

    fn swap(mut self, this_index: Int, that_index: Int):
        """Swaps elements at different indices."""
        index1 = this_index + len(self) if this_index < 0 else this_index
        index2 = that_index + len(self) if that_index < 0 else that_index

        if not 0 <= index1 < len(self) and 0 <= index2 < len(self):
            panic(
                "IntList → swap: provided index(indices) is(are) out of bounds"
            )

        if index1 != index2:
            self.elems.swap_elements(index1, index2)
            # swap((self.elem._data + index1)[], (self.elem._data + index2)[])

    fn sort_and_deduplicate(mut self):
        if len(self) == 0:
            return
        self.sort()
        write_index = 1
        for read_index in range(1, len(self)):
            if self[read_index] != self[write_index - 1]:
                self[write_index] = self[read_index]
                write_index += 1
        self.elems.shrink(len(self) - write_index + 1) 

    fn sorted(self, asc: Bool = True) -> IntList:
        copied = self.copy()
        copied.sort(asc)
        return copied

    fn sort(mut self, asc: Bool = True):
        for i in range(1, len(self)):
            elem = self[i]
            j = i
            while j > 0 and (elem < self[j - 1] if asc else elem > self[j - 1]):
                self[j] = self[j - 1]
                j -= 1
            self[j] = elem

    fn select(self, indices: IntList) -> IntList:
        n = len(self)
        result = IntList.with_capacity(len(indices))

        for i in indices:
            if i < 0 or i >= n:
                panic(
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
        out = IntList.with_capacity(len(self))

        for i in range(len(self)):
            if self[i] == val:
                out.append(i)
        return out

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
            panic("IntList -> insert: indices and values must be same length")

        n = len(self)
        m = len(indices)
        if n == 0:
            if m == 0:
                return IntList()
            # Ensure indices = [0, 1, ..., m-1]
            if not indices.is_strictly_increasing_from_zero():
                panic(
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
                    panic(
                        "IntList -> insert: ran out of source values too early"
                    )
                result.append(self[orig_cursor])
                orig_cursor += 1
        return result

    fn insert(self, at: Int, value: Int) -> IntList:
        # Insert `value` at position `at` in `self`, return a new IntList
        # `at` could be start, middle or end
        if at < 0 or at > len(self):
            panic("IntList -> insert - index out of bounds: " + String(at))

        result = IntList.with_capacity(len(self) + 1)
        for i in range(len(self) + 1):
            if i == at:
                result.append(value)
            if i < len(self):
                result.append(self[i])

        return result

    @always_inline("nodebug")
    #fn __del__(owned self):
    fn free(deinit self):
        """Destroy the `IntList` and free its memory."""
        if self.elems._data and len(self) > 0:
            log_debug("Calling IntList __del__")
            self.elems._data.free()

    fn clear(mut self):
        self.elems.clear()

    fn __rmul__(self: IntList, factor: Int) -> IntList:
        return self.__mul__(factor)

    fn __mul__(self: IntList, factor: Int) -> IntList:
        if factor < 1 or not self.elems:
            return IntList()
        result = IntList.with_capacity(len(self) * factor)
        elems = self.elems * factor
        for i in range(len(elems)):
            result[i] = elems[i]
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
        elems = self.elems[slice]
        out = IntList(elems)
        return out

    fn pop(mut self, index: Int = -1) -> Int:
        if len(self) < 1:
            panic("cannot pop from empty IntList")
        var i = index
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            panic("pop index out of bounds")

        val = self.elems.pop(i)
        return val

    fn __radd__(self: IntList, other: List[Int]) -> IntList:
        return IntList(other).__add__(self)

    fn __add__(self: IntList, other: List[Int]) -> IntList:
        return self.__add__(IntList(other))

    fn __add__(self: IntList, other: IntList) -> IntList:
        if not self.elems and not other.elems:
            return IntList()
        if not self.elems:
            return other.copy()
        if not other.elems:
            return self.copy()

        elems = self.elems + other.elems
        return IntList(elems)

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        index = idx if idx >= 0 else idx + self.__len__()
        if index < 0 or index >= len(self):
            panic("IntList __getitem__  → Out-of-bounds read: " + String(idx))

        return self.elems[index]

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, value: Int):
        if idx < 0 or idx > len(self):
            panic("IntList __setitem__ -> Cannot skip indices")
        if idx == len(self):
            self.append(value)
        else:
            self.elems[idx] = value

    fn replace(self, idx: Int, value: Int) -> Self:
        result = self.copy()
        result[idx] = value
        return result

    @staticmethod
    fn invert_permutation(perm: IntList) -> Self:
        n = len(perm)
        inverted = IntList.filled(n, 0)
        for i in range(n):
            inverted[perm[i]] = i

        return inverted

    fn replace(self: IntList, indices: IntList, values: IntList) -> Self:
        n = len(self)
        m = len(indices)

        if m != len(values):
            panic("IntList -> replace: indices and values must be same length")

        # Validate indices: no out-of-bounds, no duplicates
        for i in range(m):
            idx = indices[i]
            if idx < 0 or idx >= n:
                panic("IntList -> replace: index out of bounds: " + String(idx))

        result = self[::]

        # Apply replacements
        for i in range(m):
            result[indices[i]] = values[i]

        return result

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return len(self.elems)

    @always_inline("nodebug")
    fn len(self) -> Int:
        return len(self.elems)

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
            value = self.elems[i]
            value_str = value.__str__()
            #s += String(self[i])
            s = s + value_str
            if i < len(self) - 1:
                s += ", "
        s += "]"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn prepend(mut self, value: Int):
        self.elems.insert(0, value)

    fn append(mut self, value: Int):
        self.elems.append(value)

    fn resize(mut self, new_capacity: Int):
        self.elems.reserve(new_capacity)

    @always_inline
    fn __contains__(self, value: Int) -> Bool:
        return self.elems.__contains__(value)

    fn reversed(self) -> Self:
        copied = self.copy()
        copied.reverse()
        return copied

    fn reverse(mut self):
        self.elems.reverse()

    fn print(self, limit: Int = 20) -> None:
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

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self.index = existing.index
        self.offset = existing.offset
        self.src_this = existing.src_this
        self.src_that = existing.src_that

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
    pass
    #test_slice()
    #test_deduplicate()
    #test_new()


from testing import assert_true

fn test_with_capacity_fill() raises:
    print("test_with_capacity_fill")
    il = IntList.with_capacity(3, -10)
    assert_true(
        il == IntList(-10, -10, -10) and len(il) == 3,
        "with_capacity with fill assertion failed",
    )

fn test_new() raises:
    print("test_new")
    l = List(1, 2, 3)
    il = IntList.new(l)
    print("il: ", il)
    assert_true(il == IntList(1, 2, 3), "new assertion 1 failed")
    l = List[Int]()
    il = IntList.new(l)
    print("il2: ", il)
    assert_true(il == IntList(), "new assertion 2 failed")


fn test_deduplicate() raises:
    print("test_deduplicate")
    il = IntList(9, 2, 9, 1, 4, 3, 1, 5, 7, 2, 1, 4, 7)
    il.sort_and_deduplicate()
    print("il: ", il)
    assert_true(
        il
        == IntList(
            1,
            2,
            3,
            4,
            5,
            7,
            9,
        ),
        "deduplicate assertion failed",
    )




fn test_slice() raises:
    print("test_slice")
    il = IntList.range_list(15)
    print("il: ", il)
    sliced = il[2::3]
    print(sliced)
    assert_true(sliced == IntList(2, 5, 8, 11, 14), "slice assertion failed")
