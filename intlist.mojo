from memory import Pointer
from common_utils import panic
from layout.int_tuple import IntArray

@register_passable
struct IntList(
    Sized & ImplicitlyCopyable & Stringable & Representable & Writable
):
    var _storage: IntArray

    @always_inline
    @staticmethod
    fn Empty() -> IntList:
        return IntList()

    @always_inline("nodebug")
    fn __init__(out self, array: IntArray):
        self._storage = array

    fn __init__(out self):
        self._storage = IntArray(size=0)

    @always_inline("nodebug")
    fn __init__(out self, src: List[Int]):
        self._storage = IntArray(size=len(src))
        for i in range(len(src)):
            self._storage[i] = src[i]

    @always_inline("nodebug")
    fn __init__(out self, *elems: Int):
        self._storage = IntArray(size=len(elems))
        for i in range(len(elems)):
            self._storage[i] = elems[i]

    @always_inline("nodebug")
    fn __init__(out self, elems: VariadicList[Int]):
        self._storage = IntArray(size=len(elems))
        for i in range(len(elems)):
            self._storage[i] = elems[i]

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self._storage = existing._storage  # IntArray handles deep copy

    @staticmethod
    @always_inline
    fn new(src: List[Int]) -> IntList:
        out = IntList(src)
        return out^

    @always_inline
    fn tolist(self) -> List[Int]:
        out = List[Int](capacity=len(self))
        for i in range(len(self)):
            out.append(self[i])
        return out^

    @always_inline
    fn intarray(self) -> IntArray:
        return self._storage

    @staticmethod
    @always_inline
    fn range_list(n: Int) -> IntList:
        var out = IntList.with_capacity(n)
        for i in range(n):
            out.append(i)
        return out^

    @staticmethod
    @always_inline
    fn filled(length: Int, value: Int) -> IntList:
        var out = IntList.with_capacity(length)
        for _ in range(length):
            out.append(value)
        return out^

    @staticmethod
    @always_inline
    fn with_capacity(capacity: Int, fill: Optional[Int] = None) -> IntList:
        var out = IntList()
        if fill:
            # Create storage with exact size and fill it
            out._storage = IntArray(capacity)
            for i in range(capacity):
                out._storage[i] = fill.value()
        else:
            # Create empty storage but with allocated capacity
            out._storage = IntArray(size=0)  # Start with empty
        return out^

    @staticmethod
    @always_inline
    fn single_intarray(elem: Int) -> IntArray:
        var out = IntArray(size=1)
        out[0] = elem
        return out^

    fn product(self) -> Int:
        if len(self) == 0:
            return 0
        var prod = 1
        for i in range(len(self)):
            prod *= self[i]
        return prod

    fn sum(self) -> Int:
        var summ = 0
        for i in range(len(self)):
            summ += self[i]
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
        var permuted = IntList.with_capacity(len(axes))
        var seen = IntList.with_capacity(len(axes))
        for ax in axes:
            var axis = ax + len(self) if ax < 0 else ax
            if axis < 0 or axis >= len(self):
                panic("IntList -> permute: index out of bound", ax.__str__())
            if axis in seen:
                panic("IntList -> permute: duplicate axis", ax.__str__())
            seen.append(axis)
            permuted.append(self[axis])

        return permuted^

    fn swap(mut self, this_index: Int, that_index: Int):
        """Swaps elements at different indices."""
        var index1 = this_index + len(self) if this_index < 0 else this_index
        var index2 = that_index + len(self) if that_index < 0 else that_index

        if not 0 <= index1 < len(self) and 0 <= index2 < len(self):
            panic(
                "IntList → swap: provided index(indices) is(are) out of bounds"
            )

        if index1 != index2:
            var temp = self[index1]
            self[index1] = self[index2]
            self[index2] = temp

    fn sort_and_deduplicate(mut self):
        if len(self) <= 1:
            return
        self.sort()
        var write_index = 1
        for read_index in range(1, len(self)):
            if self[read_index] != self[write_index - 1]:
                self[write_index] = self[read_index]
                write_index += 1

        # Shrink the storage
        if write_index < len(self):
            var new_storage = IntArray(size=write_index)
            for i in range(write_index):
                new_storage[i] = self[i]
            self._storage = new_storage

    fn sorted(self, asc: Bool = True) -> IntList:
        var copied = self.copy()
        copied.sort(asc)
        return copied^

    fn sort(mut self, asc: Bool = True):
        # Simple insertion sort (you might want to optimize this later)
        for i in range(1, len(self)):
            var elem = self[i]
            var j = i
            while j > 0 and (elem < self[j - 1] if asc else elem > self[j - 1]):
                self[j] = self[j - 1]
                j -= 1
            self[j] = elem

    fn select(self, indices: IntList) -> IntList:
        var n = len(self)
        var result = IntList.with_capacity(len(indices))

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
        return result^

    fn indices_of(self, val: Int) -> IntList:
        """Returns a new IntList containing all indices where self[i] == val."""
        var out = IntList.with_capacity(len(self))

        for i in range(len(self)):
            if self[i] == val:
                out.append(i)
        return out^

    fn count(self, elem: Int) -> Int:
        var count = 0
        for i in range(len(self)):
            if self[i] == elem:
                count += 1
        return count

    fn any(self, cond: fn (Int) -> Bool) -> Bool:
        for i in range(len(self)):
            if cond(self[i]):
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

        var n = len(self)
        var m = len(indices)
        if n == 0:
            if m == 0:
                return IntList()
            # Ensure indices = [0, 1, ..., m-1]
            if not indices.is_strictly_increasing_from_zero():
                panic(
                    "IntList -> insert: invalid idices for empty list insertion"
                )
            return values.copy()

        var final_size = n + m
        # Step 3: Create dense result
        var result = IntList.with_capacity(final_size)
        var red_cursor = 0
        var orig_cursor = 0
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
        return result^

    fn insert(self, at: Int, value: Int) -> IntList:
        # Insert `value` at position `at` in `self`, return a new IntList
        # `at` could be start, middle or end
        if at < 0 or at > len(self):
            panic("IntList -> insert - index out of bounds: " + String(at))

        var result = IntList.with_capacity(len(self) + 1)
        for i in range(len(self) + 1):
            if i == at:
                result.append(value)
            if i < len(self):
                result.append(self[i])

        return result^

    fn clear(mut self):
        # For IntArray, we create a new empty storage
        self._storage = IntArray(size=0)

    fn __rmul__(self: IntList, factor: Int) -> IntList:
        return self.__mul__(factor)

    fn __mul__(self: IntList, factor: Int) -> IntList:
        if factor < 1 or len(self) == 0:
            return IntList()
        var result = IntList.with_capacity(len(self) * factor)
        for _ in range(factor):
            for i in range(len(self)):
                result.append(self[i])
        return result^

    fn __mul__(self: IntList, other: Self) -> IntList:
        if len(self) != len(other):
            panic(
                "IntList → __mul__(IntList): lengths are not equal → ",
                String(len(self)),
                "<=>",
                String(len(other)),
            )
        var result = IntList.with_capacity(len(self))
        for i in range(len(self)):
            result.append(self[i] * other[i])
        return result^

    fn __mul__(self: IntList, other: IntArray) -> IntList:
        if len(self) != other.size():
            panic(
                "IntList → __mul__(IntArray): lengths are not equal → ",
                String(len(self)),
                "<=>",
                String(other.size()),
            )
        var result = IntList.with_capacity(len(self))
        for i in range(len(self)):
            result.append(self[i] * other[i])
        return result^


    fn __getitem__(self, slice: Slice) -> Self:
        var start = slice.start.value() if slice.start else  0
        var stop = slice.end.value() if slice.end  else len(self)
        var step = slice.step.value() if slice.step else 1

        if start < 0:
            start += len(self)
        if stop < 0:
            stop += len(self)

        var result = IntList.with_capacity((stop - start + step - 1) // step)
        for i in range(start, stop, step):
            result.append(self[i])
        return result^

    fn pop(mut self, index: Int = -1) -> Int:
        if len(self) < 1:
            panic("cannot pop from empty IntList")
        var i = index
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            panic("pop index out of bounds")

        var val = self[i]

        # Create new storage without the popped element
        var new_storage = IntArray(size=(len(self) - 1))
        for j in range(i):
            new_storage[j] = self[j]
        for j in range(i + 1, len(self)):
            new_storage[j - 1] = self[j]

        self._storage = new_storage
        return val

    fn __radd__(self: IntList, other: List[Int]) -> IntList:
        return IntList(other).__add__(self)

    fn __add__(self: IntList, other: List[Int]) -> IntList:
        return self.__add__(IntList(other))

    fn __add__(self: IntList, other: IntList) -> IntList:
        if len(self) == 0 and len(other) == 0:
            return IntList()
        if len(self) == 0:
            return other.copy()
        if len(other) == 0:
            return self.copy()

        var result = IntList.with_capacity(len(self) + len(other))
        for i in range(len(self)):
            result.append(self[i])
        for i in range(len(other)):
            result.append(other[i])
        return result^

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        var index = idx if idx >= 0 else idx + len(self)
        if index < 0 or index >= len(self):
            panic("IntList __getitem__  → Out-of-bounds read: " + String(idx))

        return self._storage[index]

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, value: Int):
        var index = idx if idx >= 0 else idx + len(self)
        if index < 0 or index > len(self):
            panic("IntList __setitem__ -> Cannot skip indices")
        if index == len(self):
            self.append(value)
        else:
            self._storage[index] = value

    fn replace(self, idx: Int, value: Int) -> Self:
        var result = self.copy()
        result[idx] = value
        return result^

    @staticmethod
    fn invert_permutation(perm: IntList) -> Self:
        var n = len(perm)
        var inverted = IntList.filled(n, 0)
        for i in range(n):
            inverted[perm[i]] = i

        return inverted^

    fn replace(self: IntList, indices: IntList, values: IntList) -> Self:
        var n = len(self)
        var m = len(indices)

        if m != len(values):
            panic("IntList -> replace: indices and values must be same length")

        # Validate indices: no out-of-bounds, no duplicates
        for i in range(m):
            var idx = indices[i]
            if idx < 0 or idx >= n:
                panic("IntList -> replace: index out of bounds: " + String(idx))

        var result = self.copy()

        # Apply replacements
        for i in range(m):
            result[indices[i]] = values[i]

        return result^

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return self._storage.size()

    @always_inline("nodebug")
    fn len(self) -> Int:
        return self._storage.size()

    @always_inline("nodebug")
    fn size(self) -> Int:
        return self._storage.size()

    fn is_empty(self) -> Bool:
        return len(self) == 0

    fn __eq__(self: IntList, other: IntList) -> Bool:
        if len(self) != len(other):
            return False
        for i in range(len(self)):
            if self[i] != other[i]:
                return False
        return True

    fn __eq__(self: IntList, other: IntArray) -> Bool:
        if len(self) != other.size():
            return False
        for i in range(len(self)):
            if self[i] != other[i]:
                return False
        return True


    fn __str__(self) -> String:
        if len(self) == 0:
            return "[]"
        var result = String("[")
        for i in range(len(self)):
            if i > 0:
                result += ", "
            result += String(self[i])
        result += "]"
        return result

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(self.__str__())

    fn prepend(mut self, value: Int):
        # Create new storage with +1 capacity
        var new_storage = IntArray(size=(len(self) + 1))
        new_storage[0] = value
        for i in range(len(self)):
            new_storage[i + 1] = self[i]
        self._storage = new_storage

    fn append(mut self, value: Int):
        # Create new storage with +1 capacity
        var new_storage = IntArray(size=(len(self) + 1))
        for i in range(len(self)):
            new_storage[i] = self[i]
        new_storage[len(self)] = value
        self._storage = new_storage

    @always_inline
    fn __contains__(self, value: Int) -> Bool:
        for i in range(len(self)):
            if self[i] == value:
                return True
        return False

    fn reversed(self) -> Self:
        var copied = self.copy()
        copied.reverse()
        return copied^

    fn reverse(mut self):
        var n = len(self)
        for i in range(n // 2):
            var temp = self[i]
            self[i] = self[n - 1 - i]
            self[n - 1 - i] = temp

    fn print(self, limit: Int = 20) -> None:
        var total = len(self)
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


fn main() raises:
    test_simple_slice()
    _="""il = IntList()
    il.append(1)
    il.append(3)
    il.append(2)
    print(il)"""

fn test_simple_slice() raises:
    print("test_simple_slice")
    il = IntList(0, 1, 2, 3, 4, 5)
    sliced = il[2:5:1]  # Should be [2, 3, 4]
    print("sliced: ", sliced)
    print("expected: [2, 3, 4]")
    array = IntArray(size=3)
    array[0] = 100
    array[1] = 200
    array[2] = 300
    il = IntList(array^)
    print(il)
    print("Below")
    print(IntList.single_intarray(il[2])[0])

    a = IntList(1, 2, 3)
    result = a * a
    print(result)
