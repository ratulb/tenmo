from memory import Pointer
from common_utils import panic
from layout.int_tuple import IntArray

@register_passable
struct IntList(
    Sized & ImplicitlyCopyable & Stringable & Representable & Writable
):
    var _storage: IntArray
    var _size: Int  # Current number of elements (<= storage capacity)

    @always_inline
    @staticmethod
    fn Empty() -> IntList:
        return IntList()

    @always_inline("nodebug")
    fn __init__(out self, array: IntArray):
        self._storage = array
        self._size = array.size()  # Use full capacity initially

    fn __init__(out self):
        self._storage = IntArray(size=0)
        self._size = 0

    @always_inline("nodebug")
    fn __init__(out self, src: List[Int]):
        src_len = len(src)
        self._storage = IntArray(size=src_len)
        self._size = src_len
        for i in range(src_len):
            self._storage[i] = src[i]

    @always_inline("nodebug")
    fn __init__(out self, *elems: Int):
        elems_len = len(elems)
        self._storage = IntArray(size=elems_len)
        self._size = elems_len
        for i in range(elems_len):
            self._storage[i] = elems[i]

    @always_inline("nodebug")
    fn __init__(out self, elems: VariadicList[Int]):
        elems_len = len(elems)
        self._storage = IntArray(size=elems_len)
        self._size = elems_len
        for i in range(elems_len):
            self._storage[i] = elems[i]

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self._storage = existing._storage  # IntArray handles deep copy
        self._size = existing._size

    @staticmethod
    @always_inline
    fn new(src: List[Int]) -> IntList:
        out = IntList(src)
        return out^

    @always_inline
    fn tolist(self) -> List[Int]:
        out = List[Int](capacity=self._size)
        for i in range(self._size):
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
        if capacity > 0:
            out._storage = IntArray(capacity)
            if fill:
                out._size = capacity
                for i in range(capacity):
                    out._storage[i] = fill.value()
            else:
                out._size = 0  # Start with zero elements but allocated capacity
        # else: remains empty (size=0, storage=IntArray(0))
        return out^

    fn _ensure_capacity(mut self, required_capacity: Int):
        """Ensure we have enough capacity, reallocating if necessary."""
        current_capacity = self._storage.size()
        if required_capacity <= current_capacity:
            return

        # Grow by at least 1.5x, but at least required_capacity
        new_capacity = max(required_capacity, current_capacity * 3 // 2 + 1)
        var new_storage = IntArray(size=new_capacity)

        # Copy existing elements
        for i in range(self._size):
            new_storage[i] = self._storage[i]

        self._storage = new_storage

    fn product(self) -> Int:
        if self._size == 0:
            return 0
        var prod = 1
        for i in range(self._size):
            prod *= self[i]
        return prod

    @always_inline
    fn sum(self) -> Int:
        var accum_sum = 0
        for i in range(self._size):
            accum_sum += self[i]
        return accum_sum

    fn has_duplicates(self) -> Bool:
        if self._size <= 1:
            return False
        var sorted_list = self.sorted()

        for i in range(len(sorted_list) - 1):
            if sorted_list[i] == sorted_list[i + 1]:
                return True

        return False

    fn permute(self, axes: IntList) -> IntList:
        if not self._size == len(axes):
            panic(
                "IntList -> permute: axes length",
                len(axes).__str__(),
                "does not match list's length",
                self._size.__str__(),
            )
        var permuted = IntList.with_capacity(len(axes))
        var seen = IntList.with_capacity(len(axes))
        for ax in axes:
            var axis = ax + self._size if ax < 0 else ax
            if axis < 0 or axis >= self._size:
                panic("IntList -> permute: index out of bound", ax.__str__())
            if axis in seen:
                panic("IntList -> permute: duplicate axis", ax.__str__())
            seen.append(axis)
            permuted.append(self[axis])

        return permuted^

    fn swap(mut self, this_index: Int, that_index: Int):
        """Swaps elements at different indices."""
        var index1 = this_index + self._size if this_index < 0 else this_index
        var index2 = that_index + self._size if that_index < 0 else that_index

        if not 0 <= index1 < self._size and 0 <= index2 < self._size:
            panic(
                "IntList → swap: provided index(indices) is(are) out of bounds"
            )

        if index1 != index2:
            var temp = self[index1]
            self[index1] = self[index2]
            self[index2] = temp

    fn sort_and_deduplicate(mut self):
        if self._size <= 1:
            return
        self.sort()
        var write_index = 1
        for read_index in range(1, self._size):
            if self[read_index] != self[write_index - 1]:
                self[write_index] = self[read_index]
                write_index += 1

        # Update size to remove duplicates
        self._size = write_index
        # Note: We don't shrink storage here to avoid reallocation

    fn sorted(self, asc: Bool = True) -> IntList:
        var copied = self.copy()
        copied.sort(asc)
        return copied^

    fn sort(mut self, asc: Bool = True):
        # Simple insertion sort (you might want to optimize this later)
        for i in range(1, self._size):
            var elem = self[i]
            var j = i
            while j > 0 and (elem < self[j - 1] if asc else elem > self[j - 1]):
                self[j] = self[j - 1]
                j -= 1
            self[j] = elem

    fn select(self, indices: IntList) -> IntList:
        var n = self._size
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
        var out = IntList.with_capacity(self._size)

        for i in range(self._size):
            if self[i] == val:
                out.append(i)
        return out^

    fn count(self, elem: Int) -> Int:
        var count = 0
        for i in range(self._size):
            if self[i] == elem:
                count += 1
        return count

    fn any(self, cond: fn (Int) -> Bool) -> Bool:
        for i in range(self._size):
            if cond(self[i]):
                return True
        return False

    @always_inline
    fn is_strictly_increasing_from_zero(self) -> Bool:
        for i in range(self._size):
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

        var n = self._size
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
        if at < 0 or at > self._size:
            panic("IntList -> insert - index out of bounds: " + String(at))

        var result = IntList.with_capacity(self._size + 1)
        for i in range(self._size + 1):
            if i == at:
                result.append(value)
            if i < self._size:
                result.append(self[i])

        return result^

    fn clear(mut self):
        self._size = 0
        # Note: We keep the storage to avoid reallocation

    fn __rmul__(self: IntList, factor: Int) -> IntList:
        return self.__mul__(factor)

    fn __mul__(self: IntList, factor: Int) -> IntList:
        if factor < 1 or self._size == 0:
            return IntList()
        var result = IntList.with_capacity(self._size * factor)
        for _ in range(factor):
            for i in range(self._size):
                result.append(self[i])
        return result^

    fn __mul__(self: IntList, other: Self) -> IntList:
        if self._size != len(other):
            panic(
                "IntList → __mul__(IntList): lengths are not equal → ",
                String(self._size),
                "<=>",
                String(len(other)),
            )
        var result = IntList.with_capacity(self._size)
        for i in range(self._size):
            result.append(self[i] * other[i])
        return result^

    fn __mul__(self: IntList, other: IntArray) -> IntList:
        if self._size != other.size():
            panic(
                "IntList → __mul__(IntArray): lengths are not equal → ",
                String(self._size),
                "<=>",
                String(other.size()),
            )
        var result = IntList.with_capacity(self._size)
        for i in range(self._size):
            result.append(self[i] * other[i])
        return result^

    fn __getitem__(self, slice: Slice) -> Self:
        var start = slice.start.value() if slice.start else  0
        var stop = slice.end.value() if slice.end  else self._size
        var step = slice.step.value() if slice.step else 1

        if start < 0:
            start += self._size
        if stop < 0:
            stop += self._size

        var result = IntList.with_capacity((stop - start + step - 1) // step)
        for i in range(start, stop, step):
            result.append(self[i])
        return result^

    fn pop(mut self, index: Int = -1) -> Int:
        if self._size < 1:
            panic("cannot pop from empty IntList")
        var i = index
        if i < 0:
            i += self._size
        if i < 0 or i >= self._size:
            panic("pop index out of bounds")

        var val = self[i]

        # Shift elements left
        for j in range(i, self._size - 1):
            self._storage[j] = self._storage[j + 1]

        self._size -= 1
        return val

    fn __radd__(self: IntList, other: List[Int]) -> IntList:
        return IntList(other).__add__(self)

    fn __add__(self: IntList, other: List[Int]) -> IntList:
        return self.__add__(IntList(other))

    fn __add__(self: IntList, other: IntList) -> IntList:
        if self._size == 0 and len(other) == 0:
            return IntList()
        if self._size == 0:
            return other.copy()
        if len(other) == 0:
            return self.copy()

        var result = IntList.with_capacity(self._size + len(other))
        for i in range(self._size):
            result.append(self[i])
        for i in range(len(other)):
            result.append(other[i])
        return result^

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        var index = idx if idx >= 0 else idx + self._size
        if index < 0 or index >= self._size:
            panic("IntList __getitem__  → Out-of-bounds read: " + String(idx))

        return self._storage[index]

    @always_inline("nodebug")
    fn __setitem__(mut self, idx: Int, value: Int):
        var index = idx if idx >= 0 else idx + self._size
        if index < 0 or index > self._size:
            panic("IntList __setitem__ -> Cannot skip indices")
        if index == self._size:
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
        var n = self._size
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
        return self._size

    @always_inline("nodebug")
    fn len(self) -> Int:
        return self._size

    @always_inline("nodebug")
    fn size(self) -> Int:
        return self._size

    fn capacity(self) -> Int:
        return self._storage.size()

    fn is_empty(self) -> Bool:
        return self._size == 0

    fn __eq__(self: IntList, other: IntList) -> Bool:
        if self._size != len(other):
            return False
        for i in range(self._size):
            if self[i] != other[i]:
                return False
        return True

    fn __eq__(self: IntList, other: IntArray) -> Bool:
        if self._size != other.size():
            return False
        for i in range(self._size):
            if self[i] != other[i]:
                return False
        return True

    fn __str__(self) -> String:
        if self._size == 0:
            return "[]"
        var result = String("[")
        for i in range(self._size):
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
        self._ensure_capacity(self._size + 1)

        # Shift elements right
        for i in range(self._size, 0, -1):
            self._storage[i] = self._storage[i - 1]

        self._storage[0] = value
        self._size += 1

    fn append(mut self, value: Int):
        self._ensure_capacity(self._size + 1)
        self._storage[self._size] = value
        self._size += 1

    @always_inline
    fn __contains__(self, value: Int) -> Bool:
        for i in range(self._size):
            if self[i] == value:
                return True
        return False

    fn reversed(self) -> Self:
        var copied = self.copy()
        copied.reverse()
        return copied^

    fn reverse(mut self):
        var n = self._size
        for i in range(n // 2):
            var temp = self[i]
            self[i] = self[n - 1 - i]
            self[n - 1 - i] = temp

    fn print(self, limit: Int = 20) -> None:
        var total = self._size
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
        return Iterator[forward=False](self._size, Pointer(to=self))

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
            min(self._size, len(other)), Pointer(to=self), Pointer(to=other)
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

    a = IntList(1, 2, 3)
    result = a * a
    print(result)
