from intlist import IntList
from shapes import Shape
from common_utils import log_debug, panic
from layout.int_tuple import IntArray


fn main():
    s = Shape.of(5, 3)
    strides = Strides.default(s)
    print(strides)
    s1 = Strides(1, 3, 1)
    s2 = Strides([1, 3, 1])
    print(s1 == s2)
    ia = IntArray(3)
    ia[0] = 100
    ia[1] = 200
    ia[2] = 300
    print(s1 * ia)


@register_passable
struct Strides(
    Sized & ImplicitlyCopyable & Stringable & Representable & Writable
):
    var strides: IntArray

    @always_inline
    @staticmethod
    fn Zero() -> Strides:
        return Strides()

    fn __init__(out self):
        self.strides = IntArray()

    fn __init__(out self, values: List[Int]):
        self.strides = IntArray(size=len(values))
        for i in range(len(values)):
            self.strides[i] = values[i]

    fn __init__(out self, values: IntList):
        self.strides = IntArray(size=len(values))
        for i in range(len(values)):
            self.strides[i] = values[i]

    fn __init__(out self, *values: Int):
        ll = List[Int](capacity=UInt(len(values)))
        for v in values:
            ll.append(v)
        self = Self(ll)

    fn __copyinit__(out self, existing: Self):
        self.strides = existing.strides.copy()

    fn __str__(self) -> String:
        capacity = (self.strides.size() * 2) + 1
        var s = String(capacity=capacity)
        s += "("
        for i in range(self.strides.size()):
            s += self.strides[i].__str__()
            if i < self.strides.size() - 1:
                s += ", "
        s += ")"

        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn __getitem__(self, i: Int) -> Int:
        index = i + len(self) if i < 0 else i
        if index < 0 or index >= len(self):
            panic("Strides → __getitem(i)__: invalid index", i.__str__())
        return self.strides[index]

    fn __getitem__(self, slice: Slice) -> Self:
        result = self.list()
        strides = result[slice]
        return Strides(strides)

    fn __setitem__(mut self, i: Int, value: Int):
        index = i + len(self) if i < 0 else i
        if index < 0 or index >= len(self):
            panic("Strides → __setitem(i, value)__: invalid index", i.__str__())
        self.strides[i] = value

    fn __len__(self) -> Int:
        return self.strides.size()

    fn __eq__(self, other: Self) -> Bool:
        return self.list() == other.list()

    fn list(self) -> List[Int]:
        result = List[Int](capacity=UInt(len(self)))
        for i in range(len(self)):
            result.append(self.strides[i])
        return result^

    @always_inline
    fn intlist(self) -> IntList:
        il = IntList.with_capacity(len(self))
        for i in range(len(self)):
            il.append(self[i])
        return il^

    # Reorder dimensions (for transpose/permute)
    @always_inline
    fn permute(self, axes: IntList) -> Self:
        il = self.intlist()
        perm = il.permute(axes)
        return Strides(perm)

    @always_inline
    fn __mul__(self, other: IntArray) -> IntList:
        if len(self) != other.size():
            panic(
                "Strides → __mul__(IntArray): lengths are not equal → ",
                String(len(self)),
                "<=>",
                String(other.size()),
            )
        var result = IntList.with_capacity(len(self))
        for i in range(len(self)):
            result.append(self[i] * other[i])
        return result^

    @staticmethod
    @always_inline
    fn default(shape: Shape) -> Self:
        var rank = shape.rank()
        var strides = List[Int](capacity=UInt(rank))
        var acc = 1
        for i in reversed(range(rank)):
            strides.insert(0, acc)
            acc *= shape[i]
        return Strides(strides)

    @always_inline
    fn is_contiguous(self, shape: Shape) -> Bool:
        if shape.rank() == 0:
            return True  # scalar is trivially contiguous
        var expected_stride = 1
        for i in reversed(range(shape.rank())):
            if shape[i] > 1 and self[i] != expected_stride:
                return False
            expected_stride *= shape[i]

        return True

    @always_inline
    @staticmethod
    fn with_capacity(capacity: Int) -> Strides:
        new_strides = Self()
        new_strides.strides = IntArray(size=capacity)
        return new_strides
