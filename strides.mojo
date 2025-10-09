from intlist import IntList
from shapes import Shape
from common_utils import log_debug


fn main():
    s = Strides(IntList())
    print(s)


struct Strides(
    Sized & Copyable & Movable & Stringable & Representable & Writable
):
    var strides: List[Int]

    @always_inline
    @staticmethod
    fn Zero() -> Strides:
        return Strides()

    fn __init__(out self):
        self.strides = List[Int](capacity=0)

    fn __init__(out self, values: List[Int]):
        self.strides = values

    fn __init__(out self, values: IntList):
        self.strides = values.tolist()

    fn __copyinit__(out self, existing: Self):
        self.strides = existing.strides.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.strides = existing.strides^

    fn __str__(self) -> String:
        return self.strides.__str__()

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to(self, mut writer: Some[Writer]):
        self.strides.write_to(writer)

    fn __getitem__(self, i: Int) -> Int:
        return self.strides[i]

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[Int]:
        return self.strides.unsafe_ptr()

    fn __getitem__(self, slice: Slice) -> Self:
        strides = self.strides[slice]
        return Strides(strides)

    fn __len__(self) -> Int:
        return len(self.strides)

    fn __eq__(self, other: Self) -> Bool:
        return self.strides == other.strides

    @always_inline
    fn to_list(self) -> IntList:
        return IntList(self.strides)

    # Reorder dimensions (for transpose/permute)
    fn permute(self, axes: IntList) -> Self:
        log_debug(
            "Stride -> permute: strides "
            + self.strides.__str__()
            + ", axes: "
            + axes.__str__()
        )
        il = IntList(self.strides)
        perm = il.permute(axes)
        return Strides(perm)

    @staticmethod
    fn of(*values: Int) -> Self:
        ll = List[Int](capacity=len(values))
        for v in values:
            ll.append(v)
        return Self(ll)

    @staticmethod
    @always_inline
    fn default(shape: Shape) -> Self:
        var rank = shape.rank()
        var strides = List[Int](length=rank, fill=0)
        var acc = 1
        for i in reversed(range(rank)):
            strides[i] = acc
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
