from shapes import Shape
from intlist import IntList
from os import abort
from common_utils import log_debug


fn main():
    strides = Strides([])
    for i in strides.to_list().__reversed__():
        print(i)
    print(Strides.Zero)


@register_passable
struct Strides(
    Sized & Copyable & Stringable & Representable & Writable
):
    var strides: IntList
    alias Zero = Self(IntList.Empty)

    fn __init__(out self, values: List[Int]):
        self.strides = IntList.new(values)

    fn __init__(out self, values: IntList):
        self.strides = values.copy()

    fn __copyinit__(out self, existing: Self):
        self.strides = existing.strides.copy()

        _="""fn __moveinit__(out self, deinit existing: Self):
        self.strides = existing.strides"""

    fn __str__(self) -> String:
        var s = String("(")
        for i in range(len(self)):
            s += String(self.strides[i])
            if i < len(self) - 1:
                s += ", "
        s += ")"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    fn __getitem__(self, i: Int) -> Int:
        return self.strides[i]

    fn __getitem__(self, slice: Slice) -> Self:
        strides = self.strides[slice]
        return Strides(strides)

    fn __len__(self) -> Int:
        return len(self.strides)

    fn __eq__(self, other: Self) -> Bool:
        return self.strides == other.strides

    @always_inline
    fn to_list(self) -> IntList:
        return self.strides

    # Reorder dimensions (for transpose/permute)
    fn permute(self, axes: IntList) -> Self:
        log_debug(
            "Stride -> permute: strides "
            + self.strides.__str__()
            + ", axes: "
            + axes.__str__()
        )

        return Strides(self.strides.permute(axes))

    @staticmethod
    fn of(*values: Int) -> Self:
        return Self(IntList(values))

    @staticmethod
    fn default(shape: Shape) -> Self:
        var rank = shape.rank()
        var strides = IntList.filled(rank, 0)
        var acc = 1
        for i in reversed(range(rank)):
            strides[i] = acc
            acc *= shape[i]
        return Strides(strides)

    fn free(deinit self):
        self.strides.free()
