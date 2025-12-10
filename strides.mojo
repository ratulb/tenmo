from intarray import IntArray
from shapes import Shape


@register_passable
struct Strides(ImplicitlyCopyable, Representable, Sized, Stringable, Writable):
    """Strides for tensor indexing."""

    var data: IntArray

    @staticmethod
    @always_inline
    fn Zero() -> Strides:
        return Strides()

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
    fn __copyinit__(out self, existing: Self):
        self.data = existing.data

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return len(self.data)

    @always_inline("nodebug")
    fn __getitem__(self, i: Int) -> Int:
        return self.data[i]

    @always_inline("nodebug")
    fn __setitem__(mut self, i: Int, value: Int):
        self.data[i] = value

    @always_inline("nodebug")
    fn __getitem__(self, slice: Slice) -> Self:
        return Strides(self.data[slice])

    fn __eq__(self, other: Self) -> Bool:
        return self.data == other.data

    fn __str__(self) -> String:
        return "(" + self.data.__str__()[1:-1] + ")"

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    @always_inline
    fn tolist(self) -> List[Int]:
        return self.data.tolist()

    @always_inline
    fn intarray(self) -> IntArray:
        return self.data

    @always_inline
    fn permute(self, axes: IntArray) -> Self:
        """Reorder dimensions."""
        var result = IntArray.with_capacity(len(axes))
        for i in range(len(axes)):
            result.append(self[axes[i]])
        return Strides(result)

    @staticmethod
    @always_inline
    fn default(shape: Shape) -> Self:
        """Compute default C-contiguous strides."""
        var rank = shape.rank()
        var strides = IntArray.with_capacity(rank)
        var acc = 1
        for i in range(rank - 1, -1, -1):
            strides.prepend(acc)
            acc *= shape[i]
        return Strides(strides)

    @always_inline
    fn is_contiguous(self, shape: Shape) -> Bool:
        """Check if strides represent contiguous layout."""
        if shape.rank() == 0:
            return True
        var expected = 1
        for i in range(shape.rank() - 1, -1, -1):
            if shape[i] > 1 and self[i] != expected:
                return False
            expected *= shape[i]
        return True

    @staticmethod
    @always_inline
    fn with_capacity(capacity: Int) -> Strides:
        return Strides(IntArray.with_capacity(capacity))
