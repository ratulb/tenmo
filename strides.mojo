from shapes import Shape
from intlist import IntList
from os import abort

from testing import assert_true


fn test_compute_default_strides() raises:
    shape = Shape.of(2, 3, 4)
    strides = Strides.default(shape)
    assert_true(
        strides == Strides(IntList(12, 4, 1)),
        "stride compute assertion 1 failed",
    )


fn main() raises:
    test_compute_default_strides()


@register_passable
struct Strides(Sized & Copyable & Stringable & Representable & Writable):
    var strides: IntList

    fn __init__(out self, values: IntList):
        self.strides = values

    fn __eq__(self, other: Self) -> Bool:
        return self.strides == other.strides

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        """Initialize by copying an existing `Strides`.
        Args:
            existing: The Strides to copy from.
        """
        self.strides = existing.strides

    fn is_contiguous(self, shape: Shape) -> Bool:
        if shape.rank() != self.strides.size:
            return False

        var expected_stride = 1
        for i in reversed(range(shape.rank())):
            if self.strides[i] != expected_stride:
                return False
            expected_stride *= shape[i]

        return True

    @staticmethod
    fn of(*values: Int) -> Self:
        return Self(IntList(values))

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

    @always_inline
    fn rank(self) -> Int:
        return len(self)

    fn __getitem__(self, i: Int) -> Int:
        return self.strides[i]

    fn __len__(self) -> Int:
        return len(self.strides)

    fn to_list(self) -> IntList:
        return self.strides

    fn clone(self) -> Self:
        return Strides(self.strides.copy())

    fn print(self):
        print("Strides(" + self.strides.__str__() + ")")

    # Reorder dimensions (for transpose/permute)
    fn permute(self, axes: IntList) -> Self:
        if not len(axes) == len(self):
             abort("Strides -> permute: axes length not equal to strides' length")
        result = IntList.with_capacity(len(axes))
        for axis in axes:
            result.append(self[axis])
        return Strides(result)

    @staticmethod
    fn default(shape: Shape) -> Strides:
        var strides_list = IntList.filled(shape.rank(), 1)

        for i in reversed(range(shape.rank() - 1)):
            strides_list[i] = strides_list[i + 1] * shape[i + 1]

        return Strides(strides_list)

    _ = """# Compute strides from shape in row-major order
    @staticmethod
    fn default(shape: Shape) -> Self:
        var strides = IntList.with_capacity(shape.rank())
        var acc = 1
        for i in reversed(range(shape.rank())):
            strides.prepend(acc)
            acc *= shape[i]
        return Strides(strides)"""

    # Adjust strides for broadcasting to a new shape
    fn broadcast_to(self, from_shape: Shape, to_shape: Shape) -> Self:
        offset = to_shape.rank() - from_shape.rank()
        var result = IntList.with_capacity(to_shape.rank())

        for i in range(to_shape.rank()):
            if i < offset:
                result.append(0)  # new broadcasted dimension
            else:
                from_dim = from_shape[i - offset]
                to_dim = to_shape[i]
                if from_dim == to_dim:
                    result.append(self[i - offset])
                elif from_dim == 1:
                    result.append(0)  # broadcasted dimension
                else:
                    abort("broadcast_to: incompatible shape")

        return Strides(result)
