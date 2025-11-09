from common_utils import log_debug, panic
from intlist import IntList
from memory import Pointer
from strides import Strides
from layout.int_tuple import IntArray


@register_passable
struct ShapeIndexIterator[origin: ImmutableOrigin](ImplicitlyCopyable):
    var shape: Pointer[Shape, origin]
    var current: IntArray
    var index: Int

    fn __init__(out self, shape: Pointer[Shape, origin]):
        self.shape = shape
        self.current = IntArray(shape[].rank())
        self.index = 0
        for i in range(self.current.size()):
            self.current[i] = 0

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.current = other.current
        self.index = other.index

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> IntArray:
        result = self.current
        self.index += 1
        for i in range(self.shape[].ndim - 1, -1, -1):
            self.current[i] += 1
            if self.current[i] < self.shape[][i]:
                break
            self.current[i] = 0
        return result

    fn __len__(self) -> Int:
        return self.shape[].num_elements() - self.index

    fn __has_next__(self) -> Bool:
        return self.shape[].num_elements() - self.index > 0


@register_passable
struct IndexIterator[origin: ImmutableOrigin](Copyable):
    var shape: Pointer[Shape, origin]
    var current: IntList
    var index: Int

    fn __init__(out self, shape: Pointer[Shape, origin]):
        self.shape = shape
        self.current = IntList.filled(shape[].rank(), 0)
        self.index = 0

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.current = other.current.copy()
        self.index = other.index

    fn __iter__(self) -> Self:
        return self.copy()

    fn __next__(mut self) -> IntList:
        result = self.current[::]
        self.index += 1
        for i in range(self.shape[].ndim - 1, -1, -1):
            self.current[i] += 1
            if self.current[i] < self.shape[][i]:
                break
            self.current[i] = 0
        return result^

    fn __len__(self) -> Int:
        return self.shape[].num_elements() - self.index

    fn __has_next__(self) -> Bool:
        return self.shape[].num_elements() - self.index > 0


@register_passable
struct Shape(
    Sized & Stringable & Writable & Representable & ImplicitlyCopyable & Movable
):
    var axes_spans: IntList
    var ndim: Int
    var numels: Int

    @always_inline
    @staticmethod
    fn Void() -> Shape:
        return Shape()

    @always_inline
    @staticmethod
    fn Unit() -> Shape:
        return Shape(1)

    fn __init__(out self):
        self = Self(IntList())

    fn __init__(out self, dims: VariadicList[Int]):
        spans = IntList.with_capacity(len(dims))
        for each in dims:
            spans.append(each)
        self = Self(spans)

    fn __init__(out self, *values: Int):
        spans = IntList(values)
        self = Self(spans)

    fn __init__(out self, dims: List[Int]):
        spans = IntList.new(dims)
        self = Self(spans^)

    fn __copyinit__(out self, other: Self):
        self.axes_spans = other.axes_spans.copy()
        self.ndim = other.ndim
        self.numels = other.numels

    fn __init__(out self, dims: IntList):
        ndim = len(dims)
        # Allow scalar tensors (rank 0, i.e., Shape())
        if ndim == 0:
            self.axes_spans = IntList()
            self.ndim = 0
            self.numels = 1
            return
        numels = 1
        for i in range(ndim):
            if dims[i] < 1:
                panic(
                    "Shape → __init__: negative or zero sized dimension(s) are"
                    " not allowed →"
                    + " dimension = "
                    + String(dims[i])
                    + " @index = "
                    + String(i)
                )
            numels *= dims[i]
        self.axes_spans = dims[::]
        self.ndim = ndim
        self.numels = numels

    fn __iter__(ref self) -> ShapeIndexIterator[origin_of(self)]:
        return ShapeIndexIterator(Pointer(to=self))

    fn indices(ref self) -> IndexIterator[origin_of(self)]:
        return IndexIterator(Pointer(to=self))

    @always_inline
    fn count_axes_of_size(self, axis_size: Int) -> Int:
        return self.axes_spans.count(axis_size)

    @always_inline
    fn indices_of_axes_with_size(self, axis_size: Int) -> IntList:
        indices = IntList.with_capacity(len(self))
        for i in range(len(self.axes_spans)):
            if self[i] == axis_size:
                indices.append(i)
        return indices^

    @always_inline
    fn first_index(self) -> IntList:
        return IntList.filled(len(self), 0)

    fn __mul__(self, factor: Int) -> Shape:
        repeated = self.intlist() * factor
        return Shape(repeated)

    fn __rmul__(self, factor: Int) -> Shape:
        return self.__mul__(factor)

    fn __add__(self, other: Shape) -> Shape:
        dims = self.intlist() + other.intlist()
        return Shape(dims)

    fn __radd__(self, other: List[Int]) -> Shape:
        return Shape(IntList(other) + self.intlist())

    fn __add__(self, other: List[Int]) -> Shape:
        dims = self.intlist() + IntList.new(other)
        return Shape(dims)

    fn compute_output_shape(
        self, normalized_axes: IntList, keepdims: Bool, validated: Bool = False
    ) -> Shape:
        """Compute the output shape after reduction along specified axes.

        Args:
            normalized_axes: Sorted list of axes to reduce over.
                - Empty list means reduce over ALL axes.
            keepdims: Whether to keep reduced dimensions as size 1.
            validated: If True, skips axis validation (assumes axes are sorted, unique, and in bounds).

        Returns:
            Shape after reduction

        Behavior:
            - If reducing all axes (empty list or all indices) and keepdims=False → returns Shape() (scalar)
            - Otherwise:
                - For reduced axes: keep as 1 if keepdims=True, else remove
                - For non-reduced axes: keep original size.
        """
        var rank = self.rank()

        # Handle empty axes case: reduce over ALL axes
        if len(normalized_axes) == 0:
            if keepdims:
                # Return shape of all 1's
                var ones = IntList.filled(rank, 1)
                return Shape(ones)
            else:
                return Shape()  # Scalar

        # Validate axes only if not already validated
        if not validated:
            for i in range(len(normalized_axes)):
                var axis = normalized_axes[i]
                if axis < 0 or axis >= rank:
                    panic(
                        (
                            "Shape → compute_output_shape: reduction axis out"
                            " of bounds: normalized_axes: "
                        ),
                        normalized_axes.__str__(),
                        "keepdims: ",
                        keepdims.__str__(),
                        "→ for shape: ",
                        self.__str__(),
                    )
                if i > 0 and axis <= normalized_axes[i - 1]:
                    panic(
                        (
                            "Shape → compute_output_shape: reduction axes must"
                            " be sorted and unique. normalized_axes: "
                        ),
                        normalized_axes.__str__(),
                        "keepdims: ",
                        keepdims.__str__(),
                        "→ for shape: ",
                        self.__str__(),
                    )

        # Full reduction case (all specified axes reduced, not keeping dims)
        if len(normalized_axes) == rank and not keepdims:
            return Shape()

        # Build output shape efficiently using sorted axes property
        var spans = IntList.with_capacity(
            rank if keepdims else rank - len(normalized_axes)
        )
        var axes_index = 0

        for dim in range(rank):
            if (
                axes_index < len(normalized_axes)
                and dim == normalized_axes[axes_index]
            ):
                # This dimension is being reduced
                if keepdims:
                    spans.append(1)
                axes_index += 1
            else:
                # This dimension is kept as-is
                spans.append(self[dim])

        return Shape(spans)

    fn reverse(self) -> Self:
        dims = self.intlist()
        dims.reverse()
        return Shape(dims)

    fn replace(self, axis: Int, extent: Int) -> Shape:
        if axis < 0 or axis >= self.ndim:
            panic(
                "Shape → replace: Invalid axis: "
                + String(axis)
                + " for shape: "
                + self.__str__()
            )
        if extent < 1:
            panic("Shape → replace: Invalid extent: " + String(extent))
        axes = self.intlist()
        axes[axis] = extent
        return Shape(axes)

    @always_inline
    fn __len__(self) -> Int:
        return self.ndim

    @always_inline
    fn rank(self) -> Int:
        return self.ndim

    @always_inline
    fn __getitem__(self, idx: Int) -> Int:
        index = idx if idx >= 0 else idx + self.__len__()
        if 0 <= index < self.ndim:
            return self.axes_spans[index]
        else:
            return -1

    fn __getitem__(self, slice: Slice) -> Self:
        l = self.axes_spans[slice]
        return Self(l)

    fn permute(self, axes: IntList) -> Self:
        log_debug(
            "Stride -> permute: strides "
            + self.axes_spans.__str__()
            + ", axes: "
            + axes.__str__()
        )

        return Shape(self.axes_spans.permute(axes))

    fn __eq__(self, other: List[Int]) -> Bool:
        shape = Self(other)
        return self.__eq__(shape)

    fn __eq__(self, other: Self) -> Bool:
        if self.ndim != other.ndim:
            return False
        for idx in range(self.ndim):
            if self.axes_spans[idx] != other.axes_spans[idx]:
                return False
        return True

    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    @always_inline
    fn num_elements(self) -> Int:
        return self.numels

    fn __str__(self) -> String:
        var s = String("(")
        for i in range(self.ndim):
            s += String(self.axes_spans[i])
            if i < self.ndim - 1:
                s += ", "
        s += ")"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    @always_inline
    fn intlist(self) -> IntList:
        return self.axes_spans.copy()

    @always_inline
    fn tolist(self) -> List[Int]:
        return self.axes_spans.tolist()

    @always_inline
    fn product(shape: Shape) -> Int:
        return shape.intlist().product()

    @staticmethod
    fn of(*dims: Int) -> Shape:
        return Shape(dims)


from common_utils import IntArrayHelper
from testing import assert_true


fn main() raises:
    test_compute_output_shape_with_validation_flag()


fn test_compute_output_shape_with_validation_flag() raises:
    print("test_compute_output_shape_with_validation_flag")
    var shape = Shape(2, 3, 4)

    # Test with default validation (should work)
    assert_true(shape.compute_output_shape(IntList([1]), False) == Shape(2, 4))

    # Test with validated=True (should work and be faster)
    var pre_validated_axes = IntList([0, 2])
    assert_true(
        shape.compute_output_shape(pre_validated_axes, True, validated=True)
        == Shape(1, 3, 1)
    )

    # Test with invalid axes but validated=True (dangerous but allowed)
    # This would panic if validated=False, but with validated=True it assumes caller knows what they're doing
    var risky_axes = IntList([5])  # Out of bounds
    var output_shape = shape.compute_output_shape(
        risky_axes, False, validated=False
    )  # Could cause undefined behavior
    print(output_shape)
