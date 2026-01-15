from common_utils import panic
from intarray import IntArray


@register_passable
struct Shape(
    EqualityComparable,
    ImplicitlyCopyable,
    Movable,
    Representable,
    Sized,
    Stringable,
    Writable,
):
    """Shape of a tensor."""

    var dims: IntArray
    var _numels: Int

    @staticmethod
    @always_inline
    fn Void() -> Shape:
        return Shape()

    @staticmethod
    @always_inline
    fn Unit() -> Shape:
        return Shape(1)

    @always_inline("nodebug")
    fn __init__(out self):
        self.dims = IntArray()
        self._numels = 1

    @always_inline("nodebug")
    fn __init__(out self, *values: Int):
        self.dims = IntArray(values)
        self._numels = 0
        self._numels = self._compute_numels()

    @always_inline("nodebug")
    fn __init__(out self, values: VariadicList[Int]):
        self.dims = IntArray(values)
        self._numels = 0
        self._numels = self._compute_numels()

    @always_inline("nodebug")
    fn __init__(out self, values: List[Int]):
        self.dims = IntArray(values)
        self._numels = 0
        self._numels = self._compute_numels()

    @always_inline("nodebug")
    fn __init__(out self, values: IntArray):
        self.dims = values
        self._numels = 0
        self._numels = self._compute_numels()

    @always_inline("nodebug")
    fn __copyinit__(out self, existing: Self):
        self.dims = existing.dims
        self._numels = existing._numels

    @always_inline("nodebug")
    fn _compute_numels(self) -> Int:
        """Compute number of elements, validating dimensions."""
        if len(self.dims) == 0:
            return 1
        var prod = 1
        for i in range(len(self.dims)):
            if self.dims[i] < 1:
                panic(
                    "Shape: dimension must be >= 1, got " + String(self.dims[i])
                )
            prod *= self.dims[i]
        return prod

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return len(self.dims)

    @always_inline("nodebug")
    fn rank(self) -> Int:
        return len(self.dims)

    @always_inline("nodebug")
    fn ndim(self) -> Int:
        return len(self.dims)

    @always_inline("nodebug")
    fn num_elements(self) -> Int:
        return self._numels

    @always_inline("nodebug")
    fn numels(self) -> Int:
        return self._numels

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        return self.dims[idx]

    @always_inline
    fn __getitem__(self, slice: Slice) -> Self:
        s = self.dims[slice]
        return Self(s)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return self.dims == other.dims

    @always_inline("nodebug")
    fn __eq__(self, other: List[Int]) -> Bool:
        return self.dims == other

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self.dims == other.dims)

    fn __str__(self) -> String:
        return "(" + self.dims.__str__()[1:-1] + ")"

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    @always_inline("nodebug")
    fn __iter__(ref self) -> ShapeIndexIterator[origin_of(self)]:
        return ShapeIndexIterator(Pointer(to=self))

    @always_inline("nodebug")
    fn tolist(self) -> List[Int]:
        return self.dims.tolist()

    @always_inline("nodebug")
    fn intarray(self) -> IntArray:
        return self.dims

    # ========== Operations ==========

    @always_inline("nodebug")
    fn __add__(self, other: Shape) -> Shape:
        """Concatenate shapes."""
        # Use extend for bulk copy instead of loop
        dims = self.dims + other.dims
        return Shape(dims^)

    @always_inline("nodebug")
    fn __add__(self, other: List[Int]) -> Shape:
        """Concatenate with list."""
        dims = self.dims + other
        return Shape(dims^)

    @always_inline("nodebug")
    fn __radd__(self, other: List[Int]) -> Shape:
        """Concatenate list + shape."""
        var from_list = IntArray(other)
        dims = from_list + self.dims
        return Shape(dims^)

    @always_inline("nodebug")
    fn __mul__(self, factor: Int) -> Shape:
        """Repeat shape."""
        if factor <= 0:
            return Shape()
        var result = self.dims
        for _ in range(factor - 1):
            result = result + self.dims
        return Shape(result)

    @always_inline("nodebug")
    fn __rmul__(self, factor: Int) -> Shape:
        return self.__mul__(factor)

    @always_inline("nodebug")
    fn reverse(self) -> Self:
        """Return reversed shape."""
        return Shape(self.dims.reversed())

    @always_inline("nodebug")
    fn replace(self, axis: Int, extent: Int) -> Shape:
        """Replace dimension at axis."""
        if axis < 0 or axis >= len(self):
            panic("Shape: invalid axis " + String(axis))
        if extent < 1:
            panic("Shape: invalid extent " + String(extent))
        var result = self.dims
        result[axis] = extent
        return Shape(result)

    @always_inline("nodebug")
    fn permute(self, axes: IntArray) -> Self:
        """Reorder dimensions."""
        var result = IntArray.with_capacity(len(axes))
        result.reserve(len(axes))  # Guarantee no realloc
        for i in range(len(axes)):
            result.append(self[axes[i]])
        return Shape(result)

    @always_inline
    fn count_axes_of_size(self, size: Int) -> Int:
        """Count dimensions with given size."""
        var count = 0
        for i in range(len(self)):
            if self[i] == size:
                count += 1
        return count

    @always_inline("nodebug")
    fn indices_of_axes_with_size(self, size: Int) -> IntArray:
        """Get indices of dimensions with given size."""
        var result = IntArray.with_capacity(len(self))
        for i in range(len(self)):
            if self[i] == size:
                result.append(i)
        return result^

    @always_inline("nodebug")
    fn first_index(self) -> IntArray:
        """Get first index (all zeros)."""
        return IntArray.filled(len(self), 0)

    @always_inline("nodebug")
    fn compute_output_shape(
        self, normalized_axes: IntArray, keepdims: Bool, validated: Bool = False
    ) -> Shape:
        """Compute output shape after reduction.

        Args:
            normalized_axes: Sorted axes to reduce (empty = reduce all).
            keepdims: Keep reduced dims as 1.
            validated: Skip validation if True.
        """
        var rank = self.rank()

        # Reduce all axes
        if len(normalized_axes) == 0:
            if keepdims:
                return Shape(IntArray.filled(rank, 1))
            else:
                return Shape()

        # Validate if needed
        if not validated:
            for i in range(len(normalized_axes)):
                var axis = normalized_axes[i]
                if axis < 0 or axis >= rank:
                    panic("Shape: reduction axis out of bounds")
                if i > 0 and axis <= normalized_axes[i - 1]:
                    panic("Shape: reduction axes must be sorted and unique")

        # Full reduction without keepdims
        if len(normalized_axes) == rank and not keepdims:
            return Shape()

        # Build output shape
        var expected_size = rank if keepdims else rank - len(normalized_axes)
        var result = IntArray.with_capacity(expected_size)
        result.reserve(expected_size)  # Guarantee no realloc
        var axes_idx = 0

        for dim in range(rank):
            if (
                axes_idx < len(normalized_axes)
                and dim == normalized_axes[axes_idx]
            ):
                if keepdims:
                    result.append(1)
                axes_idx += 1
            else:
                result.append(self[dim])

        return Shape(result)

    @always_inline("nodebug")
    fn reduced_shape(self, axes: IntArray) -> Shape:
        if len(axes) > self.rank():
            panic("Shape -> reduced_shape: axes greater that shape rank")
        var reduced_axes = IntArray.with_capacity(len(axes))
        for i in range(len(axes)):
            reduced_axes.append(self[axes[i]])
        return Shape(reduced_axes)

    @staticmethod
    fn of(*dims: Int) -> Shape:
        return Shape(dims)

    @always_inline
    fn product(self) -> Int:
        return self._numels


@register_passable
struct ShapeIndexIterator[origin: ImmutOrigin](ImplicitlyCopyable):
    """Iterator over IntArray coordinates of a shape."""

    var shape: Pointer[Shape, Self.origin]
    var current: IntArray
    var index: Int

    fn __init__(out self, shape: Pointer[Shape, Self.origin]):
        self.shape = shape
        self.current = IntArray.filled(shape[].rank(), 0)
        self.index = 0

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.current = other.current
        self.index = other.index

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> IntArray:
        var result = self.current
        self.index += 1
        # This loop is hot - uses __setitem__ which is already optimized
        ref shape = self.shape[]
        var rank = shape.rank()

        for i in range(rank - 1, -1, -1):
            self.current[i] += 1
            if self.current[i] < shape[i]:
                break
            self.current[i] = 0
        return result

    fn __len__(self) -> Int:
        return self.shape[].num_elements() - self.index

    fn __has_next__(self) -> Bool:
        return self.index < self.shape[].num_elements()
