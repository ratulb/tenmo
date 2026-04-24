from .common_utils import panic
from .intarray import IntArray
from .array import Array


struct Shape(
    Equatable,
    ImplicitlyCopyable,
    Iterable,
    Movable,
    RegisterPassable,
    Sized,
    Writable,
):
    """Represents the shape (dimensions) of a tensor.

    A Shape instance stores the dimensions of a multi-dimensional array,
    where each dimension represents the size along a particular axis.
    For example, a 2x3 matrix has shape (2, 3).

    Shape instances are immutable - operations that modify dimensions
    return new Shape instances rather than modifying in place.
    """
    var dims: IntArray
    var _numels: Int

    @staticmethod
    @always_inline
    fn Void() -> Shape:
        """Create a void (scalar) shape.

        Returns:
            Shape representing a scalar (0-dimensional tensor).
            Equivalent to shape with no dimensions.
        """
        return Shape()

    @staticmethod
    @always_inline
    fn Unit() -> Shape:
        """Create a unit shape with single element.

        Returns:
            Shape representing a single element (1-dimensional with size 1).
        """
        return Shape(1)

    @always_inline("nodebug")
    fn __init__(out self):
        """Create an empty (scalar) shape.

        Creates a scalar shape with no dimensions, representing a single value.
        The number of elements is 1 (the multiplicative identity).
        """
        self.dims = IntArray()
        self._numels = 1

    @always_inline("nodebug")
    fn __init__(out self, *values: Int):
        """Create a shape from variadic integer arguments.

        Args:
            *values: Variable number of integers representing dimension sizes

        Example:
            Shape(2, 3, 4) creates a shape with dimensions (2, 3, 4)
        """
        self.dims = IntArray(values)
        self._numels = 0
        self._numels = self._compute_numels()

    @always_inline("nodebug")
    fn __init__(out self, values: VariadicList[Int, _]):
        """Create a shape from a VariadicList of integers.

        Args:
            values: VariadicList of integers representing dimension sizes

        Example:
            Shape(VariadicList(2, 3, 4)) creates a shape with dimensions (2, 3, 4)
        """
        self.dims = IntArray(values)
        self._numels = 0
        self._numels = self._compute_numels()

    @always_inline("nodebug")
    fn __init__(out self, values: List[Int]):
        """Create a shape from a Mojo List of integers.

        Args:
            values: Mojo List of integers representing dimension sizes

        Example:
            Shape([2, 3, 4]) creates a shape with dimensions (2, 3, 4)
        """
        self.dims = IntArray(values)
        self._numels = 0
        self._numels = self._compute_numels()

    @always_inline("nodebug")
    fn __init__(out self, values: IntArray):
        """Create a shape from an IntArray.

        Args:
            values: IntArray representing dimension sizes

        Example:
            Shape(IntArray(2, 3, 4)) creates a shape with dimensions (2, 3, 4)
        """
        self.dims = values
        self._numels = 0
        self._numels = self._compute_numels()

    @always_inline("nodebug")
    fn __copyinit__(out self, copy: Self):
        """Create a copy of another Shape instance.

        Args:
            copy: Shape instance to copy from
        """
        self.dims = copy.dims
        self._numels = copy._numels

    @always_inline("nodebug")
    fn _compute_numels(self) -> Int:
        """Compute number of elements, validating dimensions.

        Returns:
            Total number of elements (product of all dimensions)

        Raises:
            Panic if any dimension is less than 1.
        """
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
        """Get the rank (number of dimensions) of the shape.

        Returns:
            Number of dimensions in the shape.
            This enables len() function support.
        """
        return len(self.dims)

    @always_inline("nodebug")
    fn rank(self) -> Int:
        """Get the rank (number of dimensions) of the shape.

        Returns:
            Number of dimensions in the shape.
        """
        return len(self.dims)

    @always_inline("nodebug")
    fn ndim(self) -> Int:
        """Get the number of dimensions (alias for rank()).

        Returns:
            Number of dimensions in the shape.
        """
        return len(self.dims)

    @always_inline("nodebug")
    fn num_elements(self) -> Int:
        """Get the total number of elements in a tensor with this shape.

        Returns:
            Product of all dimensions.
        """
        return self._numels

    @always_inline("nodebug")
    fn numels(self) -> Int:
        """Get the total number of elements (alias for num_elements()).

        Returns:
            Product of all dimensions.
        """
        return self._numels

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Int:
        """Get the size of a specific dimension.

        Args:
            idx: Index of the dimension (supports negative indexing)

        Returns:
            Size of the dimension at the specified index

        Raises:
            Panic if index is out of bounds.
        """
        return self.dims[idx]

    @always_inline
    fn __getitem__(self, slice: Slice) -> Self:
        """Get a slice of the shape dimensions.

        Args:
            slice: Slice specifying which dimensions to extract

        Returns:
            New Shape with the sliced dimensions
        """
        s = self.dims[slice]
        return Self(s)

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        """Check equality with another Shape.

        Args:
            other: Shape instance to compare with

        Returns:
            True if both shapes have the same dimensions, False otherwise
        """
        return self.dims == other.dims

    @always_inline("nodebug")
    fn __eq__(self, other: List[Int]) -> Bool:
        """Check equality with a Mojo List.

        Args:
            other: Mojo List of integers to compare with

        Returns:
            True if shape dimensions equal the list, False otherwise
        """
        return self.dims == other

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        """Check inequality with another Shape.

        Args:
            other: Shape instance to compare with

        Returns:
            True if shapes have different dimensions, False otherwise
        """
        return not (self.dims == other.dims)

    @no_inline
    fn __str__(self) -> String:
        """Get string representation of the shape.

        Returns:
            String in the format "(d0, d1, ..., dn)"
        """
        var s = self.dims.__str__()
        return "(" + s[byte=1:len(s)-1] + ")"

    @no_inline
    fn __repr__(self) -> String:
        """Get official string representation of the shape.

        Returns:
            String representation suitable for debugging.
        """
        return self.__str__()

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """Write shape to a writer.

        Args:
            writer: Writer to write the shape to
        """
        writer.write(self.__str__())

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = ShapeIndexIterator[iterable_origin]

    @always_inline("nodebug")
    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Get iterator over indices of the shape.

        Returns:
            Iterator that yields all valid index combinations.
        """
        return {Pointer(to=self).get_immutable()}

    @always_inline("nodebug")
    fn tolist(self) -> List[Int]:
        """Convert shape dimensions to a Mojo List.

        Returns:
            Mojo List containing the dimension sizes
        """
        return self.dims.tolist()

    @always_inline("nodebug")
    fn intarray(self) -> IntArray:
        """Get the underlying IntArray of dimensions.

        Returns:
            IntArray containing the dimension sizes
        """
        return self.dims

    @always_inline("nodebug")
    fn array(self) -> Array:
        """Convert shape to an Array.

        Returns:
            Array containing the dimension sizes
        """
        var result = Array()
        result.size = len(self)
        for i in range(len(self)):
            result[i] = self[i]
        return result^


    @always_inline("nodebug")
    fn __add__(self, other: Shape) -> Shape:
        """Concatenate two shapes.

        Args:
            other: Shape to concatenate with this shape

        Returns:
            New Shape with dimensions from both shapes combined

        Example:
            Shape(2, 3) + Shape(4, 5) -> Shape(2, 3, 4, 5)
        """
        # Use extend for bulk copy instead of loop
        dims = self.dims + other.dims
        return Shape(dims^)

    @always_inline("nodebug")
    fn __add__(self, other: List[Int]) -> Shape:
        """Concatenate shape with a Mojo List.

        Args:
            other: Mojo List of integers to concatenate

        Returns:
            New Shape with dimensions from both combined
        """
        dims = self.dims + other
        return Shape(dims^)

    @always_inline("nodebug")
    fn __radd__(self, other: List[Int]) -> Shape:
        """Concatenate a Mojo List with a shape.

        Args:
            other: Mojo List of integers to prepend

        Returns:
            New Shape with list dimensions followed by shape dimensions
        """
        var from_list = IntArray(other)
        dims = from_list + self.dims
        return Shape(dims^)

    @always_inline("nodebug")
    fn __mul__(self, factor: Int) -> Shape:
        """Repeat shape dimensions.

        Args:
            factor: Number of times to repeat the shape dimensions

        Returns:
            New Shape with dimensions repeated factor times

        Example:
            Shape(2, 3) * 2 -> Shape(2, 3, 2, 3)
        """
        if factor <= 0:
            return Shape()
        var result = self.dims
        for _ in range(factor - 1):
            result = result + self.dims
        return Shape(result)

    @always_inline("nodebug")
    fn __rmul__(self, factor: Int) -> Shape:
        """Repeat shape dimensions (reverse multiplication).

        Args:
            factor: Number of times to repeat the shape dimensions

        Returns:
            New Shape with dimensions repeated factor times

        Note:
            Enables expressions like 2 * Shape(2, 3)
        """
        return self.__mul__(factor)

    @always_inline("nodebug")
    fn reverse(self) -> Self:
        """Return reversed shape.

        Returns:
            New Shape with dimensions in reverse order

        Example:
            Shape(2, 3, 4).reverse() -> Shape(4, 3, 2)
        """
        return Shape(self.dims.reversed())

    @always_inline("nodebug")
    fn replace(self, axis: Int, extent: Int) -> Shape:
        """Replace dimension at a specific axis.

        Args:
            axis: Index of the dimension to replace
            extent: New size for the dimension

        Returns:
            New Shape with the specified dimension replaced

        Raises:
            Panic if axis is out of bounds or extent is less than 1.
        """
        if axis < 0 or axis >= len(self):
            panic("Shape: invalid axis " + String(axis))
        if extent < 1:
            panic("Shape: invalid extent " + String(extent))
        var result = self.dims
        result[axis] = extent
        return Shape(result)

    @always_inline("nodebug")
    fn permute(self, axes: IntArray) -> Self:
        """Reorder dimensions according to specified axes.

        Args:
            axes: IntArray specifying the new order of dimensions
                  (e.g., [2, 0, 1] means dim0->dim2, dim1->dim0, dim2->dim1)

        Returns:
            New Shape with dimensions reordered according to axes
        """
        var result = IntArray.with_capacity(len(axes))
        result.reserve(len(axes))  # Guarantee no realloc
        for i in range(len(axes)):
            result.append(self[axes[i]])
        return Shape(result)

    @always_inline
    fn count_axes_of_size(self, size: Int) -> Int:
        """Count dimensions with a specific size.

        Args:
            size: The dimension size to count

        Returns:
            Number of dimensions that have the specified size
        """
        var count = 0
        for i in range(len(self)):
            if self[i] == size:
                count += 1
        return count

    @always_inline("nodebug")
    fn indices_of_axes_with_size(self, size: Int) -> IntArray:
        """Get indices of dimensions with a specific size.

        Args:
            size: The dimension size to find

        Returns:
            IntArray containing indices of all dimensions with the specified size

        Example:
            For Shape(2, 3, 2), indices_of_axes_with_size(2) returns [0, 2]
        """
        var result = IntArray.with_capacity(len(self))
        for i in range(len(self)):
            if self[i] == size:
                result.append(i)
        return result^

    @always_inline("nodebug")
    fn first_index(self) -> IntArray:
        """Get first index (all zeros) for iteration.

        Returns:
            IntArray filled with zeros, representing the first valid index
        """
        return IntArray.filled(len(self), 0)

    @always_inline("nodebug")
    fn compute_output_shape(
        self, normalized_axes: IntArray, keepdims: Bool, validated: Bool = False
    ) -> Shape:
        """Compute output shape after a reduction operation.

        Args:
            normalized_axes: Sorted IntArray of axes to reduce (empty = reduce all)
            keepdims: Whether to keep reduced dimensions as size 1
            validated: If True, skip validation of axes

        Returns:
            New Shape representing the result of the reduction

        Raises:
            Panic if any axis is out of bounds or axes are not sorted/unique
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
        """Get the shape with specific axes reduced (dimensions removed).

        Args:
            axes: IntArray of axis indices to reduce

        Returns:
            New Shape with the specified axes removed

        Raises:
            Panic if axes length exceeds shape rank.
        """
        if len(axes) > self.rank():
            panic("Shape -> reduced_shape: axes greater that shape rank")
        if len(axes) == 0:
            return self
        var reduced_axes = IntArray.with_capacity(len(axes))
        for i in range(len(axes)):
            reduced_axes.append(self[axes[i]])
        return Shape(reduced_axes)

    @staticmethod
    fn of(*dims: Int) -> Shape:
        """Create a shape from variadic integers (factory method).

        Args:
            *dims: Variable number of integers representing dimension sizes

        Returns:
            Shape instance with the specified dimensions

        Example:
            Shape.of(2, 3, 4) creates a shape with dimensions (2, 3, 4)
        """
        return Shape(dims)

    @always_inline
    fn product(self) -> Int:
        """Get the total number of elements (alias for num_elements()).

        Returns:
            Product of all dimensions
        """
        return self._numels


struct ShapeIndexIterator[origin: ImmutOrigin](
    RegisterPassable & ImplicitlyCopyable & Iterable & Iterator & Sized
):
    """Iterator over all valid indices of a tensor with a given shape.

    This iterator generates all possible index combinations for a tensor
    with the specified shape, useful for iteration over tensor elements.
    """

    comptime Element = IntArray
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var shape: Pointer[Shape, Self.origin]
    var current: IntArray
    var index: Int

    fn __init__(out self, shape: Pointer[Shape, Self.origin]):
        """Initialize the iterator.

        Args:
            shape: Pointer to the Shape to iterate over
        """
        self.shape = shape
        self.current = IntArray.filled(shape[].rank(), 0)
        self.index = 0

    fn __copyinit__(out self, copy: Self):
        """Create a copy of the iterator."""
        self.shape = copy.shape
        self.current = copy.current
        self.index = copy.index

    fn __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    fn __next__(mut self) raises StopIteration -> Self.Element:
        if not self.__has_next__():
            raise StopIteration()
        var result = self.current
        self.index += 1
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

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    @always_inline
    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len = len(self)
        return (iter_len, {iter_len})
