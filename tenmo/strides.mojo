from .intarray import IntArray
from .shapes import Shape
from .common_utils import panic
from .array import Array


struct Strides(
    Defaultable,
    ImplicitlyCopyable,
    RegisterPassable,
    Sized,
    Writable,
):
    """Strides for tensor indexing.

    Represents the stride values for each dimension of a tensor, indicating
    how many elements to skip in memory to move to the next element along each dimension.

    For a tensor with shape (d0, d1, ..., dn), the strides (s0, s1, ..., sn) define
    how many elements to traverse in memory to increment the index in each dimension.
    In C-contiguous layout, strides are calculated as (d1*d2*...*dn, d2*...*dn, ..., dn, 1).
    """

    var data: IntArray

    @staticmethod
    fn Zero() -> Strides:
        """Create an empty strides instance with zero dimensions."""
        return Strides()

    @staticmethod
    fn zeros(rank: Int) -> Strides:
        """Create strides filled with zeros for the given rank.

        Args:
            rank: Number of dimensions for the strides

        Returns:
            Strides instance with all elements set to 0
        """
        return Strides(IntArray.filled(rank, 0))

    fn __init__(out self):
        """Create an empty strides instance."""
        self.data = IntArray()

    @always_inline("nodebug")
    fn __init__(out self, values: IntArray):
        """Create strides from an IntArray.

        Args:
            values: IntArray containing stride values for each dimension
        """
        self.data = values

    @always_inline("nodebug")
    fn __init__(out self, values: List[Int]):
        """Create strides from a Mojo List of integers.

        Args:
            values: Mojo List of integers representing stride values for each dimension
        """
        self.data = IntArray(values)

    @always_inline("nodebug")
    fn __init__(out self, *values: Int):
        """Create strides from individual integer values.

        Args:
            *values: Variable number of integers representing stride values for each dimension
        """
        self.data = IntArray(values)

    @always_inline("nodebug")
    fn __copyinit__(out self, copy: Self):
        """Create a copy of another strides instance.

        Args:
             copy: Strides instance to copy from
        """
        self.data = copy.data

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the number of dimensions (rank) of the strides.

        Returns:
             Number of dimensions in the strides instance
        """
        return len(self.data)

    @always_inline("nodebug")
    fn array(self) -> Array:
        """Convert strides to an Array.

        Returns:
            Array containing the same elements as the strides
        """
        var result = Array()
        result.size = len(self)
        for i in range(len(self)):
            result[i] = self[i]
        return result^

    @always_inline("nodebug")
    fn __getitem__(self, i: Int) -> Int:
        """Get the stride value at the specified dimension index.

        Args:
            i: Dimension index

        Returns:
            Stride value for dimension i
        """
        return self.data[i]

    @always_inline("nodebug")
    fn __setitem__(mut self, i: Int, value: Int):
        """Set the stride value at the specified dimension index.

        Args:
            i: Dimension index
            value: New stride value for dimension i
        """
        self.data[i] = value

    @always_inline("nodebug")
    fn __getitem__(self, slice: Slice) -> Self:
        """Get a slice of strides.

        Args:
             slice: Slice specifying the range to extract

        Returns:
            New Strides instance containing the sliced elements
        """
        return Strides(self.data[slice])

    fn __eq__(self, other: Self) -> Bool:
        """Check if two strides instances are equal.

        Args:
             other: Strides instance to compare with

        Returns:
            True if strides are equal, False otherwise
        """
        return self.data == other.data

    fn __str__(self) -> String:
        """Get string representation of the strides.

        Returns:
            String representation in the format '(s0, s1, ..., sn)'
        """
        var s = self.data.__str__()
        return "(" + s[byte = 1 : len(s) - 1] + ")"

    fn __repr__(self) -> String:
        """Get official string representation of the strides.

        Returns:
            String representation suitable for debugging
        """
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        """Write strides to a writer.

        Args:
             writer: Writer to write to
        """
        writer.write(self.__str__())

    @always_inline
    fn tolist(self) -> List[Int]:
        """Convert strides to a Mojo List.

        Returns:
            Mojo List containing the stride values for each dimension
        """
        return self.data.tolist()

    @always_inline
    fn intarray(self) -> IntArray:
        """Get the underlying IntArray of the strides.

        Returns:
            IntArray containing the stride values
        """
        return self.data

    @always_inline
    fn permute(self, axes: IntArray) -> Self:
        """Reorder dimensions according to the specified permutation.

        Args:
            axes: IntArray specifying the new order of dimensions
                  (e.g., [2,0,1] means dim0->dim2, dim1->dim0, dim2->dim1)

        Returns:
             New Strides instance with dimensions reordered according to axes

        Example:
            ```mojo
            var s = Strides([12, 4, 1])
            var permuted = s.permute([2, 0, 1])  # Returns (1, 12, 4)
            ```
        """
        var result = IntArray.with_capacity(len(axes))
        for i in range(len(axes)):
            result.append(self[axes[i]])
        return Strides(result)

    @staticmethod
    @always_inline
    fn default(shape: Shape) -> Self:
        """Compute default C-contiguous strides for the given shape.

        Args:
            shape: Shape representing tensor dimensions

        Returns:
             Strides instance representing C-contiguous memory layout

        Example:
            For shape (2, 3, 4), returns strides (12, 4, 1)
            where 12 = 3*4, 4 = 4, 1 = 1
        """
        var rank = shape.rank()
        var strides = IntArray.with_capacity(rank)
        var acc = 1
        for i in range(rank - 1, -1, -1):
            strides.prepend(acc)
            acc *= shape[i]
        return Strides(strides)

    @always_inline
    fn is_contiguous(self, shape: Shape) -> Bool:
        """Check if the strides represent a contiguous memory layout for the given shape.

        Args:
            shape: Shape representing tensor dimensions

        Returns:
            True if strides represent contiguous layout, False otherwise

        Note:
            A layout is contiguous if elements are stored in memory without gaps,
            following the C-contiguous (row-major) ordering convention.

        Example:
            ```mojo
            var shape = Shape([2, 3, 4])
            var cont = Strides([12, 4, 1])    # True - C-contiguous
            var noncont = Strides([1, 12, 4]) # False - transposed layout
            ```
        """
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
        """Create an empty strides instance with pre-allocated capacity.

        Args:
            capacity: Pre-allocated capacity for the strides data

        Returns:
            Empty Strides instance with specified capacity

        Note:
            This avoids frequent reallocations when you know approximately
            how many dimensions you'll need.
        """
        return Strides(IntArray.with_capacity(capacity))
