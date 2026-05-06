from .shapes import Shape
from .strides import Strides
from .intarray import IntArray
from .common_utils import panic


@fieldwise_init
struct IndexIterator[shape_origin: ImmutOrigin, strides_origin: ImmutOrigin](
    ImplicitlyCopyable, Iterable, Iterator, RegisterPassable, Sized
):
    """Iterator over memory offsets for a tensor with given shape and strides.

    This iterator generates the physical memory offsets for each logical element
    in a multi-dimensional tensor. It handles both contiguous and strided tensors
    efficiently, using an odometer-like increment strategy for non-contiguous data.
    """
    comptime Element = Int
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var shape: Pointer[Shape, Self.shape_origin]
    var strides: Pointer[Strides, Self.strides_origin]
    var start_offset: Int
    var current_offset: Int
    var current_index: Int
    var total_elements: Int
    var rank: Int
    var coords: IntArray
    var contiguous: Bool

    @always_inline("nodebug")
    fn __init__(
        out self,
        shape: Pointer[Shape, Self.shape_origin],
        strides: Pointer[Strides, Self.strides_origin],
        start_offset: Int = 0,
    ):
        """Initialize the index iterator.

        Args:
            shape: Pointer to the Shape of the tensor
            strides: Pointer to the Strides of the tensor
            start_offset: Initial memory offset (default: 0)
        """
        self.shape = shape
        self.strides = strides
        self.start_offset = start_offset
        self.current_offset = start_offset
        self.current_index = 0
        self.total_elements = shape[].num_elements()
        self.rank = shape[].rank()
        self.coords = IntArray.filled(self.rank, 0)
        self.contiguous = self.strides[].is_contiguous(self.shape[])

    fn __iter__(
        ref self,
    ) -> Self.IteratorType[
        origin_of(self, Self.shape_origin, Self.strides_origin)
    ]:
        """Get the iterator itself.

        Returns:
            Self reference as iterator.
        """
        return self

    fn __next__(mut self) raises StopIteration -> Self.Element:
        """Return next memory offset and advance iterator.

        Incrementally updates coordinates like an odometer.
        Uses fast path for contiguous tensors and odometer-style
        increment for strided tensors.

        Returns:
            Physical memory offset for current logical element.

        Raises:
            StopIteration: When all elements have been iterated.
        """
        if not self.__has_next__():
            raise StopIteration()

        var result = self.current_offset

        # Fast path: contiguous tensor
        if self.contiguous:
            self.current_offset += 1
            self.current_index += 1
            return result

        # Strided path: increment coordinates like an odometer
        # Start from rightmost (fastest-changing) dimension
        self.current_index += 1
        ref shape = self.shape[]
        ref strides = self.strides[]

        for i in range(self.rank - 1, -1, -1):
            self.coords[i] += 1

            # Check if we need to carry to next dimension
            if self.coords[i] < shape[i]:
                # No carry needed - just update offset and done!
                self.current_offset += strides[i]
                break
            else:
                # Carry: reset this dimension and continue to next
                self.coords[i] = 0
                self.current_offset -= (shape[i] - 1) * strides[i]

        return result

    @always_inline("nodebug")
    fn __has_next__(self) -> Bool:
        """Check if there are more elements to iterate.

        Returns:
            True if there are more elements, False otherwise.
        """
        return self.current_index < self.total_elements

    fn bounds(self) -> Tuple[Int, Optional[Int]]:
        """Get the bounds of the iterator.

        Returns:
            Tuple of (remaining length, Optional of the same length).
        """
        var iter_len: Int = len(self)
        return (iter_len, {iter_len})

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Get the number of remaining elements.

        Returns:
            Number of elements yet to be iterated.
        """
        return self.total_elements - self.current_index

    @always_inline("nodebug")
    fn skip(mut self, n: Int, small_skip: Int = 100):
        """Skip n elements forward efficiently.

        Uses hybrid strategy:
        - Small skips (n < small_skip): Incremental updates.
        - Large skips (n >= small_skip): Direct computation.

        Args:
            n: Number of elements to skip (must be >= 0).
            small_skip: Threshold for deciding whether to use incremental or direct calculation.
        """
        if n <= 0:
            return

        var target_index = min(self.current_index + n, self.total_elements)
        var skip_distance = target_index - self.current_index

        # Fast path: contiguous
        if self.contiguous:
            self.current_offset += skip_distance
            self.current_index = target_index
            return

        if skip_distance < small_skip:
            # Inline the strided increment logic
            ref shape = self.shape[]
            ref strides = self.strides[]
            for _ in range(skip_distance):
                self.current_index += 1
                for i in range(self.rank - 1, -1, -1):
                    self.coords[i] += 1
                    if self.coords[i] < shape[i]:
                        self.current_offset += strides[i]
                        break
                    else:
                        self.coords[i] = 0
                        self.current_offset -= (shape[i] - 1) * strides[i]

        else:
            # Large skip: compute directly (faster for large n)
            if target_index == self.total_elements:
                self.current_index = self.total_elements
                return

            self.current_index = target_index

            # Convert linear index to coordinates using running divisor
            var remaining = target_index
            var divisor = self.total_elements
            ref shape = self.shape[]
            ref strides = self.strides[]
            self.current_offset = self.start_offset

            for i in range(self.rank):
                divisor //= shape[i]
                var coord = remaining // divisor
                self.coords[i] = coord
                self.current_offset += coord * strides[i]
                remaining %= divisor

    @always_inline("nodebug")
    fn reset(mut self):
        """Reset iterator to the beginning.

        Resets the current offset, index, and coordinates to their initial values.
        """
        self.current_offset = self.start_offset
        self.current_index = 0
        for i in range(self.rank):
            self.coords[i] = 0


    @always_inline("nodebug")
    fn peek(self) -> Int:
        """Get current offset without advancing.

        Returns:
            The current memory offset without modifying iterator state.
        """
        return self.current_offset


@fieldwise_init
struct IndexCalculator(ImplicitlyCopyable, RegisterPassable):
    """Utility for calculating flat indices and coordinates in multi-dimensional arrays.

    Provides static methods for converting between flat (linear) indices and
    multi-dimensional coordinates, which is essential for tensor operations.
    """

    @always_inline
    @staticmethod
    fn flatten_index(
        shape: Shape, indices: IntArray, strides: Strides, offset: Int = 0
    ) -> Int:
        """Calculate flat (linear) index from multi-dimensional indices.

        Args:
            shape: Shape of the tensor
            indices: IntArray of indices, one per dimension
            strides: Strides of the tensor
            offset: Base offset to add (default: 0)

        Returns:
            The flat memory offset corresponding to the given indices

        Raises:
            Panic if indices or strides have wrong rank, or if any index is out of bounds.
        """
        # 1. Rank check
        var rank = shape.rank()
        if len(indices) != rank or len(strides) != rank:
            if len(indices) != rank and len(strides) != rank:
                panic(
                    "flatten_index: indices(",
                    String(len(indices)),
                    ") and strides(",
                    String(len(strides)),
                    ") both ≠ required rank(",
                    String(rank),
                    ")",
                )
            elif len(indices) != rank:
                panic(
                    "flatten_index: indices(",
                    String(len(indices)),
                    ") ≠ required rank(",
                    String(rank),
                    ")",
                )
            else:
                panic(
                    "flatten_index: strides(",
                    String(len(strides)),
                    ") ≠ required rank(",
                    String(rank),
                    ")",
                )

        var flat = offset  # absolute base offset
        # 2. Normalize negative indices, bounds-check, and accumulate
        for dim_idx in range(len(indices)):
            var idx = indices[dim_idx]
            dim_size = shape[dim_idx]
            idx = idx + dim_size if idx < 0 else idx
            if idx < 0 or idx >= dim_size:
                panic(
                    "flatten_index: index out of bounds: axis",
                    String(dim_idx),
                    ", got",
                    String(indices[dim_idx]),
                    ", size",
                    String(dim_size),
                )
            flat = flat + idx * strides[dim_idx]
        return flat

    @always_inline
    @staticmethod
    fn index_to_coord(shape: Shape, flat_index: Int) -> IntArray:
        """Convert flat (linear) index to multi-dimensional coordinates.

        Args:
            shape: Shape of the tensor
            flat_index: Linear index to convert

        Returns:
            IntArray of coordinates corresponding to the flat index

        Raises:
            Panic if flat_index is out of bounds.
        """
        if flat_index < 0 or flat_index >= shape.num_elements():
            panic(
                "IndexCalculator → index_to_coord: flat_index",
                String(flat_index),
                "out of bounds.",
                "Should be between 0 <= and <",
                String(shape.num_elements()),
            )
        var rank = shape.rank()
        var indices = IntArray.filled(rank, 0)
        var remaining = flat_index
        for i in range(rank - 1, -1, -1):  # from last axis backward
            var dim = shape[i]
            indices[i] = remaining % dim
            remaining //= dim
        return indices^

    @staticmethod
    @always_inline
    fn max_index_good(shape: Shape, strides: Strides, offset: Int) -> Int:
        """Calculate the maximum valid flat index for the given shape and strides.

        Args:
            shape: Shape of the tensor
            strides: Strides of the tensor
            offset: Base offset to add

        Returns:
            The maximum flat index that can be accessed in the tensor.
        """
        var max_idx = offset
        for i in range(shape.rank()):
            if strides[i] > 0:
                max_idx += (shape[i] - 1) * strides[i]
        return max_idx

    @staticmethod
    fn min_index(shape: Shape, strides: Strides, offset: Int) -> Int:
        """Compute the minimum buffer index touched by this view."""
        var min_idx = offset
        for i in range(shape.rank()):
            var dim    = shape[i]
            var stride = strides[i]
            if dim <= 1:
                continue
            if stride < 0:
                min_idx += (dim - 1) * stride   # negative, so min_idx decreases
            # stride > 0 contributes nothing — offset is already the minimum
        return min_idx

    @staticmethod
    fn max_index(shape: Shape, strides: Strides, offset: Int) -> Int:
        """Compute the maximum buffer index touched by this view."""
        var max_idx = offset
        for i in range(shape.rank()):
            var dim    = shape[i]
            var stride = strides[i]
            if dim <= 1:
                continue
            if stride > 0:
                max_idx += (dim - 1) * stride
            # stride < 0 contributes nothing — offset is already the maximum
        return max_idx

