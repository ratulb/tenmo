from shapes import Shape
from strides import Strides
from intarray import IntArray
from common_utils import panic


@fieldwise_init
@register_passable
struct IndexIterator[shape_origin: ImmutOrigin, strides_origin: ImmutOrigin](
    ImplicitlyCopyable
):
    var shape: Pointer[Shape, shape_origin]
    var strides: Pointer[Strides, strides_origin]
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
        self.shape = shape
        self.strides = strides
        self.start_offset = start_offset
        self.current_offset = start_offset
        self.current_index = 0
        self.total_elements = shape[].num_elements()
        self.rank = shape[].rank()
        self.coords = IntArray.filled(self.rank, 0)
        self.contiguous = self.strides[].is_contiguous(self.shape[])

    @always_inline("nodebug")
    fn __iter__(self) -> Self:
        return self

    @always_inline("nodebug")
    fn __next__(mut self) -> Int:
        """
        Return next memory offset and advance iterator.

        Incrementally update coordinates like an odometer.

        Returns:
            Physical memory offset for current logical element.
        """
        var result = self.current_offset

        # Fast path: contiguous tensor (ultra-fast)
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
        return self.current_index < self.total_elements

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        return self.total_elements - self.current_index

    @always_inline("nodebug")
    fn skip(mut self, n: Int, small_skip: Int = 100):
        """
        Skip n elements forward.

        Uses hybrid strategy:
        - Small skips (n < 100): Incremental updates.
        - Large skips (n >= 100): Direct computation.

        Args:
            n: Number of elements to skip (must be >= 0).
            small_skip: Threshold for deciding whether to call __next__ or calculate.
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
        """Reset iterator to beginning."""
        self.current_offset = self.start_offset
        self.current_index = 0
        for i in range(self.rank):
            self.coords[i] = 0

    @always_inline("nodebug")
    fn peek(self) -> Int:
        """Get current offset without advancing."""
        return self.current_offset


@register_passable
struct IndexCalculator:
    """Utility for calculating flat indices and coordinates in multi-dimensional arrays.
    """

    @always_inline
    @staticmethod
    fn flatten_index(
        shape: Shape, indices: IntArray, strides: Strides, offset: Int = 0
    ) -> Int:
        """Calculate flat index from IntArray indices."""
        # 1. Rank check
        var rank = shape.rank()
        if indices.size() != rank or len(strides) != rank:
            if indices.size() != rank and len(strides) != rank:
                panic(
                    "flatten_index: indices(",
                    indices.size().__str__(),
                    ") and strides(",
                    len(strides).__str__(),
                    ") both ≠ required rank(",
                    rank.__str__(),
                    ")",
                )
            elif indices.size() != rank:
                panic(
                    "flatten_index: indices(",
                    indices.size().__str__(),
                    ") ≠ required rank(",
                    rank.__str__(),
                    ")",
                )
            else:
                panic(
                    "flatten_index: strides(",
                    len(strides).__str__(),
                    ") ≠ required rank(",
                    rank.__str__(),
                    ")",
                )

        var flat = offset  # absolute base offset

        # print("var flat: ", flat, indices.size(), shape)
        # IntArrayHelper.print(indices)
        # 2. Normalize negative indices, bounds-check, and accumulate
        for dim_idx in range(indices.size()):
            var idx = indices[dim_idx]
            dim_size = shape[dim_idx]
            # allow negative indexing like Python/NumPy: -1 => last element
            idx = idx + dim_size if idx < 0 else idx

            # now validate

            if idx < 0 or idx >= dim_size:
                panic(
                    "flatten_index: index out of bounds: axis",
                    dim_idx.__str__(),
                    ", got",
                    indices[dim_idx].__str__(),
                    ", size",
                    dim_size.__str__(),
                )

            flat = flat + idx * strides[dim_idx]

        return flat

    @always_inline
    @staticmethod
    fn flatten_index(
        shape: Shape, indices: List[Int], strides: Strides, offset: Int = 0
    ) -> Int:
        """Calculate flat index from List[Int] indices."""
        # 1. Rank check
        var rank = shape.rank()
        if len(indices) != rank or len(strides) != rank:
            if len(indices) != rank and len(strides) != rank:
                panic(
                    "flatten_index: indices(",
                    len(indices).__str__(),
                    ") and strides(",
                    len(strides).__str__(),
                    ") both ≠ required rank(",
                    rank.__str__(),
                    ")",
                )
            elif len(indices) != rank:
                panic(
                    "flatten_index: indices(",
                    len(indices).__str__(),
                    ") ≠ required rank(",
                    rank.__str__(),
                    ")",
                )
            else:
                panic(
                    "flatten_index: strides(",
                    len(strides).__str__(),
                    ") ≠ required rank(",
                    rank.__str__(),
                    ")",
                )

        var flat = offset  # absolute base offset

        # 2. Normalize negative indices, bounds-check, and accumulate
        for dim_idx in range(len(indices)):
            var idx = indices[dim_idx]
            dim_size = shape[dim_idx]

            # allow negative indexing like Python/NumPy: -1 => last element
            idx = idx + dim_size if idx < 0 else idx

            # now validate
            if idx < 0 or idx >= dim_size:
                panic(
                    "flatten_index: index out of bounds: axis",
                    dim_idx.__str__(),
                    ", got",
                    indices[dim_idx].__str__(),
                    ", size",
                    dim_size.__str__(),
                )

            flat = flat + idx * strides[dim_idx]

        return flat

    @always_inline
    @staticmethod
    fn flatten_index(
        shape: Shape,
        indices: VariadicList[Int],
        strides: Strides,
        offset: Int = 0,
    ) -> Int:
        """Calculate flat index from List[Int] indices."""
        # 1. Rank check
        var rank = shape.rank()
        if len(indices) != rank or len(strides) != rank:
            if len(indices) != rank and len(strides) != rank:
                panic(
                    "flatten_index: indices(",
                    len(indices).__str__(),
                    ") and strides(",
                    len(strides).__str__(),
                    ") both ≠ required rank(",
                    rank.__str__(),
                    ")",
                )
            elif len(indices) != rank:
                panic(
                    "flatten_index: indices(",
                    len(indices).__str__(),
                    ") ≠ required rank(",
                    rank.__str__(),
                    ")",
                )
            else:
                panic(
                    "flatten_index: strides(",
                    len(strides).__str__(),
                    ") ≠ required rank(",
                    rank.__str__(),
                    ")",
                )

        var flat = offset  # absolute base offset

        # 2. Normalize negative indices, bounds-check, and accumulate
        for dim_idx in range(len(indices)):
            var idx = indices[dim_idx]
            dim_size = shape[dim_idx]

            # allow negative indexing like Python/NumPy: -1 => last element
            idx = idx + dim_size if idx < 0 else idx

            # now validate
            if idx < 0 or idx >= dim_size:
                panic(
                    "flatten_index: index out of bounds: axis",
                    dim_idx.__str__(),
                    ", got",
                    indices[dim_idx].__str__(),
                    ", size",
                    dim_size.__str__(),
                )

            flat = flat + idx * strides[dim_idx]

        return flat

    @always_inline
    @staticmethod
    fn index_to_coord(shape: Shape, flat_index: Int) -> IntArray:
        """Convert flat index to multi-dimensional coordinates."""
        if flat_index < 0 or flat_index >= shape.num_elements():
            panic(
                "IndexCalculator → index_to_coord: flat_index",
                flat_index.__str__(),
                "out of bounds.",
                "Should be between 0 <= and <",
                shape.num_elements().__str__(),
            )
        var rank = shape.rank()
        var indices = IntArray.filled(rank, 0)
        var remaining = flat_index
        for i in range(rank - 1, -1, -1):  # from last axis backward
            var dim = shape[i]
            indices[i] = remaining % dim
            remaining //= dim

        return indices^
