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


struct ShapeIndexIter1[origin: ImmutableOrigin](Copyable & Movable):
    var shape: Pointer[Shape, origin]
    var current: IntList
    var index: Int

    fn __init__(
        out self, shape: Pointer[Shape, origin], curr_index: IntList, pos: Int
    ):
        self.shape = shape
        self.current = curr_index.copy()
        self.index = pos

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.current = other.current.copy()
        self.index = other.index

    fn __moveinit__(out self, deinit other: Self):
        self.shape = other.shape
        self.current = other.current^
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

        _="""fn __moveinit__(out self, deinit other: Self):
        self.axes_spans = other.axes_spans^
        self.ndim = other.ndim
        self.numels = other.numels"""

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

    @always_inline
    fn broadcastable(self, to: Shape) -> Bool:
        dims1 = self.intlist()
        dims2 = to.intlist()
        zip_reversed = dims1.zip_reversed(dims2)
        for dims in zip_reversed:
            if dims[0] != dims[1]:
                if dims[0] != 1 and dims[1] != 1:
                    return False
        return True

    @always_inline
    fn broadcast_mask(self, target_shape: Shape) -> IntArray:
        mask = IntArray(size=target_shape.ndim)
        offset = target_shape.ndim - self.ndim
        if offset < 0:
            panic(
                "Shape → broadcast_mask → target_shape.ndim is smaller than"
                " self.ndim: "
                + String(target_shape.ndim)
                + ", "
                + String(self.ndim)
            )

        for i in range(target_shape.ndim):
            if i < offset:
                mask[i] = 1  # self has no dimension here
            else:
                base_dim = self[i - offset]
                target_dim = target_shape[i]
                if base_dim == 1 and target_dim != 1:
                    mask[i] = 1  # self is being expanded
                else:
                    mask[i] = 0  # match or both 1 → not broadcasted

        return mask^

    @always_inline
    fn flatten_index(
        self, indices: IntList, strides: Strides, offset: Int = 0
    ) -> Int:
        # 1. Rank check
        if indices.size() != self.rank():
            panic(
                (
                    "Shape → flatten_index(IntList): number of indices does not"
                    " match"
                ),
                " shape rank",
                ": len indices →",
                indices.size().__str__(),
                "rank →",
                self.rank().__str__(),
            )

        var flat = offset  # absolute base offset

        # 2. Normalize negative indices, bounds-check, and accumulate
        for dim_idx in range(indices.size()):
            var idx = indices[dim_idx]
            dim_size = self[dim_idx]

            # allow negative indexing like Python/NumPy: -1 => last element
            idx = idx + dim_size if idx < 0 else idx

            # now validate
            if idx < 0 or idx >= dim_size:
                panic(
                    "Shape → flatten_index(IntList): index out of bounds: axis",
                    dim_idx.__str__(),
                    ", got",
                    indices[dim_idx].__str__(),
                    ", size",
                    dim_size.__str__(),
                )

            flat = flat + idx * strides[dim_idx]

        return flat

    @always_inline
    fn flatten_index(
        self, indices: IntArray, strides: Strides, offset: Int = 0
    ) -> Int:
        # 1. Rank check
        if indices.size() != self.rank():
            panic(
                (
                    "Shape → flatten_index[IntArray]: number of indices does"
                    " not match"
                ),
                " shape rank",
                ": len indices →",
                indices.size().__str__(),
                "rank →",
                self.rank().__str__(),
            )

        var flat = offset  # absolute base offset

        # 2. Normalize negative indices, bounds-check, and accumulate
        for dim_idx in range(indices.size()):
            var idx = indices[dim_idx]
            dim_size = self[dim_idx]

            # allow negative indexing like Python/NumPy: -1 => last element
            idx = idx + dim_size if idx < 0 else idx

            # now validate
            if idx < 0 or idx >= dim_size:
                panic(
                    (
                        "Shape → flatten_index[IntArray]: index out of bounds:"
                        " axis"
                    ),
                    dim_idx.__str__(),
                    ", got",
                    indices[dim_idx].__str__(),
                    ", size",
                    dim_size.__str__(),
                )

            flat = flat + idx * strides[dim_idx]

        return flat

    @always_inline
    fn flatten_index(
        self, indices: List[Int], strides: Strides, offset: Int = 0
    ) -> Int:
        # 1. Rank check
        if len(indices) != self.rank():
            panic(
                (
                    "Shape → flatten_index(List[Int]): number of indices does"
                    " not match"
                ),
                " shape rank",
                ": indices →",
                indices.__str__(),
                "rank →",
                self.rank().__str__(),
            )

        var flat = offset  # absolute base offset

        # 2. Normalize negative indices, bounds-check, and accumulate
        for dim_idx in range(len(indices)):
            var idx = indices[dim_idx]
            dim_size = self[dim_idx]

            # allow negative indexing like Python/NumPy: -1 => last element
            idx = idx + dim_size if idx < 0 else idx

            # now validate
            if idx < 0 or idx >= dim_size:
                panic(
                    (
                        "Shape → flatten_index(List[Int]): index out of bounds:"
                        " axis"
                    ),
                    dim_idx.__str__(),
                    ", got",
                    indices[dim_idx].__str__(),
                    ", size",
                    dim_size.__str__(),
                )

            flat = flat + idx * strides[dim_idx]

        return flat

    @always_inline
    fn index_to_coord(self, flat_index: Int) -> IntList:
        # fn unravel_index(self, flat_index: Int) -> IntList:
        if flat_index < 0 or flat_index >= self.num_elements():
            panic(
                "Shape → unravel_index: flat_index",
                flat_index.__str__(),
                "out of bounds.",
                "Should be between 0 <= and <",
                self.num_elements().__str__(),
            )
        rank = self.rank()
        indices = IntList.filled(rank, 0)
        remaining = flat_index
        for i in range(rank - 1, -1, -1):  # from last axis backward
            dim = self[i]
            indices[i] = remaining % dim
            remaining //= dim

        return indices^

    @always_inline
    fn count_axes_of_size(self, axis_size: Int) -> Int:
        return self.axes_spans.count(axis_size)

    @always_inline
    fn indices_of_axes_with_size(self, axis_size: Int) -> IntList:
        indices = IntList.with_capacity(len(self))
        for i in self.axes_spans:
            if self.axes_spans[i] == axis_size:
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
        self, normalized_axes: IntList, keepdims: Bool
    ) -> Shape:
        """Compute the output shape after reduction along specified axes.

        Args:
            normalized_axes: Sorted list of axes to reduce over.
            keepdims: Whether to keep reduced dimensions as size 1.

        Returns:
            Shape after reduction

        Behavior:
            - If reducing all axes and keepdims=False → returns Shape() (scalar)
            - Otherwise:
                - For reduced axes: keep as 1 if keepdims=True, else remove
                - For non-reduced axes: keep original size.
        """
        rank = self.rank()

        # Full reduction case (return scalar shape if not keeping dims)
        if rank == 0 or (len(normalized_axes) == rank and not keepdims):
            return Shape()

        var spans = IntList.with_capacity(rank)
        for dim in range(rank):
            if dim in normalized_axes:
                if keepdims:
                    spans.append(1)  # Keep reduced dim as size 1
            else:
                spans.append(self[dim])  # Keep original size

        return Shape(spans)

    @always_inline
    @staticmethod
    fn validate_matrix_shapes_nd(A_shape: Shape, B_shape: Shape):
        if len(A_shape) < 2 or len(B_shape) < 2:
            panic(
                "Tensor → validate_matrix_shapes_nd: matmul_nd expects rank >="
                " 2. Got A = "
                + A_shape.__str__()
                + ", B = "
                + B_shape.__str__()
            )

        if A_shape[-1] != B_shape[-2]:
            panic(
                "Tensor → validate_matrix_shapes_nd: inner dimensions"
                " mismatch: "
                + "A(...,"
                + A_shape[-1].__str__()
                + ") vs "
                + "B("
                + B_shape[-2].__str__()
                + ",...). "
                + "Full A = "
                + A_shape.__str__()
                + ", B = "
                + B_shape.__str__()
            )

        A_batch = A_shape[0:-2]
        B_batch = B_shape[0:-2]
        _ = Shape.broadcast_shape(
            A_batch, B_batch
        )  # will panic internally if not compatible

    @always_inline
    @staticmethod
    fn validate_matrix_shapes_2d(A_shape: Shape, B_shape: Shape):
        if len(A_shape) != 2 or len(B_shape) != 2:
            panic(
                "Tensor → validate_matrix_shapes_2d: matmul_2d expects rank =="
                " 2. Got A = "
                + A_shape.__str__()
                + ", B = "
                + B_shape.__str__()
            )

        if A_shape[1] != B_shape[0]:
            panic(
                "Tensor → validate_matrix_shapes_2d: inner dimensions mismatch"
                " in matmul_2d: "
                + "A(m,"
                + A_shape[1].__str__()
                + ") vs "
                + "B("
                + B_shape[0].__str__()
                + ",n). "
                + "Full A = "
                + A_shape.__str__()
                + ", B = "
                + B_shape.__str__()
            )

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
    @staticmethod
    fn pad_shapes(shape1: Shape, shape2: Shape) -> Tuple[Shape, Shape]:
        if shape1 == shape2:
            return shape1.copy(), shape2.copy()
        if shape1 == Shape():
            return Shape(1) * len(shape2), shape2.copy()
        if shape2 == Shape():
            return shape1.copy(), Shape(1) * len(shape1)

        max_len = max(len(shape1), len(shape2))

        # Pad with 1s
        padded1 = Shape(1) * (max_len - len(shape1)) + shape1
        padded2 = Shape(1) * (max_len - len(shape2)) + shape2

        return padded1^, padded2^

    @always_inline
    @staticmethod
    fn broadcast_shape(this: Shape, that: Shape) -> Shape:
        if not this.broadcastable(that):
            panic(
                "Shape → broadcast_shape - not broadcastable: "
                + this.__str__()
                + " <=> "
                + that.__str__()
            )
        # Explicitly handle true scalars (Shape())
        if this == Shape():
            return that.copy()  # Scalar + Tensor -> Tensor's shape
        if that == Shape():
            return this.copy()  # Tensor + Scalar -> Tensor's shape
        padded = Self.pad_shapes(this, that)
        shape1 = padded[0].copy()
        shape2 = padded[1].copy()
        result_shape = IntList.with_capacity(len(shape1))
        s1 = shape1.intlist()
        s2 = shape2.intlist()

        for dims in s1.zip(s2):
            if dims[0] == dims[1]:
                result_shape.append(dims[0])
            elif dims[0] == 1:
                result_shape.append(dims[1])
            elif dims[1] == 1:
                result_shape.append(dims[0])
            else:
                panic(
                    "Shape → broadcast_shape - cannot broadcast shapes: "
                    + this.__str__()
                    + ", "
                    + that.__str__()
                )

        return Shape(result_shape)

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

    @always_inline
    fn translate_index(
        self, indices: IntArray, mask: IntArray, broadcast_shape: Shape
    ) -> IntArray:
        """Translate broadcasted indices to original tensor indices.

        Args:
            indices: Position in broadcasted tensor.
            mask: 1 for broadcasted dims, 0 for original.
            broadcast_shape: Shape after broadcasting.

        Returns:
            Indices in original tensor's space.
        """
        # Input Validation
        if self.ndim > broadcast_shape.ndim:
            panic(
                "Shape translate_index: original dims greater than broadcast"
                " dims"
            )
        if mask.size() != broadcast_shape.ndim:
            panic(
                "Shape translate_index: mask size does not match broadcast ndim"
            )
        if indices.size() != broadcast_shape.ndim:
            panic(
                "Shape translate_index: indices size does not match broadcast"
                " ndim"
            )

        translated = IntArray(size=self.ndim)
        offset = broadcast_shape.ndim - self.ndim

        # Perform the translation
        for i in range(self.ndim):
            broadcast_axis = i + offset

            if mask[broadcast_axis] == 1:
                translated[i] = 0  # Broadcasted dim
            else:
                original_index = indices[broadcast_axis]
                # CRITICAL: Check if the index is valid for the original shape
                if original_index >= self[i]:
                    panic(
                        "Shape translate_index: index out of bounds for"
                        " original tensor"
                    )
                translated[i] = original_index

        return translated^

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
    @staticmethod
    fn validate1(shape: Shape):
        for idx in range(shape.ndim):
            if shape.axes_spans[idx] < 1:
                panic(
                    "Shape → validate: shape dimension not valid: "
                    + String(shape.axes_spans[idx])
                )

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


fn main() raises:
    _ = """s = Shape(5, 2, 1)
    for coord in s:
        # print(coord, end="\n")
        IntArrayHelper.print(coord)
        extended = IntArrayHelper.extend(coord, 2, 3)
        print()
        IntArrayHelper.print(extended)"""
    s = Shape(6)
    print(len(s))
    pass
