from common_utils import variadiclist_as_intlist, log_debug, panic
from intlist import IntList
from memory import Pointer


fn main() raises:
    pass


struct ShapeIndexIter[origin: ImmutableOrigin](Copyable):
    var shape: Pointer[Shape, origin]
    var current: IntList
    var index: Int

    fn __init__(out self, shape: Pointer[Shape, origin]):
        self.shape = shape
        self.current = IntList.with_capacity(shape[].ndim)
        self.index = 0
        for _ in range(shape[].ndim):
            self.current.append(0)

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> IntList:
        result = self.current.copy()
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
struct Shape(Sized & Stringable & Writable & Representable & Copyable):
    alias Unit = Shape.of(1)
    alias Void = Shape(IntList.Empty)
    var axes_spans: IntList
    var ndim: Int
    var numels: Int

    fn __init__(out self):
        self = Self.Void

    fn __init__(out self, dims: VariadicList[Int]):
        _dims = IntList.with_capacity(len(dims))
        for each in dims:
            _dims.append(each)
        self = Self(_dims)

    fn __init__(out self, dims: List[Int]):
        self = Self(IntList.new(dims))

    fn __init__(out self, dims: IntList):
        _ = """if len(dims) < 1:
            panic("Shape -> __init__: Shape dimension count should be > 0")"""
        _ndims = len(dims)
        # Allow scalar tensors (rank 0, i.e., Shape())
        if _ndims == 0:
            self.axes_spans = IntList.Empty
            self.ndim = 0
            self.numels = 1
            return
        for i in range(_ndims):
            if dims[i] < 1:
                panic(
                    "Shape → __init__: negative or zero sized dimension(s) are"
                    " not allowed →"
                    + " dimension = "
                    + String(dims[i])
                    + " @index = "
                    + String(i)
                )
        _numels = 1
        for idx in range(_ndims):
            _numels *= dims[idx]
        self.axes_spans = dims
        self.ndim = _ndims
        self.numels = _numels

    fn __iter__(ref self) -> ShapeIndexIter[__origin_of(self)]:
        return ShapeIndexIter(Pointer(to=self))

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
    fn broadcast_mask(self, target_shape: Shape) -> IntList:
        mask = IntList.with_capacity(target_shape.ndim)
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
                mask.append(1)  # self has no dimension here
            else:
                base_dim = self[i - offset]
                target_dim = target_shape[i]
                if base_dim == 1 and target_dim != 1:
                    mask.append(1)  # self is being expanded
                else:
                    mask.append(0)  # match or both 1 → not broadcasted

        return mask

    @staticmethod
    fn broadcast_strides(
        original_shape: Shape,
        target_shape: Shape,
        original_strides: IntList,  # use original strides - for transpose etc default is not enough
    ) -> IntList:
        mask = original_shape.broadcast_mask(target_shape)
        var strides = IntList.with_capacity(len(mask))
        var orig_index = 0
        for i in range(len(mask)):
            if mask[i] == 1:
                strides.append(0)  # broadcasted dim → zero stride
            else:
                strides.append(original_strides[orig_index])
                orig_index += 1
        return strides

    @always_inline
    fn count_axes_of_size(self, axis_size: Int) -> Int:
        return self.axes_spans.count(axis_size)

    @always_inline
    fn indices_of_axes_with_size(self, axis_size: Int) -> IntList:
        indices = IntList.with_capacity(len(self))
        for i in self.axes_spans:
            if self.axes_spans[i] == axis_size:
                indices.append(i)
        return indices

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

    fn __add__(self, other: List[Int]) -> Shape:
        dims = self.intlist() + IntList.new(other)
        return Shape(dims)

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

    fn drop_axis(self, axis: Int) -> Shape:
        if axis < 0 or axis >= self.ndim:
            panic(
                "Shape → drop_axis: Invalid axis: "
                + String(axis)
                + " for shape: "
                + self.__str__()
            )
        if self.ndim == 1:
            shape = self
            return shape
        axes = self.intlist()[:axis] + self.intlist()[axis + 1 :]
        return Shape(axes)

    @always_inline
    @staticmethod
    fn pad_shapes(shape1: Shape, shape2: Shape) -> (Shape, Shape):
        if shape1 == shape2:
            return shape1, shape2
        if shape1 == Shape.Void:
            return Shape.Unit * len(shape2), shape2
        if shape2 == Shape.Void:
            return shape1, Shape.Unit * len(shape1)

        max_len = max(len(shape1), len(shape2))

        # Pad with 1s
        padded1 = Shape.Unit * (max_len - len(shape1)) + shape1
        padded2 = Shape.Unit * (max_len - len(shape2)) + shape2

        return padded1, padded2

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
        # Explicitly handle true scalars (Shape.Void)
        if this == Shape.Void:
            return that  # Scalar + Tensor -> Tensor's shape
        if that == Shape.Void:
            return this  # Tensor + Scalar -> Tensor's shape
        shape1, shape2 = Self.pad_shapes(this, that)
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
        if not len(axes) == len(self):
            panic(
                "Shape → permute: axes length not equal to shape's dimension"
                " count"
            )
        result = IntList.with_capacity(len(axes))
        for axis in axes:
            result.append(self[axis])
        return Shape(result)

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
        self, indices: IntList, mask: IntList, broadcast_shape: Shape
    ) -> IntList:
        """Translate broadcasted indices to original tensor indices.

        Args:
            indices: Position in broadcasted tensor.
            mask: 1 for broadcasted dims, 0 for original.
            broadcast_shape: Shape after broadcasting.

        Returns:
            Indices in original tensor's space.
        """
        if not self.ndim <= broadcast_shape.ndim:
            panic("Shape → translate_index: original dims > broadcast dims")
        if not mask.size == broadcast_shape.ndim:
            panic("Shape → translate_index: mask/broadcast shape mismatch")
        if not indices.size == broadcast_shape.ndim:
            panic("Shape → translate_index: indices/broadcast shape mismatch")

        translated = IntList.with_capacity(self.ndim)
        offset = broadcast_shape.ndim - self.ndim

        for i in range(self.ndim):
            broadcast_axis = i + offset
            if not broadcast_axis < mask.size:
                panic("Shape → translate_index: invalid axis")

            if mask[broadcast_axis] == 1:
                translated.append(0)  # Broadcasted dim
            else:
                translated.append(indices[broadcast_axis])

        return translated

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

        _ = """fn __moveinit__(out self, deinit other: Self):
        self.axes_spans = other.axes_spans
        self.ndim = other.ndim
        self.numels = other.numels"""

    fn __copyinit__(out self, other: Self):
        self.axes_spans = other.axes_spans
        self.ndim = other.ndim
        self.numels = other.numels

    fn free(deinit self):
        log_debug("Freeing Shape")
        self.axes_spans.free()
        _ = self^

    @always_inline
    @staticmethod
    fn validate(shape: Shape):
        for idx in range(shape.ndim):
            if shape.axes_spans[idx] < 1:
                panic(
                    "Shape → validate: shape dimension not valid: "
                    + String(shape.axes_spans[idx])
                )

    @always_inline
    fn intlist(self) -> IntList:
        return self.axes_spans

    @always_inline
    fn product(shape: Shape) -> Int:
        return shape.intlist().product()

    @staticmethod
    fn of(*dims: Int) -> Shape:
        return Shape(dims)
