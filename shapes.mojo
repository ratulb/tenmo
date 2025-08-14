from common_utils import variadiclist_as_intlist, log_debug
from intlist import IntList
from os import abort
from memory import Pointer


fn main() raises:
    test_negative_indices()
    test_slice_shape()


from testing import assert_true


fn test_slice_shape() raises:
    shape = Shape([1, 2, 3, 4])
    assert_true(
        shape[:-1] == Shape.of(1, 2, 3)
        and shape[:-2] == Shape.of(1, 2)
        and shape[:-3] == Shape(1)
        and shape[2::4] == Shape.of(3)
        and shape[-1:] == Shape.of(4)
        and shape[-2:] == Shape.of(3, 4),
        "Shape slice assertion failed",
    )


fn test_negative_indices() raises:
    shape = Shape([1, 2, 3])
    assert_true(
        shape[-1] == 3 and shape[-2] == 2 and shape[-3] == 1,
        "Shape negative indices assertion failed",
    )


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


struct Shape(
    Sized & Stringable & Writable & Representable & Copyable & Movable
):
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
            abort("Shape -> __init__: Shape dimension count should be > 0")"""
        _ndims = len(dims)
        # Allow scalar tensors (rank 0, i.e., Shape())
        if _ndims == 0:
            self.axes_spans = IntList.Empty
            self.ndim = 0
            self.numels = 1
            return
        for i in range(_ndims):
            if dims[i] < 1:
                abort(
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

    fn slice_from(self, axis: Int) -> Shape:
        if axis < 0 or axis > self.rank():
            abort(
                "Shape -> slice_from: axis "
                + String(axis)
                + " out of bounds for ndim "
                + String(self.ndim)
            )

        new_axes_spans = self.axes_spans[axis:]
        return Shape(new_axes_spans)

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
            abort(
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

    fn reverse(self) -> Self:
        dims = self.intlist()
        dims.reverse()
        return Shape(dims)

    fn replace(self, axis: Int, extent: Int) -> Shape:
        if axis < 0 or axis >= self.ndim:
            abort(
                "Shape → replace: Invalid axis: "
                + String(axis)
                + " for shape: "
                + self.__str__()
            )
        if extent < 1:
            abort("Shape → replace: Invalid extent: " + String(extent))
        axes = self.intlist()
        axes[axis] = extent
        return Shape(axes)

    fn drop_axis(self, axis: Int) -> Shape:
        if axis < 0 or axis >= self.ndim:
            abort(
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
        # Handle scalar cases first
        if shape1 == Shape.Void and shape2 == Shape.Void:
            return Shape.Void, Shape.Void
        one = IntList(1)
        if shape1 == Shape.Void:
            # return Shape(1) * len(shape2), shape2  # Scalar becomes [1,1...] of matching rank
            return (
                Shape(one * len(shape2)),
                shape2,
            )  # Scalar becomes [1,1...] of matching rank
        if shape2 == Shape.Void:
            # return shape1, Shape(1) * len(shape1)  # Same for other side
            return shape1, Shape(one * len(shape1))  # Same for other side

        # Now handle empty tensor case explicitly
        if 0 in shape1.intlist() or 0 in shape2.intlist():
            abort("Cannot broadcast shapes with zero dimensions")
        # Normal case for non-scalars
        if shape1 == shape2:
            return shape1, shape2

        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)

        # Pad with 1s (never 0s)
        padded1 = one * (max_len - len1) + shape1.intlist()
        padded2 = one * (max_len - len2) + shape2.intlist()

        # Validate no zero dimensions
        if 0 in padded1 or 0 in padded2:
            abort("Invalid shape padding: resulted in zero dimension")

        return Shape(padded1), Shape(padded2)

    @always_inline
    @staticmethod
    fn broadcast_shape(this: Shape, that: Shape) -> Shape:
        if not this.broadcastable(that):
            abort(
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
                abort(
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
            abort(
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

    fn flatten_index(self, indices: VariadicList[Int], offset: Int = 0) -> Int:
        list = variadiclist_as_intlist(indices)
        return self.flatten_index(list, offset)

    fn flatten_index(self, indices: IntList, offset: Int = 0) -> Int:
        if self.ndim == 0:
            if len(indices) != 0:
                abort(
                    "Shape → flatten_index: scalar tensor should receive empty"
                    " indices[IntList]"
                )
            return 0 + offset
        if len(indices) != self.ndim:
            print(
                (
                    "Shape → flatten_index: shape mismatch len(indices) !="
                    " self.ndim -> "
                ),
                len(indices),
                "<=>",
                self.ndim,
            )
            return -1
        var index = offset
        var stride = 1
        for i in reversed(range(self.ndim)):
            idx = indices[i]
            dim = self.axes_spans[i]
            if idx < 0:
                idx += dim  # Negative index
            if idx >= dim or idx < 0:  # Negative index
                print(
                    (
                        "Shape → flatten_index: invalid index >= dim span or"
                        " negative: "
                    ),
                    idx,
                    dim,
                )
                return -1
            index += idx * stride
            stride *= dim
        return index

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
            abort("Shape → translate_index: original dims > broadcast dims")
        if not mask.size == broadcast_shape.ndim:
            abort("Shape → translate_index: mask/broadcast shape mismatch")
        if not indices.size == broadcast_shape.ndim:
            abort("Shape → translate_index: indices/broadcast shape mismatch")

        translated = IntList.with_capacity(self.ndim)
        offset = broadcast_shape.ndim - self.ndim

        for i in range(self.ndim):
            broadcast_axis = i + offset
            if not broadcast_axis < mask.size:
                abort("Shape → translate_index: invalid axis")

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

    fn __moveinit__(out self, owned other: Self):
        self.axes_spans = other.axes_spans
        self.ndim = other.ndim
        self.numels = other.numels

    fn __copyinit__(out self, other: Self):
        self.axes_spans = other.axes_spans
        self.ndim = other.ndim
        self.numels = other.numels

    fn free(owned self):
        log_debug("Freeing Shape")
        self.axes_spans.free()
        _ = self^

    @staticmethod
    fn validate(shape: Shape):
        for idx in range(shape.ndim):
            if shape.axes_spans[idx] < 1:
                abort(
                    "Shape → validate: shape dimension not valid: "
                    + String(shape.axes_spans[idx])
                )

    fn intlist(self) -> IntList:
        return self.axes_spans.copy()

    @staticmethod
    fn of(*dims: Int) -> Shape:
        return Shape(dims)
