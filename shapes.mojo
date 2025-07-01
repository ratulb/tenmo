from common_utils import variadiclist_as_intlist, log_debug
from intlist import IntList
from os import abort
from memory import Pointer

from tensors import Tensor


from testing import assert_true, assert_raises


fn test_equivalence() raises:
    assert_true(Shape(IntList(1, 4)) == Shape.of(1, 4), "Not equivalent")


fn test_empty_shape() raises:
    shape = Shape(IntList.Empty)
    assert_true(shape[0] == -1, "Empty shape __getitem__ assertion failed")
    for each in shape:
        assert_true(
            each == IntList.Empty, "Empty shape iteration assertion failed"
        )
    tensor = Tensor[DType.bool](shape)
    assert_true(
        tensor[IntList.Empty] == False, "Scalar tensor get assertion 1 failed"
    )
    tensor[IntList.Empty] = True
    assert_true(
        tensor[IntList.Empty] == True, "Scalar tensor get assertion 2 failed"
    )
    assert_true(tensor.item() == True, "Scalar tensor item() assertion failed")
    assert_true(
        shape.broadcastable(Shape.of(1)),
        "broadcastable assertion 1 failed for empty shape",
    )
    assert_true(
        Shape.of(1).broadcastable(shape),
        "broadcastable assertion 1 failed for empty shape",
    )

    broadcast_shape = Shape.broadcast_shape(shape, Shape.of(1))
    assert_true(
        broadcast_shape == Shape.of(1),
        "Empty shape broadcast to Shape.of(1) assertion failed",
    )

    broadcast_shape = Shape.broadcast_shape(Shape.of(1), shape)
    assert_true(
        broadcast_shape == Shape.of(1),
        "Shape.of(1) broadcast with empty shape assertion failed",
    )
    broadcast_mask = shape.broadcast_mask(Shape.of(1))
    assert_true(
        broadcast_mask == IntList(1),
        "Empty shape broadcast mask assertion failed",
    )


fn test_replace() raises:
    shape = Shape.of(3, 4, 2)
    shape = shape.replace(2, 5)
    assert_true(shape == Shape.of(3, 4, 5), "replace assertion failed")


fn test_broadcast_shape() raises:
    shape1 = Shape.of(32, 16)
    shape2 = Shape.of(
        16,
    )
    result = Shape.broadcast_shape(shape1, shape2)
    assert_true(
        result == Shape.of(32, 16), "Shape broadcast 1 assertion failed"
    )

    shape1 = Shape.of(4, 16, 32)
    shape2 = Shape.of(
        32,
    )
    result = Shape.broadcast_shape(shape1, shape2)
    assert_true(
        result == Shape.of(4, 16, 32), "Shape broadcast 2 assertion failed"
    )

    shape1 = Shape.of(4, 16, 32, 64)
    shape2 = Shape.of(
        64,
    )
    result = Shape.broadcast_shape(shape1, shape2)
    assert_true(
        result == Shape.of(4, 16, 32, 64), "Shape broadcast 3 assertion failed"
    )

    _ = """shape1 = Shape.of(5, 3, 1)
    shape2 = Shape.of(5, 1)

    with assert_raises():
        _ = Shape.broadcast_shape(shape1, shape2)"""

    shape1 = Shape.of(1)
    shape2 = Shape.of(
        3,
        4,
    )
    result = Shape.broadcast_shape(shape1, shape2)
    assert_true(result == Shape.of(3, 4), "Shape broadcast 4 assertion failed")

    result = Shape.broadcast_shape(Shape.of(2, 1), Shape.of(4, 2, 5))
    assert_true(
        result == Shape.of(4, 2, 5), "Shape broadcast 5 assertion failed"
    )

    tensor1 = Tensor.d2([[1, 2, 3], [4, 5, 6]])
    tensor2 = Tensor.d2([[1, 1, 1]])

    shape1 = tensor1.broadcast_shape(tensor2)
    shape2 = tensor2.broadcast_shape(tensor1)
    shape3 = tensor1.broadcast_shape(tensor1)
    shape4 = tensor2.broadcast_shape(tensor2)
    print(shape1, shape2, shape3, shape3)
    assert_true(
        shape1 == Shape.of(2, 3)
        and shape2 == Shape.of(2, 3)
        and shape3 == Shape.of(2, 3)
        and shape4 == Shape.of(1, 3),
        (
            "Tensor broadcast shape assertion failed for tensor sizes 2 by 3"
            " and 1 by 3"
        ),
    )


fn test_index_iter() raises:
    shape = Shape.of(1)
    for each in shape:
        assert_true(
            each == IntList(0),
            "Unit shape(Shape.of(1)) index iteration assertion failed",
        )
    shape = Shape.of(2, 1)
    indices = shape.__iter__()
    assert_true(
        indices.__next__() == IntList(0, 0)
        and indices.__next__() == IntList(1, 0),
        "Shape(2,1) iteration assertion failed",
    )


fn test_broadcastable() raises:
    assert_true(
        Shape.of(1).broadcastable(Shape.of(1)),
        "broadcastable assertion 1 failed",
    )
    assert_true(
        Shape.of(4, 5).broadcastable(Shape.of(1)),
        "broadcastable assertion 2 failed",
    )
    assert_true(
        Shape.of(2, 3, 5).broadcastable(Shape.of(1, 5)),
        "broadcastable assertion 3 failed",
    )
    assert_true(
        Shape.of(2, 3, 5).broadcastable(Shape.of(3, 5)),
        "broadcastable assertion 4 failed",
    )
    tensor1 = Tensor.of(1, 2, 3, 4, 5)
    tensor2 = Tensor.of(6)
    assert_true(
        tensor1.shape.broadcastable(tensor2.shape)
        and tensor2.broadcastable(tensor1),
        "Tensor shape broadcastable assertion failed",
    )


fn test_shape_as_intlist() raises:
    shape = Shape.of(2, 4, 5)
    fa = shape.intlist()
    assert_true(
        fa[0] == 2 and fa[1] == 4 and fa[2] == 5,
        "Shape to IntList assertion failed",
    )


fn test_pad_shapes() raises:
    shape1 = Shape.of(3, 4)
    shape2 = Shape.of(
        4,
    )
    padded1, padded2 = Shape.pad_shapes(shape1, shape2)
    assert_true(
        padded1 == shape1 and padded2 == Shape.of(1, 4),
        "Padding of shapes (3,4) and (4,) assertion failed",
    )
    shape1 = Shape.of(5, 3, 1)
    shape2 = Shape.of(5, 1)
    padded1, padded2 = Shape.pad_shapes(shape1, shape2)
    assert_true(
        padded1 == shape1 and padded2 == Shape.of(1, 5, 1),
        "Padding of shapes (5,3,1) and (5,1) assertion failed",
    )
    shape1 = Shape.of(
        1,
    )
    shape2 = Shape.of(3, 4)
    padded1, padded2 = Shape.pad_shapes(shape1, shape2)
    assert_true(
        padded1 == Shape.of(1, 1) and padded2 == shape2,
        "Padding of shapes (1, ) and (3,4) assertion failed",
    )
    shape1 = Shape.of(3, 4, 5, 2)
    shape2 = Shape.of(3, 4, 5, 2)
    padded1, padded2 = Shape.pad_shapes(shape1, shape2)
    assert_true(
        padded1 == shape1 and padded2 == shape2,
        "Padding of shapes (3,4,5,2 ) and (3,4,5,2) assertion failed",
    )


fn test_zip_reversed() raises:
    shape1 = Shape.of(1, 2, 3, 4, 5)
    shape2 = Shape.of(6)
    rzipped = shape1.intlist().zip_reversed(shape2.intlist())
    for each in rzipped:
        assert_true(
            each[0] == 5 and each[1] == 6, "zip_reversed assertion failed"
        )


fn main() raises:
    test_equivalence()
    test_empty_shape()
    test_replace()
    test_broadcastable()
    test_pad_shapes()
    test_broadcast_shape()
    test_shape_as_intlist()
    test_index_iter()
    test_zip_reversed()


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

    fn __next__(mut self) -> IntList:  #
        result = self.current.copy()  #
        self.index += 1  #
        for i in range(self.shape[].ndim - 1, -1, -1):
            self.current[i] += 1
            # self.current.print()
            if self.current[i] < self.shape[][i]:  #
                break
            self.current[i] = 0  #
            # self.current.print() #
        return result  #

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

    _ = """fn __init__(out self):
        self = Self.Void"""

    fn __init__(out self, dims: VariadicList[Int]):
        _dims = IntList.with_capacity(len(dims))
        for each in dims:
            _dims.append(each)
        self = Self(_dims)

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
                    "Shape -> __init__: Wrong shape dimension."
                    + "Dim = "
                    + String(dims[i])
                    + " at index = "
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

    fn broadcastable(self, to: Shape) -> Bool:
        dims1 = self.intlist()
        dims2 = to.intlist()
        zip_reversed = dims1.zip_reversed(dims2)
        for dims in zip_reversed:
            if dims[0] != dims[1]:
                if dims[0] != 1 and dims[1] != 1:
                    return False
        return True

    fn broadcast_mask(self, target_shape: Shape) -> IntList:
        mask = IntList.with_capacity(target_shape.ndim)
        offset = target_shape.ndim - self.ndim
        if offset < 0:
            abort(
                "Shape -> broadcast_mask -> target_shape.ndim is smaller than"
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

    fn replace(self, axis: Int, extent: Int) -> Shape:
        if axis < 0 or axis >= self.ndim:
            abort(
                "Shape -> replace: Invalid axis: "
                + String(axis)
                + " for shape: "
                + self.__str__()
            )
        if extent < 1:
            abort("Shape -> replace: Invalid extent: " + String(extent))
        axes = self.intlist()
        axes[axis] = extent
        return Shape(axes)

    fn drop_axis(self, axis: Int) -> Shape:
        if axis < 0 or axis >= self.ndim:
            abort(
                "Shape -> drop_axis: Invalid axis: "
                + String(axis)
                + " for shape: "
                + self.__str__()
            )
        if self.ndim == 1:
            shape = self
            return shape
        axes = self.intlist()[:axis] + self.intlist()[axis + 1 :]
        return Shape(axes)

    _ = """@staticmethod
    fn pad_shapes(shape1: Shape, shape2: Shape) -> (Shape, Shape):
        if shape1 == shape2:
            return shape1, shape2
        one = IntList(1)
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)
        padded1 = one * (max_len - len1) + shape1.intlist()
        padded2 = one * (max_len - len2) + shape2.intlist()
        return Shape(padded1), Shape(padded2)"""

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

    @staticmethod
    fn broadcast_shape(this: Shape, that: Shape) -> Shape:
        if not this.broadcastable(that):
            abort(
                "Shape -> broadcast_shape - not broadcastable: "
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
                    "Shape -> broadcast_shape - cannot broadcast shapes: "
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
        return len(self)

    fn __getitem__(self, idx: Int) -> Int:
        if 0 <= idx < self.ndim:
            return self.axes_spans[idx]
        else:
            return -1

    fn __eq__(self, other: Self) -> Bool:
        if self.ndim != other.ndim:
            return False
        for idx in range(self.ndim):
            if self.axes_spans[idx] != other.axes_spans[idx]:
                return False
        return True

    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    fn num_elements(self) -> Int:
        return self.numels

    fn flatten_index(self, indices: VariadicList[Int]) -> Int:
        list = variadiclist_as_intlist(indices)
        return self.flatten_index(list)

    fn flatten_index(self, indices: IntList) -> Int:
        if self.ndim == 0:
            if len(indices) != 0:
                abort(
                    "Shape.flatten_index: Scalar tensor should receive no"
                    " indices"
                )
            return 0
        if len(indices) != self.ndim:
            print(
                (
                    "Shape fltatten_index -> shape mismatch len(indices) !="
                    " self.ndim"
                ),
                len(indices),
                self.ndim,
            )
            return -1
        var index = 0
        var stride = 1
        for i in reversed(range(self.ndim)):
            idx = indices[i]
            dim = self.axes_spans[i]
            if idx >= dim:
                print("Shape fltatten_index -> index >= dim span", idx, dim)
                return -1
            index += idx * stride
            stride *= dim
        return index

    _ = """fn translate_index(
        self, indices: IntList, mask: IntList, broadcast_shape: Shape
    ) -> IntList:
        translated = IntList.with_capacity(self.ndim)
        offset = broadcast_shape.ndim - self.ndim

        for i in range(self.ndim):
            broadcast_axis = i + offset
            if mask[broadcast_axis] == 1:
                translated.append(0)
            else:
                translated.append(indices[broadcast_axis])

        return translated"""

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
            abort("Original dims > broadcast dims")
        if not mask.size == broadcast_shape.ndim:
            abort("Mask/broadcast shape mismatch")
        if not indices.size == broadcast_shape.ndim:
            abort("Indices/broadcast shape mismatch")

        translated = IntList.with_capacity(self.ndim)
        offset = broadcast_shape.ndim - self.ndim

        for i in range(self.ndim):
            broadcast_axis = i + offset
            if not broadcast_axis < mask.size:
                abort("Invalid axis")

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
        log_debug("Shape free kicking in alright")
        self.axes_spans.free()
        _ = self^

    @staticmethod
    fn validate(shape: Shape):
        for idx in range(shape.ndim):
            if shape.axes_spans[idx] < 1:
                abort(
                    "Shape -> validate - Shape dimension not valid: "
                    + String(shape.axes_spans[idx])
                )

    fn intlist(self) -> IntList:
        return self.axes_spans.copy()

    @staticmethod
    fn of(*dims: Int) -> Shape:
        return Shape(dims)
