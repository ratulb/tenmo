from common_utils import variadiclist_as_intlist, log_debug
from intlist import IntList
from os import abort
from memory import Pointer

from tensors import Tensor


fn main() raises:
    test_replace()
    tensor1 = Tensor.of(1, 2, 3, 4, 5)
    tensor2 = Tensor.of(6)
    shape1 = tensor1.shape
    shape2 = tensor2.shape
    print("shapes: ", shape1, shape2)
    rzipped = shape1.intlist().zip_reversed(shape2.intlist())
    for each in rzipped:
        print(each[0], each[1])

    _ = """test_broadcastable()
    tensor1 = Tensor.rand(3, 1)
    print(tensor1.shape)
    tensor2 = Tensor.rand(1, 2)
    print(tensor2.shape)
    print("ndims: ", tensor1.shape.ndim, tensor2.shape.ndim)
    if tensor1.shape != tensor2.shape:
        broadcast_shape = Shape.broadcast_shape(tensor1.shape, tensor2.shape)
        print("broadcast_shape: ", broadcast_shape)
        mask = tensor1.shape.broadcast_mask(tensor2.shape)
        print("mask")
        mask.print()

        for indices in broadcast_shape:
            print("Broadcast shape index")
            indices.print()
            translated = tensor1.shape.translate_index(
                indices, mask, broadcast_shape
            )
            print("translated for shape1: ")
            translated.print()
        mask = tensor2.shape.broadcast_mask(tensor1.shape)
        for indices in broadcast_shape:
            #indices.print()
            translated = tensor2.shape.translate_index(indices, mask, broadcast_shape)
            print("translated for shape2: ")
            translated.print()
        print("mask")
        mask.print()

    else:
        summ = tensor1 + tensor2
        summ.print()"""

    _ = """test_broadcast_shape()
    test_index_iter()
    shape = Shape.of(1)
    # print(shape)
    for e in shape:
        e.print()
    print("==========")
    shape1 = Shape(IntList(2, 3, 4))
    print(shape1)
    dropped = shape1.drop_axis(2)
    print(dropped)
    for each in shape1:
        each.print()
    print(shape1)
    shape2 = Shape.of(1, 2, 3, 4, 5, 6)
    print(shape2)
    test_pad_shapes()
    test_shape_as_intlist()"""


from testing import assert_true, assert_raises

fn test_replace() raises:
    shape = Shape.of(3,4,2)
    shape = shape.replace(2, 5)
    assert_true(shape == Shape.of(3,4,5), "replace assertion failed")

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

    shape1 = Shape.of(5, 3, 1)
    shape2 = Shape.of(5, 1)
    with assert_raises():
        _ = Shape.broadcast_shape(shape1, shape2)

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
    print(
        tensor1.shape.broadcastable(tensor2.shape),
        tensor2.broadcastable(tensor1),
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


struct Shape(Sized & Writable & Copyable & Movable):
    alias UnitShape = Shape.of(1)
    var axes_spans: IntList
    var ndim: Int
    var numels: Int

    fn __init__(out self, dims: VariadicList[Int]):
        _dims = IntList.with_capacity(len(dims))
        for each in dims:
            _dims.append(each)
        self = Self(_dims)

    fn __init__(out self, dims: IntList):
        if len(dims) < 1:
            abort("Shape -> __init__: Shape dimension count should be > 0")
        _ndims = len(dims)
        for i in range(_ndims):
            if dims[i] < 1:
                abort("Shape -> __init__: Wrong shape dimension")
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

    @staticmethod
    fn pad_shapes(shape1: Shape, shape2: Shape) -> (Shape, Shape):
        if shape1 == shape2:
            return shape1, shape2
        one = IntList(1)
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)
        padded1 = one * (max_len - len1) + shape1.intlist()
        padded2 = one * (max_len - len2) + shape2.intlist()
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

    fn __len__(self) -> Int:
        return self.ndim

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

    fn translate_index(
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
                abort("Shape dimension not valid")

    fn intlist(self) -> IntList:
        result = IntList.with_capacity(len(self))
        for i in range(len(self)):
            result.append(self[i])
        return result

    @staticmethod
    fn of(*dims: Int) -> Shape:
        return Shape(dims)


from testing import assert_true


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
