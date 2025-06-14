from common_utils import variadiclist_as_intlist
from intlist import IntList
from os import abort
from memory import Pointer

from testing import assert_true, assert_raises


fn test_broadcast_shapes() raises:
    shape1 = Shape.of(3, 4)  # 2D tensor
    shape2 = Shape.of(
        4,
    )  # 1D tensor
    result = Shape.broadcast_shapes(shape1, shape2)
    assert_true(result == Shape.of(3, 4), "Shape broadcast 1 assertion failed")

    shape1 = Shape.of(5, 3, 1)
    shape2 = Shape.of(5, 1)
    with assert_raises():
        _ = Shape.broadcast_shapes(shape1, shape2)

    shape1 = Shape.of(1)
    shape2 = Shape.of(
        3,
        4,
    )
    result = Shape.broadcast_shapes(shape1, shape2)
    assert_true(result == Shape.of(3, 4), "Shape broadcast 1 assertion failed")


fn main() raises:
    test_broadcast_shapes()
    _ = """shape = Shape.of(1)
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

    fn drop_axis(self, axis: Int) -> Shape:
        if axis < 0 or axis >= self.ndim:
            abort("Shape -> drop_axis: Invalid axis " + String(axis))
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
    def broadcast_shapes(this: Shape, that: Shape) -> Shape:
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
                raise Error(
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
        print("Shape free kicking in alright")
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
