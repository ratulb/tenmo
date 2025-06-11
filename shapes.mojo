from common_utils import variadiclist_as_list
from flexarray import FlexArray
from os import abort


struct Shape(Sized & Writable & Copyable & Movable):
    var axes_spans: FlexArray
    var ndim: Int
    var numels: Int

    fn __init__(out self, dims: VariadicList[Int]):
        if len(dims) < 1:
            abort("Shape dimension count should be > 0")
        for each in dims:
            if each < 1:
                abort("Wrong shape dimension")
        _spans = FlexArray(dims)
        _ndims = len(_spans)
        _numels = 1
        for idx in range(_ndims):
            _numels *= _spans[idx].__int__()
        self.axes_spans = _spans
        self.ndim = _ndims
        self.numels = _numels

    fn __init__(out self, dims: FlexArray):
        if len(dims) < 1:
            abort("Shape dimension count should be > 0")
        _ndims = len(dims)
        _spans = FlexArray.with_capacity(len(dims))
        for i in range(_ndims):
            if dims[i] < 1:
                abort("Wrong shape dimension")
            _spans[i] = dims[i].__int__()
        _numels = 1
        for idx in range(_ndims):
            _numels *= _spans[idx].__int__()
        self.axes_spans = _spans
        self.ndim = _ndims
        self.numels = _numels

    fn __len__(self) -> Int:
        return self.ndim

    fn __getitem__(self, index: Int) -> Int:
        var idx: Int
        if 0 <= index < self.ndim:
            idx = self.axes_spans[index].__int__()
        else:
            idx = -1
        return idx

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
        list = variadiclist_as_list(indices)
        return self.flatten_index(list)

    fn flatten_index(self, indices: List[Int]) -> Int:
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
            dim = self.axes_spans[i].__int__()
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

    @staticmethod
    fn validate(shape: Shape):
        for idx in range(shape.ndim):
            if shape.axes_spans[idx] < 1:
                abort("Shape dimension not valid")

    fn flexarray(self) -> FlexArray:
        result = FlexArray.with_capacity(len(self))
        for i in range(len(self)):
            result.append(self[i])
        return result

    @staticmethod
    fn of(*dims: Int) -> Shape:
        return Shape(dims)

    @staticmethod
    fn pad_shapes(shape1: Shape, shape2: Shape) -> (Shape, Shape):
        if shape1 == shape2:
            return shape1, shape2
        one = FlexArray(1)
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)
        padded1 = one * (max_len - len1) + shape1.flexarray()
        padded2 = one * (max_len - len2) + shape2.flexarray()
        return Shape(padded1), Shape(padded2)


from testing import assert_true


fn test_shape_as_flexarray() raises:
    shape = Shape.of(2, 4, 5)
    fa = shape.flexarray()
    assert_true(
        fa[0] == 2 and fa[1] == 4 and fa[2] == 5,
        "Shape to FlexArray assertion failed",
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


fn main() raises:
    shape1 = Shape(FlexArray(1, 2, 3, 4, 5, 6))
    print(shape1)
    shape2 = Shape.of(1, 2, 3, 4, 5, 6)
    print(shape2)
    test_pad_shapes()
    test_shape_as_flexarray()
