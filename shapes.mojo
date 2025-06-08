from common_utils import variadiclist_as_list
from testing import assert_true
from utils import StaticTuple

# Max supported dimensions is currently limited to 5
alias max_dim_count = 5


struct Shape(Sized & Writable):
    var axes_spans: StaticTuple[Int, max_dim_count]
    var ndim: Int
    var numels: Int

    fn __init__(out self, dims: VariadicList[Int]) raises:
        assert_true(len(dims) > 0, "Shape dimension count should be > 0")
        for each in dims:
            assert_true(each > 0, "Wrong shape dimension")
        _spans = StaticTuple[Int, max_dim_count](dims)
        _ndims = min(len(dims), len(_spans))
        _numels = 1
        for idx in range(_ndims):
            _numels *= _spans[idx]
        for idx in range(_ndims, max_dim_count):
            _spans[idx] = -1  # Negate anything beyond idx >=_ndims
        self.axes_spans = _spans
        self.ndim = _ndims
        self.numels = _numels
    fn __len__(self) -> Int:
        return self.ndim

    fn __getitem__(self, index: Int) -> Int:
        var idx: Int
        if 0 <= index < self.ndim:
            idx = self.axes_spans[index]
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

    @staticmethod
    fn validate(shape: Shape) raises:
        for idx in range(shape.ndim):
            assert_true(shape.axes_spans[idx] > 0, "Shape dimension not valid")

    fn as_list(self) -> List[Int]:
        result = List[Int](capacity=len(self))
        for i in range(len(self)):
            result.append(self[i])
        return result




        
