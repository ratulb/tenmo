from common_utils import varia_list_to_list


struct Shape(Sized & Writable):
    var axes_spans: VariadicList[Int]
    var ndim: Int
    var numels: Int

    fn __init__(out self, dims: VariadicList[Int]):
        self.axes_spans = dims
        self.ndim = len(self.axes_spans)
        if len(self.axes_spans) > 0:
            self.numels = 1
            for i in range(len(self.axes_spans)):
                self.numels *= self.axes_spans[i]
        else:
            self.numels = 0

    fn spans(self) -> VariadicList[Int]:
        return self.axes_spans

    fn __len__(self) -> Int:
        return len(self.axes_spans)

    fn __getitem__(self, index: Int) -> Int:
        var idx: Int
        if 0 <= index < len(self.axes_spans):
            idx = self.axes_spans[index]
        else:
            idx= -1
        print("Shape -> __getitem__: ", index, idx)
        return idx

    fn __eq__(self, other: Self) -> Bool:
        if len(self.axes_spans) != len(other.axes_spans):
            return False
        for i in range(len(self.axes_spans)):
            if self.axes_spans[i] != other.axes_spans[i]:
                return False
        return True

    fn __ne__(self, other: Self) -> Bool:
        return not self.__eq__(other)

    fn num_elements(self) -> Int:
        if len(self.axes_spans) == 0:
            return 0
        var numels = 1
        for i in range(len(self.axes_spans)):
            numels *= self.axes_spans[i]

        print("num_elements: ", numels)
        return numels

    fn flatten_index(self, indices: VariadicList[Int]) -> Int:
        list = varia_list_to_list(indices)
        return self.flatten_index(list)

    fn flatten_index(self, indices: List[Int]) -> Int:
        var elem_index: Int
        if len(indices) != len(self.axes_spans):
            elem_index = -1  # shape mismatch
        var index = 0
        var stride = 1
        for i in reversed(range(len(self.axes_spans))):
            idx = indices[i]
            dim = self.axes_spans[i]
            if idx >= dim:
                elem_index = -1
                print("flatten_index inner: ", elem_index, indices.__str__())
                return elem_index
            index += idx * stride
            stride *= dim
        elem_index = index
        print("flatten_index: ", elem_index, indices.__str__())
        return elem_index

    fn __str__(self) -> String:
        var s = String("(")
        for i in range(len(self.axes_spans)):
            s += String(self.axes_spans[i])
            if i < len(self.axes_spans) - 1:
                s += ", "
        s += ")"
        print("Shape -> __str__", s)
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
