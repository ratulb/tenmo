from memory import UnsafePointer, memcpy
from random import randn, seed
from utils import StaticTuple


def main():
    # tensor = Tensor[5].rand(4, 3, 2, 1)
    tensor = Tensor[2].rand(4, 3)
    var indices = List[Int]()
    tensor.print_tensor_recursive(indices, 1)


struct Tensor[axes_sizes: Int = 1, dtype: DType = DType.float32]:
    var shape: Shape[axes_sizes]
    var data: UnsafePointer[Scalar[dtype]]
    var datatype: DType

    fn __init__(out self, *tensor_shapes: Int) raises:
        axes_dims = Self.init_shape(tensor_shapes)
        self = Self(axes_dims)

    @staticmethod
    fn init_shape(
        shapes: VariadicList[Int],
    ) raises -> StaticTuple[Int, axes_sizes]:
        if len(shapes) != axes_sizes:
            err = (
                "Tensor dimension = "
                + String(axes_sizes)
                + " and args count = "
                + String(len(shapes))
                + " mismatch"
            )
            print(err)
            raise Error("Dimension mismatch")
        axes_dims = StaticTuple[Int, axes_sizes](0)
        for i in range(axes_sizes):
            axes_dims[i] = shapes[i]
        return axes_dims

    fn __init__(out self, axes_dims: StaticTuple[Int, axes_sizes]):
        self.shape = Shape[axes_sizes](axes_dims)
        self.datatype = dtype
        self.data = UnsafePointer[Scalar[dtype]].alloc(self.shape.numels)

    fn __getitem__(self, indices: List[Int]) raises -> Scalar[dtype]:
        static_tuple = StaticTuple[Int, axes_sizes](0)
        for i in range(axes_sizes):
            static_tuple[i] = indices[i]
        index = self.shape.flatten_index(static_tuple)
        if index == -1:
            raise Error("Invalid indices")
        return (self.data + index)[]

    fn __getitem__(self, *indices: Int) raises -> Scalar[dtype]:
        index = self.shape.flatten_index(indices)
        if index == -1:
            raise Error("Invalid indices")
        return (self.data + index)[]

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]) raises:
        index = self.shape.flatten_index(indices)
        if index == -1:
            raise Error("Invalid indices")
        (self.data + index)[] = value

    fn __moveinit__(out self, owned other: Self):
        self.data = UnsafePointer[Scalar[dtype]].alloc(other.numels())
        memcpy(self.data, other.data, other.numels())
        self.shape = other.shape^
        self.datatype = other.datatype

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.data = UnsafePointer[Scalar[dtype]].alloc(other.numels())
        memcpy(self.data, other.data, other.numels())
        self.datatype = other.datatype

    fn __del__(owned self):
        self.data.free()

    @always_inline
    fn numels(self) -> Int:
        return self.shape.numels

    @always_inline
    fn ndim(self) -> Int:
        return self.shape.ndim

    fn __eq__(self: Self, rhs: Self) -> Bool:
        return True

    fn __ne__(self: Self, rhs: Self) -> Bool:
        return True

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self.data

    fn __str__(self) -> String:
        s = String("[")
        if axes_sizes == 1:
            s += "1D Tensor"
        elif axes_sizes == 2:
            s += "2D Tensor"
        elif axes_sizes == 3:
            s += "3D Tensor"
        elif axes_sizes == 4:
            s += "4D Tensor"
        else:
            s += "Unsupported Tensor"
        s += self.shape.__str__()
        s += ", Type: " + self.datatype.__str__()
        s += "]"
        return s

    @staticmethod
    fn rand(
        *tensor_shapes: Int, init_seed: Optional[Int] = None
    ) raises -> Tensor[axes_sizes, dtype]:
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        axes_dims = Self.init_shape(tensor_shapes)
        tensor = Tensor[axes_sizes, dtype](axes_dims)
        randn(tensor.data, tensor.numels())
        return tensor

    # fn print_tensor_recursive(self, mut indices: List[Int], level: Int) raises:
    # try:
    # current_dim = len(indices)
    # indent = " " * (level * 2)
    #
    # if current_dim == self.ndim() - 1:
    # print(indent + "[", end="")
    # for i in range(self.shape[current_dim]):
    # indices.append(i)
    # print(self[indices], end="")
    # _ = indices.pop()
    # if i < self.shape[current_dim] - 1:
    # print(", ", end="")
    # print("]")
    # else:
    # print(indent + "[")
    # for i in range(self.shape[current_dim]):
    # indices.append(i)
    # self.print_tensor_recursive(indices, level + 1)
    # _ = indices.pop()
    # if i < self.shape[current_dim] - 1:
    # print(",")
    # print(indent + "]")
    # except e:
    # print(e)
    # fn print_tensor_recursive(self, mut indices: List[Int], level: Int) raises:
    # try:
    # current_dim = len(indices)
    # indent = " " * (level * 2)
    #
    ##max_items_to_show = 5  # How many items per dimension to display at most
    # num_first = 3          # Show first 3
    # num_last = 2           # Show last 2
    #
    # if current_dim == self.ndim() - 1:
    # print(indent + "[", end="")
    # size = self.shape[current_dim]
    #
    # for i in range(size):
    # if i < num_first or i >= size - num_last:
    # indices.append(i)
    # print(self[indices], end="")
    # _ = indices.pop()
    # elif i == num_first:
    # print("...", end="")
    #
    # if i < size - 1:
    # print(", ", end="")
    # print("]")
    # else:
    # print(indent + "[")
    # size = self.shape[current_dim]
    #
    # for i in range(size):
    # if i < num_first or i >= size - num_last:
    # indices.append(i)
    # self.print_tensor_recursive(indices, level + 1)
    # _ = indices.pop()
    # elif i == num_first:
    # print(indent + "  ...")  # indent deeper
    #
    # if i < size - 1 and (i < num_first - 1 or i >= size - num_last):
    # print(",")
    # print(indent + "]")
    # except e:
    # print(e)
    fn print_tensor_recursive(self, mut indices: List[Int], level: Int) raises:
        try:
            current_dim = len(indices)
            indent = " " * (level * 2)

            num_first = 3  # Show first 3 elements
            num_last = 2  # Show last 2 elements

            if current_dim == self.ndim() - 1:
                print(indent + "[", end="")
                size = self.shape[current_dim]

                for i in range(size):
                    if i < num_first:
                        indices.append(i)
                        print(
                            self[indices],
                            end=", " if (
                                i != num_first - 1
                                or size > num_first + num_last
                            ) else "",
                        )
                        _ = indices.pop()
                    elif i == num_first:
                        if size > num_first + num_last:
                            print("..., ", end="")
                        # Skip printing middle elements
                    elif i >= size - num_last:
                        indices.append(i)
                        print(self[indices], end=", " if i != size - 1 else "")
                        _ = indices.pop()
                print("]", end="")
            else:
                print(indent + "[")
                size = self.shape[current_dim]

                for i in range(size):
                    if i < num_first:
                        indices.append(i)
                        self.print_tensor_recursive(indices, level + 1)
                        _ = indices.pop()
                        if i != num_first - 1 or size > num_first + num_last:
                            print(",")
                    elif i == num_first:
                        if size > num_first + num_last:
                            print(indent + "  ...,")
                    elif i >= size - num_last:
                        indices.append(i)
                        self.print_tensor_recursive(indices, level + 1)
                        _ = indices.pop()
                        if i != size - 1:
                            print(",")
                print(indent + "]", end="")
        except e:
            print(e)


struct Shape[axes: Int]:
    var axes_sizes: StaticTuple[Int, axes]
    var ndim: Int
    var numels: Int

    fn __init__(out self, array: StaticTuple[Int, axes]):
        self.axes_sizes = array
        self.ndim = axes
        if len(array) > 0:
            self.numels = 1
            for i in range(axes):
                self.numels *= self.axes_sizes[i]
        else:
            self.numels = 0

    fn flatten_index(self, indices: StaticTuple[Int, size=axes]) -> Int:
        index = 0
        stride = 1
        for i in reversed(range(self.ndim)):
            idx = indices[i]
            self_idx = self[i]
            if idx >= self_idx:
                return -1
            index += idx * stride
            stride *= self_idx
        return index

    fn __str__(self) -> String:
        s = String("(")
        for i in range(axes):
            s += String(self.axes_sizes[i])
            if i < axes - 1:
                s += ", "
        s += ")"
        return s

    fn __repr__(self) -> String:
        return self.__str__()

    fn write_to[W: Writer](self, mut writer: W):
        s = self.__str__()
        writer.write(s)

    fn __getitem__(self, index: Int) -> Int:
        if 0 <= index < axes:
            return self.axes_sizes[index]
        else:
            return -1

    fn __moveinit__(out self, owned other: Self):
        self.axes_sizes = other.axes_sizes
        self.ndim = other.ndim
        self.numels = other.numels

    fn __copyinit__(out self, other: Self):
        self.axes_sizes = other.axes_sizes
        self.ndim = other.ndim
        self.numels = other.numels
