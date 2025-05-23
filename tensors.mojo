from math import iota
from random import randn, seed
from utils import StaticTuple
from time import perf_counter_ns
from algorithm import vectorize
from sys import simdwidthof
from memory import UnsafePointer, memcpy, memset, memset_zero
from common_utils import int_varia_list_to_str


fn is_true(x: Scalar[DType.bool]) -> Bool:
    return x


def main():
    t1 = Tensor(5)
    for i in range(5):
        t1[i] = i
    added_3 = t1 + 3
    print("Added\n")
    Tensor.print(added_3)
    expected = Tensor.arange(8, start=3)
    print("Expected\n")
    Tensor.print(expected)
    result = added_3 == expected
    print("Result\n")
    Tensor.print(result)

    print(result.for_all(is_true))
    # tensor = Tensor[5].rand(4, 3, 2, 1)
    _ = """Tensor.print(Tensor.arange(7, start=3).reshape[2](2, 2))
    tensor = Tensor[2].rand(4, 3)
    print("Am I gone: ")
    Tensor.print(tensor)
    print()
    multiplied = tensor / 2
    print("I am multiplied: ")
    Tensor.print(multiplied)
    print()

    rival = tensor == multiplied
    print("rival")
    Tensor.print(rival)

    tensor = Tensor[2].rand(4, 3)
    print("Original")
    Tensor.print(tensor)
    reshaped = tensor.reshape[3](2, 2, 3)
    print("Reshaped")
    Tensor.print(reshaped)

    tensor_false = Tensor[2, DType.bool].zeros(4, 3)
    indices = List[Int]()
    tensor_false.print_tensor_recursive(indices, 1)

    tensor_true = Tensor[2, DType.bool].ones(4, 3)
    indices = List[Int]()
    tensor_true.print_tensor_recursive(indices, 1)

    tensor = Tensor[2].ones(4, 3)
    indices = List[Int]()
    tensor.print_tensor_recursive(indices, 1)

    t16 = Tensor[2, DType.uint16].zeros(5, 5)
    t16[0, 0] = 1
    t16[0, 1] = 2
    t16[0, 2] = 3
    t16[0, 3] = 4
    t16[0, 4] = 5

    t16[1, 0] = 6
    t16[1, 1] = 7
    t16[1, 2] = 8
    t16[1, 3] = 9
    t16[1, 4] = 10

    t16[2, 0] = 11
    t16[2, 1] = 12
    t16[2, 2] = 13
    t16[2, 3] = 14
    t16[2, 4] = 15

    t16[3, 0] = 16
    t16[3, 1] = 17
    t16[3, 2] = 18
    t16[3, 3] = 19
    t16[3, 4] = 20

    t16[4, 0] = 21
    t16[4, 1] = 22
    t16[4, 2] = 23
    t16[4, 3] = 24
    t16[4, 4] = 25

    other = Tensor[2, DType.uint16].zeros(5, 5)
    other[0, 0] = 10
    other[0, 1] = 2
    other[0, 2] = 3
    other[0, 3] = 4
    other[0, 4] = 7

    other[1, 0] = 6
    other[1, 1] = 7
    other[1, 2] = 8
    other[1, 3] = 9
    other[1, 4] = 10

    other[2, 0] = 13
    other[2, 1] = 14
    other[2, 2] = 15
    other[2, 3] = 16
    other[2, 4] = 17

    other[3, 0] = 18
    other[3, 1] = 19
    other[3, 2] = 20
    other[3, 3] = 21
    other[3, 4] = 22

    other[4, 0] = 23
    other[4, 1] = 24
    other[4, 2] = 25
    other[4, 3] = 26
    other[4, 4] = 25

    # Tensor.print(t16.matmal_v2(other))
    print()

    # Tensor.print(t16.matmal_v3(other))
    print()

    # Tensor.print(t16.matmal(other))
    Tensor.print(t16 == other)
    Tensor.print(t16 != other)

    tensor_big1 = Tensor[2].rand(1024, 4096)
    tensor_big2 = Tensor[2].rand(4096, 512)

    # Tensor.print(tensor_big1.matmal_v3(tensor_big2))
    """


struct Tensor[axes_sizes: Int = 1, dtype: DType = DType.float32](Stringable):
    var shape: Shape[axes_sizes]
    var data: UnsafePointer[Scalar[dtype]]
    var datatype: DType

    fn __init__(out self, *tensor_shapes: Int) raises:
        axes_dims = Self.init_shape[axes_sizes](tensor_shapes)
        self = Self(axes_dims)

    @staticmethod
    fn init_shape[
        axes_count: Int
    ](axes_spans: VariadicList[Int],) raises -> StaticTuple[Int, axes_count]:
        axes_dims = StaticTuple[Int, axes_count](-1)
        if len(axes_spans) != axes_count:
            err = (
                "Tensor dimension = "
                + String(axes_count)
                + " and args count = "
                + String(len(axes_spans))
                + " mismatch"
            )
            raise Error(err)
        for i in range(axes_count):
            axes_dims[i] = axes_spans[i]
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
            raise Error("__getitem__(indices): Invalid indices")
        return (self.data + index)[]

    fn __getitem__(self, *indices: Int) raises -> Scalar[dtype]:
        index = self.shape.flatten_index(indices)
        if index == -1:
            raise Error("__getitem__(*indices): Invalid indices")
        return (self.data + index)[]

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]) raises:
        index = self.shape.flatten_index(indices)
        if index == -1:
            raise Error("__setitem__(*indices): Invalid indices")
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

    fn for_all[
        simd_width: Int = simdwidthof[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        num_elems = self.numels()
        simd_blocks = num_elems // simd_width
        remaining = num_elems % simd_width

        for i in range(simd_blocks):
            vector = self.data.load[width=simd_width](i * simd_width)
            for j in range(simd_width):
                if not pred(vector[j]):
                    return False
        for k in range(remaining):
            if not pred(self.data.load[width=1](simd_blocks * simd_width + k)):
                return False
        return True

    fn __eq__(self: Self, other: Self) raises -> Tensor[axes_sizes, DType.bool]:
        if self.shape != other.shape:
            raise Error("Dimension mismatch")
        copy = Tensor[axes_sizes, DType.bool](StaticTuple[Int, axes_sizes](0))
        copy.shape = self.shape

        @parameter
        fn compare_elems[simd_width: Int](idx: Int):
            copy.data.store[width=simd_width, volatile=True](
                idx,
                self.data.load[width=simd_width](idx)
                == other.data.load[width=simd_width](idx),
            )

        vectorize[compare_elems, simdwidthof[DType.bool]()](copy.numels())
        return copy

    fn __add__(self: Self, other: Self) raises -> Tensor[axes_sizes, dtype]:
        if self.shape != other.shape:
            raise Error("Add -> Dimension mismatch")
        result = Tensor[axes_sizes, dtype](StaticTuple[Int, axes_sizes]())
        result.shape = self.shape

        @parameter
        fn add_elems[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx,
                (
                    self.data.load[width=simd_width](idx)
                    + other.data.load[width=simd_width](idx)
                ),
            )

        vectorize[add_elems, simdwidthof[dtype]()](result.numels())
        return result

    fn __ne__(self: Self, other: Self) raises -> Tensor[axes_sizes, DType.bool]:
        result = self == other

        @parameter
        fn invert[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, ~result.data.load[width=simd_width](idx)
            )

        vectorize[invert, simdwidthof[DType.bool]()](result.numels())
        return result

    fn __mul__(self: Self, factor: Scalar[dtype]) -> Self:
        copy = self

        @parameter
        fn mul_by_factor[simd_width: Int](idx: Int):
            copy.data.store[width=simd_width](
                idx, copy.data.load[width=simd_width](idx) * factor
            )

        vectorize[mul_by_factor, simdwidthof[dtype]()](copy.numels())
        return copy

    fn __add__(self: Self, value: Scalar[dtype]) -> Self:
        copy = self

        @parameter
        fn add_value[simd_width: Int](idx: Int):
            copy.data.store[width=simd_width](
                idx, copy.data.load[width=simd_width](idx) + value
            )

        vectorize[add_value, simdwidthof[dtype]()](copy.numels())
        return copy

    fn __sub__(self: Self, value: Scalar[dtype]) -> Self:
        copy = self

        @parameter
        fn subtract_value[simd_width: Int](idx: Int):
            copy.data.store[width=simd_width](
                idx, copy.data.load[width=simd_width](idx) - value
            )

        vectorize[subtract_value, simdwidthof[dtype]()](copy.numels())
        return copy

    fn __truediv__(self: Self, factor: Scalar[dtype]) -> Self:
        copy = self

        @parameter
        fn div_by_factor[simd_width: Int](idx: Int):
            copy.data.store[width=simd_width](
                idx, copy.data.load[width=simd_width](idx).__truediv__(factor)
            )

        vectorize[div_by_factor, simdwidthof[dtype]()](copy.numels())
        return copy

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self.data

    fn matmal(self, other: Self) raises -> Tensor[axes_sizes, dtype]:
        start = perf_counter_ns()
        from testing import assert_equal

        result = Tensor[axes_sizes, dtype].zeros(self.shape[0], other.shape[1])
        try:
            assert_equal(self.shape[1], other.shape[0], "matmul - Dim mismatch")
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    for k in range(self.shape[1]):
                        result[i, j] += self[i, k] * other[k, j]
        except e:
            raise e
        end = perf_counter_ns()
        print("Total: ", end - start)
        return result

    fn load[
        nelts: Int = 1
    ](self, rows: Int, cols: Int) raises -> SIMD[dtype, nelts]:
        from testing import assert_equal

        try:
            assert_equal(2, self.ndim(), "load is supported only for 2d tensor")
        except e:
            raise e
        return self.data.load[width=nelts](rows * self.shape[1] + cols)

    fn store[
        nelts: Int = 1
    ](self, rows: Int, cols: Int, val: SIMD[dtype, nelts]) raises:
        from testing import assert_equal

        try:
            assert_equal(
                2, self.ndim(), "store is supported only for 2d tensor"
            )
        except e:
            raise e
        self.data.store(rows * self.shape[1] + cols, val)

    fn matmal_v2(self, other: Self) raises -> Tensor[axes_sizes, dtype]:
        start = perf_counter_ns()
        from testing import assert_equal

        result = Tensor[axes_sizes, dtype].zeros(self.shape[0], other.shape[1])
        try:
            assert_equal(self.shape[1], other.shape[0], "matmul - Dim mismatch")
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    for k in range(other.shape[1]):
                        result[i, k] += self[i, j] * other[j, k]
        except e:
            raise e
        end = perf_counter_ns()
        print("Total: ", end - start)

        return result

    fn matmal_v3(self, other: Self) raises -> Tensor[axes_sizes, dtype]:
        start = perf_counter_ns()
        from testing import assert_equal

        result = Tensor[axes_sizes, dtype].zeros(self.shape[0], other.shape[1])
        try:
            assert_equal(self.shape[1], other.shape[0], "matmul - Dim mismatch")
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):

                    @parameter
                    fn dot[simd_width: Int](idx: Int):
                        try:
                            result.store[simd_width](
                                i,
                                idx,
                                result.load[simd_width](i, idx)
                                + self[i, j] * other.load[simd_width](j, idx),
                            )
                        except e:
                            print(e)

                    vectorize[dot, 2 * simdwidthof[dtype]()](other.shape[1])
        except e:
            raise e
        end = perf_counter_ns()
        print("Total: ", end - start)

        return result

    fn reshape[
        new_axes_sizes: Int
    ](self, *newdims: Int) raises -> Tensor[new_axes_sizes, dtype]:
        shape = Self.init_shape[new_axes_sizes](newdims)  # StaticTuple
        numels = 1
        for idx in range(len(shape)):
            numels *= shape[idx]
        if numels != self.numels():
            raise Error(
                "Tensor with "
                + String(self.numels())
                + " elements can't be converted to "
                + int_varia_list_to_str(newdims)
                + " dimensional tensor"
            )
        result = Tensor[new_axes_sizes, dtype](shape)

        @parameter
        fn copy_elements[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx)
            )

        vectorize[copy_elements, simdwidthof[dtype]()](self.numels())
        return result

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
        axes_dims = Self.init_shape[axes_sizes](tensor_shapes)
        tensor = Tensor[axes_sizes, dtype](axes_dims)
        randn(tensor.data, tensor.numels())
        return tensor

    @staticmethod
    fn arange[
        d_type: DType = DType.float32
    ](end: Int, start: Int = 0) raises -> Tensor[1, d_type]:
        len = end - start
        result = Tensor[dtype=d_type](len)
        iota(result.data, len, offset=start)
        return result

    @staticmethod
    fn zeros(*tensor_shapes: Int) raises -> Tensor[axes_sizes, dtype]:
        axes_dims = Self.init_shape[axes_sizes](tensor_shapes)
        tensor = Tensor[axes_sizes, dtype](axes_dims)
        memset_zero(tensor.data, tensor.numels())
        return tensor

    @staticmethod
    fn ones(*tensor_shapes: Int) raises -> Tensor[axes_sizes, dtype]:
        axes_dims = Self.init_shape[axes_sizes](tensor_shapes)
        tensor = Tensor[axes_sizes, dtype](axes_dims)
        var value: SIMD[dtype, 1]

        @parameter
        if dtype.is_floating_point():
            value = SIMD[dtype, 1](1.0)
        else:
            value = SIMD[dtype, 1](1)
        for i in range(tensor.numels()):
            tensor.data.store(i, value)
        return tensor

    fn print_tensor_recursive(self, mut indices: List[Int], level: Int) raises:
        try:
            current_dim = len(indices)
            indent = " " * (level * 2)

            num_first = 5  # Show first 3 elements
            num_last = 5  # Show last 2 elements

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
                print("\n")
        except e:
            print(e)

    @staticmethod
    fn print(t: Tensor):
        print(t.__str__())
        print()
        l = List[Int]()
        try:
            t.print_tensor_recursive(l, 1)
        except e:
            print(e)


struct Shape[axes: Int]:
    var axes_sizes: StaticTuple[Int, axes]
    var ndim: Int
    var numels: Int

    fn __ne__(self, other: Self) -> Bool:
        return self.__eq__(other) == False

    fn __eq__(self, other: Self) -> Bool:
        length1 = len(self.axes_sizes)
        length2 = len(other.axes_sizes)
        if length1 != length2:
            return False
        for i in range(length1):
            if self.axes_sizes[i] != other.axes_sizes[i]:
                return False
        return self.ndim == other.ndim and self.numels == other.numels

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
