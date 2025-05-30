### Mojo Tensor
### Implement tensor library in mojo from first principles

from math import iota, exp
from random import randn, seed
from time import perf_counter_ns
from algorithm import vectorize
from sys import simdwidthof
from memory import UnsafePointer, memcpy, memset, memset_zero
from shapes import Shape
from common_utils import (
    int_varia_list_to_str,
    validate_shape,
)
from operators import GradFn, AddScalar

from testing import assert_true


struct Tensor[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable
):
    # Gradients are float32
    var shape: Shape
    var data: UnsafePointer[Scalar[dtype]]
    var requires_grad: Bool
    var grad: UnsafePointer[Self]
    var op: Optional[StaticString]
    var parents: List[UnsafePointer[Tensor[dtype], origin=MutableAnyOrigin]]
    var grad_fn: GradFn[dtype]

    fn __init__(out self, *axes_spans: Int, requires_grad: Bool = False) raises:
        shape = Shape(axes_spans)
        self = Self(shape, requires_grad)

    fn __init__(out self, shape: Shape, requires_grad: Bool = False) raises:
        self.shape = shape
        validate_shape(shape)
        self.requires_grad = requires_grad
        self.op = None
        self.parents = List[
            UnsafePointer[Tensor[dtype], origin=MutableAnyOrigin]
        ]()
        self.grad_fn = GradFn[dtype]()
        self.grad = UnsafePointer[__type_of(self)]()
        self.data = UnsafePointer[Scalar[self.dtype]].alloc(
            self.shape.num_elements()
        )

    fn grad_func(self) -> GradFn[dtype]:
        return self.grad_fn

    fn __getitem__(self, indices: List[Int]) raises -> Scalar[dtype]:
        index = self.shape.flatten_index(indices)
        if index == -1:
            raise Error("__getitem__(indices): Invalid indices")
        # return (self.data + index)[]
        return self.data.load[volatile=True](index)

    fn __getitem__(self, *indices: Int) raises -> Scalar[dtype]:
        index = self.shape.flatten_index(indices)
        if index == -1:
            raise Error("__getitem__(*indices): Invalid indices")
        # return (self.data + index)[]
        # return self.data.load(index)
        return self.data.load[volatile=True](index)

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]) raises:
        index = self.shape.flatten_index(indices)
        if index == -1:
            raise Error("__setitem__(*indices): Invalid indices")
        # (self.data + index)[] = value
        self.data.store[volatile=True](index, value)

    fn __moveinit__(out self, owned other: Self):
        print("Move is being called?")
        self.shape = other.shape
        self.data = UnsafePointer[Scalar[dtype]].alloc(other.numels())
        memcpy(self.data, other.data, other.numels())
        # self.datatype = other.datatype
        self.requires_grad = other.requires_grad
        self.grad = UnsafePointer[__type_of(other)]()  # Not moving grad yet
        # self.grad = UnsafePointer[__type_of(other)].alloc(1)
        # other.grad.move_pointee_into(self.grad)  # moving grad
        # other.grad.free()
        self.op = other.op
        """self.parents = List[
            UnsafePointer[Tensor[dtype], origin=MutableAnyOrigin]
        ](
            capacity=0
        )"""
        self.parents = other.parents
        self.grad_fn = other.grad_fn
        #Self.transfer_grad(self, other, False)
        #other.data.free()
        if other.grad_tensor_initialized():
            #other.grad.free()
            pass

    @staticmethod
    fn transfer_grad[
        dtype: DType, //
    ](mut this: Tensor[dtype], that: Tensor[dtype], copy: Bool) raises:
        print("Transfer is being called?")
        if (
            that.grad_tensor_initialized()
            and not this.grad_tensor_initialized()
        ):
            print("Transfer inner 1 is being called?")
            this.init_grad_tensor()
            if this.grad_tensor_initialized():
                if copy:
                    print("Getting called from copy")
                    this.grad.init_pointee_copy(that.grad[])
                else:
                    print("I was called")
                    that.grad.move_pointee_into(this.grad)
                    that.data.free()
                    that.grad.free()

    fn __copyinit__(out self, other: Self):
        print("Copy is being called?")
        self.shape = other.shape
        self.data = UnsafePointer[Scalar[dtype]].alloc(other.numels())
        memcpy(self.data, other.data, other.numels())
        self.requires_grad = other.requires_grad
        self.grad = UnsafePointer[__type_of(other)]()  # Not copying grad
        # self.grad = other.grad^  # moving grad
        self.op = other.op
        _="""self.parents = List[
            UnsafePointer[Tensor[dtype], origin=MutableAnyOrigin]
        ](
            capacity=0
        )"""
        self.parents = other.parents
        self.grad_fn = other.grad_fn
        print(
            "Copy is being called?",
            self.requires_grad,
            self.grad.__as_bool__(),
            other.grad_tensor_initialized(),
        )
        #Self.transfer_grad(self, other, True)

    fn __del__(owned self):
        self.data.free()

    fn __len__(self) -> Int:
        return self.numels()

    @always_inline
    fn numels(self) -> Int:
        return self.shape.num_elements()

    @always_inline
    fn ndim(self) -> Int:
        return self.shape.ndim

    fn all_true(self: Tensor[DType.bool]) -> Bool:
        fn all_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == True

        return self.for_all(all_truthy)

    fn any_true(self: Tensor[DType.bool]) -> Bool:
        fn any_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == True

        return self.any(any_truthy)

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

    fn any[
        simd_width: Int = simdwidthof[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        num_elems = self.numels()
        simd_blocks = num_elems // simd_width
        remaining = num_elems % simd_width

        for i in range(simd_blocks):
            vector = self.data.load[width=simd_width](i * simd_width)
            for j in range(simd_width):
                if pred(vector[j]):
                    return True
        for k in range(remaining):
            if pred(self.data.load[width=1](simd_blocks * simd_width + k)):
                return True
        return False

    fn __eq__(self, other: Self) raises -> Tensor[DType.bool]:
        if self.shape != other.shape:
            raise Error("eq -> Dimension mismatch")
        copy = Tensor[DType.bool](self.shape, False)

        @parameter
        fn compare_elems[simd_width: Int](idx: Int):
            copy.data.store[width=simd_width, volatile=True](
                idx,
                self.data.load[width=simd_width](idx)
                == other.data.load[width=simd_width](idx),
            )

        vectorize[compare_elems, simdwidthof[DType.bool]()](copy.numels())
        return copy

    fn __add__(self, other: Self) raises -> Tensor[dtype]:
        if self.shape != other.shape:
            raise Error("add -> Dimension mismatch")

        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor[dtype](self.shape, requires_grad)

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

    fn __iadd__(self, other: Self) raises:
        if self.shape != other.shape:
            raise Error("iadd -> Dimension mismatch")

        @parameter
        fn add_elems[simd_width: Int](idx: Int):
            self.data.store[width=simd_width](
                idx,
                (
                    self.data.load[width=simd_width](idx)
                    + other.data.load[width=simd_width](idx)
                ),
            )

        vectorize[add_elems, simdwidthof[dtype]()](self.numels())

    fn exp(self) raises -> Tensor[dtype]:
        requires_grad = self.requires_grad
        result = Tensor[dtype](self.shape, requires_grad)

        @parameter
        fn exp_elems[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, exp(self.data.load[width=simd_width](idx))
            )

        vectorize[exp_elems, simdwidthof[dtype]()](result.numels())
        return result

    fn __ne__(self, other: Self) raises -> Tensor[DType.bool]:
        if self.shape != other.shape:
            raise Error("ne -> Dimension mismatch")
        result = self == other

        @parameter
        fn invert[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, ~result.data.load[width=simd_width](idx)
            )

        vectorize[invert, simdwidthof[DType.bool]()](result.numels())
        return result

    @always_inline
    fn grad_required(self) -> Bool:
        return self.requires_grad

    @always_inline
    fn grad_tensor_initialized(self) -> Bool:
        return self.grad.__as_bool__()

    fn zero_grad(self):
        print("ok - zero grading coming")
        if self.grad_required() and self.grad_tensor_initialized():
            print("ok - zero grading")
            memset_zero(self.grad[].data, self.numels())

    fn init_grad_tensor(mut self) raises:
        if self.grad_required() and not self.grad_tensor_initialized():
            gradients = Tensor[dtype](self.shape)
            self.grad = UnsafePointer[__type_of(self)].alloc(1)
            self.grad.init_pointee_move(gradients^)
            print("This should be printed")
            self.zero_grad()

    @staticmethod
    fn _init_grad_(
        mut tensor: Tensor[dtype],
        shape: Shape,
    ) raises:
        if tensor.requires_grad and not tensor.grad:
            # Gradients are float32
            gradients = Tensor[dtype](shape)
            tensor.grad = UnsafePointer[__type_of(gradients)].alloc(1)
            tensor.grad.init_pointee_move(gradients)
            tensor.zero_grad()
            Tensor.print(tensor.grad[])

    fn __mul__(mut self, factor: Scalar[dtype]) raises -> Tensor[dtype]:
        out = self

        @parameter
        fn mul_by_factor[simd_width: Int](idx: Int):
            out.data.store[width=simd_width](
                idx, out.data.load[width=simd_width](idx) * factor
            )

        vectorize[mul_by_factor, simdwidthof[dtype]()](out.numels())

        out.requires_grad = self.requires_grad
        out.parents.append(UnsafePointer(to=self))

        _ = """fn grad_fn() raises -> None:
            if self.requires_grad:
                # Self._init_grad_(self, self.shape)
                self.init_grad_tensor()
                # Self._init_grad_(out, out.shape)
                out.init_grad_tensor()
                print("in _backward")
                self.grad[] += out.grad[] * factor
                Tensor.print(self.grad[])"""

        # out.grad_fn = Optional(grad_fn)
        return out

    fn __add__(mut self, value: Scalar[dtype]) raises -> Tensor[dtype]:
        #out = self
        out = Tensor[dtype](self.shape, self.requires_grad)

        @parameter
        fn add_value[simd_width: Int](idx: Int):
            out.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx) + value
            )

        vectorize[add_value, simdwidthof[dtype]()](out.numels())

        out.requires_grad = self.requires_grad
        out.parents.append(UnsafePointer(to=self))
        print("Out tensor parents len: ", len(out.parents))
        if self.requires_grad:
            grad_func = GradFn[dtype](
                AddScalar,
                UnsafePointer(to=self),
                UnsafePointer(to=out),
                value,
            )
            out.grad_fn = grad_func
        _ = """fn grad_fn() raises -> None:
            if self.requires_grad:
                # out.requires_grad = True
                # out.parents.append(UnsafePointer(to=self))
                self.init_grad_tensor()
                out.init_grad_tensor()
                print("add value in _backward")
                self.grad[] += out.grad[]
                Tensor.print(self.grad[])"""

        return out

    fn __iadd__(self, value: Scalar[dtype]):
        @parameter
        fn add_value[simd_width: Int](idx: Int):
            self.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx) + value
            )

        vectorize[add_value, simdwidthof[dtype]()](self.numels())

    fn to_dtype[NewType: DType](self) raises -> Tensor[NewType]:
        result = Tensor[NewType](self.shape, self.requires_grad)

        @parameter
        fn cast_values[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx).cast[NewType]()
            )

        vectorize[cast_values, simdwidthof[NewType]()](result.numels())
        return result

    fn __sub__(self: Self, value: Scalar[dtype]) -> Self:
        copy = self

        @parameter
        fn subtract_value[simd_width: Int](idx: Int):
            copy.data.store[width=simd_width](
                idx, copy.data.load[width=simd_width](idx) - value
            )

        vectorize[subtract_value, simdwidthof[dtype]()](copy.numels())
        return copy

    fn __truediv__(self, factor: Scalar[dtype]) -> Tensor[dtype]:
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

    fn matmal(self, other: Self) raises -> Tensor[dtype]:
        start = perf_counter_ns()
        from testing import assert_equal

        result = Tensor[dtype].zeros(self.shape[0], other.shape[1])
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

    fn matmal_v2(self, other: Self) raises -> Tensor[dtype]:
        start = perf_counter_ns()
        from testing import assert_equal

        result = Tensor[dtype].zeros(self.shape[0], other.shape[1])
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

    fn matmal_v3(self, other: Self) raises -> Tensor[dtype]:
        start = perf_counter_ns()
        from testing import assert_equal

        result = Tensor[dtype].zeros(self.shape[0], other.shape[1])
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

    fn reshape(self, *newdims: Int) raises -> Tensor[dtype]:
        shape = Shape(newdims)
        if shape == self.shape:
            return self
        if shape.num_elements() != self.numels():
            raise Error(
                "Tensor with "
                + String(self.numels())
                + " elements can't be converted to "
                + int_varia_list_to_str(newdims)
                + " dimensional tensor"
            )
        result = Tensor[dtype](shape, self.requires_grad)

        @parameter
        fn copy_elements[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx)
            )

        vectorize[copy_elements, simdwidthof[dtype]()](self.numels())
        return result

    fn __str__(self) -> String:
        dims = self.ndim()
        s = String("[")
        if dims == 1:
            s += "1D Tensor"
        elif dims == 2:
            s += "2D Tensor"
        elif dims == 3:
            s += "3D Tensor"
        elif dims == 4:
            s += "4D Tensor"
        else:
            s += "Unsupported Tensor"
        s += self.shape.__str__()
        s += ", Type: " + self.dtype.__str__()
        s += ", requires_grad: " + String(self.requires_grad)
        s += "]"
        return s

    @staticmethod
    fn rand(
        *axes_spans: Int,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) raises -> Tensor[dtype]:
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        shape = Shape(axes_spans)
        tensor = Tensor[dtype](shape, requires_grad)
        randn(tensor.data, tensor.numels())
        return tensor

    @staticmethod
    fn arange[
        end: Int,
        start: Int = 0,
        datatype: DType = DType.int64,
        length: Int = end - start,
    ](requires_grad: Bool = False) raises -> Tensor[datatype]:
        constrained[
            end > start and end - start == length,
            (
                "arange -> invalid parameters - end should be > start and end -"
                " start = length"
            ),
        ]()
        shape = Shape(length)
        result = Tensor[dtype=datatype](shape, requires_grad)
        # print(result.dtype, __type_of(result[0]).__str__(result[0]))
        # print(__type_of(result).__str__(result))
        iota(result.unsafe_ptr(), length, offset=start)

        # casted = result.unsafe_ptr().bitcast[Scalar[datatype]]()
        # memcpy(result.unsafe_ptr(), casted, result.numels())
        return result

    @staticmethod
    fn zeros(
        *axes_spans: Int, requires_grad: Bool = False
    ) raises -> Tensor[dtype]:
        shape = Shape(axes_spans)
        tensor = Tensor[dtype](shape, requires_grad)
        memset_zero(tensor.data, tensor.numels())
        return tensor

    @staticmethod
    fn ones(
        *axes_spans: Int, requires_grad: Bool = False
    ) raises -> Tensor[dtype]:
        tensor = Tensor[dtype](Shape(axes_spans), requires_grad)
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
            num_first = 5
            num_last = 5
            _ = self.shape.__str__()
            # Defensive check
            if current_dim >= self.ndim():
                print(
                    "ERROR: current_dim (",
                    current_dim,
                    ") >= ndim (",
                    self.ndim(),
                    ")",
                )
                return

            size = self.shape[current_dim]

            # Size sanity check
            if size < 0 or size > 1_000_000:
                print(
                    "ERROR: suspicious size: ",
                    size,
                    "at dim ",
                    current_dim,
                    self.shape.__str__(),
                )
                return

            # Base case: last dimension (print actual elements)
            if current_dim == self.ndim() - 1:
                print(indent + "[", end="")

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
                    elif i >= size - num_last:
                        indices.append(i)
                        print(self[indices], end=", " if i != size - 1 else "")
                        _ = indices.pop()
                    else:
                        # Handles middle region not explicitly caught
                        continue

                print("]", end="")

            else:
                print(indent + "[")
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
                    else:
                        # This path was previously missing, which caused silent looping!
                        continue

                print(indent + "]", end="")
                print("\n")

        except e:
            print("ERROR during tensor printing: ", e)

    _ = """fn print_tensor_recursive(self, mut indices: List[Int], level: Int) raises:
        try:
            current_dim = len(indices)
            indent = " " * (level * 2)

            num_first = 5  # Show first 5 elements
            num_last = 5  # Show last 5 elements

            if current_dim == self.ndim() - 1:
                print(indent + "[", end="")
                size = self.shape[current_dim]
                print(self.shape.__str__(), "size: ", size)

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
                    else:
                        #print("Am I stuck here - of course")
                        continue
                        #print(self.shape.__str__())
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
                print("I am stuck here........")
                print("\n")
        except e:
            print(e)
        print("Out of print........")

    @staticmethod
    fn print(t: Tensor):
        print(t.__str__())
        print()
        l = List[Int]()
        try:
            t.print_tensor_recursive(l, 1)
        except e:
            print(e)"""

    fn print(self):
        print(self.__str__())
        print()
        empty = List[Int]()
        try:
            self.print_tensor_recursive(empty, 1)
        except e:
            print(e)


def main():
    # tensor = Tensor.rand(4, 3, 2, 1)
    tensor = Tensor.rand(2, 3, requires_grad=True)
    tensor.print()
    out_tensor = tensor + 2
    print("Out tensor parents len in main: ", len(out_tensor.parents))
    # tensor.grad_func()()
    out_tensor.print()
    out_tensor.grad_func()()
    print("Tensor.grad: ")
    print("Out tensor requires grad: ", out_tensor.requires_grad)
    tensor.grad[].print()
    print(
        "I want grad here: ",
        out_tensor.grad.__as_bool__(),
        "requires_grad",
        out_tensor.requires_grad,
        len(out_tensor.parents),
    )
    # if out_tensor.grad_fn:
    out_tensor.grad[].print()
    # out_tensor.grad_fn.value()()
    # multiplied.grad_fn.value()()
    # output = out_tensor * 2
    # output.grad[] += 1
    # output.grad_func()()
    # print(output.grad.__as_bool__())

    # Tensor.print(Tensor.arange(7, start=3).reshape[2](2, 2))
    # Tensor.print(Tensor.arange(7, start=3).reshape[2](2, 2))
    # tensor = Tensor.arange[5]().to_dtype[DType.float32]()
    # l = List[Int]()
    # tensor.print_tensor_recursive(l, 1)
    # tensor.print()
    # tensor.print()
    # Tensor.print(tensor)
    # tensor1 = Tensor.arange[4]()
    # tensor1.print()
    # tensor1.print()

    _ = """# tensor._init_grad_()
    print(multiplied.dtype)
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

    tensor = Tensor.rand(4, 3)
    print("Original")
    Tensor.print(tensor)
    reshaped = tensor.reshape(2, 2, 3)
    print("Reshaped")
    Tensor.print(reshaped)

    tensor_false = Tensor.zeros(4, 3)
    indices = List[Int]()
    tensor_false.print_tensor_recursive(indices, 1)

    tensor_true = Tensor.ones(4, 3)
    indices = List[Int]()
    tensor_true.print_tensor_recursive(indices, 1)

    tensor = Tensor.ones(4, 3)
    indices = List[Int]()
    tensor.print_tensor_recursive(indices, 1)

    t16 = Tensor.zeros(5, 5)
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

    other = Tensor.zeros(5, 5)
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

    tensor_big1 = Tensor.rand(1024, 4096)
    tensor_big2 = Tensor.rand(4096, 512)

    # Tensor.print(tensor_big1.matmal_v3(tensor_big2))
    """
