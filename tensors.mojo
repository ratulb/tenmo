fn test_broadcast_add_2_tensors() raises:
    print("Test broadcast add 2 tensors")

    tensor1 = Tensor.of(1, 2, 3, 4, requires_grad=True)
    tensor2 = Tensor.of(6, requires_grad=True)

    result = tensor1 + tensor2

    assert_true(
        (result == Tensor.of(7, 8, 9, 10)).all_true(),
        "broadcast add assertion 1 failed",
    )
    # result.backward()
    Tensor.walk_backward(result)

    assert_true(
        (
            tensor1.grad[]
            == Tensor.d1(
                [
                    1,
                    1,
                    1,
                    1,
                ]
            )
        ).all_true(),
        "grad check 1 - assertion failed",
    )

    assert_true(
        (tensor2.grad[] == Tensor.of([4])).all_true(),
        "grad check 2 - assertion failed",
    )

    tensor1 = Tensor.of[3](1, 2, 3, 4, 5, 6, requires_grad=True)
    tensor2 = Tensor.of(6, requires_grad=True)

    result = tensor1 + tensor2

    Tensor.walk_backward(result)

    assert_true(
        (result == Tensor.of[3](7, 8, 9, 10, 11, 12)).all_true(),
        "broadcast add assertion 2 failed",
    )

    assert_true(
        (
            tensor1.grad[]
            == Tensor.d2(
                [
                    [
                        1,
                        1,
                        1,
                    ],
                    [
                        1,
                        1,
                        1,
                    ],
                ]
            )
        ).all_true(),
        "grad check 3 - assertion failed",
    )

    assert_true(
        (tensor2.grad[] == Tensor.of([6])).all_true(),
        "grad check 4 - assertion failed",
    )

    tensor1 = Tensor.rand(
        2, 1, 4, 1, init_seed=Optional(42), requires_grad=True
    )
    tensor2 = Tensor.rand(2, 1, 5, init_seed=Optional(42), requires_grad=True)

    result = tensor1 + tensor2

    Tensor.walk_backward(result)

    assert_true(
        (
            tensor1.grad[]
            == Tensor.d4(
                [
                    [
                        [
                            [
                                10,
                            ],
                            [
                                10,
                            ],
                            [
                                10,
                            ],
                            [
                                10,
                            ],
                        ],
                    ],
                    [
                        [
                            [
                                10,
                            ],
                            [
                                10,
                            ],
                            [
                                10,
                            ],
                            [
                                10,
                            ],
                        ],
                    ],
                ]
            )
        ).all_true(),
        "grad check 5 - assertion failed",
    )

    assert_true(
        (
            tensor2.grad[]
            == Tensor.d3(
                [
                    [
                        [
                            8.0,
                            8.0,
                            8.0,
                            8.0,
                            8.0,
                        ],
                    ],
                    [
                        [
                            8.0,
                            8.0,
                            8.0,
                            8.0,
                            8.0,
                        ],
                    ],
                ]
            )
        ).all_true(),
        "grad check 6 - assertion failed",
    )


def main():
    test_broadcast_add_2_tensors()
    test_sum()
    test_reshape()
    test_scalar_tensor()
    test_tensor_of_list()
    test_add_2_tensors()
    test_item()
    test_arange()
    test_mul_by_factor()
    test_random()
    test_transpose_matmul()
    test_add_value()
    test_factor_mul_by()
    test_view()


### Mojo Tensor
### Implement tensor library in mojo from first principles

from math import iota, exp, floor
from random import seed, random_float64
from time import perf_counter_ns
from algorithm import vectorize
from sys import simdwidthof
from utils.numerics import max_finite
from os import abort
from memory import UnsafePointer, memcpy, memset, memset_zero
from shapes import Shape
from intlist import IntList
from ancestry import Ancestors
from common_utils import log_debug, piped
from operators import (
    __tensor_op_tensor__,
    AddTensor,
    SubtractTensor,
    MulTensor,
    __tensor_op_scalar__,
    AddScalar,
    SubtractScalar,
    MulScalar,
    sum_across_rows,
    sum_across_cols,
)
from collections import Set


struct Tensor[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable
):
    alias GradBox = UnsafePointer[Self]
    alias Address = UnsafePointer[Tensor[dtype]]
    var shape: Shape
    var data: UnsafePointer[Scalar[dtype]]
    var requires_grad: Bool
    var grad: Self.GradBox
    var ancestors: Ancestors[dtype]
    var grad_fn: Optional[fn () escaping raises -> None]

    fn __init__(out self, *axes_spans: Int, requires_grad: Bool = False):
        shape = Shape(axes_spans)
        self = Self(shape, requires_grad)

    fn __init__(out self, shape: Shape, requires_grad: Bool = False):
        Shape.validate(shape)
        self.shape = shape
        self.requires_grad = requires_grad
        self.ancestors = Ancestors[dtype].untracked()

        self.grad_fn = None
        self.grad = UnsafePointer[__type_of(self)]()
        if shape.ndim == 0:  # Tensor with Shape ()
            self.data = UnsafePointer[Scalar[self.dtype]].alloc(1)
        else:
            self.data = UnsafePointer[Scalar[self.dtype]].alloc(
                self.shape.num_elements()
            )
        self.init_gradbox()

    @staticmethod
    fn trace_ancestry[
        dtype: DType, //
    ](
        tensor: Tensor[dtype],
        mut visited: Set[Int],
        mut traced: Ancestors[dtype],
    ):
        if tensor.int_addr() not in visited:
            visited.add(Int(tensor.address()))
            for ancestor in tensor.ancestors:
                Self.trace_ancestry(ancestor[], visited, traced)
            traced.append(tensor.address())

    @staticmethod
    fn walk_backward[
        dtype: DType, //
    ](
        tensor: Tensor[dtype],
        start_grad: Scalar[dtype] = 1.0,
        verbose: Bool = False,
    ) raises:
        if tensor.has_grad() == False:
            return
        visited = Set[Int]()
        traced = Ancestors[dtype]()
        Self.trace_ancestry(tensor, visited, traced)
        tensor.grad[].fill(start_grad)
        for each in traced.__reversed__():
            each[].invoke_grad_fn(verbose)

    fn backward(
        self,
        start_grad: Scalar[dtype] = 1.0,
        start: Bool = True,
        verbose: Bool = False,
    ) raises:
        if start:
            self.grad[].fill(start_grad)
        self.invoke_grad_fn(verbose)
        for ancestor in self.ancestors:
            ancestor[].backward(start=False)

    fn grad_func(self) -> Optional[fn () escaping raises -> None]:
        return self.grad_fn

    @always_inline
    fn address(self) -> UnsafePointer[Self]:
        return UnsafePointer(to=self)

    @always_inline
    fn int_addr(self) -> Int:
        return Int(self.address())

    fn invoke_grad_fn(self, verbose: Bool = False) raises -> None:
        if self.grad_fn:
            if verbose:
                print("\nInvoking  grad_fn\n")
            self.grad_fn.value()()
        else:
            if verbose:
                print("\nNo grad_fn\n")
            pass

    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        if self.shape.ndim == 0 and len(indices) != 0:  # Tensor with Shape ()
            abort("Tensor -> __getitem__: Scalar tensor expects no indices")
        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__getitem__(indices): Invalid indices")
        return self.data.load[volatile=True](index)

    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        if self.shape.ndim == 0:  # Tensor with Shape ()
            abort(
                "Tensor -> __getitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __getitem__(IntList())"
            )

        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__getitem__(*indices): Invalid indices")
        return self.data.load[volatile=True](index)

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]):
        if self.shape.ndim == 0:  # Tensor with Shape ()
            abort(
                "Tensor -> __setitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __setitem__(IntList())"
            )
        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__setitem__(*indices): Invalid indices")
        self.data.store[volatile=True](index, value)

    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        if self.shape.ndim == 0 and len(indices) != 0:  # Tensor with Shape ()
            abort("Tensor -> __setitem__: Scalar tensor expects no indices")
        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__setitem__(IntList): Invalid indices")
        self.data.store[volatile=True](index, value)

    fn item(self) -> Scalar[self.dtype]:
        if (
            self.shape != Shape.Unit and self.shape.ndim != 0
        ):  # Tensor with Shape ()
            abort(
                "Tensor.item(): Only valid for scalar or singleton tensors, got"
                " shape: "
                + self.shape.__str__()
            )
        return self[0] if self.shape == Shape.Unit else self[IntList.Empty]

    fn __moveinit__(out self, owned other: Self):
        self.shape = other.shape
        self.data = UnsafePointer[Scalar[other.dtype]].alloc(other.numels())
        memcpy(self.data, other.data, other.numels())
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self.ancestors = other.ancestors
        self.grad_fn = other.grad_fn

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.data = UnsafePointer[Scalar[other.dtype]].alloc(other.numels())
        memcpy(self.data, other.data, other.numels())
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self.ancestors = other.ancestors
        self.grad_fn = other.grad_fn

    fn init_gradbox(mut self):
        if self.requires_grad and self.grad.__as_bool__() == False:
            gradients = Tensor[self.dtype](self.shape)
            self.grad = UnsafePointer[__type_of(self)].alloc(1)
            self.grad.init_pointee_move(gradients^)
            self.zero_grad()

    fn print_grad(self):
        if self.requires_grad == False:
            print("Requires grad? No.")
        elif self.requires_grad and self.has_grad() == False:
            print("Gradbox not initialized")
        else:
            self.grad[].print()

    fn open_gradbox(
        mut self,
    ) raises -> ref [self.grad[]] Tensor[self.dtype]:
        if self.requires_grad == False or self.has_grad() == False:
            err_s = String(
                "Requires grad is: {0}, gradbox: uninitialized? {1}"
            ).format(self.requires_grad, self.has_grad() == False)
            raise Error(err_s)
        return self.grad[]

    # fn __del__(owned self):
    fn free(owned self):
        if self.has_grad():
            for i in range(self.numels()):
                (self.data + i).destroy_pointee()
                (self.grad[].data + i).destroy_pointee()
            self.grad.free()
            log_debug(
                "Tensor__del__ -> freed grad(and pointees) and self data"
                " pointees"
            )
        else:
            for i in range(self.numels()):
                (self.data + i).destroy_pointee()
            log_debug("Tensor__del__ -> freed self data pointees")
        log_debug("Tensor__del__ -> discarded ancestors")
        self.ancestors.free()
        self.shape.free()
        if self.data:
            self.data.free()
        log_debug("Tensor__del__ -> called free on data")
        _ = self^

    fn __len__(self) -> Int:
        return self.numels()

    fn numels(self) -> Int:
        return self.shape.num_elements()

    fn ndim(self) -> Int:
        return self.shape.ndim

    @always_inline
    fn broadcastable(self, to: Tensor[dtype]) -> Bool:
        return self.shape.broadcastable(to.shape)

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

    fn all_close[
        simd_width: Int = simdwidthof[dtype](),
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Self) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "all_close is for floating point data types only",
        ]()

        if self.shape != other.shape:
            print(
                "all_close expects same shape 2D tensors: ",
                self.shape,
                ", ",
                other.shape,
            )

        num_elems = self.numels()
        simd_blocks = num_elems // simd_width
        remaining = num_elems % simd_width

        for i in range(simd_blocks):
            vector1 = self.data.load[width=simd_width](i * simd_width)
            vector2 = other.data.load[width=simd_width](i * simd_width)
            diff = abs(vector1 - vector2)
            tolerance = atol + rtol * abs(vector2)
            all_checks_out = (diff < tolerance).reduce_and()
            if all_checks_out == False:
                return False
        for k in range(remaining):
            value1 = self.data.load[width=1](simd_blocks * simd_width + k)
            value2 = other.data.load[width=1](simd_blocks * simd_width + k)
            value_diff = abs(value1 - value2)
            value_tolerance = atol + rtol * abs(value2)
            checks_out = value_diff < value_tolerance
            if checks_out == False:
                return False

        return True

    fn fill(self, value: Scalar[dtype]):
        @parameter
        fn set_value[simd_width: Int](idx: Int):
            self.data.store[width=simd_width](idx, value)

        vectorize[set_value, simdwidthof[dtype]()](self.numels())

    fn __eq__(self, other: Tensor[self.dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            abort(
                "Tensor __eq__ -> Dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        result = Tensor[DType.bool](self.shape, False)

        @parameter
        fn compare_elems[simd_width: Int](idx: Int):
            result.data.store[width=simd_width, volatile=True](
                idx,
                self.data.load[width=simd_width](idx)
                == other.data.load[width=simd_width](idx),
            )

        vectorize[compare_elems, simdwidthof[DType.bool]()](result.numels())
        return result

    fn __iadd__(self, other: Self) raises:
        if self.shape != other.shape:
            raise Error(
                "iadd -> Dimension mismatch: ", self.shape, ", ", other.shape
            )

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
            raise Error(
                "__ne__ -> Dimension mismatch: ", self.shape, ", ", other.shape
            )
        result = self == other

        @parameter
        fn invert[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, ~result.data.load[width=simd_width](idx)
            )

        vectorize[invert, simdwidthof[DType.bool]()](result.numels())
        return result

    fn grad_required(self) -> Bool:
        return self.requires_grad

    fn has_grad(self) -> Bool:
        return self.grad.__as_bool__() == True

    fn zero_grad(self):
        if self.grad_required() and self.has_grad():
            memset_zero(self.grad[].data, self.grad[].numels())

    fn add_ancestry(
        mut self,
        left: UnsafePointer[Tensor[dtype]],
        right: UnsafePointer[Tensor[dtype]] = UnsafePointer[Tensor[dtype]](),
    ):
        if right.__as_bool__():
            self.ancestors.add_ancestry(left[], right[])
        else:
            self.ancestors.add_ancestry(left[])

    fn __rmul__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__mul__(scalar)

    fn __mul__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, MulScalar](
            self.address()[], scalar
        )

        if self.address()[].requires_grad:

            fn grad_fn() raises -> None:
                out_grad_scaled = __tensor_op_scalar__[dtype, MulScalar](
                    out.address()[].grad[], scalar
                )
                self.address()[].grad[] = __tensor_op_tensor__[
                    dtype, AddTensor
                ](self.address()[].grad[], out_grad_scaled)

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self.address())

        return out

    # Element wise multiplication of two tensors
    fn __mul__(self, other: Self) raises -> Tensor[dtype]:
        if self.address()[].shape != other.address()[].shape:
            raise Error(
                "__mul__self * other -> Dimension mismatch:",
                self.address()[].shape,
                other.address()[].shape,
            )
        var out = __tensor_op_tensor__[dtype, MulTensor](
            self.address()[], other.address()[]
        )

        if self.address()[].requires_grad or other.address()[].requires_grad:

            fn grad_fn() raises -> None:
                out_grad = out.address()[].grad[]

                if self.address()[].requires_grad:
                    requires_grad_original = other.address()[].requires_grad
                    other.address()[].requires_grad = (
                        False  # Prevent requires_grad for grads
                    )
                    product = __tensor_op_tensor__[dtype, MulTensor](
                        out_grad, other.address()[]
                    )
                    other.address()[].requires_grad = requires_grad_original
                    self.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](self.address()[].grad[], product)

                if other.address()[].requires_grad:
                    requires_grad_original = self.address()[].requires_grad
                    self.address()[].requires_grad = False
                    product = __tensor_op_tensor__[dtype, MulTensor](
                        out_grad, self.address()[]
                    )
                    self.address()[].requires_grad = requires_grad_original
                    other.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](other.address()[].grad[], product)

            out.add_ancestry(self.address(), other.address())
            out.grad_fn = Optional(grad_fn)

        return out

    fn broadcast_to(self, target_shape: Shape) -> Tensor[dtype]:
        if not self.shape.broadcastable(target_shape):
            abort(
                "Tensor -> broadcast_to: shape "
                + self.shape.__str__()
                + " not broadcastable to "
                + target_shape.__str__()
            )

        mask = self.shape.broadcast_mask(target_shape)
        out = Tensor[dtype](target_shape, requires_grad=self.requires_grad)

        for idx in target_shape:
            src_idx = self.shape.translate_index(idx, mask, target_shape)
            out[idx] = self[src_idx]

        return out

    @always_inline
    fn broadcast_shape(self, other: Self) -> Shape:
        return Shape.broadcast_shape(self.shape, other.shape)

    @always_inline
    fn broadcast_mask(self, broadcast_shape: Shape) -> IntList:
        return self.shape.broadcast_mask(broadcast_shape)

    @always_inline
    fn translate_index(
        self, indices: IntList, mask: IntList, broadcast_shape: Shape
    ) -> IntList:
        return self.shape.translate_index(indices, mask, broadcast_shape)

    fn broadcast_add(self, other: Self) -> Tensor[dtype]:
        result_shape = self.broadcast_shape(other)
        # print("result_shape: ", result_shape)  # result_shape:  (4)
        mask1 = self.broadcast_mask(result_shape)
        # mask1.print()  # IntList[ 1 ] = 0
        mask2 = other.broadcast_mask(result_shape)
        # mask2.print()  # IntList[ 1 ] = 1
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor[dtype](result_shape, requires_grad=requires_grad)
        # result.print()  # [1D Tensor(4), Type: float32, requires_grad: True][0.0, 0.0, 0.0, 0.0,]
        for indices in result_shape:
            # indices.print() #  IntList[1] = 0 IntList[1] = 1  IntList[1] = 2 IntList[1] = 3
            self_indices = self.translate_index(indices, mask1, result_shape)
            other_indices = other.translate_index(indices, mask2, result_shape)
            # other_indices.print()  # IntList[1]=0 IntList[1]=0 IntList[1]=0 IntList[1]=0
            result[indices] = self[self_indices] + other[other_indices]

        if self.requires_grad or other.requires_grad:

            fn grad_fn() raises -> None:
                out_grad = result.address()[].grad[]

                if self.address()[].requires_grad:
                    grad_self = out_grad.sum(
                        sorted_axes=self.address()[]
                        .broadcast_mask(result.address()[].shape)
                        .indices_of(1),
                        keepdims=True,
                    ).reshape(self.address()[].shape)
                    self.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](self.address()[].grad[], grad_self)

                if other.address()[].requires_grad:
                    grad_other = out_grad.sum(
                        sorted_axes=other.address()[]
                        .broadcast_mask(result.address()[].shape)
                        .indices_of(1),
                        keepdims=True,
                    ).reshape(other.address()[].shape)
                    other.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](other.address()[].grad[], grad_other)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self.address(), other.address())
        return result

    fn __add__(self, other: Self) -> Tensor[dtype]:
        if self.address() == other.address():
            return self.__mul__(2)
        if not self.address()[].broadcastable(other.address()[]):
            abort(
                "__add__ -> Dimension mismatch: "
                + self.address()[].shape.__str__()
                + " <=> "
                + other.address()[].shape.__str__()
            )

        if self.address()[].shape != other.address()[].shape:
            return self.address()[].broadcast_add(other.address()[])

        var out = __tensor_op_tensor__[dtype, AddTensor](
            self.address()[], other.address()[]
        )

        if self.address()[].requires_grad or other.address()[].requires_grad:

            fn grad_fn() raises -> None:
                out_grad = out.address()[].grad[]
                if self.address()[].requires_grad:
                    self.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](self.address()[].grad[], out_grad)
                if other.address()[].requires_grad:
                    other.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](other.address()[].grad[], out_grad)

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self.address(), other.address())

        return out

    fn __radd__(self, scalar: Scalar[dtype]) raises -> Tensor[dtype]:
        return self.__add__(scalar)

    fn __add__(self, scalar: Scalar[dtype]) raises -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, AddScalar](
            self.address()[], scalar
        )

        if self.address()[].requires_grad:

            fn grad_fn() raises -> None:
                self_grad = self.address()[].grad[]
                out_grad = out.address()[].grad[]
                self.address()[].grad[] = __tensor_op_tensor__[
                    dtype, AddTensor
                ](self_grad, out_grad)

                print("in __add__(scalar) grad_fn")

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self.address())
        print("Out pointer address: ", out.address())

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

    fn __sub__(self, scalar: Scalar[dtype]) raises -> Self:
        var out = __tensor_op_scalar__[dtype, SubtractScalar](
            self.address()[], scalar
        )

        if self.address()[].requires_grad:

            fn grad_fn() raises -> None:
                self_grad = self.address()[].grad[]
                out_grad = out.address()[].grad[]
                self.address()[].grad[] = __tensor_op_tensor__[
                    dtype, AddTensor
                ](self_grad, out_grad)

                print("in __sub__(scalar) grad_fn")

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self.address())
        return out

    fn __sub__(self, other: Self) -> Tensor[dtype]:
        requires_grad = (
            self.address()[].requires_grad or other.address()[].requires_grad
        )
        if self.address() == other.address():
            out = Tensor[dtype].zeros_like(self.address()[], requires_grad)
            out.add_ancestry(self.address())
            return out
        if self.address()[].shape != other.address()[].shape:
            abort(
                "Tensor ->__sub__(other) - Dimension mismatch:"
                + self.address()[].shape.__str__()
                + ", "
                + other.address()[].shape.__str__()
            )

        out = __tensor_op_tensor__[dtype, SubtractTensor](
            self.address()[], other.address()[]
        )

        if requires_grad:

            fn grad_fn() raises -> None:
                out_grad = out.address()[].grad[]
                if self.address()[].requires_grad:
                    self.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](self.address()[].grad[], out_grad)
                if other.address()[].requires_grad:
                    other.address()[].grad[] = __tensor_op_tensor__[
                        dtype, SubtractTensor
                    ](other.address()[].grad[], out_grad)

            out.grad_fn = Optional(grad_fn)

            out.add_ancestry(self.address(), other.address())

        return out

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

    fn matmal(self, other: Self) -> Tensor[dtype]:
        start = perf_counter_ns()
        if self.shape[1] != other.shape[0]:
            abort("matmul - Dim mismatch")
        result = Tensor[dtype].zeros(self.shape[0], other.shape[1])
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                for k in range(self.shape[1]):
                    result[i, j] += self[i, k] * other[k, j]
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

    fn matmal_v2(self, other: Self) -> Tensor[dtype]:
        start = perf_counter_ns()
        if self.shape[1] != other.shape[0]:
            abort("matmul - Dim mismatch")

        result = Tensor[dtype].zeros(self.shape[0], other.shape[1])
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(other.shape[1]):
                    result[i, k] += self[i, j] * other[j, k]
        end = perf_counter_ns()
        print("Total: ", end - start)

        return result

    fn mat(self, other: Self) raises -> Tensor[dtype]:
        start = perf_counter_ns()
        if self.shape[1] != other.shape[0]:
            abort(
                "Tensor -> mat - Dim mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )

        result = Tensor[dtype].zeros(self.shape[0], other.shape[1])
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
        end = perf_counter_ns()
        print("Total: ", end - start)

        return result

    fn T(self, tile_size: Int = 32) raises -> Tensor[dtype]:
        if self.shape.ndim != 2:
            abort("Tensor -> transpose allowed only for 2D tensors")
        rows, cols = (self.shape[0], self.shape[1])
        output = Tensor[dtype](
            Shape(piped(cols, rows)), requires_grad=self.requires_grad
        )

        for i in range(0, rows, tile_size):
            for j in range(0, cols, tile_size):
                for ii in range(i, min(i + tile_size, rows)):
                    for jj in range(j, min(j + tile_size, cols)):
                        output[jj, ii] = self[ii, jj]

        return output

    @always_inline
    fn is_scalar(self) -> Bool:
        return self.numels() == 1 and self.shape == Shape.Void

    fn reshape(self, requires_grad: Bool = False) -> Tensor[dtype]:
        if self.is_scalar():
            return self
        if self.numels() != 1:
            abort(
                "Only tensor with single element can be reshaped to scalar"
                " tensor"
            )
        result = Tensor[dtype].scalar(self.data[], requires_grad)
        return result

    fn reshape(
        self, *newdims: Int, requires_grad: Bool = False
    ) -> Tensor[dtype]:
        return self.reshape(Shape(newdims), requires_grad)

    fn reshape(
        self, shape: Shape, requires_grad: Bool = False
    ) -> Tensor[dtype]:
        # shape = Shape(newdims)
        if shape == self.shape:
            return self
        if shape.num_elements() != self.numels():
            abort(
                "Tensor with "
                + String(self.numels())
                + " elements can't be converted to "
                + String(shape.num_elements())
                + " dimensional tensor"
            )
        result = Tensor[dtype](shape, requires_grad=requires_grad)
        memcpy(result.data, self.data, self.numels())
        return result

    fn sum(self, sorted_axes: IntList, keepdims: Bool = False) -> Tensor[dtype]:
        _rank = self.shape.rank()

        for axis in sorted_axes:
            if axis < 0 or axis >= _rank:
                abort(
                    "Tensor -> sum - invalid axis in sum: "
                    + String(axis)
                    + " for tensor with shape: "
                    + self.shape.__str__()
                )

        spans = IntList.with_capacity(_rank)

        for i in range(_rank):
            if i in sorted_axes:
                if keepdims:
                    spans.append(1)
                else:
                    continue
            else:
                spans.append(self.shape[i])

        out_shape = Shape(spans)

        var out = Tensor[dtype].zeros(
            out_shape, requires_grad=self.requires_grad
        )

        reduced_shape = Shape(self.shape.axes_spans.select(sorted_axes))
        for out_idx in out_shape:
            var summ = Scalar[dtype](0)

            for red_idx in reduced_shape:
                if keepdims:
                    full_idx = out_idx.replace(sorted_axes, red_idx)
                else:
                    full_idx = out_idx.insert(sorted_axes, red_idx)

                summ += self[full_idx]

            out[out_idx] = summ

        if self.requires_grad:

            fn grad_fn() raises -> None:
                out_grad = out.address()[].grad[]
                expanded = out_grad
                if not keepdims:
                    expanded = out_grad.reshape(
                        Shape(
                            out_grad.shape.intlist().insert(
                                sorted_axes,
                                IntList.with_capacity(len(sorted_axes), 1),
                            )
                        )
                    )

                self.address()[].grad[] = __tensor_op_tensor__[
                    dtype, AddTensor
                ](
                    self.address()[].grad[],
                    expanded.broadcast_to(self.address()[].shape),
                )

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self.address())

        return out

    fn sum(
        self, axes: List[Int] = [-1], keepdims: Bool = False
    ) -> Tensor[dtype]:
        _rank = self.shape.rank()

        sorted_axes = IntList(_rank - 1) if (
            len(axes) == 0 or axes == [-1]
        ) else IntList.new(axes).sorted()
        return self.sum(sorted_axes, keepdims)

    _ = """fn sum(self, axis: Int = -1, keepdim: Bool = False) -> Tensor[dtype]:
        _axis = axis
        if _axis != -1:
            if _axis < 0 or _axis >= self.ndim():
                abort("Invalid axis for tensor sum: " + String(_axis))
        else:
            _axis = self.ndim() - 1

        if self.ndim() == 1:
            result = Tensor[dtype].zeros(1, requires_grad=self.requires_grad)

            @parameter
            fn sum_elems[simd_width: Int](idx: Int):
                result[0] += self.data.load[width=simd_width](idx).reduce_add()

            vectorize[sum_elems, simdwidthof[dtype]()](self.numels())
            return result

        #elif self.ndim() == 2:
            #out = sum_across_rows(self) if _axis == 1 else sum_across_cols(self)
            #return out

        else:
            shape = self.shape
            out_shape = shape.replace(_axis, 1) if keepdim else shape.drop_axis(
                _axis
            )
            out = Tensor[dtype].zeros(
                out_shape, requires_grad=self.requires_grad
            )
            for indices in out_shape:  # all indices of output tensor
                sum_val = Scalar[dtype](0)
                for i in range(self.shape[_axis]):
                    full_idx = indices.replace(
                        _axis, i
                    ) if keepdim else indices.insert(_axis, i)
                    sum_val += self[full_idx]
                out[indices] = sum_val

            return out"""

    fn __str__(self) -> String:
        dims = len(self.shape)
        s = String("[")
        if dims == 1:
            s += "1D Tensor"
        elif dims == 2:
            s += "2D Tensor"
        elif dims == 3:
            s += "3D Tensor"
        elif dims == 4:
            s += "4D Tensor"
        elif dims == 5:
            s += "5D Tensor"
        else:
            s += "Tensor"
        s += self.shape.__str__()
        s += ", Type: " + self.dtype.__str__()
        s += ", requires_grad: " + String(self.requires_grad)
        s += "]"
        return s

    @staticmethod
    fn rand(
        *axes_spans: Int,
        min: Scalar[dtype] = 0,
        max: Scalar[dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        shape = Shape(axes_spans)
        tensor = Tensor[dtype](shape, requires_grad)
        # randn(tensor.data, tensor.numels())
        for i in range(tensor.numels()):  # vectorize?
            tensor.data.store[volatile=True](
                i,
                random_float64(
                    min.cast[DType.float64](), max.cast[DType.float64]()
                ).cast[dtype](),
            )
        return tensor

    @staticmethod
    fn arange[
        dtype: DType = DType.float32
    ](
        start: Scalar[dtype] = 0,
        end: Scalar[dtype] = max_finite[dtype](),
        step: Scalar[dtype] = 1,
        epsilon: Scalar[dtype] = 1e-8,
    ) -> Tensor[dtype]:
        if step == 0:
            abort("step can not be zero")
        if (step > 0 and start >= end) or (step < 0 and start <= end):
            abort("Invalid range for the given step")
        delta = end - start
        size = floor(delta / step + epsilon)
        if size <= 0:
            abort("Error: computed arange size is zero")
        tensor = Tensor[dtype](size.__int__())

        @parameter
        fn fill(i: Int) -> Scalar[dtype]:
            return (i * step + start) % end

        @parameter
        fn mapper[simd_width: Int](idx: Int):
            first_entry = fill(idx).cast[dtype]()
            data = SIMD[dtype, simd_width](first_entry)
            for i in range(1, simd_width):
                data[i] = fill(idx + i).cast[dtype]()
            tensor.data.store[width=simd_width](idx, data)

        vectorize[mapper, simdwidthof[dtype]()](tensor.numels())

        return tensor

    @staticmethod
    fn zeros(*axes_spans: Int, requires_grad: Bool = False) -> Tensor[dtype]:
        shape = Shape(axes_spans)
        return Self.zeros(shape, requires_grad)

    @staticmethod
    fn zeros_like(
        tensor: Tensor[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        out = Tensor[dtype](tensor.shape, requires_grad)
        memset_zero(out.data, out.numels())
        return out

    @staticmethod
    fn zeros(shape: Shape, requires_grad: Bool = False) -> Tensor[dtype]:
        out = Tensor[dtype](shape, requires_grad)
        memset_zero(out.data, out.numels())
        return out

    alias Row = List[Scalar[dtype]]

    @staticmethod
    fn d1(row: Self.Row, requires_grad: Bool = False) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "d1")
        shape = Shape(IntList(len(row)))
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, row.data, len(row))
        return tensor

    @staticmethod
    fn d2(rows: List[Self.Row], requires_grad: Bool = False) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "d2")
        dims = IntList(len(rows), len(rows[0]))
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for row in rows:
            if len(row) != dims[1]:
                abort("Tensor -> d2 -> not all rows equal in length")
            flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    alias Rows = List[Self.Row]

    @staticmethod
    fn d3(
        blocks: List[Self.Rows], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "d3")
        dims = IntList(len(blocks), len(blocks[0]), len(blocks[0][0]))
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for block in blocks:
            if len(block) != dims[1]:
                abort("Tensor -> d3 -> not all blocks equal in length")
            for row in block:
                if len(row) != dims[2]:
                    abort("Tensor -> d3 -> not all rows equal in length")

                flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    alias Block = List[Self.Rows]

    @staticmethod
    fn d4(
        blockgrid: List[Self.Block], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "d4")
        dims = IntList(
            len(blockgrid),
            len(blockgrid[0]),
            len(blockgrid[0][0]),
            len(blockgrid[0][0][0]),
        )
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for block in blockgrid:
            if len(block) != dims[1]:
                abort(
                    "Tensor -> d4 -> not all blocks are of equal length in the"
                    " blockgrid"
                )
            for matrix in block:
                if len(matrix) != dims[2]:
                    abort(
                        "Tensor -> d4 -> not all matrices are of equal length"
                        " in block"
                    )
                for row in matrix:
                    if len(row) != dims[3]:
                        abort(
                            "Tensor -> d4 not all rows are of equal length in"
                            " matrix"
                        )
                    flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    alias Blocks = List[Self.Block]

    @staticmethod
    fn d5(
        blockhive: List[Self.Blocks], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "d5")
        dims = IntList(
            len(blockhive),
            len(blockhive[0]),
            len(blockhive[0][0]),
            len(blockhive[0][0][0]),
            len(blockhive[0][0][0][0]),
        )
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for blocks in blockhive:
            if len(blocks) != dims[1]:
                abort(
                    "Tensor -> d5 -> not all blocks are of equal length in the"
                    " input"
                )
            for block in blocks:
                if len(block) != dims[2]:
                    abort("Tensor -> d5 -> unequal block length")
                for matrix in block:
                    if len(matrix) != dims[3]:
                        abort(
                            "Tensor -> d5 not all matrices are of equal length"
                            " in block"
                        )
                    for row in matrix:
                        if len(row) != dims[4]:
                            abort(
                                "Tensor -> d5 not all rows are of equal length"
                                " in matrix"
                            )
                        flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    @staticmethod
    fn of(*elems: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "of")
        shape = Shape(piped(len(elems)))
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(len(elems)):
            tensor[i] = elems[i]
        return tensor

    @staticmethod
    fn of[
        row_size: Int
    ](*elems: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "of[row_size]")

        if not (row_size >= 1 and row_size <= len(elems)):
            abort(
                (
                    "Tensor -> of[row_size] -> invalid row size or not enough"
                    " elements"
                ),
            )
        num_rows = len(elems) // row_size
        axes_spans = piped(num_rows, row_size)
        shape = Shape(axes_spans)
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(num_rows):
            for j in range(row_size):
                tensor[i, j] = elems[i * row_size + j]
        return tensor

    @staticmethod
    fn scalar(val: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        result = Tensor[dtype](Shape.Void, requires_grad=requires_grad)
        result[IntList.Empty] = val
        return result

    @staticmethod
    fn ones(*axes_spans: Int, requires_grad: Bool = False) -> Tensor[dtype]:
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

    @staticmethod
    fn validate_dtype_consistency(
        dtype: DType, requires_grad: Bool, label: String
    ):
        if requires_grad:
            if not (dtype.is_floating_point()):
                abort(
                    "Tensor → "
                    + label
                    + " → requires_grad=True is only supported for floating"
                    " point types. "
                )

    fn print_tensor_recursive(
        self,
        mut indices: IntList,
        level: Int,
        num_first: Int = 10,
        num_last: Int = 10,
    ) raises:
        try:
            if self.ndim() == 0:  # Tensor with Shape ()
                print(self[IntList.Empty])
                return
            current_dim = len(indices)
            indent = " " * (level * 2)
            # Defensive check
            if current_dim >= self.ndim():
                # if current_dim > self.ndim():
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
                # print("\n")

        except e:
            print("ERROR during tensor printing: ", e)

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        print(self.__str__())
        empty = IntList()
        try:
            self.print_tensor_recursive(
                empty, 1, num_first=num_first, num_last=num_last
            )
        except e:
            print(e)

    @staticmethod
    fn free_all[dtype: DType, //](*tensors: Tensor[dtype]):
        for each in tensors:
            each.free()

    fn view(ref self) -> View[__origin_of(self), self.dtype]:
        return View(Pointer(to=self))


@fieldwise_init
struct View[
    mutability: Bool, //,
    origin: Origin[mutability],
    dtype: DType = DType.float32,
]:
    var target: Pointer[Tensor[dtype], origin]


from testing import assert_true


fn test_add_2_tensors() raises:
    print("test_add_2_tensors")

    tensor1 = Tensor.rand(1024, 1024, requires_grad=True)
    tensor2 = Tensor.rand(1024, 1024, requires_grad=True)
    assert_true(
        tensor1.shape == tensor2.shape,
        "Input tensors shape match assertion failed",
    )
    out_tensor = tensor1 + tensor2
    assert_true(
        tensor1.shape == out_tensor.shape,
        "Input/output tensors shape match assertion failed",
    )
    print("Tensor1 grad shape: ", tensor1.open_gradbox().shape)
    print("Tensor2 grad shape: ", tensor2.open_gradbox().shape)
    print("Out tensor grad shape: ", out_tensor.open_gradbox().shape)
    parent1 = out_tensor.ancestors.get(0)[]
    parent2 = out_tensor.ancestors.get(1)[]
    left_parent_is_tensor1 = (parent1 == tensor1).all_true()
    right_parent_is_tensor2 = (parent2 == tensor2).all_true()
    assert_true(
        left_parent_is_tensor1 == True and right_parent_is_tensor2 == True,
        "Output tensor ancestry validation failed",
    )
    print("The following is out tensor gradient")
    out_tensor.open_gradbox().print()
    print("Before invoking grad_fn")
    print("Tensor1 grad shape: ", tensor1.open_gradbox().shape)
    print("Tensor2 grad shape: ", tensor2.open_gradbox().shape)
    print("Out tensor grad shape: ", out_tensor.open_gradbox().shape)

    out_tensor.invoke_grad_fn()
    out_tensor.free()
    tensor1.free()
    tensor2.free()
    Tensor.free_all(tensor1, tensor1, out_tensor)


fn test_factor_mul_by() raises:
    print("test_factor_mul_by")

    tensor = Tensor.rand(256, 256, requires_grad=True)
    out_tensor = 2 * tensor
    assert_true(
        len(out_tensor.ancestors.get(0)[]) == 65536,
        "Output tensor ancestors length validation failed",
    )
    out_tensor.invoke_grad_fn()
    print("The following is out tensor gradient")
    out_tensor.open_gradbox().print()
    Tensor.free_all(tensor, out_tensor)


fn test_mul_by_factor() raises:
    print("test_mul_by_factor")
    tensor = Tensor.rand(128, 256, requires_grad=True)
    out_tensor = tensor * 100
    assert_true(
        len(out_tensor.ancestors.get(0)[]) == 32768,
        "Output tensor ancestors length validation failed",
    )
    out_tensor.invoke_grad_fn()
    print("The following is out tensor gradient")
    out_tensor.open_gradbox().print()
    Tensor.free_all(tensor, out_tensor)


fn test_add_value() raises:
    print("test_add_value")

    tensor = Tensor.rand(1024, 64, requires_grad=True)
    out_tensor = 2 * tensor
    assert_true(
        len(out_tensor.ancestors.get(0)[]) == 65536,
        "Output tensor ancestors length validation failed",
    )
    out_tensor.invoke_grad_fn()
    print("The following is out tensor gradient")
    out_tensor.open_gradbox().print()
    Tensor.free_all(tensor, out_tensor)


fn test_arange() raises:
    tensor = Tensor.arange(0, 10)
    # expected = Tensor.of(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    expected = Tensor.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    # print(tensor.dtype, expected.dtype)
    is_true = (tensor == expected).all_true()
    assert_true(is_true, "arange gen check assertion failed")

    Tensor.free_all(tensor, expected)

    tensor1 = Tensor.arange(0, -5, -0.5)
    # expected = Tensor[DType.float32].of(
    expected = Tensor.of(
        0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5
    ).to_dtype[DType.float32]()
    is_true = (tensor1 == expected).all_true()
    assert_true(is_true, "arange negative step assertion failed")
    Tensor.free_all(tensor1, expected)


fn test_transpose_matmul() raises:
    A = Tensor.rand(3, 3)
    A_T = A.T()
    A.print()
    A_T.print()
    B = Tensor.rand(3, 3)
    C = A.matmal(B)
    D = A.mat(B)
    R = C == D
    C.print()
    D.print()
    R.print()
    assert_true(C.all_close(D), "Matmal and at implementations are not same")
    Tensor.free_all(A, A_T, B, C, D)
    Tensor.free_all(R)
    Tensor.free_all(R)


fn test_random() raises:
    rand_tensor = Tensor.rand(10)
    rand_tensor.print()

    fn each(e: Scalar[DType.float32]) -> Bool:
        return e >= 0 and e < 1

    holds_true = rand_tensor.for_all(each)
    assert_true(holds_true, "rand min and max range assertion failed")

    rand_tensor2 = Tensor.rand(10, 20, min=-2, max=2)

    fn each2(e: Scalar[DType.float32]) -> Bool:
        return e >= -2 and e < 2

    holds_true = rand_tensor2.for_all(each2)
    assert_true(holds_true, "rand min(-2) and max(2) range assertion failed")
    Tensor.free_all(rand_tensor, rand_tensor2)


fn test_sum() raises:
    ones = Tensor.ones(3, 3)
    summed = ones.sum(axes=[0], keepdims=True)
    assert_true(
        (summed == Tensor.d2([[3, 3, 3]])).all_true(),
        "keepdim = True sum assertion 1 failed",
    )
    ones = Tensor.ones(3, 3)
    summed = ones.sum(axes=[0])
    expect = Tensor.of(3, 3, 3)
    assert_true((summed == expect).all_true(), "1D sum assertion failed")

    tensor = Tensor.arange(1, 21).reshape(2, 5, 2)
    summed = tensor.sum(axes=[1])
    _ = """[2D Tensor(2, 2), Type: float32, requires_grad: False]
        [
            [25.0, 30.0, ],
            [75.0, 80.0, ],
    ]"""
    expect = Tensor.of[2](25, 30, 75, 80)
    assert_true(
        (summed == expect).all_true(), "Sum across axis 1 assertion failed"
    )

    summed = tensor.sum(axes=[0])
    expect = Tensor.of[2](12, 14, 16, 18, 20, 22, 24, 26, 28, 30)
    assert_true(
        (summed == expect).all_true(), "Sum across axis 0 assertion failed"
    )

    expect = Tensor.of[5](3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
    summed = tensor.sum()
    assert_true(
        (summed == expect).all_true(), "Sum across axis 2 assertion failed"
    )


fn test_item() raises:
    tensor = Tensor.of(42)
    assert_true(tensor.item() == 42)


fn test_view() raises:
    tensor = Tensor.rand(2, 4, 5)
    view = tensor.view()
    assert_true(
        tensor.shape == view.target[].shape,
        "Tensor and view shape equality asserttion failed",
    )


fn test_tensor_of_list() raises:
    tensor = Tensor.d1([1, 3, 4, 5])
    assert_true(
        tensor.numels() == 4 and tensor.dtype == DType.float32,
        "Tensor from list assertion 1 failed",
    )
    tensor_int32 = Tensor[DType.int32].d1([1, 3, 4, 5])
    assert_true(
        tensor_int32.numels() == 4 and tensor_int32.dtype == DType.int32,
        "Tensor from list assertion 2 failed",
    )
    tensor_2d = Tensor.d2(
        List(List(1.0, 2, 3), List(4.0, 5, 6), List(7.0, 8, 9))
    )
    assert_true(
        tensor_2d.shape == Shape.of(3, 3) and tensor_2d.numels() == 9,
        "Tensor from assertion 3 failed",
    )
    tensor2d = Tensor.d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert_true(
        tensor2d.shape == Shape.of(3, 3) and tensor2d.numels() == 9,
        "Tensor from assertion 3 failed",
    )

    tensor3d = Tensor.d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert_true(
        tensor3d.shape == Shape.of(2, 2, 2) and tensor3d.numels() == 8,
        "Tensor from assertion 4 failed",
    )


fn test_scalar_tensor() raises:
    tensor = Tensor.scalar(42)
    assert_true(
        (
            tensor.item() == 42.0
            and tensor.shape == Shape.Void
            and tensor.numels() == 1
        ),
        "Scalar tensor item and shape assertion failed",
    )


fn test_reshape() raises:
    tensor = Tensor.rand(3, 3)
    reshaped = tensor.reshape(9)
    assert_true(
        tensor[2, 2] == reshaped[8], "reshape __getitem__ assertion failed"
    )
    tensor = Tensor.of(42)
    assert_true(tensor.shape == Shape.Unit, "Tensor shape assertion failure")
    reshaped = tensor.reshape(1, 1)
    assert_true(
        reshaped.shape == Shape.of(1, 1) and reshaped[0, 0] == tensor[0],
        "post reshape shape and get assertion failed",
    )
    tensor = Tensor.scalar(42)
    reshaped = tensor.reshape(1, 1)
    assert_true(
        reshaped.shape == Shape.of(1, 1) and reshaped[0, 0] == tensor.item(),
        "post reshape shape and get assertion failed for scalar tensor",
    )
    reshaped = tensor.reshape(1)
    assert_true(
        reshaped.shape == Shape.Unit and reshaped[0] == tensor.item(),
        "post reshape 2 - shape and get assertion failed for scalar tensor",
    )
    tensor = Tensor.rand(1, 1)
    reshaped = tensor.reshape()
    assert_true(
        reshaped.shape == Shape.Void and reshaped.item() == tensor[0, 0],
        "post reshape random tensor - shape and get assertion failed",
    )
