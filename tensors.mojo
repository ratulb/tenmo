# %s/^\(fn test_\(.*\)() raises:\)$/&\r    print("test_\2")/
from testing import assert_true


fn test_scalar_addition() raises:
    print("test_scalar_addition")
    var a = Tensor.scalar(3.0, requires_grad=True)
    var b = Tensor.scalar(4.0, requires_grad=True)
    var c = a + b
    c.backward()
    assert_true(c.item() == 7.0)
    assert_true(a.grad[].item() == 1.0)
    assert_true(b.grad[].item() == 1.0)


fn test_broadcast_addition() raises:
    print("test_broadcast_addition")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var b = Tensor.d1([10, 20], requires_grad=True)
    var c = a + b  # shape (2,2)
    s = c.sum()
    s.backward()
    assert_true((c == Tensor.d2([[11, 22], [13, 24]])).all_true())
    assert_true(a.grad[].all_close(Tensor.d2([[1, 1], [1, 1]])))
    assert_true(
        b.grad[].all_close(Tensor.d1([2, 2]))
    )  # Summed over broadcast dim


fn test_sum_all_dims() raises:
    print("test_sum_all_dims")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum()  # scalar
    s.backward()
    assert_true(s.item() == 10.0)
    assert_true(a.grad[].all_close(Tensor.d2([[1, 1], [1, 1]])))


fn test_sum_specific_axis() raises:
    print("test_sum_specific_axis")
    var a = Tensor.d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    var s = a.sum(axes=[1], keepdims=True)  # shape (2,1,2)
    s.backward()
    assert_true((s == Tensor.d3([[[4, 6]], [[12, 14]]])).all_true())
    assert_true(a.grad[].all_close(Tensor.ones_like(a)))


fn test_mean_with_keepdims() raises:
    print("test_mean_with_keepdims")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var m = a.mean(axes=[0], keepdims=True)  # shape (1,2)
    print("Ok here1")
    m.print()
    s = m.sum()
    s.print()
    print("Ok here2")
    s.backward()
    print("Ok here3")
    assert_true(m.all_close(Tensor.d2([[2, 3]])))
    assert_true(a.grad[].all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])))
    s.free()
    m.free()
    a.free()


fn test_matmul_shapes() raises:
    print("test_matmul_shapes")
    # Test various matmul shape combinations
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var b = Tensor.d2([[5, 6], [7, 8]], requires_grad=True)
    var c = a.matmul(b)
    # c.sum().backward()
    assert_true(c.all_close(Tensor.d2([[19, 22], [43, 50]])))
    # assert_true(a.grad[].all_close(b.sum(axes=[0], keepdims=True).T()))
    # assert_true(b.grad[].all_close(a.sum(axes=[1], keepdims=True)))


fn test_matmul_broadcasting() raises:
    print("test_matmul_broadcasting")
    # Batch matmul
    var a = Tensor.d3([[[1, 2]], [[3, 4]]], requires_grad=True)  # shape (2,1,2)
    var b = Tensor.d3([[[5], [6]]], requires_grad=True)  # shape (1,2,1)
    var c = a.matmul(b)  # shape (2,2,1)
    # c.sum().backward()
    assert_true(c.all_close(Tensor.d3([[[17], [39]], [[23], [53]]])))


fn test_nested_operations() raises:
    print("test_nested_operations")
    var a = Tensor.d1([1, 2], requires_grad=True)
    var b = Tensor.d1([3, 4], requires_grad=True)
    # var c = (a * b).sum() + (a + b).prod()
    # c.backward()
    # Verify gradients numerically
    assert_true(abs(a.grad[][0] - 11.0) < 1e-6)  # 3 + (3+4)*1
    assert_true(abs(b.grad[][0] - 8.0) < 1e-6)  # 1 + (1+2)*1


_ = """fn test_detach() raises:
    print("test_detach")
    var a = Tensor.d1([1,2], requires_grad=True)
    var b = a.detach() * 2  # Should not propagate grad
    var c = a * b
    c.sum().backward()
    assert_true(a.grad[].all_close(Tensor.d1([2,4])))  # Only from c = a*b"""

_ = """fn test_empty_tensor() raises:
    print("test_empty_tensor")
    var a = Tensor.d1([], requires_grad=True)
    var s = a.sum()
    s.backward()
    assert_true(s.item() == 0.0)
    assert_true(a.grad[].shape == Shape.of(0))"""


fn test_zero_grad() raises:
    print("test_zero_grad")
    var a = Tensor.scalar(1.0, requires_grad=True)
    var b = a * 2
    # b.backward()
    a.zero_grad()
    assert_true(a.grad[].item() == 0.0)


fn test_transpose_grad() raises:
    print("test_transpose_grad")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var b = a.T()
    var c = b * Tensor.d2([[10, 30], [20, 40]])
    # c.sum().backward()
    assert_true(a.grad[].all_close(Tensor.d2([[10, 20], [30, 40]])))


_ = """fn test_slice_grad() raises:
    print("test_slice_grad")
    var a = Tensor.d1([1,2,3,4], requires_grad=True)
    var b = a[1:3]  # [2,3]
    var c = b * Tensor.d1([10,20])
    c.sum().backward()
    assert_true(a.grad[].all_close(Tensor.d1([0,10,20,0])))"""


_ = """fn test_large_tensor_backprop() raises:
    print("test_large_tensor_backprop")
    # Test memory efficiency
    var a = Tensor.rand(500, 128, requires_grad=True)
    var b = Tensor.rand(128, 100, requires_grad=True)
    var c = a.matmul(b).sum()
    c.backward()
    assert_true(a.grad[].shape == a.shape)
    assert_true(b.grad[].shape == b.shape)"""


fn main() raises:
    test_mean_with_keepdims()
    _="""test_scalar_addition()
    test_broadcast_addition()
    test_sum_all_dims()
    test_sum_specific_axis()"""

    _ = """var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var b = Tensor.d1([10, 20], requires_grad=True)
    print("Inputs a, b: ", a.id(), b.id())
    var c = a + b  # shape (2,2)
    s = c.sum()
    # print("Before all s.addr and c.addr: ", s.address(), c.address())
    print("Before all c.id: ", c.id())
    # s.backward()
    Tensor.walk_backward(s.into_tensorlike())
    #_= a^
    #_= b^
    #_= c^
    assert_true((c == Tensor.d2([[11, 22], [13, 24]])).all_true())
    assert_true(a.grad[].all_close(Tensor.d2([[1, 1], [1, 1]])))
    assert_true(
        b.grad[].all_close(Tensor.d1([2, 2]))
    )  # Summed over broadcast dim"""

    _ = """test_scalar_addition()
    test_broadcast_addition()
    test_sum_all_dims()
    test_sum_specific_axis()
    test_mean_with_keepdims()
    test_matmul_shapes()
    test_zero_grad()"""
    # test_matmul_broadcasting()
    # test_nested_operations()
    # test_large_tensor_backprop()


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
from views import TensorView
from strides import Strides
from shared import TensorLike, Differentiable
from common_utils import log_debug, piped, is_null
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
    Power,
    scalar_ops,
    Add,
    Subtract,
    Multiply,
)
from summ import SumGradFn

# from collections import Set
from graphs import Graph


struct Tensor[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable & Differentiable
):
    alias datatype: DType = dtype
    alias Row = List[Scalar[dtype]]
    alias Rows = List[Self.Row]
    alias Block = List[Self.Rows]
    alias Blocks = List[Self.Block]
    var shape: Shape
    var data: UnsafePointer[Scalar[dtype]]
    var requires_grad: Bool
    var grad: UnsafePointer[Self]
    var ancestors: Ancestors[dtype]
    var grad_fn: Optional[fn () escaping raises -> None]
    var base: UnsafePointer[Tensor[dtype]]  # Only allocated on need basis

    fn __init__(out self, *axes_spans: Int, requires_grad: Bool = False):
        shape = Shape(axes_spans)
        self = Self(shape, requires_grad)

    fn __init__(out self, row: Self.Row, requires_grad: Bool = False):
        self = Self.d1(row, requires_grad=requires_grad)

    fn __init__(
        out self,
        shape: Shape,
        data: UnsafePointer[Scalar[dtype]],
        requires_grad: Bool = False,
    ):
        Shape.validate(shape)
        self.shape = shape
        self.requires_grad = requires_grad
        self.ancestors = Ancestors[dtype].untracked()
        self.grad_fn = None
        self.grad = UnsafePointer[__type_of(self)]()
        self.base = UnsafePointer[Tensor[dtype]]()
        self.data = data
        self.init_grad()

    fn is_tensor(self) -> Bool:
        return True

    fn is_view(self) -> Bool:
        return False

    fn into_view(self) -> TensorView[dtype]:
        abort("Tensor -> into_view(self) - not supported")
        return TensorView[dtype].Blank

    fn into_tensor(self) -> Tensor[dtype]:
        return self

    fn __init__(out self, shape: Shape, requires_grad: Bool = False):
        Shape.validate(shape)
        self.shape = shape
        self.requires_grad = requires_grad
        self.ancestors = Ancestors[dtype].untracked()
        self.base = UnsafePointer[Tensor[dtype]]()
        self.grad_fn = None
        self.grad = UnsafePointer[__type_of(self)]()
        if shape.ndim == 0:  # Tensor with Shape ()
            self.data = UnsafePointer[Scalar[self.dtype]].alloc(1)
        else:
            self.data = UnsafePointer[Scalar[self.dtype]].alloc(
                self.shape.num_elements()
            )
        self.init_grad()

    fn is_contiguous(self) -> Bool:
        return True

    fn into_tensorlike(self) -> TensorLike[dtype]:
        return TensorLike[dtype](self.address())

    fn backward(self):
        graph = Graph[dtype]()
        graph.walk_backward(self)

    _ = """@staticmethod
    fn trace_ancestry[
        dtype: DType, //
    ](
        node: TensorLike[dtype],
        mut visited: IntList,
        mut traced: Ancestors[dtype],
    ):
        if node.id() not in visited:
            visited.append(node.id())
            var ptr = UnsafePointer[TensorLike[dtype]].alloc(1)
            ptr.init_pointee_copy(node)
            traced.append(ptr)
            msg = String(
                "DAG node inner id: "
                + String(node.inner_id())
                + " => kind:"
                + "Tensor" if ptr[].is_tensor() else "View"
            )
            log_debug(msg)

            for ancestor in node.tensor().ancestors:
                Self.trace_ancestry(ancestor[], visited, traced)

    @staticmethod
    fn walk_backward(

        node: TensorLike[dtype],
        start_grad: Scalar[dtype] = 1.0,
        verbose: Bool = False,
    ) raises:
        traced = Ancestors[dtype]()
        visited = IntList()
        Self.trace_ancestry(node, visited, traced)
        print("Printing traced")
        traced.print()
        seen_ids = Set[Int]()
        for each in traced:
            id = each[].inner_id()
            if id in seen_ids:
                print("Duplicate TensorLike id in DAG:", id)
            seen_ids.add(id)
            print("About to call grad_fn for", id)
            ptr = each
            if not ptr.__as_bool__():
                print("Null pointer found!")
                continue

            node_ = ptr[]
            print(
                "About to call grad_fn for", node_.inner_id()
            )  # <-- Match this against `traced.print()`

            if each[].is_view():
                print("→ It's a TensorView")
                v = each[].view()
                if v.base_tensor[].grad_fn:
                    # Mojo recommended way for checking Optional
                    print("  grad_fn present")
                    print("Calling grad_fn on view id:", each[].inner_id())
                    try:
                        each[].invoke_grad_fn(verbose)
                    except e:
                        print("grad_fn threw error:", e)
                else:
                    print("Skipping empty grad_fn on view id:", each[].inner_id())
            else:
                print("→ It's a Tensor")
                t = each[].tensor()
                if t.grad_fn:
                    print("  grad_fn present")
                    print("Calling grad_fn on tensor id:", each[].inner_id())
                    try:
                        #each[].invoke_grad_fn(verbose)
                        t.grad_fn.value()()
                    except e:
                        print("🔥 grad_fn threw error:", e)
                else:
                    print("Skipping empty grad_fn on tensor id:", each[].inner_id())"""

    fn ancestry(self) -> Ancestors[dtype]:
        return self.ancestors

    fn grad_func(self) -> Optional[fn () escaping raises -> None]:
        return self.grad_fn

    fn view(self, shape: Shape) -> TensorView[dtype]:
        if shape == self.shape and self.is_contiguous():
            return TensorView(
                UnsafePointer(to=self),
                self.shape,
                Strides.default(self.shape),
                offset=0,  # or self.offset if needed
            )
        if not self.shape.num_elements() == shape.num_elements():
            abort("Mismatch in elements for view")
        return TensorView(
            UnsafePointer(to=self), shape, Strides.default(shape), offset=0
        )

    fn address(self) -> UnsafePointer[Tensor[dtype]]:
        return UnsafePointer(to=self)

    fn id(self) -> Int:
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
            abort("Tensor → __getitem__: Scalar tensor expects no indices")
        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__getitem__(indices): Invalid indices")
        return self.data.load[volatile=True](index)

    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        if self.shape.ndim == 0:  # Tensor with Shape ()
            abort(
                "Tensor → __getitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __getitem__(IntList())"
            )

        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__getitem__(*indices): Invalid indices")
        return self.data.load[volatile=True](index)

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]):
        if self.shape.ndim == 0:  # Tensor with Shape ()
            abort(
                "Tensor → __setitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __setitem__(IntList())"
            )
        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__setitem__(*indices): Invalid indices")
        self.data.store[volatile=True](index, value)

    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        if self.shape.ndim == 0 and len(indices) != 0:  # Tensor with Shape ()
            abort("Tensor → __setitem__: Scalar tensor expects no indices")
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
        self.base = other.base
        self.ancestors = other.ancestors
        self.grad_fn = other.grad_fn

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.data = UnsafePointer[Scalar[other.dtype]].alloc(other.numels())
        memcpy(self.data, other.data, other.numels())
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self.base = other.base
        self.ancestors = other.ancestors
        self.grad_fn = other.grad_fn
        self.init_grad()

    fn copy(self) -> Self:
        result = Tensor[dtype](self.shape, requires_grad=self.requires_grad)
        memcpy(result.data, self.data, self.numels())
        if result.requires_grad:
            memcpy(result.grad, self.grad, self.numels())
        return result

    fn init_grad(mut self):
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

    # fn __del__(owned self):
    fn free(owned self):
        if self.has_grad():
            for i in range(self.numels()):
                (self.data + i).destroy_pointee()
                (self.grad[].data + i).destroy_pointee()
            self.grad.free()
            log_debug(
                "Tensor__del__ → freed grad(and pointees) and self data"
                " pointees"
            )
        else:
            for i in range(self.numels()):
                (self.data + i).destroy_pointee()
            log_debug("Tensor__del__ → freed self data pointees")
        log_debug("Tensor__del__ → discarded ancestors")
        self.ancestors.free()
        self.shape.free()
        if self.data:
            self.data.free()
        log_debug("Tensor__del__ → called free on data")
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

    fn seed_grad(self, value: Scalar[dtype]):
        if self.has_grad():
            self.grad[].fill(value)

    fn fill(self, value: Scalar[dtype]):
        @parameter
        fn set_value[simd_width: Int](idx: Int):
            self.data.store[width=simd_width](idx, value)

        vectorize[set_value, simdwidthof[dtype]()](self.numels())

    fn __eq__(self, other: Tensor[self.dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            abort(
                "Tensor __eq__ → Dimension mismatch: "
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
                "iadd → Dimension mismatch: ", self.shape, ", ", other.shape
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

    fn exp(self) -> Tensor[dtype]:
        requires_grad = self.requires_grad
        result = Tensor[dtype](self.shape, requires_grad)

        @parameter
        fn exp_elems[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, exp(self.data.load[width=simd_width](idx))
            )

        vectorize[exp_elems, simdwidthof[dtype]()](result.numels())
        return result

    fn __neg__(self) -> Tensor[dtype]:
        requires_grad = self.requires_grad
        result = Tensor[dtype](self.shape, requires_grad)

        @parameter
        fn negate_elems[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx).__neg__()
            )

        vectorize[negate_elems, simdwidthof[dtype]()](result.numels())
        return result

    fn __ne__(self, other: Self) raises -> Tensor[DType.bool]:
        if self.shape != other.shape:
            raise Error(
                "__ne__ → Dimension mismatch: ", self.shape, ", ", other.shape
            )
        result = self == other

        @parameter
        fn invert[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, ~result.data.load[width=simd_width](idx)
            )

        vectorize[invert, simdwidthof[DType.bool]()](result.numels())
        return result

    fn has_grad(self) -> Bool:
        return self.requires_grad and self.grad.__as_bool__() == True

    fn _requires_grad(self) -> Bool:
        return self.requires_grad

    fn grad_is_zero(self) -> Bool:
        if not self.requires_grad:
            abort(
                "Tensor → grad_is_zero: checking grad on a tensor that does"
                " have grad"
            )

        fn all_zero(val: Scalar[dtype]) -> Bool:
            return val == Scalar[dtype](0)

        return self.has_grad() and self.grad[].for_all(all_zero)

    fn zero_grad(self):
        if self.requires_grad and self.has_grad():
            memset_zero(self.grad[].data, self.grad[].numels())

    fn add_ancestry(mut self, *parents: Tensor[dtype]):
        for parent in parents:
            stable = parent.into_tensorlike()
            var ptr = UnsafePointer[TensorLike[dtype]].alloc(1)
            ptr.init_pointee_move(stable)
            self.ancestors.append(ptr)
            try:
                msg = String(
                    "DAG node inner id: {0}, self id: {1} => kind: "
                    + "Tensor" if ptr[].is_tensor() else "View"
                ).format(
                    stable.inner_id(), self.id()
                )  # Critical compiler issue c = a + b, s = c.sum(), results in s.id() == b.id() if self.id() is not printed!
                log_debug(msg)
            except e:
                print(e)

    fn __rmul__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__mul__(scalar)

    fn __mul__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, MulScalar](
            self,
            scalar,
        )

        if self.requires_grad:

            fn grad_fn() raises -> None:
                out_grad_scaled = __tensor_op_scalar__[dtype, MulScalar](
                    out.address()[].grad[], scalar
                )
                self.address()[].update_grad[AddTensor](out_grad_scaled)

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self)

        return out

    fn __pow__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, Power](
            self,
            scalar,
        )

        if self.requires_grad:

            fn grad_fn() raises -> None:
                self_powed_one_less = __tensor_op_scalar__[dtype, Power](
                    self.address()[], scalar - 1
                )
                self_powed_one_less_scaled = __tensor_op_scalar__[
                    dtype, MulScalar
                ](self_powed_one_less, scalar)

                product = __tensor_op_tensor__[dtype, MulTensor](
                    out.address()[].grad[], self_powed_one_less_scaled
                )
                self.address()[].update_grad[AddTensor](product)

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self)

        return out

    fn broadcast_to(self, target_shape: Shape) -> Tensor[dtype]:
        if not self.shape.broadcastable(target_shape):
            abort(
                "Tensor → broadcast_to: shape "
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

    _ = """@staticmethod
    fn grad_unreduced(
        tensor: Tensor[dtype], upstream_grad: Tensor[dtype]
    ) -> Tensor[dtype]:
        upstream_grad_shape = upstream_grad.shape
        tensor_view = tensor.view(upstream_grad_shape)
        result = Tensor[dtype](upstream_grad_shape)
        for indices in upstream_grad_shape:
            result[indices] = upstream_grad[indices] * tensor_view[indices]
        return result"""

    @always_inline
    fn broadcast_mask(self, broadcast_shape: Shape) -> IntList:
        return self.shape.broadcast_mask(broadcast_shape)

    @always_inline
    fn translate_index(
        self, indices: IntList, mask: IntList, broadcast_shape: Shape
    ) -> IntList:
        return self.shape.translate_index(indices, mask, broadcast_shape)

    fn broadcast_op(
        self,
        other: Self,
        op: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    ) -> Tensor[dtype]:
        if self.shape.rank() == 0 or other.shape.rank() == 0:
            return self.broadcast_scalar_op(other, op)
        else:
            return self.broadcast_tensor_op(other, op)

    fn broadcast_scalar_op(
        self,
        other: Self,
        op: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    ) -> Tensor[dtype]:
        # Decide result shape
        result_shape = other.shape if self.shape.rank() == 0 else self.shape
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor[dtype](result_shape, requires_grad=requires_grad)

        for indices in result_shape:
            self_val = self.item() if self.shape.rank() == 0 else self[indices]
            other_val = (
                other.item() if other.shape.rank() == 0 else other[indices]
            )
            result[indices] = op(self_val, other_val)

        return result

    fn broadcast_tensor_op(
        self,
        other: Self,
        op: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    ) -> Tensor[dtype]:
        result_shape = Shape.broadcast_shape(self.shape, other.shape)
        mask1 = self.broadcast_mask(result_shape)
        mask2 = other.broadcast_mask(result_shape)
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor[dtype](result_shape, requires_grad=requires_grad)

        for indices in result_shape:
            self_indices = self.translate_index(indices, mask1, result_shape)
            other_indices = other.translate_index(indices, mask2, result_shape)
            result[indices] = op(self[self_indices], other[other_indices])

        return result

    fn update_grad[op: Int](self, incoming: Tensor[dtype]):
        self.grad[] = __tensor_op_tensor__[dtype, op](self.grad[], incoming)

    fn broadcast_operation[
        element_wise_op: Int, tensor_op_first: Int, tensor_op_second: Int
    ](self, other: Self) -> Tensor[dtype]:
        result = self.broadcast_op(other, scalar_ops[dtype, element_wise_op])
        if self.requires_grad or other.requires_grad:
            self_ptr = self.address()
            that_ptr = other.address()

            fn grad_fn() raises -> None:
                this = self_ptr[]
                that = that_ptr[]

                if this.requires_grad:
                    upstream_grad = result.address()[].grad[]
                    grad_contrib = this.backward_grad_contrib(
                        that, upstream_grad, False
                    )

                    this.update_grad[tensor_op_first](grad_contrib)

                if that.requires_grad:
                    upstream_grad = result.address()[].grad[]
                    grad_contrib = that.backward_grad_contrib(
                        this, upstream_grad, False
                    )

                    that.update_grad[tensor_op_second](grad_contrib)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self, other)
        return result

    fn backward_grad_contrib(
        self,
        other: Tensor[dtype],
        upstream_grad: Tensor[dtype],
        do_multiply: Bool,
    ) -> Tensor[dtype]:
        var grad_contrib: Tensor[dtype]

        if upstream_grad.shape == Shape.Void:
            grad_contrib = Tensor[dtype].full(
                self.shape, upstream_grad.item(), requires_grad=False
            )
        else:
            grad_contrib = (
                upstream_grad * other if do_multiply else upstream_grad
            )
            if grad_contrib.shape != self.shape:
                axes = self.broadcast_mask(grad_contrib.shape).indices_of(1)
                print("Reaching here alright 1", axes)
                grad_contrib = grad_contrib.sum(axes=axes, keepdims=True)
                print("Reaching here alright 2", axes)
                grad_contrib = grad_contrib.sum(axes=axes, keepdims=True)
                print("Reaching here alright 3", axes)
            if grad_contrib.shape != self.shape:
                print("Reaching here alright 4", grad_contrib.shape, self.shape)
            if grad_contrib.shape != self.shape:
                grad_contrib = grad_contrib.reshape(self.shape)
            grad_contrib.requires_grad = False

        return grad_contrib

    fn broadcast_mul(
        self: Self,
        other: Self,
    ) -> Tensor[dtype]:
        result = self.broadcast_op(other, scalar_ops[dtype, Multiply])
        requires_grad = self.requires_grad or other.requires_grad
        if requires_grad:

            fn grad_fn() raises -> None:
                this = self.address()[]
                that = other.address()[]
                output = result.address()[]
                upstream_grad = output.grad[]
                if this.requires_grad:
                    grad_contrib = this.backward_grad_contrib(
                        that, upstream_grad, True
                    )
                    this.update_grad[AddTensor](grad_contrib)
                if that.requires_grad:
                    grad_contrib = that.backward_grad_contrib(
                        this, upstream_grad, True
                    )
                    that.update_grad[AddTensor](grad_contrib)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self, other)

        return result

    fn __add__(self, other: Self) -> Tensor[dtype]:
        if self.address() == other.address():
            return self.__mul__(2)
        if not self.broadcastable(other):
            abort(
                "__add__ → Dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__()
            )

        if self.shape != other.shape:
            return self.broadcast_operation[Add, AddTensor, AddTensor](
                other,
            )

        var out = __tensor_op_tensor__[dtype, AddTensor](self, other)

        if self.requires_grad or other.requires_grad:

            fn grad_fn() raises -> None:
                out_grad = out.address()[].grad[]
                if self.address()[].requires_grad:
                    self.address()[].update_grad[AddTensor](out_grad)
                if other.address()[].requires_grad:
                    other.address()[].update_grad[AddTensor](out_grad)

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self, other)

        return out

    fn __radd__(self, scalar: Scalar[dtype]) raises -> Tensor[dtype]:
        return self.__add__(scalar)

    fn __add__(self, scalar: Scalar[dtype]) raises -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, AddScalar](
            self,
            scalar,
        )
        if self.requires_grad:

            fn grad_fn() raises -> None:
                out_grad = out.address()[].grad[]
                self.address()[].update_grad[AddTensor](out_grad)

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self)

        return out

    fn __iadd__(self, value: Scalar[dtype]):
        @parameter
        fn add_value[simd_width: Int](idx: Int):
            self.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx) + value
            )

        vectorize[add_value, simdwidthof[dtype]()](self.numels())

    fn float(self) -> Tensor[DType.float32]:
        if self.dtype == DType.float32:
            return rebind[Tensor[DType.float32]](self)
        return self.to_dtype[DType.float32]()

    fn float64(self) -> Tensor[DType.float64]:
        if self.dtype == DType.float64:
            return rebind[Tensor[DType.float64]](self)
        return self.to_dtype[DType.float64]()

    fn to_dtype[NewType: DType](self) -> Tensor[NewType]:
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
                out_grad = out.address()[].grad[]
                self.address()[].update_grad[AddTensor](out_grad)

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self)
        return out

    # Element wise multiplication of two tensors
    fn __mul__(self, other: Self) -> Tensor[dtype]:
        if not self.broadcastable(other):
            abort(
                "__mul__(self * other) → Dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__()
            )

        if self.shape != other.shape:
            return self.broadcast_mul(other)

        var out = __tensor_op_tensor__[dtype, MulTensor](
            self,
            other,
        )

        if self.requires_grad or other.requires_grad:

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
                    self.address()[].update_grad[AddTensor](product)

                if other.address()[].requires_grad:
                    requires_grad_original = self.address()[].requires_grad
                    self.address()[].requires_grad = False
                    product = __tensor_op_tensor__[dtype, MulTensor](
                        out_grad, self.address()[]
                    )
                    self.address()[].requires_grad = requires_grad_original
                    other.address()[].update_grad[AddTensor](product)

            out.add_ancestry(self, other)
            out.grad_fn = Optional(grad_fn)

        return out

    fn __sub__(self, other: Self) -> Tensor[dtype]:
        if not self.broadcastable(other):
            abort(
                "__sub__ → Dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__()
            )

        if self.shape != other.shape:
            return self.broadcast_operation[
                Subtract, AddTensor, SubtractTensor
            ](other)
        requires_grad = (
            self.address()[].requires_grad or other.address()[].requires_grad
        )

        out = __tensor_op_tensor__[dtype, SubtractTensor](
            self.address()[], other.address()[]
        )

        if requires_grad:

            fn grad_fn() raises -> None:
                out_grad = out.address()[].grad[]
                if self.address()[].requires_grad:
                    self.address()[].update_grad[AddTensor](out_grad)
                if other.address()[].requires_grad:
                    other.address()[].update_grad[SubtractTensor](out_grad)

            out.grad_fn = Optional(grad_fn)

            out.add_ancestry(self, other)

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

    fn unsafe_addr(
        ref self,
    ) -> UnsafePointer[
        Self,
        mut = Origin(__origin_of(self)).mut,
        origin = __origin_of(self),
    ]:
        return UnsafePointer(to=self).origin_cast[
            mut = Origin(__origin_of(self)).mut, origin = __origin_of(self)
        ]()

    fn mse(self, target: Tensor[dtype]) -> Tensor[dtype]:
        return ((self - target) ** 2).mean()

    fn matmul_v1(self, other: Self) -> Tensor[dtype]:
        if self.shape[1] != other.shape[0]:
            abort("Tensor matmul_v1 - Dim mismatch")
        result = Tensor[dtype].zeros(self.shape[0], other.shape[1])
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                for k in range(self.shape[1]):
                    result[i, j] += self[i, k] * other[k, j]
        return result

    fn load[nelts: Int = 1](self, rows: Int, cols: Int) -> SIMD[dtype, nelts]:
        if not self.ndim() == 2:
            abort("Tensor - load is supported only for 2d tensor")
        result = self.data.load[width=nelts](rows * self.shape[1] + cols)
        return result

    fn store[
        nelts: Int = 1
    ](self, rows: Int, cols: Int, val: SIMD[dtype, nelts]):
        if not self.ndim() == 2:
            abort("Tensor - store is supported only for 2d tensor")
        self.data.store(rows * self.shape[1] + cols, val)

    fn matmul_v2(self, other: Self) -> Tensor[dtype]:
        if self.shape[1] != other.shape[0]:
            abort("Tensor matmul_v2 - Dim mismatch")
        requires_grad = self.requires_grad or other.requires_grad

        result = Tensor[dtype](
            self.shape[0], other.shape[1], requires_grad=requires_grad
        )

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(other.shape[1]):
                    result[i, k] += self[i, j] * other[j, k]
        if requires_grad:

            fn grad_fn() raises -> None:
                a = self.address()[]
                b = other.address()[]
                out = result.address()[]
                upstream = out.grad[]

                if a.requires_grad:
                    a_grad = upstream.matmul(b.T())
                    a.update_grad[AddTensor](a_grad)

                if b.requires_grad:
                    b_grad = a.T().matmul(upstream)
                    b.update_grad[AddTensor](b_grad)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self, other)

        return result

    fn matmul_optim[
        simd_width: Int = simdwidthof[dtype](), nelts: Int = 1
    ](self, other: Self) -> Tensor[dtype]:
        rows, cols = self.shape[0], self.shape[1]
        other_rows, other_cols = other.shape[0], other.shape[1]

        if cols != other_rows:
            abort(
                "Tensor → matmul_optim - Dim mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor[dtype].zeros(
            rows, other_cols, requires_grad=requires_grad
        )
        for i in range(rows):
            for j in range(cols):

                @parameter
                fn dot[simd_width: Int](idx: Int):
                    result.store[nelts](
                        i,
                        idx,
                        result.load[nelts](i, idx)
                        + self[i, j] * other.load[nelts](j, idx),
                    )

                vectorize[dot, simd_width](other.shape[1])

        if requires_grad:

            fn grad_fn() raises -> None:
                self_ref = self.address()
                other_ref = other.address()
                result_ref = result.address()
                upstream_grad = result_ref[].grad[]

                if self_ref[].requires_grad:
                    transposed = other_ref[].T()
                    grad = upstream_grad.matmul_optim(transposed)
                    self_ref[].update_grad[AddTensor](grad)

                if other_ref[].requires_grad:
                    transposed = self_ref[].T()
                    grad = transposed.matmul_optim(upstream_grad)
                    other.address()[].update_grad[AddTensor](grad)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self, other)

        return result

    fn matmul(self: Tensor[dtype], other: Tensor[dtype]) -> Tensor[dtype]:
        if not self.shape.rank() == 2:
            abort("Only supports 2D matmul for now")
        if not other.shape.rank() == 2:
            abort("Other must be 2D")
        if not self.shape[1] == other.shape[0]:
            abort("Incompatible shapes")

        m, k = self.shape[0], self.shape[1]
        n = other.shape[1]

        requires_grad = self.requires_grad or other.requires_grad
        var result = Tensor[dtype](m, n, requires_grad=requires_grad)

        for i in range(m):
            for j in range(n):
                var summ = Scalar[dtype](0)
                for p in range(k):
                    summ += self[IntList(i, p)] * other[IntList(p, j)]
                result[IntList(i, j)] = summ

        if requires_grad:

            fn grad_fn() raises -> None:
                a = self.address()[]
                b = other.address()[]
                out = result.address()[]
                upstream = out.grad[]

                if a.requires_grad:
                    a_grad = upstream.matmul(b.T())
                    a.update_grad[AddTensor](a_grad)

                if b.requires_grad:
                    b_grad = a.T().matmul(upstream)
                    b.update_grad[AddTensor](b_grad)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self, other)

        return result

    fn T(self, tile_size: Int = 32) raises -> Tensor[dtype]:
        if self.shape.ndim != 2:
            abort("Tensor → transpose allowed only for 2D tensors")
        rows, cols = (self.shape[0], self.shape[1])
        result = Tensor[dtype](
            self.shape.reverse(), requires_grad=self.requires_grad
        )

        for i in range(0, rows, tile_size):
            for j in range(0, cols, tile_size):
                for ii in range(i, min(i + tile_size, rows)):
                    for jj in range(j, min(j + tile_size, cols)):
                        result[jj, ii] = self[ii, jj]

        if self.requires_grad:

            fn grad_fn() raises:
                upstream_grad = result.address()[].grad[]
                self.address()[].update_grad[AddTensor](upstream_grad.T())

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self)

        return result

    @always_inline
    fn is_scalar(self) -> Bool:
        return self.numels() == 1 and self.shape == Shape.Void

    fn reshape(self) -> Tensor[dtype]:
        if self.numels() != 1:
            abort(
                "Only tensor with single element can be reshaped to scalar"
                " tensor"
            )
        return self.reshape(Shape.Void)

    fn reshape(self, *newdims: Int) -> Tensor[dtype]:
        if len(newdims) == 1 and newdims[0] == 0:
            print("reshape *newdims: ", len(newdims), newdims[0])
            return self.reshape()
        return self.reshape(Shape(newdims))

    fn reshape(self, new_shape: Shape) -> Tensor[dtype]:
        if self.numels() != new_shape.num_elements():
            # if self.shape.product() != new_shape.product():
            abort(
                "Tensor with "
                + String(self.numels())
                + " element(s) can't be converted to a tensor containing "
                + String(new_shape.num_elements())
                + " element(s)"
            )

        requires_grad = self.requires_grad
        result = Tensor[dtype](
            new_shape, self.data, requires_grad=requires_grad
        )

        if requires_grad:
            # Only allocate base if needed
            base = Tensor[dtype].zeros(self.shape)
            result.base = UnsafePointer[Tensor[dtype]].alloc(1)
            result.base.init_pointee_move(base^)

            fn grad_fn() raises -> None:
                upstream_grad = (
                    result.address()[].grad[].reshape(self.address()[].shape)
                )
                # Calculate new contribution (exclude already accumulated)
                new_contrib = __tensor_op_tensor__[dtype, SubtractTensor](
                    upstream_grad, result.address()[].base[]
                )
                # Update parent gradient
                self.address()[].update_grad[AddTensor](new_contrib)
                # Update accumulator
                result.address()[].base.init_pointee_move(upstream_grad^)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self)

        return result

    _ = """fn unsqueeze(self, dim: Int) -> View[__origin_of(self), self.dtype]:
        new_shape = self.shape.intlist().insert(dim, 1)
        # result = Tensor[dtype](Shape(new_shape), requires_grad=self.requires_grad)
        # result.data = self.data  # share same data
        return self.view(Shape(new_shape))
        # return result"""

    fn mean(
        self, axes: List[Int] = [], keepdims: Bool = False
    ) -> Tensor[dtype]:
        return self.mean(IntList.new(axes), keepdims)

    fn mean(self, axes: IntList, keepdims: Bool = False) -> Tensor[dtype]:
        sorted_axes = Self.validate_and_normalize_axes(self.shape, axes)
        # Compute total count of elements being reduced
        reduce_dims = self.shape.axes_spans.select(sorted_axes)
        var count = 1
        for span in reduce_dims:
            count *= span

        if count == 0:
            abort("Mean reduction over zero elements not allowed.")

        # Perform sum
        summed = self.sum(sorted_axes, keepdims)
        # Divide by count
        var result = summed / Scalar[dtype](count)

        # Gradient logic
        if self.requires_grad:

            fn grad_fn() raises -> None:
                print("This is the place")
                upstream_grad = result.address()[].grad[]
                if upstream_grad.shape == Shape.Void:
                    scalar_grad = (
                        upstream_grad.item()
                        / self.address()[].shape.num_elements()
                    )
                    grad_contrib = Tensor[dtype].full(
                        self.address()[].shape, scalar_grad, requires_grad=False
                    )
                    self.address()[].update_grad[AddTensor](grad_contrib)
                    return

                var expanded = upstream_grad

                if not keepdims:
                    expanded = upstream_grad.reshape(
                        Shape(
                            upstream_grad.shape.intlist().insert(
                                sorted_axes,
                                IntList.with_capacity(len(sorted_axes), 1),
                            )
                        )
                    )

                # Broadcast and divide
                broadcasted = expanded.broadcast_to(self.address()[].shape)
                scaled = broadcasted / Scalar[dtype](count)
                self.address()[].update_grad[AddTensor](scaled)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self)

        return result

    fn sum(self, axes: List[Int] = [], keepdims: Bool = False) -> Tensor[dtype]:
        return self.sum(IntList.new(axes), keepdims)

    fn sum(self: Self, axes: IntList, keepdims: Bool = False) -> Tensor[dtype]:
        _axes = Self.validate_and_normalize_axes(self.shape, axes)
        requires_grad = self.requires_grad
        # original_shape = self.shape
        rank = self.shape.rank()
        print("rank rank rank rank: ", rank)

        # Early scalar return - already correct
        if rank == 0:
            scalar_out = Tensor[dtype].zeros(
                Shape.Void, requires_grad=self.requires_grad
            )
            scalar_out[IntList.Empty] = self[IntList.Empty]

            if self.requires_grad:
                self_ptr_ = UnsafePointer(to=self)
                out_ptr_ = UnsafePointer(to=scalar_out)

                fn scalar_grad_fn() raises -> None:
                    print("Inside scalar_grad_fn")
                    self_ptr_[].update_grad[AddTensor](out_ptr_[].grad[])

                scalar_out.grad_fn = Optional(scalar_grad_fn)
                scalar_out.add_ancestry(self)
                print("returned scalar_out")
            return scalar_out

        # FIX 1: Handle full reduction case explicitly
        var out_shape: Shape
        reducing_all = len(_axes) == rank
        if reducing_all and not keepdims:
            # Explicit scalar output for full reduction
            out_shape = Shape.Void
        else:
            spans = IntList.with_capacity(rank)
            for i in range(rank):
                if i in _axes:
                    if keepdims:
                        spans.append(1)
                    else:
                        continue
                else:
                    spans.append(self.shape[i])
            out_shape = Shape(spans)
        print("requires_grad requires_grad: ", requires_grad)
        out = Tensor[dtype].zeros(out_shape, requires_grad=requires_grad)
        result_addr = out.address()
        print("The address of result: ", result_addr, Int(result_addr))
        reduced_shape = Shape(self.shape.axes_spans.select(_axes))
        # Special handling for full reduction case
        if reducing_all and not keepdims:
            summ = Scalar[dtype](0)
            for idx in self.shape:
                summ += self[idx]
            out[IntList.Empty] = summ
        else:
            for out_idx in out_shape:
                summ = Scalar[dtype](0)
                for red_idx in reduced_shape:
                    if keepdims:
                        full_idx = out_idx.replace(_axes, red_idx)
                    else:
                        full_idx = out_idx.insert(_axes, red_idx)
                    summ += self[full_idx]
                out[out_idx] = summ

        if requires_grad:

            fn grad_fn() raises -> None:
                # sum_grad_fn = SumGradFn(self.address(), out.address(), keepdims, _axes.address())
                # sum_grad_fn()
                print("Actually this is place")
                outstream_grad = out.address()[].grad[]
                this = self.address()[]
                original_shape = this.shape
                print("sum grad function original shape: ", original_shape)
                var grad_contrib: Tensor[dtype]

                # Handle scalar gradient case (sum reduced to scalar)
                if outstream_grad.shape == Shape.Void:
                    grad_contrib = Tensor[dtype].full(
                        original_shape,
                        outstream_grad.item(),
                        requires_grad=False,
                    )
                else:
                    # Handle keepdims=False case (need to reshape gradient)
                    if not keepdims:
                        # Determine axes/unsqueeze (insert dims of size 1)
                        axes = outstream_grad.shape.intlist().insert(
                            _axes,
                            IntList.with_capacity(len(_axes), 1),
                        )
                        unsqueezed_shape = Shape(axes)

                        unsqueezed_grad = outstream_grad.reshape(
                            unsqueezed_shape
                        )
                        grad_contrib = unsqueezed_grad.broadcast_to(
                            original_shape
                        )
                    else:
                        # keepdims=True: shapes match except for broadcasting
                        grad_contrib = outstream_grad.broadcast_to(
                            original_shape
                        )
                grad_contrib.requires_grad = False
                this.update_grad[AddTensor](grad_contrib)
                print("Out of sum grad_fn successfully")

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self)
            print(
                "sum result and self ids: ",
                Int(out.id()),
                Int(self.id()),
            )

        print("Gone out of sum")
        return out

    @staticmethod
    fn validate_and_normalize_axes(shape: Shape, axes: IntList) -> IntList:
        """Ensure axes are unique, sorted, and within bounds."""
        rank = shape.rank()

        if rank == 0:
            if len(axes) == 1 and axes[0] == -1:
                return (
                    IntList()
                )  # Interpret `[-1]` as "reduce all axes" for scalars
            if len(axes) > 0:
                abort(
                    "Tensor → validate_and_normalize_axes - cannot reduce over"
                    " axes "
                    + axes.__str__()
                    + " for scalar tensor with shape: "
                    + shape.__str__()
                )
            return IntList()  # Scalar sum over [] is valid

        if len(axes) == 0:
            return IntList.range_list(rank)
        normalized = IntList.with_capacity(len(axes))
        for _axis in axes:
            axis = _axis
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                abort(
                    "Tensor → validate_and_normalize_axes - invalid axis: "
                    + String(_axis)
                    + " for tensor shape: "
                    + shape.__str__()
                )
            normalized.append(axis)
        # Sort and deduplicate
        normalized.sort_and_deduplicate()
        return normalized

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
    fn full(
        shape: Shape, value: Scalar[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        tensor = Tensor[dtype](shape, requires_grad=requires_grad)
        tensor.fill(value)
        return tensor

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
        for i in range(tensor.numels()):  # vectorize?
            tensor.data.store[volatile=True](
                i,
                random_float64(
                    min.cast[DType.float64](), max.cast[DType.float64]()
                ).cast[dtype](),
            )
        return tensor

    @staticmethod
    fn arange(
        *args: Scalar[dtype],
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        start: Scalar[dtype] = 0
        end: Scalar[dtype] = max_finite[dtype]()
        step: Scalar[dtype] = 1

        n = len(args)
        if n == 1:
            end = args[0]
        elif n == 2:
            start = args[0]
            end = args[1]
        elif n == 3:
            start = args[0]
            end = args[1]
            step = args[2]
        else:
            abort(
                "Tensor.arange expects 1 to 3 arguments:\n"
                + "- arange(end)\n"
                + "- arange(start, end)\n"
                + "- arange(start, end, step)\n"
                + "Got: "
                + String(len(args))
                + " argument(s)"
            )

        if step == 0:
            abort("step can not be zero")
        if (step > 0 and start >= end) or (step < 0 and start <= end):
            abort("Invalid range for the given step")
        delta = end - start
        size = floor(delta / step)
        if size <= 0:
            abort("Error: computed arange size is zero")
        count = size.__int__()
        tensor = Tensor[dtype](count, requires_grad=requires_grad)

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
    fn ones_like(
        tensor: Tensor[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        out = Tensor[dtype].full(tensor.shape, 1, requires_grad=requires_grad)
        return out

    @staticmethod
    fn zeros(shape: Shape, requires_grad: Bool = False) -> Tensor[dtype]:
        out = Tensor[dtype](shape, requires_grad)
        memset_zero(out.data, out.numels())
        return out

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
                abort("Tensor → d2 → not all rows equal in length")
            flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    @staticmethod
    fn d3(
        blocks: List[Self.Rows], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "d3")
        dims = IntList(len(blocks), len(blocks[0]), len(blocks[0][0]))
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for block in blocks:
            if len(block) != dims[1]:
                abort("Tensor → d3 → not all blocks equal in length")
            for row in block:
                if len(row) != dims[2]:
                    abort("Tensor → d3 → not all rows equal in length")

                flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

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
                    "Tensor → d4 → not all blocks are of equal length in the"
                    " blockgrid"
                )
            for matrix in block:
                if len(matrix) != dims[2]:
                    abort(
                        "Tensor → d4 → not all matrices are of equal length"
                        " in block"
                    )
                for row in matrix:
                    if len(row) != dims[3]:
                        abort(
                            "Tensor → d4 not all rows are of equal length in"
                            " matrix"
                        )
                    flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

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
                    "Tensor → d5 → not all blocks are of equal length in the"
                    " input"
                )
            for block in blocks:
                if len(block) != dims[2]:
                    abort("Tensor → d5 → unequal block length")
                for matrix in block:
                    if len(matrix) != dims[3]:
                        abort(
                            "Tensor → d5 not all matrices are of equal length"
                            " in block"
                        )
                    for row in matrix:
                        if len(row) != dims[4]:
                            abort(
                                "Tensor → d5 not all rows are of equal length"
                                " in matrix"
                            )
                        flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    @staticmethod
    fn of(*elems: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "of(*elems)")
        # shape = Shape.of(len(elems))
        shape = Shape(IntList(len(elems)))
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(len(elems)):
            tensor[i] = elems[i]
        return tensor

    @staticmethod
    fn of(
        elems: Self.Row,
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "of(elems)")
        shape = Shape.of(len(elems))
        tensor = Tensor[Self.dtype](shape, requires_grad)
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
                    "Tensor → of[row_size] → invalid row size or not enough"
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
        return Self.ones(Shape(axes_spans), requires_grad)

    @staticmethod
    fn ones(shape: Shape, requires_grad: Bool = False) -> Tensor[dtype]:
        tensor = Tensor[dtype](shape, requires_grad=requires_grad)
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
    ):
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
                            i != num_first - 1 or size > num_first + num_last
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

            print("]", end="\n")

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

                print(indent + "]", end="\n")
                # print("\n")

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        print(self.__str__(), end="\n")
        empty = IntList()
        self.print_tensor_recursive(
            empty, 1, num_first=num_first, num_last=num_last
        )

    @staticmethod
    fn free_all[dtype: DType, //](*tensors: Tensor[dtype]):
        for each in tensors:
            each.free()
            _ = each

    _ = """fn view(
        ref self, target_shape: Shape
    ) -> View[__origin_of(self), self.dtype]:
        concrete = True if target_shape == self.shape else False
        mask = self.shape.broadcast_mask(target_shape)
        return View(Pointer(to=self), concrete, mask, target_shape)"""

    fn view2(self) -> View2[self.dtype]:
        return View2(UnsafePointer(to=self))


@fieldwise_init
struct View2[
    dtype: DType = DType.float32,
]:
    var target: UnsafePointer[Tensor[dtype]]


@fieldwise_init
struct View[
    mutability: Bool, //,
    origin: Origin[mutability],
    dtype: DType = DType.float32,
]:
    var target: Pointer[Tensor[dtype], origin]
    var concrete: Bool
    var mask: IntList
    var target_shape: Shape

    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        if self.concrete:
            return self.target[][indices]
        else:
            target_idx = self.target[].shape.translate_index(
                indices, self.mask, self.target_shape
            )
            return self.target[][target_idx]

    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        if self.concrete:
            self.target[].__setitem__(indices, value)
        else:
            target_idx = self.target[].shape.translate_index(
                indices, self.mask, self.target_shape
            )
            self.target[].__setitem__(target_idx, value)
