# %s/^\(fn test_\(.*\)() raises:\)$/&\r    print("test_\2")/
from testing import assert_true, assert_raises


fn main() raises:
    # test_matmul_broadcasting()
    # test_matmul_shapes()
    test_mean_with_keepdims()
    test_scalar_addition()
    test_sum_all_dims()
    test_broadcast_addition()
    test_sum_specific_axis()

    # test_matmul_broadcasting()
    # test_nested_operations()
    # test_large_tensor_backprop()
    test_reshape_gradient_2d()
    test_reshape_gradient_flatten()
    test_multiple_reshapes()
    test_reshape_noop()
    test_reshape_reused_twice_correct_grad()

    test_mean_scalar()
    test_mean_1d()
    test_mean_2d_all_axes()
    test_mean_axis0()
    test_mean_axis1_keepdims()
    test_mean_multiple_axes()
    test_mean_no_axes()
    test_mean_no_grad()

    test_tensor_div_scalar_2d()
    test_tensor_div_scalar_nonuniform()
    test_tensor_div_scalar()
    test_tensor_scalar_subtract()
    test_tensor_scalar_add_mul_pow()
    test_sum_all_elements()
    test_sum_axis0()
    test_sum_axis1_keepdims()
    test_sum_multi_axes()
    test_sum_all_axes_keepdims()
    test_sum_gradient_accumulation()
    test_scalar_sum_forward()
    test_scalar_sum_backward()
    test_scalar_sum_custom_grad()
    test_scalar_sum_keepdims_true()
    test_scalar_sum_explicit_axes()

    test_add_scalar_scalar()
    test_add_scalar_1d()
    test_add_1d_1d()
    test_add_2d_scalar()
    test_add_2d_1d()
    test_add_3d_1d()
    test_add_3d_2d()
    test_add_broadcast_degenerate()
    # test_add_mismatch_shapes()


fn test_add_scalar_scalar() raises:
    print("test_add_scalar_scalar")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = Tensor.scalar(3.0, requires_grad=True)
    var c = a + b
    assert_true(c.item() == 5.0, "Scalar addition failed")
    c.backward()
    assert_true(a.grad[].item() == 1.0)
    assert_true(b.grad[].item() == 1.0)


fn test_add_scalar_1d() raises:
    print("test_add_scalar_1d")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var c = a + b
    assert_true(c.all_close(Tensor.d1([3.0, 4.0, 5.0])))
    c.sum().backward()
    assert_true(a.grad[].item() == 3.0, "a broadcast to 3 elements")
    assert_true(b.grad[].all_close(Tensor.d1([1.0, 1.0, 1.0])))


fn test_add_1d_1d() raises:
    print("test_add_1d_1d")
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor.d1([4.0, 5.0, 6.0], requires_grad=True)
    var c = a + b
    assert_true(c.all_close(Tensor.d1([5.0, 7.0, 9.0])))
    c.sum().backward()
    # assert_true(a.grad[].all_close(Tensor.d1([1.0, 1.0, 1.0])))
    # assert_true(b.grad[].all_close(Tensor.d1([1.0, 1.0, 1.0])))


fn test_add_2d_scalar() raises:
    print("test_add_2d_scalar")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.scalar(5.0, requires_grad=True).float()
    var c = a + b
    assert_true(c.all_close(Tensor.d2([[6.0, 7.0], [8.0, 9.0]])))
    c.sum().backward()
    assert_true(a.grad[].all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))
    assert_true(b.grad[].item() == 4.0, "b broadcast to 4 elements")


fn test_add_2d_1d() raises:
    print("test_add_2d_1d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.d1([10.0, 20.0], requires_grad=True).float()
    var c = a + b  # b gets broadcasted to both rows
    assert_true(c.all_close(Tensor.d2([[11.0, 22.0], [13.0, 24.0]])))
    c.sum().backward()
    assert_true(a.grad[].all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))
    assert_true(b.grad[].all_close(Tensor.d1([2.0, 2.0]).float()))


fn test_add_3d_1d() raises:
    print("test_add_3d_1d")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )

    var b = Tensor.d1([10.0, 20.0], requires_grad=True).float()

    var c = a + b  # shape (2, 2, 2)
    c.sum().backward()
    assert_true(a.grad[].all_close(Tensor.full(a.shape, 1.0).float()))
    assert_true(b.grad[].all_close(Tensor.d1([4.0, 4.0]).float()))


fn test_add_3d_2d() raises:
    print("test_add_3d_2d")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )

    var b = Tensor.d2([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)

    var c = a + b  # b gets broadcast along dim 0
    assert_true(c.shape == a.shape)
    c.sum().backward()
    assert_true(a.grad[].all_close(Tensor.full(a.shape, 1.0).float()))
    assert_true(
        b.grad[].all_close(Tensor.full(b.shape, 2.0).float())
    )  # repeated twice


fn test_add_broadcast_degenerate() raises:
    print("test_add_broadcast_degenerate")
    var a = Tensor.d3(
        [[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True
    )  # Shape (2, 2, 1)

    var b = Tensor.d1([5.0], requires_grad=True).float()  # Shape (1,)

    var c = a + b
    assert_true(c.shape == a.shape)
    c.sum().backward()
    assert_true(b.grad[].item() == 4.0, "Broadcasted across 4 elements")


fn test_add_mismatch_shapes() raises:
    print("test_add_mismatch_shapes")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var b = Tensor.d2([[1.0], [2.0], [3.0]])  # Shape mismatch

    with assert_raises():
        _ = a + b


fn test_mean_scalar() raises:
    print("test_mean_scalar")
    var a = Tensor.scalar(4.2, requires_grad=True)
    var m = a.mean()
    assert_true(m.item() == 4.2, "Mean of scalar should be the scalar itself")
    m.backward()
    assert_true(a.grad[].item() == 1.0, "Grad of scalar mean should be 1.0")


fn test_mean_1d() raises:
    print("test_mean_1d")
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var m = a.mean()
    assert_true(m.item() == 2.0, "Mean of [1, 2, 3] is 2.0")
    m.backward()
    assert_true(
        a.grad[].all_close(Tensor.d1([1 / 3, 1 / 3, 1 / 3])),
        "Equal gradient distribution",
    )


fn test_mean_2d_all_axes() raises:
    print("test_mean_2d_all_axes")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var m = a.mean()
    assert_true(m.item() == 2.5, "Mean of all elements is 2.5")
    m.backward()
    assert_true(
        a.grad[].all_close(Tensor.d2([[0.25, 0.25], [0.25, 0.25]])),
        "Each grad is 1/4",
    )


fn test_mean_axis0() raises:
    print("test_mean_axis0")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var m = a.mean(axes=[0])
    assert_true(m.all_close(Tensor.d1([2.0, 3.0]).float()), "Mean along axis 0")
    m.backward()
    assert_true(
        a.grad[].all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])),
        "Each input contributes 1/2 to mean(axis=0)",
    )


fn test_mean_axis1_keepdims() raises:
    print("test_mean_axis1_keepdims")
    var a = Tensor.d2([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
    var m = a.mean(axes=[1], keepdims=True)
    assert_true(m.all_close(Tensor.d2([[3.0], [7.0]])), "Mean across rows")
    m.backward()
    assert_true(
        a.grad[].all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])),
        "Row-wise mean: each contributes 1/2",
    )


fn test_mean_multiple_axes() raises:
    print("test_mean_multiple_axes")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var m = a.mean(axes=[0, 2])
    assert_true(m.shape == Shape.of(2), "Shape after reducing [0, 2]")
    m.backward()
    assert_true(
        a.grad[].sum().item() == 2.0,
        "Total gradient distributed across all elements",
    )


fn test_mean_no_axes() raises:
    print("test_mean_no_axes")
    var a = Tensor.d2([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
    var m = a.mean(axes=[])
    assert_true(m.item() == 5.0, "Mean of all elements")
    m.backward()
    assert_true(a.grad[].all_close(Tensor.d2([[0.25, 0.25], [0.25, 0.25]])))


fn test_mean_no_grad() raises:
    print("test_mean_no_grad")
    var a = Tensor.d2([[10.0, 20.0], [30.0, 40.0]], requires_grad=False)
    var m = a.mean()
    assert_true(m.item() == 25.0, "Correct mean without grad")


fn test_scalar_sum_explicit_axes() raises:
    print("test_scalar_sum_explicit_axes")
    var a = Tensor.scalar(10.0)
    var result = a.sum(axes=[])
    assert_true(
        result.item() == 10.0, "Explicit empty axes should work on scalar"
    )


fn test_scalar_sum_keepdims_true() raises:
    print("test_scalar_sum_keepdims_true")
    var a = Tensor.scalar(7.0)
    var result = a.sum(axes=[], keepdims=True)
    assert_true(
        result.shape.rank() == 0,
        "keepdims=True should still return a scalar shape",
    )
    assert_true(result.item() == 7.0, "Sum with keepdims on scalar")


fn test_scalar_sum_custom_grad() raises:
    print("test_scalar_sum_custom_grad")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var result = a.sum()
    result.backward(Tensor.scalar(5.0))  # Upstream gradient is 5.0
    assert_true(
        a.grad[].item() == 5.0,
        "Custom upstream grad should be passed correctly",
    )


# This test needs to be enabled once sum is migrated
fn test_reshape_reused_twice_correct_grad() raises:
    print("test_reshape_reused_twice_correct_grad")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var r = x.reshape(Shape.of(2, 2))
    var y = r + r  # <-- r used twice
    y.backward()

    assert_true(
        x.grad[].all_close(Tensor.d1([2.0, 2.0, 2.0, 2.0])),
        "∂y/∂x should be 2s — not duplicated",
    )


# This test need to be enabled
fn test_sum_gradient_accumulation() raises:
    print("test_sum_gradient_accumulation")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s1 = a.sum()
    var s2 = a.sum()
    var s = s1 + s2
    s.backward()

    # ∂s/∂a = ∂s1/∂a + ∂s2/∂a = 1 + 1 = 2
    assert_true(
        a.grad[].all_close(Tensor.d2([[2, 2], [2, 2]])),
        "Gradient should accumulate from both paths",
    )


fn test_scalar_sum_backward() raises:
    print("test_scalar_sum_backward")
    var a = Tensor.scalar(3.14, requires_grad=True)
    var result = a.sum()  # Should just return a
    result.backward()
    assert_true(result.item() == 3.14, "Forward sum check")
    assert_true(a.grad[].item() == 1.0, "Gradient of scalar sum should be 1.0")


fn test_scalar_sum_forward() raises:
    print("test_scalar_sum_forward")
    var a = Tensor.scalar(42.0)
    var result = a.sum()
    assert_true(
        result.item() == 42.0, "Scalar sum should return the same value"
    )


fn test_sum_all_axes_keepdims() raises:
    print("test_sum_all_axes_keepdims")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum(axes=[0, 1], keepdims=True)
    s.backward(Tensor.d2([[100]]))

    assert_true(s.shape == Shape.of(1, 1), "keepdims should preserve (1,1)")
    assert_true(a.grad[].all_close(Tensor.d2([[100, 100], [100, 100]])))


fn test_sum_multi_axes() raises:
    var a = Tensor.d3(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        requires_grad=True,
    )  # shape: (2,2,2)

    var s = a.sum(axes=[0, 2])
    s.backward(Tensor.d1([10, 20]))  # shape: (2,)

    # Reduced to shape: (2,)
    # Incoming grad should be broadcasted back to shape (2,2,2)
    assert_true(
        a.grad[].all_close(
            Tensor.d3(
                [
                    [[10, 10], [20, 20]],
                    [[10, 10], [20, 20]],
                ]
            )
        )
    )


fn test_sum_axis1_nokeepdims() raises:
    print("test_sum_axis1_nokeepdims")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum(axes=[1])
    s.backward(Tensor.d1([5, 6]))  # shape: (2,)
    s.print()
    # Broadcast (2,) to (2,2)
    assert_true(a.grad[].all_close(Tensor.d2([[5, 5], [6, 6]])))


fn test_sum_axis1_keepdims() raises:
    print("test_sum_axis1_keepdims")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum(axes=[1], keepdims=True)
    s.backward(Tensor.d2([[10], [20]]))

    # ∂s/∂a = [[10, 10], [20, 20]]
    assert_true(
        a.grad[].all_close(Tensor.d2([[10, 10], [20, 20]])),
        "Keepdims should preserve dimension during broadcast",
    )


fn test_sum_axis0() raises:
    print("test_sum_axis0")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum(axes=[0])
    s.backward(Tensor.d1([10, 20]))  # incoming grad shape must match output
    assert_true(s.shape == Shape.of(2), "Sum axis=0 → shape (2,)")

    # ∂s/∂a = [[10, 20], [10, 20]]
    assert_true(
        a.grad[].all_close(Tensor.d2([[10, 20], [10, 20]])),
        "Gradient must be broadcast correctly",
    )


fn test_sum_all_elements() raises:
    print("test_sum_all_elements")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum()
    s.backward()
    assert_true(s.item() == 10.0, "Sum of all elements should be 10")
    assert_true(a.grad[].all_close(Tensor.d2([[1, 1], [1, 1]])), "∂s/∂a = ones")


# Basic reshape gradient: forward shape is changed, but grads match original shape
fn test_reshape_gradient_2d() raises:
    print("test_reshape_gradient_2d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.reshape(Shape.of(4))  # Flatten
    b.backward()
    assert_true(
        a.grad[].all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])),
        "∂b/∂a should be ones reshaped",
    )


fn test_reshape_gradient_flatten() raises:
    print("test_reshape_gradient_flatten")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var y = x.reshape(Shape.of(2, 2))  # Reshape to 2x2
    var z = y * 2.0
    z.backward()
    assert_true(
        x.grad[].all_close(Tensor.d1([2.0, 2.0, 2.0, 2.0])), "∂z/∂x should be 2"
    )


fn test_multiple_reshapes() raises:
    print("test_multiple_reshapes")
    var t = Tensor.d1([10.0, 20.0, 30.0, 40.0], requires_grad=True)
    var t2 = t.reshape(Shape.of(2, 2))
    var t3 = t2.reshape(Shape.of(4))
    var y = t3 * 3.0
    y.backward()
    assert_true(
        t.grad[].all_close(Tensor.d1([3.0, 3.0, 3.0, 3.0])),
        "Chain of reshapes should yield correct grad",
    )


fn test_reshape_noop() raises:
    print("test_reshape_noop")
    var m = Tensor.d2([[5.0, 6.0]], requires_grad=True)
    var reshaped = m.reshape(Shape.of(1, 2))  # No shape change
    reshaped.backward()
    assert_true(
        m.grad[].all_close(Tensor.d2([[1.0, 1.0]])),
        "No-op reshape still propagates grad",
    )


fn test_tensor_div_scalar_2d() raises:
    print("test_tensor_div_scalar_2d")
    var a = Tensor.d2([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
    var b = a / 2.0

    assert_true(
        b.all_close(Tensor.d2([[1.0, 2.0], [3.0, 4.0]])),
        "Forward division failed",
    )

    b.backward()
    assert_true(
        a.grad[].all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])),
        "Gradient should be 1/2",
    )


fn test_tensor_div_scalar_nonuniform() raises:
    print("test_tensor_div_scalar_nonuniform")
    var a = Tensor.d1([10.0, 20.0, 30.0], requires_grad=True)
    var out = a / 10.0

    assert_true(
        out.all_close(Tensor.d1([1.0, 2.0, 3.0])), "Forward output incorrect"
    )

    out.backward()
    # Gradient of each is 1/10
    assert_true(
        a.grad[].all_close(Tensor.d1([0.1, 0.1, 0.1])), "Gradient of a wrong"
    )


fn test_tensor_div_scalar() raises:
    print("test_tensor_div_scalar")
    var a = Tensor.d1([4.0, 6.0], requires_grad=True)
    var s = a / 2.0

    assert_true(s.all_close(Tensor.d1([2.0, 3.0])), "Forward result of a / 2")

    s.backward()
    assert_true(
        a.grad[].all_close(Tensor.d1([0.5, 0.5])),
        "Grad of a: 1/2 for each element",
    )


fn test_tensor_scalar_subtract() raises:
    print("test_tensor_scalar_subtract")
    # test_scalar_sub
    var a = Tensor.scalar(5.0, requires_grad=True)
    var b = a - 3.0
    b.backward()
    assert_true(a.grad[].item() == 1.0, "∂(a - 3)/∂a = 1")

    # test_scalar_rsub
    a = Tensor.scalar(5.0, requires_grad=True)
    b = 10.0 - a
    b.backward()
    assert_true(a.grad[].item() == -1.0, "∂(10 - a)/∂a = -1")


fn test_tensor_scalar_add_mul_pow() raises:
    print("test_tensor_scalar_add_mul_pow")
    # ─────── Tensor + scalar ───────
    var a = Tensor.scalar(3.0, requires_grad=True)
    var b = a + 2.0
    b.backward()
    # Expect: b = 5.0, ∂c/∂a = 1 → grad[a] = 1
    assert_true(b.item() == 5.0, "3.0 + 2.0 should be 5.0")
    assert_true(a.grad[].item() == 1.0, "∂(a + 2.0)/∂a = 1")

    # ─────── scalar + Tensor ───────
    a = Tensor.scalar(3.0, requires_grad=True)
    b = 2.0 + a  # should dispatch __radd__
    b.backward()
    assert_true(b.item() == 5.0, "2.0 + 3.0 should be 5.0")
    assert_true(a.grad[].item() == 1.0, "∂(a + 2.0)/∂a = 1")

    # ─────── Tensor * scalar ───────
    var c = Tensor.scalar(4.0, requires_grad=True)
    var d = c * 3.0
    d.backward()
    assert_true(d.item() == 12.0, "4.0 * 3.0")
    assert_true(c.grad[].item() == 3.0, "∂(c * 3)/∂c = 3")

    # ─────── scalar * Tensor ───────
    var e = Tensor.scalar(5.0, requires_grad=True)
    var f = 4.0 * e  # should dispatch __rmul__
    f.backward()
    assert_true(f.item() == 20.0, "4.0 * 5.0")
    assert_true(e.grad[].item() == 4.0, "∂(4 * e)/∂e = 4")

    # ─────── Tensor ** scalar ───────
    var g = Tensor.scalar(2.0, requires_grad=True)
    var h = g**3.0  # 2 ** 3 = 8
    h.backward()
    assert_true(h.item() == 8.0, "2.0 ** 3.0 = 8.0")
    assert_true(g.grad[].item() == 12.0, "∂(g ** 3)/∂g = 3 * g^2 = 3 * 4 = 12")


### Mojo Tensor
### Implement tensor library in mojo from first principles

from math import iota, exp, floor
from random import seed, random_float64
from algorithm import vectorize
from sys import simdwidthof
from utils.numerics import max_finite
from os import abort
from memory import memcpy, memset, memset_zero
from shapes import Shape
from intlist import IntList
from views import TensorView
from strides import Strides
from shared import TensorLike
from common_utils import log_debug, variadic1or2, is_null
from operators import (
    __tensor_op_tensor__,
    AddTensor,
    SubtractTensor,
    MulTensor,
    __tensor_op_scalar__,
    AddScalar,
    SubtractScalar,
    SubtractFromScalar,
    DivideByScalar,
    DivideScalar,
    MulScalar,
    sum_across_rows,
    sum_across_cols,
    Power,
    scalar_ops,
    Add,
    Subtract,
    Multiply,
)


struct Tensor[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable
):
    alias Opcode = Int
    alias GradTensor = Tensor[dtype]
    alias GradOutputs = List[(TensorLike[dtype], Self.GradTensor, Self.Opcode)]
    alias BackwardFn = fn (
        gradients: Self.GradTensor
    ) escaping -> Self.GradOutputs
    alias Row = List[Scalar[dtype]]
    alias Rows = List[Self.Row]
    alias Block = List[Self.Rows]
    alias Blocks = List[Self.Block]
    var shape: Shape
    var data: UnsafePointer[Scalar[dtype]]
    var requires_grad: Bool
    var grad: UnsafePointer[Self]
    var base: UnsafePointer[Tensor[dtype]]  # Only allocated on need basis
    var grad_fn: UnsafePointer[Self.BackwardFn]

    fn gradients(self) -> UnsafePointer[Self]:
        return self.grad

    fn capture_grad_fn(mut self, grad_fn: Self.BackwardFn):
        self.grad_fn = UnsafePointer[Self.BackwardFn].alloc(1)
        self.grad_fn.init_pointee_move(grad_fn)

    fn backward(self, start_grad: Scalar[dtype] = 1.0):
        if not self.requires_grad:
            return
        seed_tensor = Tensor[dtype].full(self.shape, start_grad)
        self.backward(seed_tensor)

    fn backward(self, seed_tensor: Tensor[dtype]):
        if not self.requires_grad:
            return
        self.seed_grad(seed_tensor)

        visited = IntList.Empty
        stack = [self.into_tensorlike()]

        while stack:
            node = stack.pop()
            if node.has_grad_fn():
                for ancestor, local_grad, opcode in node.grad_fn()[](
                    node.gradients()[]
                ):
                    if opcode == AddTensor:
                        ancestor.update_grad[AddTensor](local_grad)
                    elif opcode == SubtractTensor:
                        ancestor.update_grad[SubtractTensor](local_grad)

                    if ancestor.inner_id() not in visited:
                        stack.append(ancestor)
                        visited.append(ancestor.inner_id())

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
        self.grad_fn = UnsafePointer[Self.BackwardFn]()
        self.grad = UnsafePointer[__type_of(self)]()
        self.base = UnsafePointer[Tensor[dtype]]()
        self.data = data
        self.init_grad()

    fn is_tensor(self) -> Bool:
        return True

    fn is_leaf(self) -> Bool:
        return self.requires_grad and self.has_grad_fn()

    fn is_view(self) -> Bool:
        return False

    fn into_view(self) -> TensorView[dtype]:
        return TensorView(
            UnsafePointer(to=self),
            self.shape,
            Strides.default(self.shape),
            offset=0,
        )

    fn into_tensor(self) -> Tensor[dtype]:
        return self

    fn __init__(out self, shape: Shape, requires_grad: Bool = False):
        Shape.validate(shape)
        self.shape = shape
        self.requires_grad = requires_grad
        self.base = UnsafePointer[Tensor[dtype]]()
        self.grad_fn = UnsafePointer[Self.BackwardFn]()
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

    _ = """fn backward(self, start_grad: Scalar[dtype] = 1.0):
        graph = Graph[dtype]()
        graph.walk_backward(self.into_tensorlike(), start_grad)

    fn backward(self, with_tensor: Tensor[dtype]):
        graph = Graph[dtype]()
        graph.walk_backward(self.into_tensorlike(), with_tensor)"""

    fn backward_fn(self) -> UnsafePointer[Self.BackwardFn]:
        return self.grad_fn

    fn has_grad_fn(self) -> Bool:
        return self.backward_fn().__as_bool__()

    fn address(self) -> UnsafePointer[Tensor[dtype]]:
        return UnsafePointer(to=self)

    fn id(self) -> Int:
        return Int(self.address())

    _ = """fn invoke_grad_fn(self, verbose: Bool = False) raises -> None:
        if self.grad_fn:
            if verbose:
                print("\nInvoking  grad_fn\n")
            self.grad_fn.value()()
        else:
            if verbose:
                print("\nNo grad_fn\n")
            pass"""

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
        self.grad_fn = other.grad_fn

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.data = UnsafePointer[Scalar[other.dtype]].alloc(other.numels())
        memcpy(self.data, other.data, other.numels())
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self.base = other.base
        self.grad_fn = other.grad_fn
        self.init_grad()

    _ = """fn copy(self) -> Self:
        result = Tensor[dtype](self.shape, requires_grad=self.requires_grad)
        memcpy(result.data, self.data, self.numels())
        if result.requires_grad:
            memcpy(result.grad, self.grad, self.numels())
        return result"""

    fn init_grad(mut self):
        if self.requires_grad and self.grad.__as_bool__() == False:
            gradients = Tensor[self.dtype](self.shape)
            self.grad = UnsafePointer[__type_of(self)].alloc(1)
            self.grad.init_pointee_move(gradients^)
            self.zero_grad()

    fn gprint(self):
        if not self.requires_grad:
            print("Tensor is non-differentiable")
        elif self.requires_grad and self.grad.__as_bool__() == False:
            print("Requires grad but grad not initialized")
        else:
            self.grad[].print()

    # fn __del__(owned self):
    fn free(owned self):
        if self.has_grad_fn():
            log_debug("Tensor__del__ → freed grad_fn")
            self.grad_fn.destroy_pointee()
            self.grad_fn.free()
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
        self.shape.free()
        if self.data:
            self.data.free()
        log_debug("Tensor__del__ → called free on data")
        if self.base:
            self.base[].free()
            self.base.destroy_pointee()
            self.base.free()
            log_debug("Tensor__del__ → called free on base")
        _ = self^

    fn __len__(self) -> Int:
        return self.numels()

    fn len(self) -> Int:
        return self.numels()

    fn size(self) -> Int:
        return self.numels()

    fn numels(self) -> Int:
        return self.shape.num_elements()

    fn ndim(self) -> Int:
        return self.shape.ndim

    fn rank(self) -> Int:
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
            "Tensor → all_close is for floating point data types only",
        ]()

        if self.shape != other.shape:
            abort(
                "Tensor → all_close expects same shaped tensors: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
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

    fn seed_grad(self, with_tensor: Tensor[dtype]):
        if not self.has_grad():
            return
        if self.shape != with_tensor.shape:
            abort(
                "Tensor -> seed_grad: Shapes not equal -> "
                + self.shape.__str__()
                + "<=>"
                + with_tensor.shape.__str__()
            )
        memcpy(self.grad[].data, with_tensor.data, with_tensor.numels())

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

    fn __iadd__(self, other: Self):
        if self.is_leaf():
            abort(
                "Tensor → __iadd__(self, other): Cannot perform in-place"
                " operation on a leaf tensor requiring grad."
            )

        if self.shape != other.shape:
            abort(
                "Tensor → __iadd__(self, other): Dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
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

    fn __ne__(self, other: Self) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            abort(
                "Tensor → __ne__(self, other): Dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
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
        return self.requires_grad and self.grad.__as_bool__()

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
    fn full_like(
        like: Tensor[dtype], value: Scalar[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        shape = like.shape
        return Tensor[dtype].full(shape, value, requires_grad=requires_grad)

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
        axes_spans = variadic1or2(num_rows, row_size)
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

    fn update_grad[opcode: Int](self, incoming: Tensor[dtype]):
        self.grad[] = __tensor_op_tensor__[dtype, opcode](self.grad[], incoming)

    fn is_scalar(self) -> Bool:
        return self.numels() == 1 and self.shape == Shape.Void

    fn data_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self.data

    fn unsafe_address(
        ref self,
    ) -> UnsafePointer[
        Self,
        mut = Origin(__origin_of(self)).mut,
        origin = __origin_of(self),
    ]:
        return UnsafePointer(to=self).origin_cast[
            mut = Origin(__origin_of(self)).mut, origin = __origin_of(self)
        ]()

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

    _ = """fn __rtruediv__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, DivideScalar](self, scalar)

        if self.requires_grad:
            fn grad_fn(gradients: Self.GradTensor) -> Self.GradOutputs:
                var base_squared = self.__pow__(2)
                base_squared_reciprocal =  __tensor_op_scalar__[dtype, DivideScalar](base_squared, 1.0)
                var grad = (gradients * scalar) * base_squared_reciprocal
                return [(self.into_tensorlike(), grad, SubtractTensor)]

            out.capture_grad_fn(grad_fn)

        return out"""
    _ = """
    fn test_scalar_div_tensor() raises:
        var a = Tensor.d1([2.0, 4.0], requires_grad=True)
        var out = 8.0 / a

        assert_true(out.all_close(Tensor.d1([4.0, 2.0])), "Forward: scalar / tensor incorrect")

        out.sum().backward()

        # dz/da = -8 / a^2 ⇒ [-2.0, -0.5]
        assert_true(a.grad[].all_close(Tensor.d1([-2.0, -0.5])), "Backward gradient incorrect")

    fn test_scalar_div_tensor_multiple() raises:
        var a = Tensor.d1([1.0, 2.0, 4.0], requires_grad=True)
        var out = 8.0 / a

        assert_true(out.all_close(Tensor.d1([8.0, 4.0, 2.0])), "Forward scalar / tensor")

        out.sum().backward()

        # ∂/∂a: -8 / a^2 ⇒ [-8.0, -2.0, -0.5]
        assert_true(a.grad[].all_close(Tensor.d1([-8.0, -2.0, -0.5])), "Backward grad mismatch")

    fn test_scalar_div_tensor_2d() raises:
        var a = Tensor.d2([[1.0, 2.0], [4.0, 8.0]], requires_grad=True)
        var out = 16.0 / a

        assert_true(out.all_close(Tensor.d2([[16.0, 8.0], [4.0, 2.0]])), "Forward output incorrect")

        out.sum().backward()

        # Gradient: -16 / a^2
        assert_true(a.grad[].all_close(
            Tensor.d2([[-16.0, -4.0], [-1.0, -0.25]])
        ), "Backward gradient failed")
        """

    fn __truediv__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, DivideByScalar](
            self,
            scalar,
        )
        if self.requires_grad:

            fn grad_fn(
                gradients: Self.GradTensor,
            ) -> Self.GradOutputs:
                # ∂(x / s)/∂x = 1/s → incoming_grad / scalar
                var scaled = gradients / scalar
                return [(self.into_tensorlike(), scaled, AddTensor)]

            out.capture_grad_fn(grad_fn)

        return out

    fn __radd__(self, scalar: Scalar[dtype]) raises -> Tensor[dtype]:
        return self.__add__(scalar)

    fn __add__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, AddScalar](self, scalar)

        if self.requires_grad:

            fn grad_fn(
                gradients: Self.GradTensor,
            ) -> Self.GradOutputs:
                # Gradient of addition is 1 → just pass through incoming grad
                return [(self.into_tensorlike(), gradients, AddTensor)]

            out.capture_grad_fn(grad_fn)

        return out

    fn __rmul__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__mul__(scalar)

    fn __mul__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, MulScalar](
            self,
            scalar,
        )

        if self.requires_grad:

            fn grad_fn(
                gradients: Self.GradTensor,
            ) -> Self.GradOutputs:
                scaled_gradients = __tensor_op_scalar__[dtype, MulScalar](
                    gradients, scalar
                )
                return [(self.into_tensorlike(), scaled_gradients, AddTensor)]

            out.capture_grad_fn(grad_fn)

        return out

    fn __pow__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, Power](
            self,
            scalar,
        )

        if self.requires_grad:

            fn grad_fn(
                gradients: Self.GradTensor,
            ) -> Self.GradOutputs:
                # ∂(x**n)/∂x = n * x**(n-1)
                requires_grad = self.requires_grad
                # Set requires_grad to False and restore it back - requires self to be mutable
                # But self is not mutable - Hence the pointer indirection
                self.address()[].requires_grad = False
                # Need to see if base_pow gets a grad_fn or not - we don't want it to have one!
                # var base_pow = self ** (scalar - 1.0)
                var base_pow = __tensor_op_scalar__[dtype, Power](
                    self, (scalar - 1.0)
                )
                self.address()[].requires_grad = requires_grad
                var local_grad = __tensor_op_scalar__[dtype, MulScalar](
                    base_pow, scalar
                )
                product = __tensor_op_tensor__[dtype, MulTensor](
                    gradients, local_grad
                )
                return [(self.into_tensorlike(), product, AddTensor)]

            out.capture_grad_fn(grad_fn)

        return out

    fn __rsub__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, SubtractFromScalar](self, scalar)
        if self.requires_grad:

            fn grad_fn(
                gradients: Self.GradTensor,
            ) -> Self.GradOutputs:
                return [(self.into_tensorlike(), gradients, SubtractTensor)]

            out.capture_grad_fn(grad_fn)
        return out

    fn __sub__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, SubtractScalar](self, scalar)

        if self.requires_grad:

            fn grad_fn(
                gradients: Self.GradTensor,
            ) -> Self.GradOutputs:
                return [(self.into_tensorlike(), gradients, AddTensor)]

            out.capture_grad_fn(grad_fn)
        return out

    fn reshape(self) -> Tensor[dtype]:
        if self.numels() != 1:
            abort(
                "Only tensor with single element can be reshaped to scalar"
                " tensor"
            )
        return self.reshape(Shape.Void)

    fn reshape(self, *newdims: Int) -> Tensor[dtype]:
        if len(newdims) == 1 and newdims[0] == 0:
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
        out = Tensor[dtype](new_shape, self.data, requires_grad=requires_grad)

        if requires_grad:
            # Only allocate base if needed
            base = Tensor[dtype].zeros(self.shape)
            out.base = UnsafePointer[Tensor[dtype]].alloc(1)
            out.base.init_pointee_move(base^)

            fn grad_fn(
                gradients: Self.GradTensor,
            ) -> Self.GradOutputs:
                reshaped = gradients.reshape(self.shape)
                # Deduct already contributed portion
                new_contrib = __tensor_op_tensor__[dtype, SubtractTensor](
                    reshaped, out.base[]
                )

                # Update base accumulator
                out.base.init_pointee_move(reshaped^)
                return [(self.into_tensorlike(), new_contrib, AddTensor)]

            out.capture_grad_fn(grad_fn)

        return out

    fn sum(self, axes: List[Int] = [], keepdims: Bool = False) -> Tensor[dtype]:
        return self.sum(IntList.new(axes), keepdims)

    fn sum(self: Self, axes: IntList, keepdims: Bool = False) -> Tensor[dtype]:
        _axes = Self.validate_and_normalize_axes(self.shape, axes)
        requires_grad = self.requires_grad
        rank = self.shape.rank()

        # Early scalar return - already correct
        if rank == 0:
            scalar_out = Tensor[dtype].zeros(
                Shape.Void, requires_grad=self.requires_grad
            )
            scalar_out[IntList.Empty] = self[IntList.Empty]

            if self.requires_grad:

                fn scalar_grad_fn(
                    gradients: Self.GradTensor,
                ) -> Self.GradOutputs:
                    return [(self.into_tensorlike(), gradients, AddTensor)]

                scalar_out.capture_grad_fn(scalar_grad_fn)
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
        out = Tensor[dtype].zeros(out_shape, requires_grad=requires_grad)
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

            fn grad_fn(
                gradients: Self.GradTensor,
            ) -> Self.GradOutputs:
                original_shape = self.address()[].shape
                var grad_contrib: Tensor[dtype]

                # Handle scalar gradient case (sum reduced to scalar)
                if gradients.shape == Shape.Void:
                    grad_contrib = Tensor[dtype].full(
                        original_shape,
                        gradients.item(),
                        requires_grad=False,
                    )
                else:
                    # Handle keepdims=False case (need to reshape gradient)
                    if not keepdims:
                        # Determine axes/unsqueeze (insert dims of size 1)
                        axes = gradients.shape.intlist().insert(
                            _axes,
                            IntList.with_capacity(len(_axes), 1),
                        )
                        unsqueezed_shape = Shape(axes)

                        unsqueezed_grad = gradients.reshape(unsqueezed_shape)
                        grad_contrib = unsqueezed_grad.broadcast_to(
                            original_shape
                        )
                    else:
                        # keepdims=True: shapes match except for broadcasting
                        grad_contrib = gradients.broadcast_to(original_shape)
                grad_contrib.requires_grad = False
                tl = self.address()[].into_tensorlike()

                return [
                    (
                        self.address()[].into_tensorlike(),
                        grad_contrib,
                        AddTensor,
                    )
                ]

            out.capture_grad_fn(grad_fn)

        return out

    fn mean(
        self, axes: List[Int] = [], keepdims: Bool = False
    ) -> Tensor[dtype]:
        return self.mean(IntList.new(axes), keepdims)

    fn mean(self, axes: IntList, keepdims: Bool = False) -> Tensor[dtype]:
        sorted_axes = Self.validate_and_normalize_axes(self.shape, axes)
        # Compute total count of elements being reduced
        count = self.shape.axes_spans.select(sorted_axes).product()

        # Perform sum and divide by count
        out = self.sum(sorted_axes, keepdims) / Scalar[dtype](count)

        # Gradient logic
        if self.requires_grad:

            fn grad_fn(
                gradients: Self.GradTensor,
            ) -> Self.GradOutputs:
                if gradients.shape == Shape.Void:
                    scalar_grad = gradients.item() / self.address()[].numels()
                    grad_contrib = Tensor[dtype].full(
                        self.address()[].shape, scalar_grad, requires_grad=False
                    )
                    return [
                        (
                            self.address()[].into_tensorlike(),
                            grad_contrib,
                            AddTensor,
                        )
                    ]

                var expanded = gradients

                if not keepdims:
                    expanded = gradients.reshape(
                        Shape(
                            gradients.shape.intlist().insert(
                                sorted_axes,
                                IntList.with_capacity(len(sorted_axes), 1),
                            )
                        )
                    )

                # Broadcast and divide
                broadcasted = expanded.broadcast_to(self.address()[].shape)
                scaled = broadcasted / Scalar[dtype](count)
                return [(self.address()[].into_tensorlike(), scaled, AddTensor)]

            out.capture_grad_fn(grad_fn)

        return out

    @staticmethod
    fn validate_and_normalize_axes(shape: Shape, axes: IntList) -> IntList:
        # Ensure axes are unique, sorted, and within bounds.
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

            fn grad_fn(
                gradients: Self.GradTensor,
            ) -> Self.GradOutputs:
                grad_outputs = Self.GradOutputs()

                if self.address()[].requires_grad:
                    grad_outputs.append(
                        (
                            self.address()[].into_tensorlike(),
                            gradients,
                            AddTensor,
                        )
                    )

                if other.address()[].requires_grad:
                    grad_outputs.append(
                        (
                            other.address()[].into_tensorlike(),
                            gradients,
                            AddTensor,
                        )
                    )
                return grad_outputs

            out.capture_grad_fn(grad_fn)

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

        out = __tensor_op_tensor__[dtype, SubtractTensor](
            self.address()[], other.address()[]
        )

        if self.requires_grad or other.requires_grad:

            fn grad_fn(
                gradients: Self.GradTensor,
            ) -> Self.GradOutputs:
                grad_outputs = Self.GradOutputs()

                if self.address()[].requires_grad:
                    grad_outputs.append(
                        (
                            self.address()[].into_tensorlike(),
                            gradients,
                            AddTensor,
                        )
                    )

                if other.address()[].requires_grad:
                    grad_outputs.append(
                        (
                            other.address()[].into_tensorlike(),
                            gradients,
                            SubtractTensor,
                        )
                    )
                return grad_outputs

            out.capture_grad_fn(grad_fn)

        return out

    fn broadcast_operation[
        element_wise_op: Int, tensor_op_first: Int, tensor_op_second: Int
    ](self, other: Self) -> Tensor[dtype]:
        var result = self.broadcast_op(
            other, scalar_ops[dtype, element_wise_op]
        )

        if self.requires_grad or other.requires_grad:

            fn grad_fn(incoming: Self.GradTensor) -> Self.GradOutputs:
                var grad_outputs: Self.GradOutputs = []

                if self.address()[].requires_grad:
                    var grad_self = self.address()[].backward_grad_contrib(
                        other.address()[], incoming, False
                    )
                    grad_outputs.append(
                        (
                            self.address()[].into_tensorlike(),
                            grad_self,
                            tensor_op_first,
                        )
                    )

                if other.address()[].requires_grad:
                    var grad_other = other.address()[].backward_grad_contrib(
                        self.address()[], incoming, False
                    )
                    grad_outputs.append(
                        (
                            other.address()[].into_tensorlike(),
                            grad_other,
                            tensor_op_second,
                        )
                    )

                return grad_outputs

            result.capture_grad_fn(grad_fn)

        return result

    _ = """fn broadcast_operation[
        element_wise_op: Int, tensor_op_first: Int, tensor_op_second: Int
    ](self, other: Self) -> Tensor[dtype]:
        out = self.broadcast_op(other, scalar_ops[dtype, element_wise_op])

        print("Coming inside broadcast_operation alright")
        if self.requires_grad or other.requires_grad:
            #self_ptr = self.address()
            #that_ptr = other.address()

            print("Coming inside broadcast_operation alright: ", self.requires_grad, other.requires_grad)
            fn grad_fn(
                gradients: Self.GradTensor,
            ) -> Self.GradOutputs:
                #this = self.address()[]
                #that = other.address()[]
                print("Coming inside gran_fn alright")
                grad_outputs = Self.GradOutputs()
                if self.address()[].requires_grad:
                    # upstream_grad = out.address()[].grad[]
                    grad_contrib = self.address()[].backward_grad_contrib(
                        other.address()[], gradients, False
                    )
                    # this.update_grad[tensor_op_first](grad_contrib)
                    grad_outputs.append(
                        (self.address()[].into_tensorlike(), grad_contrib, tensor_op_first)
                    )

                if other.address()[].requires_grad:
                    # upstream_grad = out.address()[].grad[]
                    grad_contrib = other.address()[].backward_grad_contrib(
                        self.address()[], gradients, False
                    )

                    grad_outputs.append(
                        (other.address()[].into_tensorlike(), grad_contrib, tensor_op_second)
                    )
                    # that.update_grad[tensor_op_second](grad_contrib)
                return grad_outputs

            out.capture_grad_fn(grad_fn)
        return out"""

    _ = """fn __add__(self, scalar: Scalar[dtype]) raises -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, AddScalar](
            self,
            scalar,
        )
        if self.requires_grad:

            fn grad_fn() raises -> None:
                out_grad = out.address()[].grad[]
                self.address()[].update_grad[AddTensor](out_grad)

            out.grad_fn = Optional(grad_fn)

        return out"""

    fn __iadd__(self, value: Scalar[dtype]):
        if self.is_leaf():
            abort(
                "Cannot perform in-place operation on a leaf tensor requiring"
                " grad."
            )

        @parameter
        fn add_value[simd_width: Int](idx: Int):
            self.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx) + value
            )

        vectorize[add_value, simdwidthof[dtype]()](self.numels())

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

    fn broadcast_mask(self, broadcast_shape: Shape) -> IntList:
        return self.shape.broadcast_mask(broadcast_shape)

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
                # upstream_grad * other if do_multiply else upstream_grad
                upstream_grad
            )
            if grad_contrib.shape != self.shape:
                axes = self.broadcast_mask(grad_contrib.shape).indices_of(1)
                grad_contrib = grad_contrib.sum(axes=axes, keepdims=True)
            if grad_contrib.shape != self.shape:
                grad_contrib = grad_contrib.reshape(self.shape)
            grad_contrib.requires_grad = False

        return grad_contrib

    _ = """fn broadcast_mul(
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

        return result



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

            out.grad_fn = Optional(grad_fn)

        return out



    fn view(self, shape: Shape, offset: Int = 0) -> TensorView[dtype]:
        if offset < 0 or offset >= self.numels():
            abort(
                "Tensor → view(shape): offset out of bounds: offset => "
                + String(offset)
                + "and self.numels() => "
                + String(self.numels())
            )
        if shape == self.shape and offset == 0:  # Tensor offset is always 0
            return self.into_view()
        if shape.num_elements() + offset > self.numels():
            abort("Tensor → view(shape): shape numels exceeds base tensor size")

        return TensorView(
            UnsafePointer(to=self), shape, Strides.default(shape), offset=offset
        )

    fn view(
        self, shape: Shape, strides: Strides, offset: Int = 0
    ) -> TensorView[dtype]:
        if offset < 0 or offset >= self.numels():
            abort("Tensor → view: offset out of bounds")

        if strides.rank() != shape.rank():
            abort("Tensor → view: shape and strides must have same rank")

        var min_index = offset
        var max_index = offset
        for i in range(shape.rank()):
            stride = strides[i]
            extent = (shape[i] - 1) * stride
            if extent > 0:
                max_index += extent
            else:
                min_index += extent

        if min_index < 0 or max_index >= self.numels():
            abort("Tensor → view: requested view accesses out-of-bounds data")

        return TensorView(UnsafePointer(to=self), shape, strides, offset=offset)



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

        return result

    fn transpose(self, axes: List[Int] = []) -> TensorView[dtype]:
        return self.transpose(IntList.new(axes))

    fn transpose(self, axes: IntList = IntList.Empty) -> TensorView[dtype]:
        _axes = axes
        if len(_axes) == 0:
            if not self.rank() == 2:
                abort("Default transpose only valid for 2D")
            _axes = IntList(1, 0)

        if len(_axes) != self.rank():
            abort("transpose: axes must match tensor rank")

        # Permute shape and create strides
        var new_shape = self.shape.permute(_axes)
        var new_strides = Strides.default(self.shape).permute(_axes)
        result = self.view(new_shape, new_strides)
        return result"""


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
    s = m.sum()
    s.backward()
    assert_true(m.all_close(Tensor.d2([[2, 3]])))
    assert_true(a.grad[].all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])))
    m.free()
    a.free()


_ = """fn test_matmul_shapes() raises:
    print("test_matmul_shapes")
    # Test various matmul shape combinations
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var b = Tensor.d2([[5, 6], [7, 8]], requires_grad=True)
    var c = a.matmul(b)
    c.sum().backward()
    assert_true(c.all_close(Tensor.d2([[19, 22], [43, 50]])))
    assert_true(a.grad[].all_close(Tensor.d2([[11, 15], [11, 15]])))
    assert_true(b.grad[].all_close(Tensor.d2([[4, 4], [6, 6]])))


fn test_matmul_broadcasting() raises:
    print("test_matmul_broadcasting")
    # Batch matmul
    var a = Tensor.d3([[[1, 2]], [[3, 4]]], requires_grad=True)  # shape (2,1,2)
    var b = Tensor.d3([[[5], [6]]], requires_grad=True)  # shape (1,2,1)
    var c = a.matmul(b)  # shape (2,2,1)
    c.sum().backward()
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
    assert_true(a.grad[].all_close(Tensor.d2([[10, 20], [30, 40]])))"""


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
