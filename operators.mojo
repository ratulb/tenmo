from tensors import Tensor
from algorithm import vectorize, parallelize
from sys import simdwidthof
from shapes import Shape
from os import abort

# from runtime.asyncrt import num_physical_cores
from sys import num_logical_cores, num_physical_cores

alias Noop = 0
alias AddScalar = 1
alias SubtractScalar = 2
alias MulScalar = 3
alias MulTensor = 4
alias AddTensor = 5
alias SubtractTensor = 6
alias Power = 7
alias Add = 8
alias Subtract = 9
alias Multiply = 10
alias SubtractFromScalar = 11
alias DivideByScalar = 12
alias DivideScalar = 13


@always_inline
fn scalar_ops[
    dtype: DType, op: Int
](lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Scalar[dtype]:
    result = Scalar[dtype](0)
    if op == Add:
        result = lhs + rhs
    elif op == Subtract:
        result = lhs - rhs
    elif op == Multiply:
        result = lhs * rhs
    else:
        abort("operators -> scalar_ops: unsupported operation")
    return result


# Element wise operatorns
fn __tensor_op_tensor__[
    dtype: DType, op: Int
](this: Tensor[dtype], that: Tensor[dtype]) -> Tensor[dtype]:
    if this.shape != that.shape:
        abort(
            "operator -> __tensor_op__tensor("
            + String(op)
            + ")  -> Dimension mismatch: "
            + this.shape.__str__()
            + " != "
            + that.shape.__str__()
        )
    requires_grad = this.requires_grad or that.requires_grad
    out = Tensor[dtype](this.shape, requires_grad)

    @parameter
    fn mul_elems[simd_width: Int](idx: Int):
        out.data.store[width=simd_width, volatile=True](
            idx,
            (
                this.data.load[width=simd_width, volatile=True](idx)
                * that.data.load[width=simd_width, volatile=True](idx)
            ),
        )

    @parameter
    fn add_elems[simd_width: Int](idx: Int):
        out.data.store[width=simd_width, volatile=True](
            idx,
            (
                this.data.load[width=simd_width, volatile=True](idx)
                + that.data.load[width=simd_width, volatile=True](idx)
            ),
        )

    @parameter
    fn subtract_elems[simd_width: Int](idx: Int):
        out.data.store[width=simd_width, volatile=True](
            idx,
            (
                this.data.load[width=simd_width, volatile=True](idx)
                - that.data.load[width=simd_width, volatile=True](idx)
            ),
        )

    if op == MulTensor:
        vectorize[mul_elems, simdwidthof[dtype]()](out.numels())
    elif op == AddTensor:
        vectorize[add_elems, simdwidthof[dtype]()](out.numels())
    elif op == SubtractTensor:
        vectorize[subtract_elems, simdwidthof[dtype]()](out.numels())

    return out


# Tensor and scalar ops
fn __tensor_op_scalar__[
    dtype: DType, op: Int
](this: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[dtype]:
    var out = Tensor[dtype](this.shape, this.requires_grad)

    @parameter
    fn add_scalar[simd_width: Int](idx: Int):
        out.data.store[width=simd_width, volatile=True](
            idx, this.data.load[width=simd_width, volatile=True](idx) + scalar
        )

    @parameter
    fn subtract_scalar[simd_width: Int](idx: Int):
        out.data.store[width=simd_width, volatile=True](
            idx, this.data.load[width=simd_width, volatile=True](idx) - scalar
        )

    @parameter
    fn subtract_from_scalar[simd_width: Int](idx: Int):
        out.data.store[width=simd_width, volatile=True](
            idx, scalar - this.data.load[width=simd_width, volatile=True](idx)
        )

    @parameter
    fn mul_by_scalar[simd_width: Int](idx: Int):
        out.data.store[width=simd_width, volatile=True](
            idx, this.data.load[width=simd_width, volatile=True](idx) * scalar
        )

    @parameter
    fn powered_by_scalar[simd_width: Int](idx: Int):
        out.data.store[width=simd_width, volatile=True](
            idx,
            this.data.load[width=simd_width, volatile=True](idx).__pow__(
                scalar
            ),
        )

    @parameter
    fn div_by_factor[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx,
            this.data.load[width=simd_width, volatile=True](idx).__truediv__(
                scalar
            ),
        )

    @parameter
    fn divide_scalar[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx,
            this.data.load[width=simd_width, volatile=True](idx).__rtruediv__(
                scalar
            ),
        )

    if op == MulScalar:
        vectorize[mul_by_scalar, simdwidthof[dtype]()](out.numels())
    elif op == AddScalar:
        vectorize[add_scalar, simdwidthof[dtype]()](out.numels())
    elif op == SubtractScalar:
        vectorize[subtract_scalar, simdwidthof[dtype]()](out.numels())
    elif op == SubtractFromScalar:
        vectorize[subtract_from_scalar, simdwidthof[dtype]()](out.numels())
    elif op == DivideByScalar:
        vectorize[div_by_factor, simdwidthof[dtype]()](out.numels())
    elif op == DivideScalar:
        vectorize[divide_scalar, simdwidthof[dtype]()](out.numels())

    elif op == Power:
        vectorize[powered_by_scalar, simdwidthof[dtype]()](out.numels())

    return out


from testing import assert_true


fn main() raises:
    pass
