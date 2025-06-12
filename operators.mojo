from memory import UnsafePointer
from tensors import Tensor
from algorithm import vectorize
from sys import simdwidthof

alias Noop = 0
alias AddScalar = 1
alias SubtractScalar = 2
alias MulScalar = 3
alias MulTensor = 4
alias AddTensor = 5
alias SubtractTensor = 6


# Element wise operatorns
fn __tensor_op_tensor__[
    dtype: DType, op: Int
](this: Tensor[dtype], that: Tensor[dtype]) raises -> Tensor[dtype]:
    if this.shape != that.shape:
        raise Error(
            "__tensor_op__tensor(" + String(op) + ")  -> Dimension mismatch:",
            this.shape,
            that.shape,
        )
    requires_grad = this.requires_grad or that.requires_grad
    out = Tensor[dtype](this.shape, requires_grad)

    @parameter
    fn mul_elems[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx,
            (
                this.data.load[width=simd_width](idx)
                * that.data.load[width=simd_width](idx)
            ),
        )

    @parameter
    fn add_elems[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx,
            (
                this.data.load[width=simd_width](idx)
                + that.data.load[width=simd_width](idx)
            ),
        )

    @parameter
    fn subtract_elems[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx,
            (
                this.data.load[width=simd_width](idx)
                - that.data.load[width=simd_width](idx)
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
](this: Tensor[dtype], scalar: Scalar[dtype]) raises -> Tensor[dtype]:
    var out = Tensor[dtype](this.shape, this.requires_grad)

    @parameter
    fn add_scalar[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx, this.data.load[width=simd_width](idx) + scalar
        )

    @parameter
    fn subtract_scalar[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx, this.data.load[width=simd_width](idx) - scalar
        )

    @parameter
    fn mul_by_scalar[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx, this.data.load[width=simd_width](idx) * scalar
        )

    if op == MulScalar:
        vectorize[mul_by_scalar, simdwidthof[dtype]()](out.numels())
    elif op == AddScalar:
        vectorize[add_scalar, simdwidthof[dtype]()](out.numels())
    elif op == SubtractScalar:
        vectorize[subtract_scalar, simdwidthof[dtype]()](out.numels())

    return out


fn main():
    pass
