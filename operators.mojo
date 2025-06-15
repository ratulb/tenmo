from memory import UnsafePointer
from tensors import Tensor
from algorithm import vectorize, parallelize
from sys import simdwidthof
from shapes import Shape

alias Noop = 0
alias AddScalar = 1
alias SubtractScalar = 2
alias MulScalar = 3
alias MulTensor = 4
alias AddTensor = 5
alias SubtractTensor = 6


fn sum_across_rows[
    dtype: DType = DType.float32
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    rows = tensor.shape[0]
    cols = tensor.shape[1]
    shape = Shape.of(rows)
    out = Tensor[dtype].zeros(shape, requires_grad=tensor.requires_grad)

    @parameter
    fn sum_row(row: Int):
        row_start = tensor.data.offset(row * cols)

        @parameter
        fn sum_row_elems[simd_width: Int](idx: Int):
            out[row] += row_start.load[width=simd_width](idx).reduce_add()

        vectorize[sum_row_elems, simdwidthof[dtype]()](cols)

    parallelize[sum_row](rows)
    return out


fn sum_across_cols[
    dtype: DType = DType.float32
](tensor: Tensor[dtype]) -> Tensor[dtype]:
    cols = tensor.shape[1]
    shape = Shape.of(cols)
    out = Tensor[dtype].zeros(shape, requires_grad=tensor.requires_grad)

    @parameter
    fn sum_cols(col: Int):
        col_start = tensor.data.offset(col)
        out[col] += col_start.strided_load[width = simdwidthof[dtype]()](
            cols
        ).reduce_add()

    parallelize[sum_cols](cols)
    return out


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


fn main() raises:
    tensor = Tensor.of[5](1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    tensor.print()
    # summ = sum_across_rows(tensor)
    # summ.print()
    # sl = tensor.data.strided_load[width=16](5)
    # print(sl.__str__())
    # print(len(sl))
    # print(sl.reduce_add().__str__())
    summ = sum_across_cols(tensor)
    summ.print()
