from tensors import Tensor
from common_utils import panic
from algorithm import vectorize, parallelize
from sys import simdwidthof
from shapes import Shape
from os import abort
from buffers import Buffer

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
alias Equal = 14
alias NotEqual = 15
alias LessThan = 16
alias LessThanEqual = 17
alias GreaterThan = 18
alias GreaterThanEqual = 19


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
        out.buffer.store[simdwidth=simd_width](
            idx,
            (
                this.buffer.load[simdwidth=simd_width](idx)
                * that.buffer.load[simdwidth=simd_width](idx)
            ),
        )

    @parameter
    fn add_elems[simd_width: Int](idx: Int):
        out.buffer.store[simdwidth=simd_width](
            idx,
            (
                this.buffer.load[simdwidth=simd_width](idx)
                + that.buffer.load[simdwidth=simd_width](idx)
            ),
        )

    @parameter
    fn subtract_elems[simd_width: Int](idx: Int):
        out.buffer.store[simdwidth=simd_width](
            idx,
            (
                this.buffer.load[simdwidth=simd_width](idx)
                - that.buffer.load[simdwidth=simd_width](idx)
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
        out.buffer.store[simdwidth=simd_width](
            idx, this.buffer.load[simdwidth=simd_width](idx) + scalar
        )

    @parameter
    fn subtract_scalar[simd_width: Int](idx: Int):
        out.buffer.store[simdwidth=simd_width](
            idx, this.buffer.load[simdwidth=simd_width](idx) - scalar
        )

    @parameter
    fn subtract_from_scalar[simd_width: Int](idx: Int):
        out.buffer.store[simdwidth=simd_width](
            idx, scalar - this.buffer.load[simdwidth=simd_width](idx)
        )

    @parameter
    fn mul_by_scalar[simd_width: Int](idx: Int):
        out.buffer.store[simdwidth=simd_width](
            idx, this.buffer.load[simdwidth=simd_width](idx) * scalar
        )

    @parameter
    fn powered_by_scalar[simd_width: Int](idx: Int):
        out.buffer.store[simdwidth=simd_width](
            idx,
            this.buffer.load[simdwidth=simd_width](idx).__pow__(scalar),
        )

    @parameter
    fn div_by_factor[simd_width: Int](idx: Int):
        out.buffer.store[simdwidth=simd_width](
            idx,
            this.buffer.load[simdwidth=simd_width](idx).__truediv__(scalar),
        )

    @parameter
    fn divide_scalar[simd_width: Int](idx: Int):
        out.buffer.store[simdwidth=simd_width](
            idx,
            this.buffer.load[simdwidth=simd_width](idx).__rtruediv__(scalar),
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


fn tensor_compare_scalar[
    dtype: DType, //, op: Int
](this: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[DType.bool]:
    result = Tensor[DType.bool](this.shape, False)

    @parameter
    fn compare_elems[simd_width: Int](idx: Int):
        if op == Equal:
            result.buffer.store[simdwidth=simd_width](
                idx,
                this.buffer.load[simdwidth=simd_width](idx) == scalar,
            )

        if op == NotEqual:
            result.buffer.store[simdwidth=simd_width](
                idx,
                this.buffer.load[simdwidth=simd_width](idx) != scalar,
            )

        if op == LessThan:
            result.buffer.store[simdwidth=simd_width](
                idx,
                this.buffer.load[simdwidth=simd_width](idx) < scalar,
            )

        if op == LessThanEqual:
            result.buffer.store[simdwidth=simd_width](
                idx,
                this.buffer.load[simdwidth=simd_width](idx) <= scalar,
            )

        if op == GreaterThan:
            result.buffer.store[simdwidth=simd_width](
                idx,
                this.buffer.load[simdwidth=simd_width](idx) > scalar,
            )

        if op == GreaterThanEqual:
            result.buffer.store[simdwidth=simd_width](
                idx,
                this.buffer.load[simdwidth=simd_width](idx) >= scalar,
            )

    vectorize[compare_elems, simdwidthof[DType.bool]()](result.numels())
    return result


alias messages = {Equal: "__eq__", NotEqual: "__ne__"}


@fieldwise_init
struct Comparator(Copyable):
    @staticmethod
    fn compare[
        dtype: DType, //, op: Int
    ](this: Tensor[dtype], that: Tensor[dtype]) -> Tensor[DType.bool]:
        if this.shape != that.shape:
            panic(
                "Tensor compare → Dimension mismatch:",
                this.shape.__str__(),
                ",",
                that.shape.__str__(),
            )
        var this_buffer: Buffer[dtype] = Buffer[dtype].Empty
        var that_buffer: Buffer[dtype] = Buffer[dtype].Empty
        if this.is_contiguous() and that.is_contiguous():
            if this.owns_data and that.owns_data:
                this_buffer = this.buffer
                that_buffer = that.buffer
            elif this.owns_data and not that.owns_data:
                this_buffer = this.buffer
                that_buffer = that.base_address()[].buffer[
                    that.offset : that.offset + that.numels()
                ]
            elif not this.owns_data and that.owns_data:
                this_buffer = this.base_address()[].buffer[
                    this.offset : this.offset + this.numels()
                ]
                that_buffer = that.buffer
            else:
                this_buffer = this.base_address()[].buffer[
                    this.offset : this.offset + this.numels()
                ]
                that_buffer = that.base_address()[].buffer[
                    that.offset : that.offset + that.numels()
                ]
        if op == Equal:
            return Tensor[DType.bool](this.shape, (this_buffer == that_buffer), False)

        return Tensor[DType.bool].scalar(False)


fn tensor_compare[
    dtype: DType, //, op: Int
](this: Tensor[dtype], other: Tensor[dtype]) -> Tensor[DType.bool]:
    if this.shape != other.shape:
        panic(
            "Tensor __eq__ → Dimension mismatch:",
            this.shape.__str__(),
            ",",
            other.shape.__str__(),
        )
    result = Tensor[DType.bool](this.shape, False)

    @parameter
    fn compare_elems[simd_width: Int](idx: Int):
        if op == Equal:
            result.buffer.store[simdwidth=simd_width](
                idx,
                this.buffer.load[simdwidth=simd_width](idx)
                == other.buffer.load[simdwidth=simd_width](idx),
            )

        if op == NotEqual:
            result.buffer.store[simdwidth=simd_width](
                idx,
                this.buffer.load[simdwidth=simd_width](idx)
                != other.buffer.load[simdwidth=simd_width](idx),
            )

        if op == LessThan:
            result.buffer.store[simdwidth=simd_width](
                idx,
                this.buffer.load[simdwidth=simd_width](idx)
                < other.buffer.load[simdwidth=simd_width](idx),
            )

        if op == LessThanEqual:
            result.buffer.store[simdwidth=simd_width](
                idx,
                this.buffer.load[simdwidth=simd_width](idx)
                <= other.buffer.load[simdwidth=simd_width](idx),
            )

        if op == GreaterThan:
            result.buffer.store[simdwidth=simd_width](
                idx,
                this.buffer.load[simdwidth=simd_width](idx)
                > other.buffer.load[simdwidth=simd_width](idx),
            )

        if op == GreaterThanEqual:
            result.buffer.store[simdwidth=simd_width](
                idx,
                this.buffer.load[simdwidth=simd_width](idx)
                >= other.buffer.load[simdwidth=simd_width](idx),
            )

    vectorize[compare_elems, simdwidthof[DType.bool]()](result.numels())
    return result


fn sum_all[dtype: DType, //](input: Tensor[dtype]) -> Scalar[dtype]:
    constrained[
        dtype.is_numeric(),
        "operators → sumup is for numeric data types only",
    ]()
    summ = Scalar[dtype](0)

    @parameter
    fn sum_elems[simd_width: Int](idx: Int):
        summ += input.buffer.load[simdwidth=simd_width](idx).reduce_add()

    vectorize[sum_elems, simdwidthof[dtype]()](input.numels())
    return summ


fn main() raises:
    this = Tensor.d1([1, 5, 3, 5])
    this_view = this[1::]
    that = Tensor.d1([1, 2, 3, 4])
    that_view = that.view(shape=[3], offset=1)
    this_view.print()
    print()
    that_view.print()
    print()
    cmp = Comparator.compare[Equal](this_view, that_view)
    cmp.print()
    _ = this
    _ = that
