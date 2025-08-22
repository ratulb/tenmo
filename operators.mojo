from tensors import Tensor
from common_utils import panic, log_debug
from algorithm import vectorize, parallelize
from sys import simdwidthof
from shapes import Shape
from os import abort
from buffers import Buffer

# from runtime.asyncrt import num_physical_cores
from sys import num_logical_cores, num_physical_cores

alias Noop = 0
#alias AddScalar = 1
#alias SubtractScalar = 2
#alias MulScalar = 3
alias MulTensor = 1
alias AddTensor = 5
alias SubtractTensor = 6
#alias Power = 7
alias Add = 8
alias Subtract = 9
alias Multiply = 10
#alias SubtractFromScalar = 11
#alias DivideByScalar = 12
#alias DivideScalar = 13
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

@fieldwise_init
struct Comparator(Copyable):
    @staticmethod
    fn compare[
        dtype: DType, //, op: Int, simd_width: Int = simdwidthof[dtype]()
    ](this: Tensor[dtype], that: Tensor[dtype]) -> Tensor[DType.bool]:
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
                return Tensor[DType.bool](
                    this.shape,
                    this_buffer.__eq__[dtype, simd_width](that_buffer),
                    False,
                )
            if op == NotEqual:
                return Tensor[DType.bool](
                    this.shape,
                    this_buffer.__ne__[dtype, simd_width](that_buffer),
                    False,
                )
            if op == LessThan:
                return Tensor[DType.bool](
                    this.shape,
                    this_buffer.__lt__[dtype, simd_width](that_buffer),
                    False,
                )
            if op == LessThanEqual:
                return Tensor[DType.bool](
                    this.shape,
                    this_buffer.__le__[dtype, simd_width](that_buffer),
                    False,
                )
            if op == GreaterThan:
                return Tensor[DType.bool](
                    this.shape,
                    this_buffer.__gt__[dtype, simd_width](that_buffer),
                    False,
                )
            if op == GreaterThanEqual:
                return Tensor[DType.bool](
                    this.shape,
                    this_buffer.__ge__[dtype, simd_width](that_buffer),
                    False,
                )
        else:
            out = Tensor[DType.bool].full(this.shape, False)

            if op == Equal:
                for indices in this.shape:
                    out[indices] = this[indices] == that[indices]
                return out

            if op == NotEqual:
                for indices in this.shape:
                    out[indices] = this[indices] != that[indices]
                return out

            if op == LessThan:
                for indices in this.shape:
                    out[indices] = this[indices] < that[indices]
                return out

            if op == LessThanEqual:
                for indices in this.shape:
                    out[indices] = this[indices] <= that[indices]
                return out

            if op == GreaterThan:
                for indices in this.shape:
                    out[indices] = this[indices] > that[indices]
                return out

            if op == GreaterThanEqual:
                for indices in this.shape:
                    out[indices] = this[indices] >= that[indices]
                return out

        log_debug(
            "Tensor compare → we should never reach here - if we have, something"
            " has gone terribly wrong"
        )
        return Tensor[DType.bool].scalar(False)

    @staticmethod
    fn compare_scalar[
        dtype: DType, //, op: Int, simd_width: Int = simdwidthof[dtype]()
    ](this: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[DType.bool]:
        if this.is_contiguous():
            if this.owns_data:
                this_buffer = this.buffer
            else:
                this_buffer = this.base_address()[].buffer[
                    this.offset : this.offset + this.numels()
                ]
            if op == Equal:
                return Tensor[DType.bool](
                    this.shape,
                    this_buffer.__eq__[simd_width](scalar),
                    False,
                )
            if op == NotEqual:
                return Tensor[DType.bool](
                    this.shape,
                    this_buffer.__ne__[simd_width](scalar),
                    False,
                )
            if op == LessThan:
                return Tensor[DType.bool](
                    this.shape,
                    this_buffer.__lt__[simd_width](scalar),
                    False,
                )
            if op == LessThanEqual:
                return Tensor[DType.bool](
                    this.shape,
                    this_buffer.__le__[simd_width](scalar),
                    False,
                )
            if op == GreaterThan:
                return Tensor[DType.bool](
                    this.shape,
                    this_buffer.__gt__[simd_width](scalar),
                    False,
                )
            if op == GreaterThanEqual:
                return Tensor[DType.bool](
                    this.shape,
                    this_buffer.__ge__[simd_width](scalar),
                    False,
                )
        else:
            out = Tensor[DType.bool].full(this.shape, False)

            if op == Equal:
                for indices in this.shape:
                    out[indices] = this[indices] == scalar
                return out

            if op == NotEqual:
                for indices in this.shape:
                    out[indices] = this[indices] != scalar
                return out

            if op == LessThan:
                for indices in this.shape:
                    out[indices] = this[indices] < scalar
                return out

            if op == LessThanEqual:
                for indices in this.shape:
                    out[indices] = this[indices] <= scalar
                return out

            if op == GreaterThan:
                for indices in this.shape:
                    out[indices] = this[indices] > scalar
                return out

            if op == GreaterThanEqual:
                for indices in this.shape:
                    out[indices] = this[indices] >= scalar
                return out

        log_debug(
            "Tensor compare_scalar → we should never reach here - if we have, something"
            " has gone terribly wrong"
        )
        return Tensor[DType.bool].scalar(False)


from testing import assert_true, assert_false


fn main() raises:
    A = Tensor.arange(10)
    C = A.reshape(5, 2)
    C.print()
    D = C.transpose()
    D.print()
    E = D.reshape(5, 2).transpose()
    print()
    E.print()
    print(E.is_contiguous(), D.is_contiguous())
    result = Comparator.compare[Equal](E, D)
    assert_true(
        result.all_true(),
        "Equality assertion check failed for non-contguous views",
    )

    B = Tensor.arange(10)
    result = Comparator.compare[Equal](A, B)
    assert_true(result.all_true(), "Equality assertion 1 failed")
    B[0] = 100
    result = Comparator.compare[Equal](A, B)
    assert_false(result.all_true(), "Nonequality assertion 1 failed")
    _ = D
    _ = E
    _ = """this = Tensor.d1([1, 5, 3, 5])
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
    _ = that"""
