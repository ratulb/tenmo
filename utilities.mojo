from math import log, exp
from buffers import Buffer

@fieldwise_init
@register_passable
struct Utils[dtype: DType](ImplicitlyCopyable):

    @staticmethod
    @always_inline
    fn log_scalar(s: Scalar[dtype]) -> Scalar[dtype]:
        return log(s)

    @staticmethod
    @always_inline
    fn log_buffer(b: Buffer[dtype]) -> Buffer[dtype]:
        return b.log()

    @staticmethod
    @always_inline
    fn sum_buffer(
        b: Buffer[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[dtype]:
        return b.sum(start_index, end_index)

    @staticmethod
    @always_inline
    fn sum_scalars(
        this: Scalar[dtype], that: Scalar[dtype]
    ) -> Scalar[dtype]:
        return this + that

    @staticmethod
    @always_inline
    fn product_buffer(
        b: Buffer[dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[dtype]:
        return b.product(start_index, end_index)

    @staticmethod
    @always_inline
    fn product_scalars(
        this: Scalar[dtype], that: Scalar[dtype]
    ) -> Scalar[dtype]:
        return this * that

    @staticmethod
    @always_inline
    fn exp_buffer(b: Buffer[dtype]) -> Buffer[dtype]:
        return b.exp()

    @staticmethod
    @always_inline
    fn exp_scalar(s: Scalar[dtype]) -> Scalar[dtype]:
        return exp(s)

    @staticmethod
    @always_inline
    fn abs_buffer(b: Buffer[dtype]) -> Buffer[dtype]:
        return b.__abs__()

    @staticmethod
    @always_inline
    fn abs_scalar(s: Scalar[dtype]) -> Scalar[dtype]:
        return s.__abs__()

    @staticmethod
    @always_inline
    fn negate_buffer(b: Buffer[dtype]) -> Buffer[dtype]:
        return -b

    @staticmethod
    @always_inline
    fn negate_scalar(s: Scalar[dtype]) -> Scalar[dtype]:
        return -s

    @staticmethod
    @always_inline
    fn invert_buffer(b: Buffer[DType.bool]) -> Buffer[DType.bool]:
        return ~b

    @staticmethod
    @always_inline
    fn invert_scalar(s: Scalar[DType.bool]) -> Scalar[DType.bool]:
        return ~s


fn main() raises:
    pass
