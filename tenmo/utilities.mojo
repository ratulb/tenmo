from std.math import log, exp
from .buffers import Buffer


@fieldwise_init
struct Utils[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    @always_inline
    fn log_scalar(
        s: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype] where Self.dtype.is_floating_point():
        return log(s)

    @staticmethod
    @always_inline
    fn log_buffer(
        b: Buffer[Self.dtype],
    ) -> Buffer[Self.dtype] where Self.dtype.is_floating_point():
        return b.log()

    @staticmethod
    @always_inline
    fn sum_buffer(
        b: Buffer[Self.dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[Self.dtype]:
        return b.sum(start_index, end_index)

    @staticmethod
    @always_inline
    fn sum_scalars(
        this: Scalar[Self.dtype], that: Scalar[Self.dtype]
    ) -> Scalar[Self.dtype]:
        return this + that

    @staticmethod
    @always_inline
    fn product_buffer(
        b: Buffer[Self.dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[Self.dtype]:
        return b.product(start_index, end_index)

    @staticmethod
    @always_inline
    fn product_scalars(
        this: Scalar[Self.dtype], that: Scalar[Self.dtype]
    ) -> Scalar[Self.dtype]:
        return this * that

    @staticmethod
    @always_inline
    fn exp_buffer(
        b: Buffer[Self.dtype],
    ) -> Buffer[Self.dtype] where Self.dtype.is_floating_point():
        return b.exp()

    @staticmethod
    @always_inline
    fn exp_scalar(
        s: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype] where Self.dtype.is_floating_point():
        return exp(s)

    @staticmethod
    @always_inline
    fn abs_buffer(b: Buffer[Self.dtype]) -> Buffer[Self.dtype]:
        return b.__abs__()

    @staticmethod
    @always_inline
    fn abs_scalar(s: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        return s.__abs__()

    @staticmethod
    @always_inline
    fn negate_buffer(b: Buffer[Self.dtype]) -> Buffer[Self.dtype]:
        return -b

    @staticmethod
    @always_inline
    fn negate_scalar(s: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        return -s

    @staticmethod
    @always_inline
    fn invert_buffer(b: Buffer[DType.bool]) -> Buffer[DType.bool]:
        return ~b

    @staticmethod
    @always_inline
    fn invert_scalar(s: Scalar[DType.bool]) -> Scalar[DType.bool]:
        return ~s

    @staticmethod
    @always_inline
    fn tanh_stable(
        x: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype] where Self.dtype.is_floating_point():
        """
        More numerically stable tanh implementation.
        """
        if x > 0:
            return (1 - exp(-2 * x)) / (1 + exp(-2 * x))
        else:
            return (exp(2 * x) - 1) / (exp(2 * x) + 1)
