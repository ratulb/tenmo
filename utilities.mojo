from math import log, exp
from buffers import Buffer


trait Utils:
    alias datatype: DType

    @staticmethod
    @always_inline
    fn log_scalar(s: Scalar[Self.datatype]) -> Scalar[Self.datatype]:
        return log(s)

    @staticmethod
    @always_inline
    fn log_buffer(b: Buffer[Self.datatype]) -> Buffer[Self.datatype]:
        return b.log()

    @staticmethod
    @always_inline
    fn sum_buffer(
        b: Buffer[Self.datatype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[Self.datatype]:
        return b.sum(start_index, end_index)

    @staticmethod
    @always_inline
    fn sum_scalars(
        this: Scalar[Self.datatype], that: Scalar[Self.datatype]
    ) -> Scalar[Self.datatype]:
        return this + that

    @staticmethod
    @always_inline
    fn product_buffer(
        b: Buffer[Self.datatype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Scalar[Self.datatype]:
        return b.product(start_index, end_index)

    @staticmethod
    @always_inline
    fn product_scalars(
        this: Scalar[Self.datatype], that: Scalar[Self.datatype]
    ) -> Scalar[Self.datatype]:
        return this * that

    @staticmethod
    @always_inline
    fn exp_buffer(b: Buffer[Self.datatype]) -> Buffer[Self.datatype]:
        return b.exp()

    @staticmethod
    @always_inline
    fn exp_scalar(s: Scalar[Self.datatype]) -> Scalar[Self.datatype]:
        return exp(s)

    @staticmethod
    @always_inline
    fn abs_buffer(b: Buffer[Self.datatype]) -> Buffer[Self.datatype]:
        return b.__abs__()

    @staticmethod
    @always_inline
    fn abs_scalar(s: Scalar[Self.datatype]) -> Scalar[Self.datatype]:
        return s.__abs__()

    @staticmethod
    @always_inline
    fn negate_buffer(b: Buffer[Self.datatype]) -> Buffer[Self.datatype]:
        return -b

    @staticmethod
    @always_inline
    fn negate_scalar(s: Scalar[Self.datatype]) -> Scalar[Self.datatype]:
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
