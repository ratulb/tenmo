"""Per-dtype numeric constants.

Only depends on `shared.panic` and the stdlib.
"""

from std.utils.numerics import min_finite
from std.sys.defines import get_defined_int
from .panic import panic

comptime _GELU_K0: Float64 = 0.7978845608028654  # sqrt(2/pi)
comptime _GELU_C: Float64 = 0.044715
comptime _GELU_C3: Float64 = 3.0 * _GELU_C

comptime LOG2E = 1.44269504088896340735992468100189214
comptime LN2 = 0.69314718055966295651160180568695068359375

comptime LAYERNORM_DEFAULT_EPS = 1e-5

comptime UNKNOWN_VALUE = -1

# Single source of truth for the maximum tensor rank: static capacity of
# RankArray (shared/array.mojo), stack-temp sizes in kernels, and every
# rank-dispatch bound. Override with -D MAX_RANK=N (must stay >= every rank
# the library builds, currently 5, and <= what kernels instantiate).
comptime MAX_RANK = get_defined_int["MAX_RANK", 8]()

struct Epsilon[dtype: DType](RegisterPassable):
    @staticmethod
    def value() -> Scalar[Self.dtype]:
        comptime if Self.dtype == DType.float32:
            return Scalar[Self.dtype](1e-7)
        elif Self.dtype == DType.float64:
            return Scalar[Self.dtype](1e-12)
        elif Self.dtype == DType.float16:
            return rebind[Scalar[Self.dtype]](min_finite[DType.float16]())
        elif Self.dtype == DType.bool:
            return Scalar[Self.dtype](UInt8(0))
        elif Self.dtype.is_integral():
            # Integer types have no "epsilon" concept — use 0 so that
            # b + epsilon == b (preserves integer division, no overflow).
            return Scalar[Self.dtype](0)
        else:
            panic("Epsilon value not supported for: ", String(Self.dtype))
            return Scalar[Self.dtype](0)


struct CloseTol[dtype: DType](RegisterPassable):
    @staticmethod
    def rtol() -> Scalar[Self.dtype]:
        comptime if Self.dtype == DType.float64:
            return Scalar[Self.dtype](1e-5)
        elif Self.dtype == DType.float32:
            return Scalar[Self.dtype](1e-4)
        elif Self.dtype == DType.float16:
            return Scalar[Self.dtype](1e-3)
        else:
            panic("CloseTol.rtol not supported for: ", String(Self.dtype))
            return Scalar[Self.dtype](0)

    @staticmethod
    def atol() -> Scalar[Self.dtype]:
        comptime if Self.dtype == DType.float64:
            return Scalar[Self.dtype](1e-8)
        elif Self.dtype == DType.float32:
            return Scalar[Self.dtype](1e-5)
        elif Self.dtype == DType.float16:
            return Scalar[Self.dtype](1e-5)
        else:
            panic("CloseTol.atol not supported for: ", String(Self.dtype))
            return Scalar[Self.dtype](0)


struct One[dtype: DType](RegisterPassable):
    @staticmethod
    def value() -> Scalar[Self.dtype]:
        comptime if Self.dtype.is_floating_point():
            return Scalar[Self.dtype](1.0)
        elif Self.dtype == DType.bool:
            return Scalar[Self.dtype](UInt8(1))
        else:
            return Scalar[Self.dtype](1)


struct Zero[dtype: DType](RegisterPassable):
    @staticmethod
    def value() -> Scalar[Self.dtype]:
        comptime if Self.dtype.is_floating_point():
            return Scalar[Self.dtype](0.0)
        elif Self.dtype == DType.bool:
            return Scalar[Self.dtype](UInt8(0))
        else:
            return Scalar[Self.dtype](0)
