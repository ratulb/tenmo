# === Scalar Operations  —  Single home for all per-element functions ===
#
# Every function here is a pure Scalar[dtype] → Scalar[dtype] (or → Bool)
# transformation.  No buffer, shape, stride, or device knowledge.
#
# Both NDBuffer and CpuBroadcast import from here instead of duplicating
# or calling across module boundaries.  Test files should import too.

from tenmo.common_utils import Epsilon, One
from tenmo.mnemonics import (
    Multiply,
    Add,
    Subtract,
    ReverseSubtract,
    Divide,
    ReverseDivide,
    POW,
    NEGATE,
    SQRT,
    ABS,
    LOG,
    EXP,
    SIGMOID_FORWARD,
    SIGMOID_BACKWARD,
    TANH_FORWARD,
    TANH_BACKWARD,
    SQRT_BACKWARD,
    MAX,
    MIN,
    Equal,
    NotEqual,
    GreaterThanEqual,
    GreaterThan,
    LessThanEqual,
    LessThan,
)
from std.math import sqrt, log, exp, tanh


struct ScalarOps[dtype: DType](
    ImplicitlyCopyable & Movable & Equatable & Writable
):
    """Namespace for per-element arithmetic, unary, loss, and comparison
    operations on Scalar[dtype] values.

    All methods are @staticmethod; the struct exists only to carry the
    dtype parameter so that Self.dtype resolves correctly inside each.
    """

    # ------------------------------------------------------------------
    # Binary scalar arithmetic (op_code dispatch)
    # ------------------------------------------------------------------

    @staticmethod
    @always_inline
    def scalar_fn[
        op_code: Int,
    ](
        lhs: Scalar[Self.dtype],
        rhs: Scalar[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> Scalar[Self.dtype]:
        comptime if op_code == Add:
            return lhs + rhs
        elif op_code == Subtract:
            return lhs - rhs
        elif op_code == ReverseSubtract:
            return rhs - lhs
        elif op_code == Multiply:
            return lhs * rhs
        elif op_code == Divide:
            return lhs / rhs
        elif op_code == MAX:
            return max(lhs, rhs)
        elif op_code == MIN:
            return min(lhs, rhs)
        elif op_code == SIGMOID_BACKWARD:
            return rhs * lhs * (One[Self.dtype].value() - lhs)
        elif op_code == SQRT_BACKWARD:
            return rhs * (
                One[Self.dtype].value()
                / (epsilon + Scalar[Self.dtype](2) * sqrt(lhs))
            )
        elif op_code == TANH_BACKWARD:
            return rhs * (One[Self.dtype].value() - lhs * lhs)
        elif op_code == POW:
            return lhs**rhs
        else:  # op_code == ReverseDivide
            return rhs / lhs

    # ------------------------------------------------------------------
    # Floating-point unary operations
    # ------------------------------------------------------------------

    @staticmethod
    @always_inline
    def float_unary_fn_helper[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](scalar: Scalar[Self.dtype]) -> Scalar[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        comptime if op_code == LOG:
            return log(max(scalar, epsilon))
        elif op_code == SIGMOID_FORWARD:
            return One[Self.dtype].value() / (
                One[Self.dtype].value() + exp(scalar)
            )
        elif op_code == TANH_FORWARD:
            return tanh(scalar)
        else:  # op_code == EXP
            return exp(scalar)

    # ------------------------------------------------------------------
    # General unary operations (any numeric dtype)
    # ------------------------------------------------------------------

    @staticmethod
    @always_inline
    def unary_fn_helper[
        op_code: Int
    ](scalar: Scalar[Self.dtype]) -> Scalar[Self.dtype]:
        comptime if op_code == NEGATE:
            return -scalar
        elif op_code == SQRT:
            return sqrt(scalar)
        else:  # op_code == ABS
            return scalar.__abs__()

    # ------------------------------------------------------------------
    # Float64 cast helper  (used by PRODUCT reduction)
    # ------------------------------------------------------------------

    @staticmethod
    @always_inline
    def cast_result[
        datatype: DType
    ](val: Scalar[DType.float64]) -> Scalar[datatype]:
        """Cast float64 log-space result back to dtype.
        Rounds to nearest integer for integral types before casting —
        prevents log/exp precision loss from producing 23 instead of 24.
        For floating point types, direct cast (no rounding needed).
        """
        comptime if datatype.is_integral():
            return round(val).cast[datatype]()
        else:
            return val.cast[datatype]()

    @staticmethod
    def excl_one_cpu(
        val: Scalar[DType.float64],
        total_log: Scalar[DType.float64],
        total_neg: Int,
        total_zero: Int,
        f64_zero: Scalar[DType.float64],
    ) -> Scalar[Self.dtype]:
        """Compute exclusive product for one element. CPU helper."""
        if total_zero > 1:
            return Scalar[Self.dtype](0)
        elif total_zero == 1:
            if val == f64_zero:
                var sign = Scalar[DType.float64](
                    -1 if total_neg % 2 == 1 else 1
                )
                return Self.cast_result[Self.dtype](sign * exp(total_log))
            else:
                return Scalar[Self.dtype](0)
        else:
            if val == f64_zero:
                return Scalar[Self.dtype](0)
            var val_neg = 1 if val < f64_zero else 0
            var excl_log = total_log - log(abs(val))
            var excl_neg = total_neg - val_neg
            var sign = Scalar[DType.float64](-1 if excl_neg % 2 == 1 else 1)
            return Self.cast_result[Self.dtype](sign * exp(excl_log))

    # ------------------------------------------------------------------
    # Scalar comparison  (from Buffer.compare_pair)
    # ------------------------------------------------------------------

    @staticmethod
    @always_inline
    def compare_pair[
        op_code: Int
    ](left: Scalar[Self.dtype], right: Scalar[Self.dtype]) -> Bool:
        comptime if op_code == Equal:
            return left == right
        elif op_code == NotEqual:
            return left != right
        elif op_code == GreaterThanEqual:
            return left >= right
        elif op_code == GreaterThan:
            return left > right
        elif op_code == LessThanEqual:
            return left <= right
        else:  # op_code == LessThan
            return left < right
