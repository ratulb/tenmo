# =============================================================================
# Shared kernel helpers — index computation and op dispatch.
#
# - output_to_input_base / rank_to_reduced_offset: reduction index helpers
#   (used by reduction_kernel.mojo, minmax_kernel.mojo,
#    std_variance_backward_kernel.mojo)
# - simd_op / scalar_op: binary operation dispatch helpers
#   (used by binary_ops_kernel.mojo)
# =============================================================================

from tenmo.common_utils import One, Epsilon
from tenmo.array import Array
from std.math import rsqrt


@always_inline
def simd_op[
    op_code: Int,
    dtype: DType,
    simd_width: Int,
](
    a: SIMD[dtype, simd_width],
    b: SIMD[dtype, simd_width],
    epsilon: Scalar[dtype],
) -> SIMD[dtype, simd_width]:
    var one = SIMD[dtype, simd_width](One[dtype].value())
    var eps = SIMD[dtype, simd_width](epsilon)

    comptime if op_code == Add:
        return a + b
    elif op_code == Subtract:
        return a - b
    elif op_code == Multiply:
        return a * b
    elif op_code == Divide:
        return a / b
    elif op_code == SIGMOID_BACKWARD:
        return b * a * (one - a)
    elif op_code == TANH_BACKWARD:
        return b * (one - a * a)
    elif op_code == SQRT_BACKWARD:
        return b * (SIMD[dtype, simd_width](0.5) * rsqrt(max(a, eps)))
    else:  # LOG_BACKWARD
        return b / max(a, eps)


@always_inline
def scalar_op[
    op_code: Int,
    dtype: DType,
](a: Scalar[dtype], b: Scalar[dtype], epsilon: Scalar[dtype],) -> Scalar[dtype]:
    var one = One[dtype].value()

    comptime if op_code == Add:
        return a + b
    elif op_code == Subtract:
        return a - b
    elif op_code == Multiply:
        return a * b
    elif op_code == Divide:
        return a / b
    elif op_code == SIGMOID_BACKWARD:
        return b * a * (one - a)
    elif op_code == TANH_BACKWARD:
        return b * (one - a * a)
    elif op_code == SQRT_BACKWARD:
        return b * (Scalar[dtype](0.5) * rsqrt(max(a, epsilon)))
    else:  # LOG_BACKWARD
        return b / max(a, epsilon)


# =============================================================================
# Shared index helpers — used by reduction_kernel and std_variance_backward_kernel
# =============================================================================

@always_inline
def output_to_input_base(
    out_idx: Int,
    in_shape: Array,
    in_strides: Array,
    reduction_axes: Array,
) -> Int:
    var remaining = out_idx
    var input_base = 0

    if len(reduction_axes) == 0:
        return 0

    for k in reversed(range(len(in_shape))):
        if k not in reduction_axes:
            var coord = remaining % in_shape[k]
            remaining //= in_shape[k]
            input_base += coord * in_strides[k]

    return input_base


@always_inline
def rank_to_reduced_offset(
    rank: Int, in_shape: Array, in_strides: Array, reduction_axes: Array
) -> Int:
    var tmp = rank
    var offset = 0
    var reduce_all = len(reduction_axes) == 0

    for k in reversed(range(len(in_shape))):
        if reduce_all or k in reduction_axes:
            var coord = tmp % in_shape[k]
            tmp //= in_shape[k]
            offset += coord * in_strides[k]

    return offset
