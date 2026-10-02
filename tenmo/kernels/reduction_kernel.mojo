# reduction_kernel.mojo
#
# GPU reduction kernels for sum, mean, and product operations.
#
# DESIGN OVERVIEW
# Three kernel functions, one unified launcher (ReductionKernel.launch[op_code]):
#
#   reduce[dtype, max_block_size, op_code]
#       Handles SUM and MEAN.
#       No dtype constraint — works for all numeric types.
#       op_code replaces the old mean: Bool flag.
#       One block per output element; threads stripe across reduced_volume.
#       Standard parallel tree reduction in shared memory.
#
#   product_reduce[dtype, max_block_size]
#       Handles PRODUCT for all numeric types.
#       No dtype constraint on the kernel signature — launcher stays clean.
#       Accumulates in float64 log-space regardless of input dtype:
#           - Overflow-safe for all practical inputs
#           - Silent wraparound (NumPy-style) is explicitly rejected
#           - Three shared memory arrays: log_abs_sum, neg_count, zero_count
#           - Zero handling: one zero → whole slice is zero
#           - Sign tracking: neg_count % 2 determines output sign
#       Precision note: int64 values beyond 2^53 (~9 * 10^15) lose mantissa
#       precision in the float64 accumulator. Results for such inputs are
#       approximate. All other types (int8 through int32, float32, float64)
#       are exact or within float64 precision.
#
#   log_sum_exp_f32 / log_sum_exp_f64
#       Dedicated log-sum-exp kernels for softmax / cross-entropy.
#       Separate from reduce — different mathematical operation.
#       Kept as dtype-specialised functions (f32/f64) matching existing design.
#
# BACKWARD SUPPORT (product only)
# Product backward requires per-element "product of all others in slice".
# This is stored or recomputed depending on a comptime flag:
#
#   store_excl_product: Bool = True  (default — faster backward, more memory)
#       excl_product buffer is computed during forward and stored in ProductArg.
#       Backward uses it directly — no second kernel launch.
#
#   store_excl_product: Bool = False  (recompute — less memory, slower backward)
#       Only the input buffer and zero_counts are stored.
#       Backward recomputes excl_product via a second kernel launch.
#
# ProductArg (defined in product_reduction.mojo — consumer side):
#   var input:          NDBuffer[dtype]          — original input, always stored
#   var excl_product:   Optional[NDBuffer[dtype]] — None if recompute=True
#   var zero_counts:    NDBuffer[DType.int32]     — per output: zeros in slice
#   var axes:           IntArray
#   var keepdims:       Bool
#   var reduced_volume: Int
#
# The kernel returns raw (Layout, Storage) pairs; the consumer assembles
# ProductArg via NDBuffer.with_layout_storage.
#
# ZERO HANDLING IN PRODUCT BACKWARD
# For a reduction slice:
#   zero_count == 0  → grad_x[i] = grad_out * excl_product[i]   (standard)
#   zero_count == 1  → grad_x[i] = grad_out * excl_product[i]   (only the
#                      zero element gets non-zero grad; others get 0 because
#                      excl_product[i≠zero] contains the zero, making it 0)
#   zero_count >= 2  → grad_x[i] = 0 for all i in slice
#
# excl_product is computed treating each element as excluded — the product
# of all others. For the single-zero case, excl_product[zero_pos] equals
# the product of all non-zero elements (correct gradient). For non-zero
# elements in the single-zero case, excl_product contains the zero, giving
# grad = 0 (correct). No special-casing needed in backward.
#
# LAUNCHER API
# ReductionKernel[dtype].launch[op_code](A_layout, A_device_state, axes, keepdims)
#     → (Layout, DeviceState[dtype])   for SUM / MEAN
# ReductionKernel[dtype].launch_product[store_excl_product](A_layout, A_device_state, ...)
#     → (out_pair, zero_counts_pair, excl_optional_pair)  for PRODUCT
#     (consumer assembles ProductArg — moved to product_reduction.mojo)
#
# NDBuffer public API (CPU + GPU unified):
#     ndb.sum(axes, keepdims)      → NDBuffer
#     ndb.mean(axes, keepdims)     → NDBuffer
#     ndb.product(axes, keepdims)  → NDBuffer   (grad arg handled at Tensor level)
#     (GPU branch dispatches through Reduction launchers above.)
#
# CHANGE MAP (vs previous reduction_kernel.mojo)
# kernels:
#   reduce[mean: Bool]  →  reduce[op_code: Int]   (SUM=mnemonics.SUM, MEAN=mnemonics.MEAN)
#   NEW: product_reduce[dtype, max_block_size]     (PRODUCT, all dtypes, log-space)
#   NEW: excl_product_kernel                       (prefix×suffix for backward)
#   log_sum_exp_f32 / log_sum_exp_f64              UNCHANGED
#
# launcher:
#   Reduction.launch[mean: Bool]  →  Reduction.launch[op_code: Int]
#   Reduction.launch_log_sum      UNCHANGED
#   NEW: Reduction.launch_product[store_excl_product: Bool]
#   NEW: Reduction.compute_excl_product
#
# backward arg:
#   NEW: ProductArg[dtype]  (implements ArgumentType)
#   MOVED: ProductArg definition → product_reduction.mojo (consumer side),
#   because NDBuffer.storage() pairs are only valid for the enclosing call.
#   The kernel returns raw (Layout, Storage) pairs; launch_product now returns
#   (out_pair, zero_counts_pair, excl_optional_pair).
#

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from max.gpu import barrier
from std.memory import AddressSpace, stack_allocation
from std.sys import simd_width_of
from std.math import log, exp, max, round

from ..shared.layout import Layout
from ..gpu.device import DeviceState
from ..shared.constants import Epsilon
from ..shared.panic import panic
from ..shared.shapes import Shape
from ..shared.mnemonics import SUM, MEAN, PRODUCT
from ..shared.array import RankArray
from ..shared.intarray import IntArray

from .kernel_helpers import (
    output_to_input_base,
    rank_to_reduced_offset,
    reduction_launch_config,
)

def reduce[
    dtype: DType,
    max_block_size: Int = 512,
    op_code: Int = SUM,
](
    out_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: RankArray,
    in_strides: RankArray,
    reduction_axes: RankArray,
    total_output_: Int64,
    reduced_volume_: Int64,
):
    """Sum / mean reduction kernel.

    One block per output element. Threads stripe across reduced_volume,
    accumulate a local sum, then perform a parallel tree reduction in
    shared memory.

    op_code must be SUM or MEAN. PRODUCT is handled by product_reduce.
    No floating point constraint — works for all numeric dtypes.

    Args:
        out_buffer:      Output pointer (total_output elements).
        in_buffer:       Input pointer (contiguous, strided via in_strides).
        in_shape:        Shape of input as RankArray.
        in_strides:      Strides of input as RankArray.
        reduction_axes:  Axes being reduced as RankArray.
        total_output_:   Number of output elements (== grid_dim).
        reduced_volume_: Number of elements reduced per output element.

    SECTION 2 — reduce kernel: SUM and MEAN
    op_code replaces the old mean: Bool flag.
    No dtype constraint — integer and floating point types both supported.
    Behaviour:
      SUM  → smem[0] written directly
      MEAN → smem[0] / reduced_volume written
    PRODUCT is NOT handled here — see product_reduce below.
    Mixing log/exp into this kernel would impose a floating point constraint
    on the entire kernel, breaking integer sum/mean.
    """
    var total_output = Int(total_output_)
    var reduced_volume = Int(reduced_volume_)
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    var smem = stack_allocation[
        max_block_size, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var block_size = block_dim.x
    var out_idx = block_idx.x

    if out_idx >= total_output:
        return

    smem[unsafe_offset=tid] = Scalar[dtype](0)

    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )
    var local = Scalar[dtype](0)
    var rank = tid

    while rank < reduced_volume:
        local += (
            in_buffer
            .unsafe_offset(input_base
            + rank_to_reduced_offset(rank, in_shape, in_strides, reduction_axes))
        )[]
        rank += block_size

    smem[unsafe_offset=tid] = local
    barrier()

    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem[unsafe_offset=tid] += smem[unsafe_offset=tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        comptime if op_code == MEAN:
            (out_buffer .unsafe_offset(out_idx))[] = smem[unsafe_offset=0] / Scalar[dtype](
                max(reduced_volume, 1)
            )
        else:  # SUM
            (out_buffer .unsafe_offset(out_idx))[] = smem[unsafe_offset=0]


def product_reduce[
    dtype: DType,
    max_block_size: Int = 512,
](
    out_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    zero_counts_buffer: Pointer[Scalar[DType.int32], MutAnyOrigin],
    in_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: RankArray,
    in_strides: RankArray,
    reduction_axes: RankArray,
    total_output_: Int64,
    reduced_volume_: Int64,
):
    """Product reduction kernel — all dtypes, float64 log-space accumulation.

    No floating point constraint on signature. Accumulates in float64
    log-space regardless of input dtype for overflow safety.

    Writes:
        out_buffer[out_idx]         = product of slice (cast to dtype)
        zero_counts_buffer[out_idx] = number of zeros in slice (int32)

    Zero counts are stored for use in backward pass (excl_product computation
    and gradient zeroing for slices with 2+ zeros).

    Precision note: int64/uint64 values beyond 2^53 are approximate.
    All other types are exact within float64 precision.

    Args:
        out_buffer:         Output pointer (total_output elements).
        zero_counts_buffer: Zero count per output element (int32).
        in_buffer:          Input pointer (strided via in_strides).
        in_shape:           Shape of input as RankArray.
        in_strides:         Strides of input as RankArray.
        reduction_axes:     Axes being reduced.
        total_output_:     Number of output elements.
        reduced_volume_:   Elements reduced per output element.

    SECTION 3 — product_reduce kernel
    Handles PRODUCT for ALL numeric dtypes without a floating point constraint
    on the kernel signature. The launcher calls this for any dtype — clean.
    Strategy: accumulate in float64 log-space, cast back to dtype at write.
    Why log-space for all dtypes:
      Direct integer multiply overflows silently (e.g. int8 wraps at 128).
      NumPy does this and it is a constant source of user confusion.
      float64 log-space gives overflow safety for all practical inputs.
    Precision contract:
      int8, int16, int32, uint8, uint16, uint32:
          All representable values fit exactly in float64 mantissa (< 2^53).
          Results are exact.
      int64, uint64:
          Values beyond 2^53 (~9 * 10^15) lose mantissa precision in float64.
          Results for such inputs are approximate. This is documented and
          unavoidable without arbitrary precision arithmetic.
      float32:
          Accumulated in float64, cast back. More precise than direct float32
          accumulation would be.
      float64:
          Native — no precision loss.
    Three shared memory arrays (all float64 or int32 — never dtype):
      smem_log:  accumulated log(abs(x)) per thread
      smem_neg:  count of negative elements per thread
      smem_zero: count of zero elements per thread
    Final write (thread 0 only):
      zero_count > 0  → output = 0
      else            → output = sign * exp(log_abs_sum), cast to dtype
    Zero count is also written to zero_counts_buffer for use in backward.
    """
    var total_output = Int(total_output_)
    var reduced_volume = Int(reduced_volume_)
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    # Three shared memory arrays — typed independently of dtype
    var smem_log = stack_allocation[
        max_block_size, Scalar[DType.float64], address_space=AddressSpace.SHARED
    ]()
    var smem_neg = stack_allocation[
        max_block_size, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()
    var smem_zero = stack_allocation[
        max_block_size, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var block_size = block_dim.x
    var out_idx = block_idx.x

    if out_idx >= total_output:
        return

    smem_log[unsafe_offset=tid] = Scalar[DType.float64](0)
    smem_neg[unsafe_offset=tid] = Scalar[DType.int32](0)
    smem_zero[unsafe_offset=tid] = Scalar[DType.int32](0)

    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )

    var local_log = Scalar[DType.float64](0)
    var local_neg = Scalar[DType.int32](0)
    var local_zero = Scalar[DType.int32](0)

    var f64_zero = Scalar[DType.float64](0)
    var f64_one = Scalar[DType.float64](1)

    var rank = tid
    while rank < reduced_volume:
        # Cast to float64 here — the only place dtype touches float64
        var val = (
            in_buffer
            .unsafe_offset(input_base
            + rank_to_reduced_offset(rank, in_shape, in_strides, reduction_axes))
        )[].cast[DType.float64]()

        if val == f64_zero:
            local_zero += Scalar[DType.int32](1)
        else:
            if val < f64_zero:
                local_neg += Scalar[DType.int32](1)
            # log(abs(val)) — safe since val != 0
            # max(x, -x) instead of abs(): abs() lowers to the llvm.nvvm.fabs
            # intrinsic, which this toolchain's NVPTX backend cannot select.
            local_log += log(max(val, -val))

        rank += block_size

    smem_log[unsafe_offset=tid] = local_log
    smem_neg[unsafe_offset=tid] = local_neg
    smem_zero[unsafe_offset=tid] = local_zero

    barrier()

    # Parallel tree reduction across all three arrays simultaneously
    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem_log[unsafe_offset=tid] += smem_log[unsafe_offset=tid + stride]
            smem_neg[unsafe_offset=tid] += smem_neg[unsafe_offset=tid + stride]
            smem_zero[unsafe_offset=tid] += smem_zero[unsafe_offset=tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        # Write zero count for backward regardless of output value
        (zero_counts_buffer .unsafe_offset(out_idx))[] = smem_zero[unsafe_offset=0]

        if smem_zero[unsafe_offset=0] > Scalar[DType.int32](0):
            # Any zero in slice → product is zero
            (out_buffer .unsafe_offset(out_idx))[] = Scalar[dtype](0)
        else:
            # sign: odd number of negatives → negative result
            var sign = Scalar[DType.float64](
                -1 if smem_neg[unsafe_offset=0] % Scalar[DType.int32](2)
                == Scalar[DType.int32](1) else 1
            )
            # Cast back to dtype — the only other place dtype is named
            # (out_buffer + out_idx)[] = (sign * exp(smem_log[0])).cast[dtype]()
            (out_buffer .unsafe_offset(out_idx))[] = _cast_result[dtype](
                sign * exp(smem_log[unsafe_offset=0])
            )


def excl_product_kernel[
    dtype: DType,
    max_block_size: Int = 512,
](
    excl_out: Pointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: RankArray,
    in_strides: RankArray,
    reduction_axes: RankArray,
    total_output_: Int64,
    reduced_volume_: Int64,
):
    """Compute product-of-all-others for each input element.

    excl_out[i] = product of all elements in i's reduction slice except i.

    Used in product backward. Accumulates in float64 log-space — same
    overflow safety and precision contract as product_reduce.

    For slices containing zeros:
        excl_out[zero_pos]  = product of non-zero others (may be non-zero)
        excl_out[non_zero]  = 0 (slice product excluding non-zero is 0)
    Backward uses zero_counts to decide whether to apply excl_out.

    Args:
        excl_out:       Output: input-shaped buffer of excl products.
        in_buffer:      Input pointer.
        in_shape:       Input shape.
        in_strides:     Input strides.
        reduction_axes: Axes being reduced.
        total_output_:  Number of output elements (== number of slices).
        reduced_volume_: Elements per slice.

    SECTION 4 — excl_product_kernel (for backward)
    Computes the "product of all others" for each element in the input,
    within its reduction slice. This is the gradient multiplier for product
    backward when there are no zeros in the slice (or exactly one zero).
    Algorithm: prefix × suffix product along each reduction axis.
    One block per output element (same as product_reduce).
    Threads stripe across reduced_volume.
    Output buffer is input-shaped: excl_product[i] = product of all elements
    in i's reduction slice except element i itself.
    For the single-zero case:
      excl_product[zero_pos]  = product of all non-zero elements (correct grad)
      excl_product[non_zero]  = 0 (contains the zero — correct, grad = 0)
    No special-casing needed in backward — zero handling falls out naturally.
    Accumulates in float64 log-space (same rationale as product_reduce).
    Sign tracked separately per element.
    """
    var total_output = Int(total_output_)
    var reduced_volume = Int(reduced_volume_)
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    var tid = thread_idx.x
    var block_size = block_dim.x
    var out_idx = block_idx.x

    if out_idx >= total_output:
        return

    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )

    var f64_zero = Scalar[DType.float64](0)
    var f64_one = Scalar[DType.float64](1)

    # Pass 1: compute total log_abs_sum, total neg_count, total zero_count
    # for this slice — same as product_reduce accumulation
    var smem_log = stack_allocation[
        max_block_size, Scalar[DType.float64], address_space=AddressSpace.SHARED
    ]()
    var smem_neg = stack_allocation[
        max_block_size, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()
    var smem_zero = stack_allocation[
        max_block_size, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()

    smem_log[unsafe_offset=tid] = f64_zero
    smem_neg[unsafe_offset=tid] = Scalar[DType.int32](0)
    smem_zero[unsafe_offset=tid] = Scalar[DType.int32](0)

    var local_log = f64_zero
    var local_neg = Scalar[DType.int32](0)
    var local_zero = Scalar[DType.int32](0)

    var rank = tid
    while rank < reduced_volume:
        var offset = rank_to_reduced_offset(
            rank, in_shape, in_strides, reduction_axes
        )
        var val = (in_buffer .unsafe_offset(input_base + offset))[].cast[DType.float64]()
        if val == f64_zero:
            local_zero += Scalar[DType.int32](1)
        else:
            if val < f64_zero:
                local_neg += Scalar[DType.int32](1)
            local_log += log(abs(val))
        rank += block_size

    smem_log[unsafe_offset=tid] = local_log
    smem_neg[unsafe_offset=tid] = local_neg
    smem_zero[unsafe_offset=tid] = local_zero
    barrier()

    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem_log[unsafe_offset=tid] += smem_log[unsafe_offset=tid + stride]
            smem_neg[unsafe_offset=tid] += smem_neg[unsafe_offset=tid + stride]
            smem_zero[unsafe_offset=tid] += smem_zero[unsafe_offset=tid + stride]
        barrier()
        stride >>= 1

    # Now smem_log[0], smem_neg[0], smem_zero[0] = totals for this slice

    # Pass 2: each thread computes excl_product for its elements
    # excl_product[i] = total_product / x[i]
    # In log-space: log_excl[i] = total_log - log(abs(x[i]))
    #               neg_excl[i] = total_neg - (1 if x[i] < 0 else 0)

    var total_log = smem_log[unsafe_offset=0]
    var total_neg = smem_neg[unsafe_offset=0]
    var total_zero = smem_zero[unsafe_offset=0]

    rank = tid
    while rank < reduced_volume:
        var offset = rank_to_reduced_offset(
            rank, in_shape, in_strides, reduction_axes
        )
        var flat_input_idx = input_base + offset
        var val = (in_buffer .unsafe_offset(flat_input_idx))[].cast[DType.float64]()

        var excl: Scalar[dtype]

        if total_zero > Scalar[DType.int32](1):
            # 2+ zeros in slice → all excl products are 0
            excl = Scalar[dtype](0)

        elif total_zero == Scalar[DType.int32](1):
            if val == f64_zero:
                # This IS the zero — excl = product of all non-zero others
                # total_log already excludes zeros (we only added log for non-zero)
                var sign = Scalar[DType.float64](
                    -1 if total_neg % Scalar[DType.int32](2)
                    == Scalar[DType.int32](1) else 1
                )
                # excl = (sign * exp(total_log)).cast[dtype]()
                excl = _cast_result[dtype](sign * exp(total_log))
            else:
                # Another element — its excl contains the zero → result is 0
                excl = Scalar[dtype](0)

        else:
            # No zeros — standard log-space division
            if val == f64_zero:
                # Shouldn't reach here (total_zero == 0), but guard
                excl = Scalar[dtype](0)
            else:
                var val_neg = Scalar[DType.int32](1 if val < f64_zero else 0)
                var excl_log = total_log - log(max(val, -val))
                var excl_neg = total_neg - val_neg
                var sign = Scalar[DType.float64](
                    -1 if excl_neg % Scalar[DType.int32](2)
                    == Scalar[DType.int32](1) else 1
                )
                # excl = (sign * exp(excl_log)).cast[dtype]()
                excl = _cast_result[dtype](sign * exp(excl_log))

        (excl_out .unsafe_offset(flat_input_idx))[] = excl
        rank += block_size


@always_inline
def _cast_result[dtype: DType](val: Scalar[DType.float64]) -> Scalar[dtype]:
    """Cast float64 log-space result back to dtype.
    Rounds to nearest integer for integral types before casting —
    prevents log/exp precision loss from producing 23 instead of 24.
    For floating point types, direct cast (no rounding needed).
    """
    comptime if dtype.is_integral():
        return round(val).cast[dtype]()
    else:
        return val.cast[dtype]()


# SECTION 5 — log_sum_exp kernels (unchanged)


def log_sum_exp_f32[
    simd_width: Int = simd_width_of[DType.float32](),
    max_block_size: Int = 512,
    epsilon: Scalar[DType.float32] = Epsilon[DType.float32].value(),
](
    out_buffer: Pointer[Scalar[DType.float32], MutAnyOrigin],
    in_buffer: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
    in_shape: RankArray,
    in_strides: RankArray,
    reduction_axes: RankArray,
    total_output_: Int64,
    reduced_volume_: Int64,
):
    var total_output = Int(total_output_)
    var reduced_volume = Int(reduced_volume_)
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    var smem = stack_allocation[
        max_block_size, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var block_size = block_dim.x
    var out_idx = block_idx.x

    if out_idx >= total_output:
        return

    smem[unsafe_offset=tid] = Scalar[DType.float32](0)
    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )
    var local = Scalar[DType.float32](0)
    var rank = tid

    while rank < reduced_volume:
        local += exp(
            (
                in_buffer
                .unsafe_offset(input_base
                + rank_to_reduced_offset(
                    rank, in_shape, in_strides, reduction_axes
                ))
            )[]
        )
        rank += block_size

    smem[unsafe_offset=tid] = local
    barrier()

    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem[unsafe_offset=tid] += smem[unsafe_offset=tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        (out_buffer .unsafe_offset(out_idx))[] = log(max(smem[unsafe_offset=0], epsilon))


def log_sum_exp_f64[
    simd_width: Int = simd_width_of[DType.float64](),
    max_block_size: Int = 512,
    epsilon: Scalar[DType.float64] = Epsilon[DType.float64].value(),
](
    out_buffer: Pointer[Scalar[DType.float64], MutAnyOrigin],
    in_buffer: Pointer[Scalar[DType.float64], ImmutAnyOrigin],
    in_shape: RankArray,
    in_strides: RankArray,
    reduction_axes: RankArray,
    total_output_: Int64,
    reduced_volume_: Int64,
):
    var total_output = Int(total_output_)
    var reduced_volume = Int(reduced_volume_)
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    var smem = stack_allocation[
        max_block_size, Scalar[DType.float64], address_space=AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var block_size = block_dim.x
    var out_idx = block_idx.x

    if out_idx >= total_output:
        return

    smem[unsafe_offset=tid] = Scalar[DType.float64](0)
    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )
    var local = Scalar[DType.float64](0)
    var rank = tid

    while rank < reduced_volume:
        local += exp(
            (
                in_buffer
                .unsafe_offset(input_base
                + rank_to_reduced_offset(
                    rank, in_shape, in_strides, reduction_axes
                ))
            )[]
        )
        rank += block_size

    smem[unsafe_offset=tid] = local
    barrier()

    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem[unsafe_offset=tid] += smem[unsafe_offset=tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        (out_buffer .unsafe_offset(out_idx))[] = log(max(smem[unsafe_offset=0], epsilon))


# SECTION 7 — Welford kernel

def welford_reduce[
    dtype: DType,
    max_block_size: Int = 512,
](
    mean_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    M2_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: RankArray,
    in_strides: RankArray,
    reduction_axes: RankArray,
    total_output_: Int64,
    reduced_volume_: Int64,
):
    """Welford online mean + M2 reduction kernel.

    One block per output element. Threads stripe across reduced_volume,
    each running a serial Welford accumulation. Tree reduction merges
    thread-local accumulators via the Welford parallel merge formula.

    Three shared memory arrays: smem_mean, smem_M2, smem_count.
    Outputs mean and M2 (unscaled sum of squared deviations).
    Caller divides M2 by n or n-1 to get variance.

    No dtype constraint — consistent with reduce kernel.

    Args:
        mean_buffer:     Output pointer for mean (total_output elements).
        M2_buffer:       Output pointer for M2 (total_output elements).
        in_buffer:       Input pointer (strided via in_strides).
        in_shape:        Shape of input as RankArray.
        in_strides:      Strides of input as RankArray.
        reduction_axes:  Axes being reduced as RankArray.
        total_output_:   Number of output elements (== grid_dim).
        reduced_volume_: Number of elements reduced per output element.

    DESIGN NOTES
    Welford online algorithm — single pass, numerically stable:
      M_0 = 0, S_0 = 0
      for k = 1..n:
          delta  = x_k - M_{k-1}
          M_k    = M_{k-1} + delta / k
          delta2 = x_k - M_k
          S_k    = S_{k-1} + delta * delta2
      mean     = M_n
      var_pop  = S_n / n          (biased)
      var_samp = S_n / (n - 1)    (unbiased)
    Welford parallel merge of two accumulators (a, b):
      combined.count = a.count + b.count
      delta          = b.mean - a.mean
      combined.mean  = a.mean + delta * b.count / combined.count
      combined.M2    = a.M2 + b.M2 + delta^2 * a.count * b.count / combined.count
    GPU kernel:
      One block per output element (same as reduce).
      Threads stripe across reduced_volume — each thread runs serial Welford.
      Three shared memory arrays: smem_mean, smem_M2, smem_count.
      Tree reduction using Welford merge — NOT simple addition.
      Returns two output buffers: mean and M2 (unscaled).
      Caller divides M2 by n or n-1 for variance.
      Std: caller takes sqrt of variance output.
    No dtype constraint at kernel level — consistent with reduce kernel.
    Floating point arithmetic is inherent to Welford math, not enforced here.
    Index helpers reused verbatim:
      output_to_input_base
      rank_to_reduced_offset
    Launcher returns (mean pair, M2 pair).
    Consumer divides M2 by n/n-1 and optionally sqrts for std.
    """
    var total_output = Int(total_output_)
    var reduced_volume = Int(reduced_volume_)
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    # Three shared memory arrays — mean, M2, count
    var smem_mean = stack_allocation[
        max_block_size, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var smem_M2 = stack_allocation[
        max_block_size, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var smem_count = stack_allocation[
        max_block_size, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var block_size = block_dim.x
    var out_idx = block_idx.x

    if out_idx >= total_output:
        return

    # Initialize shared memory
    smem_mean[unsafe_offset=tid] = Scalar[dtype](0)
    smem_M2[unsafe_offset=tid] = Scalar[dtype](0)
    smem_count[unsafe_offset=tid] = Int32(0)

    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )

    # Serial Welford over this thread's stripe
    var local_mean = Scalar[dtype](0)
    var local_M2 = Scalar[dtype](0)
    var local_count = Int32(0)
    var rank = tid

    while rank < reduced_volume:
        var x = (
            in_buffer
            .unsafe_offset(input_base
            + rank_to_reduced_offset(rank, in_shape, in_strides, reduction_axes))
        )[]
        local_count += Int32(1)
        var delta = x - local_mean
        local_mean += delta / Scalar[dtype](Int(local_count))
        var delta2 = x - local_mean
        local_M2 += delta * delta2
        rank += block_size

    smem_mean[unsafe_offset=tid] = local_mean
    smem_M2[unsafe_offset=tid] = local_M2
    smem_count[unsafe_offset=tid] = local_count
    barrier()

    # Tree reduction via Welford parallel merge
    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            var a_mean = smem_mean[unsafe_offset=tid]
            var a_M2 = smem_M2[unsafe_offset=tid]
            var a_count = smem_count[unsafe_offset=tid]
            var b_mean = smem_mean[unsafe_offset=tid + stride]
            var b_M2 = smem_M2[unsafe_offset=tid + stride]
            var b_count = smem_count[unsafe_offset=tid + stride]

            var combined_count = a_count + b_count
            if combined_count > Int32(0):
                var delta = b_mean - a_mean
                var b_count_f = Scalar[dtype](Int(b_count))
                var combined_count_f = Scalar[dtype](Int(combined_count))
                smem_mean[unsafe_offset=tid] = a_mean + delta * b_count_f / combined_count_f
                smem_M2[unsafe_offset=tid] = (
                    a_M2
                    + b_M2
                    + delta
                    * delta
                    * Scalar[dtype](Int(a_count))
                    * b_count_f
                    / combined_count_f
                )
                smem_count[unsafe_offset=tid] = combined_count
        barrier()
        stride >>= 1

    if tid == 0:
        (mean_buffer .unsafe_offset(out_idx))[] = smem_mean[unsafe_offset=0]
        (M2_buffer .unsafe_offset(out_idx))[] = smem_M2[unsafe_offset=0]


@fieldwise_init
struct ReductionKernel[dtype: DType = DType.float32](
    ImplicitlyCopyable, RegisterPassable
):
    """SECTION 8 — Reduction launcher
    Unified launcher for SUM, MEAN, PRODUCT via op_code.
    launch[op_code]  → (Layout, Storage)  (SUM / MEAN — result only)
    launch_product   → (out_pair, zero_pair, excl_optional_pair)  (PRODUCT)
    launch_log_sum   → (Layout, Storage)  (log-sum-exp)
    Sum/mean call sites:
      BEFORE: Reduction.launch[mean=False](A, axes, keepdims)
              Reduction.launch[mean=True](A, axes, keepdims)
      AFTER:  Reduction.launch[SUM](A, axes, keepdims)
              Reduction.launch[MEAN](A, axes, keepdims)
    """
    # SUM / MEAN

    @staticmethod
    def launch[
        op_code: Int = SUM,
        max_block_width: Int = 512,
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        """Launch sum or mean reduction.

        op_code must be SUM or MEAN. For PRODUCT use launch_product.

        Args:
            A_layout:         Input layout.
            A_device_state:        Input storage. Must be on GPU.
            normalized_axes: Validated, normalised reduction axes.
            keepdims:        Whether to keep reduced dimensions.
            sync:            Whether to sync GPU after operation.

        Returns:
            (Layout, Storage) — result with reduction applied.
        """
        comptime assert op_code == SUM or op_code == MEAN, (
            "launch[op_code] only accepts SUM or MEAN — use launch_product for"
            " PRODUCT"
        )
        var shape_A = A_layout.shape
        var strides_A = A_layout.strides
        var output_shape = shape_A.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )

        var normalized_axes_copy = normalized_axes
        if len(normalized_axes_copy) == 0:
            normalized_axes_copy = IntArray(len(shape_A))
            for i in range(len(shape_A)):
                normalized_axes_copy[i] = i

        var reduction_axes: RankArray = RankArray(normalized_axes_copy)
        var reduced_shape = shape_A.reduced_shape(normalized_axes)
        var in_shape: RankArray = shape_A.array()
        var in_strides: RankArray = strides_A.array()
        var total_output: Int = output_shape.product()
        var reduced_volume: Int = reduced_shape.product()

        var (threads_per_block, num_blocks) = Self.launch_config[
            max_block_width
        ](total_output, reduced_volume)

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )

        ref A_buffer = A_device_state.device_buffer()

        var compiled_func = device_context.compile_function[
            reduce[Self.dtype, max_block_width, op_code],
        ]()

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            in_shape,
            in_strides,
            reduction_axes,
            Int64(total_output),
            Int64(reduced_volume),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()
        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return (
            Layout(output_shape),
            device_state^,
        )

    # PRODUCT

    @staticmethod
    def launch_product[
        store_excl_product: Bool = True,
        max_block_width: Int = 512,
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
        sync: Bool = False,
    ) raises -> Tuple[
        Tuple[Layout, DeviceState[Self.dtype]],
        Tuple[Layout, DeviceState[DType.int32]],
        Optional[Tuple[Layout, DeviceState[Self.dtype]]],
    ]:
        """Launch product reduction.

        Runs product_reduce kernel (log-space, all dtypes, overflow-safe).
        Optionally computes and stores excl_product for backward.

        store_excl_product=True  (default):
            Runs excl_product_kernel immediately after product_reduce.
            Returns it as Some(pair).
            Backward uses it directly — no second kernel launch needed.
            Memory cost: one input-shaped buffer of dtype.

        store_excl_product=False:
            Returns None for excl_product.
            Backward recomputes excl_product via excl_product_kernel.
            No extra memory cost, but backward is slower.

        Args:
            A_layout:         Input layout.
            A_device_state:        Input storage. Must be on GPU.
            normalized_axes: Validated, normalised reduction axes.
            keepdims:        Whether to keep reduced dimensions.
            sync:            Whether to sync GPU after operation.

        Returns:
            (out_pair, zero_counts_pair, excl_optional_pair) where out is
            output-shaped, zero_counts is int32 output-shaped (always stored
            for backward), and excl_product is input-shaped (Some iff
            store_excl_product=True). The consumer assembles ProductArg.
        """
        var shape_A = A_layout.shape
        var strides_A = A_layout.strides
        var output_shape = shape_A.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )

        var normalized_axes_copy = normalized_axes
        if len(normalized_axes_copy) == 0:
            normalized_axes_copy = IntArray(len(shape_A))
            for i in range(len(shape_A)):
                normalized_axes_copy[i] = i

        var reduction_axes: RankArray = RankArray(normalized_axes_copy)
        var reduced_shape = shape_A.reduced_shape(normalized_axes)
        var in_shape: RankArray = shape_A.array()
        var in_strides: RankArray = strides_A.array()
        var total_output: Int = output_shape.product()
        var reduced_volume: Int = reduced_shape.product()
        var input_numels: Int = A_layout.numel()

        var (threads_per_block, num_blocks) = Self.launch_config[
            max_block_width
        ](total_output, reduced_volume)

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        # Output buffer (dtype)
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )
        # Zero counts buffer (int32 — always stored for backward)
        var zero_counts_buffer = device_context.enqueue_create_buffer[
            DType.int32
        ](total_output)

        ref A_buffer = A_device_state.device_buffer()

        # Kernel 1: product_reduce
        var compiled_product = device_context.compile_function[
            product_reduce[Self.dtype, max_block_width],
        ]()

        device_context.enqueue_function(
            compiled_product,
            result_buffer,
            zero_counts_buffer,
            A_buffer,
            in_shape,
            in_strides,
            reduction_axes,
            Int64(total_output),
            Int64(reduced_volume),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        var out_pair = (
            Layout(output_shape),
            result_state^,
        )

        var zero_state = DeviceState[DType.int32](zero_counts_buffer^, gpu)
        var zero_pair = (
            Layout(output_shape),
            zero_state^,
        )

        # Kernel 2: excl_product (only if store_excl_product=True)
        var excl_optional: Optional[Tuple[Layout, DeviceState[Self.dtype]]] = None

        comptime if store_excl_product:
            var excl_buffer = device_context.enqueue_create_buffer[Self.dtype](
                input_numels
            )

            # excl_product uses same launch config as product_reduce
            var (excl_threads, excl_blocks) = Self.launch_config[
                max_block_width
            ](total_output, reduced_volume)

            var compiled_excl = device_context.compile_function[
                excl_product_kernel[Self.dtype, max_block_width],
            ]()

            device_context.enqueue_function(
                compiled_excl,
                excl_buffer,
                A_buffer,
                in_shape,
                in_strides,
                reduction_axes,
            Int64(total_output),
            Int64(reduced_volume),
                grid_dim=excl_blocks,
                block_dim=excl_threads,
            )

            if sync:
                device_context.synchronize()

            var excl_state = DeviceState[Self.dtype](excl_buffer^, gpu)
            excl_optional = Optional(
                (
                    Layout(shape_A),
                    excl_state^,
                )
            )

        return (out_pair, zero_pair, excl_optional^)

    @staticmethod
    def compute_excl_product(
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var shape_A = A_layout.shape
        var strides_A = A_layout.strides
        var output_shape = shape_A.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var normalized_axes_copy = normalized_axes
        if len(normalized_axes_copy) == 0:
            normalized_axes_copy = IntArray(len(shape_A))
            for i in range(len(shape_A)):
                normalized_axes_copy[i] = i

        var reduction_axes: RankArray = RankArray(normalized_axes_copy)
        var in_shape: RankArray = shape_A.array()
        var in_strides: RankArray = strides_A.array()
        var total_output: Int = output_shape.product()
        var reduced_volume: Int = shape_A.reduced_shape(
            normalized_axes
        ).product()
        var input_numels: Int = A_layout.numel()

        var (threads_per_block, num_blocks) = Self.launch_config[512](
            total_output, reduced_volume
        )

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        var excl_buffer = device_context.enqueue_create_buffer[Self.dtype](
            input_numels
        )
        ref A_buffer = A_device_state.device_buffer()

        var compiled = device_context.compile_function[
            excl_product_kernel[Self.dtype, 512],
        ]()

        device_context.enqueue_function(
            compiled,
            excl_buffer,
            A_buffer,
            in_shape,
            in_strides,
            reduction_axes,
            Int64(total_output),
            Int64(reduced_volume),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var excl_state = DeviceState[Self.dtype](excl_buffer^, gpu)
        return (
            Layout(shape_A),
            excl_state^,
        )

    # LOG SUM EXP (unchanged)

    @staticmethod
    def launch_log_sum[
        max_block_width: Int = 512,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var shape_A = A_layout.shape
        var strides_A = A_layout.strides
        var output_shape = shape_A.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )

        var normalized_axes_copy = normalized_axes
        if len(normalized_axes_copy) == 0:
            normalized_axes_copy = IntArray(len(shape_A))
            for i in range(len(shape_A)):
                normalized_axes_copy[i] = i

        var reduction_axes: RankArray = RankArray(normalized_axes_copy)
        var reduced_shape = shape_A.reduced_shape(normalized_axes)
        var in_shape: RankArray = shape_A.array()
        var in_strides: RankArray = strides_A.array()
        var total_output: Int = output_shape.product()
        var reduced_volume: Int = reduced_shape.product()

        var (threads_per_block, num_blocks) = Self.launch_config[
            max_block_width
        ](total_output, reduced_volume)

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )
        ref A_buffer = A_device_state.device_buffer()

        comptime if Self.dtype == DType.float32:
            var compiled_func = device_context.compile_function[
                log_sum_exp_f32[
                    max_block_size=max_block_width,
                    epsilon=epsilon.cast[DType.float32](),
                ],
            ]()
            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                in_shape,
                in_strides,
                reduction_axes,
            Int64(total_output),
            Int64(reduced_volume),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        elif Self.dtype == DType.float64:
            var compiled_func = device_context.compile_function[
                log_sum_exp_f64[
                    max_block_size=max_block_width,
                    epsilon=epsilon.cast[DType.float64](),
                ],
            ]()
            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                in_shape,
                in_strides,
                reduction_axes,
            Int64(total_output),
            Int64(reduced_volume),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        else:
            panic(
                "Reduction.launch_log_sum: only float32 and float64 supported"
            )

        if sync:
            device_context.synchronize()
        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return (
            Layout(output_shape),
            device_state^,
        )

    @staticmethod
    def launch_welford[
        max_block_width: Int = 512,
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
        sync: Bool = False,
    ) raises -> Tuple[
        Tuple[Layout, DeviceState[Self.dtype]], Tuple[Layout, DeviceState[Self.dtype]]
    ]:
        """Launch Welford mean + M2 reduction on GPU.

        Returns ((mean_layout, mean_storage), (M2_layout, M2_storage)). M2 is
        the unscaled sum of squared deviations. Divide by n for population
        variance, n-1 for sample.

        Args:
            A_layout:         Input layout.
            A_device_state:        Input storage. Must be on GPU.
            normalized_axes: Validated, normalised reduction axes.
            keepdims:        Whether to keep reduced dimensions.
            sync:            Whether to sync GPU after operation.

        Returns:
            (mean pair, M2 pair), same shape as sum/mean output.
        """
        var shape_A = A_layout.shape
        var strides_A = A_layout.strides
        var output_shape = shape_A.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )

        var normalized_axes_copy = normalized_axes
        if len(normalized_axes_copy) == 0:
            normalized_axes_copy = IntArray(len(shape_A))
            for i in range(len(shape_A)):
                normalized_axes_copy[i] = i

        var reduction_axes: RankArray = RankArray(normalized_axes_copy)
        var reduced_shape = shape_A.reduced_shape(normalized_axes)
        var in_shape: RankArray = shape_A.array()
        var in_strides: RankArray = strides_A.array()
        var total_output: Int = output_shape.product()
        var reduced_volume: Int = reduced_shape.product()

        var (threads_per_block, num_blocks) = Self.launch_config[
            max_block_width
        ](total_output, reduced_volume)

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        # Two output buffers — mean and M2
        var mean_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )
        var M2_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )

        ref A_buffer = A_device_state.device_buffer()

        var compiled_func = device_context.compile_function[
            welford_reduce[Self.dtype, max_block_width],
        ]()

        device_context.enqueue_function(
            compiled_func,
            mean_buffer,
            M2_buffer,
            A_buffer,
            in_shape,
            in_strides,
            reduction_axes,
            Int64(total_output),
            Int64(reduced_volume),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var mean_state = DeviceState[Self.dtype](mean_buffer^, gpu)
        var M2_state = DeviceState[Self.dtype](M2_buffer^, gpu)

        return (
            (
                Layout(output_shape),
                mean_state^,
            ),
            (
                Layout(output_shape),
                M2_state^,
            ),
        )

    # launch_config

    @staticmethod
    def launch_config[
        max_block_size: Int
    ](total_output: Int, reduced_volume: Int) -> Tuple[Int, Int]:
        return reduction_launch_config[max_block_size](total_output, reduced_volume)
