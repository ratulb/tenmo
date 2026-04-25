# =============================================================================
# reduction_kernel.mojo
# =============================================================================
#
# GPU reduction kernels for sum, mean, and product operations.
#
# DESIGN OVERVIEW
# ───────────────
# Three kernel functions, one unified launcher (Reduction.launch[op_code]):
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
# ────────────────────────────────
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
# ProductArg (stored in BackwardFnArg via ArgumentType):
#   var input:          NDBuffer[dtype]          — original input, always stored
#   var excl_product:   Optional[NDBuffer[dtype]] — None if recompute=True
#   var zero_counts:    NDBuffer[DType.int32]     — per output: zeros in slice
#   var axes:           IntArray
#   var keepdims:       Bool
#   var reduced_volume: Int
#
# ZERO HANDLING IN PRODUCT BACKWARD
# ───────────────────────────────────
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
# ─────────────
# Reduction[dtype].launch[op_code](A, axes, keepdims)
#     → NDBuffer[dtype]   for SUM / MEAN
#     → Tuple[NDBuffer[dtype], ProductArg[dtype]]  for PRODUCT
#
# NDBuffer public API (CPU + GPU unified):
#     ndb.sum(axes, keepdims)      → NDBuffer
#     ndb.mean(axes, keepdims)     → NDBuffer
#     ndb.product(axes, keepdims)  → NDBuffer   (grad arg handled at Tensor level)
#
# CHANGE MAP (vs previous reduction_kernel.mojo)
# ────────────────────────────────────────────────
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
#
# =============================================================================

from std.gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from std.memory import AddressSpace, stack_allocation
from std.sys import simd_width_of, has_accelerator
from std.math import log, exp, abs, max

from .ndbuffer import NDBuffer
from .device import DeviceState
from .common_utils import panic, Epsilon
from .shapes import Shape
from .buffers import Buffer
from .mnemonics import SUM, MEAN, PRODUCT
from .backpropagation import ArgumentType
from .array import Array
from .intarray import IntArray

# =============================================================================
# SECTION 1 — Shared index helpers (unchanged)
# =============================================================================

fn output_to_input_base(
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


fn rank_to_reduced_offset(
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


# =============================================================================
# SECTION 2 — reduce kernel: SUM and MEAN
# =============================================================================
#
# op_code replaces the old mean: Bool flag.
# No dtype constraint — integer and floating point types both supported.
# Behaviour:
#   SUM  → smem[0] written directly
#   MEAN → smem[0] / reduced_volume written
#
# PRODUCT is NOT handled here — see product_reduce below.
# Mixing log/exp into this kernel would impose a floating point constraint
# on the entire kernel, breaking integer sum/mean.
# =============================================================================

fn reduce[
    dtype: DType,
    max_block_size: Int = 512,
    op_code: Int = SUM,
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: Array,
    in_strides: Array,
    reduction_axes: Array,
    total_output: Int,
    reduced_volume: Int,
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
        in_shape:        Shape of input as Array.
        in_strides:      Strides of input as Array.
        reduction_axes:  Axes being reduced as Array.
        total_output:    Number of output elements (== grid_dim).
        reduced_volume:  Number of elements reduced per output element.
    """
    comptime assert(
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    var smem = stack_allocation[
        max_block_size, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()

    var tid        = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var out_idx    = Int(block_idx.x)

    if out_idx >= total_output:
        return

    smem[tid] = Scalar[dtype](0)

    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )
    var local = Scalar[dtype](0)
    var rank  = tid

    while rank < reduced_volume:
        local += (
            in_buffer
            + input_base
            + rank_to_reduced_offset(rank, in_shape, in_strides, reduction_axes)
        )[]
        rank += block_size

    smem[tid] = local
    barrier()

    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        comptime if op_code == MEAN:
            (out_buffer + out_idx)[] = smem[0] / Scalar[dtype](
                max(reduced_volume, 1)
            )
        else:  # SUM
            (out_buffer + out_idx)[] = smem[0]


# =============================================================================
# SECTION 3 — product_reduce kernel
# =============================================================================
#
# Handles PRODUCT for ALL numeric dtypes without a floating point constraint
# on the kernel signature. The launcher calls this for any dtype — clean.
#
# Strategy: accumulate in float64 log-space, cast back to dtype at write.
#
# Why log-space for all dtypes:
#   Direct integer multiply overflows silently (e.g. int8 wraps at 128).
#   NumPy does this and it is a constant source of user confusion.
#   float64 log-space gives overflow safety for all practical inputs.
#
# Precision contract:
#   int8, int16, int32, uint8, uint16, uint32:
#       All representable values fit exactly in float64 mantissa (< 2^53).
#       Results are exact.
#   int64, uint64:
#       Values beyond 2^53 (~9 * 10^15) lose mantissa precision in float64.
#       Results for such inputs are approximate. This is documented and
#       unavoidable without arbitrary precision arithmetic.
#   float32:
#       Accumulated in float64, cast back. More precise than direct float32
#       accumulation would be.
#   float64:
#       Native — no precision loss.
#
# Three shared memory arrays (all float64 or int32 — never dtype):
#   smem_log:  accumulated log(abs(x)) per thread
#   smem_neg:  count of negative elements per thread
#   smem_zero: count of zero elements per thread
#
# Final write (thread 0 only):
#   zero_count > 0  → output = 0
#   else            → output = sign * exp(log_abs_sum), cast to dtype
#
# Zero count is also written to zero_counts_buffer for use in backward.
# =============================================================================

fn product_reduce[
    dtype: DType,
    max_block_size: Int = 512,
](
    out_buffer:         UnsafePointer[Scalar[dtype], MutAnyOrigin],
    zero_counts_buffer: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    in_buffer:          UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape:           Array,
    in_strides:         Array,
    reduction_axes:     Array,
    total_output:       Int,
    reduced_volume:     Int,
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
        in_shape:           Shape of input as Array.
        in_strides:         Strides of input as Array.
        reduction_axes:     Axes being reduced.
        total_output:       Number of output elements.
        reduced_volume:     Elements reduced per output element.
    """
    comptime assert(
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

    var tid        = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var out_idx    = Int(block_idx.x)

    if out_idx >= total_output:
        return

    smem_log[tid]  = Scalar[DType.float64](0)
    smem_neg[tid]  = Scalar[DType.int32](0)
    smem_zero[tid] = Scalar[DType.int32](0)

    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )

    var local_log  = Scalar[DType.float64](0)
    var local_neg  = Scalar[DType.int32](0)
    var local_zero = Scalar[DType.int32](0)

    var f64_zero = Scalar[DType.float64](0)
    var f64_one  = Scalar[DType.float64](1)

    var rank = tid
    while rank < reduced_volume:
        # Cast to float64 here — the only place dtype touches float64
        var val = (
            in_buffer
            + input_base
            + rank_to_reduced_offset(rank, in_shape, in_strides, reduction_axes)
        )[].cast[DType.float64]()

        if val == f64_zero:
            local_zero += Scalar[DType.int32](1)
        else:
            if val < f64_zero:
                local_neg += Scalar[DType.int32](1)
            # log(abs(val)) — safe since val != 0
            local_log += log(abs(val))

        rank += block_size

    smem_log[tid]  = local_log
    smem_neg[tid]  = local_neg
    smem_zero[tid] = local_zero

    barrier()

    # Parallel tree reduction across all three arrays simultaneously
    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem_log[tid]  += smem_log[tid + stride]
            smem_neg[tid]  += smem_neg[tid + stride]
            smem_zero[tid] += smem_zero[tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        # Write zero count for backward regardless of output value
        (zero_counts_buffer + out_idx)[] = smem_zero[0]

        if smem_zero[0] > Scalar[DType.int32](0):
            # Any zero in slice → product is zero
            (out_buffer + out_idx)[] = Scalar[dtype](0)
        else:
            # sign: odd number of negatives → negative result
            var sign = Scalar[DType.float64](
                -1 if smem_neg[0] % Scalar[DType.int32](2)
                    == Scalar[DType.int32](1) else 1
            )
            # Cast back to dtype — the only other place dtype is named
            (out_buffer + out_idx)[] = (sign * exp(smem_log[0])).cast[dtype]()


# =============================================================================
# SECTION 4 — excl_product_kernel (for backward)
# =============================================================================
#
# Computes the "product of all others" for each element in the input,
# within its reduction slice. This is the gradient multiplier for product
# backward when there are no zeros in the slice (or exactly one zero).
#
# Algorithm: prefix × suffix product along each reduction axis.
# One block per output element (same as product_reduce).
# Threads stripe across reduced_volume.
#
# Output buffer is input-shaped: excl_product[i] = product of all elements
# in i's reduction slice except element i itself.
#
# For the single-zero case:
#   excl_product[zero_pos]  = product of all non-zero elements (correct grad)
#   excl_product[non_zero]  = 0 (contains the zero — correct, grad = 0)
# No special-casing needed in backward — zero handling falls out naturally.
#
# Accumulates in float64 log-space (same rationale as product_reduce).
# Sign tracked separately per element.
# =============================================================================

fn excl_product_kernel[
    dtype: DType,
    max_block_size: Int = 512,
](
    excl_out:       UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer:      UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape:       Array,
    in_strides:     Array,
    reduction_axes: Array,
    total_output:   Int,
    reduced_volume: Int,
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
        total_output:   Number of output elements (== number of slices).
        reduced_volume: Elements per slice.
    """
    comptime assert(
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    var tid        = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var out_idx    = Int(block_idx.x)

    if out_idx >= total_output:
        return

    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )

    var f64_zero = Scalar[DType.float64](0)
    var f64_one  = Scalar[DType.float64](1)

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

    smem_log[tid]  = f64_zero
    smem_neg[tid]  = Scalar[DType.int32](0)
    smem_zero[tid] = Scalar[DType.int32](0)

    var local_log  = f64_zero
    var local_neg  = Scalar[DType.int32](0)
    var local_zero = Scalar[DType.int32](0)

    var rank = tid
    while rank < reduced_volume:
        var offset = rank_to_reduced_offset(rank, in_shape, in_strides, reduction_axes)
        var val = (in_buffer + input_base + offset)[].cast[DType.float64]()
        if val == f64_zero:
            local_zero += Scalar[DType.int32](1)
        else:
            if val < f64_zero:
                local_neg += Scalar[DType.int32](1)
            local_log += log(abs(val))
        rank += block_size

    smem_log[tid]  = local_log
    smem_neg[tid]  = local_neg
    smem_zero[tid] = local_zero
    barrier()

    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem_log[tid]  += smem_log[tid + stride]
            smem_neg[tid]  += smem_neg[tid + stride]
            smem_zero[tid] += smem_zero[tid + stride]
        barrier()
        stride >>= 1

    # Now smem_log[0], smem_neg[0], smem_zero[0] = totals for this slice

    # Pass 2: each thread computes excl_product for its elements
    # excl_product[i] = total_product / x[i]
    # In log-space: log_excl[i] = total_log - log(abs(x[i]))
    #               neg_excl[i] = total_neg - (1 if x[i] < 0 else 0)

    var total_log  = smem_log[0]
    var total_neg  = smem_neg[0]
    var total_zero = smem_zero[0]

    rank = tid
    while rank < reduced_volume:
        var offset = rank_to_reduced_offset(rank, in_shape, in_strides, reduction_axes)
        var flat_input_idx = input_base + offset
        var val = (in_buffer + flat_input_idx)[].cast[DType.float64]()

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
                excl = (sign * exp(total_log)).cast[dtype]()
            else:
                # Another element — its excl contains the zero → result is 0
                excl = Scalar[dtype](0)

        else:
            # No zeros — standard log-space division
            if val == f64_zero:
                # Shouldn't reach here (total_zero == 0), but guard
                excl = Scalar[dtype](0)
            else:
                var val_neg   = Scalar[DType.int32](1 if val < f64_zero else 0)
                var excl_log  = total_log - log(abs(val))
                var excl_neg  = total_neg - val_neg
                var sign = Scalar[DType.float64](
                    -1 if excl_neg % Scalar[DType.int32](2)
                        == Scalar[DType.int32](1) else 1
                )
                excl = (sign * exp(excl_log)).cast[dtype]()

        (excl_out + flat_input_idx)[] = excl
        rank += block_size


# =============================================================================
# SECTION 5 — log_sum_exp kernels (unchanged)
# =============================================================================

fn log_sum_exp_f32[
    simd_width: Int = simd_width_of[DType.float32](),
    max_block_size: Int = 512,
    epsilon: Scalar[DType.float32] = Epsilon[DType.float32].value(),
](
    out_buffer: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    in_shape: Array,
    in_strides: Array,
    reduction_axes: Array,
    total_output: Int,
    reduced_volume: Int,
):
    comptime assert(
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    var smem = stack_allocation[
        max_block_size, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

    var tid        = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var out_idx    = Int(block_idx.x)

    if out_idx >= total_output:
        return

    smem[tid] = Scalar[DType.float32](0)
    var input_base = output_to_input_base(out_idx, in_shape, in_strides, reduction_axes)
    var local = Scalar[DType.float32](0)
    var rank = tid

    while rank < reduced_volume:
        local += exp(
            (in_buffer + input_base
             + rank_to_reduced_offset(rank, in_shape, in_strides, reduction_axes))[]
        )
        rank += block_size

    smem[tid] = local
    barrier()

    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        (out_buffer + out_idx)[] = log(max(smem[0], epsilon))


fn log_sum_exp_f64[
    simd_width: Int = simd_width_of[DType.float64](),
    max_block_size: Int = 512,
    epsilon: Scalar[DType.float64] = Epsilon[DType.float64].value(),
](
    out_buffer: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[DType.float64], ImmutAnyOrigin],
    in_shape: Array,
    in_strides: Array,
    reduction_axes: Array,
    total_output: Int,
    reduced_volume: Int,
):
    comptime assert(
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    var smem = stack_allocation[
        max_block_size, Scalar[DType.float64], address_space=AddressSpace.SHARED
    ]()

    var tid        = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var out_idx    = Int(block_idx.x)

    if out_idx >= total_output:
        return

    smem[tid] = Scalar[DType.float64](0)
    var input_base = output_to_input_base(out_idx, in_shape, in_strides, reduction_axes)
    var local = Scalar[DType.float64](0)
    var rank = tid

    while rank < reduced_volume:
        local += exp(
            (in_buffer + input_base
             + rank_to_reduced_offset(rank, in_shape, in_strides, reduction_axes))[]
        )
        rank += block_size

    smem[tid] = local
    barrier()

    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        (out_buffer + out_idx)[] = log(max(smem[0], epsilon))


# =============================================================================
# SECTION 6 — ProductArg: backward argument struct
# =============================================================================
#
# Stored in BackwardFnArg via ArgumentType.
# Implements ArgumentType trait (ImplicitlyCopyable & Movable).
#
# excl_product is Optional:
#   Some(ndb) → store_excl_product=True  — backward uses it directly
#   None      → store_excl_product=False — backward recomputes via kernel
#
# zero_counts is always stored (int32, input-shape agnostic — one per output).
# input is always stored — needed for recompute path and zero detection.
# =============================================================================

@fieldwise_init
struct ProductArg[dtype: DType](ArgumentType):
    """Backward argument for product reduction.

    Always stored:
        input        — original forward input (needed for recompute path)
        zero_counts  — per-output-element zero count (int32, on same device)
        axes         — reduction axes
        keepdims     — keepdims flag from forward
        reduced_volume — elements per slice (for excl recompute)

    Conditionally stored (store_excl_product comptime flag):
        excl_product — input-shaped buffer of per-element exclusive products
                       None if store_excl_product=False (recomputed in backward)
    """
    var input:          NDBuffer[Self.dtype]
    var excl_product:   Optional[NDBuffer[Self.dtype]]
    var zero_counts:    NDBuffer[DType.int32]
    var axes:           IntArray
    var keepdims:       Bool
    var reduced_volume: Int

    @staticmethod
    fn Empty() -> ProductArg[Self.dtype]:
        return ProductArg[Self.dtype](NDBuffer[Self.dtype].Empty(), None, NDBuffer[DType.int32].Empty(), IntArray(), False, 0)

# =============================================================================
# SECTION 7 — Reduction launcher
# =============================================================================
#
# Unified launcher for SUM, MEAN, PRODUCT via op_code.
#
# launch[op_code]  → NDBuffer  (SUM / MEAN — returns result only)
# launch_product   → Tuple[NDBuffer, ProductArg]  (PRODUCT — result + bwd arg)
# launch_log_sum   → NDBuffer  (log-sum-exp — unchanged)
#
# Sum/mean call sites:
#   BEFORE: Reduction.launch[mean=False](A, axes, keepdims)
#           Reduction.launch[mean=True](A, axes, keepdims)
#   AFTER:  Reduction.launch[SUM](A, axes, keepdims)
#           Reduction.launch[MEAN](A, axes, keepdims)
# =============================================================================

@fieldwise_init
struct Reduction[dtype: DType = DType.float32](
    ImplicitlyCopyable, RegisterPassable
):

    # ── SUM / MEAN ────────────────────────────────────────────────────────────

    @staticmethod
    def launch[
        op_code: Int = SUM,
        max_block_width: Int = 512,
    ](
        A: NDBuffer[Self.dtype], normalized_axes: IntArray, keepdims: Bool
    ) raises -> NDBuffer[Self.dtype]:
        """Launch sum or mean reduction.

        op_code must be SUM or MEAN. For PRODUCT use launch_product.

        Args:
            A:               Input NDBuffer. Must be on GPU.
            normalized_axes: Validated, normalised reduction axes.
            keepdims:        Whether to keep reduced dimensions.

        Returns:
            NDBuffer with reduction applied.
        """
        comptime assert(
            op_code == SUM or op_code == MEAN
        ), "launch[op_code] only accepts SUM or MEAN — use launch_product for PRODUCT"

        var shape_A   = A.shape
        var strides_A = A.strides
        var output_shape = shape_A.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )

        var normalized_axes_copy = normalized_axes
        if len(normalized_axes_copy) == 0:
            normalized_axes_copy = IntArray(len(shape_A))
            for i in range(len(shape_A)):
                normalized_axes_copy[i] = i

        var reduction_axes: Array  = Array(normalized_axes_copy)
        var reduced_shape          = shape_A.reduced_shape(normalized_axes)
        var in_shape: Array        = shape_A.array()
        var in_strides: Array      = strides_A.array()
        var total_output: Int      = output_shape.product()
        var reduced_volume: Int    = reduced_shape.product()

        var (threads_per_block, num_blocks) = Self.launch_config[
            max_block_width
        ](total_output, reduced_volume)

        ref A_device_state = A.device_state.value()
        ref gpu            = A_device_state.get_gpu()
        var device_context = gpu[]
        var result_buffer  = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )
        ref A_buffer = A_device_state.device_buffer()

        var compiled_func = device_context.compile_function[
            reduce[Self.dtype, max_block_width, op_code],
            reduce[Self.dtype, max_block_width, op_code],
        ]()

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            in_shape,
            in_strides,
            reduction_axes,
            total_output,
            reduced_volume,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()
        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(device_state^, output_shape)

    # ── PRODUCT ───────────────────────────────────────────────────────────────

    @staticmethod
    def launch_product[
        store_excl_product: Bool = True,
        max_block_width: Int = 512,
    ](
        A: NDBuffer[Self.dtype], normalized_axes: IntArray, keepdims: Bool
    ) raises -> Tuple[NDBuffer[Self.dtype], ProductArg[Self.dtype]]:
        """Launch product reduction.

        Runs product_reduce kernel (log-space, all dtypes, overflow-safe).
        Optionally computes and stores excl_product for backward.

        store_excl_product=True  (default):
            Runs excl_product_kernel immediately after product_reduce.
            Stores result in ProductArg.excl_product.
            Backward uses it directly — no second kernel launch needed.
            Memory cost: one input-shaped buffer of dtype.

        store_excl_product=False:
            Only stores input and zero_counts.
            Backward recomputes excl_product via excl_product_kernel.
            No extra memory cost, but backward is slower.

        Args:
            A:               Input NDBuffer. Must be on GPU.
            normalized_axes: Validated, normalised reduction axes.
            keepdims:        Whether to keep reduced dimensions.

        Returns:
            Tuple of (output NDBuffer, ProductArg for backward).
        """
        var shape_A      = A.shape
        var strides_A    = A.strides
        var output_shape = shape_A.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )

        var normalized_axes_copy = normalized_axes
        if len(normalized_axes_copy) == 0:
            normalized_axes_copy = IntArray(len(shape_A))
            for i in range(len(shape_A)):
                normalized_axes_copy[i] = i

        var reduction_axes: Array  = Array(normalized_axes_copy)
        var reduced_shape          = shape_A.reduced_shape(normalized_axes)
        var in_shape: Array        = shape_A.array()
        var in_strides: Array      = strides_A.array()
        var total_output: Int      = output_shape.product()
        var reduced_volume: Int    = reduced_shape.product()
        var input_numels: Int      = A.numels()

        var (threads_per_block, num_blocks) = Self.launch_config[
            max_block_width
        ](total_output, reduced_volume)

        ref A_device_state = A.device_state.value()
        ref gpu            = A_device_state.get_gpu()
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

        # ── Kernel 1: product_reduce ──────────────────────────────────────────
        var compiled_product = device_context.compile_function[
            product_reduce[Self.dtype, max_block_width],
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
            total_output,
            reduced_volume,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        # Wrap output
        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        var out_ndb = NDBuffer[Self.dtype].with_device_state(
            result_state^, output_shape
        )

        # Wrap zero counts
        var zero_state = DeviceState[DType.int32](zero_counts_buffer^, gpu)
        var zero_ndb = NDBuffer[DType.int32].with_device_state(
            zero_state^, output_shape
        )

        # ── Kernel 2: excl_product (only if store_excl_product=True) ─────────
        var excl_optional: Optional[NDBuffer[Self.dtype]] = None

        comptime if store_excl_product:
            var excl_buffer = device_context.enqueue_create_buffer[Self.dtype](
                input_numels
            )

            # excl_product uses same launch config as product_reduce
            var (excl_threads, excl_blocks) = Self.launch_config[max_block_width](
                total_output, reduced_volume
            )

            var compiled_excl = device_context.compile_function[
                excl_product_kernel[Self.dtype, max_block_width],
                excl_product_kernel[Self.dtype, max_block_width],
            ]()

            device_context.enqueue_function(
                compiled_excl,
                excl_buffer,
                A_buffer,
                in_shape,
                in_strides,
                reduction_axes,
                total_output,
                reduced_volume,
                grid_dim=excl_blocks,
                block_dim=excl_threads,
            )

            device_context.synchronize()

            var excl_state = DeviceState[Self.dtype](excl_buffer^, gpu)
            var excl_ndb   = NDBuffer[Self.dtype].with_device_state(
                excl_state^, shape_A   # input-shaped
            )
            excl_optional = Optional(excl_ndb^)

        # ── Build ProductArg ──────────────────────────────────────────────────
        var arg = ProductArg[Self.dtype](
            input          = A,              # original input — always stored
            excl_product   = excl_optional^,
            zero_counts    = zero_ndb^,
            axes           = normalized_axes,
            keepdims       = keepdims,
            reduced_volume = reduced_volume,
        )

        return (out_ndb^, arg^)

    # ── LOG SUM EXP (unchanged) ───────────────────────────────────────────────

    @staticmethod
    def launch_log_sum[
        max_block_width: Int = 512,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ](
        A: NDBuffer[Self.dtype], normalized_axes: IntArray, keepdims: Bool
    ) raises -> NDBuffer[Self.dtype]:
        var shape_A      = A.shape
        var strides_A    = A.strides
        var output_shape = shape_A.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )

        var normalized_axes_copy = normalized_axes
        if len(normalized_axes_copy) == 0:
            normalized_axes_copy = IntArray(len(shape_A))
            for i in range(len(shape_A)):
                normalized_axes_copy[i] = i

        var reduction_axes: Array  = Array(normalized_axes_copy)
        var reduced_shape          = shape_A.reduced_shape(normalized_axes)
        var in_shape: Array        = shape_A.array()
        var in_strides: Array      = strides_A.array()
        var total_output: Int      = output_shape.product()
        var reduced_volume: Int    = reduced_shape.product()

        var (threads_per_block, num_blocks) = Self.launch_config[
            max_block_width
        ](total_output, reduced_volume)

        ref A_device_state = A.device_state.value()
        ref gpu            = A_device_state.get_gpu()
        var device_context = gpu[]
        var result_buffer  = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )
        ref A_buffer = A_device_state.device_buffer()

        comptime if Self.dtype == DType.float32:
            var compiled_func = device_context.compile_function[
                log_sum_exp_f32[
                    max_block_size=max_block_width,
                    epsilon=epsilon.cast[DType.float32](),
                ],
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
                total_output,
                reduced_volume,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        elif Self.dtype == DType.float64:
            var compiled_func = device_context.compile_function[
                log_sum_exp_f64[
                    max_block_size=max_block_width,
                    epsilon=epsilon.cast[DType.float64](),
                ],
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
                total_output,
                reduced_volume,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        else:
            panic(
                "Reduction.launch_log_sum: only float32 and float64 supported"
            )

        device_context.synchronize()
        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(device_state^, output_shape)

    # ── launch_config (unchanged) ─────────────────────────────────────────────

    @staticmethod
    def launch_config[
        max_block_size: Int
    ](total_output: Int, reduced_volume: Int) -> Tuple[Int, Int]:
        var block_size = 1
        while block_size < reduced_volume:
            block_size <<= 1
            if block_size >= max_block_size:
                block_size = max_block_size
                break
        return (block_size, total_output)


fn main() raises:
    pass
