# =============================================================================
# norm_backward_kernel.mojo
#
# Fused backward kernels for Variance and Std backward passes.
#
# DESIGN OVERVIEW
# ───────────────
# Both variance and std backward share the same structure:
#
#   Variance backward:
#     local_grad[i] = (x[i] - mean_row) * scale   # per element
#     result[i]     = local_grad[i] * upstream[i]  # per element
#
#   Std backward:
#     local_grad[i] = (x[i] - mean_row) / denom_row  # per element
#     result[i]     = local_grad[i] * upstream[i]     # per element
#
#   Where:
#     mean_row  — per-row mean, shape (*, 1), one scalar per reduction slice
#     scale     — variance: scalar 2/divisor (same for all rows)
#     denom_row — std: (std_row + eps) * divisor, shape (*, 1), per-row
#
# Both reduce from 3-4 passes over (*, D) to 2 passes:
#   BEFORE: sub → mul/div → mul(upstream)  = 3 kernel launches
#   AFTER:  fused_normalize → mul(upstream) = 2 kernel launches
#
# STRIDE CORRECTNESS
# ──────────────────
# parent.buffer() in backward returns the NDBuffer with original strides.
# A view (transpose, slice) captured as a parent has non-trivial strides.
# The kernels must NOT assume contiguous layout for the input x.
#
# Fix: pass in_shape, in_strides, in_offset to both kernels.
# Output is always a fresh contiguous allocation — out indexing stays flat.
# Mean and denom are (*, 1) from Welford/forward — always contiguous.
#
# Strided x access (CPU and GPU):
#   flat_idx = in_offset + sum over dims(coord[d] * stride[d])
#   For a (*, D) tensor reduced over last dim:
#     outer coord = out_idx decomposed over non-last dims
#     inner coord = i (position within D)
#   → flat_idx = in_offset + outer_flat_stride * row + in_strides[-1] * i
#
#   We compute outer_flat_stride as the stride to step one "row" in x.
#   For axis=-1 reduction: outer stride = in_strides[-2] for 2D,
#   or more generally the product of trailing strides — use index helpers.
#
# PRACTICAL APPROACH
# ──────────────────
# Rather than reimplementing full strided index arithmetic, we delegate:
#   GPU: pass in_shape, in_strides, in_offset as Arrays — same as reduce kernel.
#        Use output_to_input_base + stride walk for x[row][i].
#   CPU: use NDBuffer.index_iterator() for the outer loop,
#        then self[coord] for element access — handles all stride cases.
#
# KERNELS
# ───────
# variance_backward_normalize[dtype, max_block_size]:
#   One block per row. Threads stride over D.
#   Computes: out[row_base + i] = (x[row][i] - mean[row]) * scale
#   x accessed via strided index arithmetic.
#   out written contiguously.
#
# std_backward_normalize[dtype, max_block_size]:
#   One block per row. Threads stride over D.
#   Computes: out[row_base + i] = (x[row][i] - mean[row]) / denom[row]
#   Same stride handling as variance kernel.
#
# LAUNCHER
# ────────
# NormBackwardKernel[dtype].launch_variance_backward(x, mean, scale)
#   → NDBuffer (*, D) contiguous — (x - mean) * scale
#
# NormBackwardKernel[dtype].launch_std_backward(x, mean, denom)
#   → NDBuffer (*, D) contiguous — (x - mean) / denom
#
# NDBuffer API (CPU + GPU unified):
#   ndb.variance_backward_normalize(mean, scale) → NDBuffer
#   ndb.std_backward_normalize(mean, denom)      → NDBuffer
#
# UPDATED BACKWARD STRUCTS
# ────────────────────────
# VarianceBackward:
#   BEFORE (3 passes over *, D):
#     diff       = input - mean          pass 1
#     local_grad = diff * (2/divisor)    pass 2
#     result     = local_grad * upstream pass 3
#   AFTER (2 passes over *, D):
#     normed  = x.variance_backward_normalize(mean, 2/divisor)  pass 1
#     result  = normed * upstream                                pass 2
#
# StdBackward:
#   BEFORE (3 passes over *, D):
#     diff       = input - mean                     pass 1
#     local_grad = diff / ((std+eps)*divisor)       pass 2
#     result     = local_grad * upstream            pass 3
#   AFTER (2 passes over *, D):
#     denom  = (std+eps) * divisor                  over (*, 1) — tiny
#     normed = x.std_backward_normalize(mean, denom) pass 1
#     result = normed * upstream                     pass 2
#
# EPSILON NOTE
# ────────────
# epsilon was removed from the forward pass signature.
# In StdBackward we use Epsilon[Self.dtype].value() — the type's machine
# epsilon — purely as a numerical guard in the denominator.
# This is a small fixed constant, not a user-tunable parameter.
# =============================================================================

from std.gpu import thread_idx, block_dim, block_idx

from .ndbuffer import NDBuffer
from .buffers import Buffer
from .device import DeviceState
from .common_utils import panic, Epsilon
from .shapes import Shape
from .array import Array
from .reduction_kernel import output_to_input_base, rank_to_reduced_offset


# =============================================================================
# SECTION 1 — variance_backward_normalize GPU kernel
#
# Computes: out[row_base + i] = (x[row][i] - mean[row]) * scale
#
# x is accessed via strided indexing — handles views and non-contiguous inputs.
# out is written contiguously — always a fresh allocation.
# mean is (*, 1) contiguous — one scalar per block loaded once.
# scale is a uniform scalar across all rows.
#
# Grid:  outer_size blocks — one block per row
# Block: threads stride over D
#
# Args:
#   out_buffer:  Output (*, D) contiguous — flat write.
#   x_buffer:    Input x — strided, accessed via in_shape/in_strides/in_offset.
#   mean_buffer: Per-row mean (*, 1) contiguous.
#   in_shape:    Shape of x as Array.
#   in_strides:  Strides of x as Array.
#   in_offset:   Base offset of x in its buffer.
#   reduction_axes: Axes being reduced (last axis) as Array.
#   scale:       2 / divisor — uniform scalar.
#   D:           Last dimension size.
#   outer_size:  Number of rows.
# =============================================================================


fn variance_backward_normalize[
    dtype: DType,
    max_block_size: Int = 512,
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mean_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: Array,
    in_strides: Array,
    in_offset: Int,
    reduction_axes: Array,
    scale: Scalar[dtype],
    D: Int,
    outer_size: Int,
):
    """Fused variance backward normalize — stride-aware x access.

    Computes (x[row][i] - mean[row]) * scale for each element.
    x may be non-contiguous (strided view). out is always contiguous.

    One block per row. Threads stride over D.
    """
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var block_size = Int(block_dim.x)

    if bid >= outer_size:
        return

    # Per-row mean — loaded once per block
    var row_mean = mean_buffer[bid]

    # Base index into x for this row — handles non-contiguous strides
    var x_row_base = (
        output_to_input_base(bid, in_shape, in_strides, reduction_axes)
        + in_offset
    )

    # Contiguous output base
    var out_row_base = bid * D

    var i = tid
    while i < D:
        # Strided read from x — rank_to_reduced_offset gives offset within row
        var x_i = (
            x_buffer
            + x_row_base
            + rank_to_reduced_offset(i, in_shape, in_strides, reduction_axes)
        )[]
        out_buffer[out_row_base + i] = (x_i - row_mean) * scale
        i += block_size


# =============================================================================
# SECTION 2 — std_backward_normalize GPU kernel
#
# Computes: out[row_base + i] = (x[row][i] - mean[row]) / denom[row]
#
# Same stride handling as variance_backward_normalize.
# denom is (*, 1) contiguous — one scalar per block.
#
# Grid:  outer_size blocks — one block per row
# Block: threads stride over D
#
# Args:
#   out_buffer:   Output (*, D) contiguous — flat write.
#   x_buffer:     Input x — strided via in_shape/in_strides/in_offset.
#   mean_buffer:  Per-row mean (*, 1) contiguous.
#   denom_buffer: Per-row denominator (*, 1) contiguous — (std+eps)*divisor.
#   in_shape:     Shape of x as Array.
#   in_strides:   Strides of x as Array.
#   in_offset:    Base offset of x in its buffer.
#   reduction_axes: Axes being reduced (last axis) as Array.
#   D:            Last dimension size.
#   outer_size:   Number of rows.
# =============================================================================


fn std_backward_normalize[
    dtype: DType,
    max_block_size: Int = 512,
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mean_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    denom_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: Array,
    in_strides: Array,
    in_offset: Int,
    reduction_axes: Array,
    D: Int,
    outer_size: Int,
):
    """Fused std backward normalize — stride-aware x access.

    Computes (x[row][i] - mean[row]) / denom[row] for each element.
    x may be non-contiguous (strided view). out is always contiguous.

    One block per row. Threads stride over D.
    """
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var block_size = Int(block_dim.x)

    if bid >= outer_size:
        return

    # Per-row scalars — loaded once per block
    var row_mean = mean_buffer[bid]
    var row_denom = denom_buffer[bid]

    # Base index into x for this row
    var x_row_base = (
        output_to_input_base(bid, in_shape, in_strides, reduction_axes)
        + in_offset
    )

    var out_row_base = bid * D

    var i = tid
    while i < D:
        var x_i = (
            x_buffer
            + x_row_base
            + rank_to_reduced_offset(i, in_shape, in_strides, reduction_axes)
        )[]
        out_buffer[out_row_base + i] = (x_i - row_mean) / row_denom
        i += block_size


# =============================================================================
# SECTION 3 — NormBackwardKernel launcher
# =============================================================================


@fieldwise_init
struct StdVarianceBackwardKernel[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def launch_variance_backward(
        x: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        scale: Scalar[Self.dtype],
    ) raises -> NDBuffer[Self.dtype]:
        """Launch fused variance backward normalize kernel.

        Computes (x - mean) * scale in single GPU pass.
        x may be non-contiguous — strides handled in kernel.
        Result must be multiplied by upstream grad by caller.

        Args:
            x:     Input NDBuffer (*, D). Must be on GPU.
            mean:  Per-row mean (*, 1) from Welford forward.
            scale: 2 / divisor uniform scalar.

        Returns:
            Contiguous NDBuffer (*, D) — (x - mean) * scale.
        """
        debug_assert(x.is_on_gpu())

        var out_shape = x.shape
        var D = out_shape[-1]
        var outer_size = x.numels() // D
        var numels = x.numels()

        var in_shape = x.shape.array()
        var in_strides = x.strides.array()
        var in_offset = x.offset

        # reduction_axes = [last axis] — consistent with welford/reduce
        var reduction_axes = Array(1)
        reduction_axes[0] = out_shape.rank() - 1

        var (threads_per_block, num_blocks) = Self.launch_config(D, outer_size)

        ref x_device_state = x.device_state.value()
        ref gpu = x_device_state.get_gpu()
        var device_context = gpu[]

        var out_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        ref mean_state = mean.device_state.value()

        comptime max_block = 512
        var compiled = device_context.compile_function[
            variance_backward_normalize[Self.dtype, max_block],
            variance_backward_normalize[Self.dtype, max_block],
        ]()

        device_context.enqueue_function(
            compiled,
            out_buffer,
            x_device_state.device_buffer(),
            mean_state.device_buffer(),
            in_shape,
            in_strides,
            in_offset,
            reduction_axes,
            scale,
            D,
            outer_size,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var out_state = DeviceState[Self.dtype](out_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(out_state^, out_shape)

    @staticmethod
    def launch_std_backward(
        x: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        denom: NDBuffer[Self.dtype],
    ) raises -> NDBuffer[Self.dtype]:
        """Launch fused std backward normalize kernel.

        Computes (x - mean) / denom in single GPU pass.
        x may be non-contiguous — strides handled in kernel.
        Result must be multiplied by upstream grad by caller.

        Args:
            x:     Input NDBuffer (*, D). Must be on GPU.
            mean:  Per-row mean (*, 1) from Welford forward.
            denom: Per-row denominator (*, 1) — (std+eps)*divisor.

        Returns:
            Contiguous NDBuffer (*, D) — (x - mean) / denom.
        """
        debug_assert(x.is_on_gpu())

        var out_shape = x.shape
        var D = out_shape[-1]
        var outer_size = x.numels() // D
        var numels = x.numels()

        var in_shape = x.shape.array()
        var in_strides = x.strides.array()
        var in_offset = x.offset

        var reduction_axes = Array(1)
        reduction_axes[0] = out_shape.rank() - 1

        var (threads_per_block, num_blocks) = Self.launch_config(D, outer_size)

        ref x_device_state = x.device_state.value()
        ref gpu = x_device_state.get_gpu()
        var device_context = gpu[]

        var out_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        ref mean_state = mean.device_state.value()
        ref denom_state = denom.device_state.value()

        comptime max_block = 512
        var compiled = device_context.compile_function[
            std_backward_normalize[Self.dtype, max_block],
            std_backward_normalize[Self.dtype, max_block],
        ]()

        device_context.enqueue_function(
            compiled,
            out_buffer,
            x_device_state.device_buffer(),
            mean_state.device_buffer(),
            denom_state.device_buffer(),
            in_shape,
            in_strides,
            in_offset,
            reduction_axes,
            D,
            outer_size,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var out_state = DeviceState[Self.dtype](out_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(out_state^, out_shape)

    @staticmethod
    fn launch_config(D: Int, outer_size: Int) -> Tuple[Int, Int]:
        """One block per row. Block size = next power-of-two up to 512."""
        var block_size = 1
        while block_size < D and block_size < 512:
            block_size <<= 1
        return (block_size, outer_size)
