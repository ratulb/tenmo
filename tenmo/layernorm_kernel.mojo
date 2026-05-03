# =============================================================================
# layernorm_kernel.mojo
#
# Fused LayerNorm normalize kernel — Pass 2 of the two-pass forward.
# Pass 1 (Welford) already ran and produced mean + var per row.
#
# This kernel fuses:
#   rstd    = 1/sqrt(var + eps)          # scalar per row — rsqrt directly
#   x_hat   = (x - mean) * rstd          # per element
#   out     = gamma * x_hat + beta       # per element
#
# Three output buffers written in single pass:
#   out_buffer    — final LayerNorm output (*, D)
#   x_hat_buffer  — normalized input, saved for backward (*, D)
#   rstd_buffer   — reciprocal std per row (*, 1), saved for backward
#
# Grid:  one block per row (outer_size blocks)
# Block: threads stride across D (last dim)
#
# Design mirrors unary_ops_with_mask — single pass, multiple outputs.
# No dtype constraint at kernel level.
#
# CPU path:
#   Serial loop over rows, element-wise per row.
#   NDBuffer.layernorm_normalize() handles CPU + GPU dispatch.
# =============================================================================

from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of
from std.math import rsqrt

from .ndbuffer import NDBuffer
from .buffers import Buffer
from .device import DeviceState
from .common_utils import panic
from .shapes import Shape
from .array import Array


fn layernorm_normalize[
    dtype: DType,
    max_block_size: Int = 512,
](
    out_buffer:   UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_hat_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rstd_buffer:  UnsafePointer[Scalar[dtype], MutAnyOrigin],
    x_buffer:     UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mean_buffer:  UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    var_buffer:   UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta_buffer:  UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    D: Int,
    outer_size: Int,
    eps: Scalar[dtype],
):
    """Fused LayerNorm normalize kernel.

    One block per row. Threads stride across D.
    Reads mean and var (already computed by Welford pass 1).
    Writes out, x_hat, and rstd in a single pass.

    rstd = rsqrt(var + eps) = 1/sqrt(var + eps).
    Thread 0 writes rstd_buffer[bid] once per row.

    Args:
        out_buffer:   Output (*, D) — gamma * x_hat + beta.
        x_hat_buffer: Normalized input (*, D) — saved for backward.
        rstd_buffer:  Reciprocal std per row (*, 1) — saved for backward.
        x_buffer:     Input (*, D) — contiguous.
        mean_buffer:  Per-row mean (*, 1) — from Welford.
        var_buffer:   Per-row variance (*, 1) — from Welford.
        gamma_buffer: Scale (D,).
        beta_buffer:  Shift (D,).
        D:            Last dimension size.
        outer_size:   Number of independent rows.
        eps:          Numerical stability constant.
    """
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    var bid        = Int(block_idx.x)
    var tid        = Int(thread_idx.x)
    var block_size = Int(block_dim.x)

    if bid >= outer_size:
        return

    var row_var  = var_buffer[bid]
    var row_mean = mean_buffer[bid]

    # rsqrt(var + eps) = 1/sqrt(var + eps)  — rstd directly, no inversion needed
    var safe_var = row_var + eps
    var rstd     = rsqrt(safe_var if safe_var > Scalar[dtype](0) else Scalar[dtype](eps))

    # Thread 0 writes rstd once per row — cheap, one write per block
    if tid == 0:
        rstd_buffer[bid] = rstd

    var row_base = bid * D
    var i = tid
    while i < D:
        var x_i     = x_buffer[row_base + i]
        var x_hat_i = (x_i - row_mean) * rstd
        var out_i   = gamma_buffer[i] * x_hat_i + beta_buffer[i]
        x_hat_buffer[row_base + i] = x_hat_i
        out_buffer[row_base + i]   = out_i
        i += block_size


@fieldwise_init
struct LayerNormKernel[dtype: DType](ImplicitlyCopyable, RegisterPassable):

    @staticmethod
    def launch(
        x:     NDBuffer[Self.dtype],   # (*, D) contiguous GPU
        mean:  NDBuffer[Self.dtype],   # (*, 1) from Welford
        var_:  NDBuffer[Self.dtype],   # (*, 1) from Welford
        gamma: NDBuffer[Self.dtype],   # (D,)
        beta:  NDBuffer[Self.dtype],   # (D,)
        eps:   Scalar[Self.dtype],
    ) raises -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        """Launch fused LayerNorm normalize kernel.

        Returns (out_ndb, x_hat_ndb, rstd_ndb).
        All three are free — written in the same pass.
        x_hat and rstd saved for backward at zero extra cost.

        Args:
            x:     Input NDBuffer. Must be on GPU and contiguous.
            mean:  Per-row mean from Welford. Shape (*, 1).
            var_:  Per-row variance from Welford. Shape (*, 1).
            gamma: Scale parameters. Shape (D,).
            beta:  Shift parameters. Shape (D,).
            eps:   Numerical stability constant.

        Returns:
            Tuple of (output, x_hat, rstd) NDBuffers.
            output and x_hat are shape (*, D).
            rstd is shape (*, 1) — one per row.
        """
        debug_assert(x.is_on_gpu())
        debug_assert(x.is_contiguous())

        var out_shape  = x.shape
        var D          = out_shape[-1]
        var outer_size = x.numels() // D
        var numels     = x.numels()

        var (threads_per_block, num_blocks) = Self.launch_config(D, outer_size)

        ref x_device_state = x.device_state.value()
        ref gpu            = x_device_state.get_gpu()
        var device_context = gpu[]

        var contig_x = x.contiguous_device_state()

        var out_buffer   = device_context.enqueue_create_buffer[Self.dtype](numels)
        var x_hat_buffer = device_context.enqueue_create_buffer[Self.dtype](numels)
        var rstd_buffer  = device_context.enqueue_create_buffer[Self.dtype](outer_size)

        ref mean_state  = mean.device_state.value()
        ref var_state   = var_.device_state.value()
        ref gamma_state = gamma.device_state.value()
        ref beta_state  = beta.device_state.value()

        comptime max_block = 512
        var compiled = device_context.compile_function[
            layernorm_normalize[Self.dtype, max_block],
            layernorm_normalize[Self.dtype, max_block],
        ]()

        device_context.enqueue_function(
            compiled,
            out_buffer,
            x_hat_buffer,
            rstd_buffer,
            contig_x.device_buffer(),
            mean_state.device_buffer(),
            var_state.device_buffer(),
            gamma_state.device_buffer(),
            beta_state.device_buffer(),
            D,
            outer_size,
            eps,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        # rstd shape is (*, 1) — same as mean/var from Welford
        var rstd_shape = out_shape[0:-1] + [1]

        var out_state   = DeviceState[Self.dtype](out_buffer^,   gpu)
        var x_hat_state = DeviceState[Self.dtype](x_hat_buffer^, gpu)
        var rstd_state  = DeviceState[Self.dtype](rstd_buffer^,  gpu)

        var out_ndb   = NDBuffer[Self.dtype].with_device_state(out_state^,   out_shape)
        var x_hat_ndb = NDBuffer[Self.dtype].with_device_state(x_hat_state^, out_shape)
        var rstd_ndb  = NDBuffer[Self.dtype].with_device_state(rstd_state^,  rstd_shape)

        return (out_ndb^, x_hat_ndb^, rstd_ndb^)

    @staticmethod
    fn launch_config(D: Int, outer_size: Int) -> Tuple[Int, Int]:
        """One block per row. Block size = smallest power of 2 >= D, capped at 512."""
        var block_size = 1
        while block_size < D and block_size < 512:
            block_size <<= 1
        return (block_size, outer_size)
