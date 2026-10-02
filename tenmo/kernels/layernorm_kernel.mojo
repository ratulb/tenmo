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

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.math import rsqrt

from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from ..gpu.device import DeviceState
from ..shared.shapes import Shape


def layernorm_normalize[
    dtype: DType,
    max_block_size: Int = 512,
](
    out_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    x_hat_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    rstd_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    x_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    mean_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    var_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    gamma_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    beta_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    D_: Int64,
    outer_size_: Int64,
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
        D_:            Last dimension size.
        outer_size_:   Number of independent rows.
        eps:          Numerical stability constant.
    """
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "max_block_size must be a power of 2 less than 1024"

    var D = Int(D_)
    var outer_size = Int(outer_size_)
    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var block_size = Int(block_dim.x)

    if bid >= outer_size:
        return

    var row_var = var_buffer[unsafe_offset=bid]
    var row_mean = mean_buffer[unsafe_offset=bid]

    # rsqrt(var + eps) = 1/sqrt(var + eps)  — rstd directly, no inversion needed
    var safe_var = row_var + eps
    var rstd = rsqrt(
        safe_var if safe_var > Scalar[dtype](0) else Scalar[dtype](eps)
    )

    # Thread 0 writes rstd once per row — cheap, one write per block
    if tid == 0:
        rstd_buffer[unsafe_offset=bid] = rstd

    var row_base = bid * D
    var i = tid
    while i < D:
        var x_i = x_buffer[unsafe_offset=row_base + i]
        var x_hat_i = (x_i - row_mean) * rstd
        var out_i = gamma_buffer[unsafe_offset=i] * x_hat_i + beta_buffer[unsafe_offset=i]
        x_hat_buffer[unsafe_offset=row_base + i] = x_hat_i
        out_buffer[unsafe_offset=row_base + i] = out_i
        i += block_size


@fieldwise_init
struct LayerNormKernel[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def launch(
        x_layout: Layout,  # (*, D) contiguous GPU
        x_device_state: DeviceState[Self.dtype],
        mean_layout: Layout,  # (*, 1) from Welford
        mean_device_state: DeviceState[Self.dtype],
        var__layout: Layout,  # (*, 1) from Welford
        var__device_state: DeviceState[Self.dtype],
        gamma_layout: Layout,  # (D,)
        gamma_device_state: DeviceState[Self.dtype],
        beta_layout: Layout,  # (D,)
        beta_device_state: DeviceState[Self.dtype],
        eps: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[
        Tuple[Layout, DeviceState[Self.dtype]],
        Tuple[Layout, DeviceState[Self.dtype]],
        Tuple[Layout, DeviceState[Self.dtype]],
    ]:
        """Launch fused LayerNorm normalize kernel.

        Returns ((out_layout, out_storage), (x_hat_layout, x_hat_storage),
                 (rstd_layout, rstd_storage)).
        All three are unsafe_free — written in the same pass.
        x_hat and rstd saved for backward at zero extra cost.

        Args:
            x_layout:          Input layout. Must be on GPU and contiguous.
            x_device_state:    Input device state on GPU.
            mean_layout:       Per-row mean layout from Welford. Shape (*, 1).
            mean_device_state: Per-row mean device state on GPU.
            var__layout:       Per-row variance layout from Welford. Shape (*, 1).
            var__device_state: Per-row variance device state on GPU.
            gamma_layout:      Scale parameters layout. Shape (D,).
            gamma_device_state: Scale parameters device state on GPU.
            beta_layout:       Shift parameters layout. Shape (D,).
            beta_device_state: Shift parameters device state on GPU.
            eps:               Numerical stability constant.
            sync:              Whether to sync GPU after operation.

        Returns:
            ((out_layout, out_storage), (x_hat_layout, x_hat_storage),
             (rstd_layout, rstd_storage)).
            output and x_hat are shape (*, D).
            rstd is shape (*, 1) — one per row.
        """
        debug_assert(x_layout.is_contiguous())

        var out_shape = x_layout.shape
        var D = out_shape[-1]
        var outer_size = x_layout.numel() // D
        var numels = x_layout.numel()

        var (threads_per_block, num_blocks) = Self.launch_config(D, outer_size)

        ref gpu = x_device_state.get_gpu()
        var device_context = gpu[]

        var contig_x = materialize_contiguous(
            x_device_state, x_layout
        )

        var out_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )
        var x_hat_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )
        var rstd_buffer = device_context.enqueue_create_buffer[Self.dtype](
            outer_size
        )

        ref mean_state = mean_device_state
        ref var_state = var__device_state
        ref gamma_state = gamma_device_state
        ref beta_state = beta_device_state

        comptime max_block = 512
        var compiled = device_context.compile_function[
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
            Int64(D),
            Int64(outer_size),
            eps,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        # rstd shape is (*, 1) — same as mean/var from Welford
        var rstd_shape = out_shape[0:-1] + [1]

        var out_state = DeviceState[Self.dtype](out_buffer^, gpu)
        var x_hat_state = DeviceState[Self.dtype](x_hat_buffer^, gpu)
        var rstd_state = DeviceState[Self.dtype](rstd_buffer^, gpu)

        var out_pair = (
            Layout(out_shape),
            out_state^,
        )
        var x_hat_pair = (
            Layout(out_shape),
            x_hat_state^,
        )
        var rstd_pair = (
            Layout(rstd_shape),
            rstd_state^,
        )

        return (out_pair, x_hat_pair, rstd_pair)

    @staticmethod
    def launch_config(D: Int, outer_size: Int) -> Tuple[Int, Int]:
        """One block per row. Block size = smallest power of 2 >= D, capped at 512.
        """
        var block_size = 1
        while block_size < D and block_size < 512:
            block_size <<= 1
        return (block_size, outer_size)
