"""Host-side launcher for the dtype cast GPU kernel.

Follows the kernel convention:
  - tenmo/gpu/kernels/cast.mojo: device function body (raw Pointers)
  - tenmo/kernels/cast_kernel.mojo: this file (host-side launcher)

Handles DType.bool: stored as uint8 on GPU. Allocates DeviceBuffer[dst_datatype]
(uint8 for bool), wraps with DeviceState[dst_dtype].__init__[True] to bypass
create_sub_buffer.
"""

from std.sys import simd_width_of
from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from ..gpu.device import DeviceState
from ..shared.panic import panic
from .kernel_helpers import elementwise_launch_config
from .cast_device import dtype_cast


@fieldwise_init
struct CastKernel(ImplicitlyCopyable, RegisterPassable):
    """GPU dtype cast launcher — element-wise src→dst conversion.

    Single kernel launch on device. Avoids GPU→CPU→CPU cast→CPU→GPU
    round-trip for cross-dtype conversions.
    """

    @staticmethod
    def launch[
        src_dtype: DType,
        dst_dtype: DType,
    ](
        src_layout: Layout,
        src_device_state: DeviceState[src_dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[dst_dtype]]:
        """Launch dtype cast kernel on GPU.

        Args:
            src_layout: Layout of the source tensor.
            src_device_state: GPU-resident source data.
            sync: If True, synchronize after launch.

        Returns:
            (output_layout, output_device_state) — same shape as input.
        """
        comptime src_datatype = (
            DType.uint8 if src_dtype == DType.bool else src_dtype
        )
        comptime dst_datatype = (
            DType.uint8 if dst_dtype == DType.bool else dst_dtype
        )

        var numels = src_layout.numel()
        comptime simdwidth = simd_width_of[dst_datatype]()

        var (num_blocks, threads_per_block) = elementwise_launch_config(
            numels, simdwidth
        )

        ref gpu = src_device_state.get_gpu()
        var device_context = gpu[]

        var contig_state = materialize_contiguous(
            src_device_state, src_layout, sync=sync
        )

        # Allocate output buffer using storage dtype (uint8 for bool)
        var result_buffer = device_context.enqueue_create_buffer[dst_datatype](
            numels
        )

        var compiled = device_context.compile_function[
            dtype_cast[
                src_dtype=src_dtype,
                dst_dtype=dst_dtype,
                src_datatype=src_datatype,
                dst_datatype=dst_datatype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ]
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_state.device_buffer(),
            Int64(numels),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        # bool-as-uint8: skip sub-buffer creation (buffer dtype != logical dtype when bool)
        var result_state = DeviceState[dst_dtype].__init__[True](
            result_buffer^, gpu
        )
        return (Layout(src_layout.shape), result_state^)
