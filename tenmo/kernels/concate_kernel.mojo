# =============================================================================
# concate_kernel.mojo — GPU concatenation copy kernel
#
# Strategy: one kernel launch per input tensor. A comptime bool `forward`
# controls the copy direction:
#   forward=True (scatter):   dst[mapped(flat)] = src[flat]
#   forward=False (gather):   dst[flat] = src[mapped(flat)]
#
# The mapping function maps a flat index in the smaller (parent) tensor to
# its corresponding flat index in the larger (concatenated) tensor, using
# coordinate decomposition and the concat-axis offset:
#
#   mapped(flat) = before * output_axis_size * stride_axis
#                + (coord_axis + offset) * stride_axis
#                + after_axis
#
# where:
#   before_axis  = flat // (input_axis_size * stride_axis)
#   coord_axis   = (flat // stride_axis) % input_axis_size
#   after_axis   = flat % stride_axis
#
# This is correct for any concat axis because row-major strides depend only
# on later dimensions (which are identical between parent and output), and
# the integer arithmetic correctly handles the different axis sizes.
# =============================================================================

from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import has_accelerator
from tenmo.ndbuffer import NDBuffer
from tenmo.device import GPU, DeviceState
from tenmo.shapes import Shape
from tenmo.common_utils import panic
from .kernel_helpers import elementwise_launch_config


def concate_copy_kernel[
    dtype: DType,
    forward: Bool,
](
    src: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    num_elements: Int,
    input_axis_size: Int,
    output_axis_size: Int,
    stride_axis: Int,
    offset: Int,
):
    """
    GPU kernel for concatenation copy.

    Iterates over the smaller (parent) tensor with a grid-stride loop.
    For each flat index, decomposes to coordinates, applies the concat-axis
    offset, recomputes the flat index in the larger tensor, and copies.

    Args:
        src: Source buffer pointer.
        dst: Destination buffer pointer.
        num_elements: Number of elements in the parent-sized tensor.
        input_axis_size: Parent's concat axis size.
        output_axis_size: Total concat axis size (output).
        stride_axis: Stride of the concat axis (= product of later dims).
        offset: Cumulative concat axis offset for this parent.
    """
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var gstride = Int(block_dim.x * grid_dim.x)
    var before_divisor = input_axis_size * stride_axis

    var flat = gtid
    while flat < num_elements:
        var before = flat // before_divisor
        var coord = (flat // stride_axis) % input_axis_size
        var after = flat % stride_axis

        var mapped = before * output_axis_size * stride_axis
                   + (coord + offset) * stride_axis
                   + after

        comptime if forward:
            dst[mapped] = src[flat]
        else:
            dst[flat] = src[mapped]

        flat += gstride


@fieldwise_init
struct ConcateGpuKernel[dtype: DType](
    ImplicitlyCopyable & Movable
):
    """
    GPU concatenation kernel launcher.

    Provides `launch_forward` and `launch_backward` static methods for
    GPU-resident concatenation and its backward pass.

    Each launch creates a contiguous copy of the source on GPU, enqueues
    the kernel, and synchronises.  The contiguous copy's lifetime is
    managed within the launch method, so the caller does not need to keep
    any buffers alive beyond the launch call.
    """

    @staticmethod
    def _launch[
        forward: Bool,
    ](
        src_ndb: NDBuffer[Self.dtype],
        dst_ndb: NDBuffer[Self.dtype],
        input_axis_size: Int,
        output_axis_size: Int,
        stride_axis: Int,
        offset: Int,
    ) raises -> None:
        """
        Internal launch helper.

        Makes src contiguous (if needed), enqueues the copy kernel, and
        synchronises.  dst must already be contiguous (freshly allocated).
        """
        debug_assert(src_ndb.is_on_gpu(), "ConcateGpuKernel requires GPU src")
        debug_assert(dst_ndb.is_on_gpu(), "ConcateGpuKernel requires GPU dst")

        var num_elements = src_ndb.numels()

        ref dst_state = dst_ndb.device_state.value()
        ref gpu = dst_state.get_gpu()
        var device_context = gpu[]

        comptime simdwidth = 1
        var (num_blocks, threads_per_block) = elementwise_launch_config(
            num_elements, simdwidth
        )

        # Materialize contiguous src — stored locally so the DeviceBuffer
        # reference stays valid through the sync.
        var contig_src_state = src_ndb.contiguous_device_state()

        var compiled = device_context.compile_function[
            concate_copy_kernel[Self.dtype, forward],
            concate_copy_kernel[Self.dtype, forward],
        ]()

        device_context.enqueue_function(
            compiled,
            contig_src_state.device_buffer(),
            dst_state.device_buffer(),
            num_elements,
            input_axis_size,
            output_axis_size,
            stride_axis,
            offset,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        # Sync so the local contig_src_state can be freed on return.
        device_context.synchronize()

    @staticmethod
    def launch_forward(
        src: NDBuffer[Self.dtype],
        dst: NDBuffer[Self.dtype],
        input_axis_size: Int,
        output_axis_size: Int,
        stride_axis: Int,
        offset: Int,
    ) raises -> None:
        """Forward concate: scatter src elements into dst at offset.

        dst[mapped(flat)] = src[flat]  for all flat in [0, src.numels())
        """
        ConcateGpuKernel[Self.dtype]._launch[True](
            src, dst,
            input_axis_size, output_axis_size, stride_axis, offset,
        )

    @staticmethod
    def launch_backward(
        grad_output: NDBuffer[Self.dtype],
        grad_input: NDBuffer[Self.dtype],
        parent_axis_size: Int,
        output_axis_size: Int,
        stride_axis: Int,
        offset: Int,
    ) raises -> None:
        """Backward concate: gather grad_output slices into grad_input.

        grad_input[flat] = grad_output[mapped(flat)] for all flat.
        """
        ConcateGpuKernel[Self.dtype]._launch[False](
            grad_output, grad_input,
            parent_axis_size, output_axis_size, stride_axis, offset,
        )
