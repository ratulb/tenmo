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

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from ..gpu.device import GPU, DeviceState
from .kernel_helpers import elementwise_launch_config


def concate_copy_kernel[
    dtype: DType,
    forward: Bool,
](
    src: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    num_elements_: Int64,
    input_axis_size_: Int64,
    output_axis_size_: Int64,
    stride_axis_: Int64,
    offset_: Int64,
):
    """
        GPU kernel for concatenation copy.

    Iterates over the smaller (parent) tensor with a grid-stride loop.
    For each flat index, decomposes to coordinates, applies the concat-axis
    offset, recomputes the flat index in the larger tensor, and copies.

    Args:
        src: Source buffer pointer.
        dst: Destination buffer pointer.
        num_elements_: Number of elements in the parent-sized tensor.
        input_axis_size_: Parent's concat axis size.
        output_axis_size_: Total concat axis size (output).
        stride_axis_: Stride of the concat axis (= product of later dims).
        offset_: Cumulative concat axis offset for this parent.
    """
    var num_elements = Int(num_elements_)
    var input_axis_size = Int(input_axis_size_)
    var output_axis_size = Int(output_axis_size_)
    var stride_axis = Int(stride_axis_)
    var offset = Int(offset_)
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var gstride = block_dim.x * grid_dim.x
    var before_divisor = input_axis_size * stride_axis

    var flat = gtid
    while flat < num_elements:
        var before = flat // before_divisor
        var coord = (flat // stride_axis) % input_axis_size
        var after = flat % stride_axis

        var mapped = (
            before * output_axis_size * stride_axis
            + (coord + offset) * stride_axis
            + after
        )

        comptime if forward:
            dst[unsafe_offset=mapped] = src[unsafe_offset=flat]
        else:
            dst[unsafe_offset=flat] = src[unsafe_offset=mapped]

        flat += gstride


@fieldwise_init
struct ConcatKernel[dtype: DType](ImplicitlyCopyable):
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
        src_layout: Layout,
        src_device_state: DeviceState[Self.dtype],
        dst_layout: Layout,
        dst_device_state: DeviceState[Self.dtype],
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

        var num_elements = src_layout.numel()

        ref dst_state = dst_device_state
        ref gpu = dst_state.get_gpu()
        var device_context = gpu[]

        comptime simdwidth = 1
        var (num_blocks, threads_per_block) = elementwise_launch_config(
            num_elements, simdwidth
        )

        # Materialize contiguous src — stored locally so the DeviceBuffer
        # reference stays valid through the sync.
        var contig_src_state = materialize_contiguous(
            src_device_state, src_layout
        )

        var compiled = device_context.compile_function[
            concate_copy_kernel[Self.dtype, forward],
        ]()

        device_context.enqueue_function(
            compiled,
            contig_src_state.device_buffer(),
            dst_state.device_buffer(),
            Int64(num_elements),
            Int64(input_axis_size),
            Int64(output_axis_size),
            Int64(stride_axis),
            Int64(offset),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        # Sync so the local contig_src_state can be unsafe_freed on return.
        device_context.synchronize()

    @staticmethod
    def launch_forward(
        src_layout: Layout,
        src_device_state: DeviceState[Self.dtype],
        dst_layout: Layout,
        dst_device_state: DeviceState[Self.dtype],
        input_axis_size: Int,
        output_axis_size: Int,
        stride_axis: Int,
        offset: Int,
    ) raises -> None:
        """Forward concate: scatter src elements into dst at offset.

        dst[mapped(flat)] = src[flat]  for all flat in [0, src.numels())
        """
        ConcatKernel[Self.dtype]._launch[True](
            src_layout,
            src_device_state,
            dst_layout,
            dst_device_state,
            input_axis_size,
            output_axis_size,
            stride_axis,
            offset,
        )

    @staticmethod
    def launch_backward(
        grad_output_layout: Layout,
        grad_output_device_state: DeviceState[Self.dtype],
        grad_input_layout: Layout,
        grad_input_device_state: DeviceState[Self.dtype],
        parent_axis_size: Int,
        output_axis_size: Int,
        stride_axis: Int,
        offset: Int,
    ) raises -> None:
        """Backward concate: gather grad_output slices into grad_input.

        grad_input[flat] = grad_output[mapped(flat)] for all flat.
        """
        ConcatKernel[Self.dtype]._launch[False](
            grad_output_layout,
            grad_output_device_state,
            grad_input_layout,
            grad_input_device_state,
            parent_axis_size,
            output_axis_size,
            stride_axis,
            offset,
        )
