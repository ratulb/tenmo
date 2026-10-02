# Cumsum GPU kernel  —  thread-per-frame sequential scan
#
# For a tensor with shape folded to (outer, axis_size, inner), each
# thread scans one frame (fixed outer x inner coordinate) along the
# axis dimension.  Coalesced when inner=1 (axis is innermost);
# L1-cache-friendly otherwise for small-to-moderate inner sizes.

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from .kernel_helpers import elementwise_launch_config
from ..gpu.device import DeviceState


def cumsum_kernel[
    dtype: DType,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    axis_size_: Int64,
    inner_: Int64,
    outer_: Int64,
):
    var axis_size = Int(axis_size_)
    var inner = Int(inner_)
    var outer = Int(outer_)
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    var total_frames = outer * inner

    var idx = gtid
    while idx < total_frames:
        var o = idx / inner
        var i_local = idx % inner
        var base = (o * axis_size + 0) * inner + i_local

        var running = A[unsafe_offset=base]
        result[unsafe_offset=base] = running

        for k in range(1, axis_size):
            var pos = base + k * inner
            running += A[unsafe_offset=pos]
            result[unsafe_offset=pos] = running

        idx += stride


def cumsum_backward_kernel[
    dtype: DType,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    grad: Pointer[Scalar[dtype], ImmutAnyOrigin],
    axis_size_: Int64,
    inner_: Int64,
    outer_: Int64,
):
    var axis_size = Int(axis_size_)
    var inner = Int(inner_)
    var outer = Int(outer_)
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    var total_frames = outer * inner
    var last_k = axis_size - 1

    var idx = gtid
    while idx < total_frames:
        var o = idx / inner
        var i_local = idx % inner
        var base = (o * axis_size + last_k) * inner + i_local

        var running = grad[unsafe_offset=base]
        result[unsafe_offset=base] = running

        for k in range(1, axis_size):
            var pos = base - k * inner
            running += grad[unsafe_offset=pos]
            result[unsafe_offset=pos] = running

        idx += stride


struct CumsumKernel[dtype: DType](ImplicitlyCopyable):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def launch(
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        axis: Int,
        outer: Int,
        axis_size: Int,
        inner: Int,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var total_frames = outer * inner
        var numels = A_layout.numel()

        comptime simdwidth = simd_width_of[Self.datatype]()
        var (num_blocks, threads_per_block) = elementwise_launch_config(
            total_frames, simdwidth
        )

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]
        var contig_state = materialize_contiguous(
            A_device_state, A_layout
        )

        var result_buffer = device_context.enqueue_create_buffer[Self.datatype](
            numels
        )

        var compiled = device_context.compile_function[
            cumsum_kernel[Self.datatype],
        ]()
        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_state.device_buffer(),
            Int64(axis_size),
            Int64(inner),
            Int64(outer),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        # bool-as-uint8: skip sub-buffer creation (buffer dtype != Self.dtype when bool)
        var result_state = DeviceState[Self.dtype].__init__[True](
            result_buffer^, gpu
        )

        return (
            Layout(A_layout.shape),
            result_state^,
        )

    @staticmethod
    def launch_backward(
        grad_layout: Layout,
        grad_device_state: DeviceState[Self.dtype],
        axis: Int,
        outer: Int,
        axis_size: Int,
        inner: Int,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var total_frames = outer * inner
        var numels = grad_layout.numel()

        comptime simdwidth = simd_width_of[Self.datatype]()
        var (num_blocks, threads_per_block) = elementwise_launch_config(
            total_frames, simdwidth
        )

        ref gpu = grad_device_state.get_gpu()
        var device_context = gpu[]
        var contig_state = materialize_contiguous(
            grad_device_state, grad_layout
        )

        var result_buffer = device_context.enqueue_create_buffer[Self.datatype](
            numels
        )

        var compiled = device_context.compile_function[
            cumsum_backward_kernel[Self.datatype],
        ]()
        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_state.device_buffer(),
            Int64(axis_size),
            Int64(inner),
            Int64(outer),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        # bool-as-uint8: skip sub-buffer creation (buffer dtype != Self.dtype when bool)
        var result_state = DeviceState[Self.dtype].__init__[True](
            result_buffer^, gpu
        )

        return (
            Layout(grad_layout.shape),
            result_state^,
        )
