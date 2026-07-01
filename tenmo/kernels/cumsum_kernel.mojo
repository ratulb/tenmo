# === Cumsum GPU kernel  —  thread-per-frame sequential scan ===
#
# For a tensor with shape folded to (outer, axis_size, inner), each
# thread scans one frame (fixed outer x inner coordinate) along the
# axis dimension.  Coalesced when inner=1 (axis is innermost);
# L1-cache-friendly otherwise for small-to-moderate inner sizes.

from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from tenmo.ndbuffer import NDBuffer
from . import elementwise_launch_config
from tenmo.device import DeviceState
from tenmo.common_utils import panic


def cumsum_kernel[
    dtype: DType,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    axis_size: Int,
    inner: Int,
    outer: Int,
):
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    var total_frames = outer * inner

    var idx = gtid
    while idx < total_frames:
        var o = idx / inner
        var i_local = idx % inner
        var base = (o * axis_size + 0) * inner + i_local

        var running = A[base]
        result[base] = running

        for k in range(1, axis_size):
            var pos = base + k * inner
            running += A[pos]
            result[pos] = running

        idx += stride


def cumsum_backward_kernel[
    dtype: DType,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    axis_size: Int,
    inner: Int,
    outer: Int,
):
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

        var running = grad[base]
        result[base] = running

        for k in range(1, axis_size):
            var pos = base - k * inner
            running += grad[pos]
            result[pos] = running

        idx += stride


struct CumsumGpuKernel[dtype: DType](ImplicitlyCopyable & Movable):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def launch(
        A: NDBuffer[Self.dtype],
        axis: Int,
        outer: Int,
        axis_size: Int,
        inner: Int,
        sync: Bool = False,
    ) raises -> NDBuffer[Self.dtype]:
        debug_assert(A.is_on_gpu())
        var total_frames = outer * inner
        var numels = A.numels()

        comptime simdwidth = simd_width_of[Self.datatype]()
        var (num_blocks, threads_per_block) = elementwise_launch_config(
            total_frames, simdwidth
        )

        ref device_state = A.device_state.value()
        var device_context = device_state.gpu[]
        var contig_state = A.contiguous_device_state()

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
            axis_size,
            inner,
            outer,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype].__init__[True](
            result_buffer^, device_state.gpu
        )
        return NDBuffer[Self.dtype].with_device_state(result_state^, A.shape)

    @staticmethod
    def launch_backward(
        grad: NDBuffer[Self.dtype],
        axis: Int,
        outer: Int,
        axis_size: Int,
        inner: Int,
        sync: Bool = False,
    ) raises -> NDBuffer[Self.dtype]:
        debug_assert(grad.is_on_gpu())
        var total_frames = outer * inner
        var numels = grad.numels()

        comptime simdwidth = simd_width_of[Self.datatype]()
        var (num_blocks, threads_per_block) = elementwise_launch_config(
            total_frames, simdwidth
        )

        ref device_state = grad.device_state.value()
        var device_context = device_state.gpu[]
        var contig_state = grad.contiguous_device_state()

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
            axis_size,
            inner,
            outer,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype].__init__[True](
            result_buffer^, device_state.gpu
        )
        return NDBuffer[Self.dtype].with_device_state(result_state^, grad.shape)
