from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from tenmo.ndbuffer import NDBuffer
from . import elementwise_launch_config
from tenmo.device import DeviceState
from tenmo.common_utils import panic


def triu_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
    M: Int,
    N: Int,
    diagonal: Int,
):
    """Fused triu: result[i] = A[i] if col >= row + diagonal else 0."""
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE
    var batch_stride = M * N
    var zero = SIMD[dtype, simd_width](0)

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var vec_result = SIMD[dtype, simd_width](0)

                for j in range(simd_width):
                    var idx = i + j
                    var within = idx % batch_stride
                    var row = within // N
                    var col = within % N
                    if col >= row + diagonal:
                        vec_result[j] = vec_a[j]

                result.store[width=simd_width](i, vec_result)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var within = idx % batch_stride
                    var row = within // N
                    var col = within % N
                    result[idx] = A[idx] if col >= row + diagonal else Scalar[
                        dtype
                    ](0)

        base_idx += stride * CHUNK_SIZE


def triu_backward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
    M: Int,
    N: Int,
    diagonal: Int,
):
    """Backward: grad_input = grad_output * triu_mask. Same mask as forward."""
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE
    var batch_stride = M * N
    var zero = SIMD[dtype, simd_width](0)

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_g = grad.load[width=simd_width](i)
                var vec_result = SIMD[dtype, simd_width](0)

                for j in range(simd_width):
                    var idx = i + j
                    var within = idx % batch_stride
                    var row = within // N
                    var col = within % N
                    if col >= row + diagonal:
                        vec_result[j] = vec_g[j]

                result.store[width=simd_width](i, vec_result)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var within = idx % batch_stride
                    var row = within // N
                    var col = within % N
                    result[idx] = grad[
                        idx
                    ] if col >= row + diagonal else Scalar[dtype](0)

        base_idx += stride * CHUNK_SIZE


struct TriuGpuKernel[dtype: DType](ImplicitlyCopyable & Movable):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def launch(
        A: NDBuffer[Self.dtype],
        diagonal: Int,
        sync: Bool = False,
    ) raises -> NDBuffer[Self.dtype]:
        debug_assert(A.is_on_gpu())
        var numels = A.numels()
        var shape = A.shape
        var rank = shape.rank()
        var M = shape[rank - 2]
        var N = shape[rank - 1]
        comptime simdwidth = simd_width_of[Self.datatype]()

        var (num_blocks, threads_per_block) = elementwise_launch_config(
            numels, simdwidth
        )
        ref device_state = A.device_state.value()
        var device_context = device_state.gpu[]
        var contig_state = A.contiguous_device_state()

        var result_buffer = device_context.enqueue_create_buffer[Self.datatype](
            numels
        )

        var compiled = device_context.compile_function[
            triu_kernel[Self.datatype, simdwidth, 2 * simdwidth],
        ]()
        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_state.device_buffer(),
            numels,
            M,
            N,
            diagonal,
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
        diagonal: Int,
        sync: Bool = False,
    ) raises -> NDBuffer[Self.dtype]:
        debug_assert(grad.is_on_gpu())
        var numels = grad.numels()
        var shape = grad.shape
        var rank = shape.rank()
        var M = shape[rank - 2]
        var N = shape[rank - 1]
        comptime simdwidth = simd_width_of[Self.datatype]()

        var (num_blocks, threads_per_block) = elementwise_launch_config(
            numels, simdwidth
        )
        ref device_state = grad.device_state.value()
        var device_context = device_state.gpu[]
        var contig_state = grad.contiguous_device_state()

        var result_buffer = device_context.enqueue_create_buffer[Self.datatype](
            numels
        )

        var compiled = device_context.compile_function[
            triu_backward_kernel[Self.datatype, simdwidth, 2 * simdwidth],
        ]()
        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_state.device_buffer(),
            numels,
            M,
            N,
            diagonal,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype].__init__[True](
            result_buffer^, device_state.gpu
        )
        return NDBuffer[Self.dtype].with_device_state(result_state^, grad.shape)
