# =============================================================================
# shuffle_kernel.mojo — GPU shuffle kernels
# =============================================================================

from std.gpu import thread_idx, block_idx, block_dim, grid_dim
from std.gpu.host import DeviceBuffer
from std.memory import AddressSpace
from tenmo.device import DeviceState, GPU
from tenmo.ndbuffer import NDBuffer
from tenmo.array import Array


def shuffle_gather[
    dtype: DType
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    perm_buffer: UnsafePointer[Int64, ImmutAnyOrigin],
    in_shape: Array,
    in_strides: Array,
    axis: Int,
    total_elements: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= total_elements:
        return

    var remaining = tid
    var src_flat = 0
    var rank = len(in_shape)

    for k in reversed(range(rank)):
        var coord = remaining % in_shape[k]
        remaining //= in_shape[k]
        var src_coord = Int(perm_buffer[coord]) if k == axis else coord
        src_flat += src_coord * in_strides[k]

    out_buffer[tid] = in_buffer[src_flat]


def shuffle_scatter[
    dtype: DType
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    perm_buffer: UnsafePointer[Int64, ImmutAnyOrigin],
    in_shape: Array,
    in_strides: Array,
    axis: Int,
    total_elements: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= total_elements:
        return

    var remaining = tid
    var dst_flat = 0
    var rank = len(in_shape)

    for k in reversed(range(rank)):
        var coord = remaining % in_shape[k]
        remaining //= in_shape[k]
        var dst_coord = Int(perm_buffer[coord]) if k == axis else coord
        dst_flat += dst_coord * in_strides[k]

    out_buffer[dst_flat] = in_buffer[tid]


@fieldwise_init
struct ShuffleGPU[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def _upload_permutation(
        permutation: List[Int],
        gpu: GPU,
    ) raises -> DeviceBuffer[DType.int64]:
        var device_context = gpu[]
        var n = len(permutation)
        var perm_buffer = device_context.enqueue_create_buffer[DType.int64](n)
        with perm_buffer.map_to_host() as host:
            for i in range(n):
                host[i] = Int64(permutation[i])
        return perm_buffer^

    @staticmethod
    def launch_gather(
        A: NDBuffer[Self.dtype],
        permutation: List[Int],
        axis: Int,
    ) raises -> NDBuffer[Self.dtype]:
        var shape = A.shape
        var total_elements = shape.num_elements()

        ref device_state = A.device_state.value()
        ref gpu = device_state.get_gpu()
        var device_context = gpu[]

        var in_shape = shape.array()
        var in_strides = A.strides.array()

        var perm_device = Self._upload_permutation(permutation, gpu)

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_elements
        )

        var threads_per_block = 256
        var num_blocks = (
            total_elements + threads_per_block - 1
        ) // threads_per_block

        var compiled = device_context.compile_function[
            shuffle_gather[Self.dtype],
            shuffle_gather[Self.dtype],
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,
            device_state.device_buffer(),
            perm_device,
            in_shape,
            in_strides,
            axis,
            total_elements,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(result_state^, shape)

    @staticmethod
    def launch_scatter(
        grad: NDBuffer[Self.dtype],
        permutation: List[Int],
        axis: Int,
    ) raises -> NDBuffer[Self.dtype]:
        var shape = grad.shape
        var total_elements = shape.num_elements()

        ref device_state = grad.device_state.value()
        ref gpu = device_state.get_gpu()
        var device_context = gpu[]

        var in_shape = shape.array()
        var in_strides = grad.strides.array()

        var perm_device = Self._upload_permutation(permutation, gpu)

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_elements
        )
        result_buffer.enqueue_fill(Scalar[Self.dtype](0))

        var threads_per_block = 256
        var num_blocks = (
            total_elements + threads_per_block - 1
        ) // threads_per_block

        var compiled = device_context.compile_function[
            shuffle_scatter[Self.dtype],
            shuffle_scatter[Self.dtype],
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,
            device_state.device_buffer(),
            perm_device,
            in_shape,
            in_strides,
            axis,
            total_elements,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(result_state^, shape)
