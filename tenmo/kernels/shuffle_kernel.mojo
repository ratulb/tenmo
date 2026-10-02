# shuffle_kernel.mojo — GPU shuffle kernels

from max.gpu import thread_idx, block_idx, block_dim, grid_dim
from max.gpu.host import DeviceBuffer
from ..gpu.device import DeviceState, GPU
from ..shared.layout import Layout
from ..shared.array import RankArray


def shuffle_gather[
    dtype: DType
](
    out_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    perm_buffer: Pointer[Int64, ImmutAnyOrigin],
    in_shape: RankArray,
    in_strides: RankArray,
    axis_: Int64,
    total_elements_: Int64,
):
    var axis = Int(axis_)
    var total_elements = Int(total_elements_)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= total_elements:
        return

    var remaining = tid
    var src_flat = 0
    var rank = len(in_shape)

    for k in reversed(range(rank)):
        var coord = remaining % in_shape[k]
        remaining //= in_shape[k]
        var src_coord = Int(perm_buffer[unsafe_offset=coord]) if k == axis else coord
        src_flat += src_coord * in_strides[k]

    out_buffer[unsafe_offset=tid] = in_buffer[unsafe_offset=src_flat]


def shuffle_scatter[
    dtype: DType
](
    out_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    perm_buffer: Pointer[Int64, ImmutAnyOrigin],
    in_shape: RankArray,
    in_strides: RankArray,
    axis_: Int64,
    total_elements_: Int64,
):
    var axis = Int(axis_)
    var total_elements = Int(total_elements_)
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= total_elements:
        return

    var remaining = tid
    var dst_flat = 0
    var rank = len(in_shape)

    for k in reversed(range(rank)):
        var coord = remaining % in_shape[k]
        remaining //= in_shape[k]
        var dst_coord = Int(perm_buffer[unsafe_offset=coord]) if k == axis else coord
        dst_flat += dst_coord * in_strides[k]

    out_buffer[unsafe_offset=dst_flat] = in_buffer[unsafe_offset=tid]


@fieldwise_init
struct ShuffleKernel[dtype: DType](ImplicitlyCopyable, RegisterPassable):
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
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        permutation: List[Int],
        axis: Int,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var shape = A_layout.shape
        var total_elements = shape.num_elements()

        ref device_state = A_device_state
        ref gpu = device_state.get_gpu()
        var device_context = gpu[]

        var in_shape = shape.array()
        var in_strides = A_layout.strides.array()

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
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,
            device_state.device_buffer(),
            perm_device,
            in_shape,
            in_strides,
            Int64(axis),
            Int64(total_elements),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return (
            Layout(shape),
            result_state^,
        )

    @staticmethod
    def launch_scatter(
        grad_layout: Layout,
        grad_device_state: DeviceState[Self.dtype],
        permutation: List[Int],
        axis: Int,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var shape = grad_layout.shape
        var total_elements = shape.num_elements()

        ref device_state = grad_device_state
        ref gpu = device_state.get_gpu()
        var device_context = gpu[]

        var in_shape = shape.array()
        var in_strides = grad_layout.strides.array()

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
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,
            device_state.device_buffer(),
            perm_device,
            in_shape,
            in_strides,
            Int64(axis),
            Int64(total_elements),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return (
            Layout(shape),
            result_state^,
        )
