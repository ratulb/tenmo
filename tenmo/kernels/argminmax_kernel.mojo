# =============================================================================
# argminmax_kernel.mojo — GPU argmin/argmax kernel
# =============================================================================

from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.memory import AddressSpace, stack_allocation
from std.utils.numerics import max_finite, min_finite
from tenmo.array import Array
from tenmo.device import DeviceState
from tenmo.ndbuffer import NDBuffer
from tenmo.shapes import Shape
from tenmo.common_utils import panic


def reduce_argminmax[
    dtype: DType,
    max_block_size: Int = 512,
    is_max: Bool = True,
](
    out_buffer: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: Array,
    in_strides: Array,
    reduction_axis: Int,
    total_output: Int,
    reduced_volume: Int,
):
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "Invalid max_block_size"

    var smem_val = stack_allocation[
        max_block_size, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var smem_idx = stack_allocation[
        max_block_size, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var out_idx = Int(block_idx.x)

    if out_idx >= total_output:
        return

    var remaining = out_idx
    var input_base = 0
    var rank = len(in_shape)

    for k in reversed(range(rank)):
        if k != reduction_axis:
            var dim = in_shape[k]
            var coord = remaining % dim
            remaining //= dim
            input_base += coord * in_strides[k]

    var axis_stride = in_strides[reduction_axis]

    var local_val: Scalar[dtype]
    var local_idx: Scalar[DType.int32] = 0

    comptime if is_max:
        local_val = min_finite[dtype]()
    else:
        local_val = max_finite[dtype]()

    var r = tid
    while r < reduced_volume:
        var val = (in_buffer + input_base + r * axis_stride)[]

        comptime if is_max:
            if val > local_val:
                local_val = val
                local_idx = Int32(r)
        else:
            if val < local_val:
                local_val = val
                local_idx = Int32(r)

        r += block_size

    smem_val[tid] = local_val
    smem_idx[tid] = local_idx
    barrier()

    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            comptime if is_max:
                if smem_val[tid + stride] > smem_val[tid]:
                    smem_val[tid] = smem_val[tid + stride]
                    smem_idx[tid] = smem_idx[tid + stride]
            else:
                if smem_val[tid + stride] < smem_val[tid]:
                    smem_val[tid] = smem_val[tid + stride]
                    smem_idx[tid] = smem_idx[tid + stride]

        barrier()
        stride >>= 1

    if tid == 0:
        (out_buffer + out_idx)[] = smem_idx[0]


@fieldwise_init
struct ArgMinMaxGpu[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def _gpu_reduce[
        is_max: Bool,
        max_block_size: Int,
    ](
        A: NDBuffer[Self.dtype],
        ax: Int,
        keepdims: Bool,
        out_shape: Shape,
        total_output: Int,
        reduced_volume: Int,
    ) raises -> NDBuffer[DType.int32]:
        ref A_device_state = A.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        var in_shape: Array = A.shape.array()
        var in_strides: Array = A.strides.array()

        var (threads_per_block, num_blocks) = Self._launch_config[
            max_block_size
        ](total_output, reduced_volume)

        var out_device_buf = device_context.enqueue_create_buffer[DType.int32](
            total_output
        )

        var compiled = device_context.compile_function[
            reduce_argminmax[Self.dtype, max_block_size, is_max],
            reduce_argminmax[Self.dtype, max_block_size, is_max],
        ]()

        device_context.enqueue_function(
            compiled,
            out_device_buf,
            A_device_state.device_buffer(),
            in_shape,
            in_strides,
            ax,
            total_output,
            reduced_volume,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        device_context.synchronize()

        var out_state = DeviceState[DType.int32](out_device_buf^, gpu)
        return NDBuffer[DType.int32].with_device_state(out_state^, out_shape)

    @staticmethod
    def _launch_config[
        max_block_size: Int
    ](total_output: Int, reduced_volume: Int) -> Tuple[Int, Int]:
        var block_size = 1
        while block_size < reduced_volume:
            block_size <<= 1
            if block_size >= max_block_size:
                block_size = max_block_size
                break
        return (block_size, total_output)
