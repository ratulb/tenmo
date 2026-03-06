from sys import simd_width_of
from gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from memory import AddressSpace, stack_allocation

from array import Array

from device import DeviceState
from ndbuffer import NDBuffer
from intarray import IntArray


fn output_to_input_base(
    out_idx: Int,
    in_shape: Array,
    in_strides: Array,
    reduction_axes: Array,
) -> Int:
    var remaining = out_idx
    var input_base = 0

    for k in reversed(range(len(in_shape))):
        if k not in reduction_axes:
            var coord = remaining % in_shape[k]
            remaining //= in_shape[k]
            input_base += coord * in_strides[k]

    return input_base


fn rank_to_reduced_offset(
    rank: Int, in_shape: Array, in_strides: Array, reduction_axes: Array
) -> Int:
    var tmp = rank
    var offset = 0

    for k in reversed(range(len(in_shape))):
        if k in reduction_axes:
            var coord = tmp % in_shape[k]
            offset += coord * in_strides[k]

    return offset


fn sum_hybrid[
    dtype: DType, max_block_size: Int = 512  # Needs to be power of 2
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: Array,
    in_strides: Array,
    reduction_axes: Array,
    total_output: Int,
    reduced_volume: Int,
):
    constrained[
        max_block_size.is_power_of_two() and max_block_size < 1024,
        "Invalid max_block_size",
    ]()
    var smem = stack_allocation[
        max_block_size, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var out_idx = Int(block_idx.x)

    if out_idx >= total_output:
        return

    smem[tid] = Scalar[dtype](0)

    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )
    var local = Scalar[dtype](0)

    var rank = tid

    while rank < reduced_volume:
        local += (
            in_buffer
            + input_base
            + rank_to_reduced_offset(rank, in_shape, in_strides, reduction_axes)
        )[]
        rank += block_size

    smem[tid] = local

    barrier()

    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        (out_buffer + out_idx)[] = smem[0]


@fieldwise_init
@register_passable
struct SumReduction[dtype: DType = DType.float32](ImplicitlyCopyable & Movable):
    @staticmethod
    fn launch[
        max_block_width: Int = 512,
    ](
        A: NDBuffer[Self.dtype], normalized_axes: IntArray, keepdims: Bool
    ) raises -> NDBuffer[Self.dtype]:
        var shape_A = A.shape
        var strides_A = A.strides
        var output_shape = shape_A.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var reduced_shape = shape_A.reduced_shape(normalized_axes)
        var in_shape: Array = shape_A.array()
        var in_strides: Array = strides_A.array()
        var reduction_axes: Array = Array(normalized_axes)
        var total_output: Int = output_shape.product()
        var reduced_volume: Int = reduced_shape.product()

        # Launch configuration - num_blocks represent output size/total_output
        var (threads_per_block, num_blocks) = Self.launch_config[
            max_block_width
        ](total_output, reduced_volume)

        ref A_device_state = A.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )

        ref A_buffer = A_device_state.device_buffer()

        var compiled_func = device_context.compile_function[
            sum_hybrid[Self.dtype, max_block_width],
            sum_hybrid[Self.dtype, max_block_width],
        ]()

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            in_shape,
            in_strides,
            reduction_axes,
            total_output,
            reduced_volume,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()
        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        var out = NDBuffer[Self.dtype].with_device_state(
            device_state^, output_shape
        )
        return out^

    @staticmethod
    fn launch_config[
        max_block_size: Int
    ](total_output: Int, reduced_volume: Int,) -> Tuple[Int, Int]:
        var block_size = 1
        while block_size < reduced_volume:
            block_size <<= 1
            if block_size >= max_block_size:
                block_size = max_block_size
                break
        # total_output -> num_blocks
        return (block_size, total_output)


fn main() raises:
    test_sum_all()
    test_sum_partial()


from tenmo import Tensor
from testing import assert_true


fn test_sum_all() raises:
    print("test_sum_all")
    comptime dtype = DType.float32
    var a = Tensor[dtype].ones(2, 3, 4)
    var cpu_result = a.sum()
    cpu_result.print()
    assert_true(cpu_result == Tensor[dtype].scalar(24))
    var a_gpu = a.to_gpu()
    var gpu_result = a_gpu.sum()
    var cpu_result_gpu = cpu_result.to_gpu()
    gpu_result.print()
    assert_true(cpu_result_gpu.all_close(gpu_result))

    cpu_result = a.sum(keepdims=True)
    gpu_result = a_gpu.sum(keepdims=True)
    cpu_result.print()
    gpu_result.print()
    cpu_result_gpu = cpu_result.to_gpu()
    assert_true(cpu_result_gpu== gpu_result)
    assert_true(cpu_result.to_gpu().all_close(gpu_result))
    assert_true(cpu_result.all_close(gpu_result.to_cpu()))


fn test_sum_partial() raises:
    print("test_sum_partial")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(2 * 3 * 4)
    a = a.reshape(2, 3, 4)
    var cpu_result = a.sum(axes=[1])
    cpu_result.print()
    var a_gpu = a.to_gpu()
    var gpu_result = a_gpu.sum(axes=[1])
    var cpu_result_gpu = cpu_result.to_gpu()
    gpu_result.print()
    assert_true(cpu_result_gpu.all_close(gpu_result))

    cpu_result = a.sum(axes=[0], keepdims=True)
    gpu_result = a_gpu.sum(axes=[0], keepdims=True)
    cpu_result.print()
    gpu_result.print()
    cpu_result_gpu = cpu_result.to_gpu()
    assert_true(cpu_result_gpu== gpu_result)
    assert_true(cpu_result.to_gpu().all_close(gpu_result))
    assert_true(cpu_result.all_close(gpu_result.to_cpu()))
