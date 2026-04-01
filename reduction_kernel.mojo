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

    # When reducing all axes, there's only one output element
    if len(reduction_axes) == 0:
        return 0

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

    # Determine which axes are being reduced
    var reduce_all = len(reduction_axes) == 0

    for k in reversed(range(len(in_shape))):
        if reduce_all or k in reduction_axes:
            var coord = tmp % in_shape[k]
            tmp //= in_shape[k]  # Consume this dimension
            offset += coord * in_strides[k]

    return offset


fn reduce[
    dtype: DType,
    max_block_size: Int = 512,  # Needs to be power of 2
    mean: Bool = False,
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

        @parameter
        if mean:
            (out_buffer + out_idx)[] = smem[0] / Scalar[dtype](
                max(reduced_volume, 1)
            )
        else:
            (out_buffer + out_idx)[] = smem[0]


@fieldwise_init
@register_passable
struct Reduction[dtype: DType = DType.float32](ImplicitlyCopyable & Movable):
    @staticmethod
    fn launch[
        max_block_width: Int = 512, mean: Bool = False
    ](
        A: NDBuffer[Self.dtype], normalized_axes: IntArray, keepdims: Bool
    ) raises -> NDBuffer[Self.dtype]:
        var shape_A = A.shape
        var strides_A = A.strides
        var output_shape = shape_A.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )

        var normalized_axes_copy = normalized_axes
        if len(normalized_axes_copy) == 0:
            # Reduce all axes
            normalized_axes_copy = IntArray(len(shape_A))
            for i in range(len(shape_A)):
                normalized_axes_copy[i] = i

        var reduction_axes: Array = Array(normalized_axes_copy)

        var reduced_shape = shape_A.reduced_shape(normalized_axes)
        var in_shape: Array = shape_A.array()
        var in_strides: Array = strides_A.array()
        #var reduction_axes: Array = Array(normalized_axes)
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
            reduce[Self.dtype, max_block_width, mean],
            reduce[Self.dtype, max_block_width, mean],
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
    test_mean()

from tenmo import Tensor
from testing import assert_true

from device import GPU
from shapes import Shape
fn test_mean() raises:
    print("test_mean")
    comptime dtype = DType.float32
    var ndb = NDBuffer[dtype](10, 20, 30)
    #ndb.print()
    s1 = ndb.sum(IntArray())
    s1.print()
    var ndb_gpu = ndb.to_gpu(GPU())
    #ndb_gpu.print()
    s2 = ndb_gpu.sum(IntArray())
    s2.print()
