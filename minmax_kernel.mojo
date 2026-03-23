from gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from memory import AddressSpace, stack_allocation
from array import Array
from device import DeviceState
from ndbuffer import NDBuffer
from intarray import IntArray
from utils.numerics import min_or_neg_inf, max_or_inf


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


fn reduce_minmax[
    dtype: DType,
    max_block_size: Int = 512,
    is_max: Bool = True,
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

    # Identity: -inf for max, +inf for min
    # Identity for local accumulator
    var local: Scalar[dtype]

    @parameter
    if is_max:
        smem[tid] = min_or_neg_inf[dtype]()
        local = min_or_neg_inf[dtype]()
    else:
        smem[tid] = max_or_inf[dtype]()
        local = max_or_inf[dtype]()

    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )

    # Grid-stride loop over reduced dimension
    var rank = tid
    while rank < reduced_volume:
        var val = (
            in_buffer
            + input_base
            + rank_to_reduced_offset(rank, in_shape, in_strides, reduction_axes)
        )[]

        @parameter
        if is_max:
            if val > local:
                local = val
        else:
            if val < local:
                local = val

        rank += block_size

    smem[tid] = local
    barrier()

    # Tree reduction in shared memory
    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:

            @parameter
            if is_max:
                if smem[tid + stride] > smem[tid]:
                    smem[tid] = smem[tid + stride]
            else:
                if smem[tid + stride] < smem[tid]:
                    smem[tid] = smem[tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        (out_buffer + out_idx)[] = smem[0]


fn main() raises:
    pass
