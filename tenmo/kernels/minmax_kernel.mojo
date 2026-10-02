from max.gpu import thread_idx, block_idx, block_dim, grid_dim
from max.gpu import barrier
from std.memory import AddressSpace, stack_allocation
from ..shared.array import RankArray
from ..gpu.device import DeviceState
from ..shared.layout import Layout
from ..shared.intarray import IntArray
from std.utils.numerics import min_or_neg_inf, max_or_inf
from .kernel_helpers import (
    output_to_input_base,
    rank_to_reduced_offset,
    reduction_launch_config,
)


def reduce_minmax[
    dtype: DType,
    max_block_size: Int = 512,
    is_max: Bool = True,
](
    out_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: RankArray,
    in_strides: RankArray,
    reduction_axes: RankArray,
    total_output_: Int64,
    reduced_volume_: Int64,
):
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "Invalid max_block_size"

    var total_output = Int(total_output_)
    var reduced_volume = Int(reduced_volume_)

    var smem = stack_allocation[
        max_block_size, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var out_idx = Int(block_idx.x)

    if out_idx >= total_output:
        return

    # Identity: -inf for max, +inf for min
    # Identity for local accumulator
    var local: Scalar[dtype]

    comptime if is_max:
        smem[unsafe_offset=tid] = min_or_neg_inf[dtype]()
        local = min_or_neg_inf[dtype]()
    else:
        smem[unsafe_offset=tid] = max_or_inf[dtype]()
        local = max_or_inf[dtype]()

    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )

    # Grid-stride loop over reduced dimension
    var reduced_idx = tid
    while reduced_idx < reduced_volume:
        var val = (
            in_buffer
            .unsafe_offset(input_base
            + rank_to_reduced_offset(
                reduced_idx, in_shape, in_strides, reduction_axes
            ))
        )[]

        comptime if is_max:
            if val > local:
                local = val
        else:
            if val < local:
                local = val

        reduced_idx += block_size

    smem[unsafe_offset=tid] = local
    barrier()

    # Tree reduction in shared memory
    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            comptime if is_max:
                if smem[unsafe_offset=tid + stride] > smem[unsafe_offset=tid]:
                    smem[unsafe_offset=tid] = smem[unsafe_offset=tid + stride]
            else:
                if smem[unsafe_offset=tid + stride] < smem[unsafe_offset=tid]:
                    smem[unsafe_offset=tid] = smem[unsafe_offset=tid + stride]
        barrier()
        stride >>= 1

    if tid == 0:
        (out_buffer .unsafe_offset(out_idx))[] = smem[unsafe_offset=0]


def build_minmax_mask[
    dtype: DType,
    max_block_size: Int = 512,
    is_max: Bool = True,
](
    mask_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    result_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: RankArray,
    in_strides: RankArray,
    reduction_axes: RankArray,
    total_output_: Int64,
    reduced_volume_: Int64,
):
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "Invalid max_block_size"

    var total_output = Int(total_output_)
    var reduced_volume = Int(reduced_volume_)

    # smem[0..block_size)  : tie counts (Int32 cast to dtype)
    var smem = stack_allocation[
        max_block_size, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var out_idx = Int(block_idx.x)

    if out_idx >= total_output:
        return

    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )
    var best = (result_buffer .unsafe_offset(out_idx))[]

    # Pass 1: count ties in this thread's slice
    var local_count: Scalar[dtype] = 0
    var reduced_idx = tid
    while reduced_idx < reduced_volume:
        var offset = rank_to_reduced_offset(
            reduced_idx, in_shape, in_strides, reduction_axes
        )
        var val = (in_buffer .unsafe_offset(input_base + offset))[]
        if val == best:
            local_count += 1
        reduced_idx += block_size

    smem[unsafe_offset=tid] = local_count
    barrier()

    # Tree reduction to get total tie count for this output slot
    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem[unsafe_offset=tid] += smem[unsafe_offset=tid + stride]
        barrier()
        stride >>= 1

    # smem[0] now holds total tie count for out_idx
    var tie_count = smem[unsafe_offset=0]
    var inv = Scalar[dtype](1) / tie_count if tie_count > 0 else Scalar[dtype](
        0
    )

    barrier()

    # Pass 2: write normalised mask — each thread handles its slice
    reduced_idx = tid
    while reduced_idx < reduced_volume:
        var offset = rank_to_reduced_offset(
            reduced_idx, in_shape, in_strides, reduction_axes
        )
        var val = (in_buffer .unsafe_offset(input_base + offset))[]
        (mask_buffer .unsafe_offset(input_base + offset))[] = inv if val == best else Scalar[
            dtype
        ](0)
        reduced_idx += block_size


@fieldwise_init
struct MinMaxKernel[dtype: DType = DType.float32](
    RegisterPassable, ImplicitlyCopyable
):
    @staticmethod
    def launch[
        max_block_width: Int = 512,
        is_max: Bool = True,
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
        sync: Bool = False,
    ) raises -> Tuple[Tuple[Layout, DeviceState[Self.dtype]], Tuple[Layout, DeviceState[Self.dtype]]]:
        """
                Returns ((result_layout, result_storage), (mask_layout, mask_storage)).
        both on GPU.
        result: min/max values with output_shape.
        mask:   normalised gradient mask with A.shape (same shape as input).
        """
        var shape_A = A_layout.shape
        var strides_A = A_layout.strides
        var output_shape = shape_A.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var reduced_shape = shape_A.reduced_shape(normalized_axes)
        var in_shape: RankArray = shape_A.array()
        var in_strides: RankArray = strides_A.array()
        var reduction_axes: RankArray = RankArray(normalized_axes)
        var total_output: Int = output_shape.product()
        var reduced_volume: Int = reduced_shape.product()
        var total_input: Int = shape_A.num_elements()

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]
        ref A_buffer = A_device_state.device_buffer()

        var (threads_per_block, num_blocks) = Self.launch_config[
            max_block_width
        ](total_output, reduced_volume)

        # Pass 1: compute min/max values
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )
        var compiled_reduce = device_context.compile_function[
            reduce_minmax[Self.dtype, max_block_width, is_max],
        ]()
        device_context.enqueue_function(
            compiled_reduce,
            result_buffer,
            A_buffer,
            in_shape,
            in_strides,
            reduction_axes,
            Int64(total_output),
            Int64(reduced_volume),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        if sync:
            device_context.synchronize()

        # Pass 2: build normalised mask
        var mask_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_input
        )
        # Zero-initialise mask first (positions not matching get 0)
        mask_buffer.enqueue_fill(Scalar[Self.dtype](0))

        var compiled_mask = device_context.compile_function[
            build_minmax_mask[Self.dtype, max_block_width, is_max],
        ]()
        device_context.enqueue_function(
            compiled_mask,
            mask_buffer,
            A_buffer,
            result_buffer,
            in_shape,
            in_strides,
            reduction_axes,
            Int64(total_output),
            Int64(reduced_volume),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        var result_pair = (
            Layout(output_shape),
            result_state^,
        )

        var mask_state = DeviceState[Self.dtype](mask_buffer^, gpu)
        var mask_pair = (
            Layout(shape_A),
            mask_state^,
        )

        return (result_pair, mask_pair)

    @staticmethod
    def launch_config[
        max_block_size: Int
    ](total_output: Int, reduced_volume: Int,) -> Tuple[Int, Int]:
        return reduction_launch_config[max_block_size](total_output, reduced_volume)
