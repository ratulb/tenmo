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

    # Collect non-reduction axes in forward order — these form the output shape
    var non_reduction_axes = Array()
    for k in range(len(in_shape)):
        if k not in reduction_axes:
            non_reduction_axes.append(k)

    # Decode out_idx using output dimensions in reverse
    for i in reversed(range(len(non_reduction_axes))):
        var k = non_reduction_axes[i]
        var coord = remaining % in_shape[k]
        remaining //= in_shape[k]
        input_base += coord * in_strides[k]

    return input_base


fn output_to_input_base_orig(
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

    # Collect reduction axes in forward order so we decode tmp correctly
    var red_axes = Array()
    for k in range(len(in_shape)):
        if k in reduction_axes:
            red_axes.append(k)

    # Decode tmp in reverse over reduction axes only
    for i in reversed(range(len(red_axes))):
        var k = red_axes[i]
        var coord = tmp % in_shape[k]
        tmp //= in_shape[k]
        offset += coord * in_strides[k]

    return offset

fn rank_to_reduced_offset_orig(
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


fn build_minmax_mask[
    dtype: DType,
    max_block_size: Int = 512,
    is_max: Bool = True,
](
    mask_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
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

    # smem[0..block_size)  : tie counts (Int32 cast to dtype)
    var smem = stack_allocation[
        max_block_size, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var out_idx = Int(block_idx.x)

    if out_idx >= total_output:
        return

    var input_base = output_to_input_base(
        out_idx, in_shape, in_strides, reduction_axes
    )
    var best = (result_buffer + out_idx)[]

    # Pass 1: count ties in this thread's slice
    var local_count: Scalar[dtype] = 0
    var rank = tid
    while rank < reduced_volume:
        var offset = rank_to_reduced_offset(
            rank, in_shape, in_strides, reduction_axes
        )
        var val = (in_buffer + input_base + offset)[]
        if val == best:
            local_count += 1
        rank += block_size

    smem[tid] = local_count
    barrier()

    # Tree reduction to get total tie count for this output slot
    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride >>= 1

    # smem[0] now holds total tie count for out_idx
    var tie_count = smem[0]
    var inv = Scalar[dtype](1) / tie_count if tie_count > 0 else Scalar[dtype](
        0
    )

    barrier()

    # Pass 2: write normalised mask — each thread handles its slice
    rank = tid
    while rank < reduced_volume:
        var offset = rank_to_reduced_offset(
            rank, in_shape, in_strides, reduction_axes
        )
        var val = (in_buffer + input_base + offset)[]
        (mask_buffer + input_base + offset)[] = inv if val == best else Scalar[
            dtype
        ](0)
        rank += block_size


@fieldwise_init
@register_passable
struct ReductionMinMax[dtype: DType = DType.float32](
    ImplicitlyCopyable & Movable
):
    @staticmethod
    fn launch[
        max_block_width: Int = 512,
        is_max: Bool = True,
    ](
        A: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
    ) raises -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        """
        Returns (result, mask) both on GPU.
        result: min/max values with output_shape.
        mask:   normalised gradient mask with A.shape (same shape as input).
        """
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
        var total_input: Int = shape_A.num_elements()

        ref A_device_state = A.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()
        ref A_buffer = A_device_state.device_buffer()

        var (threads_per_block, num_blocks) = Self.launch_config[
            max_block_width
        ](total_output, reduced_volume)

        # ── Pass 1: compute min/max values ──────────────────────────────────
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )
        var compiled_reduce = device_context.compile_function[
            reduce_minmax[Self.dtype, max_block_width, is_max],
            reduce_minmax[Self.dtype, max_block_width, is_max],
        ]()
        device_context.enqueue_function(
            compiled_reduce,
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

        # ── Pass 2: build normalised mask ────────────────────────────────────
        var mask_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_input
        )
        # Zero-initialise mask first (positions not matching get 0)
        mask_buffer.enqueue_fill(Scalar[Self.dtype](0))

        var compiled_mask = device_context.compile_function[
            build_minmax_mask[Self.dtype, max_block_width, is_max],
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
            total_output,
            reduced_volume,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        var result_ndb = NDBuffer[Self.dtype].with_device_state(
            result_state^, output_shape
        )

        var mask_state = DeviceState[Self.dtype](mask_buffer^, gpu)
        var mask_ndb = NDBuffer[Self.dtype].with_device_state(
            mask_state^, shape_A
        )

        return (result_ndb^, mask_ndb^)

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
    pass
