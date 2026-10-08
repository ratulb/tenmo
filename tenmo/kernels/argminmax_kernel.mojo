# argminmax_kernel.mojo — GPU argmin/argmax kernel

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu import barrier
from std.memory import AddressSpace, stack_allocation
from std.utils.numerics import max_finite, min_finite
from ..shared.array import RankArray
from ..gpu.device import DeviceState
from ..shared.layout import Layout
from ..shared.shapes import Shape
from ..shared.mnemonics import DEFAULT_INDEX_DTYPE
from .kernel_helpers import reduction_launch_config


def reduce_argminmax[
    dtype: DType,
    index_dtype: DType = DEFAULT_INDEX_DTYPE,
    max_block_size: Int = 512,
    is_max: Bool = True,
](
    out_buffer: Pointer[Scalar[index_dtype], MutAnyOrigin],
    in_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: RankArray,
    in_strides: RankArray,
    reduction_axis_: Int64,
    total_output_: Int64,
    reduced_volume_: Int64,
    offset_: Int64,
):
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size < 1024
    ), "Invalid max_block_size"

    var reduction_axis = Int(reduction_axis_)
    var total_output = Int(total_output_)
    var reduced_volume = Int(reduced_volume_)

    var smem_val = stack_allocation[
        max_block_size, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var smem_idx = stack_allocation[
        max_block_size, Scalar[index_dtype], address_space=AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var out_idx = Int(block_idx.x)

    if out_idx >= total_output:
        return

    var remaining = out_idx
    var input_base = Int(offset_)  # GPU offset fix: seed base with view offset
    var rank = len(in_shape)

    for k in reversed(range(rank)):
        if k != reduction_axis:
            var dim = in_shape[k]
            var coord = remaining % dim
            remaining //= dim
            input_base += coord * in_strides[k]

    var axis_stride = in_strides[reduction_axis]

    var local_val: Scalar[dtype]
    var local_idx: Scalar[index_dtype] = 0

    comptime if is_max:
        local_val = min_finite[dtype]()
    else:
        local_val = max_finite[dtype]()

    var r = tid
    while r < reduced_volume:
        var val = (in_buffer .unsafe_offset(input_base + r * axis_stride))[]

        comptime if is_max:
            if val > local_val:
                local_val = val
                local_idx = Scalar[index_dtype](r)
        else:
            if val < local_val:
                local_val = val
                local_idx = Scalar[index_dtype](r)

        r += block_size

    smem_val[unsafe_offset=tid] = local_val
    smem_idx[unsafe_offset=tid] = local_idx
    barrier()

    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:
            comptime if is_max:
                if smem_val[unsafe_offset=tid + stride] > smem_val[unsafe_offset=tid]:
                    smem_val[unsafe_offset=tid] = smem_val[unsafe_offset=tid + stride]
                    smem_idx[unsafe_offset=tid] = smem_idx[unsafe_offset=tid + stride]
            else:
                if smem_val[unsafe_offset=tid + stride] < smem_val[unsafe_offset=tid]:
                    smem_val[unsafe_offset=tid] = smem_val[unsafe_offset=tid + stride]
                    smem_idx[unsafe_offset=tid] = smem_idx[unsafe_offset=tid + stride]

        barrier()
        stride >>= 1

    if tid == 0:
        (out_buffer .unsafe_offset(out_idx))[] = smem_idx[unsafe_offset=0]


@fieldwise_init
struct ArgMinMaxKernel[dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def _gpu_reduce[
        is_max: Bool,
        max_block_size: Int,
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        ax: Int,
        keepdims: Bool,
        out_shape: Shape,
        total_output: Int,
        reduced_volume: Int,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.index_dtype]]:
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        var in_shape: RankArray = A_layout.shape.array()
        var in_strides: RankArray = A_layout.strides.array()

        var (threads_per_block, num_blocks) = Self._launch_config[
            max_block_size
        ](total_output, reduced_volume)

        var out_device_buf = device_context.enqueue_create_buffer[
            Self.index_dtype
        ](total_output)

        var compiled = device_context.compile_function[
            reduce_argminmax[
                Self.dtype, Self.index_dtype, max_block_size, is_max
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            out_device_buf,
            A_device_state.device_buffer(),
            in_shape,
            in_strides,
            Int64(ax),
            Int64(total_output),
            Int64(reduced_volume),
            Int64(A_layout.offset),  # GPU offset fix
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        if sync:
            device_context.synchronize()

        var out_state = DeviceState[Self.index_dtype](out_device_buf^, gpu)
        return (
            Layout(out_shape),
            out_state^,
        )

    @staticmethod
    def _launch_config[
        max_block_size: Int
    ](total_output: Int, reduced_volume: Int) -> Tuple[Int, Int]:
        return reduction_launch_config[max_block_size](total_output, reduced_volume)
