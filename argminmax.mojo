from tenmo import Tensor
from intarray import IntArray
from common_utils import panic
from shapes import Shape
from utils.numerics import max_finite, min_finite
from gpu import thread_idx, block_idx, block_dim, barrier
from memory import AddressSpace, stack_allocation
from array import Array
from device import DeviceState
from ndbuffer import NDBuffer
from sys import has_accelerator


fn reduce_argminmax[
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
    """
    One block per output element.
    Each thread strides over the reduction axis, tracking local best value
    and its index. Then a two-array shared-memory tree reduction picks the
    global best index for this output slot.
    """
    constrained[
        max_block_size.is_power_of_two() and max_block_size < 1024,
        "Invalid max_block_size",
    ]()

    # Shared memory: interleaved [val, idx] pairs per thread
    # We store values in smem_val and indices in smem_idx
    var smem_val = stack_allocation[
        max_block_size, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()
    var smem_idx = stack_allocation[
        max_block_size, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var out_idx = Int(block_idx.x)

    if out_idx >= total_output:
        return

    # Compute the flat base offset in the input for this output slot,
    # skipping the reduction axis (same logic as output_to_input_base
    # but inlined here for the single-axis case)
    var remaining = out_idx
    var input_base = 0
    var rank = len(in_shape)

    # We need per-axis stride for non-reduction axes to reconstruct base
    # Walk axes in reverse, skip reduction_axis
    for k in reversed(range(rank)):
        if k != reduction_axis:
            var dim = in_shape[k]
            var coord = remaining % dim
            remaining //= dim
            input_base += coord * in_strides[k]

    var axis_stride = in_strides[reduction_axis]

    # Identity initialisation
    var local_val: Scalar[dtype]
    var local_idx: Scalar[DType.int32] = 0

    @parameter
    if is_max:
        local_val = min_finite[dtype]()
    else:
        local_val = max_finite[dtype]()

    # Grid-stride loop over reduction axis
    var r = tid
    while r < reduced_volume:
        var val = (in_buffer + input_base + r * axis_stride)[]

        @parameter
        if is_max:
            if val > local_val:
                local_val = val
                local_idx = r
        else:
            if val < local_val:
                local_val = val
                local_idx = r

        r += block_size

    smem_val[tid] = local_val
    smem_idx[tid] = local_idx
    barrier()

    # Tree reduction — keep the better (val, idx) pair
    var stride = block_size >> 1
    while stride > 0:
        if tid < stride:

            @parameter
            if is_max:
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
@register_passable
struct ArgMinMaxReducer[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Unified CPU + GPU argmin/argmax on NDBuffer.
    Returns an NDBuffer[DType.int32] with the output shape.
    """

    @staticmethod
    fn reduce[
        is_max: Bool,
        max_block_size: Int = 512,
    ](
        A: NDBuffer[Self.dtype],
        axis: Int,
        keepdims: Bool = False,
    ) raises -> NDBuffer[DType.int32]:
        var shape = A.shape
        var rank = shape.rank()
        var ax = axis if axis >= 0 else axis + rank
        if ax < 0 or ax >= rank:
            panic(
                "ArgMinMaxReducer: axis",
                axis.__str__(),
                "out of range for rank",
                rank.__str__(),
            )

        # Build output shape
        var out_axes = IntArray()
        for i in range(rank):
            if i == ax:
                if keepdims:
                    out_axes.append(1)
            else:
                out_axes.append(shape[i])
        var out_shape = Shape(out_axes)
        var reduced_volume = shape[ax]
        var total_output = out_shape.num_elements()

        @parameter
        if has_accelerator():
            if A.is_on_gpu():
                return Self._gpu_reduce[is_max, max_block_size](
                    A, ax, keepdims, out_shape, total_output, reduced_volume
                )

        return Self._cpu_reduce[is_max](A, ax, keepdims, out_shape)

    # ── GPU path ──────────────────────────────────────────────────────────────

    @staticmethod
    fn _gpu_reduce[
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
        var device_context = gpu()

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

    # ── CPU path ──────────────────────────────────────────────────────────────

    @staticmethod
    fn _cpu_reduce[
        is_max: Bool,
    ](
        A: NDBuffer[Self.dtype],
        ax: Int,
        keepdims: Bool,
        out_shape: Shape,
    ) -> NDBuffer[DType.int32]:
        var shape = A.shape
        var out = NDBuffer[DType.int32].zeros(out_shape)

        for out_idx in out_shape:
            var best_val: Scalar[Self.dtype]
            var best_pos = 0

            @parameter
            if is_max:
                best_val = min_finite[Self.dtype]()
            else:
                best_val = max_finite[Self.dtype]()

            for idx in range(shape[ax]):
                var full_idx = out_idx.insert(
                    ax, idx
                ) if not keepdims else out_idx.replace(ax, idx)
                var val = A[full_idx]

                @parameter
                if is_max:
                    if val > best_val:
                        best_val = val
                        best_pos = idx
                else:
                    if val < best_val:
                        best_val = val
                        best_pos = idx

            if keepdims:
                var write_idx = out_idx.replace(ax, 0)
                out[write_idx] = best_pos
            else:
                out[out_idx] = best_pos

        return out^

    @staticmethod
    fn _launch_config[
        max_block_size: Int
    ](total_output: Int, reduced_volume: Int) -> Tuple[Int, Int]:
        var block_size = 1
        while block_size < reduced_volume:
            block_size <<= 1
            if block_size >= max_block_size:
                block_size = max_block_size
                break
        return (block_size, total_output)


# ── Public structs (thin wrappers) ────────────────────────────────────────────


struct Argmin[dtype: DType]:
    @staticmethod
    fn argmin(
        ndb: NDBuffer[Self.dtype],
        axis: Int = 0,
        keepdims: Bool = False,
    ) -> NDBuffer[DType.int32]:
        try:
            return ArgMinMaxReducer[Self.dtype].reduce[is_max=False](
                ndb, axis, keepdims
            )
        except e:
            print(e)
            panic("Argmin failed at ArgMinMaxReducer reduce")
            # Unreachable
            return NDBuffer[DType.int32].zeros(Shape())

    # Tensor convenience overload
    @staticmethod
    fn argmin(
        tensor: Tensor[Self.dtype],
        axis: Int = 0,
        keepdims: Bool = False,
    ) -> Tensor[DType.int32]:
        try:
            var result_ndb = ArgMinMaxReducer[Self.dtype].reduce[is_max=False](
                tensor.buffer, axis, keepdims
            )
            return Tensor[DType.int32](result_ndb^, requires_grad=False)
        except e:
            print(e)
            panic("Argmin tensor failed at ArgMinMaxReducer reduce")
            # Unreachable
            return Tensor[DType.int32].scalar(0)


struct Argmax[dtype: DType]:
    @staticmethod
    fn argmax(
        ndb: NDBuffer[Self.dtype],
        axis: Int = 0,
        keepdims: Bool = False,
    ) -> NDBuffer[DType.int32]:
        try:
            return ArgMinMaxReducer[Self.dtype].reduce[is_max=True](
                ndb, axis, keepdims
            )
        except e:
            print(e)
            panic("Argmax failed at ArgMinMaxReducer reduce")
            # Unreachable
            return NDBuffer[DType.int32].zeros(Shape())

    # Tensor convenience overload
    @staticmethod
    fn argmax(
        tensor: Tensor[Self.dtype],
        axis: Int = 0,
        keepdims: Bool = False,
    ) -> Tensor[DType.int32]:
        try:
            var result_ndb = ArgMinMaxReducer[Self.dtype].reduce[is_max=True](
                tensor.buffer, axis, keepdims
            )
            return Tensor[DType.int32](result_ndb^, requires_grad=False)
        except e:
            print(e)
            panic("Argmax tensor failed at ArgMinMaxReducer reduce")
            # Unreachable
            return Tensor[DType.int32].scalar(0)


fn main() raises:
    pass
