from max.gpu import thread_idx, block_idx, block_dim
from max.gpu import barrier
from std.memory import AddressSpace, stack_allocation

from ..shared.array import RankArray
from ..gpu.device import DeviceState
from ..shared.layout import Layout
from ..shared.intarray import IntArray
from ..shared.constants import MAX_RANK
from ..shared.strides import Strides
from ..shared.broadcasthelper import ShapeBroadcaster


def vector_matmul_nd[
    dtype: DType,
    block_size: Int = 256,
](
    out_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    v_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    M_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    # broadcast-resolved batch space
    batch_shape: RankArray,
    batch_strides: RankArray,
    # per-tensor batch shapes and strides (for broadcast clamping)
    v_batch_shape: RankArray,
    v_batch_strides: RankArray,
    M_batch_shape: RankArray,
    M_batch_strides: RankArray,
    # inner dimensions
    k_: Int64,
    n_: Int64,
    total_output_: Int64,  # total_batch * n
):
    var k = Int(k_)
    var n = Int(n_)
    var total_output = Int(total_output_)
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)

    if tid >= total_output:
        return

    # Step 1: decompose flat tid → (batch_idx, col_idx)
    var batch_idx = tid // n
    var col_idx = tid % n

    # Step 2: recover batch coords from batch_idx
    # Walk dims right-to-left, same pattern as output_to_input_base.
    var batch_coords = stack_allocation[MAX_RANK, Int]()
    var remaining = batch_idx
    for dim in reversed(range(len(batch_shape))):
        batch_coords[unsafe_offset=dim] = remaining % batch_shape[dim]
        remaining //= batch_shape[dim]

    # Step 3: v base offset — right-aligned broadcast clamping
    # Mirrors ShapeBroadcaster.broadcasted_indices exactly:
    #   target_idx = len(batch_shape) - len(v_batch_shape) + i
    var v_base = 0
    var v_rank_off = len(batch_shape) - len(v_batch_shape)
    for i in range(len(v_batch_shape)):
        var coord = batch_coords[unsafe_offset=v_rank_off + i] if v_batch_shape[i] > 1 else 0
        v_base += coord * v_batch_strides[i]

    # Step 4: M base offset — right-aligned broadcast clamping
    var M_base = 0
    var M_rank_off = len(batch_shape) - len(M_batch_shape)
    for i in range(len(M_batch_shape)):
        var coord = batch_coords[unsafe_offset=M_rank_off + i] if M_batch_shape[i] > 1 else 0
        M_base += coord * M_batch_strides[i]

    # Step 5: dot product over k
    # v is contiguous (to_gpu() guarantee): v_k_stride = 1
    # M is contiguous (to_gpu() guarantee): M_k_stride = n, M_n_stride = 1
    var acc = Scalar[dtype](0)
    for i in range(k):
        var v_val = v_buffer[unsafe_offset=v_base + i]
        var m_val = M_buffer[unsafe_offset=M_base + i * n + col_idx]
        acc += v_val * m_val

    # Step 6: write result
    # Output is contiguous, batch-major then col-major: flat index = tid
    out_buffer[unsafe_offset=tid] = acc


@fieldwise_init
struct VectorMatmulKernel[dtype: DType = DType.float32](
    RegisterPassable & ImplicitlyCopyable
):
    @staticmethod
    def launch[
        block_size: Int = 256,
    ](
        v_layout: Layout,
        v_device_state: DeviceState[Self.dtype],
        M_layout: Layout,
        M_device_state: DeviceState[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var v_shape = v_layout.shape
        var M_shape = M_layout.shape

        if v_shape.rank() < 1:
            raise Error("VectorMatmulKernel: vector must have rank >= 1")
        if M_shape.rank() < 2:
            raise Error("VectorMatmulKernel: matrix must have rank >= 2")

        var k = v_shape[-1]
        var k_M = M_shape[-2]
        var n = M_shape[-1]

        if k != k_M:
            raise Error("VectorMatmulKernel: inner dims must match")

        var v_batch_shape = v_shape[:-1]  # v_shape minus last dim
        var M_batch_shape = M_shape[:-2]  # M_shape minus last 2 dims

        var batch_shape = ShapeBroadcaster.broadcast_shape(
            v_batch_shape, M_batch_shape
        )

        var out_shape = batch_shape + [n]
        var total_batch = batch_shape.product()
        var total_output = total_batch * n

        var batch_strides = Strides.default(batch_shape).array()
        var v_batch_strides = v_layout.strides[:-1].array()
        var M_batch_strides = M_layout.strides[:-2].array()

        var batch_shape_arr = batch_shape.array()
        var v_batch_shape_arr = v_batch_shape.array()
        var M_batch_shape_arr = M_batch_shape.array()

        var num_blocks = (total_output + block_size - 1) // block_size

        ref gpu = v_device_state.get_gpu()
        var device_context = gpu[]

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )

        ref v_buf = v_device_state.device_buffer()
        ref M_buf = M_device_state.device_buffer()

        var compiled_func = device_context.compile_function[
            vector_matmul_nd[Self.dtype, block_size],
        ]()

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            v_buf,
            M_buf,
            batch_shape_arr,
            batch_strides,
            v_batch_shape_arr,
            v_batch_strides,
            M_batch_shape_arr,
            M_batch_strides,
            Int64(k),
            Int64(n),
            Int64(total_output),
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        if sync:
            device_context.synchronize()

        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return (
            Layout(out_shape),
            device_state^,
        )
