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


def matrix_vector_nd[
    dtype: DType,
    block_size: Int = 256,
](
    out_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    M_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    v_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    # broadcast-resolved batch space
    batch_shape: RankArray,
    batch_strides: RankArray,
    # per-tensor batch shapes and strides (for broadcast clamping)
    M_batch_shape: RankArray,
    M_batch_strides: RankArray,
    v_batch_shape: RankArray,
    v_batch_strides: RankArray,
    # inner dimensions
    m_: Int64,  # number of rows  — output width per batch
    k_: Int64,  # contraction dim
    total_output_: Int64,  # total_batch * m
):
    var m = Int(m_)
    var k = Int(k_)
    var total_output = Int(total_output_)
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)

    if tid >= total_output:
        return

    # Step 1: decompose flat tid → (batch_idx, row_idx)
    var batch_idx = tid // m
    var row_idx = tid % m

    # Step 2: recover batch coords from batch_idx
    var batch_coords = stack_allocation[MAX_RANK, Int]()
    var remaining = batch_idx
    for dim in reversed(range(len(batch_shape))):
        batch_coords[unsafe_offset=dim] = remaining % batch_shape[dim]
        remaining //= batch_shape[dim]

    # Step 3: M base offset — right-aligned broadcast clamping
    # M[..., m, k]: batch dims are all but last 2.
    # Contiguous guarantee: M_row_stride = k, M_col_stride = 1
    var M_base = 0
    var M_rank_off = len(batch_shape) - len(M_batch_shape)
    for i in range(len(M_batch_shape)):
        var coord = batch_coords[unsafe_offset=M_rank_off + i] if M_batch_shape[i] > 1 else 0
        M_base += coord * M_batch_strides[i]

    # Advance M_base to the correct row for this thread
    M_base += row_idx * k  # M_row_stride = k (contiguous)

    # Step 4: v base offset — right-aligned broadcast clamping
    # v[..., k]: batch dims are all but last 1.
    # Contiguous guarantee: v_k_stride = 1
    var v_base = 0
    var v_rank_off = len(batch_shape) - len(v_batch_shape)
    for i in range(len(v_batch_shape)):
        var coord = batch_coords[unsafe_offset=v_rank_off + i] if v_batch_shape[i] > 1 else 0
        v_base += coord * v_batch_strides[i]

    # Step 5: dot product over k
    # M row j has stride 1 (contiguous), v has stride 1 (contiguous)
    var acc = Scalar[dtype](0)
    for j in range(k):
        acc += M_buffer[unsafe_offset=M_base + j] * v_buffer[unsafe_offset=v_base + j]

    # Step 6: write result
    out_buffer[unsafe_offset=tid] = acc


@fieldwise_init
struct MatrixVectorKernel[dtype: DType = DType.float32](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def launch[
        block_size: Int = 256,
    ](
        M_layout: Layout,
        M_device_state: DeviceState[Self.dtype],
        v_layout: Layout,
        v_device_state: DeviceState[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var M_shape = M_layout.shape
        var v_shape = v_layout.shape

        if M_shape.rank() < 2:
            raise Error("MatrixVectorKernel: matrix must have rank >= 2")
        if v_shape.rank() < 1:
            raise Error("MatrixVectorKernel: vector must have rank >= 1")

        var k = M_shape[-1]
        var k_v = v_shape[-1]
        var m = M_shape[-2]

        if k != k_v:
            raise Error("MatrixVectorKernel: inner dims must match")

        var M_batch_shape = M_shape[:-2]  # M_shape minus last 2 dims
        var v_batch_shape = v_shape[:-1]  # v_shape minus last dim

        var batch_shape = ShapeBroadcaster.broadcast_shape(
            M_batch_shape, v_batch_shape
        )

        var out_shape = batch_shape + [m]
        var total_batch = batch_shape.product()
        var total_output = total_batch * m

        var batch_strides = Strides.default(batch_shape).array()
        var M_batch_strides = M_layout.strides[:-2].array()  # slice off inner k,n
        var v_batch_strides = v_layout.strides[:-1].array()  # slice off inner k

        # Convert batch shapes to RankArray for kernel
        var batch_shape_arr = batch_shape.array()
        var M_batch_shape_arr = M_batch_shape.array()
        var v_batch_shape_arr = v_batch_shape.array()

        var num_blocks = (total_output + block_size - 1) // block_size

        ref gpu = M_device_state.get_gpu()
        var device_context = gpu[]

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )

        ref M_buf = M_device_state.device_buffer()
        ref v_buf = v_device_state.device_buffer()

        var compiled_func = device_context.compile_function[
            matrix_vector_nd[Self.dtype, block_size],
        ]()

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            M_buf,
            v_buf,
            batch_shape_arr,
            batch_strides,
            M_batch_shape_arr,
            M_batch_strides,
            v_batch_shape_arr,
            v_batch_strides,
            Int64(m),
            Int64(k),
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
