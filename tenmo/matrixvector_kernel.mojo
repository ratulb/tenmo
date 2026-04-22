from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.memory import AddressSpace, stack_allocation

from .array import Array
from .device import DeviceState
from .ndbuffer import NDBuffer
from .intarray import IntArray
from .mnemonics import max_rank
from .strides import Strides
from .broadcasthelper import ShapeBroadcaster


# ── Kernel ────────────────────────────────────────────────────────────────────


fn matrix_vector_nd[
    dtype: DType,
    block_size: Int = 256,
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    M_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    # broadcast-resolved batch space
    batch_shape: Array,
    batch_strides: Array,
    # per-tensor batch shapes and strides (for broadcast clamping)
    M_batch_shape: Array,
    M_batch_strides: Array,
    v_batch_shape: Array,
    v_batch_strides: Array,
    # inner dimensions
    m: Int,  # number of rows  — output width per batch
    k: Int,  # contraction dim
    total_output: Int,  # total_batch * m
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)

    if tid >= total_output:
        return

    # ── Step 1: decompose flat tid → (batch_idx, row_idx) ────────────────────
    var batch_idx = tid // m
    var row_idx = tid % m

    # ── Step 2: recover batch coords from batch_idx ───────────────────────────
    var batch_coords = stack_allocation[max_rank, Int]()
    var remaining = batch_idx
    for dim in reversed(range(len(batch_shape))):
        batch_coords[dim] = remaining % batch_shape[dim]
        remaining //= batch_shape[dim]

    # ── Step 3: M base offset — right-aligned broadcast clamping ─────────────
    # M[..., m, k]: batch dims are all but last 2.
    # Contiguous guarantee: M_row_stride = k, M_col_stride = 1
    var M_base = 0
    var M_rank_off = len(batch_shape) - len(M_batch_shape)
    for i in range(len(M_batch_shape)):
        var coord = batch_coords[M_rank_off + i] if M_batch_shape[i] > 1 else 0
        M_base += coord * M_batch_strides[i]

    # Advance M_base to the correct row for this thread
    M_base += row_idx * k  # M_row_stride = k (contiguous)

    # ── Step 4: v base offset — right-aligned broadcast clamping ─────────────
    # v[..., k]: batch dims are all but last 1.
    # Contiguous guarantee: v_k_stride = 1
    var v_base = 0
    var v_rank_off = len(batch_shape) - len(v_batch_shape)
    for i in range(len(v_batch_shape)):
        var coord = batch_coords[v_rank_off + i] if v_batch_shape[i] > 1 else 0
        v_base += coord * v_batch_strides[i]

    # ── Step 5: dot product over k ────────────────────────────────────────────
    # M row j has stride 1 (contiguous), v has stride 1 (contiguous)
    var acc = Scalar[dtype](0)
    for j in range(k):
        acc += M_buffer[M_base + j] * v_buffer[v_base + j]

    # ── Step 6: write result ──────────────────────────────────────────────────
    out_buffer[tid] = acc


# ── Host-side launch wrapper ──────────────────────────────────────────────────


@fieldwise_init
struct MatrixVectorNdGpu[dtype: DType = DType.float32](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    fn launch[
        block_size: Int = 256,
    ](
        M: NDBuffer[Self.dtype],
        v: NDBuffer[Self.dtype],
    ) raises -> NDBuffer[
        Self.dtype
    ]:
        # ── Shape extraction ──────────────────────────────────────────────────
        var M_shape = M.shape
        var v_shape = v.shape

        # ── Validation ────────────────────────────────────────────────────────
        if M_shape.rank() < 2:
            raise Error("MatrixVectorNdGpu: matrix must have rank >= 2")
        if v_shape.rank() < 1:
            raise Error("MatrixVectorNdGpu: vector must have rank >= 1")

        var k = M_shape[-1]
        var k_v = v_shape[-1]
        var m = M_shape[-2]

        if k != k_v:
            raise Error("MatrixVectorNdGpu: inner dims must match")

        # ── Batch shapes ──────────────────────────────────────────────────────
        var M_batch_shape = M_shape[:-2]  # M_shape minus last 2 dims
        var v_batch_shape = v_shape[:-1]  # v_shape minus last dim

        var batch_shape = ShapeBroadcaster.broadcast_shape(
            M_batch_shape, v_batch_shape
        )

        # ── Output shape and sizes ────────────────────────────────────────────
        var out_shape = batch_shape + [m]
        var total_batch = batch_shape.product()
        var total_output = total_batch * m

        # ── Strides (row-major, contiguous — to_gpu() guarantee) ─────────────
        var batch_strides = Strides.default(batch_shape).array()
        var M_batch_strides = M.strides[:-2].array()  # slice off inner k,n
        var v_batch_strides = v.strides[:-1].array()  # slice off inner k

        # Convert batch shapes to Array for kernel
        var batch_shape_arr = batch_shape.array()
        var M_batch_shape_arr = M_batch_shape.array()
        var v_batch_shape_arr = v_batch_shape.array()

        # ── Launch config ─────────────────────────────────────────────────────
        var num_blocks = (total_output + block_size - 1) // block_size

        # ── Device setup ──────────────────────────────────────────────────────
        ref M_device_state = M.device_state.value()
        ref gpu = M_device_state.get_gpu()
        var device_context = gpu()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )

        ref M_buf = M_device_state.device_buffer()
        ref v_buf = v.device_state.value().device_buffer()

        # ── Compile and enqueue ───────────────────────────────────────────────
        var compiled_func = device_context.compile_function[
            matrix_vector_nd[Self.dtype, block_size],
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
            m,
            k,
            total_output,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        device_context.synchronize()

        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        var out = NDBuffer[Self.dtype].with_device_state(
            device_state^, out_shape
        )
        return out^

