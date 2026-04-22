from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.memory import AddressSpace, stack_allocation

from array import Array
from device import DeviceState
from ndbuffer import NDBuffer
from intarray import IntArray
from mnemonics import max_rank
from strides import Strides
from broadcasthelper import ShapeBroadcaster

# ── Kernel ────────────────────────────────────────────────────────────────────


fn vector_matmul_nd[
    dtype: DType,
    block_size: Int = 256,
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    v_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    M_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    # broadcast-resolved batch space
    batch_shape: Array,
    batch_strides: Array,
    # per-tensor batch shapes and strides (for broadcast clamping)
    v_batch_shape: Array,
    v_batch_strides: Array,
    M_batch_shape: Array,
    M_batch_strides: Array,
    # inner dimensions
    k: Int,
    n: Int,
    total_output: Int,  # total_batch * n
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)

    if tid >= total_output:
        return

    # ── Step 1: decompose flat tid → (batch_idx, col_idx) ────────────────────
    var batch_idx = tid // n
    var col_idx = tid % n

    # ── Step 2: recover batch coords from batch_idx ───────────────────────────
    # Walk dims right-to-left, same pattern as output_to_input_base.
    var batch_coords = stack_allocation[max_rank, Int]()
    var remaining = batch_idx
    for dim in reversed(range(len(batch_shape))):
        batch_coords[dim] = remaining % batch_shape[dim]
        remaining //= batch_shape[dim]

    # ── Step 3: v base offset — right-aligned broadcast clamping ─────────────
    # Mirrors ShapeBroadcaster.broadcasted_indices exactly:
    #   target_idx = len(batch_shape) - len(v_batch_shape) + i
    var v_base = 0
    var v_rank_off = len(batch_shape) - len(v_batch_shape)
    for i in range(len(v_batch_shape)):
        var coord = batch_coords[v_rank_off + i] if v_batch_shape[i] > 1 else 0
        v_base += coord * v_batch_strides[i]

    # ── Step 4: M base offset — right-aligned broadcast clamping ─────────────
    var M_base = 0
    var M_rank_off = len(batch_shape) - len(M_batch_shape)
    for i in range(len(M_batch_shape)):
        var coord = batch_coords[M_rank_off + i] if M_batch_shape[i] > 1 else 0
        M_base += coord * M_batch_strides[i]

    # ── Step 5: dot product over k ────────────────────────────────────────────
    # v is contiguous (to_gpu() guarantee): v_k_stride = 1
    # M is contiguous (to_gpu() guarantee): M_k_stride = n, M_n_stride = 1
    var acc = Scalar[dtype](0)
    for i in range(k):
        var v_val = v_buffer[v_base + i]
        var m_val = M_buffer[M_base + i * n + col_idx]
        acc += v_val * m_val

    # ── Step 6: write result ──────────────────────────────────────────────────
    # Output is contiguous, batch-major then col-major: flat index = tid
    out_buffer[tid] = acc


# ── Host-side launch wrapper ──────────────────────────────────────────────────


@fieldwise_init
struct VectorMatmulNdGpu[dtype: DType = DType.float32](
    RegisterPassable & ImplicitlyCopyable & Movable
):
    @staticmethod
    fn launch[
        block_size: Int = 256,
    ](
        v: NDBuffer[Self.dtype],
        M: NDBuffer[Self.dtype],
    ) raises -> NDBuffer[
        Self.dtype
    ]:
        # ── Shape extraction ──────────────────────────────────────────────────
        var v_shape = v.shape
        var M_shape = M.shape

        # ── Validation ────────────────────────────────────────────────────────
        if v_shape.rank() < 1:
            raise Error("VectorMatmulNdGpu: vector must have rank >= 1")
        if M_shape.rank() < 2:
            raise Error("VectorMatmulNdGpu: matrix must have rank >= 2")

        var k = v_shape[-1]
        var k_M = M_shape[-2]
        var n = M_shape[-1]

        if k != k_M:
            raise Error("VectorMatmulNdGpu: inner dims must match")

        # ── Batch shapes ──────────────────────────────────────────────────────
        var v_batch_shape = v_shape[:-1]  # v_shape minus last dim
        var M_batch_shape = M_shape[:-2]  # M_shape minus last 2 dims

        var batch_shape = ShapeBroadcaster.broadcast_shape(
            v_batch_shape, M_batch_shape
        )

        # ── Output shape and sizes ────────────────────────────────────────────
        var out_shape = batch_shape + [n]
        var total_batch = batch_shape.product()
        var total_output = total_batch * n

        # ── Strides (row-major, contiguous — to_gpu() guarantee) ─────────────
        var batch_strides = Strides.default(batch_shape).array()
        var v_batch_strides = v.strides[:-1].array()
        var M_batch_strides = M.strides[:-2].array()

        # Convert batch shapes to Array for kernel
        var batch_shape_arr = batch_shape.array()
        var v_batch_shape_arr = v_batch_shape.array()
        var M_batch_shape_arr = M_batch_shape.array()

        # ── Launch config ─────────────────────────────────────────────────────
        var num_blocks = (total_output + block_size - 1) // block_size

        # ── Device setup ──────────────────────────────────────────────────────
        ref v_device_state = v.device_state.value()
        ref gpu = v_device_state.get_gpu()
        var device_context = gpu()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )

        ref v_buf = v_device_state.device_buffer()
        ref M_buf = M.device_state.value().device_buffer()

        # ── Compile and enqueue ───────────────────────────────────────────────
        var compiled_func = device_context.compile_function[
            vector_matmul_nd[Self.dtype, block_size],
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
            k,
            n,
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


fn main() raises:
    pass
