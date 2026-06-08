from std.sys import simd_width_of
from std.gpu import thread_idx, block_idx, block_dim, grid_dim
from tenmo.mnemonics import (
    Add,
    Multiply,
    Subtract,
    Divide,
    max_rank,
    SIGMOID_BACKWARD,
    TANH_BACKWARD,
    LOG_BACKWARD,
    SQRT_BACKWARD,
)
from tenmo.strides import Strides
from tenmo.broadcasthelper import ShapeBroadcaster
from tenmo.device import DeviceState
from tenmo.array import Array
from tenmo.ndbuffer import NDBuffer
from tenmo.kernels.kernel_helpers import simd_op, scalar_op


# =============================================================================
# KERNEL 1 — Both contiguous, same shape, no broadcasting.
#
# Unchanged from original — pure linear indexing, no coordinate decomposition.
# Fastest possible path. No bugs.
# =============================================================================
def arithmetic_ops_both_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
):
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var grid_stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var vec_b = B.load[width=simd_width](i)
                var vec_result = simd_op[op_code, dtype, simd_width](
                    vec_a, vec_b, epsilon
                )
                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var idx = i + j
                    result[idx] = scalar_op[op_code, dtype](
                        A[idx], B[idx], epsilon
                    )

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL 2 — Both contiguous, shapes differ — broadcast expansion needed.
#
# Preconditions (enforced by launcher):
#   A.is_contiguous() and B.is_contiguous()
#   A_shape != B_shape
#
# Strategy:
#   Both A and B are flat in memory. Compute the outer base address once per
#   SIMD vector, then load simd_width consecutive elements.
#
# CORRECTNESS FIX — double-counting bug in original:
#   The original loop ran from dim=rank-1 down to dim=1, accumulating
#   coord_{rank-1} * stride into a_base, THEN added inner_offset = i % inner_dim
#   which equals coord_{rank-1} again. This counted the innermost coordinate
#   twice, producing wrong physical addresses.
#
#   Fix: strip the innermost coordinate FIRST via i % inner_dim and i // inner_dim.
#   Then decompose only the outer dimensions (rank-2 down to 0) to build the base.
#   Finally add inner_offset once when computing the load address.
#
#   Trace verification (rank=2, shape=(4,6), i=7):
#     inner_offset = 7 % 6 = 1
#     outer_remaining = 7 // 6 = 1
#     dim=0: coord = 1 % 4 = 1,  a_base += 1 * 6 = 6,  outer_remaining = 0
#     load: a_base + inner_offset = 6 + 1 = 7  ✓
#
# PERFORMANCE: one coord decomp per SIMD vector + one SIMD load each for A and B,
# vs the old per-lane decomp + scalar loads (simd_width decomps + simd_width loads).
#
# Row-boundary handling:
#   When inner_offset + simd_width > inner_dim the vector would cross a row
#   boundary. In that case A and B elements are no longer consecutive in both
#   tensors simultaneously so we fall back to per-lane scalar decomposition.
#   This happens at most once per row per thread — negligible in practice.
#
# B inner stride cases:
#   B_strides[rank-1] == 1  → B has the inner dim, SIMD load from B
#   B_strides[rank-1] == 0  → B broadcasts across inner dim, scalar splat
# =============================================================================


def arithmetic_ops_both_contiguous_broadcast[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: Array,
    A_strides: Array,  # broadcast strides from Strides.default(A_shape)
    B_strides: Array,  # broadcast strides from Strides.default(B_shape)
    size: Int,
    rank: Int,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
):
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var grid_stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    # Innermost dimension size — controls row-boundary detection.
    var inner_dim = result_shape[rank - 1]

    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                # ── Compute outer base addresses ──────────────────────────
                # CORRECTNESS FIX: strip innermost coord first, decompose
                # outer dims only. This prevents double-counting the innermost
                # coordinate in the load address.
                var inner_offset = i % inner_dim
                var outer_remaining = i // inner_dim

                var a_base = 0
                var b_base = 0

                for dim in range(rank - 2, -1, -1):
                    var coord = outer_remaining % result_shape[dim]
                    a_base += coord * A_strides[dim]
                    b_base += coord * B_strides[dim]
                    outer_remaining //= result_shape[dim]

                # ── Fast path: vector fits within one row ─────────────────
                # All simd_width elements share the same outer coordinates.
                # A elements at [outer, inner_offset..inner_offset+simd_width]
                # are consecutive in memory (A is contiguous, A_strides[rank-1]=1).
                # B elements may be consecutive (B_strides[rank-1]==1) or
                # broadcast (B_strides[rank-1]==0).
                _="""if inner_offset + simd_width <= inner_dim:
                    var vec_a = A.load[width=simd_width](a_base + inner_offset)
                    var vec_result: SIMD[dtype, simd_width]

                    if B_strides[rank - 1] == 1:
                        # B has the inner dim — consecutive SIMD load
                        var vec_b = B.load[width=simd_width](
                            b_base + inner_offset
                        )
                        vec_result = simd_op[op_code, dtype, simd_width](
                            vec_a, vec_b, epsilon
                        )
                    else:
                        # B broadcasts across inner dim — scalar splat
                        var vec_b = SIMD[dtype, simd_width](B[b_base])
                        vec_result = simd_op[op_code, dtype, simd_width](
                            vec_a, vec_b, epsilon
                        )

                    result.store[width=simd_width](i, vec_result)"""

                if inner_offset + simd_width <= inner_dim:
                    # A stride guard — mirrors B guard
                    var vec_a: SIMD[dtype, simd_width]
                    if A_strides[rank - 1] == 1:
                        vec_a = A.load[width=simd_width](a_base + inner_offset)
                    else:
                        # A broadcasts inner dim (e.g. A_shape=(N,1)) — scalar splat
                        vec_a = SIMD[dtype, simd_width](A[a_base])

                    var vec_result: SIMD[dtype, simd_width]
                    if B_strides[rank - 1] == 1:
                        var vec_b = B.load[width=simd_width](b_base + inner_offset)
                        vec_result = simd_op[op_code, dtype, simd_width](vec_a, vec_b, epsilon)
                    else:
                        var vec_b = SIMD[dtype, simd_width](B[b_base])
                        vec_result = simd_op[op_code, dtype, simd_width](vec_a, vec_b, epsilon)

                    result.store[width=simd_width](i, vec_result)


                else:
                    # ── Slow path: crosses row boundary ───────────────────
                    # Per-lane scalar decomposition. Happens at most once per
                    # row per thread — amortised cost is negligible.
                    var vec_result: SIMD[dtype, simd_width] = 0

                    comptime for lane in range(simd_width):
                        var linear_idx = i + lane
                        var rem = linear_idx
                        var a_idx = 0
                        var b_idx = 0

                        for dim in range(rank - 1, -1, -1):
                            var coord = rem % result_shape[dim]
                            a_idx += coord * A_strides[dim]
                            b_idx += coord * B_strides[dim]
                            rem //= result_shape[dim]

                        vec_result[lane] = scalar_op[op_code, dtype](
                            A[a_idx], B[b_idx], epsilon
                        )

                    result.store[width=simd_width](i, vec_result)

            else:
                # ── Tail: fewer than simd_width elements remain ───────────
                for j in range(size - i):
                    var linear_idx = i + j
                    var rem = linear_idx
                    var a_idx = 0
                    var b_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = rem % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        b_idx += coord * B_strides[dim]
                        rem //= result_shape[dim]

                    result[linear_idx] = scalar_op[op_code, dtype](
                        A[a_idx], B[b_idx], epsilon
                    )

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL 3 — A contiguous and fills broadcast shape; B strided/broadcast.
#
# Preconditions (enforced by launcher):
#   A.is_contiguous() and A_shape == broadcast_shape
#
# Strategy:
#   A fills the broadcast shape contiguously → its linear index equals the
#   result linear index. A.load[width=simd_width](i) is always valid for a
#   full vector (no row-boundary concern for A).
#
#   B is accessed via B_broadcast_strides. For a SIMD vector starting at i,
#   B elements are consecutive when the vector stays within one row
#   (B_strides[rank-1]==1) or are a single broadcast value (B_strides[rank-1]==0).
#
# PERFORMANCE FIX vs original:
#   Original did per-lane scalar coordinate decomposition and scalar B loads
#   inside a `comptime for lane` loop — simd_width decomps and scalar loads
#   per vector. New version does ONE decomp per vector for the outer base,
#   then a single SIMD load from B when safe.
#
# CORRECTNESS: uses the same i % inner_dim / i // inner_dim split as kernel 2
#   to avoid double-counting the innermost coordinate in b_base.
#   A has no base computation — loaded at i directly — so no bug possible there.
#
# B inner stride cases:
#   B_strides[rank-1] == 1  → B consecutive in inner dim, SIMD load
#   B_strides[rank-1] == 0  → B broadcasts inner dim, scalar splat
# =============================================================================
def arithmetic_ops_A_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: Array,
    B_strides: Array,
    size: Int,
    rank: Int,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
):
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var grid_stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    var inner_dim = result_shape[rank - 1]

    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                # A fills broadcast shape contiguously — always load at i.
                # No base computation needed, no row-boundary concern for A.
                var vec_a = A.load[width=simd_width](i)

                # ── Compute B outer base ───────────────────────────────────
                # Strip innermost coord first to avoid double-counting.
                var inner_offset = i % inner_dim
                var outer_remaining = i // inner_dim

                var b_base = 0
                for dim in range(rank - 2, -1, -1):
                    var coord = outer_remaining % result_shape[dim]
                    b_base += coord * B_strides[dim]
                    outer_remaining //= result_shape[dim]

                var vec_result: SIMD[dtype, simd_width]

                # ── Fast path: vector within one row ──────────────────────
                if inner_offset + simd_width <= inner_dim:
                    if B_strides[rank - 1] == 1:
                        # B has real inner dim — consecutive SIMD load
                        var vec_b = B.load[width=simd_width](
                            b_base + inner_offset
                        )
                        vec_result = simd_op[op_code, dtype, simd_width](
                            vec_a, vec_b, epsilon
                        )
                    else:
                        # B broadcasts inner dim — scalar splat
                        var vec_b = SIMD[dtype, simd_width](B[b_base])
                        vec_result = simd_op[op_code, dtype, simd_width](
                            vec_a, vec_b, epsilon
                        )
                else:
                    # ── Slow path: B crosses row boundary ─────────────────
                    # A elements are still consecutive (A fills broadcast shape)
                    # so vec_a[lane] is correct. B needs per-lane decomposition.
                    vec_result = SIMD[dtype, simd_width](0)

                    comptime for lane in range(simd_width):
                        var linear_idx = i + lane
                        var rem = linear_idx
                        var b_idx = 0

                        for dim in range(rank - 1, -1, -1):
                            var coord = rem % result_shape[dim]
                            b_idx += coord * B_strides[dim]
                            rem //= result_shape[dim]

                        vec_result[lane] = scalar_op[op_code, dtype](
                            vec_a[lane], B[b_idx], epsilon
                        )

                result.store[width=simd_width](i, vec_result)

            else:
                # ── Tail ──────────────────────────────────────────────────
                for j in range(size - i):
                    var linear_idx = i + j
                    var rem = linear_idx
                    var b_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = rem % result_shape[dim]
                        b_idx += coord * B_strides[dim]
                        rem //= result_shape[dim]

                    result[linear_idx] = scalar_op[op_code, dtype](
                        A[linear_idx], B[b_idx], epsilon
                    )

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL 3B — A contiguous and fills broadcast shape; B is last-dim-contiguous
#              (bias_add pattern).
#
# Preconditions (enforced by launcher):
#   A.is_contiguous() and A_shape == broadcast_shape
#   B_broadcast_strides == [0, ..., 0, 1]  (B contiguous only in last dim)
#
# Strategy: A is read linearly (index == linear result index).
#           B is accessed via b_idx = linear_idx % last_dim — a single modulo
#           per SIMD group start, then increment with wrap per lane.
#           Avoids the expensive for-dim coordinate decomposition loop.
#           Critical for small last_dim (< simd_width) where every SIMD group
#           crosses a row boundary and the main kernel's slow path would run.
# =============================================================================
def arithmetic_ops_A_contiguous_lastdim_contiguous_B[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    last_dim: Int,
    size: Int,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
):
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var grid_stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width] = 0

                var b_idx = i % last_dim

                comptime for lane in range(simd_width):
                    vec_result[lane] = scalar_op[op_code, dtype](
                        vec_a[lane], B[b_idx], epsilon
                    )

                    b_idx += 1
                    if b_idx >= last_dim:
                        b_idx = 0

                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    result[linear_idx] = scalar_op[op_code, dtype](
                        A[linear_idx], B[linear_idx % last_dim], epsilon
                    )

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL 4 — B contiguous and fills broadcast shape; A strided/broadcast.
#
# Preconditions (enforced by launcher):
#   B.is_contiguous() and B_shape == broadcast_shape
#
# Mirror of Kernel 3 with A and B roles swapped.
#
# B fills the broadcast shape contiguously → B.load[width=simd_width](i) always valid.
# A is accessed via A_broadcast_strides with the same outer-base fast path.
#
# PERFORMANCE FIX: same as kernel 3 — one decomp per vector, SIMD load from A
#   when the vector stays within one row and A_strides[rank-1]==1.
#
# CORRECTNESS: i % inner_dim / i // inner_dim split for a_base to prevent
#   double-counting. B has no base computation — no bug possible there.
# =============================================================================
def arithmetic_ops_B_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: Array,
    A_strides: Array,
    size: Int,
    rank: Int,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
):
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var grid_stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    var inner_dim = result_shape[rank - 1]

    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                # B fills broadcast shape contiguously — always load at i.
                var vec_b = B.load[width=simd_width](i)

                # ── Compute A outer base ───────────────────────────────────
                var inner_offset = i % inner_dim
                var outer_remaining = i // inner_dim

                var a_base = 0
                for dim in range(rank - 2, -1, -1):
                    var coord = outer_remaining % result_shape[dim]
                    a_base += coord * A_strides[dim]
                    outer_remaining //= result_shape[dim]

                var vec_result: SIMD[dtype, simd_width]

                # ── Fast path: vector within one row ──────────────────────
                if inner_offset + simd_width <= inner_dim:
                    if A_strides[rank - 1] == 1:
                        # A has real inner dim — consecutive SIMD load
                        var vec_a = A.load[width=simd_width](
                            a_base + inner_offset
                        )
                        vec_result = simd_op[op_code, dtype, simd_width](
                            vec_a, vec_b, epsilon
                        )
                    else:
                        # A broadcasts inner dim — scalar splat
                        var vec_a = SIMD[dtype, simd_width](A[a_base])
                        vec_result = simd_op[op_code, dtype, simd_width](
                            vec_a, vec_b, epsilon
                        )
                else:
                    # ── Slow path: A crosses row boundary ─────────────────
                    vec_result = SIMD[dtype, simd_width](0)

                    comptime for lane in range(simd_width):
                        var linear_idx = i + lane
                        var rem = linear_idx
                        var a_idx = 0

                        for dim in range(rank - 1, -1, -1):
                            var coord = rem % result_shape[dim]
                            a_idx += coord * A_strides[dim]
                            rem //= result_shape[dim]

                        vec_result[lane] = scalar_op[op_code, dtype](
                            A[a_idx], vec_b[lane], epsilon
                        )

                result.store[width=simd_width](i, vec_result)

            else:
                # ── Tail ──────────────────────────────────────────────────
                for j in range(size - i):
                    var linear_idx = i + j
                    var rem = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = rem % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        rem //= result_shape[dim]

                    result[linear_idx] = scalar_op[op_code, dtype](
                        A[a_idx], B[linear_idx], epsilon
                    )

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL 5 — General fallback: both tensors non-contiguous and/or strided.
#
# Unchanged from original — no fast path attempted, no base+inner_offset bug.
# Full per-lane coordinate decomposition for both A and B. Correct as-is.
# =============================================================================
def arithmetic_ops_both_strided[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: Array,
    A_strides: Array,
    B_strides: Array,
    size: Int,
    rank: Int,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
):
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var grid_stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_result: SIMD[dtype, simd_width] = 0

                comptime for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx
                    var a_idx = 0
                    var b_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    vec_result[lane] = scalar_op[op_code, dtype](
                        A[a_idx], B[b_idx], epsilon
                    )

                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var a_idx = 0
                    var b_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    result[linear_idx] = scalar_op[op_code, dtype](
                        A[a_idx], B[b_idx], epsilon
                    )

        base_idx += grid_stride * CHUNK_SIZE


@fieldwise_init
struct BinaryOperations[dtype: DType = DType.float32](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def launch[
        op_code: Int,
    ](
        A: NDBuffer[Self.dtype],
        B: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
        sync: Bool = False,
    ) raises -> NDBuffer[Self.dtype]:
        comptime simdwidth = simd_width_of[Self.dtype]()
        var A_shape = A.shape
        var B_shape = B.shape

        var broadcast_shape = ShapeBroadcaster.broadcast_shape(A_shape, B_shape)
        var output_size = broadcast_shape.product()
        var rank = broadcast_shape.rank()

        var (num_blocks, threads_per_block) = Self.launch_config(output_size)

        ref A_device_state = A.device_state.value()
        ref B_device_state = B.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            output_size
        )

        ref A_buffer = A_device_state.device_buffer()
        ref B_buffer = B_device_state.device_buffer()

        var A_is_contiguous = A.is_contiguous()
        var B_is_contiguous = B.is_contiguous()

        # ================================================================
        # PATH 1: Both contiguous, same shape — pure linear indexing.
        #
        # No broadcasting, no strides needed. Fastest path.
        # Example: [256, 512] + [256, 512]
        # ================================================================
        if A_shape == B_shape and A_is_contiguous and B_is_contiguous:
            print("[DISPATCH] PATH 1: both_contiguous same_shape | A=", String(A_shape), " B=", String(B_shape))
            var compiled_func = device_context.compile_function[
                arithmetic_ops_both_contiguous[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
                arithmetic_ops_both_contiguous[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
            ]()
            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                B_buffer,
                output_size,
                epsilon,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
            if sync:
                device_context.synchronize()
            var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
            return NDBuffer[Self.dtype].with_device_state(
                device_state^, broadcast_shape
            )

        # ----------------------------------------------------------------
        # Broadcast strides for the remaining mixed and fully-strided paths.
        #
        # FIX: Use A.strides / B.strides (actual strides), NOT Strides.default().
        # Strides.default() assumes contiguous row-major layout. Non-contiguous
        # tensors (transposed, sliced) have different strides; using the wrong
        # ones silently reads incorrect memory locations.
        # ----------------------------------------------------------------
        var A_broadcast_strides = ShapeBroadcaster.broadcast_strides(
            A_shape, A.strides, broadcast_shape
        )
        var B_broadcast_strides = ShapeBroadcaster.broadcast_strides(
            B_shape, B.strides, broadcast_shape
        )

        # ================================================================
        # PATH 3: A contiguous and fills broadcast shape; B strided/broadcast.
        # ================================================================
        if A_is_contiguous and A_shape == broadcast_shape:
            # Check for optimized bias_add path:
            # B_broadcast_strides == [0, ..., 0, 1] — B contiguous in last dim only.
            # Replace coordinate decomposition per lane with single modulo per group.
            var is_lastdim_b = B_broadcast_strides[rank - 1] == 1
            if is_lastdim_b and rank >= 2:
                for dim in range(rank - 1):
                    if B_broadcast_strides[dim] != 0:
                        is_lastdim_b = False
                        break
            if is_lastdim_b:
                print("[DISPATCH] PATH 3B: A_contiguous lastdim_B | A=", String(A_shape), " B=", String(B_shape), " broadcast=", String(broadcast_shape))
                var compiled_func = device_context.compile_function[
                    arithmetic_ops_A_contiguous_lastdim_contiguous_B[
                        op_code, Self.dtype, simdwidth, 2 * simdwidth
                    ],
                    arithmetic_ops_A_contiguous_lastdim_contiguous_B[
                        op_code, Self.dtype, simdwidth, 2 * simdwidth
                    ],
                ]()
                device_context.enqueue_function(
                    compiled_func,
                    result_buffer,
                    A_buffer,
                    B_buffer,
                    broadcast_shape[rank - 1],
                    output_size,
                    epsilon,
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            else:
                print("[DISPATCH] PATH 3: A_contiguous fills shape | A=", String(A_shape), " B=", String(B_shape), " broadcast=", String(broadcast_shape))
                var compiled_func = device_context.compile_function[
                    arithmetic_ops_A_contiguous[
                        op_code, Self.dtype, simdwidth, 2 * simdwidth
                    ],
                    arithmetic_ops_A_contiguous[
                        op_code, Self.dtype, simdwidth, 2 * simdwidth
                    ],
                ]()
                device_context.enqueue_function(
                    compiled_func,
                    result_buffer,
                    A_buffer,
                    B_buffer,
                    broadcast_shape.array(),
                    B_broadcast_strides.array(),
                    output_size,
                    rank,
                    epsilon,
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            if sync:
                device_context.synchronize()
            var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
            return NDBuffer[Self.dtype].with_device_state(
                device_state^, broadcast_shape
            )

        # ================================================================
        # PATH 4: B contiguous and fills broadcast shape; A strided/broadcast.
        # ================================================================
        if B_is_contiguous and B_shape == broadcast_shape:
            print("[DISPATCH] PATH 4: B_contiguous fills shape | A=", String(A_shape), " B=", String(B_shape), " broadcast=", String(broadcast_shape))
            var compiled_func = device_context.compile_function[
                arithmetic_ops_B_contiguous[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
                arithmetic_ops_B_contiguous[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
            ]()
            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                B_buffer,
                broadcast_shape.array(),
                A_broadcast_strides.array(),
                output_size,
                rank,
                epsilon,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
            if sync:
                device_context.synchronize()
            var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
            return NDBuffer[Self.dtype].with_device_state(
                device_state^, broadcast_shape
            )

        # ================================================================
        # PATH 2: Both contiguous, shapes differ — broadcast expansion.
        # ================================================================
        if A_is_contiguous and B_is_contiguous:
            print("[DISPATCH] PATH 2: both_contiguous broadcast | A=", String(A_shape), " B=", String(B_shape), " broadcast=", String(broadcast_shape))
            var compiled_func = device_context.compile_function[
                arithmetic_ops_both_contiguous_broadcast[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
                arithmetic_ops_both_contiguous_broadcast[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
            ]()
            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                B_buffer,
                broadcast_shape.array(),
                A_broadcast_strides.array(),
                B_broadcast_strides.array(),
                output_size,
                rank,
                epsilon,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
            if sync:
                device_context.synchronize()
            var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
            return NDBuffer[Self.dtype].with_device_state(
                device_state^, broadcast_shape
            )

        # ================================================================
        # PATH 5: General fallback — both strided or complex broadcast.
        # ================================================================
        print("[DISPATCH] PATH 5: both_strided fallback | A=", String(A_shape), " B=", String(B_shape), " broadcast=", String(broadcast_shape), " A_contig=", A_is_contiguous, " B_contig=", B_is_contiguous)
        var compiled_func = device_context.compile_function[
            arithmetic_ops_both_strided[
                op_code, Self.dtype, simdwidth, 2 * simdwidth
            ],
            arithmetic_ops_both_strided[
                op_code, Self.dtype, simdwidth, 2 * simdwidth
            ],
        ]()
        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            B_buffer,
            broadcast_shape.array(),
            A_broadcast_strides.array(),
            B_broadcast_strides.array(),
            output_size,
            rank,
            epsilon,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        if sync:
            device_context.synchronize()
        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(
            device_state^, broadcast_shape
        )

    @staticmethod
    def launch_config(output_size: Int) -> Tuple[Int, Int]:
        # FIX: The original computed num_blocks = ceil(output_size / threads_per_block),
        # completely ignoring CHUNK_SIZE. Each thread processes CHUNK_SIZE =
        # simd_vectors_per_thread * simd_width elements (default: 2*sw*sw).
        # For fp32 with simd_width=8: CHUNK_SIZE = 128. The original over-launched
        # blocks by 128×, with all surplus blocks immediately exiting the while loop.
        # Not a correctness bug but a serious waste of GPU occupancy and launch overhead
        # for small/medium tensors.
        #
        # We use a conservative APPROX_CHUNK_SIZE = 32 (= 2 * 16, the maximum
        # simd_width for fp16/bf16). For fp32 (simd_width=8, CHUNK_SIZE=128) this
        # still slightly over-launches, but safely so — never under-subscribes.
        # If exact launch sizing is needed, pass CHUNK_SIZE as a parameter.
        comptime APPROX_CHUNK_SIZE = 32

        var threads_per_block: Int
        var num_blocks: Int

        if output_size < 4096:
            threads_per_block = 128
            num_blocks = max(
                1,
                (output_size + threads_per_block * APPROX_CHUNK_SIZE - 1)
                // (threads_per_block * APPROX_CHUNK_SIZE),
            )
        elif output_size < 65536:
            threads_per_block = 256
            num_blocks = min(
                (output_size + threads_per_block * APPROX_CHUNK_SIZE - 1)
                // (threads_per_block * APPROX_CHUNK_SIZE),
                128,
            )
        else:
            threads_per_block = 512
            num_blocks = min(
                (output_size + threads_per_block * APPROX_CHUNK_SIZE - 1)
                // (threads_per_block * APPROX_CHUNK_SIZE),
                512,
            )

        return num_blocks, threads_per_block

