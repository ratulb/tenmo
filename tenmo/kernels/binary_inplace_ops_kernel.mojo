from std.sys import simd_width_of
from std.gpu import thread_idx, block_idx, block_dim, grid_dim
from tenmo.mnemonics import Add, Multiply, Subtract, Divide
from tenmo.strides import Strides
from tenmo.broadcasthelper import ShapeBroadcaster
from tenmo.device import DeviceState
from tenmo.array import Array
from tenmo.ndbuffer import NDBuffer
from std.math import rsqrt
from tenmo.shared.scalar_ops import simd_op, scalar_op
from tenmo.common_utils import Epsilon
from . import elementwise_launch_config


# =============================================================================
# KERNEL for PATH 1 — Both contiguous, same shape, no broadcasting.
#
# Pure linear indexing. Already optimal — no changes needed.
# A read, op applied, result written back to A at the same linear index.
# =============================================================================
def arithmetic_ops_both_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
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
                    vec_a, vec_b, Epsilon[dtype].value()
                )
                A.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var idx = i + j
                    var a = A[idx]
                    var b = B[idx]
                    var res = scalar_op[op_code, dtype](
                        a, b, Epsilon[dtype].value()
                    )
                    A[idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL for PATH 2 — A contiguous (fills broadcast_shape); B strided/broadcast.
#
# OPTIMIZATION vs original scalar-per-lane version:
#   A fills broadcast_shape contiguously → A.load[width=simd_width](i) always
#   valid. A.store[width=simd_width](i, vec_result) always valid. Full SIMD
#   on both A read and write-back.
#
#   B outer base computed ONCE per vector using i // inner_dim / i % inner_dim
#   split (prevents double-counting the innermost coordinate). Per vector:
#     - B_strides[rank-1] == 1 → single SIMD load from B
#     - B_strides[rank-1] == 0 → scalar splat (B broadcasts inner dim)
#   Slow path (vector crosses row boundary): per-lane B decomposition, but
#   A elements still loaded from vec_a[lane] — no extra A decomposition.
#
# Index math saving vs original:
#   Original: rank modulos + rank divides per lane × simd_width lanes = rank×sw ops
#   New fast path: 1 outer decomp (rank-1 ops) + 0 per-lane inner ops = rank-1 ops
#   For bias_add (2,4,6)+=(6,), simd_width=8: 16 ops → 2 ops per vector.
# =============================================================================
def arithmetic_ops_A_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: Array,
    B_strides: Array,
    size: Int,
    rank: Int,
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
                # A fills broadcast_shape contiguously — always valid SIMD load.
                var vec_a = A.load[width=simd_width](i)

                # Compute B outer base once for this vector.
                # Strip innermost coord first to avoid double-counting.
                var inner_offset = i % inner_dim
                var outer_remaining = i // inner_dim
                var b_base = 0

                for dim in range(rank - 2, -1, -1):
                    var coord = outer_remaining % result_shape[dim]
                    b_base += coord * B_strides[dim]
                    outer_remaining //= result_shape[dim]

                var vec_result: SIMD[dtype, simd_width]

                if inner_offset + simd_width <= inner_dim:
                    # ── Fast path: vector fits within one row ─────────────
                    if B_strides[rank - 1] == 1:
                        # B has real inner dim → consecutive SIMD load.
                        var vec_b = B.load[width=simd_width](
                            b_base + inner_offset
                        )

                        vec_result = simd_op[op_code, dtype, simd_width](
                            vec_a, vec_b, Epsilon[dtype].value()
                        )
                    else:
                        # B broadcasts inner dim → scalar splat.
                        # b_base already points to the correct outer element;
                        # inner coord contributes inner_offset * 0 = 0.
                        var vec_b = SIMD[dtype, simd_width](B[b_base])
                        vec_result = simd_op[op_code, dtype, simd_width](
                            vec_a, vec_b, Epsilon[dtype].value()
                        )
                else:
                    # ── Slow path: B crosses row boundary ─────────────────
                    # A elements still come from vec_a[lane] — no extra decomp.
                    # B needs per-lane full decomposition.
                    vec_result = SIMD[dtype, simd_width](0)

                    comptime for lane in range(simd_width):
                        var linear_idx = i + lane
                        var rem = linear_idx
                        var b_idx = 0

                        for dim in range(rank - 1, -1, -1):
                            var coord = rem % result_shape[dim]
                            b_idx += coord * B_strides[dim]
                            rem //= result_shape[dim]

                        var a = vec_a[lane]
                        var b = B[b_idx]
                        vec_result[lane] = scalar_op[op_code, dtype](
                            a, b, Epsilon[dtype].value()
                        )

                # A is contiguous → SIMD store always valid.
                A.store[width=simd_width](i, vec_result)

            else:
                # ── Tail: fewer than simd_width elements remain ───────────
                for j in range(size - i):
                    var linear_idx = i + j
                    var rem = linear_idx
                    var b_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = rem % result_shape[dim]
                        b_idx += coord * B_strides[dim]
                        rem //= result_shape[dim]
                    var a = A[linear_idx]
                    var b = B[b_idx]
                    var res = scalar_op[op_code, dtype](
                        a, b, Epsilon[dtype].value()
                    )
                    A[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL for PATH 3 — A strided; B contiguous, no broadcasting.
#
# OPTIMIZATION vs original scalar-per-lane version:
#   B is contiguous and same shape as A → B.load[width=simd_width](i) is
#   already a single SIMD load — unchanged from original.
#
#   A outer base computed ONCE per vector (rank-1 modulo/divide ops total).
#   Per lane, only ONE multiply is needed for the innermost dimension:
#     a_idx = a_base + (inner_offset + lane) * A_strides[rank-1]
#   vs the original which did a full rank-level decomposition per lane
#   (rank modulos + rank divides × simd_width lanes).
#
#   A write-back remains per-lane scalar (A is strided — consecutive logical
#   elements are not consecutive in physical memory, so SIMD store is unsafe).
#
#   Fast path (vector within one row, A_strides[rank-1]==1):
#     A read: SIMD load at a_base + inner_offset (consecutive in memory).
#     A write: still per-lane at a_base + lane (stride==1 means consecutive,
#              so we CAN do SIMD store here too).
#
#   Fast path (vector within one row, A_strides[rank-1] != 1):
#     A read: per-lane scalar at a_base + (inner_offset+lane)*A_strides[rank-1].
#     A write: per-lane scalar at same address.
#     Saving: outer decomp done once, only 1 multiply per lane for inner dim.
#
#   Slow path (crosses row boundary):
#     Full per-lane decomposition as before — correctness over performance.
#
# Index math saving vs original (rank=3, simd_width=8):
#   Original: 3 mod + 3 div per lane × 8 lanes = 48 ops per vector
#   New fast path: 2 mod + 2 div once + 1 mul per lane × 8 = 4 + 8 = 12 ops
# =============================================================================
def arithmetic_ops_B_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: Array,
    A_strides: Array,
    size: Int,
    rank: Int,
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
                # B is contiguous, same shape → always valid SIMD load.
                var vec_b = B.load[width=simd_width](i)

                # Compute A outer base once for this vector.
                # Strip innermost coord first to avoid double-counting.
                var inner_offset = i % inner_dim
                var outer_remaining = i // inner_dim
                var a_base = 0
                var a_inner_stride = A_strides[rank - 1]

                for dim in range(rank - 2, -1, -1):
                    var coord = outer_remaining % result_shape[dim]
                    a_base += coord * A_strides[dim]
                    outer_remaining //= result_shape[dim]

                if inner_offset + simd_width <= inner_dim:
                    # ── Fast path: vector fits within one row ─────────────
                    if a_inner_stride == 1:
                        # A elements are consecutive in memory →
                        # SIMD load AND SIMD store both safe.
                        var vec_a = A.load[width=simd_width](
                            a_base + inner_offset
                        )
                        var vec_result = simd_op[op_code, dtype, simd_width](
                            vec_a, vec_b, Epsilon[dtype].value()
                        )
                        # A elements are stride-1 consecutive → SIMD store safe.
                        A.store[width=simd_width](
                            a_base + inner_offset, vec_result
                        )

                    else:
                        # A elements are strided (a_inner_stride != 1).
                        # Per-lane read and write, but outer base computed once.
                        # Each lane: a_idx = a_base + (inner_offset + lane) * a_inner_stride
                        comptime for lane in range(simd_width):
                            var a_idx = (
                                a_base + (inner_offset + lane) * a_inner_stride
                            )
                            var a = A[a_idx]
                            var b = vec_b[lane]
                            var res = scalar_op[op_code, dtype](
                                a, b, Epsilon[dtype].value()
                            )
                            A[a_idx] = res

                else:
                    # ── Slow path: crosses row boundary ───────────────────
                    # Full per-lane decomposition for A.
                    # vec_b[lane] still valid — B is contiguous.
                    comptime for lane in range(simd_width):
                        var linear_idx = i + lane
                        var rem = linear_idx
                        var a_idx = 0

                        for dim in range(rank - 1, -1, -1):
                            var coord = rem % result_shape[dim]
                            a_idx += coord * A_strides[dim]
                            rem //= result_shape[dim]

                        var a = A[a_idx]
                        var b = vec_b[lane]
                        var res = scalar_op[op_code, dtype](
                            a, b, Epsilon[dtype].value()
                        )
                        A[a_idx] = res

            else:
                # ── Tail: fewer than simd_width elements remain ───────────
                for j in range(size - i):
                    var linear_idx = i + j
                    var rem = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = rem % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        rem //= result_shape[dim]

                    var a = A[a_idx]
                    var b = B[linear_idx]
                    var res = scalar_op[op_code, dtype](
                        a, b, Epsilon[dtype].value()
                    )

                    A[a_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL for PATH 4 — Both strided / A strided + B needs broadcasting.
#
# OPTIMIZATION vs original scalar-per-lane version:
#   Both A and B outer bases computed ONCE per vector.
#   Per lane, only ONE multiply per operand for the innermost dimension:
#     a_idx = a_base + (inner_offset + lane) * A_strides[rank-1]
#     b_idx = b_base + (inner_offset + lane) * B_strides[rank-1]
#   vs original: full rank-level decomposition per lane for both.
#
#   A write-back always per-lane scalar (A is strided — cannot SIMD store).
#
#   No fast/slow path split needed here: since A is always strided in PATH 4,
#   there is never a safe SIMD store regardless of row boundary. The per-lane
#   write is always required. We keep the structure simple.
#
#   Slow path (crosses row boundary): full per-lane decomposition as before.
#
# Index math saving vs original (rank=3, simd_width=8):
#   Original: (3 mod + 3 div) × 2 operands × 8 lanes = 96 ops per vector
#   New:      (2 mod + 2 div) × 2 operands once
#             + (1 mul × 2 operands) × 8 lanes = 8 + 16 = 24 ops per vector
# =============================================================================
def arithmetic_ops_both_strided[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: Array,
    A_strides: Array,
    B_strides: Array,
    size: Int,
    rank: Int,
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
                # Compute outer bases for both A and B once per vector.
                # Strip innermost coord first — prevents double-counting.
                var inner_offset = i % inner_dim
                var outer_remaining = i // inner_dim
                var a_base = 0
                var b_base = 0
                var a_inner_stride = A_strides[rank - 1]
                var b_inner_stride = B_strides[rank - 1]

                for dim in range(rank - 2, -1, -1):
                    var coord = outer_remaining % result_shape[dim]
                    a_base += coord * A_strides[dim]
                    b_base += coord * B_strides[dim]
                    outer_remaining //= result_shape[dim]

                if inner_offset + simd_width <= inner_dim:
                    # ── Fast path: vector fits within one row ─────────────
                    # Per lane: one multiply each for innermost dim.
                    # A write-back always per-lane (A is strided).
                    comptime for lane in range(simd_width):
                        var a_idx = (
                            a_base + (inner_offset + lane) * a_inner_stride
                        )
                        var b_idx = (
                            b_base + (inner_offset + lane) * b_inner_stride
                        )
                        var a = A[a_idx]
                        var b = B[b_idx]
                        var res = scalar_op[op_code, dtype](
                            a, b, Epsilon[dtype].value()
                        )

                        A[a_idx] = res

                else:
                    # ── Slow path: crosses row boundary ───────────────────
                    # Full per-lane decomposition for both A and B.
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

                        var a = A[a_idx]
                        var b = B[b_idx]
                        var res = scalar_op[op_code, dtype](
                            a, b, Epsilon[dtype].value()
                        )

                        A[a_idx] = res

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

                    var a = A[a_idx]
                    var b = B[b_idx]
                    var res = scalar_op[op_code, dtype](
                        a, b, Epsilon[dtype].value()
                    )

                    A[a_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL for bias_add pattern — A contiguous, B contiguous only in last dim.
#
# Preconditions (enforced by launcher):
#   A.is_contiguous() and A_shape == broadcast_shape
#   B_broadcast_strides == [0, ..., 0, 1]  (B contiguous only in last dim)
#
# B is accessed via b_idx = linear_idx % last_dim — a single modulo
# per SIMD group start, then increment with wrap per lane.
# Avoids the dimension loop in the general A_contiguous kernel.
# Critical for small last_dim (< simd_width) where every SIMD group
# crosses a row boundary.
# =============================================================================
def arithmetic_ops_A_contiguous_lastdim_contiguous_B[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    last_dim: Int,
    size: Int,
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
                        vec_a[lane], B[b_idx], Epsilon[dtype].value()
                    )

                    b_idx += 1
                    if b_idx >= last_dim:
                        b_idx = 0

                A.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    A[linear_idx] = scalar_op[op_code, dtype](
                        A[linear_idx],
                        B[linear_idx % last_dim],
                        Epsilon[dtype].value(),
                    )

        base_idx += grid_stride * CHUNK_SIZE


@fieldwise_init
struct BinaryInplaceOperations[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def launch[
        op_code: Int,
    ](
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype], sync: Bool = False
    ) raises:
        comptime simdwidth = simd_width_of[Self.dtype]()
        var A_shape = A.shape
        var B_shape = B.shape

        # For A op= B, broadcasting is valid only when B's shape
        # is broadcastable TO A's shape. The broadcast result must
        # equal A's shape; A's shape must not grow. Validate here
        # so that the kernels below can assume broadcast_shape ==
        # A_shape without further checking.
        var broadcast_shape = ShapeBroadcaster.broadcast_shape(A_shape, B_shape)
        if broadcast_shape != A_shape:
            raise Error(
                "In-place op A op= B requires B to be broadcastable to A's "
                "shape, but the broadcast result differs from A's shape. "
                "A.shape="
                + String(A_shape)
                + " B.shape="
                + String(B_shape)
            )

        var output_size = broadcast_shape.product()
        var rank = broadcast_shape.rank()

        # needs_broadcasting is True when B's shape differs from
        # the output (A) shape, meaning B has stride-0 broadcast
        # axes and its physical buffer is smaller than output_size.
        var needs_broadcasting = B_shape != broadcast_shape

        var (num_blocks, threads_per_block) = Self.launch_config(
            output_size, simdwidth
        )

        ref A_device_state = A.device_state.value()
        ref B_device_state = B.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        ref A_buffer = A_device_state.device_buffer()
        ref B_buffer = B_device_state.device_buffer()

        var A_is_contiguous = A.is_contiguous()
        var B_is_contiguous = B.is_contiguous()

        # ================================================================
        # PATH 1: Both contiguous, same shape, no broadcasting.
        #
        # Fastest path: purely linear indexing, full SIMD on both A and B.
        # A_shape == B_shape is guaranteed here because needs_broadcasting
        # is False and broadcast_shape == A_shape by contract.
        # ================================================================
        if A_is_contiguous and B_is_contiguous and not needs_broadcasting:
            var compiled_func = device_context.compile_function[
                arithmetic_ops_both_contiguous[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
            ]()
            device_context.enqueue_function(
                compiled_func,
                A_buffer,
                B_buffer,
                output_size,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
            if sync:
                device_context.synchronize()
            return

        # Broadcast strides needed for paths 2–4.
        #
        # A.strides and B.strides are the
        # actual physical strides of each tensor. For contiguous tensors
        # these equal Strides.default(shape); for non-contiguous views
        # (transposed, sliced) they differ. Using actual strides is the
        # only correct choice for the general case.
        #
        # A_broadcast_strides will have no stride-0 axes since
        # broadcast_shape == A_shape (A is never broadcast-expanded
        # in-place). B_broadcast_strides may have stride-0 axes where
        # B is smaller than A and needs replication.
        var A_broadcast_strides = ShapeBroadcaster.broadcast_strides(
            A_shape, A.strides, broadcast_shape
        )
        var B_broadcast_strides = ShapeBroadcaster.broadcast_strides(
            B_shape, B.strides, broadcast_shape
        )

        # ================================================================
        # PATH 2: A contiguous; B strided or broadcast-expanded (or both).
        #
        # A is read and written at the linear index i — safe because
        # A_shape == broadcast_shape and A is contiguous.
        # B is always stride-decomposed via B_broadcast_strides.
        #
        # This also handles the sub-case: A contiguous, B contiguous,
        # needs_broadcasting=True. Even though B is a flat array, we
        # still need stride decomposition to implement the broadcast
        # (stride-0 axes tell us which physical B element to repeat).
        # A separate "both_contiguous_broadcast" kernel is NOT needed
        # here: A is always full-size and linearly addressable when
        # contiguous, and B's stride-0 broadcast always requires decomp.
        # ================================================================
        if A_is_contiguous and (not B_is_contiguous or needs_broadcasting):
            # Check for optimized bias_add path:
            # B_broadcast_strides == [0, ..., 0, 1] — B contiguous in last dim only.
            var is_lastdim_b = B_broadcast_strides[rank - 1] == 1
            if is_lastdim_b and rank >= 2:
                for dim in range(rank - 1):
                    if B_broadcast_strides[dim] != 0:
                        is_lastdim_b = False
                        break
            if is_lastdim_b:
                var compiled_func = device_context.compile_function[
                    arithmetic_ops_A_contiguous_lastdim_contiguous_B[
                        op_code, Self.dtype, simdwidth, 2 * simdwidth
                    ],
                ]()
                device_context.enqueue_function(
                    compiled_func,
                    A_buffer,
                    B_buffer,
                    broadcast_shape[rank - 1],
                    output_size,
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            else:
                var compiled_func = device_context.compile_function[
                    arithmetic_ops_A_contiguous[
                        op_code, Self.dtype, simdwidth, 2 * simdwidth
                    ],
                ]()
                device_context.enqueue_function(
                    compiled_func,
                    A_buffer,
                    B_buffer,
                    broadcast_shape.array(),
                    B_broadcast_strides.array(),
                    output_size,
                    rank,
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            if sync:
                device_context.synchronize()
            return

        # ================================================================
        # PATH 3: A strided, B contiguous, NO broadcasting.
        #
        # B's physical buffer is exactly output_size elements (because
        # needs_broadcasting is False and B_shape == A_shape). Linear
        # reads of B at index i are safe and in bounds.
        #
        # A is stride-decomposed to find a_idx for both reading and
        # writing. B is loaded directly at linear index i.
        #
        # CRITICAL: `not needs_broadcasting` is mandatory. If B needed
        # broadcasting, B's physical allocation would be smaller than
        # output_size, and B.load[width=simd_width](i) would be OOB.
        # Those cases (A strided, B contiguous, needs broadcasting)
        # fall through to PATH 4, which stride-decomposes B safely.
        # ================================================================
        if not A_is_contiguous and B_is_contiguous and not needs_broadcasting:
            var compiled_func = device_context.compile_function[
                arithmetic_ops_B_contiguous[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
            ]()
            device_context.enqueue_function(
                compiled_func,
                A_buffer,
                B_buffer,
                broadcast_shape.array(),
                A_broadcast_strides.array(),
                output_size,
                rank,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
            if sync:
                device_context.synchronize()
            return

        # ================================================================
        # PATH 4: Universal fallback — both operands stride-decomposed.
        #
        # Covers all remaining cases:
        #   • A strided, B strided, no broadcasting
        #   • A strided, B contiguous, needs broadcasting
        #     (B buffer < output_size — linear read would be OOB)
        #   • A strided, B strided, needs broadcasting
        #
        # A_broadcast_strides has no stride-0 axes (A_shape == broadcast_shape).
        # B_broadcast_strides has stride-0 on broadcast axes.
        # Result is always written at a_idx (A's physical address).
        # ================================================================
        var compiled_func = device_context.compile_function[
            arithmetic_ops_both_strided[
                op_code, Self.dtype, simdwidth, 2 * simdwidth
            ],
        ]()
        device_context.enqueue_function(
            compiled_func,
            A_buffer,
            B_buffer,
            broadcast_shape.array(),
            A_broadcast_strides.array(),
            B_broadcast_strides.array(),
            output_size,
            rank,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        if sync:
            device_context.synchronize()

    @staticmethod
    def launch_config(output_size: Int, simdwidth: Int) -> Tuple[Int, Int]:
        return elementwise_launch_config(output_size, simdwidth)
