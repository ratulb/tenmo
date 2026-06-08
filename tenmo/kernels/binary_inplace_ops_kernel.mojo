from std.sys import simd_width_of
from std.gpu import thread_idx, block_idx, block_dim, grid_dim
from tenmo.mnemonics import Add, Multiply, Subtract, Divide
from tenmo.strides import Strides
from tenmo.broadcasthelper import ShapeBroadcaster
from tenmo.device import DeviceState
from tenmo.array import Array
from tenmo.ndbuffer import NDBuffer
from tenmo.kernels.kernel_helpers import simd_op, scalar_op


# =============================================================================
# SEMANTIC CONTRACT (in-place operations: A op= B)
#
#   - A is the in-place accumulator. Its shape NEVER changes.
#   - broadcast_shape is always == A.shape.
#     B is the one that broadcasts to match A. A never
#     broadcasts to match B — that would require reallocation,
#     which is illegal for in-place ops.
#   - A_broadcast_strides == A's natural strides (since
#     A_shape == broadcast_shape, no axis is ever stride-0).
#   - B_broadcast_strides may have stride-0 axes wherever B
#     is smaller than A and needs broadcasting.
#   - "A is strided" means A.is_contiguous() is False, i.e.
#     A is a non-contiguous view (transposed, sliced, etc.).
#     Its physical size is still >= output_size in every axis.
#
# The four paths are chosen by (A_is_contiguous,
# B_is_contiguous, needs_broadcasting):
#
#   PATH 1  A contiguous, B contiguous, no broadcasting
#           — fastest: purely linear indexing, full SIMD
#   PATH 2  A contiguous, B strided OR B needs broadcasting
#           — A at linear i; B via stride decomposition.
#             NOTE: when B is contiguous but needs broadcasting,
#             we still land here and still need stride decomp for B
#             (broadcast stride-0 axes encode which B element to read).
#             No separate "both_contiguous_broadcast" kernel is needed
#             here unlike the out-of-place version, because A is always
#             the full broadcast_shape — A is always linearly indexable
#             when contiguous, and B always needs decomp when broadcasting.
#   PATH 3  A strided, B contiguous, NO broadcasting
#           — A via stride decomposition; B at linear i.
#             REQUIRES not needs_broadcasting (otherwise B's physical
#             buffer is smaller than output_size, making linear reads OOB).
#   PATH 4  Everything else (both strided, or A strided + B broadcasting,
#             or any other unhandled combination).
#             Universal fallback — both via stride decomposition.
#
# Full case table:
#   A_cont  B_cont  needs_bcast  →  Path
#   true    true    false        →  1  (pure linear)
#   true    true    true         →  2  (A linear, B decomposed)
#   true    false   false        →  2  (A linear, B decomposed)
#   true    false   true         →  2  (A linear, B decomposed)
#   false   true    false        →  3  (A decomposed, B linear)
#   false   true    true         →  4  (both decomposed — B too small for linear read)
#   false   false   false        →  4  (both decomposed)
#   false   false   true         →  4  (both decomposed)
# =============================================================================


# =============================================================================
# KERNEL for PATH 1: Both contiguous, same shape, no broadcasting.
#
# Linear index maps directly to both A and B.
# A is read, op is applied, result is written back to A at the same index.
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
                var vec_result = simd_op[op_code, dtype, simd_width](vec_a, vec_b, Scalar[dtype](0))
                A.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var idx = i + j
                    var a = A[idx]
                    var b = B[idx]
                    var res = scalar_op[op_code, dtype](a, b, Scalar[dtype](0))

                    A[idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL for PATH 2: A contiguous; B strided or broadcast-expanded.
#
# A is indexed at its linear position i (contiguous, safe for both read
# and write-back).
# B is stride-decomposed through B_strides, which carry stride-0 on any
# axis where B is broadcast-replicated.
#
# This also covers the sub-case where B IS contiguous but needs broadcasting:
# even a flat contiguous B still requires coordinate decomposition to apply
# the stride-0 broadcast axes correctly — linear access to B would read past
# B's physical allocation on broadcast dims.
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
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width] = 0

                comptime for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx
                    var b_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    vec_result[lane] = scalar_op[op_code, dtype](vec_a[lane], B[b_idx], Scalar[dtype](0))


                # A is contiguous: write back at linear index i.
                A.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var b_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    var res = scalar_op[op_code, dtype](A[linear_idx], B[b_idx], Scalar[dtype](0))

                    A[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL for PATH 3: A strided, B contiguous, NO broadcasting.
#
# Precondition (enforced by dispatcher): B_shape == A_shape and
# needs_broadcasting == False, so B's physical buffer is exactly
# output_size elements. Linear reads of B at index i are in bounds.
#
# A is stride-decomposed to find its physical address a_idx for both
# reading AND writing back. B is loaded linearly.
#
# CRITICAL: This kernel MUST NOT be called when needs_broadcasting is True.
# In that case B's physical buffer is smaller than output_size and
# B.load[width](i) would read out of bounds. Such cases must go to PATH 4.
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
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                # B is contiguous and same size as output: safe to
                # do a vectorised load at position i.
                var vec_b = B.load[width=simd_width](i)

                comptime for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        remaining //= result_shape[dim]

                    var res = scalar_op[op_code, dtype](A[a_idx], vec_b[lane], Scalar[dtype](0))

                    # A is strided: write back at a_idx, not at
                    # the linear index.
                    A[a_idx] = res

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        remaining //= result_shape[dim]

                    var res = scalar_op[op_code, dtype](A[a_idx], B[linear_idx], Scalar[dtype](0))

                    A[a_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL for PATH 4: Universal fallback.
#
# Both A and B are stride-decomposed independently.
# Result is written back at a_idx (A's physical address).
#
# Covers all remaining cases:
#   • A strided, B strided (no broadcasting)
#   • A strided, B contiguous but needs broadcasting (B buffer too small for linear read)
#   • A strided, B strided and needs broadcasting
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
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
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

                    var res = scalar_op[op_code, dtype](A[a_idx], B[b_idx], Scalar[dtype](0))

                    A[a_idx] = res

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

                    var res = scalar_op[op_code, dtype](A[a_idx], B[b_idx], Scalar[dtype](0))

                    A[a_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


@fieldwise_init
struct BinaryInplaceOperations[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def launch[
        op_code: Int,
    ](A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype], sync: Bool = False) raises:
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

        # FIX (tuple order): launch_config returns (threads_per_block, num_blocks).
        # The original code had the return order swapped, so threads_per_block
        # received the num_blocks value (potentially in the thousands) and
        # num_blocks received threads_per_block (128–512), producing entirely
        # wrong GPU dispatch dimensions. The destructuring here and the return
        # order in launch_config are now consistent.
        var (threads_per_block, num_blocks) = Self.launch_config(output_size)

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
            if sync: device_context.synchronize()
            return

        # Broadcast strides needed for paths 2–4.
        #
        # Strides used correctly here: A.strides and B.strides are the
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
                A_buffer,
                B_buffer,
                broadcast_shape.array(),
                B_broadcast_strides.array(),
                output_size,
                rank,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
            if sync: device_context.synchronize()
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
            if sync: device_context.synchronize()
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
        if sync: device_context.synchronize()

    @staticmethod
    def launch_config(output_size: Int) -> Tuple[Int, Int]:
        # FIX (CHUNK_SIZE): The original computed num_blocks as:
        #   ceil(output_size / threads_per_block)
        # ignoring that each thread processes CHUNK_SIZE elements, not 1.
        # CHUNK_SIZE = simd_vectors_per_thread * simd_width (default 2*sw*sw).
        # For fp32 with simd_width=8: CHUNK_SIZE = 128. The original over-launched
        # num_blocks by a factor of 128, wasting GPU occupancy and launch overhead.
        # Surplus blocks are not a correctness bug (they exit the while loop
        # immediately via the `base_idx < size` guard), but the waste is severe
        # for small/medium tensors — the most common in-place shapes.
        #
        # We use a conservative APPROX_CHUNK_SIZE = 32 (= 2 * 16, the maximum
        # simd_width for fp16/bf16 on modern GPUs). For wider types this slightly
        # over-launches, but never under-subscribes — a safe conservative bound.
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

        return threads_per_block, num_blocks
