from std.sys import simd_width_of
from std.gpu import thread_idx, block_idx, block_dim, grid_dim
from .mnemonics import Add, Multiply, Subtract, Divide
from .strides import Strides
from .broadcasthelper import ShapeBroadcaster
from .device import DeviceState
from .array import Array
from .ndbuffer import NDBuffer


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
# The four paths below are chosen by (A_is_contiguous,
# B_is_contiguous, needs_broadcasting):
#
#   PATH 1  A contiguous, B contiguous, no broadcasting
#           — fastest: purely linear indexing, full SIMD
#   PATH 2  A contiguous, B strided OR B needs broadcasting
#           — A at linear i; B via stride decomposition
#   PATH 3  A strided,    B contiguous, NO broadcasting
#           — A via stride decomposition; B at linear i
#           — REQUIRES not needs_broadcasting (otherwise B's
#             physical buffer is too small for linear reads)
#   PATH 4  everything else (both strided, or either/both
#             needs broadcasting and is not contiguous)
#           — both via stride decomposition; universal fallback


# PATH 1: Both contiguous, same shape, no broadcasting.
# Linear index maps directly to both A and B.
fn arithmetic_ops_both_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

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
                var vec_result: SIMD[dtype, simd_width]

                comptime if op_code == Add:
                    vec_result = vec_a + vec_b
                elif op_code == Subtract:
                    vec_result = vec_a - vec_b
                elif op_code == Multiply:
                    vec_result = vec_a * vec_b
                else:
                    vec_result = vec_a / vec_b

                A.store[width=simd_width](i, vec_result)

            else:
                # Tail: fewer than simd_width elements remain.
                for j in range(size - i):
                    var idx = i + j
                    var a = A[idx]
                    var b = B[idx]
                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = a + b
                    elif op_code == Subtract:
                        res = a - b
                    elif op_code == Multiply:
                        res = a * b
                    else:
                        res = a / b

                    A[idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# PATH 2: A contiguous, B strided or broadcast-expanded.
#
# A is indexed at its linear position i (contiguous, safe).
# B is stride-decomposed through B_strides, which carry
# stride-0 on any axis where B is broadcast-replicated.
# Result is written back to A at the same linear index.
fn arithmetic_ops_A_contiguous[
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
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

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

                # Each lane computes its own B physical address
                # via stride decomposition. comptime for unrolls
                # the loop with lane as a compile-time constant;
                # the body's runtime expressions are independent
                # per-lane — no aliasing between lanes.
                comptime for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx
                    var b_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    comptime if op_code == Add:
                        vec_result[lane] = vec_a[lane] + B[b_idx]
                    elif op_code == Subtract:
                        vec_result[lane] = vec_a[lane] - B[b_idx]
                    elif op_code == Multiply:
                        vec_result[lane] = vec_a[lane] * B[b_idx]
                    else:
                        vec_result[lane] = vec_a[lane] / B[b_idx]

                # A is contiguous: write back at linear index i.
                A.store[width=simd_width](i, vec_result)

            else:
                # Tail path: scalar loop over remaining elements.
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var b_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = A[linear_idx] + B[b_idx]
                    elif op_code == Subtract:
                        res = A[linear_idx] - B[b_idx]
                    elif op_code == Multiply:
                        res = A[linear_idx] * B[b_idx]
                    else:
                        res = A[linear_idx] / B[b_idx]

                    A[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# PATH 3: A strided, B contiguous, NO broadcasting.
#
# Precondition (enforced by the dispatcher): B_shape == A_shape,
# so B's physical buffer is exactly output_size elements.
# Linear index i is safe to use directly on B.
#
# A is stride-decomposed to find its physical address a_idx
# for both reading and writing.
#
# NOTE: This path MUST NOT be taken when needs_broadcasting is
# true, because then B's physical buffer is smaller than
# output_size and the linear load at i would be out of bounds.
fn arithmetic_ops_B_contiguous[
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
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                # B is contiguous and same size as output: safe
                # vectorised load at position i.
                var vec_b = B.load[width=simd_width](i)

                comptime for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        remaining //= result_shape[dim]

                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = A[a_idx] + vec_b[lane]
                    elif op_code == Subtract:
                        res = A[a_idx] - vec_b[lane]
                    elif op_code == Multiply:
                        res = A[a_idx] * vec_b[lane]
                    else:
                        res = A[a_idx] / vec_b[lane]

                    # A is strided: write back at a_idx, not
                    # at the linear index.
                    A[a_idx] = res

            else:
                # Tail path.
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        remaining //= result_shape[dim]

                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = A[a_idx] + B[linear_idx]
                    elif op_code == Subtract:
                        res = A[a_idx] - B[linear_idx]
                    elif op_code == Multiply:
                        res = A[a_idx] * B[linear_idx]
                    else:
                        res = A[a_idx] / B[linear_idx]

                    A[a_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# PATH 4: Both strided, or B needs broadcasting alongside
#         a non-contiguous A, or any other case.
#
# Both A and B are stride-decomposed independently.
# Result is written back at a_idx (A's physical address).
# Universal fallback: handles all remaining cases correctly.
fn arithmetic_ops_both_strided[
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
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

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

                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = A[a_idx] + B[b_idx]
                    elif op_code == Subtract:
                        res = A[a_idx] - B[b_idx]
                    elif op_code == Multiply:
                        res = A[a_idx] * B[b_idx]
                    else:
                        res = A[a_idx] / B[b_idx]

                    A[a_idx] = res

            else:
                # Tail path.
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

                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = A[a_idx] + B[b_idx]
                    elif op_code == Subtract:
                        res = A[a_idx] - B[b_idx]
                    elif op_code == Multiply:
                        res = A[a_idx] * B[b_idx]
                    else:
                        res = A[a_idx] / B[b_idx]

                    A[a_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


@fieldwise_init
struct BinaryInplaceOperations[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    fn launch[
        op_code: Int,
    ](A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]) raises:
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

        var (threads_per_block, num_blocks) = Self.launch_config(output_size)

        ref A_device_state = A.device_state.value()
        ref B_device_state = B.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        ref A_buffer = A_device_state.device_buffer()
        ref B_buffer = B_device_state.device_buffer()

        var A_is_contiguous = A.is_contiguous()
        var B_is_contiguous = B.is_contiguous()

        # PATH 1: Both contiguous, same shape, no broadcasting.
        #
        # Conditions:
        #   • A_shape == B_shape  (already guaranteed since
        #     broadcast_shape == A_shape and needs_broadcasting
        #     is false, so B_shape == A_shape)
        #   • Both physically contiguous
        #   • No broadcast expansion needed
        #
        # Fastest path: purely linear indexing, full SIMD on both.
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
            device_context.synchronize()
            return

        # Broadcast strides are only needed for paths 2-4.
        # A_broadcast_strides == A's natural strides (no stride-0
        # axes, because broadcast_shape == A_shape by contract).
        # B_broadcast_strides may have stride-0 axes where B is
        # smaller than A and needs replication.
        var A_broadcast_strides = ShapeBroadcaster.broadcast_strides(
            A_shape, Strides.default(A_shape), broadcast_shape
        )
        var B_broadcast_strides = ShapeBroadcaster.broadcast_strides(
            B_shape, Strides.default(B_shape), broadcast_shape
        )

        # PATH 2: A contiguous, B strided or broadcast-expanded.
        #
        # Conditions:
        #   • A is physically contiguous
        #   • B is either non-contiguous OR needs broadcasting
        #     (or both — the kernel handles all sub-cases via
        #     B_broadcast_strides, which encodes stride-0 for
        #     broadcast axes and physical strides elsewhere)
        #
        # A is indexed at linear i directly.
        # B is always stride-decomposed — safe regardless of
        # whether B is a strided view or a broadcast-expanded
        # smaller tensor.
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
            device_context.synchronize()
            return

        # PATH 3: A strided, B contiguous, NO broadcasting.
        #
        # Conditions:
        #   • A is non-contiguous (strided view of A's buffer)
        #   • B is physically contiguous
        #   • needs_broadcasting is False — B_shape == A_shape,
        #     so B's physical buffer is exactly output_size
        #     elements and linear reads at i are in bounds.
        #
        # CRITICAL: `not needs_broadcasting` is mandatory here.
        # If B needed broadcasting, B's physical size would be
        # smaller than output_size, and B.load[width](i) would
        # read past the end of B's allocation. Such cases fall
        # through to Path 4, which stride-decomposes B safely.
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
            device_context.synchronize()
            return

        # PATH 4: Universal fallback — both strided and/or
        #         B needs broadcasting.
        #
        # Covers:
        #   • A strided, B strided (no broadcasting)
        #   • A strided, B contiguous but needs broadcasting
        #   • A strided, B strided and needs broadcasting
        #   • A contiguous, B contiguous but needs broadcasting
        #     (edge case: B_is_contiguous and needs_broadcasting
        #      — Path 2 also covers this, but if it somehow fell
        #      through, Path 4 handles it correctly too)
        #
        # Both A and B are stride-decomposed.
        # A_broadcast_strides has no stride-0 axes (contract).
        # B_broadcast_strides has stride-0 on broadcast axes.
        # Result written at a_idx (A's physical address).
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
        device_context.synchronize()

    # ──────────────────────────────────────────────────────────
    # launch_config: returns (threads_per_block, num_blocks).
    #
    # BUG FIX: the original code returned (num_blocks,
    # threads_per_block) — i.e. the tuple was backwards relative
    # to the caller's destructuring pattern:
    #   var (threads_per_block, num_blocks) = Self.launch_config(...)
    # This caused threads_per_block to receive the num_blocks
    # value (potentially thousands) and vice versa, producing
    # entirely wrong GPU dispatch dimensions.
    #
    # The fix is simply to return in the order the caller expects.
    # ──────────────────────────────────────────────────────────
    @staticmethod
    fn launch_config(output_size: Int) -> Tuple[Int, Int]:
        var threads_per_block: Int
        var num_blocks: Int

        if output_size < 4096:
            threads_per_block = 128
            num_blocks = (output_size + 127) // 128
        elif output_size < 65536:
            threads_per_block = 256
            num_blocks = min((output_size + 255) // 256, 128)
        else:
            threads_per_block = 512
            num_blocks = min((output_size + 511) // 512, 512)

        # Return order matches caller destructuring:
        #   var (threads_per_block, num_blocks) = Self.launch_config(...)
        return threads_per_block, num_blocks

