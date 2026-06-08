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
from tenmo.common_utils import One, Epsilon
from std.math import rsqrt


# =============================================================================
# KERNEL 1 — Both contiguous, same shape, no broadcasting.
#
# Preconditions (enforced by launcher):
#   A_shape == B_shape == broadcast_shape
#   A.is_contiguous() and B.is_contiguous()
#
# Strategy: pure linear indexing. result[i] = op(A[i], B[i]).
# No strides, no coordinate decomposition needed.
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
    var one: SIMD[dtype, simd_width]
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

                comptime if dtype.is_floating_point():
                    one = SIMD[dtype, simd_width](1.0)
                else:
                    one = SIMD[dtype, simd_width](1)

                comptime if op_code == Add:
                    vec_result = vec_a + vec_b
                elif op_code == Subtract:
                    vec_result = vec_a - vec_b
                elif op_code == Multiply:
                    vec_result = vec_a * vec_b
                elif op_code == Divide:
                    vec_result = vec_a / vec_b
                elif op_code == SIGMOID_BACKWARD:
                    vec_result = vec_b * vec_a * (one - vec_a)
                elif op_code == TANH_BACKWARD:
                    vec_result = vec_b * (one - vec_a * vec_a)
                elif op_code == SQRT_BACKWARD:
                    # FIX: guard with epsilon, not 0. rsqrt(0) = +inf.
                    vec_result = vec_b * (
                        SIMD[dtype, simd_width](0.5)
                        * rsqrt(max(vec_a, SIMD[dtype, simd_width](epsilon)))
                    )
                else:  # LOG_BACKWARD
                    vec_result = vec_b / max(
                        vec_a, SIMD[dtype, simd_width](epsilon)
                    )

                result.store[width=simd_width](i, vec_result)

            else:
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
                    elif op_code == Divide:
                        res = a / b
                    elif op_code == SIGMOID_BACKWARD:
                        res = b * a * (One[dtype].value() - a)
                    elif op_code == TANH_BACKWARD:
                        res = b * (One[dtype].value() - a * a)
                    elif op_code == SQRT_BACKWARD:
                        # FIX: epsilon guard.
                        res = b * (Scalar[dtype](0.5) * rsqrt(max(a, epsilon)))
                    else:  # LOG_BACKWARD
                        res = b / max(a, epsilon)

                    result[idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL 2 — Both contiguous, shapes differ — broadcast expansion needed.
#
# Preconditions (enforced by launcher):
#   A.is_contiguous() and B.is_contiguous()
#   A_shape != B_shape  (so A_shape or B_shape != broadcast_shape, or both)
#
# Strategy: coordinate decomposition of the linear result index, then map
# coords through broadcast strides to get A and B physical indices.
#
# Strides passed in ARE Strides.default(shape) — correct because both tensors
# are contiguous. This is cheaper and avoids the general .strides field lookup.
#
# Why a separate kernel from KERNEL 4 (both_strided)?
#   • When both tensors are contiguous, their memory layout is a perfectly flat
#     row-major array. The broadcast stride values are known to be either
#     (standard row-major strides) or 0 for broadcast dims. The compiler can
#     reason about this more aggressively (e.g. constant-fold stride values for
#     common shapes, vectorize across the innermost dim when its stride==1).
#   • This path is the hot case for batched operations: e.g. bias add
#     [B,T,C] + [C], outer-product [N,1] x [1,M], batch norm scale [B,C,H,W] x [C].
#     Isolating it here enables future specializations (rank-1/2 fast paths,
#     inner-dim vectorization) without touching the general strided kernel.
#   • Passing Strides.default() explicitly avoids a correctness hazard: the
#     general path (KERNEL 4) uses A.strides which would also be default strides
#     for contiguous tensors, but the distinction is made explicit and auditable.
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
    A_strides: Array,  # broadcast strides computed from Strides.default(A_shape)
    B_strides: Array,  # broadcast strides computed from Strides.default(B_shape)
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

                    comptime if op_code == Add:
                        vec_result[lane] = A[a_idx] + B[b_idx]
                    elif op_code == Subtract:
                        vec_result[lane] = A[a_idx] - B[b_idx]
                    elif op_code == Multiply:
                        vec_result[lane] = A[a_idx] * B[b_idx]
                    elif op_code == Divide:
                        vec_result[lane] = A[a_idx] / B[b_idx]
                    elif op_code == SIGMOID_BACKWARD:
                        vec_result[lane] = (
                            B[b_idx]
                            * A[a_idx]
                            * (One[dtype].value() - A[a_idx])
                        )
                    elif op_code == TANH_BACKWARD:
                        vec_result[lane] = B[b_idx] * (
                            One[dtype].value() - A[a_idx] * A[a_idx]
                        )
                    elif op_code == SQRT_BACKWARD:
                        # FIX: epsilon guard.
                        vec_result[lane] = B[b_idx] * (
                            Scalar[dtype](0.5) * rsqrt(max(A[a_idx], epsilon))
                        )
                    else:  # LOG_BACKWARD
                        vec_result[lane] = B[b_idx] / max(A[a_idx], epsilon)

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

                    var a = A[a_idx]
                    var b = B[b_idx]
                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = a + b
                    elif op_code == Subtract:
                        res = a - b
                    elif op_code == Multiply:
                        res = a * b
                    elif op_code == Divide:
                        res = a / b
                    elif op_code == SIGMOID_BACKWARD:
                        res = b * a * (One[dtype].value() - a)
                    elif op_code == TANH_BACKWARD:
                        res = b * (One[dtype].value() - a * a)
                    elif op_code == SQRT_BACKWARD:
                        # FIX: epsilon guard.
                        res = b * (Scalar[dtype](0.5) * rsqrt(max(a, epsilon)))
                    else:  # LOG_BACKWARD
                        res = b / max(a, epsilon)

                    result[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL 3 — A contiguous and fills broadcast shape; B is strided/broadcast.
#
# Preconditions (enforced by launcher):
#   A.is_contiguous() and A_shape == broadcast_shape
#
# Strategy: A is read linearly (index == linear result index).
#           B is accessed via coordinate decomposition + B_broadcast_strides.
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

                    comptime if op_code == Add:
                        vec_result[lane] = vec_a[lane] + B[b_idx]
                    elif op_code == Subtract:
                        vec_result[lane] = vec_a[lane] - B[b_idx]
                    elif op_code == Multiply:
                        vec_result[lane] = vec_a[lane] * B[b_idx]
                    elif op_code == Divide:
                        vec_result[lane] = vec_a[lane] / B[b_idx]
                    elif op_code == SIGMOID_BACKWARD:
                        vec_result[lane] = (
                            B[b_idx]
                            * vec_a[lane]
                            * (One[dtype].value() - vec_a[lane])
                        )
                    elif op_code == TANH_BACKWARD:
                        vec_result[lane] = B[b_idx] * (
                            One[dtype].value() - vec_a[lane] * vec_a[lane]
                        )
                    elif op_code == SQRT_BACKWARD:
                        # FIX: epsilon guard.
                        vec_result[lane] = B[b_idx] * (
                            Scalar[dtype](0.5)
                            * rsqrt(max(vec_a[lane], epsilon))
                        )
                    else:  # LOG_BACKWARD
                        vec_result[lane] = B[b_idx] / max(vec_a[lane], epsilon)

                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var b_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    var a = A[linear_idx]
                    var b = B[b_idx]
                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = a + b
                    elif op_code == Subtract:
                        res = a - b
                    elif op_code == Multiply:
                        res = a * b
                    elif op_code == Divide:
                        res = a / b
                    elif op_code == SIGMOID_BACKWARD:
                        res = b * a * (One[dtype].value() - a)
                    elif op_code == TANH_BACKWARD:
                        res = b * (One[dtype].value() - a * a)
                    elif op_code == SQRT_BACKWARD:
                        # FIX: epsilon guard.
                        res = b * (Scalar[dtype](0.5) * rsqrt(max(a, epsilon)))
                    else:  # LOG_BACKWARD
                        res = b / max(a, epsilon)

                    result[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL 4 — B contiguous and fills broadcast shape; A is strided/broadcast.
#
# Preconditions (enforced by launcher):
#   B.is_contiguous() and B_shape == broadcast_shape
#
# Strategy: B is read linearly. A is accessed via A_broadcast_strides.
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
    # FIX: Original had a spurious closing parenthesis: block_idx.x)
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
                var vec_b = B.load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width] = 0

                comptime for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        remaining //= result_shape[dim]

                    comptime if op_code == Add:
                        vec_result[lane] = A[a_idx] + vec_b[lane]
                    elif op_code == Subtract:
                        vec_result[lane] = A[a_idx] - vec_b[lane]
                    elif op_code == Multiply:
                        vec_result[lane] = A[a_idx] * vec_b[lane]
                    elif op_code == Divide:
                        vec_result[lane] = A[a_idx] / vec_b[lane]
                    elif op_code == SIGMOID_BACKWARD:
                        vec_result[lane] = (
                            vec_b[lane]
                            * A[a_idx]
                            * (One[dtype].value() - A[a_idx])
                        )
                    elif op_code == TANH_BACKWARD:
                        vec_result[lane] = vec_b[lane] * (
                            One[dtype].value() - A[a_idx] * A[a_idx]
                        )
                    elif op_code == SQRT_BACKWARD:
                        # FIX: epsilon guard.
                        vec_result[lane] = vec_b[lane] * (
                            Scalar[dtype](0.5) * rsqrt(max(A[a_idx], epsilon))
                        )
                    else:  # LOG_BACKWARD
                        vec_result[lane] = vec_b[lane] / max(A[a_idx], epsilon)

                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        remaining //= result_shape[dim]

                    var a = A[a_idx]
                    var b = B[linear_idx]
                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = a + b
                    elif op_code == Subtract:
                        res = a - b
                    elif op_code == Multiply:
                        res = a * b
                    elif op_code == Divide:
                        res = a / b
                    elif op_code == SIGMOID_BACKWARD:
                        res = b * a * (One[dtype].value() - a)
                    elif op_code == TANH_BACKWARD:
                        res = b * (One[dtype].value() - a * a)
                    elif op_code == SQRT_BACKWARD:
                        # FIX: epsilon guard.
                        res = b * (Scalar[dtype](0.5) * rsqrt(max(a, epsilon)))
                    else:  # LOG_BACKWARD
                        res = b / max(a, epsilon)

                    result[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# KERNEL 5 — General fallback: both tensors non-contiguous and/or strided.
#
# Preconditions: none (handles everything not caught by kernels 1–4).
#
# Strategy: coordinate decomposition for both A and B via their actual
# broadcast strides (computed from A.strides, B.strides in the launcher).
#
# This is the most general but most expensive path. Any non-contiguous tensor
# (transposed, sliced, permuted) that wasn't caught by the earlier paths lands
# here. Correctness is guaranteed via the actual strides from the NDBuffer.
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

                    comptime if op_code == Add:
                        vec_result[lane] = A[a_idx] + B[b_idx]
                    elif op_code == Subtract:
                        vec_result[lane] = A[a_idx] - B[b_idx]
                    elif op_code == Multiply:
                        vec_result[lane] = A[a_idx] * B[b_idx]
                    elif op_code == Divide:
                        vec_result[lane] = A[a_idx] / B[b_idx]
                    elif op_code == SIGMOID_BACKWARD:
                        vec_result[lane] = (
                            B[b_idx]
                            * A[a_idx]
                            * (One[dtype].value() - A[a_idx])
                        )
                    elif op_code == TANH_BACKWARD:
                        vec_result[lane] = B[b_idx] * (
                            One[dtype].value() - A[a_idx] * A[a_idx]
                        )
                    elif op_code == SQRT_BACKWARD:
                        # FIX: epsilon guard.
                        vec_result[lane] = B[b_idx] * (
                            Scalar[dtype](0.5) * rsqrt(max(A[a_idx], epsilon))
                        )
                    else:  # LOG_BACKWARD
                        vec_result[lane] = B[b_idx] / max(A[a_idx], epsilon)

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

                    var a = A[a_idx]
                    var b = B[b_idx]
                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = a + b
                    elif op_code == Subtract:
                        res = a - b
                    elif op_code == Multiply:
                        res = a * b
                    elif op_code == Divide:
                        res = a / b
                    elif op_code == SIGMOID_BACKWARD:
                        res = b * a * (One[dtype].value() - a)
                    elif op_code == TANH_BACKWARD:
                        res = b * (One[dtype].value() - a * a)
                    elif op_code == SQRT_BACKWARD:
                        # FIX: epsilon guard.
                        res = b * (Scalar[dtype](0.5) * rsqrt(max(a, epsilon)))
                    else:  # LOG_BACKWARD
                        res = b / max(a, epsilon)

                    result[linear_idx] = res

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
            A_shape, A.strides, broadcast_shape  # FIX: actual strides
        )
        var B_broadcast_strides = ShapeBroadcaster.broadcast_strides(
            B_shape, B.strides, broadcast_shape  # FIX: actual strides
        )

        # ================================================================
        # PATH 3: A contiguous and fills broadcast shape; B strided/broadcast.
        #
        # A is indexed linearly (index == result linear index).
        # B is accessed via B_broadcast_strides (built from B.strides above).
        #
        # FIX vs original: Removed the incorrect extra guard `B_shape != broadcast_shape`.
        # The original condition was:
        #   A_is_contiguous and A_shape == broadcast_shape and B_shape != broadcast_shape
        # When B_shape == broadcast_shape but B is non-contiguous, the original
        # code skipped this path and fell to PATH 4. But B_broadcast_strides
        # correctly handles B regardless of whether B_shape == broadcast_shape.
        # The only precondition that matters is A_is_contiguous and A_shape == broadcast_shape.
        # Example: [B,T,C] (contiguous) + [B,T,C] (transposed/non-contiguous)
        # ================================================================
        if A_is_contiguous and A_shape == broadcast_shape:
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
        #
        # B is indexed linearly. A is accessed via A_broadcast_strides.
        #
        # FIX vs original: Same fix — removed extra guard `A_shape != broadcast_shape`.
        # Example: [B,T,C] (transposed) + [B,T,C] (contiguous)
        # ================================================================
        if B_is_contiguous and B_shape == broadcast_shape:
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

        # Issue  — Path 2 Should Not Exist for bias_add 🔴
        # bias_add is (64,128) + (128,) — this is exactly Path 3's use case:
        #
        # ================================================================
        # PATH 2: Both contiguous, shapes differ — broadcast expansion needed.
        #
        # Both buffers are flat in memory. Use Strides.default() for broadcast
        # strides — correct and cheap because both ARE contiguous.
        # Example: [B,T,C] + [C]    (bias add)
        #          [N,1]   + [1,M]  (outer product)
        #          [B,1,H,W] + [B,C,H,W]
        #
        # FIX vs original: the original had no PATH 2; these cases fell through
        # to PATH 4 (both_strided) which then used A.strides / B.strides —
        # also Strides.default for contiguous, so correctness was accidental —
        # but the code was conflated with the genuinely-strided case and could
        # not be independently optimized or audited.
        # ================================================================
        if A_is_contiguous and B_is_contiguous:
            # Both contiguous → safe to use Strides.default() for broadcast strides.
            _="""var A_broadcast_strides = ShapeBroadcaster.broadcast_strides(
                A_shape, Strides.default(A_shape), broadcast_shape
            )
            var B_broadcast_strides = ShapeBroadcaster.broadcast_strides(
                B_shape, Strides.default(B_shape), broadcast_shape
            )"""
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
        # PATH 5: General fallback — at least one non-contiguous tensor,
        #         and neither operand fills the broadcast shape contiguously.
        #
        # Both A and B accessed via coordinate decomposition + actual strides.
        # Handles all remaining cases: transposed + strided, non-contiguous
        # views with arbitrary strides, etc.
        # Example: A.T (transposed) + B[::2] (strided slice)
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
