from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from tenmo.tensor import Tensor
from tenmo.common_utils import panic, Epsilon
from tenmo.shapes import Shape
from tenmo.device import DeviceState
from tenmo.ndbuffer import NDBuffer
from . import elementwise_launch_config
from tenmo.array import Array
from tenmo.shared.scalar_ops import simd_op, scalar_op

# Kernel template for various arithmetic ops involving ND Tensor and a single scalar
# Simplification - views becomes contiguous when copied to device and offset becomes 0


def inplace_scalar_ops[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2
    * simd_width,  # Each thread processes twice simd size elements
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    scalar: Scalar[dtype],
    size: Int,
):
    """
    Element-wise scalar operations.

    - Each thread processes multiple items (better ILP)
    - SIMD vectorization within each item
    - Loop unrolling
    - Minimal divergence

    """

    var tid = thread_idx.x
    var gtid = Int(tid + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    # ===================================================================
    # Each thread processes CHUNK_SIZE elements
    # ===================================================================
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        # Process simd_vectors_per_thread vectors per thread
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            # Bounds check for this vector
            if i + simd_width <= size:
                # Full vector load
                var vec_a = A.load[width=simd_width](i)
                var vec_result = simd_op[op_code, dtype, simd_width](
                    vec_a, SIMD[dtype, simd_width](scalar), Epsilon[dtype].value()
                )
                A.store[width=simd_width](i, vec_result)
            elif i < size:
                # Partial vector (tail handling)
                for j in range(size - i):
                    var val = A[i + j]
                    var res = scalar_op[op_code, dtype](
                        val, scalar, Epsilon[dtype].value()
                    )
                    A[i + j] = res

        base_idx += stride * CHUNK_SIZE


# =============================================================================
# POW KERNELS for inplace — dedicated f32/f64 kernels using pow() intrinsic.
#
# The generic inplace_scalar_ops kernel uses simd_op[op_code] which relies on
# the `**` operator. GPU backends do not reliably support `**` for all dtypes
# (especially integers), so POW uses separate typed kernels with the GPU math
# library's pow() intrinsic — matching the out-of-place ScalarOperations.launch_pow.
#
# Strided variants handle non-contiguous GPU views (transposed, sliced).
# =============================================================================


def inplace_pow_op_f32[
    simd_width: Int = simd_width_of[DType.float32](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    exponent: Scalar[DType.float32],
    size: Int,
):
    """Inplace float32 pow — A[i] = pow(A[i], exponent). Contiguous only."""
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var vec_exp = SIMD[DType.float32, simd_width](exponent)
                A.store[width=simd_width](i, pow(vec_a, vec_exp))

            elif i < size:
                for j in range(size - i):
                    A[i + j] = pow(A[i + j], exponent)

        base_idx += stride * CHUNK_SIZE


def inplace_pow_op_f64[
    simd_width: Int = simd_width_of[DType.float64](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
    exponent: Scalar[DType.float64],
    size: Int,
):
    """Inplace float64 pow — A[i] = pow(A[i], exponent). Contiguous only."""
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var vec_exp = SIMD[DType.float64, simd_width](exponent)
                A.store[width=simd_width](i, pow(vec_a, vec_exp))

            elif i < size:
                for j in range(size - i):
                    A[i + j] = pow(A[i + j], exponent)

        base_idx += stride * CHUNK_SIZE


# =============================================================================
# KERNEL for non-contiguous A (strided view).
#
# Used when inplace_scalar_ops is called on a GPU view that is NOT contiguous
# (e.g., transposed, sliced). The kernel decomposes each logical index into
# coordinates via shape, then computes the physical address via strides.
#
# OPTIMIZATION: Outer-base decomposition computed ONCE per SIMD vector.
#   Per lane: only ONE multiply for the innermost dimension:
#     a_idx = a_base + (inner_offset + lane) * A_strides[rank-1]
#   Saving vs full rank-level decomposition per lane:
#     (rank-1 mod + rank-1 div) per vector instead of
#     (rank mod + rank div) * simd_width per vector
#
#   Fast path (vector within one row, a_inner_stride == 1):
#     SIMD load from a_base + inner_offset, SIMD store back.
#
#   Fast path (vector within one row, a_inner_stride != 1):
#     Per-lane scalar at a_base + (inner_offset + lane) * a_inner_stride.
#     Saving: outer decomp done once, only 1 multiply per lane for inner dim.
#
#   Slow path (crosses row boundary):
#     Full per-lane decomposition — correctness over performance.
# =============================================================================
def inplace_scalar_ops_strided[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    scalar: Scalar[dtype],
    shape: Array,
    strides: Array,
    numels: Int,
    rank: Int,
):
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var grid_stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var inner_dim = shape[rank - 1]
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < numels:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= numels:
                break

            if i + simd_width <= numels:
                # Compute A outer base once for this vector.
                # Strip innermost coord first — prevents double-counting.
                var inner_offset = i % inner_dim
                var outer_remaining = i // inner_dim
                var a_base = 0
                var a_inner_stride = strides[rank - 1]

                for dim in range(rank - 2, -1, -1):
                    var coord = outer_remaining % shape[dim]
                    a_base += coord * strides[dim]
                    outer_remaining //= shape[dim]

                if inner_offset + simd_width <= inner_dim:
                    # ── Fast path: vector fits within one row ─────────────
                    if a_inner_stride == 1:
                        # A elements are consecutive in memory →
                        # SIMD load AND SIMD store both safe.
                        var vec_a = A.load[width=simd_width](
                            a_base + inner_offset
                        )
                        var vec_result = simd_op[op_code, dtype, simd_width](
                            vec_a,
                            SIMD[dtype, simd_width](scalar),
                            Epsilon[dtype].value(),
                        )
                        A.store[width=simd_width](
                            a_base + inner_offset, vec_result
                        )

                    else:
                        # A elements are strided (a_inner_stride != 1).
                        # Per-lane read and write, outer base computed once.
                        comptime for lane in range(simd_width):
                            var a_idx = (
                                a_base
                                + (inner_offset + lane) * a_inner_stride
                            )
                            var a = A[a_idx]
                            var res = scalar_op[op_code, dtype](
                                a, scalar, Epsilon[dtype].value()
                            )
                            A[a_idx] = res

                else:
                    # ── Slow path: crosses row boundary ───────────────────
                    # Full per-lane decomposition for A.
                    comptime for lane in range(simd_width):
                        var linear_idx = i + lane
                        var rem = linear_idx
                        var a_idx = 0

                        for dim in range(rank - 1, -1, -1):
                            var coord = rem % shape[dim]
                            a_idx += coord * strides[dim]
                            rem //= shape[dim]

                        var a = A[a_idx]
                        var res = scalar_op[op_code, dtype](
                            a, scalar, Epsilon[dtype].value()
                        )
                        A[a_idx] = res

            else:
                # ── Tail: fewer than simd_width elements remain ───────────
                for j in range(numels - i):
                    var linear_idx = i + j
                    var rem = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = rem % shape[dim]
                        a_idx += coord * strides[dim]
                        rem //= shape[dim]

                    var a = A[a_idx]
                    var res = scalar_op[op_code, dtype](
                        a, scalar, Epsilon[dtype].value()
                    )
                    A[a_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


# =============================================================================
# POW KERNELS for non-contiguous inplace (strided views).
#
# Mirrors inplace_scalar_ops_strided but uses pow() intrinsic instead of
# simd_op/scalar_op. Three paths per SIMD vector:
#   1. Fast path: vector within one row, inner stride == 1  → SIMD load/store
#   2. Fast path: vector within one row, inner stride != 1  → per-lane strided
#   3. Slow path: crosses row boundary                      → full decomposition
# Plus tail handling for leftover elements.
# =============================================================================


def inplace_pow_op_f32_strided[
    simd_width: Int = simd_width_of[DType.float32](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    exponent: Scalar[DType.float32],
    shape: Array,
    strides: Array,
    numels: Int,
    rank: Int,
):
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var grid_stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var inner_dim = shape[rank - 1]
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < numels:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= numels:
                break

            if i + simd_width <= numels:
                var inner_offset = i % inner_dim
                var outer_remaining = i // inner_dim
                var a_base = 0
                var a_inner_stride = strides[rank - 1]

                for dim in range(rank - 2, -1, -1):
                    var coord = outer_remaining % shape[dim]
                    a_base += coord * strides[dim]
                    outer_remaining //= shape[dim]

                if inner_offset + simd_width <= inner_dim:
                    if a_inner_stride == 1:
                        var vec_a = A.load[width=simd_width](
                            a_base + inner_offset
                        )
                        var vec_exp = SIMD[DType.float32, simd_width](exponent)
                        A.store[width=simd_width](
                            a_base + inner_offset, pow(vec_a, vec_exp)
                        )
                    else:
                        comptime for lane in range(simd_width):
                            var a_idx = (
                                a_base
                                + (inner_offset + lane) * a_inner_stride
                            )
                            A[a_idx] = pow(A[a_idx], exponent)
                else:
                    comptime for lane in range(simd_width):
                        var linear_idx = i + lane
                        var rem = linear_idx
                        var a_idx = 0

                        for dim in range(rank - 1, -1, -1):
                            var coord = rem % shape[dim]
                            a_idx += coord * strides[dim]
                            rem //= shape[dim]

                        A[a_idx] = pow(A[a_idx], exponent)
            else:
                for j in range(numels - i):
                    var linear_idx = i + j
                    var rem = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = rem % shape[dim]
                        a_idx += coord * strides[dim]
                        rem //= shape[dim]

                    A[a_idx] = pow(A[a_idx], exponent)

        base_idx += grid_stride * CHUNK_SIZE


def inplace_pow_op_f64_strided[
    simd_width: Int = simd_width_of[DType.float64](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
    exponent: Scalar[DType.float64],
    shape: Array,
    strides: Array,
    numels: Int,
    rank: Int,
):
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var grid_stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var inner_dim = shape[rank - 1]
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < numels:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= numels:
                break

            if i + simd_width <= numels:
                var inner_offset = i % inner_dim
                var outer_remaining = i // inner_dim
                var a_base = 0
                var a_inner_stride = strides[rank - 1]

                for dim in range(rank - 2, -1, -1):
                    var coord = outer_remaining % shape[dim]
                    a_base += coord * strides[dim]
                    outer_remaining //= shape[dim]

                if inner_offset + simd_width <= inner_dim:
                    if a_inner_stride == 1:
                        var vec_a = A.load[width=simd_width](
                            a_base + inner_offset
                        )
                        var vec_exp = SIMD[DType.float64, simd_width](exponent)
                        A.store[width=simd_width](
                            a_base + inner_offset, pow(vec_a, vec_exp)
                        )
                    else:
                        comptime for lane in range(simd_width):
                            var a_idx = (
                                a_base
                                + (inner_offset + lane) * a_inner_stride
                            )
                            A[a_idx] = pow(A[a_idx], exponent)
                else:
                    comptime for lane in range(simd_width):
                        var linear_idx = i + lane
                        var rem = linear_idx
                        var a_idx = 0

                        for dim in range(rank - 1, -1, -1):
                            var coord = rem % shape[dim]
                            a_idx += coord * strides[dim]
                            rem //= shape[dim]

                        A[a_idx] = pow(A[a_idx], exponent)
            else:
                for j in range(numels - i):
                    var linear_idx = i + j
                    var rem = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = rem % shape[dim]
                        a_idx += coord * strides[dim]
                        rem //= shape[dim]

                    A[a_idx] = pow(A[a_idx], exponent)

        base_idx += grid_stride * CHUNK_SIZE


struct InplaceScalarOperations[dtype: DType = DType.float32](
    ImplicitlyCopyable & Movable
):
    @staticmethod
    def launch[
        op_code: Int,
    ](A: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype], sync: Bool = False) raises:
        var numels = A.numels()
        var rank = A.shape.rank()

        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )
        ref A_device_state = A.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        ref A_buffer = A_device_state.device_buffer()

        # Dispatch on contiguity — same pattern as BinaryInplaceOperations.launch.
        if A.is_contiguous():
            # PATH 1: Contiguous A → flat linear indexing (fast SIMD).
            var compiled_func = device_context.compile_function[
                inplace_scalar_ops[
                    op_code=op_code,
                    dtype=Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
                inplace_scalar_ops[
                    op_code=op_code,
                    dtype=Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
            ]()

            device_context.enqueue_function(
                compiled_func,
                A_buffer,
                scalar,
                numels,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        else:
            # PATH 2: Strided A → stride-decomposed indexing (correct for views).
            var compiled_func = device_context.compile_function[
                inplace_scalar_ops_strided[
                    op_code=op_code,
                    dtype=Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
                inplace_scalar_ops_strided[
                    op_code=op_code,
                    dtype=Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
            ]()

            device_context.enqueue_function(
                compiled_func,
                A_buffer,
                scalar,
                A.shape.array(),
                A.strides.array(),
                numels,
                rank,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        if sync: device_context.synchronize()

    @staticmethod
    def launch_inplace_pow(
        A: NDBuffer[Self.dtype],
        exponent: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises:
        """
        Dedicated inplace POW launcher — dispatches to typed f32/f64 kernels.
        Handles both contiguous and strided GPU views.
        POW only supported for float32 and float64.
        """
        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )
        ref A_device_state = A.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]
        ref A_buffer = A_device_state.device_buffer()

        comptime if Self.dtype == DType.float32:
            if A.is_contiguous():
                var compiled = device_context.compile_function[
                    inplace_pow_op_f32[
                        simd_width=simdwidth,
                        simd_vectors_per_thread=2 * simdwidth,
                    ],
                    inplace_pow_op_f32[
                        simd_width=simdwidth,
                        simd_vectors_per_thread=2 * simdwidth,
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    A_buffer,
                    exponent.cast[DType.float32](),
                    numels,
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            else:
                var rank = A.shape.rank()
                var compiled = device_context.compile_function[
                    inplace_pow_op_f32_strided[
                        simd_width=simdwidth,
                        simd_vectors_per_thread=2 * simdwidth,
                    ],
                    inplace_pow_op_f32_strided[
                        simd_width=simdwidth,
                        simd_vectors_per_thread=2 * simdwidth,
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    A_buffer,
                    exponent.cast[DType.float32](),
                    A.shape.array(),
                    A.strides.array(),
                    numels,
                    rank,
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
        elif Self.dtype == DType.float64:
            if A.is_contiguous():
                var compiled = device_context.compile_function[
                    inplace_pow_op_f64[
                        simd_width=simdwidth,
                        simd_vectors_per_thread=2 * simdwidth,
                    ],
                    inplace_pow_op_f64[
                        simd_width=simdwidth,
                        simd_vectors_per_thread=2 * simdwidth,
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    A_buffer,
                    exponent.cast[DType.float64](),
                    numels,
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            else:
                var rank = A.shape.rank()
                var compiled = device_context.compile_function[
                    inplace_pow_op_f64_strided[
                        simd_width=simdwidth,
                        simd_vectors_per_thread=2 * simdwidth,
                    ],
                    inplace_pow_op_f64_strided[
                        simd_width=simdwidth,
                        simd_vectors_per_thread=2 * simdwidth,
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    A_buffer,
                    exponent.cast[DType.float64](),
                    A.shape.array(),
                    A.strides.array(),
                    numels,
                    rank,
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
        else:
            panic(
                "InplaceScalarOperations: POW only supported for float32 and float64"
            )

        if sync: device_context.synchronize()

    @staticmethod
    def launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
        return elementwise_launch_config(numels, simdwidth)
