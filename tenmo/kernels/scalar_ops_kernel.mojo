from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from ..shared.constants import Epsilon
from ..shared.panic import panic
from ..shared.scalar_ops import simd_op, scalar_op
from ..gpu.device import DeviceState
from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from ..shared.array import RankArray
from .kernel_helpers import elementwise_launch_config

def scalar_ops[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2
    * simd_width,  # Each thread processes twice simd size elements
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    scalar: Scalar[dtype],
    size_: Int64,
):
    """
        Element-wise scalar operations.

    - Each thread processes multiple items (better ILP)
    - SIMD vectorization within each item
    - Loop unrolling
    - Minimal divergence


    KERNEL for contiguous A (flat linear indexing).
    Used when scalar_ops is called on a GPU buffer that IS contiguous.
    Each thread processes CHUNK_SIZE elements with SIMD vectorization.
    The A input and result output are both flat-contiguous.
    """
    var size = Int(size_)

    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    # Each thread processes CHUNK_SIZE elements
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        # Process simd_vectors_per_thread vectors per thread
        comptime for vector in range(simd_vectors_per_thread):
            var i = base_idx + vector * simd_width

            # Bounds check for this vector
            if i + simd_width <= size:
                # Full vector load
                var vec_a = A.unsafe_load[width=simd_width](i)
                var vec_result = simd_op[op_code, dtype, simd_width](
                    vec_a,
                    SIMD[dtype, simd_width](scalar),
                    Epsilon[dtype].value(),
                )
                result.unsafe_store[width=simd_width](i, vec_result)
            elif i < size:
                # Partial vector (tail handling)
                for j in range(size - i):
                    var val = A[unsafe_offset=i + j]
                    var res = scalar_op[op_code, dtype](
                        val, scalar, Epsilon[dtype].value()
                    )
                    result[unsafe_offset=i + j] = res

        base_idx += stride * CHUNK_SIZE


def scalar_ops_strided[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    scalar: Scalar[dtype],
    shape: RankArray,
    strides: RankArray,
    numels_: Int64,
    rank_: Int64,
):
    """KERNEL for non-contiguous A (strided view).
    Used when scalar_ops is called on a GPU view that is NOT contiguous
    (e.g., transposed, sliced). The kernel decomposes each logical index into
    coordinates via shape, then computes the physical address via strides.
    The result buffer is always contiguous (fresh allocation).
    Structure mirrors inplace_scalar_ops_strided but writes to a separate
    contiguous result buffer instead of modifying A in-place.
    """
    var numels = Int(numels_)
    var rank = Int(rank_)
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
                    # Fast path: vector fits within one row
                    if a_inner_stride == 1:
                        # A elements are consecutive in memory →
                        # SIMD load from strided A, SIMD store to result.
                        var vec_a = A.unsafe_load[width=simd_width](
                            a_base + inner_offset
                        )
                        var vec_result = simd_op[op_code, dtype, simd_width](
                            vec_a,
                            SIMD[dtype, simd_width](scalar),
                            Epsilon[dtype].value(),
                        )
                        result.unsafe_store[width=simd_width](i, vec_result)

                    else:
                        # A elements are strided (a_inner_stride != 1).
                        # Per-lane read from A, write to contiguous result.
                        comptime for lane in range(simd_width):
                            var a_idx = (
                                a_base + (inner_offset + lane) * a_inner_stride
                            )
                            var a = A[unsafe_offset=a_idx]
                            var res = scalar_op[op_code, dtype](
                                a, scalar, Epsilon[dtype].value()
                            )
                            result[unsafe_offset=i + lane] = res

                else:
                    # Slow path: crosses row boundary
                    # Full per-lane decomposition for A.
                    comptime for lane in range(simd_width):
                        var linear_idx = i + lane
                        var rem = linear_idx
                        var a_idx = 0

                        for dim in range(rank - 1, -1, -1):
                            var coord = rem % shape[dim]
                            a_idx += coord * strides[dim]
                            rem //= shape[dim]

                        var a = A[unsafe_offset=a_idx]
                        var res = scalar_op[op_code, dtype](
                            a, scalar, Epsilon[dtype].value()
                        )
                        result[unsafe_offset=linear_idx] = res

            else:
                # Tail: fewer than simd_width elements remain
                for j in range(numels - i):
                    var linear_idx = i + j
                    var rem = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = rem % shape[dim]
                        a_idx += coord * strides[dim]
                        rem //= shape[dim]

                    var a = A[unsafe_offset=a_idx]
                    var res = scalar_op[op_code, dtype](
                        a, scalar, Epsilon[dtype].value()
                    )
                    result[unsafe_offset=linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


def pow_op[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    exponent: Scalar[dtype],
    size_: Int64,
) where dtype.is_floating_point():
    """Elementwise pow kernel — x ** exponent. Float dtypes only.

    Dedicated pow kernel
    POW is separated from scalar_ops because:
    - pow() with float exponent needs a floating-point dtype constraint
    - A single generic kernel guarded by `comptime if dtype.is_floating_point()`
      covers both float32 and float64 — same pattern as float_unary_ops. The
      body-level guard sidesteps declaration-level constraint issues on GPU.
    """
    var size = Int(size_)
    comptime if dtype.is_floating_point():
        var tid = thread_idx.x
        var gtid = tid + block_dim.x * block_idx.x
        var stride = block_dim.x * grid_dim.x

        comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
        var base_idx = gtid * CHUNK_SIZE

        while base_idx < size:
            comptime for item in range(simd_vectors_per_thread):
                var i = base_idx + item * simd_width

                if i + simd_width <= size:
                    var vec_a = A.unsafe_load[width=simd_width](i)
                    var vec_exp = SIMD[dtype, simd_width](exponent)
                    result.unsafe_store[width=simd_width](
                        i, pow(vec_a, vec_exp)
                    )

                elif i < size:
                    for j in range(size - i):
                        result[unsafe_offset=i + j] = pow(
                            A[unsafe_offset=i + j], exponent
                        )

            base_idx += stride * CHUNK_SIZE
    else:
        panic("pow_op requires a floating point dtype")


struct ScalarKernel[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    def launch[
        op_code: Int,
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        scalar: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var numels = A_layout.numel()
        var rank = A_layout.shape.rank()

        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]
        ref A_buffer = A_device_state.device_buffer()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        # Dispatch on contiguity — same pattern as ScalarInplaceKernel.launch.
        if A_layout.is_contiguous():
            # PATH 1: Contiguous A → flat linear indexing (fast SIMD).
            var compiled_func = device_context.compile_function[
                scalar_ops[
                    op_code=op_code,
                    dtype=Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
            ]()

            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                scalar,
                Int64(numels),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        else:
            # PATH 2: Strided A → stride-decomposed indexing.
            var compiled_func = device_context.compile_function[
                scalar_ops_strided[
                    op_code=op_code,
                    dtype=Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
            ]()

            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                scalar,
                A_layout.shape.array(),
                A_layout.strides.array(),
                Int64(numels),
                Int64(rank),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        if sync:
            device_context.synchronize()
        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return (Layout(A_layout.shape), device_state^)

    @staticmethod
    def launch_pow(
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        exponent: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        """
        Dedicated POW launcher — dispatches to the generic pow_op kernel.
        POW only supported for floating point dtypes.
        """
        var numels = A_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()
        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        # Ensure contiguous GPU input
        var contig_state = materialize_contiguous(
            A_device_state, A_layout, sync=sync
        )
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        comptime if Self.dtype.is_floating_point():
            var compiled = device_context.compile_function[
                pow_op[
                    dtype=Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
            ]()
            device_context.enqueue_function(
                compiled,
                result_buffer,
                contig_state.device_buffer(),
                exponent,
                Int64(numels),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        else:
            panic(
                "ScalarKernel: POW only supported for float32 and float64"
            )

        if sync:
            device_context.synchronize()
        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return (Layout(A_layout.shape), device_state^)

    @staticmethod
    def launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
        return elementwise_launch_config(numels, simdwidth)
