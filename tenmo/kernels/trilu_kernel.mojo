# trilu_kernel.mojo — GPU tril/triu masking kernels (shared implementation)
#
# Both triangular masks use the same fused SIMD kernels; a comptime `upper`
# flag selects the mask:
#   upper=False → tril (col <= row + diagonal)
#   upper=True  → triu (col >= row + diagonal)

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from .kernel_helpers import elementwise_launch_config
from ..gpu.device import DeviceState


def trilu_kernel[
    dtype: DType,
    upper: Bool,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
    M_: Int64,
    N_: Int64,
    diagonal_: Int64,
):
    """Fused tril/triu: result[i] = A[i] if mask(row,col) else 0."""
    var size = Int(size_)
    var M = Int(M_)
    var N = Int(N_)
    var diagonal = Int(diagonal_)
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE
    var batch_stride = M * N

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.unsafe_load[width=simd_width](i)
                var vec_result = SIMD[dtype, simd_width](0)

                for j in range(simd_width):
                    var idx = i + j
                    var within = idx % batch_stride
                    var row = within // N
                    var col = within % N
                    comptime if upper:
                        if col >= row + diagonal:
                            vec_result[j] = vec_a[j]
                    else:
                        if col <= row + diagonal:
                            vec_result[j] = vec_a[j]

                result.unsafe_store[width=simd_width](i, vec_result)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var within = idx % batch_stride
                    var row = within // N
                    var col = within % N
                    comptime if upper:
                        result[unsafe_offset=idx] = A[unsafe_offset=
                            idx
                        ] if col >= row + diagonal else Scalar[dtype](0)
                    else:
                        result[unsafe_offset=idx] = A[unsafe_offset=
                            idx
                        ] if col <= row + diagonal else Scalar[dtype](0)

        base_idx += stride * CHUNK_SIZE


def trilu_backward_kernel[
    dtype: DType,
    upper: Bool,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    grad: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
    M_: Int64,
    N_: Int64,
    diagonal_: Int64,
):
    """Backward: grad_input = grad_output * tril/triu mask. Same mask as forward.
    """
    var size = Int(size_)
    var M = Int(M_)
    var N = Int(N_)
    var diagonal = Int(diagonal_)
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE
    var batch_stride = M * N

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_g = grad.unsafe_load[width=simd_width](i)
                var vec_result = SIMD[dtype, simd_width](0)

                for j in range(simd_width):
                    var idx = i + j
                    var within = idx % batch_stride
                    var row = within // N
                    var col = within % N
                    comptime if upper:
                        if col >= row + diagonal:
                            vec_result[j] = vec_g[j]
                    else:
                        if col <= row + diagonal:
                            vec_result[j] = vec_g[j]

                result.unsafe_store[width=simd_width](i, vec_result)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var within = idx % batch_stride
                    var row = within // N
                    var col = within % N
                    comptime if upper:
                        result[unsafe_offset=idx] = grad[unsafe_offset=
                            idx
                        ] if col >= row + diagonal else Scalar[dtype](0)
                    else:
                        result[unsafe_offset=idx] = grad[unsafe_offset=
                            idx
                        ] if col <= row + diagonal else Scalar[dtype](0)

        base_idx += stride * CHUNK_SIZE


@fieldwise_init
struct _TriluLaunch[dtype: DType](ImplicitlyCopyable):
    comptime datatype: DType = (
        DType.uint8 if Self.dtype == DType.bool else Self.dtype
    )

    @staticmethod
    def _run[
        upper: Bool,
        backward: Bool,
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        diagonal: Int,
        sync: Bool,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var numels = A_layout.numel()
        var shape = A_layout.shape
        var rank = shape.rank()
        var M = shape[rank - 2]
        var N = shape[rank - 1]
        comptime simdwidth = simd_width_of[Self.datatype]()

        var (num_blocks, threads_per_block) = elementwise_launch_config(
            numels, simdwidth
        )
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]
        var contig_state = materialize_contiguous(
            A_device_state, A_layout
        )

        var result_buffer = device_context.enqueue_create_buffer[Self.datatype](
            numels
        )

        comptime if backward:
            var compiled = device_context.compile_function[
                trilu_backward_kernel[
                    Self.datatype, upper, simdwidth, 2 * simdwidth
                ],
            ]()
            device_context.enqueue_function(
                compiled,
                result_buffer,
                contig_state.device_buffer(),
                Int64(numels),
                Int64(M),
                Int64(N),
                Int64(diagonal),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        else:
            var compiled = device_context.compile_function[
                trilu_kernel[Self.datatype, upper, simdwidth, 2 * simdwidth],
            ]()
            device_context.enqueue_function(
                compiled,
                result_buffer,
                contig_state.device_buffer(),
                Int64(numels),
                Int64(M),
                Int64(N),
                Int64(diagonal),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        if sync:
            device_context.synchronize()

        # bool-as-uint8: skip sub-buffer creation (buffer dtype != Self.dtype when bool)
        var result_state = DeviceState[Self.dtype].__init__[True](
            result_buffer^, gpu
        )
        return (
            Layout(A_layout.shape),
            result_state^,
        )


@fieldwise_init
struct TrilKernel[dtype: DType](ImplicitlyCopyable):
    """Lower-triangular masking kernel (col <= row + diagonal)."""

    @staticmethod
    def launch(
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        diagonal: Int,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        return _TriluLaunch[Self.dtype]._run[False, False](
            A_layout, A_device_state, diagonal, sync
        )

    @staticmethod
    def launch_backward(
        grad_layout: Layout,
        grad_device_state: DeviceState[Self.dtype],
        diagonal: Int,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        return _TriluLaunch[Self.dtype]._run[False, True](
            grad_layout, grad_device_state, diagonal, sync
        )


@fieldwise_init
struct TriuKernel[dtype: DType](ImplicitlyCopyable):
    """Upper-triangular masking kernel (col >= row + diagonal)."""

    @staticmethod
    def launch(
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        diagonal: Int,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        return _TriluLaunch[Self.dtype]._run[True, False](
            A_layout, A_device_state, diagonal, sync
        )

    @staticmethod
    def launch_backward(
        grad_layout: Layout,
        grad_device_state: DeviceState[Self.dtype],
        diagonal: Int,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        return _TriluLaunch[Self.dtype]._run[True, True](
            grad_layout, grad_device_state, diagonal, sync
        )
