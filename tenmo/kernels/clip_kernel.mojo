"""Fused GPU kernels for clip (forward = clamp, backward = range mask).

Kernels:
  clip_forward:   result[i] = clamp(x[i], min, max)
  clip_backward:  grad_in[i] = grad_out[i] * (min <= x[i] <= max)

Grid-stride launch, mirroring unary_ops_kernel/division_kernel.
"""

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from ..shared.panic import panic
from .kernel_helpers import elementwise_launch_config
from ..gpu.device import DeviceState


def clip_forward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
    min_val: Scalar[dtype],
    max_val: Scalar[dtype],
):
    var size = Int(size_)
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
                var vec_result = vec_a.clamp(
                    SIMD[dtype, simd_width](min_val),
                    SIMD[dtype, simd_width](max_val),
                )
                result.unsafe_store[width=simd_width](i, vec_result)
            elif i < size:
                for j in range(size - i):
                    var val = A[unsafe_offset=i + j]
                    result[unsafe_offset=i + j] = val.clamp(min_val, max_val)

        base_idx += stride * CHUNK_SIZE


def clip_backward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    grad_in: Pointer[Scalar[dtype], MutAnyOrigin],
    parent: Pointer[Scalar[dtype], ImmutAnyOrigin],
    grad_out: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
    min_val: Scalar[dtype],
    max_val: Scalar[dtype],
):
    """grad_in[i] = grad_out[i] where min <= parent[i] <= max else 0."""
    var size = Int(size_)
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_x = parent.unsafe_load[width=simd_width](i)
                var vec_g = grad_out.unsafe_load[width=simd_width](i)
                var in_range = vec_x.ge(
                    SIMD[dtype, simd_width](min_val)
                ) & vec_x.le(SIMD[dtype, simd_width](max_val))
                var vec_result = vec_g * in_range.cast[dtype]()
                grad_in.unsafe_store[width=simd_width](i, vec_result)
            elif i < size:
                for j in range(size - i):
                    var x = parent[unsafe_offset=i + j]
                    var g = grad_out[unsafe_offset=i + j]
                    if x >= min_val and x <= max_val:
                        grad_in[unsafe_offset=i + j] = g
                    else:
                        grad_in[unsafe_offset=i + j] = Scalar[dtype](0)

        base_idx += stride * CHUNK_SIZE


struct ClipKernel[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    def launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
        return elementwise_launch_config(numels, simdwidth)

    @staticmethod
    def launch_forward(
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        min_val: Scalar[Self.dtype],
        max_val: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var numels = A_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()
        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]
        var contig = materialize_contiguous(A_device_state, A_layout)
        var result_buffer = device_context.enqueue_create_buffer[
            Self.dtype
        ](numels)

        var compiled = device_context.compile_function[
            clip_forward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()
        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig.device_buffer(),
            Int64(numels),
            min_val,
            max_val,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return (Layout(A_layout.shape), result_state^)

    @staticmethod
    def launch_backward(
        grad_output_layout: Layout,
        grad_output_device_state: DeviceState[Self.dtype],
        parent_layout: Layout,
        parent_device_state: DeviceState[Self.dtype],
        min_val: Scalar[Self.dtype],
        max_val: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var numels = grad_output_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()
        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = grad_output_device_state.get_gpu()
        var device_context = gpu[]
        var contig_grad = materialize_contiguous(
            grad_output_device_state, grad_output_layout
        )
        var contig_parent = materialize_contiguous(
            parent_device_state, parent_layout
        )
        var result_buffer = device_context.enqueue_create_buffer[
            Self.dtype
        ](numels)

        var compiled = device_context.compile_function[
            clip_backward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()
        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_parent.device_buffer(),
            contig_grad.device_buffer(),
            Int64(numels),
            min_val,
            max_val,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return (Layout(grad_output_layout.shape), result_state^)
