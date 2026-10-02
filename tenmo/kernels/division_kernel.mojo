"""Fused division backward GPU kernels + launcher.

Kernels:
  rdiv_scalar_backward:  result[i] = scalar * grad_output[i] / (x[i] * x[i])
  divide_backward:       grad_x[i] = grad_output[i] / y[i]
                         grad_y[i] = grad_output[i] * x[i] / (y[i] * y[i])
"""

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from ..shared.intarray import IntArray
from ..shared.shapes import Shape
from ..shared.strides import Strides
from ..shared.broadcasthelper import ShapeBroadcaster
from ..shared.panic import panic
from .kernel_helpers import elementwise_launch_config
from ..gpu.device import DeviceState


def broadcast_layout(
    layout: Layout, target_shape: Shape
) -> Layout:
    """Expanded-strides broadcast view of `layout` over `target_shape`.

    Mirrors NDBuffer.broadcast_to metadata (stride 0 on broadcast dims).
    """
    if not ShapeBroadcaster.expandable_to(layout.shape, target_shape):
        panic(
            "DivisionKernel.broadcast_layout: cannot expand "
            + String(layout.shape)
            + " to "
            + String(target_shape)
        )

    var own_shape = layout.shape
    var own_rank = own_shape.rank()
    var target_rank = target_shape.rank()
    var extra_dims = target_rank - own_rank

    var new_strides = IntArray.with_capacity(target_rank)
    for _ in range(extra_dims):
        new_strides.append(0)
    for i in range(own_rank):
        var target_i = i + extra_dims
        if own_shape[i] == 1 and target_shape[target_i] > 1:
            new_strides.append(0)
        else:
            new_strides.append(layout.strides[i])

    return Layout(target_shape, Strides(new_strides), layout.offset)


def rdiv_scalar_backward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_output: Pointer[Scalar[dtype], ImmutAnyOrigin],
    divisor: Pointer[Scalar[dtype], ImmutAnyOrigin],
    scalar: Scalar[dtype],
    size_: Int64,
):
    """rdiv_scalar_backward Kernel
    For each element i:
      result[i] = scalar * grad_output[i] / (divisor[i] * divisor[i])
    """
    var size = Int(size_)
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_grad = grad_output.unsafe_load[width=simd_width](i)
                var vec_div = divisor.unsafe_load[width=simd_width](i)
                var result_vec = scalar * vec_grad / (vec_div * vec_div)
                result.unsafe_store[width=simd_width](i, result_vec)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    result[unsafe_offset=idx] = (
                        scalar
                        * grad_output[unsafe_offset=idx]
                        / (divisor[unsafe_offset=idx] * divisor[unsafe_offset=idx])
                    )

        base_idx += stride * CHUNK_SIZE


def divide_backward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    grad_x_result: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_y_result: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_output: Pointer[Scalar[dtype], ImmutAnyOrigin],
    x: Pointer[Scalar[dtype], ImmutAnyOrigin],
    y: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
):
    """divide_backward Kernel
    For each element i:
      grad_x[i] = grad_output[i] / y[i]
      grad_y[i] = grad_output[i] * x[i] / (y[i] * y[i])
    """
    var size = Int(size_)
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_grad = grad_output.unsafe_load[width=simd_width](i)
                var vec_x = x.unsafe_load[width=simd_width](i)
                var vec_y = y.unsafe_load[width=simd_width](i)

                var vec_y_sq = vec_y * vec_y
                var gx = vec_grad / vec_y
                var gy = vec_grad * vec_x / vec_y_sq

                grad_x_result.unsafe_store[width=simd_width](i, gx)
                grad_y_result.unsafe_store[width=simd_width](i, gy)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var g = grad_output[unsafe_offset=idx]
                    var xv = x[unsafe_offset=idx]
                    var yv = y[unsafe_offset=idx]
                    var yv_sq = yv * yv
                    grad_x_result[unsafe_offset=idx] = g / yv
                    grad_y_result[unsafe_offset=idx] = g * xv / yv_sq

        base_idx += stride * CHUNK_SIZE


# Launcher


struct DivisionKernel[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    def launch_rdiv_scalar_backward(
        grad_output_layout: Layout,
        grad_output_device_state: DeviceState[Self.dtype],
        divisor_layout: Layout,
        divisor_device_state: DeviceState[Self.dtype],
        scalar: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        """Fused rdiv_scalar_backward GPU kernel. Returns gradient for divisor.
        """

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
        var contig_div = materialize_contiguous(
            divisor_device_state, divisor_layout
        )

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            rdiv_scalar_backward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_grad.device_buffer(),
            contig_div.device_buffer(),
            scalar,
            Int64(numels),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        # NOTE: special=True not needed here — result_buffer uses Self.dtype (plain ctor works)
        # var result_state = DeviceState[Self.dtype].__init__[True](
        var result_state = DeviceState[Self.dtype](
            result_buffer^, gpu
        )

        return (
            Layout(grad_output_layout.shape),
            result_state^,
        )

    @staticmethod
    def launch_divide_backward(
        grad_output_layout: Layout,
        grad_output_device_state: DeviceState[Self.dtype],
        x_layout: Layout,
        x_device_state: DeviceState[Self.dtype],
        y_layout: Layout,
        y_device_state: DeviceState[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Tuple[Layout, DeviceState[Self.dtype]], Tuple[Layout, DeviceState[Self.dtype]]]:
        """Fused divide_backward GPU kernel. Returns (grad_x, grad_y).

        Broadcasts x and y to match grad_output shape before kernel launch.
        """

        var target_shape = grad_output_layout.shape
        var numels = grad_output_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = grad_output_device_state.get_gpu()
        var device_context = gpu[]

        # Broadcast-expand operands to match grad_output shape.
        # This ensures the GPU kernel accesses all operands with equal flat
        # sizes — required because the kernel uses flat SIMD indexing and
        # does not handle broadcasting internally.
        var contig_grad = materialize_contiguous(
            grad_output_device_state, grad_output_layout
        )
        var x_broadcast = broadcast_layout(x_layout, target_shape)
        var y_broadcast = broadcast_layout(y_layout, target_shape)
        var contig_x = materialize_contiguous(
            x_device_state, x_broadcast
        )
        var contig_y = materialize_contiguous(
            y_device_state, y_broadcast
        )

        var grad_x_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )
        var grad_y_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            divide_backward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            grad_x_buffer,
            grad_y_buffer,
            contig_grad.device_buffer(),
            contig_x.device_buffer(),
            contig_y.device_buffer(),
            Int64(numels),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var grad_x_state = DeviceState[Self.dtype](
            grad_x_buffer^, gpu
        )
        var grad_y_state = DeviceState[Self.dtype](
            grad_y_buffer^, gpu
        )

        var grad_x_pair = (
            Layout(grad_output_layout.shape),
            grad_x_state^,
        )
        var grad_y_pair = (
            Layout(grad_output_layout.shape),
            grad_y_state^,
        )

        return (grad_x_pair, grad_y_pair)

    @staticmethod
    def launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
        return elementwise_launch_config(numels, simdwidth)
