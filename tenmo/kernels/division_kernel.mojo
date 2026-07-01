"""Fused division backward GPU kernels + launcher.

Kernels:
  rdiv_scalar_backward:  result[i] = scalar * grad_output[i] / (x[i] * x[i])
  divide_backward:       grad_x[i] = grad_output[i] / y[i]
                         grad_y[i] = grad_output[i] * x[i] / (y[i] * y[i])
"""

from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from tenmo.ndbuffer import NDBuffer
from . import elementwise_launch_config
from tenmo.device import DeviceState


# ── rdiv_scalar_backward Kernel ────────────────────────────────────────────────
# For each element i:
#   result[i] = scalar * grad_output[i] / (divisor[i] * divisor[i])


def rdiv_scalar_backward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    divisor: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    scalar: Scalar[dtype],
    size: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_grad = grad_output.load[width=simd_width](i)
                var vec_div = divisor.load[width=simd_width](i)
                var result_vec = scalar * vec_grad / (vec_div * vec_div)
                result.store[width=simd_width](i, result_vec)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    result[idx] = (
                        scalar
                        * grad_output[idx]
                        / (divisor[idx] * divisor[idx])
                    )

        base_idx += stride * CHUNK_SIZE


# ── divide_backward Kernel ─────────────────────────────────────────────────────
# For each element i:
#   grad_x[i] = grad_output[i] / y[i]
#   grad_y[i] = grad_output[i] * x[i] / (y[i] * y[i])


def divide_backward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    grad_x_result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_y_result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    y: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_grad = grad_output.load[width=simd_width](i)
                var vec_x = x.load[width=simd_width](i)
                var vec_y = y.load[width=simd_width](i)

                var vec_y_sq = vec_y * vec_y
                var gx = vec_grad / vec_y
                var gy = vec_grad * vec_x / vec_y_sq

                grad_x_result.store[width=simd_width](i, gx)
                grad_y_result.store[width=simd_width](i, gy)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var g = grad_output[idx]
                    var xv = x[idx]
                    var yv = y[idx]
                    var yv_sq = yv * yv
                    grad_x_result[idx] = g / yv
                    grad_y_result[idx] = g * xv / yv_sq

        base_idx += stride * CHUNK_SIZE


# ── Launcher ──────────────────────────────────────────────────────────────────


struct DivisionKernel[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def launch_rdiv_scalar_backward(
        grad_output: NDBuffer[Self.dtype],
        divisor: NDBuffer[Self.dtype],
        scalar: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> NDBuffer[Self.dtype]:
        """Fused rdiv_scalar_backward GPU kernel. Returns gradient for divisor.
        """
        debug_assert(grad_output.is_on_gpu())
        debug_assert(divisor.is_on_gpu())

        var numels = grad_output.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = grad_output.device_state.value()
        var device_context = device_state.gpu[]

        var contig_grad = grad_output.contiguous_device_state()
        var contig_div = divisor.contiguous_device_state()

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
            numels,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        # var result_state = DeviceState[Self.dtype].__init__[True](
        var result_state = DeviceState[Self.dtype](
            result_buffer^, device_state.gpu
        )

        var result_ndb = NDBuffer[Self.dtype].with_device_state(
            result_state^, grad_output.shape
        )

        return result_ndb^

    @staticmethod
    def launch_divide_backward(
        grad_output: NDBuffer[Self.dtype],
        x: NDBuffer[Self.dtype],
        y: NDBuffer[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        """Fused divide_backward GPU kernel. Returns (grad_x, grad_y).

        Broadcasts x and y to match grad_output shape before kernel launch.
        """
        debug_assert(grad_output.is_on_gpu())
        debug_assert(x.is_on_gpu())
        debug_assert(y.is_on_gpu())

        var target_shape = grad_output.shape
        var numels = grad_output.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = grad_output.device_state.value()
        var device_context = device_state.gpu[]

        # Broadcast-expand operands to match grad_output shape.
        # This ensures the GPU kernel accesses all operands with equal flat
        # sizes — required because the kernel uses flat SIMD indexing and
        # does not handle broadcasting internally.
        var contig_grad = grad_output.contiguous_device_state()
        var bx = (
            x.broadcast_to(target_shape) if x.shape
            != target_shape else x.copy()
        )
        var by = (
            y.broadcast_to(target_shape) if y.shape
            != target_shape else y.copy()
        )
        var contig_x = bx.contiguous_device_state()
        var contig_y = by.contiguous_device_state()

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
            numels,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var grad_x_state = DeviceState[Self.dtype](
            grad_x_buffer^, device_state.gpu
        )
        var grad_y_state = DeviceState[Self.dtype](
            grad_y_buffer^, device_state.gpu
        )

        var grad_x_ndb = NDBuffer[Self.dtype].with_device_state(
            grad_x_state^, grad_output.shape
        )
        var grad_y_ndb = NDBuffer[Self.dtype].with_device_state(
            grad_y_state^, grad_output.shape
        )

        return (grad_x_ndb^, grad_y_ndb^)

    @staticmethod
    def launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
        return elementwise_launch_config(numels, simdwidth)
