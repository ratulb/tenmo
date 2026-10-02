"""Fused BCE / BCEWithLogits GPU kernels + launcher.

Forward kernel computes per-element loss AND sigmoid (for backward)
in a single GPU pass — eliminating 10+ separate tensor ops.

Gradient formulas (after mean reduction with N elements):
  BCEWithLogits: d(loss)/d(logits_i) = (sigmoid(logits_i) - target_i) / N
  BCELoss:       d(loss)/d(p_i) = -(target_i/clip(p_i) - (1-target_i)/(1-clip(p_i))) / N
"""

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from max.gpu import barrier
from std.sys import simd_width_of
from std.math import exp, log
from std.atomic import Atomic
from std.memory import stack_allocation, AddressSpace

from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from .kernel_helpers import elementwise_launch_config
from ..gpu.device import DeviceState
from ..shared.shapes import Shape


def bce_with_logits_forward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    loss_result: Pointer[Scalar[dtype], MutAnyOrigin],
    sigmoid_result: Pointer[Scalar[dtype], MutAnyOrigin],
    logits: Pointer[Scalar[dtype], ImmutAnyOrigin],
    target: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
    epsilon: Scalar[dtype],
) where dtype.is_floating_point():
    """BCEWithLogits Forward Kernel
    Computes loss_per_element AND sigmoid(logits) in one pass.
    For each element i:
      sig_i = 1 / (1 + exp(-logits_i))
      safe  = clip(sig_i, eps, 1-eps)
      loss  = -[y_i * log(safe) + (1-y_i) * log(1-safe)]
    """
    var size = Int(size_)
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_logits = logits.unsafe_load[width=simd_width](i)
                var vec_target = target.unsafe_load[width=simd_width](i)

                var sig = SIMD[dtype, simd_width](1) / (
                    SIMD[dtype, simd_width](1) + exp(-vec_logits)
                )
                var safe = sig.clamp(
                    SIMD[dtype, simd_width](epsilon),
                    SIMD[dtype, simd_width](1)
                    - SIMD[dtype, simd_width](epsilon),
                )
                var loss = -(
                    vec_target * log(safe)
                    + (SIMD[dtype, simd_width](1) - vec_target)
                    * log(SIMD[dtype, simd_width](1) - safe)
                )

                loss_result.unsafe_store[width=simd_width](i, loss)
                sigmoid_result.unsafe_store[width=simd_width](i, sig)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var x = logits[unsafe_offset=idx]
                    var y = target[unsafe_offset=idx]
                    var s = Scalar[dtype](1) / (Scalar[dtype](1) + exp(-x))
                    var safe = s.clamp(
                        Scalar[dtype](epsilon),
                        Scalar[dtype](1) - Scalar[dtype](epsilon),
                    )
                    loss_result[unsafe_offset=idx] = -(
                        y * log(safe)
                        + (Scalar[dtype](1) - y) * log(Scalar[dtype](1) - safe)
                    )
                    sigmoid_result[unsafe_offset=idx] = s

        base_idx += stride * CHUNK_SIZE


# BCEWithLogits Forward Reduce Kernel (mean/sum)
# Same as above but accumulates block-level sum via shared-mem tree-reduce,
# then thread 0 does Atomic.fetch_add to a 1-element scalar output.
# After kernel sync, host divides scalar by N if is_mean.

comptime MAX_BLOCK_SIZE: Int = 256


def bce_with_logits_forward_reduce_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    sigmoid_result: Pointer[Scalar[dtype], MutAnyOrigin],
    scalar_loss: Pointer[Scalar[dtype], MutAnyOrigin],
    logits: Pointer[Scalar[dtype], ImmutAnyOrigin],
    target: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
    epsilon: Scalar[dtype],
) where dtype.is_floating_point():
    var size = Int(size_)
    var cache_index = thread_idx.x
    var gtid = cache_index + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    var partial_sum = Scalar[dtype](0)
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_logits = logits.unsafe_load[width=simd_width](i)
                var vec_target = target.unsafe_load[width=simd_width](i)

                var sig = SIMD[dtype, simd_width](1) / (
                    SIMD[dtype, simd_width](1) + exp(-vec_logits)
                )
                var safe = sig.clamp(
                    SIMD[dtype, simd_width](epsilon),
                    SIMD[dtype, simd_width](1)
                    - SIMD[dtype, simd_width](epsilon),
                )
                var loss = -(
                    vec_target * log(safe)
                    + (SIMD[dtype, simd_width](1) - vec_target)
                    * log(SIMD[dtype, simd_width](1) - safe)
                )

                sigmoid_result.unsafe_store[width=simd_width](i, sig)
                partial_sum += loss.reduce_add()

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var x = logits[unsafe_offset=idx]
                    var y = target[unsafe_offset=idx]
                    var s = Scalar[dtype](1) / (Scalar[dtype](1) + exp(-x))
                    var safe = s.clamp(
                        Scalar[dtype](epsilon),
                        Scalar[dtype](1) - Scalar[dtype](epsilon),
                    )
                    sigmoid_result[unsafe_offset=idx] = s
                    partial_sum += -(
                        y * log(safe)
                        + (Scalar[dtype](1) - y) * log(Scalar[dtype](1) - safe)
                    )

        base_idx += stride * CHUNK_SIZE

    var block_shared = stack_allocation[
        MAX_BLOCK_SIZE,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()
    block_shared[unsafe_offset=cache_index] = partial_sum
    barrier()

    var sm_stride = block_dim.x // 2
    while sm_stride > 0:
        if cache_index < sm_stride:
            block_shared[unsafe_offset=cache_index] += block_shared[unsafe_offset=cache_index + sm_stride]
        barrier()
        sm_stride //= 2

    if cache_index == 0:
        _ = Atomic.fetch_add(scalar_loss, block_shared[unsafe_offset=0])


def bce_forward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    loss_result: Pointer[Scalar[dtype], MutAnyOrigin],
    safe_result: Pointer[Scalar[dtype], MutAnyOrigin],
    pred: Pointer[Scalar[dtype], ImmutAnyOrigin],
    target: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
    epsilon: Scalar[dtype],
) where dtype.is_floating_point():
    """BCELoss Forward Kernel (probabilities input, not logits)
    For each element i:
      safe  = clip(p_i, eps, 1-eps)
      loss  = -[y_i * log(safe) + (1-y_i) * log(1-safe)]
    """
    var size = Int(size_)
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_pred = pred.unsafe_load[width=simd_width](i)
                var vec_target = target.unsafe_load[width=simd_width](i)

                var safe = vec_pred.clamp(
                    SIMD[dtype, simd_width](epsilon),
                    SIMD[dtype, simd_width](1)
                    - SIMD[dtype, simd_width](epsilon),
                )
                var loss = -(
                    vec_target * log(safe)
                    + (SIMD[dtype, simd_width](1) - vec_target)
                    * log(SIMD[dtype, simd_width](1) - safe)
                )

                loss_result.unsafe_store[width=simd_width](i, loss)
                safe_result.unsafe_store[width=simd_width](i, safe)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var p = pred[unsafe_offset=idx]
                    var y = target[unsafe_offset=idx]
                    var safe = p.clamp(
                        Scalar[dtype](epsilon),
                        Scalar[dtype](1) - Scalar[dtype](epsilon),
                    )
                    loss_result[unsafe_offset=idx] = -(
                        y * log(safe)
                        + (Scalar[dtype](1) - y) * log(Scalar[dtype](1) - safe)
                    )
                    safe_result[unsafe_offset=idx] = safe

        base_idx += stride * CHUNK_SIZE


def bce_forward_reduce_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    safe_result: Pointer[Scalar[dtype], MutAnyOrigin],
    scalar_loss: Pointer[Scalar[dtype], MutAnyOrigin],
    pred: Pointer[Scalar[dtype], ImmutAnyOrigin],
    target: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
    epsilon: Scalar[dtype],
) where dtype.is_floating_point():
    """BCELoss Forward Reduce Kernel (mean/sum)
    Same as above but accumulates block-level sum for mean/sum reduction.
    """
    var size = Int(size_)
    var cache_index = thread_idx.x
    var gtid = cache_index + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    var partial_sum = Scalar[dtype](0)
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_pred = pred.unsafe_load[width=simd_width](i)
                var vec_target = target.unsafe_load[width=simd_width](i)

                var safe = vec_pred.clamp(
                    SIMD[dtype, simd_width](epsilon),
                    SIMD[dtype, simd_width](1)
                    - SIMD[dtype, simd_width](epsilon),
                )
                var loss = -(
                    vec_target * log(safe)
                    + (SIMD[dtype, simd_width](1) - vec_target)
                    * log(SIMD[dtype, simd_width](1) - safe)
                )

                safe_result.unsafe_store[width=simd_width](i, safe)
                partial_sum += loss.reduce_add()

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var p = pred[unsafe_offset=idx]
                    var y = target[unsafe_offset=idx]
                    var safe = p.clamp(
                        Scalar[dtype](epsilon),
                        Scalar[dtype](1) - Scalar[dtype](epsilon),
                    )
                    safe_result[unsafe_offset=idx] = safe
                    partial_sum += -(
                        y * log(safe)
                        + (Scalar[dtype](1) - y) * log(Scalar[dtype](1) - safe)
                    )

        base_idx += stride * CHUNK_SIZE

    var block_shared = stack_allocation[
        MAX_BLOCK_SIZE,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()
    block_shared[unsafe_offset=cache_index] = partial_sum
    barrier()

    # var sm_stride = UInt(block_dim.x // 2)
    var sm_stride = block_dim.x // 2
    while sm_stride > 0:
        if cache_index < sm_stride:
            block_shared[unsafe_offset=cache_index] += block_shared[unsafe_offset=cache_index + sm_stride]
        barrier()
        sm_stride //= 2

    if cache_index == 0:
        _ = Atomic.fetch_add(scalar_loss, block_shared[unsafe_offset=0])


def bce_with_logits_backward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    grad_result: Pointer[Scalar[dtype], MutAnyOrigin],
    sigmoid: Pointer[Scalar[dtype], ImmutAnyOrigin],
    target: Pointer[Scalar[dtype], ImmutAnyOrigin],
    grad_output: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
):
    """BCEWithLogits Backward Kernel
    For each element i:
      grad[i] = (sigmoid[i] - target[i]) * grad_output[i]
    """
    var size = Int(size_)
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_sig = sigmoid.unsafe_load[width=simd_width](i)
                var vec_target = target.unsafe_load[width=simd_width](i)
                var vec_grad = grad_output.unsafe_load[width=simd_width](i)
                var grad = (vec_sig - vec_target) * vec_grad
                grad_result.unsafe_store[width=simd_width](i, grad)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    grad_result[unsafe_offset=idx] = (
                        sigmoid[unsafe_offset=idx] - target[unsafe_offset=idx]
                    ) * grad_output[unsafe_offset=idx]

        base_idx += stride * CHUNK_SIZE


def bce_with_logits_backward_scaled_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    grad_result: Pointer[Scalar[dtype], MutAnyOrigin],
    sigmoid: Pointer[Scalar[dtype], ImmutAnyOrigin],
    target: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
    scalar_grad: Scalar[dtype],
):
    """BCEWithLogits Backward Scaled Kernel
    Takes scalar_grad (not per-element grad_output buffer).
    """
    var size = Int(size_)
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE
    var s_grad = SIMD[dtype, simd_width](scalar_grad)

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_sig = sigmoid.unsafe_load[width=simd_width](i)
                var vec_target = target.unsafe_load[width=simd_width](i)
                var grad = (vec_sig - vec_target) * s_grad
                grad_result.unsafe_store[width=simd_width](i, grad)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    grad_result[unsafe_offset=idx] = (
                        sigmoid[unsafe_offset=idx] - target[unsafe_offset=idx]
                    ) * scalar_grad

        base_idx += stride * CHUNK_SIZE


def bce_backward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    grad_result: Pointer[Scalar[dtype], MutAnyOrigin],
    safe: Pointer[Scalar[dtype], ImmutAnyOrigin],
    target: Pointer[Scalar[dtype], ImmutAnyOrigin],
    grad_output: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
):
    """BCELoss Backward Kernel
    For each element i:
      grad[i] = -(target[i]/safe[i] - (1-target[i])/(1-safe[i])) * grad_output[i]
    """
    var size = Int(size_)
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_safe = safe.unsafe_load[width=simd_width](i)
                var vec_target = target.unsafe_load[width=simd_width](i)
                var vec_grad = grad_output.unsafe_load[width=simd_width](i)
                var one = SIMD[dtype, simd_width](1.0)
                var grad = (
                    -(
                        vec_target / vec_safe
                        - (one - vec_target) / (one - vec_safe)
                    )
                    * vec_grad
                )
                grad_result.unsafe_store[width=simd_width](i, grad)

            elif i < size:
                var one = Scalar[dtype](1.0)
                for j in range(size - i):
                    var idx = i + j
                    var s = safe[unsafe_offset=idx]
                    var t = target[unsafe_offset=idx]
                    var g = grad_output[unsafe_offset=idx]
                    grad_result[unsafe_offset=idx] = -(t / s - (one - t) / (one - s)) * g

        base_idx += stride * CHUNK_SIZE


def bce_backward_scaled_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    grad_result: Pointer[Scalar[dtype], MutAnyOrigin],
    safe: Pointer[Scalar[dtype], ImmutAnyOrigin],
    target: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
    scalar_grad: Scalar[dtype],
):
    """BCELoss Backward Scaled Kernel
    Takes scalar_grad (not per-element grad_output buffer).
    """
    var size = Int(size_)
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE
    var one_simd = SIMD[dtype, simd_width](1.0)
    var s_grad = SIMD[dtype, simd_width](scalar_grad)

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_safe = safe.unsafe_load[width=simd_width](i)
                var vec_target = target.unsafe_load[width=simd_width](i)
                var grad = (
                    -(
                        vec_target / vec_safe
                        - (one_simd - vec_target) / (one_simd - vec_safe)
                    )
                    * s_grad
                )
                grad_result.unsafe_store[width=simd_width](i, grad)

            elif i < size:
                var one_s = Scalar[dtype](1.0)
                for j in range(size - i):
                    var idx = i + j
                    var si = safe[unsafe_offset=idx]
                    var ti = target[unsafe_offset=idx]
                    grad_result[unsafe_offset=idx] = (
                        -(ti / si - (one_s - ti) / (one_s - si)) * scalar_grad
                    )

        base_idx += stride * CHUNK_SIZE


# Launcher


struct BceKernel[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    def launch_forward_with_logits(
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        target_layout: Layout,
        target_device_state: DeviceState[Self.dtype],
        epsilon: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[
        Tuple[Layout, DeviceState[Self.dtype]], Tuple[Layout, DeviceState[Self.dtype]]
    ] where Self.dtype.is_floating_point():
        """Fused BCEWithLogits forward. Returns ((loss_layout, loss_storage), (sigmoid_layout, sigmoid_device_state))."""

        var numels = A_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        var contig_logits = materialize_contiguous(A_device_state, A_layout)
        var contig_target = materialize_contiguous(
            target_device_state, target_layout
        )

        var loss_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )
        var sigmoid_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_with_logits_forward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            loss_buffer,
            sigmoid_buffer,
            contig_logits.device_buffer(),
            contig_target.device_buffer(),
            Int64(numels),
            epsilon,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var loss_state = DeviceState[Self.dtype](loss_buffer^, gpu)
        var sigmoid_state = DeviceState[Self.dtype](
            sigmoid_buffer^, gpu
        )

        return (
            (
                Layout(A_layout.shape),
                loss_state^,
            ),
            (
                Layout(A_layout.shape),
                sigmoid_state^,
            ),
        )

    @staticmethod
    def launch_forward_with_logits_reduce(
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        target_layout: Layout,
        target_device_state: DeviceState[Self.dtype],
        epsilon: Scalar[Self.dtype],
        is_mean: Bool,
        sync: Bool = False,
    ) raises -> Tuple[
        Tuple[Layout, DeviceState[Self.dtype]], Tuple[Layout, DeviceState[Self.dtype]]
    ] where Self.dtype.is_floating_point():
        """Fused BCEWithLogits forward with mean/sum reduction.

        Returns ((scalar_loss_layout, scalar_loss_storage), (sigmoid_layout, sigmoid_device_state))
        where scalar_loss is 1-element.
        If is_mean, scalar_loss is divided by N on host after kernel."""

        var numels = A_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        var contig_logits = materialize_contiguous(A_device_state, A_layout)
        var contig_target = materialize_contiguous(
            target_device_state, target_layout
        )

        var scalar_buffer = device_context.enqueue_create_buffer[Self.dtype](1)
        scalar_buffer.enqueue_fill(0)

        var sigmoid_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_with_logits_forward_reduce_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            sigmoid_buffer,
            scalar_buffer,
            contig_logits.device_buffer(),
            contig_target.device_buffer(),
            Int64(numels),
            epsilon,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var scalar_state = DeviceState[Self.dtype](
            scalar_buffer^, gpu
        )

        if is_mean:
            var divisor = Scalar[DeviceState[Self.dtype].datatype](numels)
            with scalar_state.buffer.map_to_host() as host_buff:
                host_buff[0] = host_buff[0] / divisor

        var sigmoid_state = DeviceState[Self.dtype](
            sigmoid_buffer^, gpu
        )

        return (
            (
                Layout(Shape()),
                scalar_state^,
            ),
            (
                Layout(A_layout.shape),
                sigmoid_state^,
            ),
        )

    @staticmethod
    def launch_forward(
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        target_layout: Layout,
        target_device_state: DeviceState[Self.dtype],
        epsilon: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[
        Tuple[Layout, DeviceState[Self.dtype]], Tuple[Layout, DeviceState[Self.dtype]]
    ] where Self.dtype.is_floating_point():
        """Fused BCELoss forward (probabilities input). Returns ((loss_layout, loss_storage), (clipped_pred_layout, clipped_pred_storage)).
        """

        var numels = A_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        var contig_pred = materialize_contiguous(A_device_state, A_layout)
        var contig_target = materialize_contiguous(
            target_device_state, target_layout
        )

        var loss_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )
        var safe_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_forward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            loss_buffer,
            safe_buffer,
            contig_pred.device_buffer(),
            contig_target.device_buffer(),
            Int64(numels),
            epsilon,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var loss_state = DeviceState[Self.dtype](loss_buffer^, gpu)
        var safe_state = DeviceState[Self.dtype](safe_buffer^, gpu)

        return (
            (
                Layout(A_layout.shape),
                loss_state^,
            ),
            (
                Layout(A_layout.shape),
                safe_state^,
            ),
        )

    @staticmethod
    def launch_forward_reduce(
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        target_layout: Layout,
        target_device_state: DeviceState[Self.dtype],
        epsilon: Scalar[Self.dtype],
        is_mean: Bool,
        sync: Bool = False,
    ) raises -> Tuple[
        Tuple[Layout, DeviceState[Self.dtype]], Tuple[Layout, DeviceState[Self.dtype]]
    ] where Self.dtype.is_floating_point():
        """Fused BCELoss forward with mean/sum reduction (probabilities input).

        Returns ((scalar_loss_layout, scalar_loss_storage), (clipped_pred_layout, clipped_pred_storage))
        where scalar_loss is 1-element.
        If is_mean, scalar_loss is divided by N on host after kernel."""

        var numels = A_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        var contig_pred = materialize_contiguous(A_device_state, A_layout)
        var contig_target = materialize_contiguous(
            target_device_state, target_layout
        )

        var scalar_buffer = device_context.enqueue_create_buffer[Self.dtype](1)
        scalar_buffer.enqueue_fill(0)

        var safe_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_forward_reduce_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            safe_buffer,
            scalar_buffer,
            contig_pred.device_buffer(),
            contig_target.device_buffer(),
            Int64(numels),
            epsilon,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var scalar_state = DeviceState[Self.dtype](
            scalar_buffer^, gpu
        )

        if is_mean:
            var divisor = Scalar[DeviceState[Self.dtype].datatype](numels)
            with scalar_state.buffer.map_to_host() as host_ptr:
                host_ptr[0] = host_ptr[0] / divisor

        var safe_state = DeviceState[Self.dtype](safe_buffer^, gpu)

        return (
            (
                Layout(Shape()),
                scalar_state^,
            ),
            (
                Layout(A_layout.shape),
                safe_state^,
            ),
        )

    @staticmethod
    def launch_bce_with_logits_backward(
        sigmoid_layout: Layout,
        sigmoid_device_state: DeviceState[Self.dtype],
        target_layout: Layout,
        target_device_state: DeviceState[Self.dtype],
        grad_output_layout: Layout,
        grad_output_device_state: DeviceState[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        """Fused BCEWithLogits backward. Returns gradient for logits."""

        var numels = sigmoid_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = sigmoid_device_state.get_gpu()
        var device_context = gpu[]

        var contig_sig = materialize_contiguous(
            sigmoid_device_state, sigmoid_layout
        )
        var contig_target = materialize_contiguous(
            target_device_state, target_layout
        )
        var contig_grad = materialize_contiguous(
            grad_output_device_state, grad_output_layout
        )

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_with_logits_backward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_sig.device_buffer(),
            contig_target.device_buffer(),
            contig_grad.device_buffer(),
            Int64(numels),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype](
            result_buffer^, gpu
        )

        return (
            Layout(sigmoid_layout.shape),
            result_state^,
        )

    @staticmethod
    def launch_bce_with_logits_backward_scaled(
        sigmoid_layout: Layout,
        sigmoid_device_state: DeviceState[Self.dtype],
        target_layout: Layout,
        target_device_state: DeviceState[Self.dtype],
        scalar_grad: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        """Fused BCEWithLogits backward with scalar gradient. Returns gradient for logits.
        """

        var numels = sigmoid_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = sigmoid_device_state.get_gpu()
        var device_context = gpu[]

        var contig_sig = materialize_contiguous(
            sigmoid_device_state, sigmoid_layout
        )
        var contig_target = materialize_contiguous(
            target_device_state, target_layout
        )

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_with_logits_backward_scaled_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_sig.device_buffer(),
            contig_target.device_buffer(),
            Int64(numels),
            scalar_grad,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype](
            result_buffer^, gpu
        )

        return (
            Layout(sigmoid_layout.shape),
            result_state^,
        )

    @staticmethod
    def launch_bce_backward(
        safe_layout: Layout,
        safe_device_state: DeviceState[Self.dtype],
        target_layout: Layout,
        target_device_state: DeviceState[Self.dtype],
        grad_output_layout: Layout,
        grad_output_device_state: DeviceState[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        """Fused BCELoss backward. Returns gradient for pred."""

        var numels = safe_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = safe_device_state.get_gpu()
        var device_context = gpu[]

        var contig_safe = materialize_contiguous(
            safe_device_state, safe_layout
        )
        var contig_target = materialize_contiguous(
            target_device_state, target_layout
        )
        var contig_grad = materialize_contiguous(
            grad_output_device_state, grad_output_layout
        )

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_backward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_safe.device_buffer(),
            contig_target.device_buffer(),
            contig_grad.device_buffer(),
            Int64(numels),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype](
            result_buffer^, gpu
        )

        return (
            Layout(safe_layout.shape),
            result_state^,
        )

    @staticmethod
    def launch_bce_backward_scaled(
        safe_layout: Layout,
        safe_device_state: DeviceState[Self.dtype],
        target_layout: Layout,
        target_device_state: DeviceState[Self.dtype],
        scalar_grad: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        """Fused BCELoss backward with scalar gradient. Returns gradient for pred.
        """

        var numels = safe_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = safe_device_state.get_gpu()
        var device_context = gpu[]

        var contig_safe = materialize_contiguous(
            safe_device_state, safe_layout
        )
        var contig_target = materialize_contiguous(
            target_device_state, target_layout
        )

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_backward_scaled_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_safe.device_buffer(),
            contig_target.device_buffer(),
            Int64(numels),
            scalar_grad,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype](
            result_buffer^, gpu
        )

        return (
            Layout(safe_layout.shape),
            result_state^,
        )

    @staticmethod
    def launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
        return elementwise_launch_config(numels, simdwidth)
