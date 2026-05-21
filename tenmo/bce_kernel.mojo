"""Fused BCE / BCEWithLogits GPU kernels + launcher.

Forward kernel computes per-element loss AND sigmoid (for backward)
in a single GPU pass — eliminating 10+ separate tensor ops.

Gradient formulas (after mean reduction with N elements):
  BCEWithLogits: d(loss)/d(logits_i) = (sigmoid(logits_i) - target_i) / N
  BCELoss:       d(loss)/d(p_i) = -(target_i/clip(p_i) - (1-target_i)/(1-clip(p_i))) / N
"""

from std.gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from std.sys import simd_width_of
from std.math import exp, log
from std.os.atomic import Atomic, Consistency
from std.memory import stack_allocation, AddressSpace

from tenmo.ndbuffer import NDBuffer
from tenmo.device import DeviceState
from tenmo.common_utils import panic
from tenmo.shapes import Shape


# ── BCEWithLogits Forward Kernel ─────────────────────────────────────────────
# Computes loss_per_element AND sigmoid(logits) in one pass.
#
# For each element i:
#   sig_i = 1 / (1 + exp(-logits_i))
#   safe  = clip(sig_i, eps, 1-eps)
#   loss  = -[y_i * log(safe) + (1-y_i) * log(1-safe)]


def bce_with_logits_forward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    loss_result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    sigmoid_result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    logits: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    target: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
    epsilon: Scalar[dtype],
) where dtype.is_floating_point():
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_logits = logits.load[width=simd_width](i)
                var vec_target = target.load[width=simd_width](i)

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

                loss_result.store[width=simd_width](i, loss)
                sigmoid_result.store[width=simd_width](i, sig)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var x = logits[idx]
                    var y = target[idx]
                    var s = Scalar[dtype](1) / (Scalar[dtype](1) + exp(-x))
                    var safe = s.clamp(
                        Scalar[dtype](epsilon),
                        Scalar[dtype](1) - Scalar[dtype](epsilon),
                    )
                    loss_result[idx] = -(
                        y * log(safe)
                        + (Scalar[dtype](1) - y) * log(Scalar[dtype](1) - safe)
                    )
                    sigmoid_result[idx] = s

        base_idx += stride * CHUNK_SIZE


# ── BCEWithLogits Forward Reduce Kernel (mean/sum) ────────────────────────────
# Same as above but accumulates block-level sum via shared-mem tree-reduce,
# then thread 0 does Atomic.fetch_add to a 1-element scalar output.
# After kernel sync, host divides scalar by N if is_mean.

comptime MAX_BLOCK_SIZE: Int = 256


def bce_with_logits_forward_reduce_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    sigmoid_result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    scalar_loss: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    logits: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    target: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
    epsilon: Scalar[dtype],
) where dtype.is_floating_point():
    var cache_index = thread_idx.x
    var gtid = Int(cache_index + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    var partial_sum = Scalar[dtype](0)
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_logits = logits.load[width=simd_width](i)
                var vec_target = target.load[width=simd_width](i)

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

                sigmoid_result.store[width=simd_width](i, sig)
                partial_sum += loss.reduce_add()

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var x = logits[idx]
                    var y = target[idx]
                    var s = Scalar[dtype](1) / (Scalar[dtype](1) + exp(-x))
                    var safe = s.clamp(
                        Scalar[dtype](epsilon),
                        Scalar[dtype](1) - Scalar[dtype](epsilon),
                    )
                    sigmoid_result[idx] = s
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
    block_shared[cache_index] = partial_sum
    barrier()

    var sm_stride = UInt(block_dim.x // 2)
    while sm_stride > 0:
        if cache_index < sm_stride:
            block_shared[cache_index] += block_shared[cache_index + sm_stride]
        barrier()
        sm_stride //= 2

    if cache_index == 0:
        _ = Atomic.fetch_add(scalar_loss, block_shared[0])


# ── BCELoss Forward Kernel (probabilities input, not logits) ─────────────────
# For each element i:
#   safe  = clip(p_i, eps, 1-eps)
#   loss  = -[y_i * log(safe) + (1-y_i) * log(1-safe)]


def bce_forward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    loss_result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    safe_result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    pred: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    target: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
    epsilon: Scalar[dtype],
) where dtype.is_floating_point():
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_pred = pred.load[width=simd_width](i)
                var vec_target = target.load[width=simd_width](i)

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

                loss_result.store[width=simd_width](i, loss)
                safe_result.store[width=simd_width](i, safe)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var p = pred[idx]
                    var y = target[idx]
                    var safe = p.clamp(
                        Scalar[dtype](epsilon),
                        Scalar[dtype](1) - Scalar[dtype](epsilon),
                    )
                    loss_result[idx] = -(
                        y * log(safe)
                        + (Scalar[dtype](1) - y) * log(Scalar[dtype](1) - safe)
                    )
                    safe_result[idx] = safe

        base_idx += stride * CHUNK_SIZE


# ── BCELoss Forward Reduce Kernel (mean/sum) ──────────────────────────────────
# Same as above but accumulates block-level sum for mean/sum reduction.


def bce_forward_reduce_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    safe_result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    scalar_loss: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    pred: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    target: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
    epsilon: Scalar[dtype],
) where dtype.is_floating_point():
    var cache_index = thread_idx.x
    var gtid = Int(cache_index + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    var partial_sum = Scalar[dtype](0)
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_pred = pred.load[width=simd_width](i)
                var vec_target = target.load[width=simd_width](i)

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

                safe_result.store[width=simd_width](i, safe)
                partial_sum += loss.reduce_add()

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var p = pred[idx]
                    var y = target[idx]
                    var safe = p.clamp(
                        Scalar[dtype](epsilon),
                        Scalar[dtype](1) - Scalar[dtype](epsilon),
                    )
                    safe_result[idx] = safe
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
    block_shared[cache_index] = partial_sum
    barrier()

    var sm_stride = UInt(block_dim.x // 2)
    while sm_stride > 0:
        if cache_index < sm_stride:
            block_shared[cache_index] += block_shared[cache_index + sm_stride]
        barrier()
        sm_stride //= 2

    if cache_index == 0:
        _ = Atomic.fetch_add(scalar_loss, block_shared[0])


# ── BCEWithLogits Backward Kernel ────────────────────────────────────────────
# For each element i:
#   grad[i] = (sigmoid[i] - target[i]) * grad_output[i]


def bce_with_logits_backward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    grad_result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    sigmoid: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    target: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    grad_output: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
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
                var vec_sig = sigmoid.load[width=simd_width](i)
                var vec_target = target.load[width=simd_width](i)
                var vec_grad = grad_output.load[width=simd_width](i)
                var grad = (vec_sig - vec_target) * vec_grad
                grad_result.store[width=simd_width](i, grad)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    grad_result[idx] = (
                        sigmoid[idx] - target[idx]
                    ) * grad_output[idx]

        base_idx += stride * CHUNK_SIZE


# ── BCEWithLogits Backward Scaled Kernel ──────────────────────────────────────
# Takes scalar_grad (not per-element grad_output buffer).


def bce_with_logits_backward_scaled_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    grad_result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    sigmoid: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    target: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
    scalar_grad: Scalar[dtype],
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE
    var s_grad = SIMD[dtype, simd_width](scalar_grad)

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_sig = sigmoid.load[width=simd_width](i)
                var vec_target = target.load[width=simd_width](i)
                var grad = (vec_sig - vec_target) * s_grad
                grad_result.store[width=simd_width](i, grad)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    grad_result[idx] = (
                        sigmoid[idx] - target[idx]
                    ) * scalar_grad

        base_idx += stride * CHUNK_SIZE


# ── BCELoss Backward Kernel ──────────────────────────────────────────────────
# For each element i:
#   grad[i] = -(target[i]/safe[i] - (1-target[i])/(1-safe[i])) * grad_output[i]


def bce_backward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    grad_result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    safe: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    target: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    grad_output: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
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
                var vec_safe = safe.load[width=simd_width](i)
                var vec_target = target.load[width=simd_width](i)
                var vec_grad = grad_output.load[width=simd_width](i)
                var one = SIMD[dtype, simd_width](1.0)
                var grad = (
                    -(
                        vec_target / vec_safe
                        - (one - vec_target) / (one - vec_safe)
                    )
                    * vec_grad
                )
                grad_result.store[width=simd_width](i, grad)

            elif i < size:
                var one = Scalar[dtype](1.0)
                for j in range(size - i):
                    var idx = i + j
                    var s = safe[idx]
                    var t = target[idx]
                    var g = grad_output[idx]
                    grad_result[idx] = -(t / s - (one - t) / (one - s)) * g

        base_idx += stride * CHUNK_SIZE


# ── BCELoss Backward Scaled Kernel ────────────────────────────────────────────
# Takes scalar_grad (not per-element grad_output buffer).


def bce_backward_scaled_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    grad_result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    safe: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    target: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
    scalar_grad: Scalar[dtype],
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE
    var one_simd = SIMD[dtype, simd_width](1.0)
    var s_grad = SIMD[dtype, simd_width](scalar_grad)

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_safe = safe.load[width=simd_width](i)
                var vec_target = target.load[width=simd_width](i)
                var grad = (
                    -(
                        vec_target / vec_safe
                        - (one_simd - vec_target) / (one_simd - vec_safe)
                    )
                    * s_grad
                )
                grad_result.store[width=simd_width](i, grad)

            elif i < size:
                var one_s = Scalar[dtype](1.0)
                for j in range(size - i):
                    var idx = i + j
                    var si = safe[idx]
                    var ti = target[idx]
                    grad_result[idx] = (
                        -(ti / si - (one_s - ti) / (one_s - si)) * scalar_grad
                    )

        base_idx += stride * CHUNK_SIZE


# ── Launcher ──────────────────────────────────────────────────────────────────


struct BceKernel[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def launch_forward_with_logits(
        A: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
    ) raises -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused BCEWithLogits forward. Returns (per_element_loss, sigmoid)."""
        debug_assert(A.is_on_gpu())
        debug_assert(target.is_on_gpu())

        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = A.device_state.value()
        var device_context = device_state.gpu[]

        var contig_logits = A.contiguous_device_state()
        var contig_target = target.contiguous_device_state()

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
            numels,
            epsilon,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var loss_state = DeviceState[Self.dtype](loss_buffer^, device_state.gpu)
        var sigmoid_state = DeviceState[Self.dtype](
            sigmoid_buffer^, device_state.gpu
        )

        var loss_ndb = NDBuffer[Self.dtype].with_device_state(
            loss_state^, A.shape
        )
        var sigmoid_ndb = NDBuffer[Self.dtype].with_device_state(
            sigmoid_state^, A.shape
        )

        return (loss_ndb^, sigmoid_ndb^)

    @staticmethod
    def launch_forward_with_logits_reduce(
        A: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
        is_mean: Bool,
    ) raises -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused BCEWithLogits forward with mean/sum reduction.

        Returns (scalar_loss, sigmoid) where scalar_loss is 1-element.
        If is_mean, scalar_loss is divided by N on host after kernel."""
        debug_assert(A.is_on_gpu())
        debug_assert(target.is_on_gpu())

        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = A.device_state.value()
        var device_context = device_state.gpu[]

        var contig_logits = A.contiguous_device_state()
        var contig_target = target.contiguous_device_state()

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
            numels,
            epsilon,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var scalar_state = DeviceState[Self.dtype](
            scalar_buffer^, device_state.gpu
        )

        if is_mean:
            var divisor = Scalar[DeviceState[Self.dtype].datatype](numels)
            with scalar_state.buffer.map_to_host() as host_buff:
                host_buff[0] = host_buff[0] / divisor

        var sigmoid_state = DeviceState[Self.dtype](
            sigmoid_buffer^, device_state.gpu
        )

        var scalar_ndb = NDBuffer[Self.dtype].with_device_state(
            scalar_state^, Shape()
        )
        var sigmoid_ndb = NDBuffer[Self.dtype].with_device_state(
            sigmoid_state^, A.shape
        )

        return (scalar_ndb^, sigmoid_ndb^)

    @staticmethod
    def launch_forward(
        A: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
    ) raises -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused BCELoss forward (probabilities input). Returns (per_element_loss, clipped_pred).
        """
        debug_assert(A.is_on_gpu())
        debug_assert(target.is_on_gpu())

        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = A.device_state.value()
        var device_context = device_state.gpu[]

        var contig_pred = A.contiguous_device_state()
        var contig_target = target.contiguous_device_state()

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
            numels,
            epsilon,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var loss_state = DeviceState[Self.dtype](loss_buffer^, device_state.gpu)
        var safe_state = DeviceState[Self.dtype](safe_buffer^, device_state.gpu)

        var loss_ndb = NDBuffer[Self.dtype].with_device_state(
            loss_state^, A.shape
        )
        var safe_ndb = NDBuffer[Self.dtype].with_device_state(
            safe_state^, A.shape
        )

        return (loss_ndb^, safe_ndb^)

    @staticmethod
    def launch_forward_reduce(
        A: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
        is_mean: Bool,
    ) raises -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused BCELoss forward with mean/sum reduction (probabilities input).

        Returns (scalar_loss, clipped_pred) where scalar_loss is 1-element.
        If is_mean, scalar_loss is divided by N on host after kernel."""
        debug_assert(A.is_on_gpu())
        debug_assert(target.is_on_gpu())

        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = A.device_state.value()
        var device_context = device_state.gpu[]

        var contig_pred = A.contiguous_device_state()
        var contig_target = target.contiguous_device_state()

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
            numels,
            epsilon,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var scalar_state = DeviceState[Self.dtype](
            scalar_buffer^, device_state.gpu
        )

        if is_mean:
            var divisor = Scalar[DeviceState[Self.dtype].datatype](numels)
            with scalar_state.buffer.map_to_host() as host_ptr:
                host_ptr[0] = host_ptr[0] / divisor

        var safe_state = DeviceState[Self.dtype](safe_buffer^, device_state.gpu)

        var scalar_ndb = NDBuffer[Self.dtype].with_device_state(
            scalar_state^, Shape()
        )
        var safe_ndb = NDBuffer[Self.dtype].with_device_state(
            safe_state^, A.shape
        )

        return (scalar_ndb^, safe_ndb^)

    @staticmethod
    def launch_bce_with_logits_backward(
        sigmoid: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        grad_output: NDBuffer[Self.dtype],
    ) raises -> NDBuffer[Self.dtype]:
        """Fused BCEWithLogits backward. Returns gradient for logits."""
        debug_assert(sigmoid.is_on_gpu())
        debug_assert(target.is_on_gpu())
        debug_assert(grad_output.is_on_gpu())

        var numels = sigmoid.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = sigmoid.device_state.value()
        var device_context = device_state.gpu[]

        var contig_sig = sigmoid.contiguous_device_state()
        var contig_target = target.contiguous_device_state()
        var contig_grad = grad_output.contiguous_device_state()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_with_logits_backward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
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
            numels,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](
            result_buffer^, device_state.gpu
        )

        var result_ndb = NDBuffer[Self.dtype].with_device_state(
            result_state^, sigmoid.shape
        )

        return result_ndb^

    @staticmethod
    def launch_bce_with_logits_backward_scaled(
        sigmoid: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        scalar_grad: Scalar[Self.dtype],
    ) raises -> NDBuffer[Self.dtype]:
        """Fused BCEWithLogits backward with scalar gradient. Returns gradient for logits.
        """
        debug_assert(sigmoid.is_on_gpu())
        debug_assert(target.is_on_gpu())

        var numels = sigmoid.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = sigmoid.device_state.value()
        var device_context = device_state.gpu[]

        var contig_sig = sigmoid.contiguous_device_state()
        var contig_target = target.contiguous_device_state()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_with_logits_backward_scaled_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
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
            numels,
            scalar_grad,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](
            result_buffer^, device_state.gpu
        )

        var result_ndb = NDBuffer[Self.dtype].with_device_state(
            result_state^, sigmoid.shape
        )

        return result_ndb^

    @staticmethod
    def launch_bce_backward(
        safe: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        grad_output: NDBuffer[Self.dtype],
    ) raises -> NDBuffer[Self.dtype]:
        """Fused BCELoss backward. Returns gradient for pred."""
        debug_assert(safe.is_on_gpu())
        debug_assert(target.is_on_gpu())
        debug_assert(grad_output.is_on_gpu())

        var numels = safe.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = safe.device_state.value()
        var device_context = device_state.gpu[]

        var contig_safe = safe.contiguous_device_state()
        var contig_target = target.contiguous_device_state()
        var contig_grad = grad_output.contiguous_device_state()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_backward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
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
            numels,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](
            result_buffer^, device_state.gpu
        )

        var result_ndb = NDBuffer[Self.dtype].with_device_state(
            result_state^, safe.shape
        )

        return result_ndb^

    @staticmethod
    def launch_bce_backward_scaled(
        safe: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        scalar_grad: Scalar[Self.dtype],
    ) raises -> NDBuffer[Self.dtype]:
        """Fused BCELoss backward with scalar gradient. Returns gradient for pred.
        """
        debug_assert(safe.is_on_gpu())
        debug_assert(target.is_on_gpu())

        var numels = safe.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = safe.device_state.value()
        var device_context = device_state.gpu[]

        var contig_safe = safe.contiguous_device_state()
        var contig_target = target.contiguous_device_state()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_backward_scaled_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
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
            numels,
            scalar_grad,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](
            result_buffer^, device_state.gpu
        )

        var result_ndb = NDBuffer[Self.dtype].with_device_state(
            result_state^, safe.shape
        )

        return result_ndb^

    @staticmethod
    def launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
        threads_per_block: Int
        num_blocks: Int

        if numels < 4096:
            threads_per_block = 128
            num_blocks = (numels + 127) // 128
        elif numels < 65536:
            threads_per_block = 256
            num_blocks = (numels + 255) // 256
        else:
            threads_per_block = 256
            var total_chunks = (numels + (simdwidth * 2 * simdwidth - 1)) // (
                simdwidth * 2 * simdwidth
            )
            num_blocks = min((total_chunks + 255) // 256, 512)
        return threads_per_block, num_blocks
