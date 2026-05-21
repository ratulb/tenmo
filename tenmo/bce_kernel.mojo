"""Fused BCE / BCEWithLogits GPU kernels + launcher.

Forward kernel computes per-element loss AND sigmoid (for backward)
in a single GPU pass — eliminating 10+ separate tensor ops.

Gradient formulas (after mean reduction with N elements):
  BCEWithLogits: d(loss)/d(logits_i) = (sigmoid(logits_i) - target_i) / N
  BCELoss:       d(loss)/d(p_i) = -(target_i/clip(p_i) - (1-target_i)/(1-clip(p_i))) / N
"""

from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of
from std.math import exp, log

from .ndbuffer import NDBuffer
from .device import DeviceState
from .common_utils import panic


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


# ── Launcher ──────────────────────────────────────────────────────────────────


struct BceKernel[dtype: DType](ImplicitlyCopyable & Movable):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def launch_forward_with_logits(
        A: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
    ) raises -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        """Fused BCEWithLogits forward. Returns (per_element_loss, sigmoid)."""
        debug_assert(A.is_on_gpu())
        debug_assert(target.is_on_gpu())

        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.datatype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = A.device_state.value()
        var device_context = device_state.gpu[]

        var contig_logits = A.contiguous_device_state()
        var contig_target = target.contiguous_device_state()

        var loss_buffer = device_context.enqueue_create_buffer[Self.datatype](
            numels
        )
        var sigmoid_buffer = device_context.enqueue_create_buffer[
            Self.datatype
        ](numels)

        var compiled = device_context.compile_function[
            bce_with_logits_forward_kernel[
                dtype=Self.datatype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
            bce_with_logits_forward_kernel[
                dtype=Self.datatype,
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

        var loss_state = DeviceState[Self.dtype].__init__[True](
            loss_buffer^, device_state.gpu
        )
        var sigmoid_state = DeviceState[Self.dtype].__init__[True](
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
    def launch_forward(
        A: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
    ) raises -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        """Fused BCELoss forward (probabilities input). Returns (per_element_loss, clipped_pred).
        """
        debug_assert(A.is_on_gpu())
        debug_assert(target.is_on_gpu())

        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.datatype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = A.device_state.value()
        var device_context = device_state.gpu[]

        var contig_pred = A.contiguous_device_state()
        var contig_target = target.contiguous_device_state()

        var loss_buffer = device_context.enqueue_create_buffer[Self.datatype](
            numels
        )
        var safe_buffer = device_context.enqueue_create_buffer[Self.datatype](
            numels
        )

        var compiled = device_context.compile_function[
            bce_forward_kernel[
                dtype=Self.datatype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
            bce_forward_kernel[
                dtype=Self.datatype,
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

        var loss_state = DeviceState[Self.dtype].__init__[True](
            loss_buffer^, device_state.gpu
        )
        var safe_state = DeviceState[Self.dtype].__init__[True](
            safe_buffer^, device_state.gpu
        )

        var loss_ndb = NDBuffer[Self.dtype].with_device_state(
            loss_state^, A.shape
        )
        var safe_ndb = NDBuffer[Self.dtype].with_device_state(
            safe_state^, A.shape
        )

        return (loss_ndb^, safe_ndb^)

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
