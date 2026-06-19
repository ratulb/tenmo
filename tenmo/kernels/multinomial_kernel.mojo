# =============================================================================
# multinomial_kernel.mojo — GPU multinomial sampling (Gumbel-max trick)
# =============================================================================
#
# Strategy: fused kernel using the Gumbel-max trick:
#   sample = argmax_c ( log_prob[c] + Gumbel_c ),  Gumbel_c ~ Gumbel(0,1)
#
# One block per batch row.  Within a block, each thread is assigned a
# contiguous chunk of classes (contiguous-chunk assignment).  A shared-
# memory tree reduction finds the per-row argmax, repeated for each of
# K samples.
#
# With replacement:
#   Each of the K samples draws fresh independent Gumbel noise per class.
#
# Without replacement:
#   After selecting a class, its log_prob is set to -inf (neg_inf) in
#   global memory (which is a contiguous copy of the original, so the
#   caller's tensor is never modified).  The Gumbel-max's conditional
#   distribution naturally handles zeroed-out categories:
#     P(select j | not select k) = exp(log_prob[j]) / Σ_{i≠k} exp(log_prob[i])
#
# Philox key assignment:
#   seed        = user-provided global seed
#   subsequence = row index (batch-unique)
#   offset      = sample * C + class  (unique per sample–class pair)
#
# This ensures reproducible streams regardless of launch configuration.
#
# Fully utilising Philox SIMD output:
#   step_uniform() returns SIMD[float32, 4] — 4 uniform values per call.
#   Contiguous-chunk assignment lets each thread process 4 consecutive
#   classes per Philox call, consuming all 4 generated values instead
#   of wasting 3/4.
#
# =============================================================================

from std.random.philox import Random as PhiloxRandom
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.memory import AddressSpace, stack_allocation
from std.math import log
from tenmo.ndbuffer import NDBuffer
from tenmo.device import GPU, DeviceState
from tenmo.shapes import Shape


def multinomial_fused_kernel[
    dtype: DType,
    max_block_size: Int = 512,
](
    # log_probs: [B, C] GPU-resident contiguous copy of log-probabilities.
    #   Modified in-place for without-replacement: the selected class's
    #   log-prob is set to neg_inf after each draw so it cannot be
    #   re-selected.  The caller passes a deep copy; the original is
    #   never touched.
    log_probs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    # output: [B, K] GPU-resident output indices (Int32).
    #   output[row * K + s] = argmax_c (log_prob[row, c] + Gumbel[row,s,c])
    output: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    batch_size: Int,
    num_classes: Int,
    num_samples: Int,
    seed: UInt64,
    with_replacement: Int,          # 0 = without, 1 = with replacement
):
    """
    Fused Gumbel-max multinomial kernel.

    One block per batch row, K iterations per block.  Within each
    iteration each thread scores a contiguous chunk of classes
    (4 per Philox call), then a shared-memory tree reduction finds
    the per-row argmax.

    Barrier() calls between phases guarantee correctness:
      global reads → compute → shmem write → barrier → tree reduction
      → barrier → result write + zeroing → barrier → next sample
    """
    var row = block_idx.x
    # grid_dim == batch_size always — all launched blocks are valid.
    # No early-return guard needed.

    var tid = Int(thread_idx.x)
    var block_size = Int(block_dim.x)
    var row_base = row * num_classes

    var smem_val = stack_allocation[
        max_block_size, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var smem_idx = stack_allocation[
        max_block_size, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()

    # Per-thread contiguous chunk assignment — loop-invariant across samples.
    var chunk_size = (num_classes + block_size - 1) // block_size
    var chunk_start = tid * chunk_size
    var chunk_end = min(chunk_start + chunk_size, num_classes)

    for s in range(num_samples):
        # ── Phase 1: per-thread scoring ──────────────────────────────────
        #
        # Thread tid owns a contiguous chunk of ~num_classes/block_size
        # classes.  Within each chunk, 4 consecutive classes are processed
        # per Philox step_uniform() call — fully utilising the 4-wide
        # SIMD output instead of discarding 3 of 4 values.
        #
        var best_val: Scalar[dtype] = neg_inf[dtype]()
        var best_idx: Scalar[DType.int32] = 0

        var c = chunk_start
        while c < chunk_end:
            # One Philox call for 4 consecutive classes
            var rng = PhiloxRandom(
                seed=seed,
                subsequence=UInt64(row),
                offset=UInt64(s * num_classes + c),
            )
            var u4 = rng.step_uniform()          # SIMD[float32, 4]

            var remaining = chunk_end - c
            for lane in range(4):
                if lane >= remaining:
                    break
                var u = max(u4[lane], 1e-10)
                var gumbel_f32 = -log(-log(u))
                var gumbel = Scalar[dtype](gumbel_f32)
                var idx = c + lane
                var score = log_probs[row_base + idx] + gumbel

                if score > best_val:
                    best_val = score
                    best_idx = Scalar[DType.int32](idx)

            c += 4

        # ── Phase 2: tree reduction for per-row argmax ───────────────────
        smem_val[tid] = best_val
        smem_idx[tid] = best_idx
        barrier()

        var stride = block_size >> 1
        while stride > 0:
            if tid < stride:
                if smem_val[tid + stride] > smem_val[tid]:
                    smem_val[tid] = smem_val[tid + stride]
                    smem_idx[tid] = smem_idx[tid + stride]
            barrier()
            stride >>= 1

        # ── Phase 3: write result + handle without-replacement ───────────
        if tid == 0:
            output[row * num_samples + s] = smem_idx[0]

            if with_replacement == 0 and s < num_samples - 1:
                # Zero this class in log-space so it won't be re-selected.
                # No renormalisation needed — the Gumbel-max trick's
                # conditional distribution naturally handles a category
                # with log-prob = -inf (probability = 0).
                log_probs[row_base + smem_idx[0].__int__()] = neg_inf[dtype]()

        # barrier() ensures the zeroing write (by thread 0) is visible to
        # all threads before the next sample iteration reads log_probs.
        barrier()


@fieldwise_init
struct MultinomialGpuKernel[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Fused Gumbel-max multinomial sampler on GPU.

    Usage (from within a comptime if has_accelerator() guard):
        var out_ndb = MultinomialGpuKernel[dtype].launch(
            log_probs_ndb, out_shape, num_samples, seed, replacement
        )
    """

    @staticmethod
    def launch(
        log_probs: NDBuffer[Self.dtype],
        out_shape: Shape,
        num_samples: Int,
        seed: UInt64,
        with_replacement: Bool,
        sync: Bool = False,
    ) raises -> NDBuffer[DType.int32]:
        """
        Launch the fused Gumbel-max multinomial kernel.

        Args:
            log_probs:  [B, C] log-probabilities, GPU-resident.
            out_shape:  Output shape — (K,) for 1D input, (B, K) for 2D.
            num_samples: Number of samples to draw per batch row.
            seed:       Philox seed for deterministic random noise.
            with_replacement: Sample with replacement if True.
            sync:       GPU synchronize before returning.

        Returns:
            NDBuffer[DType.int32] on GPU with shape out_shape.

        Notes:
            - log_probs is NOT modified; an internal contiguous copy is
              made for the without-replacement zeroing, so the caller's
              original tensor is untouched.
            - The output indices span [0, C-1] for each sample.
            - Degenerate input (all -inf) produces index 0 for all
              samples, because neg_inf + finite_Gumbel = neg_inf, and
              neg_inf > neg_inf is false in IEEE 754.
        """
        debug_assert(
            log_probs.is_on_gpu(),
            "MultinomialGpuKernel.launch requires GPU-resident input",
        )
        if not with_replacement:
            debug_assert(
                num_samples <= log_probs.shape[-1],
                "without-replacement multinomial requires num_samples <= num_classes, "
                "but got num_samples=" + String(num_samples)
                + " > num_classes=" + String(log_probs.shape[-1]),
            )

        var batch_size = 1 if log_probs.rank() == 1 else log_probs.shape[0]
        var num_classes = log_probs.shape[-1]
        var output_flat_size = batch_size * num_samples

        debug_assert(
            out_shape.product() == output_flat_size,
            "out_shape product (" + String(out_shape.product())
            + ") does not match batch_size * num_samples ("
            + String(output_flat_size) + ")",
        )

        # Deep contiguous copy — the kernel modifies this copy in-place
        # for without-replacement zeroing.  The caller's tensor is never
        # touched.
        var contig_state = log_probs.contiguous_device_state()

        ref device_state = log_probs.device_state.value()
        ref gpu = device_state.get_gpu()
        var device_context = gpu[]

        var out_device_buf = device_context.enqueue_create_buffer[DType.int32](
            output_flat_size,
        )

        # Launch configuration:
        #   blocks  = batch_size (one block per row)
        #   threads = smallest power-of-two >= num_classes, capped at 512
        comptime MAX_BLOCK_SIZE = 512
        var block_size = 1
        while block_size < MAX_BLOCK_SIZE and block_size < num_classes:
            block_size <<= 1

        var compiled = device_context.compile_function[
            multinomial_fused_kernel[Self.dtype, MAX_BLOCK_SIZE],
            multinomial_fused_kernel[Self.dtype, MAX_BLOCK_SIZE],
        ]()

        device_context.enqueue_function(
            compiled,
            contig_state.device_buffer(),
            out_device_buf,
            batch_size,
            num_classes,
            num_samples,
            seed,
            1 if with_replacement else 0,
            grid_dim=batch_size,
            block_dim=block_size,
        )

        if sync:
            device_context.synchronize()

        var out_state = DeviceState[DType.int32](out_device_buf^, gpu)
        return NDBuffer[DType.int32].with_device_state(out_state^, out_shape)
