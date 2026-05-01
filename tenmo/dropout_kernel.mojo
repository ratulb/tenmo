from std.random.philox import Random as PhiloxRandom
from std.sys import simd_width_of
from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from .ndbuffer import NDBuffer
from .device import GPU, DeviceState



# Key design points:
#
# 1. Philox RNG — each thread gets an independent random stream:
#      rng = PhiloxRandom(seed=seed, subsequence=global_thread_id, offset=0)
#    subsequence isolates per-thread streams → no race conditions.
#    Same seed → same mask for a given forward call (reproducible).
#
# 2. step_uniform() returns SIMD[float32, 4] — four values per call.
#    For non-float32 dtypes, cast after comparison.
#
# 3. Writes TWO output buffers in one pass (same pattern as unary_ops_with_mask):
#    result[i] = input[i] * mask[i]
#    mask[i]   = scale if rand > p else 0   (scale baked in)
#
# 4. No dtype.is_floating_point() constraint needed —
#    Dropout on integer tensors is unusual but the kernel is dtype-generic.
#    Callers should guard at the Module level if desired.
#
# 5. Chunk/stride pattern mirrors existing kernels exactly.
#

fn dropout_forward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result:    UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mask_out:  UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A:         UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size:      Int,
    p:         Scalar[dtype],        # dropout probability
    scale:     Scalar[dtype],        # 1 / (1 - p)
    rng_seed:  UInt64,               # forwarded from Dropout.seed
):
    """Dropout forward kernel: generates mask via Philox RNG and applies it.

    Each thread owns an independent Philox subsequence keyed by global thread id.
    This guarantees statistically independent random streams across threads
    without any shared state or synchronisation.

    Writes:
        result[i]   = A[i] * mask[i]
        mask_out[i] = scale  if rand > p  else 0
    """
    var tid    = thread_idx.x
    var gtid   = Int(tid + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)

    # Independent Philox stream per thread
    var rng = PhiloxRandom(seed=rng_seed, subsequence=UInt64(gtid), offset=0)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    var zero_s  = Scalar[dtype](0)
    var scale_s = scale

    while base_idx < size:

        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var x_vec = A.load[width=simd_width](i)

                # Philox produces SIMD[float32, 4] per call.
                # We process simd_width elements per vector; call step_uniform
                # enough times to cover simd_width lanes.
                # For simd_width <= 4: one call, slice first simd_width values.
                # For simd_width > 4:  multiple calls (handled by scalar tail).
                var rand_f32 = rng.step_uniform()  # SIMD[float32, 4]

                var mask_vec = SIMD[dtype, simd_width](0)
                var res_vec  = SIMD[dtype, simd_width](0)

                comptime for lane in range(simd_width):
                    # Cast random float32 to dtype for threshold comparison
                    var r = rand_f32[lane % 4].cast[dtype]()
                    var m = scale_s if r > p else zero_s
                    mask_vec[lane] = m
                    res_vec[lane]  = x_vec[lane] * m

                result.store[width=simd_width](i, res_vec)
                mask_out.store[width=simd_width](i, mask_vec)

            elif i < size:
                # Scalar tail
                for j in range(size - i):
                    var rand_f32_scalar = rng.step_uniform()
                    var r = rand_f32_scalar[0].cast[dtype]()
                    var x_val = A[i + j]
                    var m     = scale_s if r > p else zero_s
                    result[i + j]   = x_val * m
                    mask_out[i + j] = m

        base_idx += stride * CHUNK_SIZE


# =============================================================================
# DropoutKernel launcher
# =============================================================================
#
#
# Returns Tuple[NDBuffer, NDBuffer] — (output, mask) — both on GPU.
# Follows the same pattern as UnaryOpsKernel.launch_with_mask:
#   1. contiguous_device_state() for non-contiguous input (single map_to_host)
#   2. Allocate two output DeviceBuffers
#   3. Compile and enqueue dropout_forward_kernel
#   4. Wrap results in DeviceState → NDBuffer via with_device_state()
#
# Non-contiguous input:
#   Same fix as ReLU — contiguous_device_state() does ONE map_to_host sweep.
#   The kernel then operates on the flat buffer. No per-element host calls.
# =============================================================================

struct DropoutKernel[dtype: DType](ImplicitlyCopyable & Movable):

    @staticmethod
    fn launch(
        A:        NDBuffer[Self.dtype],
        p:        Scalar[Self.dtype],
        scale:    Scalar[Self.dtype],
        rng_seed: UInt64,
    ) raises -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        """Launch dropout forward kernel. Returns (output, mask) on GPU.

        Args:
            A:        Input NDBuffer. Must be on GPU.
            p:        Dropout probability.
            scale:    1 / (1 - p).
            rng_seed: Seed forwarded to Philox — same seed → same mask.

        Returns:
            Tuple of (output NDBuffer, mask NDBuffer), both on GPU.
        """
        debug_assert(A.is_on_gpu())

        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        # Reuse UnaryOpsKernel launch config — same heuristic applies
        var (threads_per_block, num_blocks) = Self.launch_config(numels, simdwidth)

        ref device_state   = A.device_state.value()
        var device_context = device_state.gpu[]

        # Non-contiguous fix: single map_to_host sweep → flat contiguous buffer
        var contig_state = A.contiguous_device_state()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](numels)
        var mask_buffer   = device_context.enqueue_create_buffer[Self.dtype](numels)

        var compiled = device_context.compile_function[
            dropout_forward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2*simdwidth,
            ],
            dropout_forward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2*simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,                  # out: dropped values
            mask_buffer,                    # out: scale mask
            contig_state.device_buffer(),   # in:  input
            numels,
            p,
            scale,
            rng_seed,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, device_state.gpu)
        var mask_state   = DeviceState[Self.dtype](mask_buffer^,   device_state.gpu)

        var out_ndb  = NDBuffer[Self.dtype].with_device_state(result_state^, A.shape)
        var mask_ndb = NDBuffer[Self.dtype].with_device_state(mask_state^,   A.shape)

        return (out_ndb^, mask_ndb^)

    @staticmethod
    fn launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
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
