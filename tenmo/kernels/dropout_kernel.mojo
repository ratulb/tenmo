from std.random.philox import Random as PhiloxRandom
from std.sys import simd_width_of
from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from .kernel_helpers import elementwise_launch_config
from ..gpu.device import GPU, DeviceState


def dropout_forward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    mask_out: Pointer[Scalar[dtype], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
    p: Scalar[dtype],  # dropout probability
    scale: Scalar[dtype],  # 1 / (1 - p)
    rng_seed: UInt64,  # forwarded from Dropout.seed
):
    """Dropout forward kernel: generates mask via Philox RNG and applies it.

    Each thread owns an independent Philox subsequence keyed by global thread id.
    This guarantees statistically independent random streams across threads
    without any shared state or synchronisation.

    Writes:
        result[i]   = A[i] * mask[i]
        mask_out[i] = scale  if rand > p  else 0

    Key design points:
    1. Philox RNG — each thread gets an independent random stream:
         rng = PhiloxRandom(seed=seed, subsequence=global_thread_id, offset=0)
       subsequence isolates per-thread streams → no race conditions.
       Same seed → same mask for a given forward call (reproducible).
    2. step_uniform() returns SIMD[float32, 4] — four values per call.
       For non-float32 dtypes, cast after comparison.
    3. Writes TWO output buffers in one pass (same pattern as unary_ops_with_mask):
       result[i] = input[i] * mask[i]
       mask[i]   = scale if rand > p else 0   (scale baked in)
    4. No dtype.is_floating_point() constraint needed —
       Dropout on integer tensors is unusual but the kernel is dtype-generic.
       Callers should guard at the Module level if desired.
    5. Chunk/stride pattern mirrors existing kernels exactly.
    """
    var size = Int(size_)
    var tid = thread_idx.x
    var gtid = Int(tid + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)

    # Independent Philox stream per thread
    var rng = PhiloxRandom(seed=rng_seed, subsequence=UInt64(gtid), offset=0)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    var zero_s = Scalar[dtype](0)
    var scale_s = scale

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var x_vec = A.unsafe_load[width=simd_width](i)

                # Philox produces SIMD[float32, 4] per call.
                # We process simd_width elements per vector; call step_uniform
                # enough times to cover simd_width lanes.
                # For simd_width <= 4: one call, slice first simd_width values.
                # For simd_width > 4:  multiple calls (handled by scalar tail).
                var rand_f32 = rng.step_uniform()  # SIMD[float32, 4]

                var mask_vec = SIMD[dtype, simd_width](0)
                var res_vec = SIMD[dtype, simd_width](0)

                comptime for lane in range(simd_width):
                    # Cast random float32 to dtype for threshold comparison
                    var r = rand_f32[lane % 4].cast[dtype]()
                    var m = scale_s if r > p else zero_s
                    mask_vec[lane] = m
                    res_vec[lane] = x_vec[lane] * m

                result.unsafe_store[width=simd_width](i, res_vec)
                mask_out.unsafe_store[width=simd_width](i, mask_vec)

            elif i < size:
                # Scalar tail
                for j in range(size - i):
                    var rand_f32_scalar = rng.step_uniform()
                    var r = rand_f32_scalar[0].cast[dtype]()
                    var x_val = A[unsafe_offset=i + j]
                    var m = scale_s if r > p else zero_s
                    result[unsafe_offset=i + j] = x_val * m
                    mask_out[unsafe_offset=i + j] = m

        base_idx += stride * CHUNK_SIZE


struct DropoutKernel[dtype: DType](ImplicitlyCopyable):
    """DropoutKernel launcher
    Returns (output, mask) — both on GPU — as (Layout, Storage) pairs.
    Follows the same pattern as UnaryOpsKernel.launch_with_mask:
      1. materialize_contiguous for non-contiguous input (single map_to_host)
      2. Allocate two output DeviceBuffers
      3. Compile and enqueue dropout_forward_kernel
      4. Wrap results in DeviceState → (Layout, Storage) pairs
    Non-contiguous input:
      Same fix as ReLU — materialize_contiguous does ONE map_to_host sweep.
      The kernel then operates on the flat buffer. No per-element host calls.
    """
    @staticmethod
    def launch(
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        p: Scalar[Self.dtype],
        scale: Scalar[Self.dtype],
        rng_seed: UInt64,
        sync: Bool = False,
    ) raises -> Tuple[Tuple[Layout, DeviceState[Self.dtype]], Tuple[Layout, DeviceState[Self.dtype]]]:
        """Launch dropout forward kernel. Returns (output, mask) on GPU.

        Args:
            A_layout:       Input layout. Must be on GPU.
            A_device_state: Input device state (storage). Must be on GPU.
            p:              Dropout probability.
            scale:          1 / (1 - p).
            rng_seed:       Seed forwarded to Philox — same seed → same mask.
            sync:           Whether to sync GPU after operation.

        Returns:
            ((out_layout, out_storage), (mask_layout, mask_storage)),
            both on GPU.
        """

        var numels = A_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        # Non-contiguous fix: single map_to_host sweep → flat contiguous buffer
        var contig_state = materialize_contiguous(
            A_device_state, A_layout
        )

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )
        var mask_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            dropout_forward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,  # out: dropped values
            mask_buffer,  # out: scale mask
            contig_state.device_buffer(),  # in:  input
            Int64(numels),
            p,
            scale,
            rng_seed,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype](
            result_buffer^, gpu
        )
        var mask_state = DeviceState[Self.dtype](mask_buffer^, gpu)

        var out_pair = (
            Layout(A_layout.shape),
            result_state^,
        )
        var mask_pair = (
            Layout(A_layout.shape),
            mask_state^,
        )

        return (out_pair, mask_pair)

    @staticmethod
    def launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
        return elementwise_launch_config(numels, simdwidth)
