"""GPU launch configuration + helpers.

`elementwise_launch_config` is the canonical element-wise kernel launch
config, moved here from `tenmo/kernels/kernel_helpers.mojo` so that
standalone GPU programs (which must not import `tenmo.kernels`) can use it.
"""


@always_inline
def elementwise_launch_config(
    numels: Int,
    simd_width: Int,
) -> Tuple[Int, Int]:
    """Compute launch configuration for an elementwise GPU kernel.

        CHUNK_SIZE = simd_vectors_per_thread * simd_width
                   = 2 * simd_width          (simd_vectors_per_thread == 2)

    Each thread processes exactly CHUNK_SIZE elements per grid-stride
    iteration. The outer grid-stride while-loop means we never under-cover
    the tensor: every element is processed regardless of grid size. Grid
    sizing therefore only affects *occupancy and latency*, not correctness.

    Why CHUNK_SIZE = 2 * simd_width, not 2 * simd_width²:
    In every kernel, the comptime-unrolled inner loop runs
    simd_vectors_per_thread = 2 times, and each iteration processes one
    simd_width-wide SIMD vector. Total per thread per grid-stride pass:
        2 * simd_width elements.
    The launch config must use this same definition or it will compute the
    wrong number of chunks, over- or under-launching blocks.

    Chunk → block mapping: we want roughly one grid-stride iteration per
    thread on typical tensors, meaning we want:
        num_blocks * threads_per_block ≈ total_chunks

    Rearranged:
        num_blocks ≈ ceil(total_chunks / threads_per_block)

    We use a flat threads_per_block = 256 throughout. 256 is the standard
    sweet spot on both NVIDIA (4 warps of 64) and AMD (4 waves of 64):
      - Large enough to hide memory latency through warp/wave switching.
      - Small enough to fit multiple blocks per SM/CU (good occupancy).
      - A power of two, which keeps block-dim arithmetic compiler-friendly.
    Varying threads_per_block by tier (128 for small tensors, 256 for large)
    gives no real benefit and complicates reasoning, so we don't do it.

    Block count cap: uncapped block counts give diminishing returns. Past
    ~2× the number of SMs on the device, adding more blocks doesn't improve
    latency because the hardware has no unsafe_free execution slots. We don't know the SM count at
    compile time, so we use 512 as a conservative upper bound:
      - Covers current high-end GPUs (H100: 132 SMs, A100: 108 SMs).
      - At 512 blocks × 256 threads × 2 × simd_width elements per thread,
        float32 (simd_width=8): 512 * 256 * 16 = 2,097,152 elements covered
        per grid pass. Tensors larger than this use the grid-stride loop and
        take multiple passes — which is expected and correct.
      - For smaller GPUs (e.g. 40 SMs) this wastes some dispatch overhead,
        but the grid-stride loop amortises it quickly.

    Minimum of 1 block: for numels == 0, total_chunks == 0 and
    ceil(0 / 256) == 0, which would enqueue a kernel with zero blocks — undefined behaviour on most runtimes.
    Clamping to max(1, ...) lets the kernel launch and immediately exit via
    the `if i >= size: break` guard inside the unrolled loop. The caller is
    responsible for not passing numels == 0 in practice, but we defend here.

    Args:
        numels:     Total number of scalar elements in the output tensor.
        simd_width: Native SIMD width for the output dtype, obtained via
                    simd_width_of[dtype](). Must be a power of two ≥ 1.

    Returns:
        (num_blocks, threads_per_block) to pass as grid_dim / block_dim.
    """
    # Step 1: reproduce the kernel's own CHUNK_SIZE exactly
    # simd_vectors_per_thread is hard-coded to 2 at every kernel call site
    # (the `2 * simd_width` template argument). If that constant ever changes,
    # this line must change with it. Keeping the formula here explicit —
    # rather than using a shared constant — makes the dependency visible.
    var CHUNK_SIZE = 2 * simd_width  # elements consumed by one thread per pass

    # Step 2: total chunks = number of thread-work units
    # Ceiling division ensures the last partial chunk is still counted.
    # If numels is an exact multiple of CHUNK_SIZE, no rounding occurs.
    var total_chunks = (numels + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Step 3: threads per block — fixed at 256 for all tensor sizes
    # See docstring rationale above. Exposed as a named variable so a future
    # reader can change it in one place and have it propagate correctly.
    var THREADS_PER_BLOCK = 256

    # Step 4: blocks needed to cover all chunks in one grid-stride pass
    # Each block contributes THREADS_PER_BLOCK thread-work units per pass.
    # Ceiling division again: a partially-filled last block is still needed.
    var blocks_needed = (
        total_chunks + THREADS_PER_BLOCK - 1
    ) // THREADS_PER_BLOCK

    # Step 5: clamp to [1, 512]
    # Lower bound 1: prevents a zero-block launch (see docstring).
    # Upper bound 512: avoids over-subscribing the GPU scheduler.
    # The grid-stride loop ensures correctness regardless of how many blocks
    # are actually launched; this clamp only affects per-pass coverage.
    var num_blocks = max(1, min(blocks_needed, 512))

    return (num_blocks, THREADS_PER_BLOCK)

