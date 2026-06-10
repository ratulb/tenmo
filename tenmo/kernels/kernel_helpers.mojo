# =============================================================================
# Shared kernel helpers — index computation + launch configuration.
#
# - output_to_input_base / rank_to_reduced_offset: reduction index helpers
#   (used by reduction_kernel.mojo, minmax_kernel.mojo,
#    std_variance_backward_kernel.mojo)
# - elementwise_launch_config: unified launch config for element-wise GPU
#   kernels (used by scalar, binary, unary, bce, dropout, division,
#   compare, sgd, gather, filler kernels)
# =============================================================================

from tenmo.array import Array

@always_inline
def output_to_input_base(
    out_idx: Int,
    in_shape: Array,
    in_strides: Array,
    reduction_axes: Array,
) -> Int:
    var remaining = out_idx
    var input_base = 0

    if len(reduction_axes) == 0:
        return 0

    for k in reversed(range(len(in_shape))):
        if k not in reduction_axes:
            var coord = remaining % in_shape[k]
            remaining //= in_shape[k]
            input_base += coord * in_strides[k]

    return input_base


@always_inline
def rank_to_reduced_offset(
    rank: Int, in_shape: Array, in_strides: Array, reduction_axes: Array
) -> Int:
    var tmp = rank
    var offset = 0
    var reduce_all = len(reduction_axes) == 0

    for k in reversed(range(len(in_shape))):
        if reduce_all or k in reduction_axes:
            var coord = tmp % in_shape[k]
            tmp //= in_shape[k]
            offset += coord * in_strides[k]

    return offset


@always_inline
def elementwise_launch_config(
    numels: Int,
    simd_width: Int,
) -> Tuple[Int, Int]:
    """Returns (num_blocks, threads_per_block).

    Each thread processes 2 * simd_width² elements per grid-stride
    iteration. Launches enough blocks for ~1 iteration on most tensors,
    caps at 512 blocks.  Three tiers:
      ≤128 chunks  → 128 threads,  1 block per chunk
      ≤512 chunks  → 256 threads,  1 block per chunk
      >512 chunks  → 256 threads,  ceil(chunks/256) blocks capped at 512
    """
    var CHUNK_SIZE = 2 * simd_width * simd_width
    var total_chunks = (numels + CHUNK_SIZE - 1) // CHUNK_SIZE

    if total_chunks <= 128:
        return (max(1, total_chunks), 128)
    elif total_chunks <= 512:
        return (total_chunks, 256)
    else:
        return (min((total_chunks + 255) // 256, 512), 256)
