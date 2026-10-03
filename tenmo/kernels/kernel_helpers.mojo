# Shared kernel helpers — index computation + launch configuration.
#
# - output_to_input_base / rank_to_reduced_offset: reduction index helpers
#   (used by reduction_kernel.mojo, minmax_kernel.mojo,
#    std_variance_backward_kernel.mojo)
# - elementwise_launch_config: unified launch config for element-wise GPU
#   kernels (used by scalar, binary, unary, bce, dropout, division,
#   compare, sgd, gather, filler kernels)
#
# elementwise_launch_config now lives in tenmo/gpu/runtime.mojo and
# is re-exported here so tenmo.kernels callers keep their existing import.

from ..shared.array import RankArray
from ..gpu.runtime import elementwise_launch_config


@always_inline
def reduction_launch_config[
    max_block_size: Int
](total_output: Int, reduced_volume: Int) -> Tuple[Int, Int]:
    """Compute (threads_per_block, num_blocks) for reduction kernels.

    One block per output element. Block size is the smallest power of two
    >= reduced_volume, capped at max_block_size.

    Args:
        total_output:     Number of output elements (= number of blocks).
        reduced_volume:   Number of elements to reduce per output element.

    Returns:
        (threads_per_block, num_blocks).
    """
    comptime assert (
        max_block_size.is_power_of_two() and max_block_size <= 1024
    ), "max_block_size must be a power of two <= 1024"
    var block_size = 1
    while block_size < reduced_volume:
        block_size <<= 1
        if block_size >= max_block_size:
            block_size = max_block_size
            break
    return (block_size, total_output)


@always_inline
def output_to_input_base(
    out_idx: Int,
    in_shape: RankArray,
    in_strides: RankArray,
    reduction_axes: RankArray,
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
    rank: Int, in_shape: RankArray, in_strides: RankArray, reduction_axes: RankArray
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

