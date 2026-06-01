# =============================================================================
# tenmo/kernels/__init__.mojo
# =============================================================================
#
# All GPU kernels in one place. Every kernel source file lives under this
# directory. Re-exports let consumers `from tenmo.kernels import Reduction`.
#
# SHARED INDEX HELPERS
# ─────────────────────
# output_to_input_base / rank_to_reduced_offset are used by multiple
# kernel files (reduction_kernel.mojo, std_variance_backward_kernel.mojo).
# They live here to avoid circular imports and to keep them in one place.
# =============================================================================

from .scalar_ops_kernel import ScalarOperations
from .scalar_inplace_ops_kernel import InplaceScalarOperations
from .binary_ops_kernel import BinaryOperations
from .binary_inplace_ops_kernel import BinaryInplaceOperations
from .unary_ops_kernel import UnaryOpsKernel
from .matmul_kernel import MatmulNdGpu
from .compare_kernel import AllClose, Compare, CompareScalar
from .reduction_kernel import Reduction
from .bce_kernel import BceKernel
from .division_kernel import DivisionKernel
from .minmax_kernel import ReductionMinMax
from .std_variance_backward_kernel import StdVarianceBackwardKernel
from .layernorm_kernel import LayerNormKernel
from .matrixvector_kernel import MatrixVectorNdGpu
from .vectormatrix_kernel import VectorMatmulNdGpu
from .dropout_kernel import DropoutKernel
from .shuffle_kernel import ShuffleGPU
from .dotproduct_kernel import DotproductKernel
from .argminmax_kernel import ArgMinMaxGpu
from .filler_kernel import FillerGpu
from .gather_kernel import GatherGpu


# =============================================================================
# Shared index helpers — used by reduction_kernel and std_variance_backward_kernel
# =============================================================================


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
