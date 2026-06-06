# =============================================================================
# Welford online mean/variance — tenmo/welford.mojo
#
# Extracted from NDBuffer to keep the core infrastructure lean.
# Single-pass Welford accumulator: mean and variance computed simultaneously.
#
# Consumers:
#   Variance.forward  (variance.mojo)
#   StdDev.forward    (std_deviation.mojo)
#   LayerNorm.forward  (layernorm.mojo)
# =============================================================================

from .ndbuffer import NDBuffer
from .intarray import IntArray
from tenmo.kernels.reduction_kernel import Reduction
from std.sys import has_accelerator


struct Welford[dtype: DType]:
    """Single-pass online mean/variance via Welford's algorithm.

    Computes mean and variance in one pass. Mean is free — Welford computes
    it anyway. GPU dispatch uses Reduction.launch_welford; CPU path uses
    the classic Welford recurrence with per-coordinate serial accumulation.

    Returns (mean_ndb, var_ndb). Variance is already divided by n or n-1.
    Caller saves mean into BwdArg for zero-recomputation backward.
    """

    @staticmethod
    def forward(
        ndb: NDBuffer[Self.dtype],
        axes: IntArray,
        unbiased: Bool,
        keepdims: Bool,
        sync: Bool = False,
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    return Welford[Self.dtype].forward_gpu(
                        ndb, axes, unbiased, keepdims, sync=sync
                    )
                except e:
                    print(e)
                    panic("Welford.forward → GPU operation failed")
                    return (
                        NDBuffer[Self.dtype].Empty(),
                        NDBuffer[Self.dtype].Empty(),
                    )
        return Welford[Self.dtype].forward_cpu(ndb, axes, unbiased, keepdims)

    @staticmethod
    def forward_gpu(
        ndb: NDBuffer[Self.dtype],
        axes: IntArray,
        unbiased: Bool,
        keepdims: Bool,
        sync: Bool = False,
    ) raises -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        var (mean_ndb, M2_ndb) = Reduction[Self.dtype].launch_welford(
            ndb, axes, keepdims, sync=False
        )
        var n = ndb.shape.reduced_shape(axes).product()
        var divisor = Scalar[Self.dtype](n - 1 if unbiased and n > 1 else n)
        var var_ndb = M2_ndb.scalar_ops[Divide](divisor, sync=False)
        comptime if has_accelerator():
            if sync and var_ndb.is_on_gpu():
                var_ndb.sync()
        return (mean_ndb^, var_ndb^)

    @staticmethod
    def forward_cpu(
        ndb: NDBuffer[Self.dtype],
        axes: IntArray,
        unbiased: Bool,
        keepdims: Bool,
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        var out_shape = ndb.shape.compute_output_shape(
            axes, keepdims, validated=True
        )
        var mean_out = NDBuffer[Self.dtype].zeros(out_shape)
        var var_out = NDBuffer[Self.dtype].zeros(out_shape)

        var n = ndb.shape.reduced_shape(axes).product()
        var divisor = Scalar[Self.dtype](n - 1 if unbiased and n > 1 else n)

        if out_shape == Shape():
            # Global scalar reduction
            var local_mean = Scalar[Self.dtype](0)
            var local_M2 = Scalar[Self.dtype](0)
            var count = 0
            for idx in ndb.index_iterator():
                var x = ndb.buffer[idx]
                count += 1
                var delta = x - local_mean
                local_mean += delta / Scalar[Self.dtype](count)
                var delta2 = x - local_mean
                local_M2 += delta * delta2
            mean_out[IntArray()] = local_mean
            var_out[IntArray()] = local_M2 / divisor
        else:
            # Multi-axis reduction — mirrors reduce_cpu coord pattern exactly
            var reduction_axes_shape = ndb.shape.reduced_shape(axes)
            for out_coord in out_shape:
                var local_mean = Scalar[Self.dtype](0)
                var local_M2 = Scalar[Self.dtype](0)
                var count = 0
                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        axes, red_coord
                    ) if keepdims else out_coord.insert(axes, red_coord)
                    var x = ndb[self_coord]
                    count += 1
                    var delta = x - local_mean
                    local_mean += delta / Scalar[Self.dtype](count)
                    var delta2 = x - local_mean
                    local_M2 += delta * delta2
                mean_out[out_coord] = local_mean
                var_out[out_coord] = local_M2 / divisor

        return (mean_out^, var_out^)
