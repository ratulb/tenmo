# =============================================================================
# Sum / Mean reduction on NDBuffer — tenmo/sum_mean_reduction.mojo
#
# Extracted from NDBuffer. CPU + GPU dispatch for SUM and MEAN reductions.
# Also includes sum_all (CPU scalar sum) and sum_over_broadcasted_axes
# (broadcast-expansion utility used by backward passes).
#
# GPU dispatch goes through Reduction.launch in reduction_kernel.mojo.
# =============================================================================

from .ndbuffer import NDBuffer
from .intarray import IntArray
from .shapes import Shape
from .reduction_kernel import Reduction
from .common_utils import Epsilon
from .mnemonics import SUM, MEAN
from std.sys import has_accelerator


struct SumMeanReduction[dtype: DType]:
    """Sum/mean reduction on NDBuffer — device-dispatch + CPU fallback.

    Static methods mirror the original NDBuffer instance methods.
    GPU goes through Reduction.launch; CPU uses reduce_cpu (SIMD suffix
    fast path + coordinate-by-coordinate fallback).
    """

    @staticmethod
    def reduce[
        op_code: Int = SUM
    ](
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool = False,
    ) -> NDBuffer[Self.dtype]:
        """Sum / mean reduction. Axes must be already normalized.
        op_code: SUM or MEAN."""
        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    out = Reduction[Self.dtype].launch[op_code](
                        ndb, normalized_axes, keepdims
                    )
                except e:
                    print(e)
                    panic(
                        "SumMeanReduction reduce — GPU operation failed for op_code: ",
                        String(op_code),
                    )
                    out = NDBuffer[Self.dtype].Empty()
            else:
                out = SumMeanReduction[Self.dtype].reduce_cpu[op_code](
                    ndb, normalized_axes, keepdims
                )
        else:
            out = SumMeanReduction[Self.dtype].reduce_cpu[op_code](
                ndb, normalized_axes, keepdims
            )

        return out^

    @staticmethod
    def reduce_cpu[
        op_code: Int = SUM
    ](
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
    ) -> NDBuffer[Self.dtype]:
        """CPU sum / mean. op_code: SUM or MEAN."""
        var reduced_volume = Scalar[Self.dtype](1)

        comptime if op_code == MEAN:
            var volume = ndb.shape.reduced_shape(normalized_axes).product()
            reduced_volume = reduced_volume if volume == 0 else Scalar[
                Self.dtype
            ](volume)

        var out_shape = ndb.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var out = NDBuffer[Self.dtype].zeros(out_shape)

        if out_shape == Shape():
            comptime if op_code == MEAN:
                out[IntArray()] = SumMeanReduction[Self.dtype].sum_all(ndb) / reduced_volume
            else:
                out[IntArray()] = SumMeanReduction[Self.dtype].sum_all(ndb)
        else:
            var reduction_axes_shape = ndb.shape.reduced_shape(normalized_axes)
            # Fast path: contiguous input with suffix-axis reduction —
            # each output element maps to a contiguous block of reduced_volume
            # elements in memory, so we can call Buffer.sum() with SIMD.
            if ndb.is_contiguous():
                var rank = ndb.shape.ndim()
                var num_axes = normalized_axes.size()
                var is_suffix = (
                    num_axes > 0 and normalized_axes[num_axes - 1] == rank - 1
                )
                var idx = 0
                while is_suffix and idx < num_axes - 1:
                    if normalized_axes[idx] != rank - num_axes + idx:
                        is_suffix = False
                        break
                    idx += 1
                if is_suffix:
                    var reduced_numels = reduction_axes_shape.product()
                    var num_out = out.numels()
                    for oi in range(num_out):
                        var base = ndb.offset + oi * reduced_numels
                        comptime if op_code == MEAN:
                            out.buffer[oi] = (
                                ndb.buffer.sum(base, base + reduced_numels)
                                / reduced_volume
                            )
                        else:
                            out.buffer[oi] = ndb.buffer.sum(
                                base, base + reduced_numels
                            )
                    return out^

            # Fallback: coord-by-coord (works for any layout / any axes)
            for out_coord in out_shape:
                var accum = Scalar[Self.dtype](0)
                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    accum += ndb[self_coord]
                comptime if op_code == MEAN:
                    out[out_coord] = accum / reduced_volume
                else:
                    out[out_coord] = accum

        return out^

    @staticmethod
    def sum(
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool = False,
    ) -> NDBuffer[Self.dtype]:
        """Thin wrapper around reduce[SUM]."""
        return SumMeanReduction[Self.dtype].reduce[op_code=SUM](
            ndb, normalized_axes, keepdims
        )

    @staticmethod
    def sum_all(ndb: NDBuffer[Self.dtype]) -> Scalar[Self.dtype]:
        """CPU only operation — sum of all elements."""
        if ndb.is_contiguous():
            var start = ndb.offset
            var end = start + ndb.numels()
            return ndb.buffer.sum(start, end)
        else:
            var accum_sum: Scalar[Self.dtype] = Scalar[Self.dtype](0)
            for index in ndb.index_iterator():
                accum_sum += ndb.buffer[index]
            return accum_sum

    @staticmethod
    def sum_over_broadcasted_axes(
        extended_buffer: NDBuffer[Self.dtype],
        target_shape: Shape,
    ) -> NDBuffer[Self.dtype]:
        """Sum over broadcasted axes to match target shape."""
        if extended_buffer.shape == target_shape:
            return extended_buffer
        var result: NDBuffer[Self.dtype]
        if extended_buffer.is_on_cpu():
            result = extended_buffer.contiguous()
        else:
            result = extended_buffer
        var current_shape = result.shape
        # Sum over extra leading dimensions
        while len(current_shape) > len(target_shape):
            result = SumMeanReduction[Self.dtype].reduce(
                result, normalized_axes=IntArray(0), keepdims=False
            )
            current_shape = result.shape
        # Sum over mismatched dimensions
        for i in range(len(target_shape)):
            if current_shape[i] != target_shape[i] and current_shape[i] > 1:
                result = SumMeanReduction[Self.dtype].reduce(
                    result, normalized_axes=IntArray(i), keepdims=True
                )
                current_shape = result.shape
        return result^
