# Sum / Mean reduction on NDBuffer — tenmo/sum_mean_reduction.mojo
#
# Extracted from NDBuffer. CPU + GPU dispatch for SUM and MEAN reductions.
# Also includes sum_all (CPU scalar sum) and sum_over_broadcasted_axes
# (broadcast-expansion utility used by backward passes).
#
# GPU dispatch goes through ReductionKernel.launch in reduction_kernel.mojo.

from .backpropagation import ArgumentType

from .ndbuffer import NDBuffer
from .shared.buffers import Buffer
from .shared.intarray import IntArray
from .shared.shapes import Shape
from .shared.reduction_walk import ReductionWalk
from .shared.indexhelper import IndexCalculator
from .kernels.reduction_kernel import ReductionKernel
from .shared.panic import panic
from max.algorithm import parallelize
from std.sys.info import num_physical_cores
from .shared.mnemonics import SUM, MEAN
from std.sys import has_accelerator


@fieldwise_init
struct ReductionArg(ArgumentType):
    var axes: IntArray
    var keepdims: Bool


struct SumMeanReduction[dtype: DType]:
    """Sum/mean reduction on NDBuffer — device-dispatch + CPU fallback.

    Static methods mirror the original NDBuffer instance methods.
    GPU goes through ReductionKernel.launch; CPU uses reduce_cpu (SIMD suffix
    fast path + coordinate-by-coordinate fallback).
    """

    @staticmethod
    def reduce[
        op_code: Int = SUM
    ](
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool = False,
        sync: Bool = True,
    ) -> NDBuffer[Self.dtype]:
        """Sum / mean reduction. Axes must be already normalized.
        op_code: SUM or MEAN."""
        var out: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    var (result_layout, result_storage) = ReductionKernel[
                        Self.dtype
                    ].launch[op_code](
                        ndb.layout(),
                        ndb.device_state.value(),
                        normalized_axes,
                        keepdims,
                        sync=sync,
                    )
                    out = NDBuffer[Self.dtype].with_layout_device_state(
                        result_layout, result_storage
                    )
                except e:
                    print(e)
                    panic(
                        (
                            "SumMeanReduction reduce — GPU operation failed for"
                            " op_code: "
                        ),
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
            var volume = (
                ndb.shape.reduced_shape(normalized_axes).product()
            )
            reduced_volume = reduced_volume if volume == 0 else Scalar[
                Self.dtype
            ](volume)

        var out_shape = ndb.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var out = NDBuffer[Self.dtype].zeros(out_shape)

        if out_shape == Shape():
            comptime if op_code == MEAN:
                out[IntArray()] = (
                    SumMeanReduction[Self.dtype].sum_all(ndb) / reduced_volume
                )
            else:
                out[IntArray()] = SumMeanReduction[Self.dtype].sum_all(ndb)
        else:
            var reduction_axes_shape = ndb.shape.reduced_shape(
                normalized_axes
            )
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
                    var n_threads = num_physical_cores()
                    var ndb_offset = ndb.offset
                    var ndb_buf = ndb.buffer

                    def reduce_row(oi: Int) {imm}:
                        var base = ndb_offset + oi * reduced_numels
                        comptime if op_code == MEAN:
                            out.buffer[oi] = (
                                ndb_buf.sum(base, base + reduced_numels)
                                / reduced_volume
                            )
                        else:
                            out.buffer[oi] = ndb_buf.sum(
                                base, base + reduced_numels
                            )

                    # SIMD sum is very cheap per element: parallel only when
                    # there is enough work to amortize the ~5us launch cost
                    # (measured crossover ≈ threads * 32768 elements).
                    if (
                        num_out >= n_threads
                        and num_out * reduced_numels >= n_threads * 32768
                    ):
                        parallelize(reduce_row, num_out, n_threads)
                    else:
                        for oi in range(num_out):
                            reduce_row(oi)
                    return out^

            # Fallback: works for any layout / any axes. Hoists the output
            # coordinate once per output slot and walks the reduced axes with
            # stride arithmetic — no per-element IntArray alloc.
            var ndb_layout = ndb.layout()
            var ndb_buf = ndb.buffer
            var ndb_offset = ndb_layout.offset
            var num_out = out.numels()
            var n_threads = num_physical_cores()
            var walk = ReductionWalk.build(
                ndb_layout.shape, normalized_axes, ndb_layout.strides, keepdims
            )

            def reduce_worker(oi: Int) {imm}:
                var out_coord = IndexCalculator.index_to_coord(out_shape, oi)
                var base = walk.base_offset(out_coord, ndb_offset)
                var iter = walk.make_odometer(base, 0)
                var accum = Scalar[Self.dtype](0)
                for _ in range(walk.volume):
                    accum += ndb_buf[iter.off]
                    _ = iter.advance(walk)
                comptime if op_code == MEAN:
                    out.buffer[oi] = accum / reduced_volume
                else:
                    out.buffer[oi] = accum

            if (
                num_out >= n_threads
                and num_out * walk.volume >= n_threads * 32768
            ):
                parallelize(reduce_worker, num_out, n_threads)
            else:
                for oi in range(num_out):
                    reduce_worker(oi)

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
            var total_elements = ndb.numels()
            var n_threads = num_physical_cores()
            # Calibrated crossover from reduce_cpu (~5us parallelize launch
            # amortized at ≈ threads * 32768 elements).
            if total_elements >= n_threads * 32768:
                var n_segments = n_threads
                var partials = Buffer[Self.dtype](n_segments)
                var partials_data = partials.unsafe_ptr()
                var ndb_buf = ndb.buffer

                def seg_sum(seg: Int) {imm}:
                    var r0 = start + seg * total_elements // n_segments
                    var r1 = start + (seg + 1) * total_elements // n_segments
                    # Each segment is an independent SIMD range sum.
                    partials_data.unsafe_store(seg, ndb_buf.sum(r0, r1))

                parallelize(seg_sum, n_segments, n_threads)

                var accum = Scalar[Self.dtype](0)
                var n_seg = n_segments
                for seg in range(n_seg):
                    accum += partials_data.unsafe_load(seg)
                return accum
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
