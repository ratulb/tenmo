# Welford online mean/variance — tenmo/welford.mojo
#
# Extracted from NDBuffer to keep the core infrastructure lean.
# Single-pass Welford accumulator: mean and variance computed simultaneously.
#
# Consumers:
#   Variance.forward  (variance.mojo)
#   StdDev.forward    (std_deviation.mojo)
#   LayerNorm.forward  (layernorm.mojo)

from .ndbuffer import NDBuffer
from .shared.intarray import IntArray
from .shared.shapes import Shape
from .shared.reduction_walk import ReductionWalk
from .shared.indexhelper import IndexCalculator
from .shared.mnemonics import Divide
from .shared.panic import panic
from .kernels.reduction_kernel import ReductionKernel
from std.sys import has_accelerator, simd_width_of
from max.algorithm import parallelize
from std.sys.info import num_physical_cores


struct Welford[dtype: DType]:
    """Single-pass online mean/variance via Welford's algorithm.

    Computes mean and variance in one pass. Mean is unsafe_free — Welford computes
    it anyway. GPU dispatch uses ReductionKernel.launch_welford; CPU path uses
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
        sync: Bool = True,
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
        sync: Bool = True,
    ) raises -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        var (mean_pair, M2_pair) = ReductionKernel[Self.dtype].launch_welford(
            ndb.layout(), ndb.device_state.value(), axes, keepdims, sync=False
        )
        var mean_ndb = NDBuffer[Self.dtype].with_layout_device_state(
            mean_pair[0], mean_pair[1]
        )
        var M2_ndb = NDBuffer[Self.dtype].with_layout_device_state(
            M2_pair[0], M2_pair[1]
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
            # Global scalar reduction. For large contiguous inputs, split the
            # flat index space across cores; each segment runs the classic
            # Welford recurrence, then the (count, mean, M2) pairs are merged
            # serially with Chan et al.'s numerically-stable parallel combine.
            var total_elements = ndb.numels()
            var n_threads = num_physical_cores()
            if ndb.is_contiguous() and total_elements >= n_threads * 4096:
                var n_segments = n_threads
                var seg_mean = NDBuffer[Self.dtype].zeros(Shape(n_segments))
                var seg_m2 = NDBuffer[Self.dtype].zeros(Shape(n_segments))
                var seg_cnt = NDBuffer[DType.int64].zeros(Shape(n_segments))
                var ndb_offset = ndb.offset
                var ndb_buf = ndb.buffer

                def seg_welford(seg: Int) {imm}:
                    var r0 = seg * total_elements // n_segments
                    var r1 = (seg + 1) * total_elements // n_segments
                    var l_mean = Scalar[Self.dtype](0)
                    var l_m2 = Scalar[Self.dtype](0)
                    var l_cnt = 0
                    for i in range(r0, r1):
                        var x = ndb_buf[ndb_offset + i]
                        l_cnt += 1
                        var delta = x - l_mean
                        l_mean += delta / Scalar[Self.dtype](l_cnt)
                        var delta2 = x - l_mean
                        l_m2 += delta * delta2
                    seg_mean.buffer[seg] = l_mean
                    seg_m2.buffer[seg] = l_m2
                    seg_cnt.buffer[seg] = Scalar[DType.int64](l_cnt)

                parallelize(seg_welford, n_segments, n_threads)

                var total_mean = Scalar[Self.dtype](0)
                var total_m2 = Scalar[Self.dtype](0)
                var total_cnt: Int = 0
                for seg in range(n_segments):
                    var s_cnt = Int(seg_cnt.buffer[seg])
                    if s_cnt == 0:
                        continue
                    var s_mean = seg_mean.buffer[seg]
                    var s_m2 = seg_m2.buffer[seg]
                    if total_cnt == 0:
                        total_mean = s_mean
                        total_m2 = s_m2
                        total_cnt = s_cnt
                        continue
                    var new_cnt = total_cnt + s_cnt
                    var delta = s_mean - total_mean
                    total_mean += (
                        delta
                        * Scalar[Self.dtype](s_cnt)
                        / Scalar[Self.dtype](new_cnt)
                    )
                    total_m2 += (
                        s_m2
                        + delta
                        * delta
                        * Scalar[Self.dtype](total_cnt)
                        * Scalar[Self.dtype](s_cnt)
                        / Scalar[Self.dtype](new_cnt)
                    )
                    total_cnt = new_cnt

                mean_out[IntArray()] = total_mean
                var_out[IntArray()] = total_m2 / divisor
            else:
                # Serial fallback: non-contiguous layouts or small inputs.
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
            var reduction_axes_shape = ndb.shape.reduced_shape(axes)

            # Fast path: suffix-contiguous reduction (eliminates coord index overhead)
            if ndb.is_contiguous():
                var rank = ndb.shape.ndim()
                var num_axes = axes.size()
                var is_suffix = num_axes > 0 and axes[num_axes - 1] == rank - 1
                var idx = 0
                while is_suffix and idx < num_axes - 1:
                    if axes[idx] != rank - num_axes + idx:
                        is_suffix = False
                        break
                    idx += 1
                if is_suffix:
                    var reduced_numels = reduction_axes_shape.product()
                    var num_out = mean_out.numels()
                    var n_threads = num_physical_cores()
                    var ndb_offset = ndb.offset
                    var ndb_buf = ndb.buffer

                    def welford_row(oi: Int) {imm}:
                        var base = ndb_offset + oi * reduced_numels
                        # Numerically-stable two-pass: SIMD row sum → mean, then
                        # SIMD Σ(x-mean)². Avoids per-element divides entirely.
                        var row_sum = ndb_buf.sum(base, base + reduced_numels)
                        var row_mean = row_sum / Scalar[Self.dtype](
                            reduced_numels
                        )
                        comptime simd_width = simd_width_of[Self.dtype]()
                        var simd_mean = SIMD[Self.dtype, simd_width](row_mean)
                        var vectorized_end = (
                            reduced_numels // simd_width
                        ) * simd_width
                        var m2_accum = SIMD[Self.dtype, simd_width](0)
                        for ri in range(0, vectorized_end, simd_width):
                            var diff = (
                                ndb_buf.load[simdwidth=simd_width](
                                    base + ri
                                )
                                - simd_mean
                            )
                            m2_accum += diff * diff
                        var row_M2 = m2_accum.reduce_add()
                        for ri in range(vectorized_end, reduced_numels):
                            var tail_diff = ndb_buf[base + ri] - row_mean
                            row_M2 += tail_diff * tail_diff
                        mean_out.buffer[oi] = row_mean
                        var_out.buffer[oi] = row_M2 / divisor

                    # Parallel only when there is enough work to amortize the
                    # ~5us launch cost (measured crossover ≈ threads * 4096).
                    if (
                        num_out >= n_threads
                        and num_out * reduced_numels >= n_threads * 4096
                    ):
                        parallelize(welford_row, num_out, n_threads)
                    else:
                        for oi in range(num_out):
                            welford_row(oi)
                    return (mean_out^, var_out^)

            # Fallback: works for any layout / any axes. Hoists the output
            # coordinate once per output slot and walks the reduced axes with
            # stride arithmetic — no per-element IntArray alloc.
            var ndb_layout = ndb.layout()
            var ndb_buf = ndb.buffer
            var ndb_offset = ndb_layout.offset
            var num_out = mean_out.numels()
            var n_threads = num_physical_cores()
            var walk = ReductionWalk.build(
                ndb_layout.shape, axes, ndb_layout.strides, keepdims
            )

            def welford_worker(oi: Int) {imm}:
                var out_coord = IndexCalculator.index_to_coord(out_shape, oi)
                var base = walk.base_offset(out_coord, ndb_offset)
                var iter = walk.make_odometer(base, 0)
                var local_mean = Scalar[Self.dtype](0)
                var local_M2 = Scalar[Self.dtype](0)
                var count = 0
                for _ in range(walk.volume):
                    var x = ndb_buf[iter.off]
                    count += 1
                    var delta = x - local_mean
                    local_mean += delta / Scalar[Self.dtype](count)
                    var delta2 = x - local_mean
                    local_M2 += delta * delta2
                    _ = iter.advance(walk)
                mean_out.buffer[oi] = local_mean
                var_out.buffer[oi] = local_M2 / divisor

            if (
                num_out >= n_threads
                and num_out * walk.volume >= n_threads * 4096
            ):
                parallelize(welford_worker, num_out, n_threads)
            else:
                for oi in range(num_out):
                    welford_worker(oi)

        return (mean_out^, var_out^)
