from .ndbuffer import NDBuffer
from .shared.intarray import IntArray
from .shared.shapes import Shape
from .shared.reduction_walk import ReductionWalk
from .shared.indexhelper import IndexCalculator
from max.algorithm import parallelize
from std.sys.info import num_physical_cores
from std.utils.numerics import min_or_neg_inf, max_or_inf


@fieldwise_init
struct MinMaxReducer[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    def reduce_minmax[
        is_max: Bool
    ](
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
    ) -> NDBuffer[Self.dtype]:
        """
                Returns the min/max values with output shape.
        Pure computation — no grad tracking.
        """
        var shape = ndb.shape
        var rank = shape.rank()
        var out_shape = shape.compute_output_shape(normalized_axes, keepdims)

        # Scalar input fast path
        if rank == 0:
            var result = NDBuffer[ndb.dtype].zeros(shape)
            result[IntArray()] = ndb[IntArray()]
            return result^

        # Full reduction fast path
        if out_shape == Shape():
            return Self._full_reduction_minmax[is_max](ndb, shape)

        # General partial reduction
        return Self._partial_reduction_minmax[is_max](
            ndb, shape, normalized_axes, keepdims, out_shape
        )

    @staticmethod
    def _full_reduction_minmax[
        is_max: Bool
    ](ndb: NDBuffer[Self.dtype], shape: Shape,) -> NDBuffer[Self.dtype]:
        var total_elements = shape.num_elements()
        var result = NDBuffer[ndb.dtype].zeros(Shape())

        if total_elements == 0:
            return result^

        var n_threads = num_physical_cores()
        var n_segments = 1 if total_elements < n_threads * 4096 else n_threads

        if n_segments == 1:
            # Serial full reduction (small input)
            var walk = ReductionWalk.build(
                shape,
                IntArray.range(0, shape.rank()),
                ndb.strides,
                True,
            )
            var iter = walk.make_odometer(ndb.offset, 0)
            var best_value: Scalar[Self.dtype]
            comptime if is_max:
                best_value = min_or_neg_inf[Self.dtype]()
            else:
                best_value = max_or_inf[Self.dtype]()

            for _ in range(total_elements):
                var cur = ndb.buffer[iter.off]
                comptime if is_max:
                    if cur > best_value:
                        best_value = cur
                else:
                    if cur < best_value:
                        best_value = cur
                _ = iter.advance(walk)

            result[IntArray()] = best_value
            return result^

        # Segmented parallel full reduction: each worker reduces a contiguous
        # flat chunk, then the main thread merges the per-segment partials.
        var contiguous = ndb.is_contiguous()
        var partials = NDBuffer[ndb.dtype].zeros(Shape(n_segments))

        # Hoist metadata for the hot parallel path.
        var ndb_layout = ndb.layout()
        var ndb_storage = ndb.buffer

        def seg_minmax(seg: Int) {imm}:
            var r0 = seg * total_elements // n_segments
            var r1 = (seg + 1) * total_elements // n_segments
            if r1 <= r0:
                return

            var local: Scalar[Self.dtype]
            comptime if is_max:
                local = min_or_neg_inf[Self.dtype]()
            else:
                local = max_or_inf[Self.dtype]()

            if contiguous:
                for flat_idx in range(r0, r1):
                    var cur = ndb_storage[ndb_layout.offset + flat_idx]
                    comptime if is_max:
                        if cur > local:
                            local = cur
                    else:
                        if cur < local:
                            local = cur
            else:
                # Full-shape stride walk seeded at this segment's window.
                var start_coords = IndexCalculator.index_to_coord(shape, r0)
                var start_off = ndb_layout.offset
                for d in range(shape.rank()):
                    start_off += start_coords[d] * ndb_layout.strides[d]
                var walk = ReductionWalk.build(
                    shape,
                    IntArray.range(0, shape.rank()),
                    ndb_layout.strides,
                    True,
                )
                var iter = walk.make_odometer_at(start_off, r0, start_coords)
                for _ in range(r0, r1):
                    var cur = ndb_storage[iter.off]
                    comptime if is_max:
                        if cur > local:
                            local = cur
                    else:
                        if cur < local:
                            local = cur
                    _ = iter.advance(walk)

            partials.buffer[seg] = local

        parallelize(seg_minmax, n_segments, n_threads)

        var best_value: Scalar[Self.dtype]
        comptime if is_max:
            best_value = min_or_neg_inf[Self.dtype]()
        else:
            best_value = max_or_inf[Self.dtype]()

        for seg in range(n_segments):
            var cur = partials.buffer[seg]
            comptime if is_max:
                if cur > best_value:
                    best_value = cur
            else:
                if cur < best_value:
                    best_value = cur

        result[IntArray()] = best_value
        return result^

    @staticmethod
    def _partial_reduction_minmax[
        is_max: Bool
    ](
        ndb: NDBuffer[Self.dtype],
        shape: Shape,
        normalized_axes: IntArray,
        keepdims: Bool,
        out_shape: Shape,
    ) -> NDBuffer[ndb.dtype]:
        var reduced_shape = shape.reduced_shape(normalized_axes)
        var num_output_elements = out_shape.num_elements()
        var result = NDBuffer[ndb.dtype].zeros(out_shape)

        var ndb_layout = ndb.layout()
        var ndb_buf = ndb.buffer
        var ndb_offset = ndb_layout.offset
        var walk = ReductionWalk.build(
            shape, normalized_axes, ndb_layout.strides, keepdims
        )

        def compute_output_element(out_flat_idx: Int) {imm}:
            var out_idx = IndexCalculator.index_to_coord(
                out_shape, out_flat_idx
            )
            var base = walk.base_offset(out_idx, ndb_offset)
            var iter = walk.make_odometer(base, 0)

            var best_value: Scalar[Self.dtype]

            comptime if is_max:
                best_value = min_or_neg_inf[Self.dtype]()
            else:
                best_value = max_or_inf[Self.dtype]()

            var first_iteration = True
            for _ in range(walk.volume):
                var cur = ndb_buf[iter.off]
                if first_iteration:
                    best_value = cur
                    first_iteration = False
                else:
                    comptime if is_max:
                        if cur > best_value:
                            best_value = cur
                    else:
                        if cur < best_value:
                            best_value = cur
                _ = iter.advance(walk)

            result.buffer[out_flat_idx] = best_value

        # Scalar coord loop per output element: parallel only when there is
        # enough work to amortize the ~5us launch (crossover ≈ threads*4096).
        var n_threads = num_physical_cores()
        var num_reduced_elements = reduced_shape.num_elements()
        if (
            num_output_elements >= n_threads
            and num_output_elements * num_reduced_elements >= n_threads * 4096
        ):
            parallelize(compute_output_element, num_output_elements, n_threads)
        else:
            for out_flat_idx in range(num_output_elements):
                compute_output_element(out_flat_idx)

        return result^

    @staticmethod
    def build_minmax_mask[
        is_max: Bool
    ](
        ndb: NDBuffer[Self.dtype],
        result: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
    ) -> NDBuffer[Self.dtype]:
        """
                Returns a normalised gradient mask of same shape as ndb.
        mask[i] = 1/tie_count  where ndb[i] == result at corresponding output slot.
        mask[i] = 0            otherwise.
        """
        var shape = ndb.shape
        var mask = NDBuffer[Self.dtype].zeros(shape)

        if shape.rank() == 0:
            mask[IntArray()] = Scalar[Self.dtype](1)
            return mask^

        if result.shape == Shape():
            # Full reduction — flat scan (no per-element coordinates) when
            # contiguous; legacy coord loop otherwise (strided reads need
            # the index mapping).
            var best = result[IntArray()]
            var total = shape.num_elements()
            if ndb.is_contiguous():
                var ndb_buf = ndb.buffer
                var ndb_offset = ndb.offset
                var mask_buf = mask.buffer
                var n_threads = num_physical_cores()
                var tie_count: Int = 0
                if total >= n_threads * 4096:
                    var ties = NDBuffer[DType.int32].zeros(Shape(n_threads))

                    def count_ties(t: Int) {imm}:
                        var r0 = t * total // n_threads
                        var r1 = (t + 1) * total // n_threads
                        var c = 0
                        for i in range(r0, r1):
                            if ndb_buf[ndb_offset + i] == best:
                                c += 1
                        ties.buffer[t] = Scalar[DType.int32](c)

                    parallelize(count_ties, n_threads, n_threads)
                    for t in range(n_threads):
                        tie_count += Int(ties.buffer[t])
                else:
                    for i in range(total):
                        if ndb_buf[ndb_offset + i] == best:
                            tie_count += 1
                if tie_count > 0:
                    var inv = Scalar[Self.dtype](1) / Scalar[Self.dtype](
                        tie_count
                    )
                    if total >= n_threads * 4096:

                        def write_mask(t: Int) {imm}:
                            var r0 = t * total // n_threads
                            var r1 = (t + 1) * total // n_threads
                            for i in range(r0, r1):
                                if ndb_buf[ndb_offset + i] == best:
                                    mask_buf[i] = inv

                        parallelize(write_mask, n_threads, n_threads)
                    else:
                        for i in range(total):
                            if ndb_buf[ndb_offset + i] == best:
                                mask_buf[i] = inv
                return mask^
            var tie_count: Int = 0
            for flat_idx in range(shape.num_elements()):
                var idx = IndexCalculator.index_to_coord(shape, flat_idx)
                if ndb[idx] == best:
                    tie_count += 1
            if tie_count > 0:
                var inv = Scalar[Self.dtype](1) / Scalar[Self.dtype](tie_count)
                for flat_idx in range(shape.num_elements()):
                    var idx = IndexCalculator.index_to_coord(shape, flat_idx)
                    if ndb[idx] == best:
                        mask[idx] = inv
            return mask^

        # Partial reduction — for each output slot, find tie count then write mask
        var out_shape = result.shape
        var num_output_elements = out_shape.num_elements()
        var n_threads = num_physical_cores()
        var ndb_layout = ndb.layout()
        var ndb_buf = ndb.buffer
        var ndb_offset = ndb_layout.offset
        var walk = ReductionWalk.build(
            shape, normalized_axes, ndb_layout.strides, keepdims
        )

        def mask_worker(out_flat_idx: Int) {imm}:
            var best = result.buffer[out_flat_idx]
            var base = walk.base_offset_from_flat(
                out_shape, out_flat_idx, ndb_offset
            )
            var base_flat = walk.base_logical_from_flat(out_shape, out_flat_idx)

            # Count ties for this output slot
            var iter = walk.make_odometer(base, base_flat)
            var tie_count: Int = 0
            for _ in range(walk.volume):
                if ndb_buf[iter.off] == best:
                    tie_count += 1
                _ = iter.advance(walk)

            # Write normalised mask
            if tie_count > 0:
                var inv = Scalar[Self.dtype](1) / Scalar[Self.dtype](tie_count)
                var witer = walk.make_odometer(base, base_flat)
                for _ in range(walk.volume):
                    if ndb_buf[witer.off] == best:
                        mask.buffer[witer.logical_pos] = inv
                    _ = witer.advance(walk)

        if (
            num_output_elements >= n_threads
            and num_output_elements * walk.volume >= n_threads * 4096
        ):
            parallelize(mask_worker, num_output_elements, n_threads)
        else:
            for out_flat_idx in range(num_output_elements):
                mask_worker(out_flat_idx)

        return mask^
