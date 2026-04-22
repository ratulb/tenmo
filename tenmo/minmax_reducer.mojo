from .ndbuffer import NDBuffer
from .intarray import IntArray
from .shapes import Shape
from .indexhelper import IndexCalculator
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from std.utils.numerics import min_or_neg_inf, max_or_inf


@fieldwise_init
struct MinMaxReducer[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn reduce_minmax[
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
    fn _full_reduction_minmax[
        is_max: Bool
    ](ndb: NDBuffer[Self.dtype], shape: Shape,) -> NDBuffer[Self.dtype]:
        var total_elements = shape.num_elements()
        var result = NDBuffer[ndb.dtype].zeros(Shape())

        # Initialize with first element
        var first_idx = shape.first_index()
        var best_value = ndb[first_idx]

        for flat_idx in range(1, total_elements):
            var idx = IndexCalculator.index_to_coord(shape, flat_idx)
            var cur = ndb[idx]

            comptime if is_max:
                if cur > best_value:
                    best_value = cur
            else:
                if cur < best_value:
                    best_value = cur

        result[IntArray()] = best_value
        return result^

    @staticmethod
    fn _partial_reduction_minmax[
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

        @parameter
        fn compute_output_element(out_flat_idx: Int):
            var out_idx = IndexCalculator.index_to_coord(
                out_shape, out_flat_idx
            )

            var best_value: Scalar[Self.dtype]

            comptime if is_max:
                best_value = min_or_neg_inf[Self.dtype]()
            else:
                best_value = max_or_inf[Self.dtype]()

            var first_iteration = True
            var num_reduced_elements = reduced_shape.num_elements()

            for red_flat_idx in range(num_reduced_elements):
                var red_idx = IndexCalculator.index_to_coord(
                    reduced_shape, red_flat_idx
                )
                var full_idx = out_idx.replace(
                    normalized_axes, red_idx
                ) if keepdims else out_idx.insert(normalized_axes, red_idx)

                var cur = ndb[full_idx]

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

            result[out_idx] = best_value

        parallelize[compute_output_element](
            num_output_elements, num_physical_cores()
        )

        return result^

    @staticmethod
    fn build_minmax_mask[
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
            # Full reduction — scan all elements
            var best = result[IntArray()]
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
        var reduced_shape = shape.reduced_shape(normalized_axes)
        var num_output_elements = out_shape.num_elements()

        for out_flat_idx in range(num_output_elements):
            var out_idx = IndexCalculator.index_to_coord(
                out_shape, out_flat_idx
            )
            var best = result[out_idx]

            # Count ties for this output slot
            var tie_count: Int = 0
            var num_reduced = reduced_shape.num_elements()
            for red_flat_idx in range(num_reduced):
                var red_idx = IndexCalculator.index_to_coord(
                    reduced_shape, red_flat_idx
                )
                var full_idx = out_idx.insert(
                    normalized_axes, red_idx
                ) if not keepdims else out_idx.replace(normalized_axes, red_idx)
                if ndb[full_idx] == best:
                    tie_count += 1

            # Write normalised mask
            if tie_count > 0:
                var inv = Scalar[Self.dtype](1) / Scalar[Self.dtype](tie_count)
                for red_flat_idx in range(num_reduced):
                    var red_idx = IndexCalculator.index_to_coord(
                        reduced_shape, red_flat_idx
                    )
                    var full_idx = out_idx.insert(
                        normalized_axes, red_idx
                    ) if not keepdims else out_idx.replace(
                        normalized_axes, red_idx
                    )
                    if ndb[full_idx] == best:
                        mask[full_idx] = inv

        return mask^


from .buffers import Buffer


fn main() raises:
    comptime dtype = DType.float32
    ndb = NDBuffer[dtype](Buffer[dtype]([2, 2, 3, 4, 2, 6]), Shape(2, 3))
    ndb.print()
    var res1 = MinMaxReducer[dtype].reduce_minmax[True](ndb, IntArray(), False)
    var res2 = MinMaxReducer[dtype].reduce_minmax[True](ndb, IntArray(1), False)

    res1.print()
    res2.print()
