from .tensor import Tensor
from .shared.shapes import Shape
from .shared.intarray import IntArray
from .shared.indexhelper import IndexCalculator
from .shared.mnemonics import DEFAULT_INDEX_DTYPE
from .shared.panic import panic
from std.utils.numerics import min_finite, max_finite


@fieldwise_init
struct TopK[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward(
        tensor: Tensor[Self.dtype],
        k: Int,
        axis: Int = -1,
        largest: Bool = True,
        sorted: Bool = True,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) raises -> Tuple[Tensor[Self.dtype], Tensor[DEFAULT_INDEX_DTYPE]]:
        var shape = tensor.shape()
        var rank = shape.rank()
        if rank == 0:
            panic("topk: input must have at least one dimension")

        var ax = axis if axis >= 0 else axis + rank
        if ax < 0 or ax >= rank:
            panic("topk: axis out of range")
        var axis_dim = shape[ax]
        if k < 1 or k > axis_dim:
            panic("topk: k must be in [1, axis_dim]")

        var out_axes = IntArray()
        for i in range(rank):
            if i == ax:
                out_axes.append(k)
            else:
                out_axes.append(shape[i])
        var out_shape = Shape(out_axes)

        var values = Tensor[Self.dtype](out_shape)
        var indices = Tensor[DEFAULT_INDEX_DTYPE](out_shape)

        var outer_axes = IntArray()
        for i in range(rank):
            if i != ax:
                outer_axes.append(shape[i])
        var outer_shape = Shape(outer_axes)
        var num_rows = 1
        if outer_shape.rank() > 0:
            num_rows = outer_shape.num_elements()

        for row in range(num_rows):
            var outer_coords = IntArray()
            if outer_shape.rank() > 0:
                outer_coords = IndexCalculator.index_to_coord(outer_shape, row)

            var slice_vals = List[Scalar[Self.dtype]](capacity=axis_dim)
            for t in range(axis_dim):
                var coord = IntArray()
                var oi = 0
                for i in range(rank):
                    if i == ax:
                        coord.append(t)
                    else:
                        coord.append(outer_coords[oi])
                        oi += 1
                slice_vals.append(tensor[coord])

            var chosen = List[Bool](capacity=axis_dim)
            for _ in range(axis_dim):
                chosen.append(False)

            for pick in range(k):
                var best_pos = -1
                var best_val: Scalar[Self.dtype]
                if largest:
                    best_val = min_finite[Self.dtype]()
                    for t in range(axis_dim):
                        if not chosen[t]:
                            if best_pos == -1 or slice_vals[t] > best_val:
                                best_val = slice_vals[t]
                                best_pos = t
                else:
                    best_val = max_finite[Self.dtype]()
                    for t in range(axis_dim):
                        if not chosen[t]:
                            if best_pos == -1 or slice_vals[t] < best_val:
                                best_val = slice_vals[t]
                                best_pos = t
                chosen[best_pos] = True

                var out_coord = IntArray()
                var oi2 = 0
                for i in range(rank):
                    if i == ax:
                        out_coord.append(pick)
                    else:
                        out_coord.append(outer_coords[oi2])
                        oi2 += 1
                values[out_coord] = best_val
                indices[out_coord] = Scalar[DEFAULT_INDEX_DTYPE](best_pos)

        return values^, indices^

    @staticmethod
    def topk(
        tensor: Tensor[Self.dtype],
        k: Int,
        axis: Int = -1,
        largest: Bool = True,
        sorted: Bool = True,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) raises -> Tuple[Tensor[Self.dtype], Tensor[DEFAULT_INDEX_DTYPE]]:
        return Self.forward(
            tensor,
            k,
            axis=axis,
            largest=largest,
            sorted=sorted,
            requires_grad=requires_grad,
            sync=sync,
        )
