from tenmo import Tensor
from intarray import IntArray
from common_utils import panic
from shapes import Shape
from utils.numerics import max_finite, min_finite


struct Argmin[dtype: DType]:
    @staticmethod
    fn argmin(
        tensor: Tensor[dtype],
        axis: Int = 0,
        keepdims: Bool = False,
    ) -> Tensor[DType.int32]:
        shape = tensor.shape()
        rank = shape.rank()
        ax = axis if axis >= 0 else axis + rank
        if ax < 0 or ax >= rank:
            panic(
                "Tensor → argmin: axis",
                axis.__str__(),
                "out of range for tensor with rank",
                rank.__str__(),
            )

        # --- Build output shape ---
        var out_axes = IntArray()
        for i in range(rank):
            if i == ax:
                if keepdims:
                    out_axes.append(1)
            else:
                out_axes.append(shape[i])
        out_shape = Shape(out_axes)

        # --- Allocate output tensor ---
        var out = Tensor[DType.int32].zeros(out_shape)

        # --- Iterate over output indices ---
        for out_idx in out_shape:
            var min_val = max_finite[dtype]()
            var min_pos = 0

            # Determine the reduced index position to insert
            for idx in range(shape[ax]):
                var full_idx = out_idx.insert(
                    ax, idx
                ) if not keepdims else out_idx.replace(ax, idx)
                var val = tensor[full_idx]
                if val < min_val:
                    min_val = val
                    min_pos = idx

            # For keepdims=True, write at [ax]=0
            if keepdims:
                var write_idx = out_idx.replace(ax, 0)
                out[write_idx] = min_pos
            else:
                out[out_idx] = min_pos

        return out^


struct Argmax[dtype: DType]:
    @staticmethod
    fn argmax(
        tensor: Tensor[dtype],
        axis: Int = 0,
        keepdims: Bool = False,
    ) -> Tensor[DType.int32]:
        shape = tensor.shape()
        rank = shape.rank()
        ax = axis if axis >= 0 else axis + rank
        if ax < 0 or ax >= rank:
            panic(
                "Tensor → argmax: axis",
                axis.__str__(),
                "out of range for tensor with rank",
                rank.__str__(),
            )

        # --- Build output shape ---
        var out_axes = IntArray()
        for i in range(rank):
            if i == ax:
                if keepdims:
                    out_axes.append(1)
            else:
                out_axes.append(shape[i])
        out_shape = Shape(out_axes)

        # --- Allocate output tensor ---
        var out = Tensor[DType.int32].zeros(out_shape)

        # --- Iterate over output indices ---
        for out_idx in out_shape:
            var max_val = min_finite[dtype]()
            var max_pos = 0

            # Determine the reduced index position to insert
            for idx in range(shape[ax]):
                var full_idx = out_idx.insert(
                    ax, idx
                ) if not keepdims else out_idx.replace(ax, idx)
                var val = tensor[full_idx]
                if val > max_val:
                    max_val = val
                    max_pos = idx

            # For keepdims=True, write at [ax]=0
            if keepdims:
                var write_idx = out_idx.replace(ax, 0)
                out[write_idx] = max_pos
            else:
                out[out_idx] = max_pos

        return out^
