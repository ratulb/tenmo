from tenmo import Tensor
from intlist import IntList
from common_utils import panic
from shapes import Shape
from utils.numerics import max_finite, min_finite


struct Argmin[dtype: DType]:
    @staticmethod
    fn argmin(
        tensor: Tensor[dtype],
        axis: Int = 0,
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

        # Output shape is same as input but without the reduced axis
        var out_axes = IntList()
        for i in range(rank):
            if i != ax:
                out_axes.append(shape[i])
        out_shape = Shape(out_axes)

        var out = Tensor[DType.int32].zeros(out_shape)

        # Iterate over all indices of output shape
        for out_idx in out_shape.indices():
            var min_val = max_finite[dtype]()
            var min_pos = 0

            # Loop over reduced axis
            for idx in range(shape[axis]):
                var full_idx = out_idx.insert(ax, idx)
                val = tensor[full_idx]
                if val < min_val:
                    min_val = val
                    min_pos = idx
            out[out_idx] = min_pos

        return out^


struct Argmax[dtype: DType]:
    @staticmethod
    fn argmax(
        tensor: Tensor[dtype],
        axis: Int = 0,
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
        # Output shape is same as input but without the reduced axis
        var out_axes = IntList()
        for i in range(rank):
            if i != ax:
                out_axes.append(shape[i])
        out_shape = Shape(out_axes)

        # Output shape is same as input but without the reduced axis
        out = Tensor[DType.int32].zeros(out_shape)

        # Iterate over all indices of output shape
        for out_idx in out_shape.indices():
            max_val = min_finite[dtype]()
            max_pos = 0

            # Loop over reduced axis
            for idx in range(shape[ax]):
                full_idx = out_idx.insert(ax, idx)
                var val = tensor[full_idx]
                if val > max_val:
                    max_val = val
                    max_pos = idx
            out[out_idx] = max_pos

        return out^


fn main() raises:
    print("passes")
