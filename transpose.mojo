from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList
from validators import Validator


struct TransposeBackward[dtype: DType](Copyable & Movable):
    var axes: IntList

    fn __init__(out self, axes: IntList):
        self.axes = axes

    fn __copyinit__(out self, existing: Self):
        self.axes = existing.axes.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.axes = existing.axes^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: TensorLite[dtype]
    ) -> List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]:
        gradients = output.grad()
        ancestor = output.ancestry().get(0)
        inverted_axes = IntList.invert_permutation(self.axes)
        grad_transposed = gradients.transpose(inverted_axes)
        return [
            (
                ancestor,
                grad_transposed,
                AddTensor,
            )
        ]


@fieldwise_init
@register_passable
struct Transpose[dtype: DType](Copyable):
    fn __copyinit__(out self, existing: Self):
        pass

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut self: Tensor[dtype],
        axes: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = self.shape.copy()
        normalized_axes = (
            Validator.validate_and_normalize_axes(
                shape, axes, ordered=False, fill_missing=True
            ) if len(axes)
            > 0 else IntList.range_list(shape.rank()).reversed()
        )

        # Permute shape and create default strides and permute
        var new_shape = shape.permute(normalized_axes)
        var new_strides = self.strides.permute(normalized_axes)

        out = self.build_view(new_shape, new_strides, self.offset, False)

        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else self.requires_grad
            )

            if grad_required:
                out.requires_grad_(True)
                backward_fn = TransposeBackward[dtype](
                    normalized_axes
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(self)

        return out


fn main():
    pass
