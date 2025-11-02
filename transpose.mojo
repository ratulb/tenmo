from tenmo import Tensor
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList
from validators import Validator
from gradbox import Gradbox
from ancestry import Ancestor

@fieldwise_init
@register_passable
struct TransposeBackward[dtype: DType](ImplicitlyCopyable):
    var axes: IntList

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        gradbox = output.grad().copy()
        ancestor = output.ancestry().get(0)
        inverted_axes = IntList.invert_permutation(self.axes)
        gradbox_transposed = gradbox.transpose(inverted_axes)
        return [
            (
                ancestor^,
                gradbox_transposed^,
                AddTensor,
            )
        ]


@fieldwise_init
@register_passable
struct Transpose[dtype: DType](Copyable):

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        axes: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = self.shape()
        normalized_axes = (
            Validator.validate_and_normalize_axes(
                shape, axes, ordered=False, fill_missing=True
            ) if len(axes)
            > 0 else IntList.range_list(shape.rank()).reversed()
        )

        # Permute shape and create default strides and permute
        var new_shape = shape.permute(normalized_axes)
        var new_strides = self.strides().permute(normalized_axes)

        out = Tensor[dtype].build_view(
            self.address(), new_shape, new_strides, self.offset(), requires_grad=False
        )

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
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main():
    pass
