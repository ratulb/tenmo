from tenmo import Tensor
from intlist import IntList
from operators import AddTensor
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from validators import Validator
from ancestry import Ancestor
from gradbox import Gradbox


@register_passable
struct MeanBackward[dtype: DType](ImplicitlyCopyable):
    var axes: IntList
    var keepdims: Bool

    fn __init__(out self, axes: IntList = IntList(), keepdims: Bool = False):
        self.axes = axes
        self.keepdims = keepdims

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes.copy()
        self.keepdims = other.keepdims

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        gradbox_shape = gradbox.shape()
        ancestor = output.ancestry().get(0)
        if gradbox_shape == Shape():
            scalar_grad = gradbox.item() / ancestor.shape().num_elements()
            grad_contrib = Gradbox[dtype].full(
                ancestor.shape(),
                scalar_grad,
            )

            return [
                (
                    ancestor^,
                    grad_contrib^,
                    AddTensor,
                )
            ]

        var expanded = gradbox.copy()

        if not self.keepdims:
            expanded = expanded.reshape(
                Shape(
                    gradbox_shape.intlist().insert(
                        self.axes,
                        IntList.with_capacity(len(self.axes), 1),
                    )
                )
            )

        # Broadcast and divide
        broadcasted = expanded.broadcast_to(ancestor.shape())
        # Compute total count of elements being reduced
        count = ancestor.shape().axes_spans.select(self.axes).product()
        average = broadcasted / Scalar[dtype](count)

        return [
            (
                ancestor^,
                average^,
                AddTensor,
            )
        ]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


@fieldwise_init
@register_passable
struct Mean[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[dtype],
        axes: IntList,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        normalized_axes = Validator.validate_and_normalize_axes(
            tensor.shape(), axes
        )
        count = tensor.shape().axes_spans.select(normalized_axes).product()
        out = tensor.sum[track_grad=False](
            axes=normalized_axes, keepdims=keepdims, requires_grad=False
        ) / Scalar[dtype](count)

        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else tensor.requires_grad
            )
            if grad_required:
                out.requires_grad_(True)
                backward_fn = MeanBackward[dtype](
                    normalized_axes.copy(), keepdims
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(tensor)

        return out^


fn main():
    print("passes")
