from tensors import Tensor
from intlist import IntList
from operators import AddTensor
from shared import TensorLite
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from validators import Validator


@register_passable
struct MeanBackward[dtype: DType](Copyable):
    var axes: IntList
    var keepdims: Bool

    fn __init__(
        out self, axes: IntList = IntList.Empty, keepdims: Bool = False
    ):
        self.axes = axes
        self.keepdims = keepdims

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes.copy()
        self.keepdims = other.keepdims

        _ = """fn __moveinit__(out self, deinit other: Self):
        self.axes = other.axes
        self.keepdims = other.keepdims"""

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        if gradients.shape == Shape.Void:
            scalar_grad = gradients.item() / ancestor.shape().num_elements()
            grad_contrib = Tensor[dtype].full(
                ancestor.shape(),
                scalar_grad,
                requires_grad=False,
            )
            return [
                (
                    ancestor,
                    grad_contrib,
                    AddTensor,
                )
            ]

        var expanded = gradients

        if not self.keepdims:
            expanded = gradients.reshape(
                Shape(
                    gradients.shape.intlist().insert(
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
                ancestor,
                average,
                AddTensor,
            )
        ]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


@register_passable
struct Mean[dtype: DType]:
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
            tensor.shape, axes
        )
        count = tensor.shape.axes_spans.select(normalized_axes).product()
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
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(tensor))

        return out


fn main():
    print("passes")
