from tensors import Tensor
from intlist import IntList
from shapes import Shape
from operators import AddTensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn


struct MeanBackward[dtype: DType](Copyable & Movable & Stringable):
    var axes: IntList
    var keepdims: Bool

    fn __init__(
        out self, axes: IntList = IntList.Empty, keepdims: Bool = False
    ):
        self.axes = axes
        self.keepdims = keepdims

    fn __moveinit__(out self, var other: Self):
        self.axes = other.axes
        self.keepdims = other.keepdims

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes
        self.keepdims = other.keepdims

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

        scaled = broadcasted / Scalar[dtype](count)
        return [
            (
                ancestor,
                scaled,
                AddTensor,
            )
        ]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn __str__(self) -> String:
        return "MeanBackward"

fn main():
    print("passes")
