from tenmo import Tensor
from intarray import IntArray
from operators import AddTensor
from shapes import Shape
from backpropagation import Delegate, BackwardFn, BACKWARD_MEAN
from validators import Validator
from gradbox import Gradbox
from forwards import DivideByScalar


@register_passable
struct MeanBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_MEAN
    var axes: IntArray
    var keepdims: Bool

    fn __init__(out self, axes: IntArray = IntArray(), keepdims: Bool = False):
        self.axes = axes
        self.keepdims = keepdims

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes.copy()
        self.keepdims = other.keepdims

    fn backward(
        self, read output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        gradbox_shape = gradbox.shape()
        var ancestor = output.ancestry().get(0)
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
                    gradbox_shape.intarray().insert(
                        self.axes,
                        IntArray.filled(len(self.axes), 1),
                    )
                )
            )

        # Broadcast and divide
        var broadcasted = expanded.broadcast_to(ancestor.shape())
        # Compute total count of elements being reduced
        var count = ancestor.shape().reduced_shape(self.axes).product()
        count = count if count > 0 else 1
        var average = broadcasted / Scalar[dtype](count)

        return [
            (
                ancestor^,
                average^,
                AddTensor,
            )
        ]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct Mean[dtype: DType](Copyable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        normalized_axes = Validator.validate_and_normalize_axes(
            tensor.shape(), axes
        )
        var count = tensor.shape().reduced_shape(normalized_axes).product()
        count = count if count > 0 else 1
        total = tensor.sum[track_grad=False](
            axes=normalized_axes, keepdims=keepdims
        )
        var out = DivideByScalar[dtype].forward[track_grad=False](total, count)

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                backward_fn = MeanBackward[dtype](
                    normalized_axes.copy(), keepdims
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(tensor)

        return out^

    @always_inline
    @staticmethod
    fn forward(
        gradbox: Gradbox[dtype],
        axes: IntArray,
        keepdims: Bool = False,
    ) -> Gradbox[dtype]:
        var gradbox_shape = gradbox.shape()
        normalized_axes = Validator.validate_and_normalize_axes(
            gradbox_shape, axes
        )
        var count = gradbox_shape.reduced_shape(normalized_axes).product()
        count = count if count > 0 else 1
        out = gradbox.sum(axes=normalized_axes, keepdims=keepdims) / Scalar[
            dtype
        ](count)

        return out^


fn main():
    print("passes")
