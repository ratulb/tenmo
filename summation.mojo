from tenmo import Tensor
from operators import AddTensor
from intarray import IntArray
from shapes import Shape
from backpropagation import Delegate, BackwardFn, BACKWARD_SUM
from validators import Validator
from ancestry import Ancestor
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct SumBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_SUM
    var axes: IntArray
    var keepdims: Bool

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        ancestor = output.ancestry().get(0)
        rank = ancestor.shape().rank()
        if rank == 0:
            return [(ancestor^, gradbox.copy(), AddTensor)]
        shape = ancestor.shape()

        var grad_contrib: Gradbox[dtype]

        # Handle scalar gradient case (sum reduced to scalar)
        if gradbox.shape() == Shape():
            grad_contrib = Gradbox[dtype].full(
                shape,
                gradbox.item(),
                share=False,
            )
        else:
            # Handle keepdims=False case (need to reshape gradient)
            if not self.keepdims:
                # Determine axes/unsqueeze (insert dims of size 1)
                axes = (
                    gradbox.shape()
                    .intarray()
                    .insert(
                        self.axes,
                        IntArray.filled(len(self.axes), 1),
                    )
                )
                unsqueezed_shape = Shape(axes)

                unsqueezed_grad = gradbox.reshape(unsqueezed_shape)
                grad_contrib = unsqueezed_grad.broadcast_to(shape, share=False)
            else:
                # keepdims=True: shapes match except for broadcasting
                grad_contrib = gradbox.broadcast_to(shape, share=False)

        return [
            (
                ancestor^,
                grad_contrib^,
                AddTensor,
            )
        ]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct Summer[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = tensor.shape()
        reduction_axes = Validator.validate_and_normalize_axes(shape, axes)
        var nd_buffer = tensor.buffer.sum(reduction_axes, keepdims)
        var out = Tensor[dtype](nd_buffer^, requires_grad=False)

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                backward_fn = SumBackward[dtype](
                    reduction_axes, keepdims
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(tensor)

        return out^


fn main():
    print("passes")
