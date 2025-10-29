from tenmo import Tensor
from operators import AddTensor, ZeroGrad
from backpropagation import BackwardFn, Delegate
from shapes import Shape
from validators import Validator
from ancestry import Ancestor
from gradbox import Gradbox
from ndbuffer import NDBuffer


@fieldwise_init
@register_passable
struct ReshapeBackward[dtype: DType](ImplicitlyCopyable):
    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        gradbox = output.grad().copy()
        ancestor = output.ancestry().get(0)
        reshaped = gradbox.reshape(ancestor.shape())

        return [
            (ancestor^, reshaped^, AddTensor),
            (Ancestor(output), gradbox^, ZeroGrad),
        ]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


@fieldwise_init
@register_passable
struct Reshape[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[dtype],
        new_shape: Shape,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
    ) -> Tensor[dtype]:
        shape = new_shape if validated else Validator.validate_and_construct_new_shape(
            tensor.shape(), new_shape.intlist()
        )

        buffer = tensor.buffer.contiguous_buffer()
        nd_buffer = NDBuffer[dtype](buffer^, shape^)
        out = Tensor[dtype](nd_buffer^, requires_grad=False)

        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else tensor.requires_grad
            )

            if grad_required:
                out.requires_grad_(True)
                backward_fn = ReshapeBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(tensor)

        return out^


fn main():
    print("passes")
