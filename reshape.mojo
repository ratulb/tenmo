from tenmo import Tensor
from operators import AddTensor, ZeroGrad
from backpropagation import BackwardFn, Delegate
from shapes import Shape
from validators import Validator
from ancestry import Ancestor
from gradbox import Gradbox
from ndbuffer import NDBuffer
from strides import Strides


@fieldwise_init
@register_passable
struct ReshapeBackward[dtype: DType](ImplicitlyCopyable):
    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        ancestor = output.ancestry().get(0)
        reshaped = gradbox.reshape(ancestor.shape())
        output.zero_grad()
        return [
            (ancestor^, reshaped^, AddTensor),
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
            tensor.shape(), new_shape.intarray()
        )
        var out: Tensor[dtype]
        if tensor.is_contiguous():
            # Calculate the correct offset and strides
            var new_offset = tensor.offset()
            var new_strides = Strides.default(shape)

            out = Tensor[dtype].build_view(
                tensor.address(),
                shape^,
                Optional(new_strides^),
                new_offset,
                requires_grad=False,
            )

        else:
            nd_buffer = tensor.buffer.contiguous(shape^)
            out = Tensor[dtype](nd_buffer^, requires_grad=False)

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                backward_fn = ReshapeBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(tensor)

        return out^


fn main():
    print("passes")
