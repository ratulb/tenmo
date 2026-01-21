from tenmo import Tensor
from operators import AddTensor, ZeroGrad
from backpropagation import BackwardFn, Delegate, BACKWARD_RESHAPE
from shapes import Shape
from validators import Validator
from gradbox import Gradbox
from ndbuffer import NDBuffer
from strides import Strides


@fieldwise_init
@register_passable
struct ReshapeBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_RESHAPE

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        var reshaped = gradbox.reshape(ancestor.shape())
        output.zero_grad()
        return [
            (ancestor^, reshaped^, AddTensor),
        ]

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct Reshape[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut tensor: Tensor[Self.dtype],
        new_shape: Shape,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
    ) -> Tensor[Self.dtype]:
        shape = new_shape if validated else Validator.validate_and_construct_new_shape(
            tensor.shape(), new_shape.intarray()
        )
        var out: Tensor[Self.dtype]
        if tensor.is_contiguous():
            # Calculate the correct offset and strides
            var new_offset = tensor.offset()
            var new_strides = Strides.default(shape)

            out = Tensor[Self.dtype].build_view(
                tensor,
                shape^,
                Optional(new_strides^),
                new_offset,
                requires_grad=False,
            )

        else:
            nd_buffer = tensor.buffer.contiguous(shape^)
            out = Tensor[Self.dtype](nd_buffer^, requires_grad=False)

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                backward_fn = ReshapeBackward[Self.dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(tensor)

        return out^
