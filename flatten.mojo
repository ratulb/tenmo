from tenmo import Tensor
from backpropagation import Delegate, BackwardFn, BACKWARD_FLATTEN
from operators import AddTensor
from common_utils import panic
from ancestry import Ancestor
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct FlattenBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_FLATTEN
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        ancestor = output.ancestry().get(0)
        ancestor_shape = ancestor.shape()
        # Just reshape gradient back to original
        var reshaped_grad = gradbox.reshape(ancestor_shape)
        return [
            (ancestor^, reshaped_grad^, AddTensor),
        ]


@register_passable
struct Flatten[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        start_dim: Int = 0,
        end_dim: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        flattened_buffer = self.buffer.flatten(start_dim, end_dim)
        out = Tensor[dtype](flattened_buffer^, requires_grad=False)

        # autograd hookup
        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                backward_fn = FlattenBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    pass
