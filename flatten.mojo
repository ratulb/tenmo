from tenmo import Tensor
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from common_utils import panic
from ancestry import Ancestor
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct FlattenBackward[dtype: DType](ImplicitlyCopyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ancestor = output.ancestry().get(0)
        ancestor_shape = ancestor.shape()
        # Just reshape gradient back to original
        var reshaped_grad = output.grad().reshape(ancestor_shape)
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
        rank = self.rank()
        var endd = end_dim.value() if end_dim else rank - 1
        if endd < start_dim:
            panic("Flatten → end_dim must be >= start_dim")

        var original_shape = self.shape()
        # compute new shape
        collapsed = original_shape[start_dim : endd + 1].product()
        new_shape = (
            original_shape[:start_dim]
            + [collapsed]
            + original_shape[endd + 1 :]
        )
        nd_buffer = self.buffer.contiguous(new_shape)
        out = Tensor[dtype](nd_buffer^, requires_grad=False)

        # autograd hookup
        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else self.requires_grad
            )
            if grad_required:
                out.requires_grad_(True)
                backward_fn = FlattenBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    pass
