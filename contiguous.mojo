from tenmo import Tensor
from backpropagation import Delegate, BackwardFn, BACKWARD_CONTIGUOUS
from operators import AddTensor
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct ContiguousBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_CONTIGUOUS

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        parent = output.ancestry().get(0)
        parent_shape = parent.shape()

        parent_gradbox = Gradbox[dtype].zeros(parent_shape)
        for coord in parent_shape:
            parent_gradbox[coord] = gradbox[coord]

        return [
            (parent^, parent_gradbox^, AddTensor),
        ]


@register_passable
struct Contiguous[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        dtype
    ]:
        var ndb = self.buffer.contiguous()
        var out = Tensor[dtype](ndb^, requires_grad=False)

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                if self.requires_grad and self.has_grad():
                    out.update_grad[AddTensor](self.grad())
                backward_fn = ContiguousBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main():
    print("passes")
