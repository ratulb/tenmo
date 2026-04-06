from tenmo import Tensor
from backpropagation import Delegate, BackwardFn, BACKWARD_CONTIGUOUS
from mnemonics import AddTensor
from gradbox import Gradbox
from shapes import Shape


@fieldwise_init
struct ContiguousBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    comptime TAG = BACKWARD_CONTIGUOUS

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        ref parent_shape = parent.shape()
        var parent_gradbox: Gradbox[Self.dtype]
        if gradbox.shape() == Shape():
            parent_gradbox = Gradbox[Self.dtype].full(
                parent_shape,
                gradbox.item(),
                share=False,
                device=gradbox.device(),
            )
        else:
            parent_gradbox = Gradbox[Self.dtype].full(
                parent_shape,
                Scalar[Self.dtype](0),
                share=False,
                device=gradbox.device(),
            )
            for coord in parent_shape:
                parent_gradbox[coord] = gradbox[coord]

        return [
            (parent^, parent_gradbox^, AddTensor),
        ]


struct Contiguous[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var ndb = self.buffer.contiguous()
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                backward_fn = ContiguousBackward[
                    Self.dtype
                ]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^
