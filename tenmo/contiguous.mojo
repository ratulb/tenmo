from .tensor import Tensor
from .backpropagation import BackwardFnArg, BACKWARD_CONTIGUOUS
from .mnemonics import AddTensor
from .gradbox import Gradbox
from .shapes import Shape
from .ancestry import Ancestor


@fieldwise_init
struct ContiguousBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
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
            (parent, parent_gradbox^, AddTensor),
        ]


struct Contiguous[dtype: DType](ImplicitlyCopyable, RegisterPassable):
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
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_CONTIGUOUS
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^

