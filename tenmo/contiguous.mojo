from .tensor import Tensor
from .backpropagation import BackwardFnArg, BACKWARD_CONTIGUOUS
from .mnemonics import AddTensor
from .gradbox import Gradbox
from .shapes import Shape
from .ancestry import Ancestor


@fieldwise_init
struct ContiguousBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)
        ref parent_shape = parent.shape()
        var parent_gradbox: Gradbox[Self.dtype]
        if gradbox.shape() == Shape():
            parent_gradbox = Gradbox[Self.dtype].full(
                parent_shape,
                gradbox.item(),
                
                device=gradbox.device(),
            )
        else:
            parent_gradbox = Gradbox[Self.dtype].full(
                parent_shape,
                Scalar[Self.dtype](0),
                
                device=gradbox.device(),
            )
            for coord in parent_shape:
                parent_gradbox[coord] = gradbox[coord]

        parent.update_grad(parent_gradbox^, AddTensor, None)

        parent_ids.append(parent._id)
        if not retain_graph:
            gradbox.zero_grad()


struct Contiguous[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = False,
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
                backwardFnArg.needs_parent_data = True
                out.add_ancestry(backwardFnArg^, self)

        return out^
