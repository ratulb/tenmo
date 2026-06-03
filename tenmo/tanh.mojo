from .tensor import Tensor
from .mnemonics import AddTensor, TANH_BACKWARD
from .backpropagation import BackwardFnArg, BACKWARD_TANH
from .gradbox import Gradbox
from .ancestry import Ancestor


@fieldwise_init
struct TanhBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref gradbox = output.gradients()
        ref parent = output.ancestry().get(0)
        var ndb = output.buffer().arithmetic_ops[TANH_BACKWARD](gradbox.buffer())
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^)

        parent.update_grad(gradbox_ancestor^, AddTensor, None)

        parent_ids.append(parent._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct Tanh[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var ndb = self.buffer.tanh()
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_TANH
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^
