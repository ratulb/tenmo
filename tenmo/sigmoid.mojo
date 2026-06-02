from .tensor import Tensor
from .mnemonics import AddTensor, SIGMOID_BACKWARD
from .backpropagation import BackwardFnArg, BACKWARD_SIGMOID
from .gradbox import Gradbox
from .ancestry import Ancestor


@fieldwise_init
struct SigmoidBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)
        var ndb = output.buffer().arithmetic_ops[SIGMOID_BACKWARD](
            gradbox.buffer
        )
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^)

        parent.update_grad(gradbox_ancestor^, AddTensor, None)

        parent_ids.append(parent._id)
        if not retain_graph:
            gradbox.zero_grad()


struct Sigmoid[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var ndb = self.buffer.sigmoid()
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)
        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_SIGMOID
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^
