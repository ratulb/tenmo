from .tensor import Tensor
from .shared.mnemonics import AddTensor, NEGATE
from .backpropagation import BackwardFn, BackwardFnType

from .gradbox import Gradbox
from .ancestry import Ancestor


@fieldwise_init
struct NegateBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        # d/dx (-x) = -1, so the incoming gradient just gets negated.
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)
        var ndb = gradbox.buffer().unary_ops[NEGATE]()
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^)

        parent.update_grad(gradbox_ancestor^, AddTensor, None)

        parent_ids.append(parent._id)

        gradbox.zero_grad()


@fieldwise_init
struct Negate[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var ndb = self.buffer.unary_ops[NEGATE](sync=sync)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.null_arg[Self.dtype](
                    NegateBackward[Self.dtype]()
                )
                out.add_ancestry(backwardFn^, self)

        return out^
