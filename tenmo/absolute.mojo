from .tensor import Tensor
from .shared.mnemonics import AddTensor, ABS_BACKWARD
from .backpropagation import BackwardFn, BackwardFnType

from .gradbox import Gradbox
from .ancestry import Ancestor


@fieldwise_init
struct AbsBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)
        var ndb = parent.buffer().arithmetic_ops[ABS_BACKWARD](gradbox.buffer())
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^)

        parent.update_grad(gradbox_ancestor^, AddTensor, None)

        parent_ids.append(parent._id)

        gradbox.zero_grad()


@fieldwise_init
struct Absolute[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var ndb = self.buffer.__abs__()
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.null_arg[Self.dtype](
                    AbsBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, self)

        return out^
