from .tensor import Tensor
from .shared.mnemonics import AddTensor, TANH_BACKWARD
from .backpropagation import BackwardFn, NDBufferArg, BackwardFnType

from .gradbox import Gradbox
from .ancestry import Ancestor


@fieldwise_init
struct TanhBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref bwd_arg = (
            output.ancestry().backward_fn().get[NDBufferArg[Self.dtype]]()
        )
        var out_ndb = bwd_arg.ndb
        ref gradbox = output.gradients()
        ref parent = output.ancestry().get(0)
        var ndb = out_ndb.arithmetic_ops[TANH_BACKWARD](gradbox.buffer())
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^)

        parent.update_grad(gradbox_ancestor^, AddTensor, None)

        parent_ids.append(parent._id)
        gradbox.zero_grad()


@fieldwise_init
struct Tanh[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var ndb = self.buffer.tanh()
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var out_ndb = out.buffer.copy()
                var backwardFn = BackwardFn.from_ndbuffer[Self.dtype](
                    out_ndb^, TanhBackward[Self.dtype]()
                )
                out.add_ancestry(backwardFn^, self)

        return out^
