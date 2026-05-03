from .tensor import Tensor
from .backpropagation import (
    BackwardFnArg,
    BACKWARD_RIGHT_DIV_SCALAR,
)
from .mnemonics import AddTensor, SubtractTensor, ReverseDivide
from .gradbox import Gradbox
from .ancestry import Ancestor


@fieldwise_init
struct Reciprocal[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        # out = 1/x
        var out_ndb = self.buffer.scalar_ops[ReverseDivide](Scalar[Self.dtype](1))
        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                # null_arg — backward reads output buffer directly
                var backwardFnArg = BackwardFnArg[Self.dtype].scalar_arg(
                    BACKWARD_RIGHT_DIV_SCALAR, Scalar[Self.dtype](1)
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^
