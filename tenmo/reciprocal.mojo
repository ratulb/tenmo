from .tensor import Tensor
from .backpropagation import (
    BackwardFnType,
    BackwardFn,
)
from .shared.mnemonics import ReverseDivide
from .division import RightTrueDivBackwardScalar
from .ancestry import Ancestor


@fieldwise_init
struct Reciprocal[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        # out = 1/x
        var out_ndb = self.buffer.scalar_ops[ReverseDivide](
            Scalar[Self.dtype](1), sync=sync
        )
        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                # scalar_arg carries the numerator (1); backward reads
                # the output buffer directly.
                var backwardFn = BackwardFn.scalar_arg[Self.dtype](
                    Scalar[Self.dtype](1),
                    RightTrueDivBackwardScalar[Self.dtype](),
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, self)

        return out^
