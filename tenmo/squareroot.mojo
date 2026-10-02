from .tensor import Tensor
from .shared.mnemonics import AddTensor, SQRT, SQRT_BACKWARD
from .backpropagation import BackwardFn, ScalarArg, BackwardFnType

from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .shared.constants import Epsilon
from .ancestry import Ancestor


@fieldwise_init
struct SqrtBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var epsilon = (
            output.ancestry().backward_fn().get[ScalarArg[Self.dtype]]().value
        )
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)
        var ndb = parent.buffer().arithmetic_ops[SQRT_BACKWARD](
            gradbox.buffer(), epsilon
        )
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^)

        parent.update_grad(gradbox_ancestor^, AddTensor, None)

        parent_ids.append(parent._id)

        gradbox.zero_grad()


@fieldwise_init
struct Sqrt[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var ndb = self.buffer.unary_ops[SQRT]()
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.scalar_arg[Self.dtype](
                    epsilon, SqrtBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, self)

        return out^

    @staticmethod
    def forward(self: Gradbox[Self.dtype]) -> Gradbox[Self.dtype]:
        # No epsilon: pure forward with no backward wiring, so there is
        # nothing to stabilize.
        var out: Gradbox[Self.dtype]
        var shape = self.shape()

        var buffer = self.buffer().data_buffer().unary_ops[SQRT]()
        out = Gradbox[Self.dtype](
            NDBuffer[Self.dtype](buffer^, shape),
        )

        return out^
