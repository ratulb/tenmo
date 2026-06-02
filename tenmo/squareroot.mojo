from .tensor import Tensor
from .mnemonics import AddTensor, SQRT, SQRT_BACKWARD
from .backpropagation import BackwardFnArg, ScalarArg, BACKWARD_SQRT
from .gradbox import Gradbox
from std.math import sqrt
from .ndbuffer import NDBuffer
from .common_utils import Epsilon
from .ancestry import Ancestor


@fieldwise_init
struct SqrtBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var epsilon = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        var ndb = parent.buffer().arithmetic_ops[SQRT_BACKWARD](
            gradbox.buffer, epsilon
        )
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^)

        parent.update_grad(gradbox_ancestor^, AddTensor, None)

        parent_ids.append(parent._id)

        if not retain_graph:
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
    ) -> Tensor[Self.dtype]:
        var ndb = self.buffer.unary_ops[SQRT]()
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].scalar_arg(
                    BACKWARD_SQRT, epsilon
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^

    @staticmethod
    def forward(
        self: Gradbox[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> Gradbox[Self.dtype]:
        var out: Gradbox[Self.dtype]
        ref shape = self.shape()

        var buffer = self.buffer.data_buffer().unary_ops[SQRT]()
        out = Gradbox[Self.dtype](
            NDBuffer[Self.dtype](buffer^, shape), 
        )

        return out^
