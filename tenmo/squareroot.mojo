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
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var epsilon = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        var parent_buffer = parent.buffer()
        ref shape = parent.shape()
        var ndb = parent.buffer().arithmetic_ops[SQRT_BACKWARD](gradbox.buffer, epsilon)
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^, share=False)

        return [(parent^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
struct Sqrt[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
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
    fn forward(
        self: Gradbox[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> Gradbox[Self.dtype]:
        var out: Gradbox[Self.dtype]
        ref shape = self.shape()

        var buffer = self.buffer.data_buffer().unary_ops[SQRT]()
        out = Gradbox[Self.dtype](
            NDBuffer[Self.dtype](buffer^, shape), share=False
        )

        return out^
