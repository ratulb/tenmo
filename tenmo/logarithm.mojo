from tensor import Tensor
from mnemonics import AddTensor, LOG, LOG_BACKWARD
from backpropagation import BackwardFnArg, ScalarArg, BACKWARD_LOG
from gradbox import Gradbox
from common_utils import Epsilon
from ancestry import Ancestor


@fieldwise_init
struct LogBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
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
        var result_ndb = parent.buffer().arithmetic_ops[LOG_BACKWARD](
            gradbox.buffer, epsilon
        )
        var parent_gradbox = Gradbox[Self.dtype](result_ndb^, share=False)

        return [(parent^, parent_gradbox^, AddTensor)]


@fieldwise_init
struct Logarithm[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var result_ndb = self.buffer.log[epsilon]()
        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].scalar_arg(
                    BACKWARD_LOG, epsilon
                )

                out.add_ancestry(backwardFnArg^, self)

        return out^


fn main() raises:
    pass
