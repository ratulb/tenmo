from tenmo import Tensor
from mnemonics import AddTensor, LOG, LOG_BACKWARD
from backpropagation import BackwardFnArg, BACKWARD_LOG
from gradbox import Gradbox
from common_utils import Epsilon

@fieldwise_init
struct LogBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ]:
        var epsilon = output.bwd_fn_arg().arg[Scalar[Self.dtype]]
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        var result_ndb = parent.buffer.arithmetic_ops[LOG_BACKWARD](gradbox.buffer, epsilon)
        var parent_gradbox = Gradbox[Self.dtype](result_ndb^, share=False)

        return [(parent^, parent_gradbox^, AddTensor)]

@fieldwise_init
struct Logarithm[dtype: DType](RegisterPassable, ImplicitlyCopyable):

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
                out.bwdFnArg = Optional(BackwardFnArg[Self.dtype].scalar(BACKWARD_LOG, epsilon))
                out.add_ancestry(self)

        return out^

fn main() raises:
    pass
