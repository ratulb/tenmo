from tenmo import Tensor
from mnemonics import AddTensor, EXP
from backpropagation import BackwardFnArg, BACKWARD_EXPONENTIAL
from gradbox import Gradbox
from ancestors_newest import AncestorRef

@fieldwise_init
struct ExponentialBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: AncestorRef[Self.dtype],
    ) -> List[
        Tuple[AncestorRef[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        # Gradient of exp: incoming grad * exp(A) = incoming grad * output
        var exp_grad = Gradbox[Self.dtype](gradbox.buffer * output.buffer(), share=False)
        return [(parent, exp_grad, AddTensor)]


@fieldwise_init
struct Exponential[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var ndb = tensor.buffer.exp()
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(BACKWARD_EXPONENTIAL)
                out.add_ancestry(backwardFnArg^, tensor)

        return out^


fn main() raises:
    pass
