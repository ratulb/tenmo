from tenmo import Tensor
from mnemonics import AddTensor, EXP
from backpropagation import FnArg, BACKWARD_EXPONENTIAL
from gradbox import Gradbox


@fieldwise_init
struct ExponentialBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    comptime TAG = BACKWARD_EXPONENTIAL

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        # Gradient of exp: incoming grad * exp(A) = incoming grad * output
        var exp_grad = gradbox * output
        return [(parent^, exp_grad, AddTensor)]


@fieldwise_init
struct Exponential[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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
                out.fnArg = Optional(FnArg[Self.dtype].null(BACKWARD_EXPONENTIAL))
                out.add_ancestry(tensor)

        return out^


fn main() raises:
    pass
