from tenmo import Tensor
from mnemonics import AddTensor, EXP
from backpropagation import Delegate, BackwardFn, BACKWARD_EXPONENTIAL
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct ExponentialBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_EXPONENTIAL

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        # Gradient of exp: incoming grad * exp(A) = incoming grad * output
        var exp_grad = gradbox * output
        return [(parent^, exp_grad, AddTensor)]


@fieldwise_init
@register_passable
struct Exponential[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var ndb = tensor.buffer.exp()
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backward_fn = ExponentialBackward[
                    Self.dtype
                ]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(tensor)

        return out^


fn main() raises:
    pass
