from tenmo import Tensor
from backpropagation import BackwardFn, Delegate, BACKWARD_EXPONENTIATION
from mnemonics import AddTensor
from gradbox import Gradbox
from ndbuffer import NDBuffer


@fieldwise_init
@register_passable
struct ExponientionBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_EXPONENTIATION
    var exponent: Scalar[Self.dtype]

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)

        # ∂(x**n)/∂x = n * x**(n-1)
        # var base_pow = self ** (scalar - 1.0)
        base_pow = ancestor.__pow__[track_grad=False](self.exponent - 1.0)
        var local_grad = base_pow.__mul__[track_grad=False](self.exponent)
        gradbox_prod = local_grad * gradbox
        return [
            (
                ancestor^,
                gradbox_prod^,
                AddTensor,
            )
        ]


@fieldwise_init
@register_passable
struct Exponentiator[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], exponent: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        nd_buffer = NDBuffer[Self.dtype](
            (self.buffer.contiguous_buffer() ** exponent), self.shape()
        )
        var out = Tensor[Self.dtype](nd_buffer^, requires_grad=False)

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = ExponientionBackward[Self.dtype](
                    exponent
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^
