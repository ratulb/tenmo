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


from shapes import Shape

fn main() raises:
    test_exponentiation()
    test_negation()
    print("Did pass")

from testing import assert_true

fn test_exponentiation() raises:
    print("test_exponentiation")
    comptime dtype = DType.float32
    A = Tensor[dtype].full(Shape.of(3, 3), 2, requires_grad=True)
    a = A.to_gpu()
    expected = Tensor[dtype].full(Shape.of(3, 3), 7.389056)
    b = a.exp(True)
    assert_true(b.all_close(expected), "exponentiation assertion failed")
    b.backward()
    a.grad().print()

fn test_negation() raises:
    print("test_negation")
    comptime dtype = DType.float32
    A = Tensor[dtype].full(Shape.of(3, 3), 2, requires_grad=True)
    a = A.to_gpu()
    expected = Tensor[dtype].full(Shape.of(3, 3), -2)
    b = -a
    assert_true(b.all_close(expected), "negation assertion failed")
    b.backward()
    a.grad().print()
