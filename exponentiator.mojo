from tenmo import Tensor
from backpropagation import FnArg, ScalarArg, BACKWARD_EXPONENTIATION
from mnemonics import AddTensor, Multiply
from gradbox import Gradbox
from ndbuffer import NDBuffer

# ── ExponentiationBackward ────────────────────────────────────────────────────


@fieldwise_init
struct ExponentiationBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        """
        ∂(x**n)/∂x = n * x**(n-1)
        All ops at NDBuffer level — GPU safe, no LLVM lowering issues.
        """
        var exponent = output.fn_arg().arg[ScalarArg[Self.dtype]].scalar
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)

        # Step 1: x ** (n-1) — NDBuffer.pow, GPU safe
        var base_pow = ancestor.buffer ** (
            exponent - Scalar[Self.dtype](1)
        )

        # Step 2: n * x**(n-1) — scalar_ops[Multiply], GPU safe
        var local_grad = base_pow.scalar_ops[Multiply](exponent)

        # Step 3: local_grad * upstream_grad — arithmetic_ops[Multiply], GPU safe
        var grad_result = local_grad * gradbox.buffer

        var parent_gradbox = Gradbox[Self.dtype](grad_result^, share=False)

        return [(ancestor^, parent_gradbox^, AddTensor)]


# ── Exponentiator forward ─────────────────────────────────────────────────────


@fieldwise_init
struct Exponentiator[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        exponent: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """
        Element-wise x ** exponent.
        Delegates to NDBuffer.pow — handles GPU and CPU paths.
        """
        var result_ndb = self.buffer**exponent
        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                out.fnArg = Optional(FnArg[Self.dtype].scalar(exponent, BACKWARD_EXPONENTIATION))
                out.add_ancestry(self)

        return out^


from shapes import Shape


fn main() raises:
    test_exponentiation()
    test_negation()
    print("pass")


from std.testing import assert_true


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
