from .tensor import Tensor
from .backpropagation import BackwardFnArg, ScalarArg, BACKWARD_EXPONENTIATION
from .mnemonics import AddTensor, Multiply
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .ancestry import Ancestor


@fieldwise_init
struct ExponentiationBackward[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        """
        ∂(x**n)/∂x = n * x**(n-1)
        All ops at NDBuffer level — GPU safe, no LLVM lowering issues.
        """
        var exponent = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)

        # Step 1: x ** (n-1) — NDBuffer.pow, GPU safe
        var base_pow = ancestor.buffer() ** (exponent - Scalar[Self.dtype](1))

        # Step 2: n * x**(n-1) — scalar_ops[Multiply], GPU safe
        var local_grad = base_pow.scalar_ops[Multiply](exponent)

        # Step 3: local_grad * upstream_grad — arithmetic_ops[Multiply], GPU safe
        var grad_result = local_grad * gradbox.buffer

        var parent_gradbox = Gradbox[Self.dtype](grad_result^, share=False)

        return [(ancestor, parent_gradbox^, AddTensor)]



@fieldwise_init
struct Exponentiator[dtype: DType](ImplicitlyCopyable, RegisterPassable):
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
                var backwardFnArg = BackwardFnArg[Self.dtype].scalar_arg(
                    BACKWARD_EXPONENTIATION, exponent
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^

