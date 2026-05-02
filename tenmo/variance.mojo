from .tensor import Tensor
from .mnemonics import AddTensor
from .backpropagation import BackwardFnArg, ArgumentType, BACKWARD_VARIANCE
from .gradbox import Gradbox
from .common_utils import panic
from .ancestry import Ancestor


@fieldwise_init
struct VarianceBwdArg[dtype: DType](ArgumentType):
    var mean: NDBuffer[Self.dtype]  # saved from Welford forward — free
    var axis: Int
    var unbiased: Bool
    var keepdims: Bool
    var n: Int                      # saved — avoids shape lookup in backward


# =============================================================================
# Updated VarianceBackward — goes in variance.mojo
# =============================================================================

@fieldwise_init
struct VarianceBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = output.ancestry().backward_fn_arg().get[
            VarianceBwdArg[Self.dtype]
        ]()
        var axis = bwd_arg.axis
        var unbiased = bwd_arg.unbiased
        var keepdims = bwd_arg.keepdims
        var n = bwd_arg.n
        var gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        var input_tensor = Tensor[Self.dtype](
            parent.buffer(), requires_grad=parent.requires_grad
        )

        var divisor = Scalar[Self.dtype](n - 1 if unbiased and n > 1 else n)

        # mean already saved — no recomputation
        var mean_tensor = Tensor[Self.dtype](bwd_arg.mean)
        var diff = input_tensor.__sub__[track_grad=False](mean_tensor)
        var local_grad = diff.__mul__[track_grad=False](
            Scalar[Self.dtype](2) / divisor
        )

        var gradbox_ancestor: Gradbox[Self.dtype]
        if not keepdims:
            if axis != -100:
                gradbox_ancestor = gradbox.unsqueeze([axis])
            else:
                gradbox_ancestor = gradbox^
        else:
            gradbox_ancestor = gradbox^

        gradbox_ancestor = local_grad * gradbox_ancestor
        return [(parent^, gradbox_ancestor^, AddTensor)]


# =============================================================================
# Updated Variance.forward — goes in variance.mojo
# replaces the current forward body
# =============================================================================

@fieldwise_init
struct Variance[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        axis: Int = -100,
        keepdims: Bool = False,
        unbiased: Bool = True,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        if axis != -100 and (axis < 0 or axis >= self.rank()):
            panic("Invalid axis specified for variance")

        # Single Welford pass — returns (mean_ndb, var_ndb)
        # mean is free — Welford computes it anyway
        #var (mean_ndb, var_ndb) = self.buffer.welford(axis, unbiased, keepdims)
        #var result = Tensor[Self.dtype](var_ndb^, requires_grad=False)
        # Always save mean with keepdims=True for correct backward broadcasting
        var (mean_ndb, var_ndb) = self.buffer.welford(axis, unbiased, keepdims=True)
        # For the output, squeeze if user requested keepdims=False
        var result_ndb = var_ndb
        if not keepdims and axis != -100:
            result_ndb = var_ndb.squeeze(IntArray(axis))
        elif not keepdims and axis == -100:
            result_ndb = var_ndb.squeeze(IntArray())
        var result = Tensor[Self.dtype](result_ndb^, requires_grad=False)
        # mean_ndb stays keepdims=True — correct shape for backward broadcast

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                result.requires_grad_(True)
                var n: Int
                if axis != -100:
                    n = self.shape()[axis]
                else:
                    n = self.numels()
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_VARIANCE,
                    VarianceBwdArg[Self.dtype](
                        mean_ndb^,  # saved — free from Welford
                        axis,
                        unbiased,
                        keepdims,
                        n,          # saved — free
                    ),
                )
                result.add_ancestry(backwardFnArg^, self)

        return result^
