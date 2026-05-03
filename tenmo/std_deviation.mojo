from .tensor import Tensor
from .mnemonics import AddTensor
from .backpropagation import BackwardFnArg, StdArg, BACKWARD_STD
from .gradbox import Gradbox
from .common_utils import panic, Epsilon
from .ancestry import Ancestor


# =============================================================================
# Updated StdBwdArg — goes in std.mojo
# =============================================================================

@fieldwise_init
struct StdBwdArg[dtype: DType](ArgumentType):
    var mean: NDBuffer[Self.dtype]  # saved from Welford forward — free
    var std: NDBuffer[Self.dtype]   # saved from forward — already computed
    var axis: Int
    var unbiased: Bool
    var keepdims: Bool
    var n: Int
    var epsilon: Scalar[Self.dtype]


# =============================================================================
# Updated StdBackward — goes in std.mojo
# =============================================================================

@fieldwise_init
struct StdBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = output.ancestry().backward_fn_arg().get[
            StdBwdArg[Self.dtype]
        ]()
        var axis = bwd_arg.axis
        var unbiased = bwd_arg.unbiased
        var keepdims = bwd_arg.keepdims
        var n = bwd_arg.n
        var epsilon = bwd_arg.epsilon
        var gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        var input_tensor = Tensor[Self.dtype](
            parent.buffer(), requires_grad=parent.requires_grad
        )

        var divisor = Scalar[Self.dtype](n - 1 if unbiased and n > 1 else n)

        # Both mean and std already saved — zero recomputation
        var mean_tensor = Tensor[Self.dtype](bwd_arg.mean)
        var std_tensor = Tensor[Self.dtype](bwd_arg.std)

        var diff = input_tensor.__sub__[track_grad=False](mean_tensor)
        var denom = (
            std_tensor.__add__[track_grad=False](epsilon)
            .__mul__[track_grad=False](divisor)
        )
        var local_grad = diff.__truediv__[track_grad=False](denom)

        var gradbox_ancestor: Gradbox[Self.dtype]
        if not keepdims:
            if axis != -100:
                gradbox_ancestor = gradbox.unsqueeze([axis])
            else:
                var scalar_grad = gradbox.item()
                gradbox_ancestor = Gradbox[Self.dtype].full(
                    input_tensor.shape(), scalar_grad, share=False, device=gradbox.device()
                )
        else:
            gradbox_ancestor = gradbox^

        gradbox_ancestor = local_grad.__mul__(gradbox_ancestor^)
        return [(parent^, gradbox_ancestor^, AddTensor)]


# =============================================================================
# Updated StdDev.forward — goes in std.mojo
# replaces current forward body
# =============================================================================

@fieldwise_init
struct StdDev[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        axis: Int = -100,
        keepdims: Bool = False,
        unbiased: Bool = True,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        # Normalize negative axis — sentinel -100 flows through unchanged
        var normalized_axis = axis
        if axis != -100 and axis < 0:
            normalized_axis = self.rank() + axis

        if normalized_axis != -100 and (normalized_axis < 0 or normalized_axis >= self.rank()):
            panic("axis is not valid for standard deviation")

        # Always compute with keepdims=True — backward needs correct broadcast shape
        var (mean_ndb, var_ndb) = self.buffer.welford(normalized_axis, unbiased, keepdims=True)

        # std from var — keepdims=True shape preserved
        var std_ndb_keepdims = var_ndb.unary_ops[SQRT]()

        # Output: squeeze if user requested keepdims=False
        var result_ndb = std_ndb_keepdims
        if not keepdims and normalized_axis != -100:
            result_ndb = std_ndb_keepdims.squeeze(IntArray(normalized_axis))
        elif not keepdims and normalized_axis == -100:
            result_ndb = std_ndb_keepdims.squeeze(IntArray())

        var result = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        # Save keepdims=True versions into BwdArg — correct shape for backward
        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                result.requires_grad_(True)
                var n: Int
                if normalized_axis != -100:
                    n = self.shape()[normalized_axis]
                else:
                    n = self.numels()
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_STD,
                    StdBwdArg[Self.dtype](
                        mean_ndb^,          # keepdims=True — correct for broadcast
                        std_ndb_keepdims^,  # keepdims=True — correct for broadcast
                        normalized_axis,
                        unbiased,
                        keepdims,           # original user request — for gradbox handling
                        n,
                        epsilon,
                    ),
                )
                result.add_ancestry(backwardFnArg^, self)

        return result^
