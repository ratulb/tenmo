from .tensor import Tensor
from .mnemonics import AddTensor
from .backpropagation import BackwardFnArg, StdArg, BACKWARD_STD
from .gradbox import Gradbox
from .ancestry import Ancestor
from tenmo.common_utils import Epsilon

# =============================================================================
# Updated StdBwdArg — goes in std.mojo
# =============================================================================


@fieldwise_init
struct StdBwdArg[dtype: DType](ArgumentType):
    var mean: NDBuffer[Self.dtype]  # saved from Welford forward — free
    var std: NDBuffer[Self.dtype]  # saved from forward — already computed
    var axis: Int
    var unbiased: Bool
    var keepdims: Bool
    var n: Int


# =============================================================================
# Uses std_backward_normalize — 2 passes over (*, D) instead of 3.
# x_ndb accessed via parent.buffer() — strides preserved, kernel handles them.
# mean_ndb, std_ndb from StdBwdArg — keepdims=True (*, 1), contiguous.
# denom = (std + eps) * divisor computed over (*, 1) buffer — negligible cost.
# Epsilon[Self.dtype].value() used as numerical guard — type's machine epsilon.
# =============================================================================


@fieldwise_init
struct StdBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = (
            output.ancestry().backward_fn_arg().get[StdBwdArg[Self.dtype]]()
        )
        var axis = bwd_arg.axis
        var unbiased = bwd_arg.unbiased
        var keepdims = bwd_arg.keepdims
        var n = bwd_arg.n
        var gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)

        # x_ndb preserves original strides — may be non-contiguous
        var x_ndb = parent.buffer()  # (*, D) — strided if view
        var mean_ndb = bwd_arg.mean  # (*, 1) contiguous, keepdims=True
        var std_ndb = bwd_arg.std  # (*, 1) contiguous, keepdims=True

        var local_grad: Tensor[Self.dtype]
        var divisor = Scalar[Self.dtype](n - 1 if unbiased and n > 1 else n)
        var last_axis = x_ndb.rank() - 1

        # ── denom = (std + eps) * divisor — over (*, 1), negligible cost ─
        # Epsilon[Self.dtype].value() — machine epsilon, numerical guard only.
        # std was computed as sqrt(var) with no eps in forward,
        # so a near-zero std is theoretically possible for constant inputs.
        if axis == -100 or axis == last_axis:
            # ── fused path — only valid for last-dim or global reduction ────
            var denom_ndb = std_ndb.scalar_ops[Add](
                Epsilon[Self.dtype].value()
            ).scalar_ops[Multiply](divisor)

            # ── Pass 1: fused (x - mean) / denom — stride-aware ─────────────
            # Produces contiguous (*, D) output regardless of x layout
            var normed_ndb = x_ndb.std_backward_normalize(mean_ndb, denom_ndb)
            local_grad = Tensor[Self.dtype](normed_ndb^)
        else:
            # ── fallback — non-last axis, use original ops ───────────────────
            var mean_tensor = Tensor[Self.dtype](mean_ndb)
            var std_tensor = Tensor[Self.dtype](std_ndb)
            var input_tensor = Tensor[Self.dtype](x_ndb)
            var diff = input_tensor.__sub__[track_grad=False](mean_tensor)
            var denom = std_tensor.__add__[track_grad=False](
                Epsilon[Self.dtype].value()
            ).__mul__[track_grad=False](divisor)
            local_grad = diff.__truediv__[track_grad=False](denom)

        # ── Gradbox shaping ──────────────────────────────────────────────
        var gradbox_ancestor: Gradbox[Self.dtype]
        if not keepdims:
            if axis != -100:
                gradbox_ancestor = gradbox.unsqueeze([axis])
            else:
                # Global std: gradbox is scalar, broadcast to input shape
                var scalar_grad = gradbox.item()
                gradbox_ancestor = Gradbox[Self.dtype].full(
                    x_ndb.shape,
                    scalar_grad,
                    share=False,
                    device=gradbox.device(),
                )
        else:
            gradbox_ancestor = gradbox^

        # ── Pass 2: multiply by upstream ────────────────────────────────
        gradbox_ancestor = local_grad * gradbox_ancestor

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
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        # Normalize negative axis — sentinel -100 flows through unchanged
        var normalized_axis = axis
        if axis != -100 and axis < 0:
            normalized_axis = self.rank() + axis

        if normalized_axis != -100 and (
            normalized_axis < 0 or normalized_axis >= self.rank()
        ):
            panic("axis is not valid for standard deviation")

        # Always compute with keepdims=True — backward needs correct broadcast shape
        var axes = IntArray.range(0, self.rank()) if normalized_axis == -100 else IntArray(normalized_axis)
        var (mean_ndb, var_ndb) = self.buffer.welford(
            axes, unbiased, keepdims=True
        )

        # std from var — keepdims=True shape preserved
        var std_ndb_keepdims = var_ndb.unary_ops[SQRT]()

        # Output: squeeze if user requested keepdims=False
        var result_ndb = std_ndb_keepdims
        _="""if not keepdims and normalized_axis != -100:
            result_ndb = std_ndb_keepdims.squeeze(IntArray(normalized_axis))
        elif not keepdims and normalized_axis == -100:
            result_ndb = std_ndb_keepdims.squeeze(IntArray())"""
        if not keepdims:
            result_ndb = std_ndb_keepdims.squeeze(axes)

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
                        mean_ndb^,  # keepdims=True — correct for broadcast
                        std_ndb_keepdims
                        ^,  # keepdims=True — correct for broadcast
                        normalized_axis,
                        unbiased,
                        keepdims,  # original user request — for gradbox handling
                        n,
                    ),
                )
                result.add_ancestry(backwardFnArg^, self)

        return result^
