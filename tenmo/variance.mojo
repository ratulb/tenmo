from .tensor import Tensor
from .mnemonics import AddTensor
from .backpropagation import BackwardFnArg, ArgumentType, BACKWARD_VARIANCE
from .gradbox import Gradbox
from .common_utils import panic
from .ancestry import Ancestor
from tenmo.intarray import IntArray


@fieldwise_init
struct VarianceBwdArg[dtype: DType](ArgumentType):
    var mean: NDBuffer[Self.dtype]  # saved from Welford forward — free
    var axis: Int
    var unbiased: Bool
    var keepdims: Bool
    var n: Int  # saved — avoids shape lookup in backward


# =============================================================================
# Updated VarianceBackward — goes in variance.mojo
# =============================================================================


@fieldwise_init
struct VarianceBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = (
            output.ancestry()
            .backward_fn_arg()
            .get[VarianceBwdArg[Self.dtype]]()
        )
        var axis = bwd_arg.axis
        var unbiased = bwd_arg.unbiased
        var keepdims = bwd_arg.keepdims
        var n = bwd_arg.n
        var gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        # x_ndb preserves original strides — may be non-contiguous
        var x_ndb = parent.buffer()  # (*, D) — strided if view
        var mean_ndb = bwd_arg.mean  # (*, 1) — contiguous, keepdims=True

        var divisor = Scalar[Self.dtype](n - 1 if unbiased and n > 1 else n)
        var scale = Scalar[Self.dtype](2) / divisor  # uniform scalar
        var last_axis = x_ndb.rank() - 1

        var local_grad: Tensor[Self.dtype]

        if axis == -100 or axis == last_axis:
            # ── Pass 1: fused (x - mean) * scale — stride-aware ─────────────
            # Produces contiguous (*, D) output regardless of x layout
            var normed_ndb = x_ndb.variance_backward_normalize(mean_ndb, scale)
            local_grad = Tensor[Self.dtype](normed_ndb^)

        else:
            var input_tensor = Tensor[Self.dtype](
                x_ndb, requires_grad=parent.requires_grad
            )
            # mean already saved — no recomputation
            var mean_tensor = Tensor[Self.dtype](mean_ndb)
            var diff = input_tensor.__sub__[track_grad=False](mean_tensor)
            local_grad = diff.__mul__[track_grad=False](scale)

        var gradbox_ancestor: Gradbox[Self.dtype]
        if not keepdims:
            if axis != -100:
                gradbox_ancestor = gradbox.unsqueeze([axis])
            else:
                # Global variance: gradbox is scalar, broadcast to input shape
                var scalar_grad = gradbox.item()
                gradbox_ancestor = Gradbox[Self.dtype].full(
                    x_ndb.shape,
                    scalar_grad,
                    share=False,
                    device=gradbox.device(),
                )
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
        # Normalize negative axis — but never touch the -100 sentinel
        var normalized_axis = axis
        if axis != -100 and axis < 0:
            normalized_axis = self.rank() + axis  # -1 on rank 3 → 2

        # Validate — only for non-sentinel, non-negative result
        if normalized_axis != -100 and (
            normalized_axis < 0 or normalized_axis >= self.rank()
        ):
            panic("Invalid axis specified for variance")

        # Single Welford pass — returns (mean_ndb, var_ndb)
        # mean is free — Welford computes it anyway
        # var (mean_ndb, var_ndb) = self.buffer.welford(axis, unbiased, keepdims)
        # var result = Tensor[Self.dtype](var_ndb^, requires_grad=False)
        # Always save mean with keepdims=True for correct backward broadcasting
        var axes = IntArray.range(
            0, self.rank()
        ) if normalized_axis == -100 else IntArray(normalized_axis)
        var (mean_ndb, var_ndb) = self.buffer.welford(
            axes, unbiased, keepdims=True
        )
        # For the output, squeeze if user requested keepdims=False
        var result_ndb = var_ndb
        if not keepdims:
            result_ndb = var_ndb.squeeze(axes)
        var result = Tensor[Self.dtype](result_ndb^, requires_grad=False)
        # mean_ndb stays keepdims=True — correct shape for backward broadcast

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
                    BACKWARD_VARIANCE,
                    VarianceBwdArg[Self.dtype](
                        mean_ndb^,  # saved — free from Welford
                        normalized_axis,
                        unbiased,
                        keepdims,
                        n,
                    ),
                )
                result.add_ancestry(backwardFnArg^, self)

        return result^
