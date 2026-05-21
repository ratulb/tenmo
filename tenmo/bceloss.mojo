"""Fused BCE / BCEWithLogits Loss — forward structs + backward handlers.

Replaces the chain of 10+ separate tensor ops with a single fused
NDBuffer-level kernel for both forward and backward passes.
"""

from .tensor import Tensor
from .backpropagation import (
    BackwardFnArg,
    BCEWithLogitsBwdArg,
    BCELossBwdArg,
    BACKWARD_BCE_WITH_LOGITS,
    BACKWARD_BCE,
)
from .gradbox import Gradbox
from .ancestry import Ancestor
from .mnemonics import AddTensor


@fieldwise_init
struct BCEWithLogitsLossFused[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        logits: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9),
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var (loss_ndb, sigmoid_ndb) = logits.buffer.bce_with_logits_forward(
            target.buffer, epsilon
        )
        var out = Tensor[Self.dtype](loss_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = logits.requires_grad
            if grad_required:
                out.requires_grad_(True)
                var target_copy = target.buffer.copy()
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_BCE_WITH_LOGITS,
                    BCEWithLogitsBwdArg[Self.dtype](sigmoid_ndb^, target_copy^),
                )
                out.add_ancestry(backwardFnArg^, logits)

        return out^


@fieldwise_init
struct BCELossFused[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        pred: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-9),
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var (loss_ndb, safe_ndb) = pred.buffer.bce_forward(
            target.buffer, epsilon
        )
        var out = Tensor[Self.dtype](loss_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = pred.requires_grad
            if grad_required:
                out.requires_grad_(True)
                var target_copy = target.buffer.copy()
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_BCE,
                    BCELossBwdArg[Self.dtype](safe_ndb^, target_copy^),
                )
                out.add_ancestry(backwardFnArg^, pred)

        return out.mean[track_grad]()


# ── Backward Handlers ─────────────────────────────────────────────────────────


@fieldwise_init
struct BCEWithLogitsBackward[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ) where Self.dtype.is_floating_point():
        var bwd_arg = (
            output.ancestry()
            .backward_fn_arg()
            .get[BCEWithLogitsBwdArg[Self.dtype]]()
        )
        var sigmoid = bwd_arg.sigmoid
        var target = bwd_arg.target
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)

        var grad_ndb = sigmoid.bce_with_logits_backward(target, gradbox.buffer)
        var grad_parent = Gradbox[Self.dtype](grad_ndb^, share=False)
        parent.update_grad(grad_parent^, AddTensor, None)
        parent_ids.append(parent._id)


@fieldwise_init
struct BCELossBackward[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ) where Self.dtype.is_floating_point():
        var bwd_arg = (
            output.ancestry().backward_fn_arg().get[BCELossBwdArg[Self.dtype]]()
        )
        var safe = bwd_arg.clipped_pred
        var target = bwd_arg.target
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)

        var grad_ndb = safe.bce_backward(target, gradbox.buffer)
        var grad_parent = Gradbox[Self.dtype](grad_ndb^, share=False)
        parent.update_grad(grad_parent^, AddTensor, None)
        parent_ids.append(parent._id)
