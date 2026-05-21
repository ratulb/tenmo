"""Fused BCE / BCEWithLogits Loss — forward structs + backward handlers.

Replaces the chain of 10+ separate tensor ops with a single fused
NDBuffer-level kernel for both forward and backward passes.

Reduction modes:
  mean (default): scalar loss, gradient = (sigmoid-target) * upstream / N
  sum:           scalar loss, gradient = (sigmoid-target) * upstream
  none:          per-element loss and gradient (upstream is per-element)
"""

from tenmo.tensor import Tensor
from tenmo.backpropagation import (
    BackwardFnArg,
    BCEWithLogitsBwdArg,
    BCELossBwdArg,
    BACKWARD_BCE_WITH_LOGITS,
    BACKWARD_BCE,
)
from tenmo.gradbox import Gradbox
from tenmo.ancestry import Ancestor
from tenmo.mnemonics import AddTensor
from tenmo.common_utils import Epsilon
from tenmo.shared import Reduction

@fieldwise_init
struct BCEWithLogitsLoss[dtype: DType](ImplicitlyCopyable & RegisterPassable):
    var training: Bool
    var epsilon: Scalar[Self.dtype]

    def __init__(
        out self, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ):
        self.training = True
        self.epsilon = epsilon

    def __call__(
        self, logits: Tensor[Self.dtype], target: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        if self.training:
            return Self.forward[track_grad=True](logits, target, self.epsilon)
        else:
            return Self.forward[track_grad=False](logits, target, self.epsilon)

    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        logits: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
        reduction: Reduction = Reduction("mean"),
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var numels = logits.numels()
        var out: Tensor[Self.dtype]
        var sigmoid_ndb: NDBuffer[Self.dtype]

        if reduction.is_none():
            var (loss_ndb, bw) = logits.buffer.bce_with_logits_forward(
                target.buffer, epsilon
            )
            out = Tensor[Self.dtype](loss_ndb^, requires_grad=False)
            sigmoid_ndb = bw^
        else:
            var is_mean = reduction.is_mean()
            var (scalar_ndb, bw) = logits.buffer.bce_with_logits_forward_reduce(
                target.buffer, epsilon, is_mean
            )
            out = Tensor[Self.dtype](scalar_ndb^, requires_grad=False)
            sigmoid_ndb = bw^

        comptime if track_grad:
            var grad_required = logits.requires_grad
            if grad_required:
                out.requires_grad_(True)
                var target_copy = target.buffer.copy()
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_BCE_WITH_LOGITS,
                    BCEWithLogitsBwdArg[Self.dtype](
                        sigmoid_ndb^, target_copy^, reduction, numels
                    ),
                )
                out.add_ancestry(backwardFnArg^, logits)

        return out^

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False


@fieldwise_init
struct BCELoss[dtype: DType](ImplicitlyCopyable & RegisterPassable):
    var training: Bool
    var epsilon: Scalar[Self.dtype]

    def __init__(
        out self, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ):
        self.training = True
        self.epsilon = epsilon

    def __call__(
        self, pred: Tensor[Self.dtype], target: Tensor[Self.dtype]
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        if self.training:
            return Self.forward[track_grad=True](pred, target, self.epsilon)
        else:
            return Self.forward[track_grad=False](pred, target, self.epsilon)

    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        pred: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
        reduction: Reduction = Reduction("mean"),
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var numels = pred.numels()
        var out: Tensor[Self.dtype]
        var safe_ndb: NDBuffer[Self.dtype]

        if reduction.is_none():
            var (loss_ndb, bw) = pred.buffer.bce_forward(
                target.buffer, epsilon
            )
            out = Tensor[Self.dtype](loss_ndb^, requires_grad=False)
            safe_ndb = bw^
        else:
            var is_mean = reduction.is_mean()
            var (scalar_ndb, bw) = pred.buffer.bce_forward_reduce(
                target.buffer, epsilon, is_mean
            )
            out = Tensor[Self.dtype](scalar_ndb^, requires_grad=False)
            safe_ndb = bw^

        comptime if track_grad:
            var grad_required = pred.requires_grad
            if grad_required:
                out.requires_grad_(True)
                var target_copy = target.buffer.copy()
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_BCE,
                    BCELossBwdArg[Self.dtype](
                        safe_ndb^, target_copy^, reduction, numels
                    ),
                )
                out.add_ancestry(backwardFnArg^, pred)

        return out^

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False


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

        if bwd_arg.reduction.is_none():
            var grad_ndb = sigmoid.bce_with_logits_backward(
                target, gradbox.buffer
            )
            var grad_parent = Gradbox[Self.dtype](grad_ndb^, share=False)
            parent.update_grad(grad_parent^, AddTensor, None)
        else:
            var multiplier = gradbox.item()
            if bwd_arg.reduction.is_mean():
                multiplier /= Scalar[Self.dtype](bwd_arg.numels)
            var grad_ndb = sigmoid.bce_with_logits_backward_scaled(
                target, multiplier
            )
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

        if bwd_arg.reduction.is_none():
            var grad_ndb = safe.bce_backward(target, gradbox.buffer)
            var grad_parent = Gradbox[Self.dtype](grad_ndb^, share=False)
            parent.update_grad(grad_parent^, AddTensor, None)
        else:
            var multiplier = gradbox.item()
            if bwd_arg.reduction.is_mean():
                multiplier /= Scalar[Self.dtype](bwd_arg.numels)
            var grad_ndb = safe.bce_backward_scaled(target, multiplier)
            var grad_parent = Gradbox[Self.dtype](grad_ndb^, share=False)
            parent.update_grad(grad_parent^, AddTensor, None)
        parent_ids.append(parent._id)
