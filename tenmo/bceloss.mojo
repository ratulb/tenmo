"""Fused BCE / BCEWithLogits Loss — forward structs, backward handlers, and kernels.

Contains:
- BceElement[dtype] — scalar element functions for BCE math
- BceBuffer[dtype]  — Buffer-level SIMD kernels (contiguous, CPU only)
- BceNdBuffer[dtype] — NDBuffer-level device dispatch + strided CPU fallback
- BCEWithLogitsLoss, BCELoss  — forward structs with autograd
- BCEWithLogitsBackward, BCELossBackward — backward handlers

Reduction modes:
  mean (default): scalar loss, gradient = (sigmoid-target) * upstream / N
  sum:           scalar loss, gradient = (sigmoid-target) * upstream
  none:          per-element loss and gradient (upstream is per-element)
"""

from tenmo.tensor import Tensor
from tenmo.ndbuffer import NDBuffer
from tenmo.buffers import Buffer
from tenmo.shapes import Shape
from tenmo.kernels.bce_kernel import BceKernel
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
from tenmo.common_utils import panic
from std.sys import simd_width_of, has_accelerator
from std.math import exp, log


# =============================================================================
# Scalar element functions — pure Scalar[dtype] transformations
# =============================================================================


@fieldwise_init
struct BceElement[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def sigmoid_fn(
        x: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype] where Self.dtype.is_floating_point():
        return Scalar[Self.dtype](1.0) / (Scalar[Self.dtype](1.0) + exp(-x))

    @staticmethod
    def clip_fn(
        x: Scalar[Self.dtype], epsilon: Scalar[Self.dtype]
    ) -> Scalar[Self.dtype] where Self.dtype.is_floating_point():
        return x.clamp(epsilon, Scalar[Self.dtype](1.0) - epsilon)

    @staticmethod
    def with_logits_element(
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
        epsilon: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype] where Self.dtype.is_floating_point():
        var s = Self.sigmoid_fn(x)
        var safe = Self.clip_fn(s, epsilon)
        return -(
            y * log(safe)
            + (Scalar[Self.dtype](1.0) - y)
            * log(Scalar[Self.dtype](1.0) - safe)
        )

    @staticmethod
    def element(
        p: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
        epsilon: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype] where Self.dtype.is_floating_point():
        var safe = Self.clip_fn(p, epsilon)
        return -(
            y * log(safe)
            + (Scalar[Self.dtype](1.0) - y)
            * log(Scalar[Self.dtype](1.0) - safe)
        )

    @staticmethod
    def with_logits_backward_element(
        sigmoid: Scalar[Self.dtype],
        target: Scalar[Self.dtype],
        grad: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype] where Self.dtype.is_floating_point():
        return (sigmoid - target) * grad

    @staticmethod
    def backward_element(
        safe: Scalar[Self.dtype],
        target: Scalar[Self.dtype],
        grad: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype] where Self.dtype.is_floating_point():
        var one = Scalar[Self.dtype](1.0)
        var one_minus_target = one - target
        var one_minus_safe = one - safe
        return -(target / safe - one_minus_target / one_minus_safe) * grad


# =============================================================================
# Buffer-level SIMD kernels (contiguous CPU, same-size buffers)
# =============================================================================


@fieldwise_init
struct BceBuffer[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def forward_with_logits(
        logits: Buffer[Self.dtype],
        target: Buffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
    ) -> Tuple[
        Buffer[Self.dtype], Buffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused BCEWithLogits forward. Computes per-element loss + sigmoid in one pass.

        Args:
            logits: Raw logits buffer.
            target: Target buffer (same size as logits).
            epsilon: Clipping epsilon for numerical stability.

        Returns:
            (loss_per_element, sigmoid) buffers.
        """
        var extent = logits.size
        var loss_out = Buffer[Self.dtype](extent)
        var sig_out = Buffer[Self.dtype](extent)

        comptime simd_width = simd_width_of[Self.dtype]()

        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width
        var one = SIMD[Self.dtype, simd_width](1.0)
        var eps = SIMD[Self.dtype, simd_width](epsilon)

        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var x = logits.load[simdwidth=simd_width](idx)
            var y = target.load[simdwidth=simd_width](idx)

            var s = one / (one + exp(-x))
            var safe = s.clamp(eps, one - eps)
            var loss = -(y * log(safe) + (one - y) * log(one - safe))

            loss_out.store[simdwidth=simd_width](idx, loss)
            sig_out.store[simdwidth=simd_width](idx, s)

        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            var one_s = Scalar[Self.dtype](1.0)
            for i in range(remainder):
                var idx = start_idx + i
                var x = logits[idx]
                var y = target[idx]
                var s = one_s / (one_s + exp(-x))
                var safe = s.clamp(epsilon, one_s - epsilon)
                loss_out[idx] = -(
                    y * log(safe) + (one_s - y) * log(one_s - safe)
                )
                sig_out[idx] = s

        return (loss_out^, sig_out^)

    @staticmethod
    def forward_with_logits_reduce(
        logits: Buffer[Self.dtype],
        target: Buffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
        is_mean: Bool,
    ) -> Tuple[
        Buffer[Self.dtype], Buffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused BCEWithLogits forward with mean/sum reduction.

        Computes per-element loss + sigmoid in one pass, accumulates loss sum,
        and returns a 1-element scalar loss + full-size sigmoid buffer.

        Args:
            logits: Raw logits buffer.
            target: Target buffer (same size as logits).
            epsilon: Clipping epsilon.
            is_mean: If True, divide sum by N (mean); else return sum.

        Returns:
            (scalar_loss[1], sigmoid[N]) buffers.
        """
        var extent = logits.size
        var sig_out = Buffer[Self.dtype](extent)

        comptime simd_width = simd_width_of[Self.dtype]()

        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width
        var one = SIMD[Self.dtype, simd_width](1.0)
        var eps = SIMD[Self.dtype, simd_width](epsilon)

        var total = Scalar[Self.dtype](0)

        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var x = logits.load[simdwidth=simd_width](idx)
            var y = target.load[simdwidth=simd_width](idx)

            var s = one / (one + exp(-x))
            var safe = s.clamp(eps, one - eps)
            var loss = -(y * log(safe) + (one - y) * log(one - safe))

            sig_out.store[simdwidth=simd_width](idx, s)
            total += loss.reduce_add()

        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            var one_s = Scalar[Self.dtype](1.0)
            for i in range(remainder):
                var idx = start_idx + i
                var x = logits[idx]
                var y = target[idx]
                var s = one_s / (one_s + exp(-x))
                var safe = s.clamp(epsilon, one_s - epsilon)
                sig_out[idx] = s
                total += -(y * log(safe) + (one_s - y) * log(one_s - safe))

        if is_mean:
            total /= Scalar[Self.dtype](extent)

        var scalar_buf = Buffer[Self.dtype](1)
        scalar_buf[0] = total
        return (scalar_buf^, sig_out^)

    @staticmethod
    def forward(
        pred: Buffer[Self.dtype],
        target: Buffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
    ) -> Tuple[
        Buffer[Self.dtype], Buffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused BCELoss forward (probabilities input). Computes per-element loss + clipped pred.

        Args:
            pred: Predicted probabilities buffer.
            target: Target buffer (same size as pred).
            epsilon: Clipping epsilon for numerical stability.

        Returns:
            (loss_per_element, clipped_pred) buffers.
        """
        var extent = pred.size
        var loss_out = Buffer[Self.dtype](extent)
        var safe_out = Buffer[Self.dtype](extent)

        comptime simd_width = simd_width_of[Self.dtype]()

        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width
        var one = SIMD[Self.dtype, simd_width](1.0)
        var eps = SIMD[Self.dtype, simd_width](epsilon)

        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var p = pred.load[simdwidth=simd_width](idx)
            var y = target.load[simdwidth=simd_width](idx)

            var safe = p.clamp(eps, one - eps)
            var loss = -(y * log(safe) + (one - y) * log(one - safe))

            loss_out.store[simdwidth=simd_width](idx, loss)
            safe_out.store[simdwidth=simd_width](idx, safe)

        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            var one_s = Scalar[Self.dtype](1.0)
            for i in range(remainder):
                var idx = start_idx + i
                var p = pred[idx]
                var y = target[idx]
                var safe = p.clamp(epsilon, one_s - epsilon)
                loss_out[idx] = -(
                    y * log(safe) + (one_s - y) * log(one_s - safe)
                )
                safe_out[idx] = safe

        return (loss_out^, safe_out^)

    @staticmethod
    def forward_reduce(
        pred: Buffer[Self.dtype],
        target: Buffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
        is_mean: Bool,
    ) -> Tuple[
        Buffer[Self.dtype], Buffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused BCELoss forward with mean/sum reduction (probabilities input).

        Computes per-element loss + clipped pred in one pass, accumulates loss sum,
        and returns a 1-element scalar loss + full-size clipped_pred buffer.

        Args:
            pred: Predicted probabilities buffer.
            target: Target buffer (same size as pred).
            epsilon: Clipping epsilon.
            is_mean: If True, divide sum by N (mean); else return sum.

        Returns:
            (scalar_loss[1], clipped_pred[N]) buffers.
        """
        var extent = pred.size
        var safe_out = Buffer[Self.dtype](extent)

        comptime simd_width = simd_width_of[Self.dtype]()

        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width
        var one = SIMD[Self.dtype, simd_width](1.0)
        var eps = SIMD[Self.dtype, simd_width](epsilon)

        var total = Scalar[Self.dtype](0)

        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var p = pred.load[simdwidth=simd_width](idx)
            var y = target.load[simdwidth=simd_width](idx)

            var safe = p.clamp(eps, one - eps)
            var loss = -(y * log(safe) + (one - y) * log(one - safe))

            safe_out.store[simdwidth=simd_width](idx, safe)
            total += loss.reduce_add()

        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            var one_s = Scalar[Self.dtype](1.0)
            for i in range(remainder):
                var idx = start_idx + i
                var p = pred[idx]
                var y = target[idx]
                var safe = p.clamp(epsilon, one_s - epsilon)
                safe_out[idx] = safe
                total += -(y * log(safe) + (one_s - y) * log(one_s - safe))

        if is_mean:
            total /= Scalar[Self.dtype](extent)

        var scalar_buf = Buffer[Self.dtype](1)
        scalar_buf[0] = total
        return (scalar_buf^, safe_out^)

    @staticmethod
    def backward_with_logits(
        sigmoid: Buffer[Self.dtype],
        target: Buffer[Self.dtype],
        grad_output: Buffer[Self.dtype],
    ) -> Buffer[Self.dtype] where Self.dtype.is_floating_point():
        """Fused BCEWithLogits backward. Computes gradient in one pass.

        grad = (sigmoid - target) * grad_output

        Args:
            sigmoid: Sigmoid buffer.
            target: Target buffer.
            grad_output: Upstream gradient buffer.

        Returns:
            Gradient buffer for the input (logits).
        """
        var extent = sigmoid.size
        var grad_out = Buffer[Self.dtype](extent)

        comptime simd_width = simd_width_of[Self.dtype]()

        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width

        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var s = sigmoid.load[simdwidth=simd_width](idx)
            var y = target.load[simdwidth=simd_width](idx)
            var g = grad_output.load[simdwidth=simd_width](idx)
            var grad = (s - y) * g
            grad_out.store[simdwidth=simd_width](idx, grad)

        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            for i in range(remainder):
                var idx = start_idx + i
                var s = sigmoid[idx]
                var y = target[idx]
                var g = grad_output[idx]
                grad_out[idx] = (s - y) * g

        return grad_out^

    @staticmethod
    def backward_with_logits_scaled(
        sigmoid: Buffer[Self.dtype],
        target: Buffer[Self.dtype],
        scalar_grad: Scalar[Self.dtype],
    ) -> Buffer[Self.dtype] where Self.dtype.is_floating_point():
        """Fused BCEWithLogits backward with scalar gradient.

        grad[i] = (sigmoid[i] - target[i]) * scalar_grad

        No per-element grad_output buffer needed — single scalar multiplier.

        Args:
            sigmoid: Sigmoid buffer.
            target: Target buffer.
            scalar_grad: Scalar upstream gradient (already divided by N for mean).

        Returns:
            Gradient buffer for the input (logits).
        """
        var extent = sigmoid.size
        var grad_out = Buffer[Self.dtype](extent)

        comptime simd_width = simd_width_of[Self.dtype]()

        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width
        var s_grad = SIMD[Self.dtype, simd_width](scalar_grad)

        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var s = sigmoid.load[simdwidth=simd_width](idx)
            var y = target.load[simdwidth=simd_width](idx)
            var grad = (s - y) * s_grad
            grad_out.store[simdwidth=simd_width](idx, grad)

        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            for i in range(remainder):
                var idx = start_idx + i
                var s = sigmoid[idx]
                var y = target[idx]
                grad_out[idx] = (s - y) * scalar_grad

        return grad_out^

    @staticmethod
    def backward(
        safe: Buffer[Self.dtype],
        target: Buffer[Self.dtype],
        grad_output: Buffer[Self.dtype],
    ) -> Buffer[Self.dtype] where Self.dtype.is_floating_point():
        """Fused BCELoss backward. Computes gradient in one pass.

        grad = -(target/safe - (1-target)/(1-safe)) * grad_output

        Args:
            safe: Clipped prediction buffer (safe = clip(pred, eps, 1-eps)).
            target: Target buffer.
            grad_output: Upstream gradient buffer.

        Returns:
            Gradient buffer for the input (pred).
        """
        var extent = safe.size
        var grad_out = Buffer[Self.dtype](extent)

        comptime simd_width = simd_width_of[Self.dtype]()

        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width
        var one_n = SIMD[Self.dtype, simd_width](1.0)

        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var saf = safe.load[simdwidth=simd_width](idx)
            var y = target.load[simdwidth=simd_width](idx)
            var g = grad_output.load[simdwidth=simd_width](idx)
            var one_minus_y = one_n - y
            var one_minus_safe = one_n - saf
            var grad = -(y / saf - one_minus_y / one_minus_safe) * g
            grad_out.store[simdwidth=simd_width](idx, grad)

        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            var one_s = Scalar[Self.dtype](1.0)
            for i in range(remainder):
                var idx = start_idx + i
                var saf = safe[idx]
                var y = target[idx]
                var g = grad_output[idx]
                var one_minus_y = one_s - y
                var one_minus_safe_val = one_s - saf
                grad_out[idx] = (
                    -(y / saf - one_minus_y / one_minus_safe_val) * g
                )

        return grad_out^

    @staticmethod
    def backward_scaled(
        safe: Buffer[Self.dtype],
        target: Buffer[Self.dtype],
        scalar_grad: Scalar[Self.dtype],
    ) -> Buffer[Self.dtype] where Self.dtype.is_floating_point():
        """Fused BCELoss backward with scalar gradient.

        grad[i] = -(target[i]/safe[i] - (1-target[i])/(1-safe[i])) * scalar_grad

        No per-element grad_output buffer needed — single scalar multiplier.

        Args:
            safe: Clipped prediction buffer (safe = clip(pred, eps, 1-eps)).
            target: Target buffer.
            scalar_grad: Scalar upstream gradient (already divided by N for mean).

        Returns:
            Gradient buffer for the input (pred).
        """
        var extent = safe.size
        var grad_out = Buffer[Self.dtype](extent)

        comptime simd_width = simd_width_of[Self.dtype]()

        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width
        var one = SIMD[Self.dtype, simd_width](1.0)
        var s_grad = SIMD[Self.dtype, simd_width](scalar_grad)

        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var saf = safe.load[simdwidth=simd_width](idx)
            var y = target.load[simdwidth=simd_width](idx)
            var one_minus_y = one - y
            var one_minus_safe = one - saf
            var grad = -(y / saf - one_minus_y / one_minus_safe) * s_grad
            grad_out.store[simdwidth=simd_width](idx, grad)

        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            var one_s = Scalar[Self.dtype](1.0)
            for i in range(remainder):
                var idx = start_idx + i
                var saf = safe[idx]
                var y = target[idx]
                var one_minus_y = one_s - y
                var one_minus_safe_val = one_s - saf
                grad_out[idx] = (
                    -(y / saf - one_minus_y / one_minus_safe_val) * scalar_grad
                )

        return grad_out^


# =============================================================================
# NDBuffer-level dispatch — GPU → BceKernel, CPU → BceBuffer / strided fallback
# =============================================================================


@fieldwise_init
struct BceNdBuffer[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def forward_with_logits(
        logits: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
        sync: Bool = True,
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused BCEWithLogits forward. Returns (per_element_loss, sigmoid).

        Device-aware: GPU → BceKernel.launch_forward_with_logits.
        CPU → BceBuffer.forward_with_logits (contiguous) or scalar fallback.
        """
        var loss_ndb: NDBuffer[Self.dtype]
        var bw_ndb: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if logits.is_on_gpu():
                try:
                    var result = BceKernel[
                        Self.dtype
                    ].launch_forward_with_logits(logits, target, epsilon, sync=sync)
                    loss_ndb = result[0]
                    bw_ndb = result[1]
                except e:
                    print(e)
                    panic(
                        "BceNdBuffer forward_with_logits → GPU operation failed"
                    )
                    loss_ndb = NDBuffer[Self.dtype].Empty()
                    bw_ndb = NDBuffer[Self.dtype].Empty()
            else:
                (loss_ndb, bw_ndb) = BceNdBuffer[
                    Self.dtype
                ].forward_with_logits_cpu(logits, target, epsilon)
        else:
            (loss_ndb, bw_ndb) = BceNdBuffer[
                Self.dtype
            ].forward_with_logits_cpu(logits, target, epsilon)

        return (loss_ndb^, bw_ndb^)

    @staticmethod
    def forward_with_logits_cpu(
        logits: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        if logits.is_contiguous():
            var result = BceBuffer[Self.dtype].forward_with_logits(
                logits.buffer, target.buffer, epsilon
            )
            var loss_ndb = NDBuffer[Self.dtype](result[0], logits.shape)
            var bw_ndb = NDBuffer[Self.dtype](result[1], logits.shape)
            return (loss_ndb^, bw_ndb^)
        else:
            var extent = logits.numels()
            var loss_buf = Buffer[Self.dtype](extent)
            var sig_buf = Buffer[Self.dtype](extent)
            var idx = 0
            for coord in logits.index_iterator():
                loss_buf[idx] = BceElement[Self.dtype].with_logits_element(
                    logits.buffer[coord], target.buffer[coord], epsilon
                )
                sig_buf[idx] = BceElement[Self.dtype].sigmoid_fn(
                    logits.buffer[coord]
                )
                idx += 1
            return (
                NDBuffer[Self.dtype](loss_buf^, logits.shape),
                NDBuffer[Self.dtype](sig_buf^, logits.shape),
            )

    @staticmethod
    def forward_with_logits_reduce(
        logits: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
        is_mean: Bool,
        sync: Bool = True,
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused BCEWithLogits forward with mean/sum reduction.

        Returns (scalar_loss, sigmoid[N]).
        Device-aware: GPU → BceKernel.launch_forward_with_logits_reduce.
        CPU → BceBuffer.forward_with_logits_reduce (contiguous) or scalar fallback.
        """
        var scalar_ndb: NDBuffer[Self.dtype]
        var sig_ndb: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if logits.is_on_gpu():
                try:
                    var result = BceKernel[
                        Self.dtype
                    ].launch_forward_with_logits_reduce(
                        logits, target, epsilon, is_mean, sync=sync
                    )
                    scalar_ndb = result[0]
                    sig_ndb = result[1]
                except e:
                    print(e)
                    panic(
                        "BceNdBuffer forward_with_logits_reduce → GPU"
                        " operation failed"
                    )
                    scalar_ndb = NDBuffer[Self.dtype].Empty()
                    sig_ndb = NDBuffer[Self.dtype].Empty()
            else:
                (scalar_ndb, sig_ndb) = BceNdBuffer[
                    Self.dtype
                ].forward_with_logits_reduce_cpu(
                    logits, target, epsilon, is_mean
                )
        else:
            (scalar_ndb, sig_ndb) = BceNdBuffer[
                Self.dtype
            ].forward_with_logits_reduce_cpu(logits, target, epsilon, is_mean)

        return (scalar_ndb^, sig_ndb^)

    @staticmethod
    def forward_with_logits_reduce_cpu(
        logits: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
        is_mean: Bool,
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        if logits.is_contiguous():
            var result = BceBuffer[Self.dtype].forward_with_logits_reduce(
                logits.buffer, target.buffer, epsilon, is_mean
            )
            var scalar_ndb = NDBuffer[Self.dtype](result[0], Shape())
            var sig_ndb = NDBuffer[Self.dtype](result[1], logits.shape)
            return (scalar_ndb^, sig_ndb^)
        else:
            var extent = logits.numels()
            var sig_buf = Buffer[Self.dtype](extent)
            var total = Scalar[Self.dtype](0)
            var idx = 0
            for coord in logits.index_iterator():
                var s = BceElement[Self.dtype].sigmoid_fn(logits.buffer[coord])
                sig_buf[idx] = s
                total += BceElement[Self.dtype].with_logits_element(
                    logits.buffer[coord], target.buffer[coord], epsilon
                )
                idx += 1
            if is_mean:
                total /= Scalar[Self.dtype](extent)
            var scalar_buf = Buffer[Self.dtype](1)
            scalar_buf[0] = total
            return (
                NDBuffer[Self.dtype](scalar_buf^, Shape()),
                NDBuffer[Self.dtype](sig_buf^, logits.shape),
            )

    @staticmethod
    def forward(
        pred: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
        sync: Bool = True,
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused BCELoss forward (probabilities). Returns (per_element_loss, clipped_pred).

        Device-aware: GPU → BceKernel.launch_forward.
        CPU → BceBuffer.forward (contiguous) or scalar fallback.
        """
        var loss_ndb: NDBuffer[Self.dtype]
        var bw_ndb: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if pred.is_on_gpu():
                try:
                    var result = BceKernel[Self.dtype].launch_forward(
                        pred, target, epsilon, sync=sync
                    )
                    loss_ndb = result[0]
                    bw_ndb = result[1]
                except e:
                    print(e)
                    panic("BceNdBuffer forward → GPU operation failed")
                    loss_ndb = NDBuffer[Self.dtype].Empty()
                    bw_ndb = NDBuffer[Self.dtype].Empty()
            else:
                (loss_ndb, bw_ndb) = BceNdBuffer[Self.dtype].forward_cpu(
                    pred, target, epsilon
                )
        else:
            (loss_ndb, bw_ndb) = BceNdBuffer[Self.dtype].forward_cpu(
                pred, target, epsilon
            )

        return (loss_ndb^, bw_ndb^)

    @staticmethod
    def forward_cpu(
        pred: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        if pred.is_contiguous():
            var result = BceBuffer[Self.dtype].forward(
                pred.buffer, target.buffer, epsilon
            )
            var loss_ndb = NDBuffer[Self.dtype](result[0], pred.shape)
            var bw_ndb = NDBuffer[Self.dtype](result[1], pred.shape)
            return (loss_ndb^, bw_ndb^)
        else:
            var extent = pred.numels()
            var loss_buf = Buffer[Self.dtype](extent)
            var safe_buf = Buffer[Self.dtype](extent)
            var idx = 0
            for coord in pred.index_iterator():
                loss_buf[idx] = BceElement[Self.dtype].element(
                    pred.buffer[coord], target.buffer[coord], epsilon
                )
                safe_buf[idx] = BceElement[Self.dtype].clip_fn(
                    pred.buffer[coord], epsilon
                )
                idx += 1
            return (
                NDBuffer[Self.dtype](loss_buf^, pred.shape),
                NDBuffer[Self.dtype](safe_buf^, pred.shape),
            )

    @staticmethod
    def forward_reduce(
        pred: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
        is_mean: Bool,
        sync: Bool = True,
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused BCELoss forward with mean/sum reduction (probabilities input).

        Returns (scalar_loss, clipped_pred[N]).
        Device-aware: GPU → BceKernel.launch_forward_reduce.
        CPU → BceBuffer.forward_reduce (contiguous) or scalar fallback.
        """
        var scalar_ndb: NDBuffer[Self.dtype]
        var safe_ndb: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if pred.is_on_gpu():
                try:
                    var result = BceKernel[Self.dtype].launch_forward_reduce(
                        pred, target, epsilon, is_mean, sync=sync
                    )
                    scalar_ndb = result[0]
                    safe_ndb = result[1]
                except e:
                    print(e)
                    panic("BceNdBuffer forward_reduce → GPU operation failed")
                    scalar_ndb = NDBuffer[Self.dtype].Empty()
                    safe_ndb = NDBuffer[Self.dtype].Empty()
            else:
                (scalar_ndb, safe_ndb) = BceNdBuffer[
                    Self.dtype
                ].forward_reduce_cpu(pred, target, epsilon, is_mean)
        else:
            (scalar_ndb, safe_ndb) = BceNdBuffer[Self.dtype].forward_reduce_cpu(
                pred, target, epsilon, is_mean
            )

        return (scalar_ndb^, safe_ndb^)

    @staticmethod
    def forward_reduce_cpu(
        pred: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
        is_mean: Bool,
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        if pred.is_contiguous():
            var result = BceBuffer[Self.dtype].forward_reduce(
                pred.buffer, target.buffer, epsilon, is_mean
            )
            var scalar_ndb = NDBuffer[Self.dtype](result[0], Shape())
            var safe_ndb = NDBuffer[Self.dtype](result[1], pred.shape)
            return (scalar_ndb^, safe_ndb^)
        else:
            var extent = pred.numels()
            var safe_buf = Buffer[Self.dtype](extent)
            var total = Scalar[Self.dtype](0)
            var idx = 0
            for coord in pred.index_iterator():
                var safe = BceElement[Self.dtype].clip_fn(
                    pred.buffer[coord], epsilon
                )
                safe_buf[idx] = safe
                total += BceElement[Self.dtype].element(
                    pred.buffer[coord], target.buffer[coord], epsilon
                )
                idx += 1
            if is_mean:
                total /= Scalar[Self.dtype](extent)
            var scalar_buf = Buffer[Self.dtype](1)
            scalar_buf[0] = total
            return (
                NDBuffer[Self.dtype](scalar_buf^, Shape()),
                NDBuffer[Self.dtype](safe_buf^, pred.shape),
            )

    @staticmethod
    def backward_with_logits(
        sigmoid: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        grad_output: NDBuffer[Self.dtype],
        sync: Bool = True,
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """Fused BCEWithLogits backward. Returns gradient for logits.

        Device-aware: GPU → BceKernel.launch_bce_with_logits_backward.
        CPU → BceBuffer.backward_with_logits (contiguous) or scalar fallback.
        """
        result: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if sigmoid.is_on_gpu():
                try:
                    var gpu_target = (
                        target if target.is_on_gpu() else target.to_device(
                            sigmoid.device()
                        )[1]
                    )
                    var gpu_grad = grad_output if grad_output.is_on_gpu() else grad_output.to_device(
                        sigmoid.device()
                    )[
                        1
                    ]
                    result = BceKernel[
                        Self.dtype
                    ].launch_bce_with_logits_backward(
                        sigmoid, gpu_target, gpu_grad, sync=sync
                    )
                except e:
                    print(e)
                    panic(
                        "BceNdBuffer backward_with_logits → GPU operation"
                        " failed"
                    )
                    result = NDBuffer[Self.dtype].Empty()
            else:
                result = BceNdBuffer[Self.dtype].backward_with_logits_cpu(
                    sigmoid, target, grad_output
                )
        else:
            result = BceNdBuffer[Self.dtype].backward_with_logits_cpu(
                sigmoid, target, grad_output
            )
        return result^

    @staticmethod
    def backward_with_logits_cpu(
        sigmoid: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        grad_output: NDBuffer[Self.dtype],
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        if sigmoid.is_contiguous():
            var buf = BceBuffer[Self.dtype].backward_with_logits(
                sigmoid.buffer, target.buffer, grad_output.buffer
            )
            return NDBuffer[Self.dtype](buf^, sigmoid.shape)
        else:
            var extent = sigmoid.numels()
            var grad_buf = Buffer[Self.dtype](extent)
            var idx = 0
            for coord in sigmoid.index_iterator():
                grad_buf[idx] = BceElement[
                    Self.dtype
                ].with_logits_backward_element(
                    sigmoid.buffer[coord],
                    target.buffer[coord],
                    grad_output.buffer[coord],
                )
                idx += 1
            return NDBuffer[Self.dtype](grad_buf^, sigmoid.shape)

    @staticmethod
    def backward_with_logits_scaled(
        sigmoid: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        scalar_grad: Scalar[Self.dtype],
        sync: Bool = True,
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """Fused BCEWithLogits backward with scalar gradient.

        grad[i] = (sigmoid[i] - target[i]) * scalar_grad.
        Device-aware: GPU → BceKernel.launch_bce_with_logits_backward_scaled.
        CPU → BceBuffer.backward_with_logits_scaled (contiguous) or scalar fallback.
        """
        result: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if sigmoid.is_on_gpu():
                try:
                    result = BceKernel[
                        Self.dtype
                    ].launch_bce_with_logits_backward_scaled(
                        sigmoid, target, scalar_grad, sync=sync
                    )
                except e:
                    print(e)
                    panic(
                        "BceNdBuffer backward_with_logits_scaled → GPU"
                        " operation failed"
                    )
                    result = NDBuffer[Self.dtype].Empty()
            else:
                result = BceNdBuffer[
                    Self.dtype
                ].backward_with_logits_scaled_cpu(sigmoid, target, scalar_grad)
        else:
            result = BceNdBuffer[Self.dtype].backward_with_logits_scaled_cpu(
                sigmoid, target, scalar_grad
            )
        return result^

    @staticmethod
    def backward_with_logits_scaled_cpu(
        sigmoid: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        scalar_grad: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        if sigmoid.is_contiguous():
            var buf = BceBuffer[Self.dtype].backward_with_logits_scaled(
                sigmoid.buffer, target.buffer, scalar_grad
            )
            return NDBuffer[Self.dtype](buf^, sigmoid.shape)
        else:
            var extent = sigmoid.numels()
            var grad_buf = Buffer[Self.dtype](extent)
            var idx = 0
            for coord in sigmoid.index_iterator():
                grad_buf[idx] = BceElement[
                    Self.dtype
                ].with_logits_backward_element(
                    sigmoid.buffer[coord],
                    target.buffer[coord],
                    scalar_grad,
                )
                idx += 1
            return NDBuffer[Self.dtype](grad_buf^, sigmoid.shape)

    @staticmethod
    def backward(
        safe: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        grad_output: NDBuffer[Self.dtype],
        sync: Bool = True,
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """Fused BCELoss backward. Returns gradient for pred.

        Device-aware: GPU → BceKernel.launch_bce_backward.
        CPU → BceBuffer.backward (contiguous) or scalar fallback.
        """
        result: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if safe.is_on_gpu():
                try:
                    var gpu_target = (
                        target if target.is_on_gpu() else target.to_device(
                            safe.device()
                        )[1]
                    )
                    var gpu_grad = grad_output if grad_output.is_on_gpu() else grad_output.to_device(
                        safe.device()
                    )[
                        1
                    ]
                    result = BceKernel[Self.dtype].launch_bce_backward(
                        safe, gpu_target, gpu_grad, sync=sync
                    )
                except e:
                    print(e)
                    panic("BceNdBuffer backward → GPU operation failed")
                    result = NDBuffer[Self.dtype].Empty()
            else:
                result = BceNdBuffer[Self.dtype].backward_cpu(
                    safe, target, grad_output
                )
        else:
            result = BceNdBuffer[Self.dtype].backward_cpu(
                safe, target, grad_output
            )
        return result^

    @staticmethod
    def backward_cpu(
        safe: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        grad_output: NDBuffer[Self.dtype],
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        if safe.is_contiguous():
            var buf = BceBuffer[Self.dtype].backward(
                safe.buffer, target.buffer, grad_output.buffer
            )
            return NDBuffer[Self.dtype](buf^, safe.shape)
        else:
            var extent = safe.numels()
            var grad_buf = Buffer[Self.dtype](extent)
            var idx = 0
            for coord in safe.index_iterator():
                grad_buf[idx] = BceElement[Self.dtype].backward_element(
                    safe.buffer[coord],
                    target.buffer[coord],
                    grad_output.buffer[coord],
                )
                idx += 1
            return NDBuffer[Self.dtype](grad_buf^, safe.shape)

    @staticmethod
    def backward_scaled(
        safe: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        scalar_grad: Scalar[Self.dtype],
        sync: Bool = True,
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """Fused BCELoss backward with scalar gradient.

        grad[i] = -(target[i]/safe[i] - (1-target[i])/(1-safe[i])) * scalar_grad.
        Device-aware: GPU → BceKernel.launch_bce_backward_scaled.
        CPU → BceBuffer.backward_scaled (contiguous) or scalar fallback.
        """
        result: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if safe.is_on_gpu():
                try:
                    result = BceKernel[Self.dtype].launch_bce_backward_scaled(
                        safe, target, scalar_grad, sync=sync
                    )
                except e:
                    print(e)
                    panic("BceNdBuffer backward_scaled → GPU operation failed")
                    result = NDBuffer[Self.dtype].Empty()
            else:
                result = BceNdBuffer[Self.dtype].backward_scaled_cpu(
                    safe, target, scalar_grad
                )
        else:
            result = BceNdBuffer[Self.dtype].backward_scaled_cpu(
                safe, target, scalar_grad
            )
        return result^

    @staticmethod
    def backward_scaled_cpu(
        safe: NDBuffer[Self.dtype],
        target: NDBuffer[Self.dtype],
        scalar_grad: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        if safe.is_contiguous():
            var buf = BceBuffer[Self.dtype].backward_scaled(
                safe.buffer, target.buffer, scalar_grad
            )
            return NDBuffer[Self.dtype](buf^, safe.shape)
        else:
            var extent = safe.numels()
            var grad_buf = Buffer[Self.dtype](extent)
            var idx = 0
            for coord in safe.index_iterator():
                grad_buf[idx] = BceElement[Self.dtype].backward_element(
                    safe.buffer[coord],
                    target.buffer[coord],
                    scalar_grad,
                )
                idx += 1
            return NDBuffer[Self.dtype](grad_buf^, safe.shape)


# =============================================================================
# Forward structs
# =============================================================================


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
            var (loss_ndb, bw) = BceNdBuffer[Self.dtype].forward_with_logits(
                logits.buffer, target.buffer, epsilon
            )
            out = Tensor[Self.dtype](loss_ndb^, requires_grad=False)
            sigmoid_ndb = bw^
        else:
            var is_mean = reduction.is_mean()
            var (scalar_ndb, bw) = BceNdBuffer[
                Self.dtype
            ].forward_with_logits_reduce(
                logits.buffer, target.buffer, epsilon, is_mean
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
            var (loss_ndb, bw) = BceNdBuffer[Self.dtype].forward(
                pred.buffer, target.buffer, epsilon
            )
            out = Tensor[Self.dtype](loss_ndb^, requires_grad=False)
            safe_ndb = bw^
        else:
            var is_mean = reduction.is_mean()
            var (scalar_ndb, bw) = BceNdBuffer[Self.dtype].forward_reduce(
                pred.buffer, target.buffer, epsilon, is_mean
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


# =============================================================================
# Backward handlers
# =============================================================================


@fieldwise_init
struct BCEWithLogitsBackward[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
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
            var grad_ndb = BceNdBuffer[Self.dtype].backward_with_logits(
                sigmoid, target, gradbox.buffer
            )
            var grad_parent = Gradbox[Self.dtype](grad_ndb^)
            parent.update_grad(grad_parent^, AddTensor, None)
        else:
            var multiplier = gradbox.item()
            if bwd_arg.reduction.is_mean():
                multiplier /= Scalar[Self.dtype](bwd_arg.numels)
            var grad_ndb = BceNdBuffer[Self.dtype].backward_with_logits_scaled(
                sigmoid, target, multiplier
            )
            var grad_parent = Gradbox[Self.dtype](grad_ndb^)
            parent.update_grad(grad_parent^, AddTensor, None)
        parent_ids.append(parent._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct BCELossBackward[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ) where Self.dtype.is_floating_point():
        var bwd_arg = (
            output.ancestry().backward_fn_arg().get[BCELossBwdArg[Self.dtype]]()
        )
        var safe = bwd_arg.clipped_pred
        var target = bwd_arg.target
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)

        if bwd_arg.reduction.is_none():
            var grad_ndb = BceNdBuffer[Self.dtype].backward(
                safe, target, gradbox.buffer
            )
            var grad_parent = Gradbox[Self.dtype](grad_ndb^)
            parent.update_grad(grad_parent^, AddTensor, None)
        else:
            var multiplier = gradbox.item()
            if bwd_arg.reduction.is_mean():
                multiplier /= Scalar[Self.dtype](bwd_arg.numels)
            var grad_ndb = BceNdBuffer[Self.dtype].backward_scaled(
                safe, target, multiplier
            )
            var grad_parent = Gradbox[Self.dtype](grad_ndb^)
            parent.update_grad(grad_parent^, AddTensor, None)
        parent_ids.append(parent._id)
        if not retain_graph:
            gradbox.zero_grad()
