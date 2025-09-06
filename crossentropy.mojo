from tensors import Tensor
from shared import TensorLite
from operators import AddTensor
from intlist import IntList
from shapes import Shape
from validators import Validator
from backpropagation import Delegate, BackwardFn
from subtraction import Subtractor
from common_utils import panic
from math import log

alias SoftmaxOutput[dtype: DType] = List[(IntList, Scalar[dtype])]


@fieldwise_init
struct SoftmaxBackward[dtype: DType](Copyable & Movable):
    var axes: IntList
    var softmax_out: SoftmaxOutput[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        incoming = output.gradients()[]
        softmax_out = Tensor[dtype].zeros(incoming.shape)
        for indices, value in self.softmax_out:
            softmax_out[indices] = rebind[Scalar[dtype]](value)

        print("incoming")
        print()
        incoming.print()
        print()
        softmax_out.print()
        sum_grad = (incoming * softmax_out).sum(
            self.axes, keepdims=True, track_grad=False
        )
        print()
        print("sum_grad")
        sum_grad.print()

        grad_share = softmax_out * (incoming - sum_grad)

        print()
        print("grad_share")
        grad_share.print()
        ancestor = output.ancestry().get(0)[]
        return [(ancestor, grad_share, AddTensor)]


@fieldwise_init
@register_passable
struct Softmax[dtype: DType]:
    @staticmethod
    fn softmax(
        this: Tensor[dtype],
        axes: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = this.shape
        # Normalize axes
        normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        max_vals = this.max(normalized_axes, keepdims=True, requires_grad=False)
        # Numerical stability: subtract max along axes
        stable = this - max_vals
        # Compute exponentials
        stable_exp = stable.exp()
        exp_sum = stable_exp.sum(
            normalized_axes, keepdims=True, track_grad=False
        )
        # Softmax = exp(x) / sum(exp(x))
        out = stable_exp / exp_sum

        grad_required = (
            requires_grad.value() if requires_grad else this.requires_grad
        )

        if grad_required:
            out.requires_grad_(True)
            softmax_out = SoftmaxOutput[dtype](capacity=out.numels())
            for indices in out.shape:
                softmax_out.append((indices, out[indices]))
            print("Forward pass: ", len(softmax_out))
            backward_fn = SoftmaxBackward[dtype](
                normalized_axes, softmax_out
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(this))
        return out


fn main():
    _a = Tensor.arange(5, requires_grad=True)
    print("passes")


@fieldwise_init
@register_passable
struct CrossEntropyLoss[dtype: DType]:
    var reduction: Int  # '0-> mean', '1-> sum', '2 -> none'
    var ignore_index: Int  # index to ignore (-1 for none)
    var label_smoothing: Scalar[dtype]  # usually 0.0

    fn forward(
        self, logits: Tensor[dtype], target: Tensor[dtype]
    ) -> Tensor[dtype]:
        """
        Args:
            logits: Logits tensor of shape (N, C) or (N, C, d1, d2, ...).
            target: Ground truth of shape (N,) or (N, d1, d2, ...) with class indices.
        """
        # 1. Validate input shapes
        var input_shape = logits.shape
        var target_shape = target.shape

        if input_shape.rank() < 2:
            panic("Input must have at least 2 dimensions")

        if target_shape.rank() != input_shape.rank() - 1:
            panic("Target must have one fewer dimension than input")

        # 2. Flatten spatial dimensions if needed (for segmentation etc.)
        var N = input_shape[0]  # batch size
        var C = input_shape[1]  # number of classes
        var spatial_dims: Int
        if input_shape.rank() > 2:
            # Reshape to (N, C, -1) and (N, -1)
            spatial_dims = input_shape[2:].num_elements()
            # input = input.reshape(Shape([N, C, spatial_dims]))
            # target = target.reshape(Shape([N, spatial_dims]))
        else:
            spatial_dims = 1

        # 3. Compute softmax probabilities
        var log_softmax = self._log_softmax(logits)

        # 4. Compute loss
        var loss = self._compute_loss(log_softmax, target, N, C, spatial_dims)

        # 5. Apply reduction
        return self._apply_reduction(loss)

    fn _log_softmax(self, logits: Tensor[dtype]) -> Tensor[dtype]:
        """Numerically stable log(softmax(x))"""
        # Subtract max for numerical stability
        var max_vals = logits.max(
            IntList(1), keepdims=True, requires_grad=False
        )
        var logits_stable = Subtractor[dtype].forward[False](logits, max_vals)

        # Compute log sum exp
        var sum_exp = logits_stable.exp().sum(
            IntList(1), keepdims=True, track_grad=False
        )
        var log_sum_exp = sum_exp.log(requires_grad=False)

        # log_softmax = x - max - log(sum(exp(x - max)))
        return logits_stable - log_sum_exp

    fn _compute_loss(
        self,
        log_softmax: Tensor[dtype],
        target: Tensor[dtype],
        N: Int,
        C: Int,
        spatial_dims: Int,
    ) -> Tensor[dtype]:
        """Compute the actual loss value."""
        # Create one-hot encoding if needed, or use gather
        var loss = Tensor[dtype](
            Shape([N, spatial_dims]), requires_grad=log_softmax.requires_grad
        )

        # Use gather to get the log probability of the correct class
        for n in range(N):
            for s in range(spatial_dims):
                var class_idx = target[n, s].__int__()

                # Handle ignore_index
                if class_idx == self.ignore_index:
                    loss[n, s] = Scalar[dtype](0)
                    continue

                # Get log probability of correct class
                var log_prob = log_softmax[n, class_idx, s]

                # Apply label smoothing if needed
                if self.label_smoothing > 0.0:
                    var smooth_prob = self.label_smoothing / C
                    var smooth_term = log(smooth_prob) * (C - 1) / C
                    loss[n, s] = (
                        -log_prob * (1 - self.label_smoothing) - smooth_term
                    )
                else:
                    loss[n, s] = -log_prob

        return loss

    fn _apply_reduction(self, loss: Tensor[dtype]) -> Tensor[dtype]:
        """Apply reduction: 'mean' -> 0, 'sum -> 1', or 'none - 2'"""
        if self.reduction == 0:
            return loss.mean()
        elif self.reduction == 1:
            return loss.sum()
        else:  # "none"
            return loss

    fn backward(
        self,
        grad_output: Tensor[dtype],
        logits: Tensor[dtype],
        target: Tensor[dtype],
    ) -> Tensor[dtype]:
        """Backward pass: ∂L/∂input = softmax(input) - one_hot(target)."""
        var N = logits.shape[0]
        var _C = logits.shape[1]
        var spatial_dims = (
            logits.shape[2:].num_elements() if logits.shape.rank() > 2 else 1
        )

        # 1. Compute softmax probabilities
        var softmax_probs = logits.softmax([1], requires_grad=False)

        # 2. Create one-hot encoding of target
        var grad_input = Tensor[dtype].zeros_like(softmax_probs)

        # 3. Subtract 1 from the correct class probabilities
        for n in range(N):
            for s in range(spatial_dims):
                var class_idx = target[n, s].__int__()
                if class_idx != self.ignore_index:
                    grad_input[n, class_idx, s] = softmax_probs[
                        n, class_idx, s
                    ] - Scalar[dtype](1.0)
                # Other classes remain softmax_probs[n, j, s] - 0 = softmax_probs[n, j, s]

        # 4. Apply reduction scaling
        if self.reduction == 0:  # "mean"
            grad_input = grad_input / (N * spatial_dims)
        elif self.reduction == 1:  # "sum"
            # grad_input remains as is for sum reduction
            pass

        # 5. Multiply by upstream gradient
        return grad_input * grad_output
