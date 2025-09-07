from tensors import Tensor
from shared import TensorLite
from intlist import IntList
from shapes import Shape
from common_utils import panic
from math import log
from subtraction import Subtractor
from backpropagation import Delegate, BackwardFn
from operators import AddTensor


@fieldwise_init
@register_passable
struct CrossEntropyBackward[dtype: DType](Copyable):
    var reduction: Int  # '0-> mean', '1-> sum', '2 -> none'
    var ignore_index: Int  # index to ignore (-1 for none)
    var label_smoothing: Scalar[dtype]  # usually 0.0

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]
        logits = ancestor_1.tensor()
        target = ancestor_2.tensor()

        # Backward pass: ∂L/∂input = softmax(input) - one_hot(target)
        var input_shape = logits.shape
        var N = input_shape[0]
        var C = input_shape[1]
        var spatial_dims: Int

        # Reshape inputs consistently
        var logits_reshaped: Tensor[dtype]
        var target_reshaped: Tensor[dtype]

        if input_shape.rank() > 2:
            spatial_dims = input_shape[2:].num_elements()
            logits_reshaped = logits.reshape(Shape([N, C, spatial_dims]))
            target_reshaped = target.reshape(Shape([N, spatial_dims]))
        else:
            # Reshape 2D case to 3D for consistency
            spatial_dims = 1
            logits_reshaped = logits.reshape(Shape([N, C, 1]))
            target_reshaped = target.reshape(Shape([N, 1]))

        # 1. Compute softmax probabilities
        var softmax_probs = logits_reshaped.softmax([1], requires_grad=False)

        # 2. Create gradient tensor
        var grad_input = Tensor[dtype].zeros_like(softmax_probs)

        # 3. Compute gradients for ALL class positions (CRITICAL FIX)
        for n in range(N):
            for s in range(spatial_dims):
                var class_idx = target_reshaped[n, s].__int__()

                # Skip if ignore_index
                if class_idx == self.ignore_index:
                    continue

                # Update gradients for ALL classes, not just the correct one
                for j in range(C):
                    if j == class_idx:
                        # Correct class: softmax - 1
                        grad_input[n, j, s] = softmax_probs[n, j, s] - Scalar[
                            dtype
                        ](1.0)
                    else:
                        # Incorrect classes: softmax - 0 = softmax
                        grad_input[n, j, s] = softmax_probs[n, j, s]

        # 4. Apply reduction scaling
        if self.reduction == 0:  # "mean"
            grad_input = grad_input / (N * spatial_dims)
        elif self.reduction == 1:  # "sum"
            # grad_input remains as is for sum reduction
            pass

        # 5. Reshape grad_input back to original logits shape
        var final_grad_input = grad_input.reshape(input_shape)

        # 6. Multiply by upstream gradient
        return [(ancestor_1, final_grad_input * gradients, AddTensor)]


@fieldwise_init
@register_passable
struct CrossEntropyLoss[dtype: DType = DType.float32, track_grad: Bool = True](
    Copyable
):
    var reduction: Int  # '0-> mean', '1-> sum', '2 -> none'
    var ignore_index: Int  # index to ignore (-1 for none)
    var label_smoothing: Scalar[dtype]  # usually 0.0

    fn __init__(out self):
        self.reduction = 0
        self.ignore_index = -1
        self.label_smoothing = Scalar[dtype](0)

    fn __call__(
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
        var logits_reshaped: Tensor[dtype]
        var target_reshaped: Tensor[dtype]

        if input_shape.rank() > 2:
            # Reshape to (N, C, -1) and (N, -1)
            spatial_dims = input_shape[2:].num_elements()
            logits_reshaped = logits.reshape(Shape([N, C, spatial_dims]))
            target_reshaped = target.reshape(Shape([N, spatial_dims]))
        else:
            # CRITICAL FIX: Even for 2D case, reshape target to (N, 1)
            spatial_dims = 1
            target_reshaped = target.reshape(
                Shape([N, 1])
            )  # Reshape (N,) to (N, 1)
            logits_reshaped = logits.reshape(
                Shape([N, C, 1])
            )  # Reshape (N, C) to (N, C, 1)

        # 3. Compute softmax probabilities
        var log_softmax = self._log_softmax(logits_reshaped)

        # 4. Compute loss
        var loss = self._compute_loss(
            log_softmax, target_reshaped, N, C, spatial_dims
        )

        # 5. Apply reduction
        out = self._apply_reduction(loss)

        @parameter
        if track_grad:
            if logits.requires_grad:
                out.requires_grad_(True)
                reduction = self.reduction
                ignore_index = self.ignore_index
                label_smoothing = self.label_smoothing

                backward_fn = CrossEntropyBackward[dtype](
                    reduction, ignore_index, label_smoothing
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite[dtype].of(logits))
                out.add_ancestry(TensorLite[dtype].of(target))

        return out

    fn _log_softmax(self, logits: Tensor[dtype]) -> Tensor[dtype]:
        """Numerically stable log(softmax(x))."""
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
        var loss = Tensor[dtype](
            # Shape([N, spatial_dims]), requires_grad=log_softmax.requires_grad
            Shape([N, spatial_dims]),
            requires_grad=False,
        )

        # Use consistent 3D access pattern for all cases
        for n in range(N):
            for s in range(spatial_dims):
                var class_idx = target[n, s].__int__()

                # Handle ignore_index
                if class_idx == self.ignore_index:
                    loss[n, s] = Scalar[dtype](0)
                    continue

                # Get log probability of correct class - now always 3D access
                var log_prob = log_softmax[n, class_idx, s]

                # Apply label smoothing if needed
                if self.label_smoothing > Scalar[dtype](0):
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


fn main() raises:
    test_cross_entropy()
    print("passes")


fn test_cross_entropy() raises:
    print("Testing CrossEntropyLoss...")

    # Example 1: Basic classification
    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]],  # Sample 1  # Sample 2
        requires_grad=True,
    )

    var target = Tensor.d1([0, 1])  # Class indices

    var criterion = CrossEntropyLoss()
    var loss = criterion(logits, target)

    print("Loss:", loss.item())
    loss.backward()
    print("Gradient of logits:")
    logits.gradbox[].print()

    # Example 2: With ignore_index
    # var target_with_ignore = Tensor.d1([0, -1, 1])  # Ignore sample 2
    # var loss_ignore = criterion.forward(logits, target_with_ignore)
    # print("Loss with ignore_index:", loss_ignore.item())
