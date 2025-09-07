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


@register_passable
struct CrossEntropyLoss[dtype: DType = DType.float32, track_grad: Bool = True](
    Copyable
):
    var reduction: Int  # '0-> mean', '1-> sum', '2 -> none'
    var ignore_index: Int  # index to ignore (-1 for none)
    var label_smoothing: Scalar[dtype]  # usually 0.0

    fn __init__(
        out self,
        reduction: Int = 0,
        ignore_index: Int = -1,
        label_smoothing: Scalar[dtype] = Scalar[dtype](0),
    ):
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    fn __copyinit__(out self, existing: Self):
        self.reduction = existing.reduction
        self.ignore_index = existing.ignore_index
        self.label_smoothing = existing.label_smoothing

    fn __call__(
        self, logits: Tensor[dtype], target: Tensor[dtype]
    ) -> Tensor[dtype]:
        """
        Args:
            logits: Logits tensor of shape (N, C) or (N, C, d1, d2, ...).
            target: Ground truth of shape (N,) or (N, d1, d2, ...) with class indices.
        """
        # Validate inputs and get processed tensors
        var (N, C, spatial_dims, logits_reshaped, target_reshaped) = 
            Self.validate_cross_entropy_inputs[dtype](logits, target)

        # Compute softmax probabilities
        var log_softmax = self.log_softmax(logits_reshaped)

        # Compute loss
        var loss = self.compute_loss(
            log_softmax, target_reshaped, N, C, spatial_dims
        )

        # Apply reduction
        var out = self.reduce(loss)

        # Setup autograd if needed
        @parameter
        if track_grad:
            if logits.requires_grad:
                out.requires_grad_(True)
                var backward_fn = CrossEntropyBackward[dtype](
                    self.reduction, self.ignore_index, self.label_smoothing
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite[dtype].of(logits))
                out.add_ancestry(TensorLite[dtype].of(target))

        return out

    
    fn log_softmax(self, logits: Tensor[dtype]) -> Tensor[dtype]:
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

    fn compute_loss(
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

    fn reduce(self, loss: Tensor[dtype]) -> Tensor[dtype]:
        """Apply reduction: 'mean' -> 0, 'sum -> 1', or 'none - 2'."""
        if self.reduction == 0:
            return loss.mean()
        elif self.reduction == 1:
            return loss.sum()
        else:  # "none"
            return loss

    @staticmethod
    fn validate_cross_entropy_inputs[dtype: DType](
        logits: Tensor[dtype], target: Tensor[dtype]
    ) -> Tuple[Int, Int, Int, Tensor[dtype], Tensor[dtype]]:
        """
        Validate CrossEntropyLoss inputs and return processed tensors.
        
        Returns:
            N: batch size.
            C: number of classes.
            spatial_dims: number of spatial elements.
            logits_reshaped: reshaped logits tensor.
            target_reshaped: reshaped target tensor.
        """
        var input_shape = logits.shape
        var target_shape = target.shape
        
        # 1. Validate input ranks
        if input_shape.rank() < 2:
            panic("Input must have at least 2 dimensions")

        if target_shape.rank() != input_shape.rank() - 1:
            panic("Target must have one fewer dimension than input")

        # 2. Validate number of samples matches
        var N = input_shape[0]  # batch size
        if target_shape[0] != N:
            panic(
                "Target must have the same number of samples as logits."
                " Expected "
                + N.__str__()
                + ", got "
                + target_shape[0].__str__()
            )
        
        # 3. Validate spatial dimensions match (if any)
        if target_shape.rank() > 1:
            for i in range(1, target_shape.rank()):
                if i + 1 < input_shape.rank():
                    if target_shape[i] != input_shape[i + 1]:
                        panic(
                            "Spatial dimension mismatch at dim "
                            + i.__str__()
                            + ": expected "
                            + input_shape[i + 1].__str__()
                            + ", got "
                            + target_shape[i].__str__()
                        )

        # 4. Flatten spatial dimensions if needed
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
            # For 2D case, reshape target to (N, 1) and logits to (N, C, 1)
            spatial_dims = 1
            target_reshaped = target.reshape(Shape([N, 1]))
            logits_reshaped = logits.reshape(Shape([N, C, 1]))

        return (N, C, spatial_dims, logits_reshaped, target_reshaped)

fn main() raises:
    print("passes")
