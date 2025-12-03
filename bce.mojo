from tenmo import Tensor
from operators import AddTensor
from backpropagation import Delegate, BackwardFn, BACKWARD_BCE
from ancestry import Ancestor
from gradbox import Gradbox
from sys import simd_width_of
from math import log
from common_utils import panic


@fieldwise_init
#@register_passable
struct BCEBackward[dtype: DType](ImplicitlyCopyable & Movable):
    alias TAG = BACKWARD_BCE
    var epsilon: Scalar[dtype]
    var reduction: String  # "mean", "sum", "none"

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        """Compute BCE gradient: ∂BCE/∂pred = (pred - target) / (pred * (1 - pred)).

        Gradient formula:
            ∂/∂p [-y*log(p) - (1-y)*log(1-p)] = (p - y) / (p(1-p)).

        With mean reduction:
            ∂BCE/∂p_i = (1/N) * (p_i - y_i) / (p_i(1-p_i)) * grad_output.
        """
        ref grad_output = output.gradients()[]

        # Get pred and target ancestors
        var pred_ancestor = output.ancestry().get(0)
        var target_ancestor = output.ancestry().get(1)

        ref pred_tensor = pred_ancestor.tensor()
        ref target_tensor = target_ancestor.tensor()
        ref shape = pred_ancestor.shape()

        var numels = shape.num_elements()

        # Get incoming gradient (should be scalar 1.0 for loss.backward())
        # For scalar output, grad_output is a scalar tensor
        #var incoming_grad = grad_output.buffer.data_buffer().data[0]
        var incoming_grad = grad_output.item()

        # Scale factor depends on reduction type
        var scale_factor: Scalar[dtype]
        if self.reduction == "mean":
            scale_factor = incoming_grad / Scalar[dtype](numels)
        elif self.reduction == "sum":
            scale_factor = incoming_grad
        else:  # "none"
            scale_factor = incoming_grad

        var pred_gradbox = Gradbox[dtype].zeros(shape, share=False)

        if pred_tensor.is_contiguous() and target_tensor.is_contiguous():
            var pred_ptr = pred_tensor.buffer.data_buffer().data
            var target_ptr = target_tensor.buffer.data_buffer().data
            var grad_ptr = pred_gradbox.buffer.data_buffer().data
            var pred_offset = pred_tensor.offset()
            var target_offset = target_tensor.offset()

            alias simd_width = simd_width_of[dtype]()

            for i in range(0, numels - simd_width + 1, simd_width):
                var p = pred_ptr.load[width=simd_width](pred_offset + i)
                var y = target_ptr.load[width=simd_width](target_offset + i)

                # Clamp pred to avoid division by zero
                var p_safe = max(min(p, 1.0 - self.epsilon), self.epsilon)

                # ∂BCE/∂p = (p - y) / (p * (1 - p))
                var numerator = p_safe - y
                var denominator = p_safe * (1.0 - p_safe)
                var local_grad = numerator / denominator

                # Apply chain rule with scale factor
                var grad = scale_factor * local_grad

                grad_ptr.store[width=simd_width](i, grad)

            # Handle remainder
            for i in range(numels - numels % simd_width, numels):
                var p = pred_ptr[pred_offset + i]
                var y = target_ptr[target_offset + i]

                var p_safe = max(min(p, 1.0 - self.epsilon), self.epsilon)
                var numerator = p_safe - y
                var denominator = p_safe * (1.0 - p_safe)
                grad_ptr[i] = scale_factor * (numerator / denominator)
        else:
            # Non-contiguous fallback
            for coord in shape:
                var p = pred_tensor[coord]
                var y = target_tensor[coord]

                var p_safe = max(min(p, 1.0 - self.epsilon), self.epsilon)
                var numerator = p_safe - y
                var denominator = p_safe * (1.0 - p_safe)
                pred_gradbox[coord] = scale_factor * (numerator / denominator)

        # Target doesn't need gradients (it's ground truth)
        return [(pred_ancestor^, pred_gradbox^, AddTensor)]


@fieldwise_init
@register_passable
struct BCE[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        pred: Tensor[dtype],
        target: Tensor[dtype],
        epsilon: Scalar[dtype] = 1e-7,
        reduction: String = "mean",  # "mean", "sum", "none"
    ) -> Tensor[dtype]:
        """Binary Cross Entropy Loss: -[y*log(p) + (1-y)*log(1-p)].

        Args:
            pred: Predicted probabilities, should be in [0, 1].
            target: Ground truth labels, should be in {0, 1}.
            epsilon: Small value for numerical stability.
            reduction: How to reduce the loss.
                - "mean": Return mean of losses (default).
                - "sum": Return sum of losses.
                - "none": Return unreduced losses.

        Returns:
            BCE loss (scalar for mean/sum, tensor for none).
        """
        var shape = pred.shape()
        var numels = shape.num_elements()
        var eps = Scalar[dtype](epsilon)

        if reduction == "none":
            # Return element-wise loss (not implemented here for brevity)
            # Would need to return a tensor with same shape as input
            panic("BCE: reduction='none' not implemented yet")

        var loss_sum = Scalar[dtype](0.0)

        if pred.is_contiguous() and target.is_contiguous():
            var pred_ptr = pred.buffer.data_buffer().data
            var target_ptr = target.buffer.data_buffer().data
            var pred_offset = pred.offset()
            var target_offset = target.offset()

            alias simd_width = simd_width_of[dtype]()

            for i in range(0, numels - simd_width + 1, simd_width):
                var p = pred_ptr.load[width=simd_width](pred_offset + i)
                var y = target_ptr.load[width=simd_width](target_offset + i)

                # Clamp to [epsilon, 1-epsilon]
                var p_safe = max(min(p, 1.0 - eps), eps)

                # BCE: -[y*log(p) + (1-y)*log(1-p)]
                var loss_chunk = -(
                    y * log(p_safe) + (1.0 - y) * log(1.0 - p_safe)
                )
                loss_sum += loss_chunk.reduce_add()

            # Handle remainder
            for i in range(numels - numels % simd_width, numels):
                var p = pred_ptr[pred_offset + i]
                var y = target_ptr[target_offset + i]

                var p_safe = max(min(p, 1.0 - eps), eps)
                loss_sum += -(y * log(p_safe) + (1.0 - y) * log(1.0 - p_safe))
        else:
            # Non-contiguous fallback
            for coord in shape:
                var p = pred[coord]
                var y = target[coord]

                var p_safe = max(min(p, 1.0 - eps), eps)
                loss_sum += -(y * log(p_safe) + (1.0 - y) * log(1.0 - p_safe))

        # Apply reduction
        var loss_value: Scalar[dtype]
        if reduction == "mean":
            loss_value = loss_sum / Scalar[dtype](numels)
        else:  # "sum"
            loss_value = loss_sum

        # Create scalar output tensor
        var loss_tensor = Tensor[dtype].scalar(loss_value, requires_grad=False)

        # Attach backward if needed
        @parameter
        if track_grad:
            if pred.requires_grad or target.requires_grad:
                loss_tensor.requires_grad_(True)
                var backward_fn = BCEBackward[dtype](
                    eps, reduction
                ).into_backward_fn()
                loss_tensor.backwardFn = Optional(backward_fn^)
                loss_tensor.add_ancestry(pred)
                loss_tensor.add_ancestry(target)

        return loss_tensor^


# ========== Test Code ==========
from testing import assert_true

fn test_bce_gradient() raises:
    """Test BCE gradient with numerical gradient checking."""
    alias dtype = DType.float32

    # Simple test case
    var pred = Tensor[dtype].d1(
        [0.7, 0.3, 0.8, 0.2], requires_grad=True
    )
    var target = Tensor[dtype].d1([1.0, 0.0, 1.0, 0.0])

    var loss = BCE[dtype].forward(pred, target)
    loss.backward()

    var grad = pred.grad()

    print("Pred:")
    pred.print()
    print("Target:")
    target.print()
    print("Loss:")
    loss.print()
    print("Gradient:")
    grad.print()

    # Manual gradient check for first element
    # pred[0] = 0.7, target[0] = 1.0
    # grad = (1/4) * (0.7 - 1.0) / (0.7 * 0.3)
    #      = (1/4) * (-0.3) / (0.21)
    #      = -0.3571...
    var expected_grad_0 = Scalar[dtype]((1.0 / 4.0) * (0.7 - 1.0) / (0.7 * 0.3))
    print("Expected grad[0]:", expected_grad_0)
    print("Actual grad[0]:", grad[0])

    # Check if close (within epsilon)
    var diff = abs(grad[0] - expected_grad_0)
    assert_true(diff < 1e-5, "Gradient mismatch!")

    print("✓ BCE gradient test passed!")


fn test_bce_reduction_types() raises:
    """Test different reduction types."""
    alias dtype = DType.float32

    var pred = Tensor[dtype].d1([0.8, 0.6, 0.7, 0.3], requires_grad=True
    )
    var target = Tensor[dtype].d1([1.0, 1.0, 0.0, 0.0])

    # Test mean reduction
    var loss_mean = BCE[dtype].forward(pred, target, reduction="mean")
    print("BCE with mean reduction:")
    loss_mean.print()

    # Test sum reduction
    var loss_sum = BCE[dtype].forward(pred, target, reduction="sum")
    print("BCE with sum reduction:")
    loss_sum.print()

    # sum should be approximately 4x mean
    print("Ratio (should be ~4):", loss_sum.item() / loss_mean.item())

fn main() raises:
    #test_bce_gradient()
    #test_bce_reduction_types()
    test_gradients()
    _="""alias dtype = DType.float32
    a = Tensor.arange(10, requires_grad=True)
    r = a.reshape(2, 5)
    l = r.log()
    c = l.clip(Scalar[dtype](0.4), 2)

    c.backward()
    a.grad().print()"""

fn test_gradients():
    alias dtype = DType.float64
    var x = Tensor[dtype].d1([0.5]).sigmoid[track_grad=False]()
    x.requires_grad_(True)
    var y = Tensor[dtype].d1([1.0], requires_grad=True)
    print("x is: ")
    x.print()
    _="""var x1 = Tensor[dtype].d1([0.5], requires_grad=True)
    var y1 = Tensor[dtype].d1([1.0], requires_grad=True)
    var x2 = Tensor[dtype].d1([0.5], requires_grad=True)
    var y2 = Tensor[dtype].d1([1.0], requires_grad=True)
    var x3 = Tensor[dtype].d1([0.5], requires_grad=True)
    var y3 = Tensor[dtype].d1([1.0], requires_grad=True)"""




    var loss = x.binary_cross_entropy(y)
    _="""var loss1 = y1.binary_cross_entropy(x1)
    var loss2 = BCE[dtype].forward(x2, y2, reduction="mean")
    var loss3 = BCE[dtype].forward(y3, x3, reduction="mean")"""
    print("Loss:", loss.item())
    _="""print("Loss: 1", loss1.item())
    print("Loss: 2", loss2.item())
    print("Loss: 3", loss3.item())"""

    loss.backward()
    _="""loss1.backward()
    loss2.backward()
    loss3.backward()"""

    if x.has_grad():
        print("Gradient x:", x.grad()[0])
    else:
        print("ERROR: x No gradient!")

    if y.has_grad():
        print("Gradient: y", y.grad()[0])
    else:
        print("ERROR: y No gradient!")

    _="""if x2.has_grad():
        print("Gradient: x2 ", x2.grad()[0])
    else:
        print("ERROR: x2 No gradient!")
    if y3.has_grad():
        print("Gradient: y3", y3.grad()[0])
    else:
        print("ERROR: y3 No gradient!")"""

