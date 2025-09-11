from tensors import Tensor
from shared import TensorLite
from intlist import IntList
from shapes import Shape
from common_utils import panic
from math import log, exp
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
        var gradients = output.gradients()[]
        var ancestor_1 = output.ancestry().get(0)[]
        var ancestor_2 = output.ancestry().get(1)[]
        var logits = ancestor_1.tensor()
        var target = ancestor_2.tensor()

        var input_shape = logits.shape

        # 1. Use the SAME reshaping logic as forward pass
        var N = logits.shape[0]
        var C = logits.shape[1]

        # Calculate total number of predictions (same as forward pass)
        var total_elements = target.shape[1:].product()

        var M = N * total_elements  # Total predictions

        # Reshape logits to (M, C) and target to (M,) (same as forward pass)
        var logits_2d = logits.reshape(Shape([M, C]))
        var target_1d = target.reshape(Shape([M]))

        # 2. Compute softmax probabilities on the 2D tensor
        var softmax_probs = logits_2d.softmax[False](
            axes=IntList(1), requires_grad=False
        )

        # 3. Create gradient tensor with same shape as softmax_probs
        var grad_input_2d = Tensor[dtype].zeros_like(softmax_probs)

        # 4. Precompute fixed values for label smoothing (same as forward pass)
        var smoothing_active = rebind[Scalar[dtype]](
            self.label_smoothing
        ) > Scalar[dtype](0)
        var smooth_prob = Scalar[dtype](0)
        var one_minus_smoothing = Scalar[dtype](0)

        if smoothing_active:
            smooth_prob = rebind[Scalar[dtype]](self.label_smoothing) / C
            one_minus_smoothing = 1 - rebind[Scalar[dtype]](
                self.label_smoothing
            )
        # 5. Compute gradients for each element
        var valid_count = 0

        for m in range(M):
            var class_idx = target_1d[m].__int__()

            if class_idx == self.ignore_index:
                continue  # Gradient remains zero

            valid_count += 1

            if smoothing_active:
                # Gradient formula for label smoothing: y_pred - y_true_smooth
                for j in range(C):
                    if j == class_idx:
                        # For correct class: y_true = 1 - smoothing
                        grad_input_2d[m, j] = (
                            softmax_probs[m, j] - one_minus_smoothing
                        )
                    else:
                        # For incorrect classes: y_true = smoothing / (C - 1)
                        grad_input_2d[m, j] = softmax_probs[m, j] - smooth_prob
            else:
                # Standard cross-entropy gradient: y_pred - y_true
                for j in range(C):
                    if j == class_idx:
                        grad_input_2d[m, j] = softmax_probs[m, j] - Scalar[
                            dtype
                        ](1)
                    else:
                        grad_input_2d[m, j] = softmax_probs[m, j]

        # 6. Apply reduction scaling (must match forward pass logic)
        if self.reduction == 0:  # "mean"
            # For mean reduction, scale by 1/valid_count
            if valid_count > 0:
                var scale_factor = Scalar[dtype](1) / Scalar[dtype](valid_count)
                grad_input_2d = grad_input_2d * scale_factor
            # If valid_count == 0, gradients remain zero

        elif self.reduction == 1:  # "sum"
            # No scaling needed for sum reduction
            pass
        # reduction="none" requires no additional scaling

        # 7. Reshape grad_input back to original logits shape
        var final_grad_input = grad_input_2d.reshape(input_shape)

        # 8. Multiply by upstream gradient (chain rule)
        # The upstream gradient is a scalar for mean/sum reduction, or matches target shape for "none"
        var scaled_gradients: Tensor[dtype]

        if self.reduction == 2:  # none
            # For "none" reduction, upstream gradient has shape (N, d1, d2, ...)
            # We need to reshape it to (M,) to match our 2D gradient computation
            var gradients_1d = gradients.reshape(Shape([M]))
            var expanded_gradients = Tensor[dtype](
                Shape([M, C]), requires_grad=False
            )
            for m in range(M):
                for c in range(C):
                    expanded_gradients[m, c] = gradients_1d[m]
            final_grad_input = final_grad_input * expanded_gradients
        else:
            # For mean/sum reduction, upstream gradient is a scalar
            final_grad_input = final_grad_input * gradients.item()

        return [(ancestor_1, final_grad_input, AddTensor)]


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
        Unified implementation that handles all shapes the same way.
        """

        # 1. Validate basic input shapes
        if logits.rank() < 2:
            panic("Logits must have at least 2 dimensions")
        if target.rank() != logits.rank() - 1:
            panic("Target must have one fewer dimension than logits")
        if logits.shape[0] != target.shape[0]:
            panic("Batch size mismatch between logits and target")

        # 2. Reshape to unified 2D format: (total_elements, C)
        var N = logits.shape[0]
        var C = logits.shape[1]

        var total_elements = target.shape[1:].product()
        var M = N * total_elements  # Total predictions

        # Reshape logits to (M, C) and target to (M,)
        var logits_2d = logits.reshape(Shape([M, C]))
        var target_1d = target.reshape(Shape([M]))

        # 3. Compute log_softmax on the 2D tensor
        var log_probs = self.log_softmax(logits_2d)

        # 4. Precompute fixed values for label smoothing (OUTSIDE the loop)
        var smoothing_active = self.label_smoothing > Scalar[dtype](0)
        var smooth_prob = Scalar[dtype](0)
        var one_minus_smoothing = Scalar[dtype](0)

        if smoothing_active:
            smooth_prob = self.label_smoothing / C
            one_minus_smoothing = 1 - self.label_smoothing

        # 5. Compute loss for each element
        var losses = Tensor[dtype].zeros(Shape([M]), requires_grad=False)
        var valid_count = 0

        for m in range(M):
            var class_idx = target_1d[m].__int__()

            if class_idx == self.ignore_index:
                continue  # Already zero from initialization

            valid_count += 1

            if smoothing_active:
                # CORRECT label smoothing implementation with precomputed values
                # Main term: (1 - smoothing) * -log(predicted_prob)
                var main_term = -log_probs[m, class_idx] * one_minus_smoothing

                # Additional term: smoothing * average_negative_log_prob for other classes
                var other_terms = Scalar[dtype](0)
                for c in range(C):
                    if c != class_idx:
                        other_terms += -log_probs[m, c]
                var smoothing_term = other_terms * smooth_prob

                losses[m] = main_term + smoothing_term
            else:
                # Standard cross entropy
                losses[m] = -log_probs[m, class_idx]

        # 6. Apply reduction
        var out: Tensor[dtype] = losses

        if self.reduction == 2:  # none
            # Reshape back to original target shape
            out = losses.reshape(target.shape)
        else:
            if self.reduction == 0:  # mean
                if valid_count > 0:
                    var total_loss = losses.sum()
                    out = Tensor.scalar(
                        total_loss.item() / Scalar[dtype](valid_count)
                    )
                else:
                    out = Tensor.scalar(Scalar[dtype](0))
            elif self.reduction == 1:  # sum
                var total_loss = losses.sum()
                out = Tensor.scalar(total_loss.item())
            else:
                panic("Invalid reduction type: " + self.reduction.__str__())

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
        """Numerically stable log(softmax(x)) for 2D input."""
        var M = logits.shape[0]
        var C = logits.shape[1]
        var result = Tensor[dtype](Shape([M, C]), requires_grad=False)

        for m in range(M):
            # Find max for stability
            var max_val = logits[m, 0]
            for c in range(1, C):
                if logits[m, c] > max_val:
                    max_val = logits[m, c]

            # Calculate log-sum-exp
            var sum_exp = Scalar[dtype](0)
            for c in range(C):
                sum_exp += exp(logits[m, c] - max_val)

            var log_sum_exp = log(sum_exp + Scalar[dtype](1e-12))

            # Compute log_softmax
            for c in range(C):
                result[m, c] = (logits[m, c] - max_val) - log_sum_exp

        return result

    @staticmethod
    fn validate_cross_entropy_inputs[
        dtype: DType
    ](logits: Tensor[dtype], target: Tensor[dtype], ignore_index: Int):
        """
        Validate CrossEntropyLoss inputs and return processed tensors.
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
                "Batch size mismatch: logits has "
                + N.__str__()
                + ", target has "
                + target_shape[0].__str__()
            )

        # 3. Validate class indices are within bounds
        var C = input_shape[1]  # number of classes
        Self.validate_class_indices(target, C, ignore_index)

        # 4. Validate spatial dimensions match (if any)
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

    @staticmethod
    fn validate_class_indices[
        dtype: DType
    ](target: Tensor[dtype], num_classes: Int, ignore_index: Int):
        """Validate that target indices are within valid range."""
        # Use your built-in shape iterator - this is the clean way!
        for idx in target.shape:
            var class_idx = target[idx].__int__()
            if class_idx != ignore_index and (
                class_idx < 0 or class_idx >= num_classes
            ):
                panic(
                    "Target index "
                    + class_idx.__str__()
                    + " is out of bounds for "
                    + num_classes.__str__()
                    + " classes. Valid range: [0, "
                    + (num_classes - 1).__str__()
                    + "] or ignore_index="
                    + ignore_index.__str__()
                )


fn main() raises:
    test_ce_2d_basic()
    test_ce_2d_ignore_index()
    test_ce_2d_label_smoothing()
    test_ce_2d_logits_1d_target()
    test_ce_3d_spatial()
    test_ce_4d_spatial()
    test_ce_4d_spatial_1x2()
    test_ce_reduction_mean()
    test_ce_reduction_sum()
    test_ce_reduction_none()
    print("passes")


from testing import assert_true
from time import perf_counter_ns


fn test_ce_2d_basic() raises:
    print("test_ce_2d_basic")
    var logits = Tensor.d2([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]]).float()
    var target = Tensor.d1([0, 1])  # Class indices
    var loss_fn = CrossEntropyLoss[DType.float32]()
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_2d_basic -> Total:            ", end - start)

    # Expected: -log(softmax([2,1,0.1])[0]) for first sample, etc.
    assert_true(loss.item() > 0.0)


fn test_ce_2d_ignore_index() raises:
    print("test_ce_2d_ignore_index")
    var logits = Tensor.d2([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]]).float()
    var target = Tensor.d1([0, -1])  # Second sample ignored
    var loss_fn = CrossEntropyLoss[DType.float32](ignore_index=-1)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_2d_ignore_index -> Total:            ", end - start)

    # Should only compute loss for first sample
    assert_true(loss.item() > 0.0)


fn test_ce_2d_label_smoothing() raises:
    print("test_ce_2d_label_smoothing")
    var logits = Tensor.d2([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](label_smoothing=0.1)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_2d_label_smoothing -> Total:            ", end - start)

    # Should be slightly higher than without smoothing
    assert_true(loss.item() > 0.0)


fn test_ce_2d_logits_1d_target() raises:
    print("test_ce_2d_logits_1d_target")
    var logits = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32]()
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_2d_logits_1d_target -> Total:            ", end - start)

    assert_true(loss.item() > 0.0)


fn test_ce_3d_spatial() raises:
    print("test_ce_3d_spatial")
    var logits = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    ).float()  # (2, 2, 2)
    var target = Tensor.d2([[0, 1], [1, 0]])  # (2, 2)

    var loss_fn = CrossEntropyLoss[DType.float32]()
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_3d_spatial -> Total:            ", end - start)
    assert_true(loss.item() > 0.0)


fn test_ce_4d_spatial() raises:
    print("test_ce_4d_spatial")
    # Logits: 2 samples, 2 classes, 2x2 spatial
    var logits = Tensor.d4(
        [
            [  # Sample 0
                [[1.0, 2.0], [3.0, 4.0]],  # Class 0 spatial outputs
                [[5.0, 6.0], [7.0, 8.0]],  # Class 1 spatial outputs
            ],
            [  # Sample 1
                [[9.0, 10.0], [11.0, 12.0]],  # Class 0
                [[13.0, 14.0], [15.0, 16.0]],  # Class 1
            ],
        ]
    ).float()  # Shape: (2, 2, 2, 2)

    # Target must have matching spatial dimensions: 2x2
    var target = Tensor.d3(
        [
            [[0, 1], [1, 0]],  # Sample 0: 2x2 spatial class assignments
            [[1, 0], [0, 1]],  # Sample 1: 2x2 spatial class assignments
        ]
    )  # Shape: (2, 2, 2) - matches logits spatial dims

    var loss_fn = CrossEntropyLoss[DType.float32]()
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_4d_spatial -> Total:            ", end - start)
    assert_true(loss.item() > 0.0)


fn test_ce_4d_spatial_1x2() raises:
    print("test_ce_4d_spatial_1x2")
    # Logits: 2 samples, 2 classes, 1x2 spatial (not 2x2)
    var logits = Tensor.d4(
        [
            [  # Sample 0
                [[1.0, 2.0]],  # Class 0: 1x2 spatial
                [[3.0, 4.0]],  # Class 1: 1x2 spatial
            ],
            [  # Sample 1
                [[5.0, 6.0]],  # Class 0: 1x2 spatial
                [[7.0, 8.0]],  # Class 1: 1x2 spatial
            ],
        ]
    ).float()  # Shape: (2, 2, 1, 2)

    # Target: 1x2 spatial dimensions to match logits
    var target = Tensor.d3(
        [[[0, 1]], [[1, 0]]]  # Sample 0: 1x2 spatial  # Sample 1: 1x2 spatial
    )  # Shape: (2, 1, 2) - matches logits spatial dims

    var loss_fn = CrossEntropyLoss[DType.float32]()
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_4d_spatial_1x2 -> Total:            ", end - start)
    assert_true(loss.item() > 0.0)


fn test_ce_reduction_mean() raises:
    print("test_ce_reduction_mean")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)  # mean
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_reduction_mean -> Total:            ", end - start)
    assert_true(loss.shape.rank() == 0)  # scalar


fn test_ce_reduction_sum() raises:
    print("test_ce_reduction_sum")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=1)  # sum
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_reduction_sum -> Total:            ", end - start)

    assert_true(loss.shape.rank() == 0)  # scalar


fn test_ce_reduction_none() raises:
    print("test_ce_reduction_none")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=2)  # none
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_reduction_none -> Total:            ", end - start)

    assert_true(loss.shape == Shape([2]))  # per-sample loss
