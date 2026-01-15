from tenmo import Tensor
from shapes import Shape
from common_utils import panic
from math import log, exp
from subtraction import Subtractor
from backpropagation import Delegate, BackwardFn, BACKWARD_CROSS_ENTROPY
from operators import AddTensor
from buffers import Buffer
from gradbox import Gradbox
from ndbuffer import NDBuffer


@register_passable
struct Reduction(Copyable, EqualityComparable, ImplicitlyCopyable):
    var reduction: Int
    alias Mean = Reduction(0)
    alias Sum = Reduction(1)
    alias `None` = Reduction(2)

    fn __init__(out self, reduction: Int = 0):
        self.reduction = reduction
        if reduction < 0 or reduction > 2:
            panic(
                "Invalid reduction type. Must '0 → mean', '1 → sum', or '2 →"
                " none'"
            )

    fn __init__(out self, reduction: String):
        if reduction == "mean":
            self.reduction = 0
        elif reduction == "sum":
            self.reduction = 1
        elif reduction == "none":
            self.reduction = 2
        else:
            self.reduction = -1
            panic("Invalid reduction type. Must 'mean', 'sum', or 'none'")

    fn __copyinit__(out self, existing: Self):
        self.reduction = existing.reduction

    fn __eq__(self, other: Self) -> Bool:
        return self.reduction == other.reduction

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn value(self) -> Int:
        return self.reduction


@register_passable
struct CrossEntropyLoss[dtype: DType = DType.float32](Copyable):
    var reduction: Reduction
    var ignore_index: Int
    var label_smoothing: Scalar[Self.dtype]
    var training: Bool

    fn __init__(
        out self,
        reduction: Int = 0,
        ignore_index: Int = -100,
        label_smoothing: Scalar[Self.dtype] = Scalar[dtype](0),
        training: Bool = True,
    ):
        self.reduction = Reduction(reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.training = training

    fn __init__(
        out self,
        reduction: String,
        ignore_index: Int = -100,
        label_smoothing: Scalar[Self.dtype] = Scalar[dtype](0),
        training: Bool = True,
    ):
        self.reduction = Reduction(reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.training = training

    fn __copyinit__(out self, existing: Self):
        self.reduction = existing.reduction
        self.ignore_index = existing.ignore_index
        self.label_smoothing = existing.label_smoothing
        self.training = existing.training

    fn train(mut self):
        """Set to training mode - enables gradient tracking."""
        self.training = True

    fn eval(mut self):
        """Set to evaluation mode - disables gradient tracking."""
        self.training = False

    fn __call__(
        self,
        logits: Tensor[Self.dtype],
        target: Tensor[DType.int32],
        validate: Bool = True,
    ) -> Tensor[Self.dtype]:
        if self.training:
            return self._forward_class_indices[track_grad=True](
                logits, target, validate
            )
        else:
            return self._forward_class_indices[track_grad=False](
                logits, target, validate
            )

    fn __call__(
        self,
        logits: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        validate: Bool = True,
    ) -> Tensor[Self.dtype]:
        if self.training:
            return self._forward_probabilities[track_grad=True](
                logits, target, validate
            )
        else:
            return self._forward_probabilities[track_grad=False](
                logits, target, validate
            )

    fn _forward_class_indices[
        track_grad: Bool
    ](
        self,
        var logits: Tensor[Self.dtype],
        var target: Tensor[DType.int32],
        validate: Bool,
    ) -> Tensor[Self.dtype]:
        """Optimized forward pass for class indices targets."""
        # Validate inputs

        if validate:
            Self._validate_class_indices_inputs(
                logits, target, self.ignore_index
            )

        # Reshape to unified 2D format
        var logits_shape = logits.shape()
        var target_shape = target.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var spatial_dims = logits_shape[2:]
        var total_spatial = (
            spatial_dims.product() if spatial_dims.rank() > 0 else 1
        )
        var M = N * total_spatial

        var logits_2d = logits.reshape[track_grad=False](Shape([M, C]))
        var target_1d = target.reshape[track_grad=False](Shape([M]))

        # Precompute smoothing values
        var smoothing_active = self.label_smoothing > Scalar[Self.dtype](0)
        var true_smoothed = Scalar[Self.dtype](1)
        var non_true_smoothed = Scalar[Self.dtype](0)

        if smoothing_active:
            var uniform_val = Scalar[Self.dtype](1) / Scalar[dtype](C)
            non_true_smoothed = self.label_smoothing * uniform_val
            true_smoothed = (
                Scalar[Self.dtype](1) - self.label_smoothing + non_true_smoothed
            )

        # Fused log_softmax + loss computation
        var losses = Tensor[Self.dtype].zeros(Shape([M]), requires_grad=False)
        var valid_count = 0

        for m in range(M):
            var class_idx = target_1d[m].__int__()

            if class_idx == self.ignore_index:
                continue

            valid_count += 1

            # Compute log_softmax for this row (numerically stable)
            var max_val = logits_2d[m, 0]
            for c in range(1, C):
                if logits_2d[m, c] > max_val:
                    max_val = logits_2d[m, c]

            var sum_exp = Scalar[Self.dtype](0)
            for c in range(C):
                sum_exp += exp(logits_2d[m, c] - max_val)

            var log_sum_exp = log(sum_exp) + max_val

            # Compute loss directly
            if smoothing_active:
                var loss_val = Scalar[Self.dtype](0)
                for c in range(C):
                    var log_prob = logits_2d[m, c] - log_sum_exp
                    var true_prob = (
                        true_smoothed if c == class_idx else non_true_smoothed
                    )
                    loss_val += true_prob * -log_prob
                losses[m] = loss_val
            else:
                losses[m] = -(logits_2d[m, class_idx] - log_sum_exp)

        # Apply reduction
        var out = self._apply_reduction(losses, target_shape, valid_count)

        # Setup autograd
        @parameter
        if track_grad:
            if logits.requires_grad:
                out.requires_grad_(True)
                var backward_fn = CrossEntropyBackward[Self.dtype](
                    self.reduction.value(),
                    self.ignore_index,
                    self.label_smoothing,
                    Optional((target_shape, target.buffer.contiguous_buffer())),
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(logits)

        return out^

    fn _forward_probabilities[
        track_grad: Bool
    ](
        self,
        var logits: Tensor[Self.dtype],
        var target: Tensor[dtype],
        validate: Bool,
    ) -> Tensor[Self.dtype]:
        """Optimized forward pass for probability/one-hot targets."""
        # Validate inputs

        if validate:
            Self._validate_probability_inputs(logits, target)

        # Reshape to unified 2D format
        var logits_shape = logits.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var spatial_dims = logits_shape[2:]
        var total_spatial = (
            spatial_dims.product() if spatial_dims.rank() > 0 else 1
        )
        var M = N * total_spatial

        var logits_2d = logits.reshape[track_grad=False](Shape([M, C]))
        var target_2d = target.reshape[track_grad=False](Shape([M, C]))

        # Precompute smoothing
        var smoothing_active = self.label_smoothing > Scalar[Self.dtype](0)
        var uniform_val = Scalar[Self.dtype](0)
        if smoothing_active:
            uniform_val = Scalar[Self.dtype](1) / Scalar[dtype](C)

        # Fused log_softmax + loss computation
        var losses = Tensor[Self.dtype].zeros(Shape([M]), requires_grad=False)

        for m in range(M):
            # Compute log_softmax for this row
            var max_val = logits_2d[m, 0]
            for c in range(1, C):
                if logits_2d[m, c] > max_val:
                    max_val = logits_2d[m, c]

            var sum_exp = Scalar[Self.dtype](0)
            for c in range(C):
                sum_exp += exp(logits_2d[m, c] - max_val)

            var log_sum_exp = log(sum_exp) + max_val

            # Compute loss with optional smoothing
            var loss_val = Scalar[Self.dtype](0)
            for c in range(C):
                var log_prob = logits_2d[m, c] - log_sum_exp
                var target_prob = target_2d[m, c]

                if smoothing_active:
                    target_prob = (
                        Scalar[Self.dtype](1) - self.label_smoothing
                    ) * target_prob + self.label_smoothing * uniform_val

                loss_val += target_prob * -log_prob

            losses[m] = loss_val

        # Apply reduction
        var valid_count = M
        var out = self._apply_reduction(
            losses, target.shape()[0:-1], valid_count
        )

        # Setup autograd
        @parameter
        if track_grad:
            if logits.requires_grad:
                out.requires_grad_(True)
                var backward_fn = CrossEntropyBackward[Self.dtype](
                    self.reduction.value(),
                    self.ignore_index,
                    self.label_smoothing,
                    None,
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(logits, target)

        return out^

    fn _apply_reduction(
        self,
        var losses: Tensor[Self.dtype],
        original_target_shape: Shape,
        valid_count: Int,
    ) -> Tensor[Self.dtype]:
        """Apply reduction to per-sample losses."""
        if self.reduction == Reduction.`None`:
            return losses.reshape[track_grad=False](original_target_shape)
        elif self.reduction == Reduction.Sum:
            return losses.sum()
        else:  # Mean reduction
            if valid_count > 0:
                var mean_loss = losses.sum().item() / Scalar[Self.dtype](
                    valid_count
                )
                return Tensor.scalar(mean_loss)
            else:
                return Tensor.scalar(Scalar[Self.dtype](0))

    @staticmethod
    fn _validate_class_indices_inputs(
        logits: Tensor[Self.dtype],
        target: Tensor[DType.int32],
        ignore_index: Int,
    ):
        """Validate inputs for class indices targets."""
        var input_shape = logits.shape()
        var target_shape = target.shape()

        if input_shape.rank() < 2:
            panic("Input must have at least 2 dimensions")

        if target_shape.rank() != input_shape.rank() - 1:
            panic("Target must have one fewer dimension than input")

        var N = input_shape[0]
        if target_shape[0] != N:
            panic("Batch size mismatch between logits and target")

        var C = input_shape[1]

        if target_shape.rank() > 1:
            for i in range(1, target_shape.rank()):
                if i < input_shape.rank() - 1:
                    if target_shape[i] != input_shape[i + 1]:
                        panic(
                            "Spatial dimension mismatch at dimension "
                            + i.__str__()
                        )

        Self._validate_target_indices(target, C, ignore_index)

    @staticmethod
    fn _validate_probability_inputs(
        logits: Tensor[Self.dtype], target: Tensor[dtype]
    ):
        """Validate inputs for probability targets."""
        var input_shape = logits.shape()
        var target_shape = target.shape()

        if input_shape.rank() < 2:
            panic("Input must have at least 2 dimensions")

        if target_shape.rank() != input_shape.rank():
            panic("Target must have same rank as input for probability targets")

        var N = input_shape[0]
        var C = input_shape[1]

        if target_shape[0] != N or target_shape[1] != C:
            panic("Batch size or class count mismatch")

        if input_shape.rank() > 2:
            for i in range(2, input_shape.rank()):
                if target_shape[i] != input_shape[i]:
                    panic(
                        "Spatial dimension mismatch at dimension " + i.__str__()
                    )

    @staticmethod
    fn _validate_target_indices(
        target: Tensor[DType.int32], num_classes: Int, ignore_index: Int
    ):
        """Validate that all target indices are within valid range."""
        for coord, value in target:
            var class_idx = value.__int__()
            if class_idx != ignore_index and (
                class_idx < 0 or class_idx >= num_classes
            ):
                var pos_str = "["
                for i in range(coord.size()):
                    if i > 0:
                        pos_str += ", "
                    pos_str += coord[i].__str__()
                pos_str += "]"

                panic(
                    "Target index "
                    + class_idx.__str__()
                    + " at position "
                    + pos_str
                    + " is out of bounds for "
                    + num_classes.__str__()
                    + " classes. Valid range: [0, "
                    + (num_classes - 1).__str__()
                    + "] or ignore_index="
                    + ignore_index.__str__()
                )


struct CrossEntropyBackward[dtype: DType](
    Copyable & Movable & ImplicitlyCopyable
):
    alias TAG = BACKWARD_CROSS_ENTROPY
    var reduction: Reduction
    var ignore_index: Int
    var label_smoothing: Scalar[Self.dtype]
    var original_target: Optional[Tuple[Shape, Buffer[DType.int32]]]

    fn __init__(
        out self,
        reduction: Int = 0,
        ignore_index: Int = -100,
        label_smoothing: Scalar[Self.dtype] = Scalar[Self.dtype](0),
        original_target: Optional[Tuple[Shape, Buffer[DType.int32]]] = None,
    ):
        self.reduction = Reduction(reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.original_target = original_target

    fn __copyinit__(out self, existing: Self):
        self.reduction = existing.reduction.copy()
        self.ignore_index = existing.ignore_index
        self.label_smoothing = existing.label_smoothing
        self.original_target = existing.original_target.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.reduction = existing.reduction
        self.ignore_index = existing.ignore_index
        self.label_smoothing = existing.label_smoothing
        self.original_target = existing.original_target^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[dtype], Int]]:
        var gradients = output.grad()
        var logits = output.ancestry().get(0)

        if self.original_target:
            var shape_buffer_pair = self.original_target.value().copy()
            var shape = shape_buffer_pair[0].copy()
            var buffer = shape_buffer_pair[1].copy()
            var ndb = NDBuffer[DType.int32](buffer^, shape^)
            target = Tensor[DType.int32](ndb^, requires_grad=False)
            return self._backward_class_indices(logits^, target^, gradients^)
        else:
            var target = output.ancestry().get(1)
            return self._backward_probabilities(logits^, target^, gradients^)

    fn _backward_class_indices(
        self,
        var logits: Tensor[Self.dtype],
        var target: Tensor[DType.int32],
        upstream_grad: Gradbox[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[dtype], Int]]:
        """Optimized backward pass for class indices targets."""
        var logits_shape = logits.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var spatial_dims = logits_shape[2:]
        var total_spatial = (
            spatial_dims.product() if spatial_dims.rank() > 0 else 1
        )
        var M = N * total_spatial

        var logits_2d = logits.reshape[track_grad=False](Shape([M, C]))
        var target_1d = target.reshape[track_grad=False](Shape([M]))

        var grad_input_2d = Gradbox[Self.dtype].zeros(
            Shape([M, C]), share=False
        )
        var valid_count = 0

        # Precompute smoothing values
        var smoothing_active = self.label_smoothing > Scalar[Self.dtype](0)
        var true_smoothed = Scalar[Self.dtype](1)
        var non_true_smoothed = Scalar[Self.dtype](0)

        if smoothing_active:
            var uniform_val = Scalar[Self.dtype](1) / Scalar[dtype](C)
            non_true_smoothed = self.label_smoothing * uniform_val
            true_smoothed = (
                Scalar[Self.dtype](1) - self.label_smoothing + non_true_smoothed
            )

        # Fused softmax + gradient computation
        for m in range(M):
            var class_idx = target_1d[m].__int__()

            if class_idx == self.ignore_index:
                continue

            valid_count += 1

            # Compute softmax for this row (numerically stable)
            var max_val = logits_2d[m, 0]
            for c in range(1, C):
                if logits_2d[m, c] > max_val:
                    max_val = logits_2d[m, c]

            var sum_exp = Scalar[Self.dtype](0)
            for c in range(C):
                sum_exp += exp(logits_2d[m, c] - max_val)

            # Compute gradient: softmax - target
            for c in range(C):
                var softmax_prob = exp(logits_2d[m, c] - max_val) / sum_exp

                if smoothing_active:
                    var target_prob = (
                        true_smoothed if c == class_idx else non_true_smoothed
                    )
                    grad_input_2d[m, c] = softmax_prob - target_prob
                else:
                    var target_val = Scalar[Self.dtype](
                        1
                    ) if c == class_idx else Scalar[Self.dtype](0)
                    grad_input_2d[m, c] = softmax_prob - target_val

        # Apply reduction scaling
        grad_input_2d = self._apply_reduction_scaling(
            grad_input_2d, upstream_grad, valid_count, M, target.shape()
        )

        var final_grad_input = grad_input_2d.reshape(logits_shape)

        return [(logits, final_grad_input^, AddTensor)]

    fn _backward_probabilities(
        self,
        var logits: Tensor[Self.dtype],
        var target: Tensor[Self.dtype],
        upstream_grad: Gradbox[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[dtype], Int]]:
        """Optimized backward pass for probability/one-hot targets."""
        var logits_shape = logits.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var spatial_dims = logits_shape[2:]
        var total_spatial = (
            spatial_dims.product() if spatial_dims.rank() > 0 else 1
        )
        var M = N * total_spatial

        var logits_2d = logits.reshape[track_grad=False](Shape([M, C]))
        var target_2d = target.reshape[track_grad=False](Shape([M, C]))

        var grad_input_2d = Gradbox[Self.dtype].zeros(
            Shape([M, C]), share=False
        )

        # Precompute smoothing
        var smoothing_active = self.label_smoothing > Scalar[Self.dtype](0)
        var uniform_val = Scalar[Self.dtype](0)
        if smoothing_active:
            uniform_val = Scalar[Self.dtype](1) / Scalar[dtype](C)

        # Fused softmax + gradient computation
        for m in range(M):
            # Compute softmax for this row
            var max_val = logits_2d[m, 0]
            for c in range(1, C):
                if logits_2d[m, c] > max_val:
                    max_val = logits_2d[m, c]

            var sum_exp = Scalar[Self.dtype](0)
            for c in range(C):
                sum_exp += exp(logits_2d[m, c] - max_val)

            # Compute gradient
            for c in range(C):
                var softmax_prob = exp(logits_2d[m, c] - max_val) / sum_exp
                var target_prob = target_2d[m, c]

                if smoothing_active:
                    target_prob = (
                        Scalar[Self.dtype](1) - self.label_smoothing
                    ) * target_prob + self.label_smoothing * uniform_val

                grad_input_2d[m, c] = softmax_prob - target_prob

        # Apply reduction scaling
        var valid_count = M
        grad_input_2d = self._apply_reduction_scaling(
            grad_input_2d, upstream_grad, valid_count, M, target.shape()[0:-1]
        )

        var final_grad_input = grad_input_2d.reshape(logits_shape)

        return [(logits, final_grad_input^, AddTensor)]

    fn _apply_reduction_scaling(
        self,
        grad_input_2d: Gradbox[Self.dtype],
        upstream_grad: Gradbox[Self.dtype],
        valid_count: Int,
        total_elements: Int,
        target_shape: Shape,
    ) -> Gradbox[Self.dtype]:
        """Apply reduction-specific scaling to gradients."""
        var grad_input_2d_shape = grad_input_2d.shape()
        var scaled_grad = grad_input_2d.copy()

        # Apply reduction scaling first
        if self.reduction == Reduction.Mean and valid_count > 0:
            var scale_factor = Scalar[Self.dtype](1) / Scalar[dtype](
                valid_count
            )
            scaled_grad = grad_input_2d * scale_factor

        # Then multiply by upstream gradient (chain rule)
        if self.reduction == Reduction.`None`:
            var upstream_1d = upstream_grad.reshape(Shape([total_elements]))
            var expanded_upstream = Gradbox[Self.dtype](
                Shape([total_elements, grad_input_2d_shape[1]]),
            )

            for m in range(total_elements):
                for c in range(grad_input_2d_shape[1]):
                    expanded_upstream[m, c] = upstream_1d[m]

            scaled_grad = scaled_grad * expanded_upstream
        else:
            var upstream_scalar = upstream_grad.item()
            scaled_grad = scaled_grad * upstream_scalar

        return scaled_grad^
