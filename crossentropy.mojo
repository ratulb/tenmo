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
struct CrossEntropyLoss[dtype: DType = DType.float32, track_grad: Bool = True](
    Copyable
):
    var reduction: Reduction
    var ignore_index: Int
    var label_smoothing: Scalar[dtype]

    fn __init__(
        out self,
        reduction: Int = 0,
        ignore_index: Int = -100,
        label_smoothing: Scalar[dtype] = Scalar[dtype](0),
    ):
        self.reduction = Reduction(reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    fn __init__(
        out self,
        reduction: String,
        ignore_index: Int = -100,
        label_smoothing: Scalar[dtype] = Scalar[dtype](0),
    ):
        self.reduction = Reduction(reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    fn __copyinit__(out self, existing: Self):
        self.reduction = existing.reduction
        self.ignore_index = existing.ignore_index
        self.label_smoothing = existing.label_smoothing

    fn __call__(
        self,
        logits: Tensor[dtype],
        target: Tensor[DType.int32],
        validate: Bool = True,
    ) -> Tensor[dtype]:
        return self._forward_class_indices(logits, target, validate)

    fn __call__(
        self,
        logits: Tensor[dtype],
        target: Tensor[dtype],
        validate: Bool = True,
    ) -> Tensor[dtype]:
        return self._forward_probabilities(logits, target, validate)

    fn _forward_class_indices(
        self, logits: Tensor[dtype], target: Tensor[DType.int32], validate: Bool
    ) -> Tensor[dtype]:
        """Optimized forward pass for class indices targets."""
        # 1. Validate inputs

        if validate:
            Self._validate_class_indices_inputs(
                logits, target, self.ignore_index
            )

        # 2. Reshape to unified 2D format
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

        # 3. Precompute smoothing values
        var smoothing_active = self.label_smoothing > Scalar[dtype](0)
        var true_smoothed = Scalar[dtype](1)
        var non_true_smoothed = Scalar[dtype](0)

        if smoothing_active:
            var uniform_val = Scalar[dtype](1) / Scalar[dtype](C)
            non_true_smoothed = self.label_smoothing * uniform_val
            true_smoothed = (
                Scalar[dtype](1) - self.label_smoothing + non_true_smoothed
            )

        # 4. Fused log_softmax + loss computation
        var losses = Tensor[dtype].zeros(Shape([M]), requires_grad=False)
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

            var sum_exp = Scalar[dtype](0)
            for c in range(C):
                sum_exp += exp(logits_2d[m, c] - max_val)

            var log_sum_exp = log(sum_exp) + max_val

            # Compute loss directly
            if smoothing_active:
                var loss_val = Scalar[dtype](0)
                for c in range(C):
                    var log_prob = logits_2d[m, c] - log_sum_exp
                    var true_prob = (
                        true_smoothed if c == class_idx else non_true_smoothed
                    )
                    loss_val += true_prob * -log_prob
                losses[m] = loss_val
            else:
                losses[m] = -(logits_2d[m, class_idx] - log_sum_exp)

        # 5. Apply reduction
        var out = self._apply_reduction(losses, target_shape, valid_count)

        # Setup autograd if needed
        @parameter
        if track_grad:
            if logits.requires_grad:
                out.requires_grad_(True)
                var backward_fn = CrossEntropyBackward[dtype](
                    self.reduction.value(),
                    self.ignore_index,
                    self.label_smoothing,
                    Optional((target_shape, target.buffer.contiguous_buffer())),
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(logits)

        return out^

    fn _forward_probabilities(
        self, logits: Tensor[dtype], target: Tensor[dtype], validate: Bool
    ) -> Tensor[dtype]:
        """Optimized forward pass for probability/one-hot targets."""
        # 1. Validate inputs

        if validate:
            Self._validate_probability_inputs(logits, target)

        # 2. Reshape to unified 2D format
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

        # 3. Precompute smoothing
        var smoothing_active = self.label_smoothing > Scalar[dtype](0)
        var uniform_val = Scalar[dtype](0)
        if smoothing_active:
            uniform_val = Scalar[dtype](1) / Scalar[dtype](C)

        # 4. Fused log_softmax + loss computation
        var losses = Tensor[dtype].zeros(Shape([M]), requires_grad=False)

        for m in range(M):
            # Compute log_softmax for this row
            var max_val = logits_2d[m, 0]
            for c in range(1, C):
                if logits_2d[m, c] > max_val:
                    max_val = logits_2d[m, c]

            var sum_exp = Scalar[dtype](0)
            for c in range(C):
                sum_exp += exp(logits_2d[m, c] - max_val)

            var log_sum_exp = log(sum_exp) + max_val

            # Compute loss with optional smoothing
            var loss_val = Scalar[dtype](0)
            for c in range(C):
                var log_prob = logits_2d[m, c] - log_sum_exp
                var target_prob = target_2d[m, c]

                if smoothing_active:
                    target_prob = (
                        Scalar[dtype](1) - self.label_smoothing
                    ) * target_prob + self.label_smoothing * uniform_val

                loss_val += target_prob * -log_prob

            losses[m] = loss_val

        # 5. Apply reduction
        var valid_count = M
        var out = self._apply_reduction(
            losses, target.shape()[0:-1], valid_count
        )

        # Setup autograd if needed
        @parameter
        if track_grad:
            if logits.requires_grad:
                out.requires_grad_(True)
                var backward_fn = CrossEntropyBackward[dtype](
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
        losses: Tensor[dtype],
        original_target_shape: Shape,
        valid_count: Int,
    ) -> Tensor[dtype]:
        """Apply reduction to per-sample losses."""
        if self.reduction == Reduction.`None`:
            return losses.reshape[track_grad=False](original_target_shape)
        elif self.reduction == Reduction.Sum:
            return losses.sum()
        else:  # Mean reduction
            if valid_count > 0:
                var mean_loss = losses.sum().item() / Scalar[dtype](valid_count)
                return Tensor.scalar(mean_loss)
            else:
                return Tensor.scalar(Scalar[dtype](0))

    @staticmethod
    fn _validate_class_indices_inputs(
        logits: Tensor[dtype], target: Tensor[DType.int32], ignore_index: Int
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
        logits: Tensor[dtype], target: Tensor[dtype]
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
    var label_smoothing: Scalar[dtype]
    var original_target: Optional[Tuple[Shape, Buffer[DType.int32]]]

    fn __init__(
        out self,
        reduction: Int = 0,
        ignore_index: Int = -100,
        label_smoothing: Scalar[dtype] = Scalar[dtype](0),
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

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
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
        var logits: Tensor[dtype],
        var target: Tensor[DType.int32],
        upstream_grad: Gradbox[dtype],
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
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

        var grad_input_2d = Gradbox[dtype].zeros(Shape([M, C]), share=False)
        var valid_count = 0

        # Precompute smoothing values
        var smoothing_active = self.label_smoothing > Scalar[dtype](0)
        var true_smoothed = Scalar[dtype](1)
        var non_true_smoothed = Scalar[dtype](0)

        if smoothing_active:
            var uniform_val = Scalar[dtype](1) / Scalar[dtype](C)
            non_true_smoothed = self.label_smoothing * uniform_val
            true_smoothed = (
                Scalar[dtype](1) - self.label_smoothing + non_true_smoothed
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

            var sum_exp = Scalar[dtype](0)
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
                    var target_val = Scalar[dtype](
                        1
                    ) if c == class_idx else Scalar[dtype](0)
                    grad_input_2d[m, c] = softmax_prob - target_val

        # Apply reduction scaling
        grad_input_2d = self._apply_reduction_scaling(
            grad_input_2d, upstream_grad, valid_count, M, target.shape()
        )

        var final_grad_input = grad_input_2d.reshape(logits_shape)

        return [(logits, final_grad_input^, AddTensor)]

    fn _backward_probabilities(
        self,
        logits: Tensor[dtype],
        target: Tensor[dtype],
        upstream_grad: Gradbox[dtype],
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
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

        var grad_input_2d = Gradbox[dtype].zeros(Shape([M, C]), share=False)

        # Precompute smoothing
        var smoothing_active = self.label_smoothing > Scalar[dtype](0)
        var uniform_val = Scalar[dtype](0)
        if smoothing_active:
            uniform_val = Scalar[dtype](1) / Scalar[dtype](C)

        # Fused softmax + gradient computation
        for m in range(M):
            # Compute softmax for this row
            var max_val = logits_2d[m, 0]
            for c in range(1, C):
                if logits_2d[m, c] > max_val:
                    max_val = logits_2d[m, c]

            var sum_exp = Scalar[dtype](0)
            for c in range(C):
                sum_exp += exp(logits_2d[m, c] - max_val)

            # Compute gradient
            for c in range(C):
                var softmax_prob = exp(logits_2d[m, c] - max_val) / sum_exp
                var target_prob = target_2d[m, c]

                if smoothing_active:
                    target_prob = (
                        Scalar[dtype](1) - self.label_smoothing
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
        grad_input_2d: Gradbox[dtype],
        upstream_grad: Gradbox[dtype],
        valid_count: Int,
        total_elements: Int,
        target_shape: Shape,
    ) -> Gradbox[dtype]:
        """Apply reduction-specific scaling to gradients."""
        var grad_input_2d_shape = grad_input_2d.shape()
        var scaled_grad = grad_input_2d.copy()

        # Apply reduction scaling first
        if self.reduction == Reduction.Mean and valid_count > 0:
            var scale_factor = Scalar[dtype](1) / Scalar[dtype](valid_count)
            scaled_grad = grad_input_2d * scale_factor

        # Then multiply by upstream gradient (chain rule)
        if self.reduction == Reduction.`None`:
            var upstream_1d = upstream_grad.reshape(Shape([total_elements]))
            var expanded_upstream = Gradbox[dtype](
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


# =============================


@register_passable
struct Reduction_orig(Copyable, EqualityComparable, ImplicitlyCopyable):
    var reduction: Int
    alias Mean = Reduction(0)
    alias Sum = Reduction(1)
    alias `None` = Reduction(2)  # None is a keyword in mojo!

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
struct CrossEntropyLoss_orig[
    dtype: DType = DType.float32, track_grad: Bool = True
](Copyable):
    var reduction: Reduction
    var ignore_index: Int  # index to ignore (-100 for none)
    var label_smoothing: Scalar[dtype]  # usually 0.0

    fn __init__(
        out self,
        reduction: Int = 0,  # '0-> mean', '1-> sum', '2 -> none'
        ignore_index: Int = -100,
        label_smoothing: Scalar[dtype] = Scalar[dtype](0),
    ):
        self.reduction = Reduction(reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    fn __init__(
        out self,
        reduction: String,  # 'mean', 'sum', 'none'
        ignore_index: Int = -100,
        label_smoothing: Scalar[dtype] = Scalar[dtype](0),
    ):
        self.reduction = Reduction(reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    fn __copyinit__(out self, existing: Self):
        self.reduction = existing.reduction
        self.ignore_index = existing.ignore_index
        self.label_smoothing = existing.label_smoothing

    # For class indices targets (DType.int32)
    fn __call__(
        self, logits: Tensor[dtype], target: Tensor[DType.int32]
    ) -> Tensor[dtype]:
        return self._forward_class_indices(logits, target)

    # For probability/one-hot targets (same dtype as logits)
    fn __call__(
        self, logits: Tensor[dtype], target: Tensor[dtype]
    ) -> Tensor[dtype]:
        return self._forward_probabilities(logits, target)

    fn _forward_class_indices(
        self, logits: Tensor[dtype], target: Tensor[DType.int32]
    ) -> Tensor[dtype]:
        """
        Forward pass for class indices targets.
        """
        # 1. Validate inputs
        Self._validate_class_indices_inputs(logits, target, self.ignore_index)

        # 2. Reshape to unified 2D format: (total_elements, C)
        logits_shape = logits.shape()
        target_shape = target.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var spatial_dims = logits_shape[2:]
        var total_spatial = (
            spatial_dims.product() if spatial_dims.rank() > 0 else 1
        )
        var M = N * total_spatial  # Total predictions

        # Reshape logits to (M, C) and target to (M,)
        var logits_2d = logits.reshape[track_grad=False](Shape([M, C]))
        var target_1d = target.reshape[track_grad=False](Shape([M]))

        # 3. Compute log_softmax
        var log_probs = self._log_softmax(logits_2d)

        # 4. Compute loss for each element
        var losses = Tensor[dtype].zeros(Shape([M]), requires_grad=False)
        var valid_count = 0

        # Precompute smoothing values
        var smoothing_active = self.label_smoothing > Scalar[dtype](0)
        var non_true_smoothed = Scalar[dtype](0)

        if smoothing_active:
            # PyTorch's label smoothing formula: uniform distribution over ALL classes
            var uniform_val = Scalar[dtype](1) / Scalar[dtype](C)
            non_true_smoothed = self.label_smoothing * uniform_val
            true_smoothed = (
                Scalar[dtype](1) - self.label_smoothing + non_true_smoothed
            )
        else:
            true_smoothed = Scalar[dtype](1)

        for m in range(M):
            var class_idx = target_1d[m].__int__()

            if class_idx == self.ignore_index:
                continue  # Skip ignored indices

            valid_count += 1

            if smoothing_active:
                # PyTorch's label smoothing: loss = -sum(true_dist * log_softmax)
                var loss_val = Scalar[dtype](0)
                for c in range(C):
                    var true_prob = non_true_smoothed
                    if c == class_idx:
                        true_prob = true_smoothed
                    loss_val += true_prob * -log_probs[m, c]
                losses[m] = loss_val
            else:
                # Standard cross entropy: -log(p_true_class)
                losses[m] = -log_probs[m, class_idx]

        # 5. Apply reduction
        var out = self._apply_reduction(losses, target_shape, valid_count)

        # Setup autograd if needed
        @parameter
        if track_grad:
            if logits.requires_grad:
                out.requires_grad_(True)
                var backward_fn = CrossEntropyBackward[dtype](
                    self.reduction.value(),
                    self.ignore_index,
                    self.label_smoothing,
                    # Target -> Buffer[DType.int32] as part of backward fn
                    Optional((target_shape, target.buffer.contiguous_buffer())),
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(logits)

        return out^

    fn _forward_probabilities(
        self, logits: Tensor[dtype], target: Tensor[dtype]
    ) -> Tensor[dtype]:
        """
        Forward pass for probability/one-hot targets.
        """
        # 1. Validate inputs
        Self._validate_probability_inputs(logits, target)

        # 2. Reshape to unified 2D format
        var logits_shape = logits.shape()

        var N = logits_shape[0]
        var C = logits_shape[1]
        var spatial_dims = logits_shape[2:]
        var total_spatial = (
            spatial_dims.product() if spatial_dims.rank() > 0 else 1
        )
        var M = N * total_spatial

        # Reshape both to (M, C)
        var logits_2d = logits.reshape[track_grad=False](Shape([M, C]))
        var target_2d = target.reshape[track_grad=False](Shape([M, C]))

        # 3. Apply label smoothing to probability targets
        var smoothed_target = target_2d

        if self.label_smoothing > Scalar[dtype](0):
            smoothed_target = self._smooth_probability_targets(target_2d, C)

        # 4. Compute log_softmax
        var log_probs = self._log_softmax(logits_2d)

        # 5. Compute loss: -sum(target * log_probs) for each sample
        var losses = Tensor[dtype].zeros(Shape([M]), requires_grad=False)
        var valid_count = M  # All elements are valid for probability targets

        for m in range(M):
            var loss_val = Scalar[dtype](0)
            for c in range(C):
                loss_val += smoothed_target[m, c] * -log_probs[m, c]
            losses[m] = loss_val

        # 6. Apply reduction (ignore_index doesn't apply to probability targets)
        var out = self._apply_reduction(
            losses, target.shape()[0:-1], valid_count
        )

        # Setup autograd if needed
        @parameter
        if track_grad:
            if logits.requires_grad:
                out.requires_grad_(True)
                var backward_fn = CrossEntropyBackward[dtype](
                    self.reduction.value(),
                    self.ignore_index,
                    self.label_smoothing,
                    None,
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                # Target is not really a parent - but we need it in the backward pass!
                out.add_ancestry(logits, target)  # Target is of same dtype

        return out^

    fn _apply_reduction(
        self,
        losses: Tensor[dtype],
        original_target_shape: Shape,
        valid_count: Int,
    ) -> Tensor[dtype]:
        """
        Apply reduction to per-sample losses.
        """
        if self.reduction == Reduction.`None`:
            # Reshape back to original spatial dimensions
            return losses.reshape[track_grad=False](original_target_shape)
        elif self.reduction == Reduction.Sum:
            return losses.sum()
        else:  # Mean reduction
            if valid_count > 0:
                var mean_loss = losses.sum().item() / Scalar[dtype](valid_count)
                return Tensor.scalar(mean_loss)
            else:
                return Tensor.scalar(Scalar[dtype](0))

    fn _log_softmax(self, logits: Tensor[dtype]) -> Tensor[dtype]:
        """Numerically stable log(softmax(x)) for 2D input."""
        var logits_shape = logits.shape()
        var M = logits_shape[0]
        var C = logits_shape[1]
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

            var log_sum_exp = log(sum_exp) + max_val
            # Compute log_softmax: x - log_sum_exp
            for c in range(C):
                result[m, c] = logits[m, c] - log_sum_exp

        return result

    fn _smooth_probability_targets(
        self, target_probs: Tensor[dtype], num_classes: Int
    ) -> Tensor[dtype]:
        """
        Apply label smoothing to probability targets.
        Mix with uniform distribution: (1 - smoothing) * target + smoothing / C
        """
        var target_probs_shape = target_probs.shape()
        var M = target_probs_shape[0]
        var C = target_probs_shape[1]
        var smoothed = Tensor[dtype](Shape([M, C]), requires_grad=False)

        var uniform_val = Scalar[dtype](1) / Scalar[dtype](C)

        for m in range(M):
            for c in range(C):
                smoothed[m, c] = (
                    Scalar[dtype](1) - self.label_smoothing
                ) * target_probs[m, c] + self.label_smoothing * uniform_val

        return smoothed

    @staticmethod
    fn _validate_class_indices_inputs(
        logits: Tensor[dtype], target: Tensor[DType.int32], ignore_index: Int
    ):
        """
        Validate inputs for class indices targets.
        """
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

        # Validate spatial dimensions match
        if target_shape.rank() > 1:
            for i in range(1, target_shape.rank()):
                if i < input_shape.rank() - 1:
                    if target_shape[i] != input_shape[i + 1]:
                        panic(
                            "Spatial dimension mismatch at dimension "
                            + i.__str__()
                        )

        # Validate class indices using efficient coordinate-value iteration
        Self._validate_target_indices(target, C, ignore_index)

    @staticmethod
    fn _validate_probability_inputs(
        logits: Tensor[dtype], target: Tensor[dtype]
    ):
        """
        Validate inputs for probability targets.
        """
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

        # Validate spatial dimensions match
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
        """
        Validate that all target indices are within [0, num_classes-1] or equal to ignore_index.
        Uses efficient coordinate-value iteration.
        """
        for coord, value in target:
            var class_idx = value.__int__()
            if class_idx != ignore_index and (
                class_idx < 0 or class_idx >= num_classes
            ):
                # Build position string for helpful error message
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


struct CrossEntropyBackward_orig[dtype: DType](
    Copyable & Movable & ImplicitlyCopyable
):
    var reduction: Reduction
    var ignore_index: Int
    var label_smoothing: Scalar[dtype]
    var original_target: Optional[Tuple[Shape, Buffer[DType.int32]]]
    alias TAG = BACKWARD_CROSS_ENTROPY

    fn __init__(
        out self,
        reduction: Int = 0,
        ignore_index: Int = -100,
        label_smoothing: Scalar[dtype] = Scalar[dtype](0),
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

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        var gradients = output.grad()
        var logits = output.ancestry().get(0)  # logits

        # Class indices case
        if self.original_target:
            var shape_buffer_pair = self.original_target.value().copy()
            var shape = shape_buffer_pair[0].copy()
            var buffer = shape_buffer_pair[1].copy()
            var ndb = NDBuffer[DType.int32](buffer^, shape^)
            target = Tensor[DType.int32](ndb^, requires_grad=False)
            return self._backward_class_indices(logits^, target^, gradients^)
        else:
            # Probability targets case
            var target = output.ancestry().get(1)
            return self._backward_probabilities(logits^, target^, gradients^)

    fn _backward_class_indices(
        self,
        var logits: Tensor[dtype],
        var target: Tensor[DType.int32],
        upstream_grad: Gradbox[dtype],
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        """
        Backward pass for class indices targets.
        """
        var logits_shape = logits.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var spatial_dims = logits_shape[2:]
        var total_spatial = (
            spatial_dims.product() if spatial_dims.rank() > 0 else 1
        )
        var M = N * total_spatial

        # Reshape to 2D (same as forward pass)
        var logits_2d = logits.reshape[track_grad=False](Shape([M, C]))
        var target_1d = target.reshape[track_grad=False](Shape([M]))

        # Compute softmax probabilities
        var softmax_probs = self._softmax(logits_2d)

        # Create gradient tensor
        var grad_input_2d = Gradbox[dtype].zeros(Shape([M, C]), share=False)
        var valid_count = 0

        # Precompute smoothing values (same as forward pass)
        var smoothing_active = self.label_smoothing > Scalar[dtype](0)
        var true_smoothed = Scalar[dtype](0)
        var non_true_smoothed = Scalar[dtype](0)

        if smoothing_active:
            uniform_val = Scalar[dtype](1) / Scalar[dtype](C)
            non_true_smoothed = self.label_smoothing * uniform_val
            true_smoothed = (
                Scalar[dtype](1) - self.label_smoothing + non_true_smoothed
            )

        # Compute gradients for each element
        for m in range(M):
            var class_idx = target_1d[m].__int__()

            if class_idx == self.ignore_index:
                continue  # Gradient remains zero

            valid_count += 1

            if smoothing_active:
                # Gradient for label smoothing: softmax_probs - smoothed_target
                for c in range(C):
                    var target_prob = non_true_smoothed
                    if c == class_idx:
                        target_prob = true_smoothed
                    grad_input_2d[m, c] = softmax_probs[m, c] - target_prob
            else:
                # Standard gradient: softmax_probs - one_hot_target
                for c in range(C):
                    if c == class_idx:
                        grad_input_2d[m, c] = softmax_probs[m, c] - Scalar[
                            dtype
                        ](1)
                    else:
                        grad_input_2d[m, c] = softmax_probs[m, c]

        # Apply reduction scaling
        grad_input_2d = self._apply_reduction_scaling(
            grad_input_2d, upstream_grad, valid_count, M, target.shape()
        )

        # Reshape back to original shape
        var final_grad_input = grad_input_2d.reshape(logits_shape)

        return [(logits, final_grad_input^, AddTensor)]

    fn _backward_probabilities(
        self,
        logits: Tensor[dtype],
        target: Tensor[dtype],
        upstream_grad: Gradbox[dtype],
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        """
        Backward pass for probability/one-hot targets.
        """
        var logits_shape = logits.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var spatial_dims = logits_shape[2:]
        var total_spatial = (
            spatial_dims.product() if spatial_dims.rank() > 0 else 1
        )
        var M = N * total_spatial

        # Reshape to 2D (same as forward pass)
        var logits_2d = logits.reshape[track_grad=False](Shape([M, C]))
        var target_2d = target.reshape[track_grad=False](Shape([M, C]))

        # Apply label smoothing if needed (same as forward pass)
        var smoothed_target = target_2d

        if self.label_smoothing > Scalar[dtype](0):
            smoothed_target = self._smooth_probability_targets(target_2d, C)

        # Compute softmax probabilities
        var softmax_probs = self._softmax(logits_2d)

        # Gradient for probability targets: softmax_probs - target
        var grad_input_2d = Gradbox[dtype](
            softmax_probs.buffer - smoothed_target.buffer, share=False
        )

        # Apply reduction scaling (ignore_index doesn't apply to probability targets)
        var valid_count = M  # All elements are valid
        grad_input_2d = self._apply_reduction_scaling(
            grad_input_2d, upstream_grad, valid_count, M, target.shape()[0:-1]
        )

        # Reshape back to original shape
        var final_grad_input = grad_input_2d.reshape(logits_shape)

        return [(logits, final_grad_input^, AddTensor)]

    fn _apply_reduction_scaling(
        self,
        grad_input_2d: Gradbox[dtype],
        upstream_grad: Gradbox[dtype],
        valid_count: Int,
        total_elements: Int,
        target_shape: Shape,
    ) -> Gradbox[dtype]:
        """
        Apply reduction-specific scaling to gradients.
        """
        var grad_input_2d_shape = grad_input_2d.shape()
        var scaled_grad = grad_input_2d.copy()

        # Apply reduction scaling first
        if self.reduction == Reduction.Mean and valid_count > 0:
            # Scale by 1/valid_count for mean reduction
            var scale_factor = Scalar[dtype](1) / Scalar[dtype](valid_count)
            scaled_grad = grad_input_2d * scale_factor
        # No scaling needed for sum or none reduction at this stage

        # Then multiply by upstream gradient (chain rule)
        if self.reduction == Reduction.`None`:
            # For "none" reduction, upstream gradient matches target shape
            # We need to expand it to (M, C) shape
            var upstream_1d = upstream_grad.reshape(Shape([total_elements]))
            var expanded_upstream = Gradbox[dtype](
                Shape([total_elements, grad_input_2d_shape[1]]),
            )

            for m in range(total_elements):
                for c in range(grad_input_2d_shape[1]):
                    expanded_upstream[m, c] = upstream_1d[m]

            scaled_grad = scaled_grad * expanded_upstream
        else:
            # For mean/sum reduction, upstream gradient is a scalar
            var upstream_scalar = upstream_grad.item()
            scaled_grad = scaled_grad * upstream_scalar

        return scaled_grad^

    fn _softmax(self, logits: Tensor[dtype]) -> Tensor[dtype]:
        """Compute softmax for 2D input (numerically stable)."""
        var logits_shape = logits.shape()
        var M = logits_shape[0]
        var C = logits_shape[1]
        var result = Tensor[dtype](Shape([M, C]), requires_grad=False)

        for m in range(M):
            # Find max for stability
            var max_val = logits[m, 0]
            for c in range(1, C):
                if logits[m, c] > max_val:
                    max_val = logits[m, c]

            # Compute softmax
            var sum_exp = Scalar[dtype](0)
            for c in range(C):
                exp_val = exp(logits[m, c] - max_val)
                result[m, c] = exp_val
                sum_exp += exp_val

            # Normalize
            for c in range(C):
                result[m, c] = result[m, c] / sum_exp

        return result

    fn _smooth_probability_targets(
        self, target_probs: Tensor[dtype], num_classes: Int
    ) -> Tensor[dtype]:
        """
        Apply label smoothing to probability targets (same as forward pass).
        """
        var target_probs_shape = target_probs.shape()
        var M = target_probs_shape[0]
        var C = target_probs_shape[1]
        var smoothed = Tensor[dtype](Shape([M, C]), requires_grad=False)

        var uniform_val = Scalar[dtype](1) / Scalar[dtype](C)

        for m in range(M):
            for c in range(C):
                smoothed[m, c] = (
                    Scalar[dtype](1) - self.label_smoothing
                ) * target_probs[m, c] + self.label_smoothing * uniform_val

        return smoothed^


fn main() raises:
    pass
