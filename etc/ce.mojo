from tenmo import Tensor
from shapes import Shape
from common_utils import panic
from math import log, exp
from subtraction import Subtractor
from backpropagation import Delegate, BackwardFn, BACKWARD_CROSS_ENTROPY
from mnemonics import AddTensor
from buffers import Buffer
from gradbox import Gradbox
from ndbuffer import NDBuffer
from intarray import IntArray


# ── Permutation helpers ───────────────────────────────────────────────────────


fn class_to_last_perm(rank: Int) -> IntArray:
    """
    Builds perm that moves axis 1 (class) to last position.
    [N, C, d1, d2, ...] → [N, d1, d2, ..., C]
    rank=2: [0, 1]   (identity — no spatial dims)
    rank=3: [0, 2, 1]
    rank=4: [0, 2, 3, 1]
    rank=5: [0, 2, 3, 4, 1]
    """
    var perm = IntArray.with_capacity(rank)
    perm.append(0)
    for i in range(2, rank):
        perm.append(i)
    perm.append(1)
    return perm


fn class_to_second_perm(rank: Int) -> IntArray:
    """
    Inverse of class_to_last_perm.
    Moves last axis back to axis 1 (class dimension).
    [N, d1, d2, ..., C] → [N, C, d1, d2, ...]
    rank=2: [0, 1]
    rank=3: [0, 2, 1]
    rank=4: [0, 3, 1, 2]
    rank=5: [0, 4, 1, 2, 3]
    """
    var perm = IntArray.with_capacity(rank)
    perm.append(0)
    perm.append(rank - 1)
    for i in range(1, rank - 1):
        perm.append(i)
    return perm


fn spatial_shape_from(logits_shape: Shape, N: Int, C: Int) -> Shape:
    """
    Builds [N, spatial..., C] intermediate shape from logits [N, C, spatial...].
    Used for reshape-before-transpose in backward.
    """
    var rank = logits_shape.rank()
    var dims = IntArray.with_capacity(rank)
    dims.append(N)
    for i in range(2, rank):
        dims.append(logits_shape[i])
    dims.append(C)
    return Shape(dims)


# ── Reduction ─────────────────────────────────────────────────────────────────


@register_passable
struct Reduction(Copyable, Equatable, ImplicitlyCopyable):
    var reduction: Int
    comptime Mean = Reduction(0)
    comptime Sum = Reduction(1)
    comptime `None` = Reduction(2)

    fn __init__(out self, reduction: Int = 0):
        self.reduction = reduction
        if reduction < 0 or reduction > 2:
            panic(
                "Invalid reduction type. Must be '0 → mean', '1 → sum', or"
                " '2 → none'"
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
            panic("Invalid reduction type. Must be 'mean', 'sum', or 'none'")

    fn __copyinit__(out self, existing: Self):
        self.reduction = existing.reduction

    fn __eq__(self, other: Self) -> Bool:
        return self.reduction == other.reduction

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn value(self) -> Int:
        return self.reduction


# ── CrossEntropyLoss ──────────────────────────────────────────────────────────


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
        label_smoothing: Scalar[Self.dtype] = Scalar[Self.dtype](0),
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
        label_smoothing: Scalar[Self.dtype] = Scalar[Self.dtype](0),
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
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
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
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        if self.training:
            return self._forward_probabilities[track_grad=True](
                logits, target, validate
            )
        else:
            return self._forward_probabilities[track_grad=False](
                logits, target, validate
            )

    @always_inline
    fn _forward_class_indices[
        track_grad: Bool
    ](
        self,
        var logits: Tensor[Self.dtype],
        var target: Tensor[DType.int32],
        validate: Bool,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Forward pass for class indices targets. Works for any rank."""
        if validate:
            Self._validate_class_indices_inputs(
                logits, target, self.ignore_index
            )

        var logits_shape = logits.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var rank = logits_shape.rank()

        var total_spatial = 1
        for i in range(2, rank):
            total_spatial *= logits_shape[i]
        var M = N * total_spatial

        # [N, C, spatial...] → [N, spatial..., C] → [M, C]
        var logits_2d = (
            logits if rank == 2 else logits.permute[track_grad=False](
                class_to_last_perm(rank)
            )
        ).reshape[track_grad=False](Shape([M, C]))

        var target_1d = target.reshape[track_grad=False](Shape([M]))

        # Precompute smoothing values
        var smoothing_active = self.label_smoothing > Scalar[Self.dtype](0)
        var true_smoothed = Scalar[Self.dtype](1)
        var non_true_smoothed = Scalar[Self.dtype](0)
        if smoothing_active:
            var uniform_val = Scalar[Self.dtype](1) / Scalar[Self.dtype](C)
            non_true_smoothed = self.label_smoothing * uniform_val
            true_smoothed = (
                Scalar[Self.dtype](1) - self.label_smoothing + non_true_smoothed
            )

        var losses = Tensor[Self.dtype].zeros(Shape([M]), requires_grad=False)
        var valid_count = 0

        for m in range(M):
            var class_idx = target_1d[m].__int__()
            if class_idx == self.ignore_index:
                continue
            valid_count += 1

            var max_val = logits_2d[m, 0]
            for c in range(1, C):
                if logits_2d[m, c] > max_val:
                    max_val = logits_2d[m, c]

            var sum_exp = Scalar[Self.dtype](0)
            for c in range(C):
                sum_exp += exp(logits_2d[m, c] - max_val)
            var log_sum_exp = log(sum_exp) + max_val

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

        var out = self._apply_reduction(losses, target.shape(), valid_count)

        @parameter
        if track_grad:
            if logits.requires_grad:
                out.requires_grad_(True)
                var backward_fn = CrossEntropyBackward[Self.dtype](
                    self.reduction.value(),
                    self.ignore_index,
                    self.label_smoothing,
                    Optional(
                        (target.shape(), target.buffer.contiguous_buffer())
                    ),
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(logits)

        return out^

    @always_inline
    fn _forward_probabilities[
        track_grad: Bool
    ](
        self,
        var logits: Tensor[Self.dtype],
        var target: Tensor[Self.dtype],
        validate: Bool,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Forward pass for probability/one-hot targets. Works for any rank."""
        if validate:
            Self._validate_probability_inputs(logits, target)

        var logits_shape = logits.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var rank = logits_shape.rank()

        var total_spatial = 1
        for i in range(2, rank):
            total_spatial *= logits_shape[i]
        var M = N * total_spatial

        var perm = class_to_last_perm(rank)

        # [N, C, spatial...] → [N, spatial..., C] → [M, C]
        var logits_2d = (
            logits if rank == 2 else logits.permute[track_grad=False](perm)
        ).reshape[track_grad=False](Shape([M, C]))

        var target_2d = (
            target if rank == 2 else target.permute[track_grad=False](perm)
        ).reshape[track_grad=False](Shape([M, C]))

        var smoothing_active = self.label_smoothing > Scalar[Self.dtype](0)
        var uniform_val = Scalar[Self.dtype](0)
        if smoothing_active:
            uniform_val = Scalar[Self.dtype](1) / Scalar[Self.dtype](C)

        var losses = Tensor[Self.dtype].zeros(Shape([M]), requires_grad=False)

        for m in range(M):
            var max_val = logits_2d[m, 0]
            for c in range(1, C):
                if logits_2d[m, c] > max_val:
                    max_val = logits_2d[m, c]

            var sum_exp = Scalar[Self.dtype](0)
            for c in range(C):
                sum_exp += exp(logits_2d[m, c] - max_val)
            var log_sum_exp = log(sum_exp) + max_val

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

        var out = self._apply_reduction(
            losses, target.shape()[0:-1], M
        )

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
        else:  # Mean
            if valid_count > 0:
                return Tensor.scalar(
                    losses.sum().item() / Scalar[Self.dtype](valid_count)
                )
            else:
                return Tensor.scalar(Scalar[Self.dtype](0))

    @staticmethod
    fn _validate_class_indices_inputs(
        logits: Tensor[Self.dtype],
        target: Tensor[DType.int32],
        ignore_index: Int,
    ):
        var input_shape = logits.shape()
        var target_shape = target.shape()

        if input_shape.rank() < 2:
            panic("Input must have at least 2 dimensions")
        if target_shape.rank() != input_shape.rank() - 1:
            panic("Target must have one fewer dimension than input")
        if target_shape[0] != input_shape[0]:
            panic("Batch size mismatch between logits and target")

        var C = input_shape[1]
        for i in range(1, target_shape.rank()):
            if i < input_shape.rank() - 1:
                if target_shape[i] != input_shape[i + 1]:
                    panic(
                        "Spatial dimension mismatch at dimension " + i.__str__()
                    )

        Self._validate_target_indices(target, C, ignore_index)

    @staticmethod
    fn _validate_probability_inputs(
        logits: Tensor[Self.dtype], target: Tensor[Self.dtype]
    ):
        var input_shape = logits.shape()
        var target_shape = target.shape()

        if input_shape.rank() < 2:
            panic("Input must have at least 2 dimensions")
        if target_shape.rank() != input_shape.rank():
            panic("Target must have same rank as input for probability targets")
        if target_shape[0] != input_shape[0] or target_shape[1] != input_shape[1]:
            panic("Batch size or class count mismatch")

        for i in range(2, input_shape.rank()):
            if target_shape[i] != input_shape[i]:
                panic(
                    "Spatial dimension mismatch at dimension " + i.__str__()
                )

    @staticmethod
    fn _validate_target_indices(
        target: Tensor[DType.int32], num_classes: Int, ignore_index: Int
    ):
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


# ── CrossEntropyBackward ──────────────────────────────────────────────────────


struct CrossEntropyBackward[dtype: DType](
    Copyable & Movable & ImplicitlyCopyable
):
    comptime TAG = BACKWARD_CROSS_ENTROPY
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
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var gradients = output.grad()
        var logits = output.ancestry().get(0)

        if self.original_target:
            var shape_buffer_pair = self.original_target.value().copy()
            var shape = shape_buffer_pair[0].copy()
            var buffer = shape_buffer_pair[1].copy()
            var ndb = NDBuffer[DType.int32](buffer^, shape^)
            var target = Tensor[DType.int32](ndb^, requires_grad=False)
            return self._backward_class_indices(logits^, target^, gradients^)
        else:
            var target = output.ancestry().get(1)
            return self._backward_probabilities(logits^, target^, gradients^)

    @always_inline
    fn _backward_class_indices(
        self,
        var logits: Tensor[Self.dtype],
        var target: Tensor[DType.int32],
        upstream_grad: Gradbox[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        """Backward pass for class indices targets. Works for any rank."""
        var logits_shape = logits.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var rank = logits_shape.rank()

        var total_spatial = 1
        for i in range(2, rank):
            total_spatial *= logits_shape[i]
        var M = N * total_spatial

        # [N, C, spatial...] → [N, spatial..., C] → [M, C]
        var logits_2d = (
            logits if rank == 2 else logits.permute[track_grad=False](
                class_to_last_perm(rank)
            )
        ).reshape[track_grad=False](Shape([M, C]))

        var target_1d = target.reshape[track_grad=False](Shape([M]))

        var grad_input_2d = Gradbox[Self.dtype].zeros(
            Shape([M, C]), share=False
        )
        var valid_count = 0

        var smoothing_active = self.label_smoothing > Scalar[Self.dtype](0)
        var true_smoothed = Scalar[Self.dtype](1)
        var non_true_smoothed = Scalar[Self.dtype](0)
        if smoothing_active:
            var uniform_val = Scalar[Self.dtype](1) / Scalar[Self.dtype](C)
            non_true_smoothed = self.label_smoothing * uniform_val
            true_smoothed = (
                Scalar[Self.dtype](1) - self.label_smoothing + non_true_smoothed
            )

        for m in range(M):
            var class_idx = target_1d[m].__int__()
            if class_idx == self.ignore_index:
                continue
            valid_count += 1

            var max_val = logits_2d[m, 0]
            for c in range(1, C):
                if logits_2d[m, c] > max_val:
                    max_val = logits_2d[m, c]

            var sum_exp = Scalar[Self.dtype](0)
            for c in range(C):
                sum_exp += exp(logits_2d[m, c] - max_val)

            for c in range(C):
                var softmax_prob = exp(logits_2d[m, c] - max_val) / sum_exp
                if smoothing_active:
                    var target_prob = (
                        true_smoothed if c == class_idx else non_true_smoothed
                    )
                    grad_input_2d[m, c] = softmax_prob - target_prob
                else:
                    var target_val = (
                        Scalar[Self.dtype](1)
                        if c == class_idx
                        else Scalar[Self.dtype](0)
                    )
                    grad_input_2d[m, c] = softmax_prob - target_val

        grad_input_2d = self._apply_reduction_scaling(
            grad_input_2d, upstream_grad, valid_count, M
        )

        # [M, C] → [N, spatial..., C] → [N, C, spatial...]
        var grad_reordered: Gradbox[Self.dtype]
        if rank == 2:
            grad_reordered = grad_input_2d.reshape(logits_shape)
        else:
            var grad_temp = grad_input_2d.reshape(
                spatial_shape_from(logits_shape, N, C)
            )
            grad_reordered = grad_temp.transpose(class_to_second_perm(rank))

        return [(logits, grad_reordered^, AddTensor)]

    @always_inline
    fn _backward_probabilities(
        self,
        var logits: Tensor[Self.dtype],
        var target: Tensor[Self.dtype],
        upstream_grad: Gradbox[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        """Backward pass for probability/one-hot targets. Works for any rank."""
        var logits_shape = logits.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var rank = logits_shape.rank()

        var total_spatial = 1
        for i in range(2, rank):
            total_spatial *= logits_shape[i]
        var M = N * total_spatial

        var perm = class_to_last_perm(rank)

        # [N, C, spatial...] → [N, spatial..., C] → [M, C]
        var logits_2d = (
            logits if rank == 2 else logits.permute[track_grad=False](perm)
        ).reshape[track_grad=False](Shape([M, C]))

        var target_2d = (
            target if rank == 2 else target.permute[track_grad=False](perm)
        ).reshape[track_grad=False](Shape([M, C]))

        var grad_input_2d = Gradbox[Self.dtype].zeros(
            Shape([M, C]), share=False
        )

        var smoothing_active = self.label_smoothing > Scalar[Self.dtype](0)
        var uniform_val = Scalar[Self.dtype](0)
        if smoothing_active:
            uniform_val = Scalar[Self.dtype](1) / Scalar[Self.dtype](C)

        for m in range(M):
            var max_val = logits_2d[m, 0]
            for c in range(1, C):
                if logits_2d[m, c] > max_val:
                    max_val = logits_2d[m, c]

            var sum_exp = Scalar[Self.dtype](0)
            for c in range(C):
                sum_exp += exp(logits_2d[m, c] - max_val)

            for c in range(C):
                var softmax_prob = exp(logits_2d[m, c] - max_val) / sum_exp
                var target_prob = target_2d[m, c]
                if smoothing_active:
                    target_prob = (
                        Scalar[Self.dtype](1) - self.label_smoothing
                    ) * target_prob + self.label_smoothing * uniform_val
                grad_input_2d[m, c] = softmax_prob - target_prob

        grad_input_2d = self._apply_reduction_scaling(
            grad_input_2d, upstream_grad, M, M
        )

        # [M, C] → [N, spatial..., C] → [N, C, spatial...]
        var grad_reordered: Gradbox[Self.dtype]
        if rank == 2:
            grad_reordered = grad_input_2d.reshape(logits_shape)
        else:
            var grad_temp = grad_input_2d.reshape(
                spatial_shape_from(logits_shape, N, C)
            )
            grad_reordered = grad_temp.transpose(class_to_second_perm(rank))

        return [(logits, grad_reordered^, AddTensor)]

    fn _apply_reduction_scaling(
        self,
        grad_input_2d: Gradbox[Self.dtype],
        upstream_grad: Gradbox[Self.dtype],
        valid_count: Int,
        total_elements: Int,
    ) -> Gradbox[Self.dtype]:
        """Apply reduction-specific scaling to gradients."""
        var scaled_grad = grad_input_2d.copy()

        if self.reduction == Reduction.Mean and valid_count > 0:
            scaled_grad = scaled_grad * (
                Scalar[Self.dtype](1) / Scalar[Self.dtype](valid_count)
            )

        if self.reduction == Reduction.`None`:
            # upstream is [M] — reshape to [M, 1] and broadcast against [M, C]
            var upstream_col = upstream_grad.reshape(
                Shape([total_elements, 1])
            )
            scaled_grad = scaled_grad * upstream_col
        else:
            scaled_grad = scaled_grad * upstream_grad.item()

        return scaled_grad^

