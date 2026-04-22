from tensor import Tensor
from shapes import Shape
from common_utils import panic
from subtraction import Subtractor
from backpropagation import (
    BackwardFnArg,
    ArgumentType,
    BACKWARD_CE_CLASS_INDICES,
    BACKWARD_CE_PROBABILITIES,
)
from mnemonics import AddTensor, NotEqual
from gradbox import Gradbox
from ndbuffer import NDBuffer
from intarray import IntArray
from ancestry import Ancestor

# ═══════════════════════════════════════════════════════════════════════════════
# Reduction
# ═══════════════════════════════════════════════════════════════════════════════


struct Reduction(ImplicitlyCopyable, RegisterPassable):
    var reduction: Int

    fn __init__(out self, reduction: Int = 0):
        self.reduction = reduction
        if reduction < 0 or reduction > 2:
            panic(
                "Reduction: must be 0=mean, 1=sum, 2=none, got "
                + String(reduction)
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
            panic(
                "Reduction: must be 'mean', 'sum', or 'none', got '"
                + reduction
                + "'"
            )

    fn __copyinit__(out self, copy: Self):
        self.reduction = copy.reduction

    fn is_mean(self) -> Bool:
        return self.reduction == 0

    fn is_sum(self) -> Bool:
        return self.reduction == 1

    fn is_none(self) -> Bool:
        return self.reduction == 2


# ═════════════════════════════════════════════════════════════════════════════
#  Cetralized validation
# ═════════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct CEValidation[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn validate_label_smoothing(ls: Scalar[Self.dtype]):
        if ls < 0 or ls > 1:
            panic(
                "CrossEntropyLoss: label_smoothing must be in [0, 1], got "
                + String(ls)
            )

    @staticmethod
    fn validate_logits_rank(logits_shape: Shape):
        if logits_shape.rank() < 2:
            panic(
                "CrossEntropyLoss: logits must have rank ≥ 2 (N, C, ...), got"
                " rank "
                + String(logits_shape.rank())
            )

    @staticmethod
    fn validate_class_indices_shapes(
        logits_shape: Shape,
        target_shape: Shape,
    ):
        """
        Logits: (N, C, d1, ..., dk).
        target: (N, d1, ..., dk).
        → target rank must be logits_rank - 1.
        → batch size and all spatial dims must match.
        """
        var rank = logits_shape.rank()
        CEValidation[Self.dtype].validate_logits_rank(logits_shape)

        if logits_shape[0] != target_shape[0]:
            panic(
                "CrossEntropyLoss: batch size mismatch — logits N="
                + String(logits_shape[0])
                + ", target N="
                + String(target_shape[0])
            )

        if rank > 2:
            var expected_target_rank = rank - 1
            if target_shape.rank() != expected_target_rank:
                panic(
                    "CrossEntropyLoss: for logits rank "
                    + String(rank)
                    + ", target must have rank "
                    + String(expected_target_rank)
                    + ", got "
                    + String(target_shape.rank())
                )
            for i in range(2, rank):
                if logits_shape[i] != target_shape[i - 1]:
                    panic(
                        "CrossEntropyLoss: spatial dim mismatch at logits axis "
                        + String(i)
                        + " — logits="
                        + String(logits_shape[i])
                        + ", target="
                        + String(target_shape[i - 1])
                    )
        elif rank == 2:
            if target_shape.rank() != 1:
                panic(
                    "CrossEntropyLoss: for 2D logits (N,C), target must be 1D"
                    " (N,), got rank "
                    + String(target_shape.rank())
                )

    @staticmethod
    fn validate_probability_shapes(
        logits_shape: Shape,
        target_shape: Shape,
    ):
        """Logits and target must have identical shapes."""
        CEValidation[Self.dtype].validate_logits_rank(logits_shape)
        if logits_shape != target_shape:
            panic(
                "CrossEntropyLoss: for probability targets, logits and target"
                " must have identical shapes. Got logits="
                + String(logits_shape)
                + ", target="
                + String(target_shape)
            )

    @staticmethod
    fn validate_target_indices(
        target: Tensor[DType.int32],
        num_classes: Int,
        ignore_index: Int,
    ):
        """
        All target indices must be in [0, num_classes) or == ignore_index.
        ignore_index is always valid regardless of its value.
        """
        for coord in target.shape():
            var idx = target[coord].__int__()
            if idx == ignore_index:
                continue  # always valid
            if idx < 0 or idx >= num_classes:
                panic(
                    "CrossEntropyLoss: target index "
                    + String(idx)
                    + " at coordinate "
                    + String(coord)
                    + " is out of range [0, "
                    + String(num_classes)
                    + ")"
                )

    @staticmethod
    fn validate_num_classes(C: Int):
        if C < 1:
            panic(
                "CrossEntropyLoss: num_classes C must be ≥ 1, got " + String(C)
            )


# ═════════════════════════════════════════════════════════════════════════════
# CECommon — shared utilities for forward and backward
# ═════════════════════════════════════════════════════════════════════════════


@fieldwise_init
struct CECommon[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn flatten_spatial_class_indices(
        logits: Tensor[Self.dtype],
        target: Tensor[DType.int32],
    ) -> Tuple[
        NDBuffer[Self.dtype],  # logits_2d (M, C)
        NDBuffer[DType.int32],  # target_1d (M,)
        Int,  # M
        Int,  # C
        Shape,  # spatial_shape (for reshaping output)
        Int,  # N
    ]:
        """
        Flatten (N, C, d1..dk) → (M, C) and (N, d1..dk) → (M,).
        Permutes logits so class dim is last before flattening.
        Returns NDBuffers — safe to store in backward structs.
        """
        var logits_shape = logits.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var rank = logits_shape.rank()

        var M = N
        var spatial_dims = IntArray.with_capacity(rank - 2)
        if rank > 2:
            for i in range(2, rank):
                M *= logits_shape[i]
                spatial_dims.append(logits_shape[i])

        var logits_2d: NDBuffer[Self.dtype]
        if rank == 2:
            logits_2d = logits.buffer.reshape(Shape(M, C))
        else:
            # Permute (N, C, d1..dk) → (N, d1..dk, C) then reshape to (M, C)
            var perm = IntArray.with_capacity(rank)
            perm.append(0)
            for i in range(2, rank):
                perm.append(i)
            perm.append(1)
            var logits_buffer = logits.buffer.copy()
            var permuted = logits_buffer.permute(perm, shared=False)
            logits_2d = permuted.reshape(Shape(M, C))

        var target_1d = target.buffer.reshape(Shape(M))
        var spatial_shape = (
            Shape(spatial_dims) if len(spatial_dims) > 0 else Shape()
        )
        return (
            logits_2d^,
            target_1d^,
            M,
            C,
            spatial_shape^,
            N,
        )

    @staticmethod
    fn flatten_spatial_probabilities(
        logits: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
    ) -> Tuple[
        NDBuffer[Self.dtype],  # logits_2d (M, C)
        NDBuffer[Self.dtype],  # target_2d (M, C)
        Int,  # M
        Int,  # C
        Shape,  # spatial_shape
        Int,  # N
    ]:
        """Flatten probability targets — both logits and target (M, C)."""
        var logits_shape = logits.shape()
        var N = logits_shape[0]
        var C = logits_shape[1]
        var rank = logits_shape.rank()

        var M = N
        var spatial_dims = IntArray.with_capacity(rank - 2)
        if rank > 2:
            for i in range(2, rank):
                M *= logits_shape[i]
                spatial_dims.append(logits_shape[i])

        var logits_2d = logits.buffer.reshape(Shape(M, C))
        var target_2d = target.buffer.reshape(Shape(M, C))

        var spatial_shape = (
            Shape(spatial_dims) if len(spatial_dims) > 0 else Shape()
        )

        return logits_2d, target_2d, M, C, spatial_shape, N

    @staticmethod
    fn build_ignore_mask(
        target_1d: NDBuffer[DType.int32],
        ignore_index: Int,
    ) -> NDBuffer[Self.dtype]:
        """
        Build float mask: 1.0 where target != ignore_index, 0.0 where ignored.
        Used both in forward (zero losses) and backward (zero gradients).
        GPU safe: compare_scalar returns NDBuffer[DType.bool] → to_dtype.
        """
        return target_1d.compare_scalar[NotEqual](
            Scalar[DType.int32](ignore_index)
        ).to_dtype[Self.dtype]()

    @staticmethod
    fn apply_reduction(
        losses: NDBuffer[Self.dtype],
        reduction: Reduction,
        N: Int,
        spatial_shape: Shape,
        valid_count: Int,
    ) -> Tensor[Self.dtype]:
        """Apply reduction to per-sample losses (M,)."""
        var transformed: NDBuffer[Self.dtype]
        if reduction.is_none():
            # Reshape (M,) → (N, d1..dk)
            var spatial_rank = spatial_shape.rank()
            if spatial_rank == 0:
                transformed = losses.reshape(Shape(N))
            else:
                var out_dims = IntArray.with_capacity(spatial_rank + 1)
                out_dims.append(N)
                for i in range(spatial_rank):
                    out_dims.append(spatial_shape[i])
                transformed = losses.reshape(Shape(out_dims))
        elif reduction.is_sum():
            transformed = losses.sum(IntArray())
        else:  # mean
            transformed = losses.sum(IntArray())
            if valid_count > 0:
                transformed /= Scalar[Self.dtype](valid_count)
        return Tensor[Self.dtype](transformed^, requires_grad=False)

    @always_inline
    @staticmethod
    fn compute_log_softmax_and_softmax(
        logits_2d: NDBuffer[Self.dtype],
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """
        Returns (log_softmax, softmax) along axis=1.
        GPU safe — log_softmax and softmax are both GPU ready.
        Returns NDBuffers — safe for backward storage.
        """
        return logits_2d.log_softmax(IntArray([1]), validated=True)

    @staticmethod
    fn scale_grad_by_upstream(
        grad: NDBuffer[Self.dtype],
        upstream: Gradbox[Self.dtype],
        reduction: Reduction,
        valid_count: Int,
        M: Int,
        C: Int,
    ) -> NDBuffer[Self.dtype]:
        """
        Scale gradient by upstream grad and reduction factor.
        For none reduction: broadcast upstream (M,) → (M, C).
        For mean/sum: scalar multiply.
        GPU safe: all arithmetic ops.
        """
        if reduction.is_none():
            var ug = upstream.buffer.copy()
            var ug_flat = ug.reshape(Shape(M))
            var ug_expanded = ug_flat.unsqueeze(IntArray(-1)).broadcast_to(
                Shape(M, C)
            )
            return grad * ug_expanded
        else:
            var ug_scalar = upstream.buffer.sum(IntArray()).item()
            var scale = Scalar[Self.dtype](
                valid_count if reduction.is_mean() and valid_count > 0 else 1
            )
            return grad * (ug_scalar / scale)


# ═════════════════════════════════════════════════════════════════════════════
# CEClassIndicesBackward
# ═════════════════════════════════════════════════════════════════════════════


@fieldwise_init
struct CEClassIndicesBackward[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Backward for class index targets.

    grad[m,c] = (softmax[m,c] - (1-ls)*onehot[m,c] - ls/C)
                * ignore_mask[m] * upstream / scale

    Stores NDBuffers only.
    """

    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref bwd_arg = (
            output.ancestry()
            .backward_fn_arg()
            .get[ClassIndicesBwdArg[Self.dtype]]()
        )
        var (
            softmax_probs,
            target_1d,
            logits_shape,
            reduction,
            ignore_index,
            label_smoothing,
            valid_count,
            M,
            C,
        ) = (
            bwd_arg.softmax_probs,
            bwd_arg.target_1d,
            bwd_arg.logits_shape,
            bwd_arg.reduction,
            bwd_arg.ignore_index,
            bwd_arg.label_smoothing,
            bwd_arg.valid_count,
            bwd_arg.M,
            bwd_arg.C,
        )
        ref upstream = output.gradients()[]
        var logits = output.ancestry().get(0)

        var device = upstream.device()

        # Step 1: grad = softmax — (M, C)
        var grad = softmax_probs

        # Step 2: Build onehot from target — ignore_index rows → all zeros
        # onehot with ignore_index: those rows stay zero (no subtraction needed)
        var onehot = NDBuffer[Self.dtype].onehot(
            target_1d.to_dtype[Self.dtype](),
            C,
            device,
            ignore_index=ignore_index,
        )
        # Step 3: Apply smoothing
        var ls = label_smoothing
        if ls > Scalar[Self.dtype](0):
            var ls_uniform = ls / Scalar[Self.dtype](C)
            grad = grad - onehot * (Scalar[Self.dtype](1) - ls) - ls_uniform
        else:
            grad = grad - onehot

        # Step 4: Zero out ignored positions via ignore mask
        # ignore_mask: 1=valid, 0=ignored — built from ORIGINAL target
        var ignore_mask = CECommon[Self.dtype].build_ignore_mask(
            target_1d, ignore_index
        )
        var ignore_mask_2d = ignore_mask.unsqueeze(IntArray(-1)).broadcast_to(
            Shape(M, C)
        )
        grad = grad * ignore_mask_2d

        # Step 5: Scale by upstream grad and reduction
        var scaled = CECommon[Self.dtype].scale_grad_by_upstream(
            grad,
            upstream,
            reduction,
            valid_count,
            M,
            C,
        )
        # Step 6: Reshape back to original logits shape
        var grad_final = Gradbox[Self.dtype](scaled^, share=False).reshape(
            logits_shape
        )
        # Step 6: Reshape and UNPERMUTE back to original logits shape
        var rank = logits_shape.rank()
        if rank > 2:
            # Reshape (M, C) → (N, d1..dk, C)
            var intermediate_dims = IntArray()
            intermediate_dims.append(logits_shape[0])  # N
            for i in range(2, rank):
                intermediate_dims.append(logits_shape[i])  # d1..dk
            intermediate_dims.append(logits_shape[1])  # C last
            var reshaped = grad_final.reshape(Shape(intermediate_dims))
            # Unpermute: (N, d1..dk, C) → (N, C, d1..dk)
            # Forward perm was [0, 2, 3, ..., rank-1, 1]
            # Inverse perm is  [0, rank-1, 1, 2, ..., rank-2]
            var inv_perm = IntArray.with_capacity(rank)
            inv_perm.append(0)  # N stays first
            inv_perm.append(rank - 1)  # C was last → move to dim 1
            for i in range(1, rank - 1):
                inv_perm.append(i)  # d1..dk shift right
            grad_final = reshaped.permute(inv_perm)
            # else:
            # grad_final = grad_final.reshape(self.logits_shape)

        return [(logits, grad_final, AddTensor)]


# ═════════════════════════════════════════════════════════════════════════════
# CEClassIndicesForward
# ═════════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct ClassIndicesBwdArg[dtype: DType](ArgumentType):
    var softmax_probs: NDBuffer[Self.dtype]
    var target_1d: NDBuffer[DType.int32]
    var logits_shape: Shape
    var reduction: Reduction
    var ignore_index: Int
    var label_smoothing: Scalar[Self.dtype]
    var valid_count: Int
    var M: Int
    var C: Int


@fieldwise_init
struct CEClassIndicesForward[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    """
    Forward pass for class index targets.

    Steps:
        1. Flatten spatial: logits (N,C,d..) → (M,C), target → (M,)
        2. log_softmax + softmax along C dim
        3. ignore_mask from original target
        4. onehot from target (ignore_index rows → all zeros via onehot)
        5. NLL with optional label smoothing
        6. Zero losses at ignored positions
        7. Apply reduction
        8. Attach backward
    """

    @staticmethod
    fn forward[
        track_grad: Bool
    ](
        logits: Tensor[Self.dtype],
        target: Tensor[DType.int32],
        reduction: Reduction,
        ignore_index: Int,
        label_smoothing: Scalar[Self.dtype],
        validate: Bool,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var logits_shape = logits.shape()
        var C = logits_shape[1]

        # Validation
        if validate:
            CEValidation[Self.dtype].validate_logits_rank(logits_shape)
            CEValidation[Self.dtype].validate_num_classes(C)
            CEValidation[Self.dtype].validate_class_indices_shapes(
                logits_shape, target.shape()
            )
            CEValidation[Self.dtype].validate_target_indices(
                target, C, ignore_index
            )

        # Flatten spatial dims
        var logits_2d_ndb: NDBuffer[Self.dtype]
        var target_1d_ndb: NDBuffer[DType.int32]
        var M: Int
        var C2: Int
        var spatial_shape: Shape
        var N: Int
        logits_2d_ndb, target_1d_ndb, M, C2, spatial_shape, N = CECommon[
            Self.dtype
        ].flatten_spatial_class_indices(logits, target)

        # log_softmax + softmax
        var log_probs_ndb: NDBuffer[Self.dtype]
        var softmax_probs_ndb: NDBuffer[Self.dtype]
        log_probs_ndb, softmax_probs_ndb = CECommon[
            Self.dtype
        ].compute_log_softmax_and_softmax(logits_2d_ndb)

        # ignore_mask: 1=valid, 0=ignored — from ORIGINAL target
        var ignore_mask_ndb = CECommon[Self.dtype].build_ignore_mask(
            target_1d_ndb, ignore_index
        )
        var valid_count = (
            ignore_mask_ndb.sum(normalized_axes=IntArray()).item().__int__()
        )
        # onehot — ignore_index rows become all zeros
        var device = logits.device()
        var onehot_mask = NDBuffer[Self.dtype].onehot(
            target_1d_ndb.to_dtype[Self.dtype](),
            C2,
            device,
            ignore_index=ignore_index,
        )

        # NLL with optional label smoothing
        var ls = label_smoothing
        var losses: NDBuffer[Self.dtype]

        if ls > Scalar[Self.dtype](0):
            # loss = -(1-ls)*log_p[target] - ls * mean(log_p)
            var nll = -(onehot_mask * log_probs_ndb).sum(
                normalized_axes=IntArray(1)
            )
            var mean_log_p = log_probs_ndb.sum(
                normalized_axes=IntArray(1)
            ) / Scalar[Self.dtype](C2)
            losses = (Scalar[Self.dtype](1) - ls) * nll - ls * mean_log_p
        else:
            losses = -(onehot_mask * log_probs_ndb).sum(
                normalized_axes=IntArray(1)
            )

        # Zero ignored positions
        losses = losses * ignore_mask_ndb

        # Apply reduction
        var out = CECommon[Self.dtype].apply_reduction(
            losses, reduction, N, spatial_shape, valid_count
        )

        comptime if track_grad:
            if logits.requires_grad:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_CE_CLASS_INDICES,
                    ClassIndicesBwdArg[Self.dtype](
                        softmax_probs_ndb^,
                        target_1d_ndb^,
                        logits_shape^,
                        reduction,
                        ignore_index,
                        ls,
                        valid_count,
                        M,
                        C2,
                    ),
                )
                out.add_ancestry(backwardFnArg^, logits)

        return out^


# ═════════════════════════════════════════════════════════════════════════════
# CEProbabilitiesBackward
# ═════════════════════════════════════════════════════════════════════════════


@fieldwise_init
struct CEProbabilitiesBackward[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Backward for probability targets.

    grad = (softmax - smoothed_target) * upstream / scale

    Stores NDBuffers only.
    """

    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref bwd_arg = (
            output.ancestry()
            .backward_fn_arg()
            .get[ClassProbabilitiesBwdArg[Self.dtype]]()
        )
        var (
            softmax_probs,
            smoothed_target,
            logits_shape,
            reduction,
            valid_count,
            M,
            C,
        ) = (
            bwd_arg.softmax_probs,
            bwd_arg.smoothed_target,
            bwd_arg.logits_shape,
            bwd_arg.reduction,
            bwd_arg.valid_count,
            bwd_arg.M,
            bwd_arg.C,
        )
        ref upstream = output.gradients()[]
        var logits = output.ancestry().get(0)

        # grad = softmax - smoothed_target
        var grad = softmax_probs - smoothed_target

        # Scale by upstream and reduction
        var scaled = CECommon[Self.dtype].scale_grad_by_upstream(
            grad,
            upstream,
            reduction,
            valid_count,
            M,
            C,
        )

        # Reshape to original logits shape
        var grad_final = scaled.reshape(logits_shape)

        return [
            (
                logits,
                Gradbox[Self.dtype](grad_final, share=False),
                AddTensor,
            )
        ]


# ═════════════════════════════════════════════════════════════════════════════
# CEProbabilitiesForward
# ═════════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct ClassProbabilitiesBwdArg[dtype: DType](ArgumentType):
    var softmax_probs: NDBuffer[Self.dtype]
    var smoothed_target: NDBuffer[Self.dtype]
    var logits_shape: Shape
    var reduction: Reduction
    var valid_count: Int
    var M: Int
    var C: Int


@fieldwise_init
struct CEProbabilitiesForward[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    """
    Forward pass for soft probability targets.

    Steps:
        1. Flatten spatial: (M,C) for both logits and target
        2. log_softmax + softmax along C dim
        3. Apply label smoothing to target
        4. loss = -sum(smoothed * log_probs, axis=1)
        5. Apply reduction
        6. Attach backward
    """

    @staticmethod
    fn forward[
        track_grad: Bool
    ](
        logits: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        reduction: Reduction,
        ignore_index: Int,
        label_smoothing: Scalar[Self.dtype],
        validate: Bool,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var logits_shape = logits.shape()

        if validate:
            CEValidation[Self.dtype].validate_logits_rank(logits_shape)
            CEValidation[Self.dtype].validate_num_classes(logits_shape[1])
            CEValidation[Self.dtype].validate_probability_shapes(
                logits_shape, target.shape()
            )

        var logits_2d_ndb: NDBuffer[Self.dtype]
        var target_2d_ndb: NDBuffer[Self.dtype]
        var M: Int
        var C: Int
        var spatial_shape: Shape
        var N: Int
        logits_2d_ndb, target_2d_ndb, M, C, spatial_shape, N = CECommon[
            Self.dtype
        ].flatten_spatial_probabilities(logits, target)

        var log_probs_ndb: NDBuffer[Self.dtype]
        var softmax_probs_ndb: NDBuffer[Self.dtype]
        log_probs_ndb, softmax_probs_ndb = CECommon[
            Self.dtype
        ].compute_log_softmax_and_softmax(logits_2d_ndb)

        # Apply label smoothing to target
        var ls = label_smoothing
        var smoothed_target_ndb: NDBuffer[Self.dtype]
        if ls > Scalar[Self.dtype](0):
            var uniform = Scalar[Self.dtype](1) / Scalar[Self.dtype](C)
            smoothed_target_ndb = (
                Scalar[Self.dtype](1) - ls
            ) * target_2d_ndb + ls * uniform
        else:
            smoothed_target_ndb = target_2d_ndb

        # loss = -sum(smoothed_target * log_probs, axis=1)
        var losses = -(smoothed_target_ndb * log_probs_ndb).sum(IntArray(1))

        # M = valid_count (no ignore_index for probabilities)
        var out = CECommon[Self.dtype].apply_reduction(
            losses, reduction, N, spatial_shape, M
        )

        comptime if track_grad:
            if logits.requires_grad:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_CE_PROBABILITIES,
                    ClassProbabilitiesBwdArg[Self.dtype](
                        softmax_probs_ndb^,
                        smoothed_target_ndb^,
                        logits_shape^,
                        reduction,
                        M,
                        M,
                        C,
                    ),
                )
                out.add_ancestry(backwardFnArg^, logits)

        return out^


# ═════════════════════════════════════════════════════════════════════════════
# CrossEntropyLoss
# ═════════════════════════════════════════════════════════════════════════════


@fieldwise_init
struct CrossEntropyLoss[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    """
    CrossEntropyLoss — flexible, modular, GPU-ready.

    Features:
        Class index targets  (Tensor[DType.int32])
        Probability targets  (Tensor[dtype])
        Reduction: mean, sum, none
        Label smoothing ∈ [0, 1]
        Ignore index (any value)
        Multi-dimensional input (any rank ≥ 2)
        Full autograd — GPU-safe forward and backward
        train/eval modes

    Input:         (N, C, d1, ..., dk)  k ≥ 0
    Target class:  (N, d1, ..., dk)
    Target probs:  (N, C, d1, ..., dk)
    Output mean/sum: scalar
    Output none:   (N, d1, ..., dk)
    """

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
        CEValidation[Self.dtype].validate_label_smoothing(label_smoothing)
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
        CEValidation[Self.dtype].validate_label_smoothing(label_smoothing)
        self.reduction = Reduction(reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.training = training

    fn __copyinit__(out self, copy: Self):
        self.reduction = copy.reduction
        self.ignore_index = copy.ignore_index
        self.label_smoothing = copy.label_smoothing
        self.training = copy.training

    fn train(mut self):
        """Enable gradient tracking (training mode)."""
        self.training = True

    fn eval(mut self):
        """Disable gradient tracking (evaluation mode)."""
        self.training = False

    fn __call__(
        self,
        logits: Tensor[Self.dtype],
        target: Tensor[DType.int32],
        validate: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Class index targets."""
        if self.training:
            return CEClassIndicesForward[Self.dtype].forward[track_grad=True](
                logits,
                target,
                self.reduction,
                self.ignore_index,
                self.label_smoothing,
                validate,
            )
        else:
            return CEClassIndicesForward[Self.dtype].forward[track_grad=False](
                logits,
                target,
                self.reduction,
                self.ignore_index,
                self.label_smoothing,
                validate,
            )

    fn __call__(
        self,
        logits: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        validate: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Probability targets."""
        if self.training:
            return CEProbabilitiesForward[Self.dtype].forward[track_grad=True](
                logits,
                target,
                self.reduction,
                self.ignore_index,
                self.label_smoothing,
                validate,
            )
        else:
            return CEProbabilitiesForward[Self.dtype].forward[track_grad=False](
                logits,
                target,
                self.reduction,
                self.ignore_index,
                self.label_smoothing,
                validate,
            )


fn main() raises:
    test_ce_rank3_reduction_none_v2()


from std.testing import assert_true


fn test_ce_rank3_reduction_none_v2() raises:
    """Test rank-3 with reduction='none'."""
    print("test_ce_rank3_reduction_none_v2")

    var logits = Tensor[DType.float32].d3(
        [
            [[2.0, 1.0], [1.0, 2.0], [0.5, 0.5]],
        ],
        requires_grad=True,
    )
    var targets = Tensor[DType.int32].d2([[0, -100]])

    var criterion = CrossEntropyLoss[DType.float32](
        ignore_index=-100,
        reduction="none",
    )
    var loss = criterion(logits, targets)

    # Loss shape should match target shape [1, 2]
    assert_true(
        loss.shape().rank() == 2
        and loss.shape()[0] == 1
        and loss.shape()[1] == 2,
        "Rank-3: reduction=none should preserve target shape",
    )

    # Backward with ones
    loss.backward()

    # Ignored position should have zero gradient
    for c in range(3):
        assert_true(
            abs(logits.grad()[0, c, 1]) < 1e-10,
            "Rank-3: reduction=none - ignored class "
            + String(c)
            + " should be 0",
        )

    print("✓ Rank-3 reduction=none test passed")
