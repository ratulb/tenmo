from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.common_utils import panic, Epsilon
from tenmo.subtraction import Subtractor
from tenmo.backpropagation import (
    BackwardFnArg,
    ArgumentType,
    BACKWARD_CE_CLASS_INDICES_INT32,
    BACKWARD_CE_CLASS_INDICES_INT64,
    BACKWARD_CE_PROBABILITIES,
)
from tenmo.mnemonics import AddTensor, NotEqual, DEFAULT_INDEX_DTYPE
from tenmo.gradbox import Gradbox
from tenmo.ndbuffer import NDBuffer
from tenmo.intarray import IntArray
from tenmo.ancestry import Ancestor
from tenmo.device import Device, GPU
from tenmo.shared import Reduction
from tenmo.softmax import SoftmaxNdBuffer
from tenmo.sum_mean_reduction import SumMeanReduction
from std.math import exp, log, max
from std.sys import simd_width_of, has_accelerator

from std.utils.numerics import min_finite


# Validation
@fieldwise_init
struct CEValidation[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def validate_label_smoothing(ls: Scalar[Self.dtype]):
        if ls < 0 or ls > 1:
            panic(
                "CrossEntropyLoss: label_smoothing must be in [0, 1], got "
                + String(ls)
            )

    @staticmethod
    def validate_logits_rank(logits_shape: Shape):
        if logits_shape.rank() < 2:
            panic(
                "CrossEntropyLoss: logits must have rank ≥ 2 (N, C, ...), got"
                " rank "
                + String(logits_shape.rank())
            )

    @staticmethod
    def validate_class_indices_shapes(
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
    def validate_probability_shapes(
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
    def validate_target_indices[
        target_dtype: DType = DType.int64,
    ](target: Tensor[target_dtype], num_classes: Int, ignore_index: Int,):
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
    def validate_num_classes(C: Int):
        if C < 1:
            panic(
                "CrossEntropyLoss: num_classes C must be ≥ 1, got " + String(C)
            )


# CECommon — Shared utilities for forward and backward


@fieldwise_init
struct CECommon[dtype: DType, target_dtype: DType = DEFAULT_INDEX_DTYPE](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def flatten_spatial_class_indices(
        logits: Tensor[Self.dtype],
        target: Tensor[Self.target_dtype],
    ) -> Tuple[
        NDBuffer[Self.dtype],  # logits_2d (M, C)
        NDBuffer[Self.target_dtype],  # target_1d (M,)
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
    def flatten_spatial_probabilities(
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
    def build_ignore_mask(
        target_1d: NDBuffer[Self.target_dtype],
        ignore_index: Int,
    ) -> NDBuffer[Self.dtype]:
        """
        Build float mask: 1.0 where target != ignore_index, 0.0 where ignored.
        Used both in forward (zero losses) and backward (zero gradients).
        GPU safe: compare_scalar returns NDBuffer[DType.bool] → to_dtype.
        """
        return target_1d.compare_scalar[NotEqual](
            Scalar[Self.target_dtype](ignore_index)
        ).to_dtype[Self.dtype]()

    @staticmethod
    def apply_reduction(
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
            transformed = SumMeanReduction[Self.dtype].sum(losses, IntArray())
        else:  # mean
            transformed = SumMeanReduction[Self.dtype].sum(losses, IntArray())
            if valid_count > 0:
                transformed /= Scalar[Self.dtype](valid_count)
        return Tensor[Self.dtype](transformed^, requires_grad=False)

    @staticmethod
    def _fused_forward_class_indices[
        track_grad: Bool
    ](
        logits_2d: NDBuffer[Self.dtype],
        target_1d: NDBuffer[Self.target_dtype],
        reduction: Reduction,
        ignore_index: Int,
        label_smoothing: Scalar[Self.dtype],
        spatial_shape: Shape,
        N: Int,
        C: Int,
        M: Int,
    ) -> Tuple[
        NDBuffer[Self.dtype], Int, Tensor[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused CPU forward for class indices CE.

        3 passes per row:
        1. Find max_val across C
        2. Compute exp, sum_exp, loss (O(1) target lookup), raw-exp write
        3. Normalize softmax in-place (only if track_grad)

        Replaces 8 allocated ops with 1 softmax buffer.
        """
        comptime SIMD_WIDTH = simd_width_of[Self.dtype]()
        var stride0 = logits_2d.strides[0]
        var stride1 = logits_2d.strides[1]
        var logits_ptr = logits_2d.data_ptr()
        var has_ls = label_smoothing > Scalar[Self.dtype](0)
        var reduction_is_none = reduction.is_none()
        var simd_end = C - (C % SIMD_WIDTH)

        # Allocate softmax (only when track_grad=True — compiled out otherwise)
        var softmax_ndb = NDBuffer[Self.dtype]()
        comptime if track_grad:
            softmax_ndb = NDBuffer[Self.dtype].zeros(logits_2d.shape)

        var per_sample_loss = NDBuffer[Self.dtype]()
        if reduction_is_none:
            per_sample_loss = NDBuffer[Self.dtype].zeros(Shape(M))

        var scalar_loss = Scalar[Self.dtype](0)
        var valid_count = 0

        for row in range(M):
            var row_base = row * stride0

            # ── Pass 1: Find max ──
            var max_val = min_finite[Self.dtype]()
            for c in range(0, simd_end, SIMD_WIDTH):
                var ptr = logits_ptr + (row_base + c * stride1)
                var vec = ptr.load[width=SIMD_WIDTH]()
                for i in range(SIMD_WIDTH):
                    max_val = max(max_val, vec[i])
            for c in range(simd_end, C):
                max_val = max(max_val, logits_ptr[row_base + c * stride1])

            # ── Pass 2: exp + sum_exp + loss ──
            var sum_exp = Scalar[Self.dtype](0)
            var sum_logits = Scalar[Self.dtype](0)

            for c in range(0, simd_end, SIMD_WIDTH):
                var ptr = logits_ptr + (row_base + c * stride1)
                var vec = ptr.load[width=SIMD_WIDTH]()
                var e_vec = exp(vec - max_val)
                for i in range(SIMD_WIDTH):
                    sum_exp += e_vec[i]
                if has_ls:
                    for i in range(SIMD_WIDTH):
                        sum_logits += vec[i]
                comptime if track_grad:
                    var sp = softmax_ndb.data_ptr()
                    (sp + (row * C + c)).store[width=SIMD_WIDTH](e_vec)
            for c in range(simd_end, C):
                var val = logits_ptr[row_base + c * stride1]
                var e = exp(val - max_val)
                sum_exp += e
                if has_ls:
                    sum_logits += val
                comptime if track_grad:
                    var sp = softmax_ndb.data_ptr()
                    sp[row * C + c] = e

            var safe_sum_exp = max(sum_exp, Epsilon[Self.dtype].value())
            var log_sum_exp = log(safe_sum_exp)

            # ── Loss (O(1) per row — target lookup) ──
            var is_valid = target_1d.get(row) != Scalar[Self.target_dtype](
                ignore_index
            )
            var loss = Scalar[Self.dtype](0)
            if is_valid:
                var tgt_idx = target_1d.get(row).__int__()
                var logit_tgt = logits_ptr[row_base + tgt_idx * stride1]
                var log_softmax_tgt = logit_tgt - max_val - log_sum_exp
                loss = -log_softmax_tgt
                if has_ls:
                    var inv_C = Scalar[Self.dtype](1) / Scalar[Self.dtype](C)
                    var mean_log_softmax = (
                        sum_logits * inv_C - max_val - log_sum_exp
                    )
                    loss = (
                        Scalar[Self.dtype](1) - label_smoothing
                    ) * loss - label_smoothing * mean_log_softmax
                valid_count += 1

            if reduction_is_none:
                per_sample_loss.set(row, loss)
            else:
                scalar_loss += loss

            # ── Pass 3: Normalize softmax in-place (only if track_grad) ──
            comptime if track_grad:
                var sp = softmax_ndb.data_ptr()
                var inv_sum_exp_val = Scalar[Self.dtype](1) / sum_exp
                var sm_base = sp + (row * C)
                for c in range(0, simd_end, SIMD_WIDTH):
                    var raw = (sm_base + c).load[width=SIMD_WIDTH]()
                    (sm_base + c).store[width=SIMD_WIDTH](raw * inv_sum_exp_val)
                for c in range(simd_end, C):
                    sm_base[c] = sm_base[c] * inv_sum_exp_val

        # ── Apply reduction ──
        var out: Tensor[Self.dtype]
        if reduction_is_none:
            var spatial_rank = spatial_shape.rank()
            if spatial_rank == 0:
                out = Tensor[Self.dtype](
                    per_sample_loss.reshape(Shape(N)),
                    requires_grad=False,
                )
            else:
                var out_dims = IntArray()
                out_dims.append(N)
                for i in range(spatial_rank):
                    out_dims.append(spatial_shape[i])
                out = Tensor[Self.dtype](
                    per_sample_loss.reshape(Shape(out_dims)),
                    requires_grad=False,
                )
        elif reduction.is_mean():
            if valid_count > 0:
                scalar_loss /= Scalar[Self.dtype](valid_count)
            out = Tensor[Self.dtype].scalar(scalar_loss, requires_grad=False)
        else:
            out = Tensor[Self.dtype].scalar(scalar_loss, requires_grad=False)

        return softmax_ndb^, valid_count, out^

    @staticmethod
    def scale_grad_by_upstream(
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
            var ug = upstream.buffer().copy()
            var ug_flat = ug.reshape(Shape(M))
            var ug_expanded = ug_flat.unsqueeze(IntArray(-1)).broadcast_to(
                Shape(M, C)
            )
            return grad.arithmetic_ops[Multiply](ug_expanded)
        else:
            var ug_scalar = (
                SumMeanReduction[Self.dtype]
                .sum(upstream.buffer(), IntArray())
                .item()
            )
            var scale = Scalar[Self.dtype](
                valid_count if reduction.is_mean() and valid_count > 0 else 1
            )
            return grad.scalar_ops[Multiply](ug_scalar / scale)

    @staticmethod
    def _fused_forward_probabilities[
        track_grad: Bool
    ](
        logits_2d: NDBuffer[Self.dtype],
        target_2d: NDBuffer[Self.dtype],
        reduction: Reduction,
        label_smoothing: Scalar[Self.dtype],
        spatial_shape: Shape,
        N: Int,
        C: Int,
        M: Int,
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype], Tensor[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused CPU forward for probability-target CE.

        3 passes per row:
        1. Find max_val across C
        2. Compute exp, sum_exp, write raw exp
        3. Normalize softmax + compute smoothed_target + compute loss

        Returns (softmax, smoothed_target, out).
        smoothed_target is needed by backward: grad = softmax - smoothed.
        """
        comptime SIMD_WIDTH = simd_width_of[Self.dtype]()
        var stride0 = logits_2d.strides[0]
        var stride1 = logits_2d.strides[1]
        var logits_ptr = logits_2d.data_ptr()
        var target_ptr = target_2d.data_ptr()
        var has_ls = label_smoothing > Scalar[Self.dtype](0)
        var reduction_is_none = reduction.is_none()
        var inv_C = Scalar[Self.dtype](1) / Scalar[Self.dtype](C)
        var simd_end = C - (C % SIMD_WIDTH)

        var softmax_ndb = NDBuffer[Self.dtype]()
        var smoothed_target_ndb = NDBuffer[Self.dtype]()
        comptime if track_grad:
            softmax_ndb = NDBuffer[Self.dtype].zeros(logits_2d.shape)
            smoothed_target_ndb = NDBuffer[Self.dtype].zeros(logits_2d.shape)

        var per_sample_loss = NDBuffer[Self.dtype]()
        if reduction_is_none:
            per_sample_loss = NDBuffer[Self.dtype].zeros(Shape(M))

        var scalar_loss = Scalar[Self.dtype](0)

        for row in range(M):
            var row_base = row * stride0

            # ── Pass 1: Find max ──
            var max_val = min_finite[Self.dtype]()
            for c in range(0, simd_end, SIMD_WIDTH):
                var ptr = logits_ptr + (row_base + c * stride1)
                var vec = ptr.load[width=SIMD_WIDTH]()
                for i in range(SIMD_WIDTH):
                    max_val = max(max_val, vec[i])
            for c in range(simd_end, C):
                max_val = max(max_val, logits_ptr[row_base + c * stride1])

            # ── Pass 2: exp + sum_exp + write raw exp ──
            var sum_exp = Scalar[Self.dtype](0)
            for c in range(0, simd_end, SIMD_WIDTH):
                var ptr = logits_ptr + (row_base + c * stride1)
                var vec = ptr.load[width=SIMD_WIDTH]()
                var e_vec = exp(vec - max_val)
                for i in range(SIMD_WIDTH):
                    sum_exp += e_vec[i]
                comptime if track_grad:
                    var sp = softmax_ndb.data_ptr()
                    (sp + (row * C + c)).store[width=SIMD_WIDTH](e_vec)
            for c in range(simd_end, C):
                var val = logits_ptr[row_base + c * stride1]
                var e = exp(val - max_val)
                sum_exp += e
                comptime if track_grad:
                    var sp = softmax_ndb.data_ptr()
                    sp[row * C + c] = e

            var safe_sum_exp = max(sum_exp, Epsilon[Self.dtype].value())
            var log_sum_exp = log(safe_sum_exp)
            var inv_sum_exp = Scalar[Self.dtype](1) / safe_sum_exp

            # ── Pass 3: normalize softmax + write smoothed_target + compute loss ──
            var loss = Scalar[Self.dtype](0)
            for c in range(0, simd_end, SIMD_WIDTH):
                var t_vec = (target_ptr + (row_base + c * stride1)).load[
                    width=SIMD_WIDTH
                ]()
                var l_vec = (logits_ptr + (row_base + c * stride1)).load[
                    width=SIMD_WIDTH
                ]()

                # smoothed_target[c] = target[c] * (1-ls) + ls/C
                var smoothed: SIMD[Self.dtype, SIMD_WIDTH]
                if has_ls:
                    smoothed = (
                        t_vec * (Scalar[Self.dtype](1) - label_smoothing)
                        + label_smoothing * inv_C
                    )
                else:
                    smoothed = t_vec
                comptime if track_grad:
                    var st_ptr = smoothed_target_ndb.data_ptr()
                    (st_ptr + (row * C + c)).store[width=SIMD_WIDTH](smoothed)

                # normalized softmax
                comptime if track_grad:
                    var sp = softmax_ndb.data_ptr()
                    var sm_vec = (sp + (row * C + c)).load[
                        width=SIMD_WIDTH
                    ]() * inv_sum_exp
                    (sp + (row * C + c)).store[width=SIMD_WIDTH](sm_vec)

                # log_softmax[c] = logits[c] - max_val - log_sum_exp
                var log_sm_vec = (l_vec - max_val) - log_sum_exp

                # loss -= sum(smoothed * log_sm)
                for i in range(SIMD_WIDTH):
                    loss -= smoothed[i] * log_sm_vec[i]

            # Flush remainder
            for c in range(simd_end, C):
                var t_val = target_ptr[row_base + c * stride1]
                var l_val = logits_ptr[row_base + c * stride1]

                var smoothed_val: Scalar[Self.dtype]
                if has_ls:
                    smoothed_val = (
                        t_val * (Scalar[Self.dtype](1) - label_smoothing)
                        + label_smoothing * inv_C
                    )
                else:
                    smoothed_val = t_val
                comptime if track_grad:
                    var st_ptr = smoothed_target_ndb.data_ptr()
                    st_ptr[row * C + c] = smoothed_val

                var sm = exp(l_val - max_val) * inv_sum_exp
                comptime if track_grad:
                    var sp = softmax_ndb.data_ptr()
                    sp[row * C + c] = sm

                var log_sm = l_val - max_val - log_sum_exp
                loss -= smoothed_val * log_sm

            if reduction_is_none:
                per_sample_loss.set(row, loss)
            else:
                scalar_loss += loss

        # ── Apply reduction ──
        var out: Tensor[Self.dtype]
        if reduction_is_none:
            var spatial_rank = spatial_shape.rank()
            if spatial_rank == 0:
                out = Tensor[Self.dtype](
                    per_sample_loss.reshape(Shape(N)),
                    requires_grad=False,
                )
            else:
                var out_dims = IntArray()
                out_dims.append(N)
                for i in range(spatial_rank):
                    out_dims.append(spatial_shape[i])
                out = Tensor[Self.dtype](
                    per_sample_loss.reshape(Shape(out_dims)),
                    requires_grad=False,
                )
        elif reduction.is_mean():
            scalar_loss /= Scalar[Self.dtype](M)
            out = Tensor[Self.dtype].scalar(scalar_loss, requires_grad=False)
        else:
            out = Tensor[Self.dtype].scalar(scalar_loss, requires_grad=False)

        return softmax_ndb^, smoothed_target_ndb^, out^


# CEClassIndicesBackward


@fieldwise_init
struct CEClassIndicesBackward[dtype: DType, target_dtype: DType = DType.int32](
    ImplicitlyCopyable & Movable
):
    """
    Backward for class index targets.

    grad[m,c] = (softmax[m,c] - (1-ls)*onehot[m,c] - ls/C)
                * ignore_mask[m] * upstream / scale

    Stores NDBuffers only.
    """

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref bwd_arg = (
            output.ancestry()
            .backward_fn_arg()
            .get[ClassIndicesBwdArg[Self.dtype, Self.target_dtype]]()
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
        ref upstream = output.gradients()
        var logits = output.ancestry().get(0)

        # Steps 2-5 fused: onehot + smoothing + ignore_mask + scaling in one GPU kernel
        # Falls back to decomposed CPU path when no GPU.
        var scaled = NDBuffer[Self.dtype]()
        comptime if has_accelerator():
            from tenmo.kernels.crossentropy_fused_kernel import (
                CrossEntropyFusedKernel,
            )

            if softmax_probs.is_on_gpu():
                try:
                    scaled = CrossEntropyFusedKernel[
                        Self.dtype, Self.target_dtype
                    ].launch_backward(
                        softmax_probs,
                        target_1d,
                        upstream.buffer(),
                        reduction,
                        valid_count,
                        M,
                        C,
                        ignore_index,
                        label_smoothing,
                    )
                except e:
                    panic(
                        "CrossEntropyFusedKernel.launch_backward failed: "
                        + String(e)
                    )
            else:
                # CPU fallback — decomposed path
                var onehot = NDBuffer[Self.dtype].onehot(
                    target_1d.to_dtype[Self.dtype](),
                    C,
                    softmax_probs.device(),
                    ignore_index=ignore_index,
                )
                var grad = softmax_probs
                var ls = label_smoothing
                if ls > Scalar[Self.dtype](0):
                    var ls_uniform = ls / Scalar[Self.dtype](C)
                    grad = grad.arithmetic_ops[Subtract](
                        onehot.scalar_ops[Multiply](Scalar[Self.dtype](1) - ls)
                    ).scalar_ops[Subtract](ls_uniform)
                else:
                    grad = grad.arithmetic_ops[Subtract](onehot)
                var ignore_mask = CECommon[
                    Self.dtype, Self.target_dtype
                ].build_ignore_mask(target_1d, ignore_index)
                var ignore_mask_2d = ignore_mask.unsqueeze(
                    IntArray(-1)
                ).broadcast_to(Shape(M, C))
                grad = grad.arithmetic_ops[Multiply](ignore_mask_2d)
                scaled = CECommon[Self.dtype].scale_grad_by_upstream(
                    grad, upstream, reduction, valid_count, M, C
                )
        else:
            # No accelerator — CPU decomposed path
            var onehot = NDBuffer[Self.dtype].onehot(
                target_1d.to_dtype[Self.dtype](),
                C,
                softmax_probs.device(),
                ignore_index=ignore_index,
            )
            var grad = softmax_probs
            var ls = label_smoothing
            if ls > Scalar[Self.dtype](0):
                var ls_uniform = ls / Scalar[Self.dtype](C)
                grad = grad.arithmetic_ops[Subtract](
                    onehot.scalar_ops[Multiply](Scalar[Self.dtype](1) - ls)
                ).scalar_ops[Subtract](ls_uniform)
            else:
                grad = grad.arithmetic_ops[Subtract](onehot)
            var ignore_mask = CECommon[
                Self.dtype, Self.target_dtype
            ].build_ignore_mask(target_1d, ignore_index)
            var ignore_mask_2d = ignore_mask.unsqueeze(
                IntArray(-1)
            ).broadcast_to(Shape(M, C))
            grad = grad.arithmetic_ops[Multiply](ignore_mask_2d)
            scaled = CECommon[Self.dtype].scale_grad_by_upstream(
                grad, upstream, reduction, valid_count, M, C
            )
        # Step 6: Reshape back to original logits shape
        var _tmp0 = Gradbox[Self.dtype](scaled^)
        var grad_final = _tmp0.reshape(logits_shape)
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

        if logits.requires_grad:
            logits.update_grad(grad_final, AddTensor, None)
        parent_ids.append(logits._id)
        if not retain_graph:
            upstream.zero_grad()


# CEClassIndicesForward
@fieldwise_init
struct ClassIndicesBwdArg[
    dtype: DType, target_dtype: DType = DEFAULT_INDEX_DTYPE
](ArgumentType):
    var softmax_probs: NDBuffer[Self.dtype]
    var target_1d: NDBuffer[Self.target_dtype]
    var logits_shape: Shape
    var reduction: Reduction
    var ignore_index: Int
    var label_smoothing: Scalar[Self.dtype]
    var valid_count: Int
    var M: Int
    var C: Int


@fieldwise_init
struct CEClassIndicesForward[
    dtype: DType, target_dtype: DType = DEFAULT_INDEX_DTYPE
](ImplicitlyCopyable, RegisterPassable):
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
    def forward[
        track_grad: Bool
    ](
        logits: Tensor[Self.dtype],
        target: Tensor[Self.target_dtype],
        reduction: Reduction,
        ignore_index: Int,
        label_smoothing: Scalar[Self.dtype],
        validate: Bool,
        sync: Bool = True,
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
            CEValidation[Self.dtype].validate_target_indices[Self.target_dtype](
                target, C, ignore_index
            )

        # Flatten spatial dims
        var logits_2d_ndb: NDBuffer[Self.dtype]
        var target_1d_ndb: NDBuffer[Self.target_dtype]
        var M: Int
        var C2: Int
        var spatial_shape: Shape
        var N: Int
        logits_2d_ndb, target_1d_ndb, M, C2, spatial_shape, N = CECommon[
            Self.dtype, Self.target_dtype
        ].flatten_spatial_class_indices(logits, target)

        # ── Compute softmax + loss: GPU fused vs CPU decomposed ──
        var softmax_probs_ndb = NDBuffer[Self.dtype]()
        var valid_count = 0
        var out: Tensor[Self.dtype]

        comptime if has_accelerator():
            from tenmo.kernels.crossentropy_fused_kernel import (
                CrossEntropyFusedKernel,
            )

            if logits_2d_ndb.is_on_gpu():
                # GPU fused path
                var losses_ndb = NDBuffer[Self.dtype]()
                var scalar_loss_val = Scalar[Self.dtype](0)
                try:
                    (
                        softmax_probs_ndb,
                        losses_ndb,
                        scalar_loss_val,
                        valid_count,
                    ) = CrossEntropyFusedKernel[
                        Self.dtype, Self.target_dtype
                    ].launch(
                        logits_2d_ndb,
                        target_1d_ndb,
                        reduction,
                        ignore_index,
                        label_smoothing,
                    )
                except e:
                    panic("CrossEntropyFusedKernel.launch failed: " + String(e))

                if reduction.is_none():
                    var spatial_rank = spatial_shape.rank()
                    if spatial_rank == 0:
                        out = Tensor[Self.dtype](
                            losses_ndb.reshape(Shape(N)),
                            requires_grad=False,
                        )
                    else:
                        var out_dims = IntArray()
                        out_dims.append(N)
                        for i in range(spatial_rank):
                            out_dims.append(spatial_shape[i])
                        out = Tensor[Self.dtype](
                            losses_ndb.reshape(Shape(out_dims)),
                            requires_grad=False,
                        )
                else:
                    if reduction.is_mean() and valid_count > 0:
                        scalar_loss_val /= Scalar[Self.dtype](valid_count)
                    out = Tensor[Self.dtype].scalar(
                        scalar_loss_val, requires_grad=False
                    )
            else:
                softmax_probs_ndb, valid_count, out = CECommon[
                    Self.dtype, Self.target_dtype
                ]._fused_forward_class_indices[track_grad](
                    logits_2d_ndb,
                    target_1d_ndb,
                    reduction,
                    ignore_index,
                    label_smoothing,
                    spatial_shape,
                    N,
                    C2,
                    M,
                )
        else:
            softmax_probs_ndb, valid_count, out = CECommon[
                Self.dtype, Self.target_dtype
            ]._fused_forward_class_indices[track_grad](
                logits_2d_ndb,
                target_1d_ndb,
                reduction,
                ignore_index,
                label_smoothing,
                spatial_shape,
                N,
                C2,
                M,
            )

        comptime if track_grad:
            if logits.requires_grad:
                out.requires_grad_(True)
                var ce_op_code = BACKWARD_CE_CLASS_INDICES_INT32
                if Self.target_dtype == DType.int64:
                    ce_op_code = BACKWARD_CE_CLASS_INDICES_INT64
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    ce_op_code,
                    ClassIndicesBwdArg[Self.dtype, Self.target_dtype](
                        softmax_probs_ndb^,
                        target_1d_ndb^,
                        logits_shape^,
                        reduction,
                        ignore_index,
                        label_smoothing,
                        valid_count,
                        M,
                        C2,
                    ),
                )
                out.add_ancestry(backwardFnArg^, logits)

        comptime if has_accelerator():
            if sync and out.is_on_gpu():
                out.buffer.sync()

        return out^


# CEProbabilitiesBackward


@fieldwise_init
struct CEProbabilitiesBackward[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Backward for probability targets.

    grad = (softmax - smoothed_target) * upstream / scale

    Stores NDBuffers only.
    """

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
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
        ref upstream = output.gradients()
        var logits = output.ancestry().get(0)

        # grad = softmax - smoothed_target
        var grad = softmax_probs.arithmetic_ops[Subtract](smoothed_target)

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

        var gradbox = Gradbox[Self.dtype](grad_final)
        if logits.requires_grad:
            logits.update_grad(gradbox, AddTensor, None)
        parent_ids.append(logits._id)
        if not retain_graph:
            upstream.zero_grad()


# CEProbabilitiesForward
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
    def forward[
        track_grad: Bool
    ](
        logits: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        reduction: Reduction,
        ignore_index: Int,
        label_smoothing: Scalar[Self.dtype],
        validate: Bool,
        sync: Bool = True,
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

        # TODO: Write a fused GPU kernel for probability-target CE.
        # Fall back to CPU: transfer GPU data to CPU before the CPU-only fused function.
        # Backward handles CPU data correctly via arithmetic_ops dispatch.
        var logits_cpu = logits_2d_ndb
        var target_cpu = target_2d_ndb
        comptime if has_accelerator():
            if logits_2d_ndb.is_on_gpu():
                try:
                    logits_cpu = logits_2d_ndb.to_cpu(sync=True)
                    target_cpu = target_2d_ndb.to_cpu(sync=True)
                except e:
                    panic(
                        "CEProbabilitiesForward: GPU-to-CPU transfer failed: "
                        + String(e)
                    )

        var softmax_probs_ndb: NDBuffer[Self.dtype]
        var smoothed_target_ndb: NDBuffer[Self.dtype]
        var out: Tensor[Self.dtype]
        softmax_probs_ndb, smoothed_target_ndb, out = CECommon[
            Self.dtype
        ]._fused_forward_probabilities[track_grad](
            logits_cpu,
            target_cpu,
            reduction,
            label_smoothing,
            spatial_shape,
            N,
            C,
            M,
        )

        # Transfer result back to GPU if logits was on GPU
        comptime if has_accelerator():
            if logits.is_on_gpu():
                try:
                    out = out.to_gpu(stop_grad=False)
                    comptime if track_grad:
                        if logits.requires_grad:
                            softmax_probs_ndb = softmax_probs_ndb.to_gpu(
                                logits.device().kind[GPU]
                            )
                            smoothed_target_ndb = smoothed_target_ndb.to_gpu(
                                logits.device().kind[GPU]
                            )
                except e:
                    panic(
                        "CEProbabilitiesForward: CPU-to-GPU transfer failed: "
                        + String(e)
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

        comptime if has_accelerator():
            if sync and out.is_on_gpu():
                out.buffer.sync()

        return out^


# CrossEntropyLoss


@fieldwise_init
struct CrossEntropyLoss[
    dtype: DType, target_dtype: DType = DEFAULT_INDEX_DTYPE
](ImplicitlyCopyable, RegisterPassable):
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

    def __init__(
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

    def __init__(
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

    def __init__(out self, *, copy: Self):
        self.reduction = copy.reduction
        self.ignore_index = copy.ignore_index
        self.label_smoothing = copy.label_smoothing
        self.training = copy.training

    def train(mut self):
        """Enable gradient tracking (training mode)."""
        self.training = True

    def eval(mut self):
        """Disable gradient tracking (evaluation mode)."""
        self.training = False

    def __call__(
        self,
        logits: Tensor[Self.dtype],
        target: Tensor[Self.target_dtype],
        validate: Bool = True,
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Class index targets."""
        if self.training:
            return CEClassIndicesForward[Self.dtype, Self.target_dtype].forward[
                track_grad=True
            ](
                logits,
                target,
                self.reduction,
                self.ignore_index,
                self.label_smoothing,
                validate,
                sync=sync,
            )
        else:
            return CEClassIndicesForward[Self.dtype, Self.target_dtype].forward[
                track_grad=False
            ](
                logits,
                target,
                self.reduction,
                self.ignore_index,
                self.label_smoothing,
                validate,
                sync=sync,
            )

    def __call__(
        self,
        logits: Tensor[Self.dtype],
        target: Tensor[Self.dtype],
        validate: Bool = True,
        sync: Bool = True,
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
                sync=sync,
            )
        else:
            return CEProbabilitiesForward[Self.dtype].forward[track_grad=False](
                logits,
                target,
                self.reduction,
                self.ignore_index,
                self.label_smoothing,
                validate,
                sync=sync,
            )
