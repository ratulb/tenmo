# ===----------------------------------------------------------------------===
# Tensor Gather Operation — Complete GPU/CPU Implementation
# ===----------------------------------------------------------------------===
#
# Gather slices along an axis using index arrays.
# Supports:
#   - N-D tensors via comptime rank specialization (ranks 1-8)
#   - Optimized 2D row-gather fast path (axis=0, most common NLP use case)
#   - Fused embedding-bag fast path (axis=0, 2D, GPU): gather+sum in one kernel
#   - Direct index buffer read (no shared memory size limit)
#   - Transparent CPU/GPU dispatch in _gather_copy
#
# Thread mapping:
#   Generic kernel      : grid-stride, 1 thread per output element
#   2D row kernel       : 1 block per output row, 1 thread per column (coalesced)
#   Embedding-bag kernel: 1 block total, 1 thread per column, sums across all rows
#

from std.sys import has_accelerator
from tenmo.ndbuffer import NDBuffer
from tenmo.device import DeviceState
from tenmo.shapes import Shape
from tenmo.strides import Strides
from tenmo.intarray import IntArray
from tenmo.common_utils import panic
from tenmo.tensor import Tensor
from tenmo.ancestry import Ancestor
from tenmo.backpropagation import BackwardFnArg, BACKWARD_GATHER, ArgumentType
from tenmo.mnemonics import AddTensor, ScatterAddTensor, ZeroGrad
from tenmo.shared import Reduction
from .kernels.gather_kernel import GatherGpu


@fieldwise_init
struct GatherArg(ArgumentType):
    """Carries axis, indices, reduction and padding info for GatherBackward
    and the ScatterAddTensor engine branch.

    Stored in BackwardFnArg on the gather output node during forward.
    Retrieved by the backward engine when ScatterAddTensor fires —
    no extra channel needed in the return tuple.
    """

    var axis: Int
    var indices: IntArray
    var padding_idx: Optional[Int]
    var reduction: Reduction
    # padding_idx zeroing happens in engine's ScatterAddTensor branch
    # by reading GatherArg.padding_idx.
    # MEAN reduction: backward divides gradient by len(indices).


# =============================================================================
# SECTION 5 — GatherBackward
# =============================================================================


@fieldwise_init
struct GatherBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        """Scatter incoming gradient back to the gathered rows.

        Forward:  out[k]                = src[indices[k]]
        Backward: grad_src[indices[k]] += grad_out[k]   (scatter-add)

        For embedding_bag (reduction.is_sum()):
            Forward:  out[c] = sum_k src[indices[k], c]
            Backward: grad_src[indices[k], c] += grad_out[c]  for all k
            — identical scatter-add semantics, GatherArg.axis=0.
        For MEAN reduction:
            Forward:  out[c] = sum_k src[indices[k], c] / n
            Backward: grad_src[indices[k], c] += grad_out[c] / n  for all k

        The GatherArg (axis + indices + reduction) is already stored on this
        node's BackwardFnArg — the ScatterAddTensor engine branch reads
        padding_idx from there. GatherBackward scales by 1/n for MEAN,
        then signals the engine to use scatter-add semantics and clears
        its own gradbox via ZeroGrad.
        """
        var parent = output.ancestry().get(0)
        ref incoming_grad = output.gradients()
        ref bwd_arg = output.ancestry().backward_fn_arg().get[GatherArg]()

        var extra_arg = output.ancestry().backward_fn_arg()

        if bwd_arg.reduction.is_mean():
            var n = Scalar[Self.dtype](len(bwd_arg.indices))
            parent.update_grad(incoming_grad / n, ScatterAddTensor, extra_arg)
        else:
            parent.update_grad(incoming_grad, ScatterAddTensor, extra_arg)

        parent_ids.append(parent._id)
        if not retain_graph:
            output.gradients().zero_grad()


# =============================================================================
# SECTION 6 — Gather / EmbeddingBag forward (complete)
# =============================================================================


@fieldwise_init
struct Gather[dtype: DType](Copyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        indices: IntArray,
        axis: Int = 0,
        reduction: Reduction = Reduction(2),
        padding_idx: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
        sync: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Gather slices along `axis` at the given indices.

        When reduction.is_sum() or reduction.is_mean() (and tensor is 2D, axis=0),
        uses a fused gather+sum that produces output shape (cols,) instead of
        (n_tokens, cols). The backward pass uses ScatterAddTensor, scaled by
        1/n for MEAN.

        Always produces a fresh contiguous output tensor (copy semantics).
        On GPU the copy is done via kernel — no map_to_host per element.
        On CPU the copy uses a strided element loop.

        Grad tracking:
            If self.requires_grad (or requires_grad override is True),
            registers GatherBackward with GatherArg(axis, normalized_indices)
            on the output node. Backward fires ScatterAddTensor which uses
            Filler.scatter_add — atomic on GPU, loop on CPU.

        Args:
            self:          Tensor.
            indices:       Indices along `axis`. Negative values are normalized.
            axis:          Axis to gather along. Negative axes are normalized.
            requires_grad: Override requires_grad. Defaults to self.requires_grad.
            reduction:     How to reduce gathered rows (NONE/SUM/MEAN).
                           SUM/MEAN fuse gather+sum, output shape: (cols,).

        Returns:
            Contiguous tensor. Shape is (len(indices), ...) normally,
            or (cols,) when reduction is SUM or MEAN.

        Panics:
            - axis out of bounds after normalization.
            - indices is empty.
            - any index out of bounds after normalization.
        """
        var rank = self.shape().rank()

        # ── Normalize and validate axis ───────────────────────────────────────
        var ax = axis if axis >= 0 else axis + rank
        if ax < 0 or ax >= rank:
            panic(
                "gather: axis ",
                String(axis),
                " out of bounds for rank ",
                String(rank),
            )

        if len(indices) == 0:
            panic("gather: indices cannot be empty")

        # ── Validate and normalize indices ────────────────────────────────────
        var ax_dim = self.shape()[ax]
        var normalized = IntArray.with_capacity(len(indices))
        for k in range(len(indices)):
            var idx = indices[k]
            if idx < 0:
                idx += ax_dim
            if idx < 0 or idx >= ax_dim:
                panic(
                    "gather: index ",
                    String(indices[k]),
                    " out of bounds for axis ",
                    String(ax),
                    " with size ",
                    String(ax_dim),
                )
            normalized.append(idx)

        # ── Copy / fused embedding-bag (CPU or GPU kernel) ────────────────────
        var is_fast_path = (
            (reduction.is_sum() or reduction.is_mean())
            and ax == 0
            and rank == 2
        )

        var out: Tensor[Self.dtype]
        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if (
                grad_required
                and not is_fast_path
                and (reduction.is_sum() or reduction.is_mean())
            ):
                # General case with grad: standard gather, wire GatherBackward
                # (NONE), then chain sum/mean so backward flows through both ops.
                out = Self._gather_copy(
                    self, ax=ax, normalized=normalized, reduction=Reduction(2)
                )
                out.requires_grad_(True)
                var bfa = BackwardFnArg[Self.dtype](
                    BACKWARD_GATHER,
                    GatherArg(ax, normalized, padding_idx, Reduction(2)),
                )
                out.add_ancestry(bfa^, self)
                if reduction.is_sum():
                    out = out.sum[track_grad=True](IntArray(ax))
                elif reduction.is_mean():
                    out = out.mean[track_grad=True](IntArray(ax))
            else:
                out = Self._gather_copy(
                    self, ax=ax, normalized=normalized, reduction=reduction
                )
                if grad_required:
                    out.requires_grad_(True)
                    var bfa = BackwardFnArg[Self.dtype](
                        BACKWARD_GATHER,
                        GatherArg(ax, normalized, padding_idx, reduction),
                    )
                    out.add_ancestry(bfa^, self)
        else:
            out = Self._gather_copy(
                self,
                ax=ax,
                normalized=normalized,
                reduction=reduction if is_fast_path else Reduction(2),
            )
            if not is_fast_path:
                if reduction.is_sum():
                    out = out.sum[track_grad=False](IntArray(ax))
                elif reduction.is_mean():
                    out = out.mean[track_grad=False](IntArray(ax))

        return out^

    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        indices: Tensor[DType.int64],
        axis: Int = 0,
        reduction: Reduction = Reduction(2),
        padding_idx: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
        sync: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Gather with multi-dimensional indices.
        Output shape: (*indices.shape(), *self.shape()[1:]) for axis=0.

        Backward uses the flat indices (same as IntArray overload) —
        GatherBackward's ScatterAddTensor is shape-agnostic.

        GPU: GPU kernel doesn't support multi-dimensional indices yet,
        so we route through the IntArray forward + tracked reshape
        (reshape is zero-cost view on GPU too).
        """
        var rank = self.shape().rank()
        var ax = axis if axis >= 0 else axis + rank
        if ax < 0 or ax >= rank:
            panic(
                "gather: axis ",
                String(axis),
                " out of bounds for rank ",
                String(rank),
            )

        if indices.numels() == 0:
            panic("gather: indices tensor cannot be empty")

        # ── Validate and normalize indices ────────────────────────────────────
        var ax_dim = self.shape()[ax]
        var n = indices.numels()
        var normalized = IntArray.with_capacity(n)
        for k in range(n):
            var idx = Int(indices.get(k))
            if idx < 0:
                idx += ax_dim
            if idx < 0 or idx >= ax_dim:
                panic(
                    "gather: index ",
                    String(Int(indices.get(k))),
                    " out of bounds for axis ",
                    String(ax),
                    " with size ",
                    String(ax_dim),
                )
            normalized.append(idx)

        # ── Build output shape: (*indices.shape(), *self.shape()[ax+1:]) ──────
        var out_dims = IntArray()
        var idx_rank = indices.shape().rank()
        for d in range(idx_rank):
            out_dims.append(indices.shape()[d])
        var self_shape = self.shape()
        for d in range(ax + 1, rank):
            out_dims.append(self_shape[d])

        # ── GPU: route through IntArray forward + tracked reshape ─────────────
        comptime if has_accelerator():
            if self.is_on_gpu():
                var flat = Self.forward[track_grad](
                    self,
                    normalized,
                    axis=ax,
                    reduction=reduction,
                    padding_idx=padding_idx,
                    requires_grad=requires_grad,
                )
                return flat.reshape[track_grad](Shape(out_dims))

        # ── CPU: multi-dimensional _gather_copy ───────────────────────────────
        var indices_shape_arr = IntArray()
        for d in range(indices.shape().rank()):
            indices_shape_arr.append(indices.shape()[d])

        var is_fast_path = (
            (reduction.is_sum() or reduction.is_mean())
            and ax == 0
            and rank == 2
        )

        var out: Tensor[Self.dtype]
        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if (
                grad_required
                and not is_fast_path
                and (reduction.is_sum() or reduction.is_mean())
            ):
                out = Self._gather_copy(
                    self,
                    ax=ax,
                    normalized=normalized,
                    reduction=Reduction(2),
                    indices_shape=indices_shape_arr,
                )
                out.requires_grad_(True)
                var bfa = BackwardFnArg[Self.dtype](
                    BACKWARD_GATHER,
                    GatherArg(ax, normalized, padding_idx, Reduction(2)),
                )
                out.add_ancestry(bfa^, self)
                if reduction.is_sum():
                    out = out.sum[track_grad=True](IntArray(ax))
                elif reduction.is_mean():
                    out = out.mean[track_grad=True](IntArray(ax))
            else:
                out = Self._gather_copy(
                    self,
                    ax=ax,
                    normalized=normalized,
                    reduction=reduction,
                    indices_shape=indices_shape_arr,
                )
                if grad_required:
                    out.requires_grad_(True)
                    var bfa = BackwardFnArg[Self.dtype](
                        BACKWARD_GATHER,
                        GatherArg(ax, normalized, padding_idx, reduction),
                    )
                    out.add_ancestry(bfa^, self)
        else:
            out = Self._gather_copy(
                self,
                ax=ax,
                normalized=normalized,
                reduction=reduction if is_fast_path else Reduction(2),
                indices_shape=indices_shape_arr,
            )
            if not is_fast_path:
                if reduction.is_sum():
                    out = out.sum[track_grad=False](IntArray(ax))
                elif reduction.is_mean():
                    out = out.mean[track_grad=False](IntArray(ax))

        return out^

    @staticmethod
    def _gather_copy(
        self: Tensor[Self.dtype],
        ax: Int,
        normalized: IntArray,
        reduction: Reduction,
        indices_shape: IntArray = IntArray(),
        sync: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Gather + optional reduce, single dispatch.

        Fast path (rank==2, ax==0, SUM/MEAN): fused single-pass kernel.
        General case: standard gather then post-process with sum/mean
        via existing Tensor ops (track_grad=False since backward is
        wired separately by Gather.forward).

        Args:
            self:            Tensor.
            ax:              Normalized axis to gather along.
            normalized:      Validated, normalized indices.
            reduction:       How to reduce gathered rows (NONE/SUM/MEAN).
            indices_shape:   Optional multi-dimensional index shape.
                             Empty (default) → 1D behavior, output dim `ax` = len(normalized).
                             Non-empty → output gains `len(indices_shape)-1` extra dims
                             inserted at `ax`.

        Returns:
            Fresh contiguous tensor on the same device as self.
        """
        ref shape = self.shape()
        var rank = shape.rank()
        var use_nd = len(indices_shape) > 0
        var indices_rank: Int = len(indices_shape) if use_nd else 1

        # ── Fast path: rank==2, ax==0, SUM or MEAN ────────────────────────────
        if (
            (reduction.is_sum() or reduction.is_mean())
            and ax == 0
            and rank == 2
        ):
            comptime if has_accelerator():
                if self.is_on_gpu():
                    try:
                        var ndb = GatherGpu[Self.dtype].gather_gpu(
                            self.buffer, ax, normalized, reduction, sync=sync
                        )
                        return Tensor[Self.dtype](ndb^, requires_grad=False)
                    except e:
                        panic("gather_gpu failed: ", String(e))
                        return Tensor[Self.dtype].scalar(0)

            var cols = shape[1]
            var result = Tensor[Self.dtype].zeros(
                Shape(cols), requires_grad=False, device=self.device()
            )
            ref res_buffer = result.buffer.data_buffer()
            ref self_buffer = self.buffer.data_buffer()
            var base_offset = self.offset()
            var sorted = normalized.sorted()
            for row in sorted:
                var row_offset = base_offset + row * cols
                res_buffer += self_buffer[row_offset : row_offset + cols]
            if reduction.is_mean():
                result /= Scalar[Self.dtype](len(normalized))
            return result^

        # ── General case: standard gather (CPU or GPU) ─────────────────────────
        # GPU kernel doesn't support multi-dimensional index shapes yet;
        # fall through to CPU general case when use_nd.
        comptime if has_accelerator():
            if self.is_on_gpu() and not use_nd:
                try:
                    # Batch sync: if a follow-up sum/mean handles sync, skip
                    # gather_gpu's internal sync.
                    var has_followup = reduction.is_sum() or reduction.is_mean()
                    var ndb = GatherGpu[Self.dtype].gather_gpu(
                        self.buffer, ax, normalized, Reduction(2),
                        sync=sync and not has_followup
                    )
                    var gathered = Tensor[Self.dtype](ndb^, requires_grad=False)
                    if reduction.is_sum():
                        gathered = gathered.sum[track_grad=False](IntArray(ax))
                    elif reduction.is_mean():
                        gathered = gathered.mean[track_grad=False](IntArray(ax))
                    return gathered^
                except e:
                    panic("gather_gpu failed: ", String(e))
                    return Tensor[Self.dtype].scalar(0)

        # Compute output shape:
        #   1D indices: replace dim `ax` with len(normalized) → rank stays the same
        #   ND indices: replace dim `ax` with indices_shape → rank += len(indices_shape) - 1
        var out_rank = rank if not use_nd else rank + len(indices_shape) - 1
        var out_shape_arr = IntArray.with_capacity(out_rank)
        for d in range(ax):
            out_shape_arr.append(shape[d])
        if not use_nd:
            out_shape_arr.append(len(normalized))
        else:
            for d in range(len(indices_shape)):
                out_shape_arr.append(indices_shape[d])
        for d in range(ax + 1, rank):
            out_shape_arr.append(shape[d])

        var gathered = Tensor[Self.dtype].zeros(
            Shape(out_shape_arr), requires_grad=False, device=self.device()
        )
        var total = gathered.shape().num_elements()

        for flat in range(total):
            var coords = IntArray.with_capacity(out_rank)
            var rem = flat
            for d in range(out_rank - 1, -1, -1):
                coords.prepend(rem % gathered.shape()[d])
                rem //= gathered.shape()[d]

            # Compute flat index into `normalized` from multi-dimensional coords
            var flat_idx: Int
            if not use_nd:
                flat_idx = coords[ax]
            else:
                flat_idx = 0
                var mul = 1
                for r in range(len(indices_shape) - 1, -1, -1):
                    flat_idx += coords[ax + r] * mul
                    mul *= indices_shape[r]
            var src_idx = normalized[flat_idx]

            # Map output coords to weight offset
            var src_offset = self.offset()
            for d in range(ax):
                src_offset += coords[d] * self.strides()[d]
            src_offset += src_idx * self.strides()[ax]
            for k in range(rank - ax - 1):
                src_offset += (
                    coords[ax + indices_rank + k] * self.strides()[ax + 1 + k]
                )

            gathered.set(flat, self.get(src_offset))

        if reduction.is_sum():
            gathered = gathered.sum[track_grad=False](IntArray(ax))
        elif reduction.is_mean():
            gathered = gathered.mean[track_grad=False](IntArray(ax))
        return gathered^
