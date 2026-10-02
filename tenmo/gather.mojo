# Tensor Gather Operation — Complete GPU/CPU Implementation
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
from std.memory import unsafe_memcpy
from std.sys.info import num_physical_cores
from max.algorithm import parallelize
from .ndbuffer import NDBuffer
from .shared.shapes import Shape
from .shared.strides import Strides
from .shared.intarray import IntArray
from .shared.panic import panic
from .tensor import Tensor
from .ancestry import Ancestor
from .backpropagation import BackwardFn, ArgumentType, BackwardFnType

from .shared.mnemonics import (
    AddTensor,
    ScatterAddTensor,
    ZeroGrad,
    DEFAULT_INDEX_DTYPE,
)
from .shared import Reduction
from .kernels.gather_kernel import GatherKernel


@fieldwise_init
struct GatherArg(ArgumentType):
    """Carries axis, indices.
    Reduction and padding info for GatherBackward
    and the ScatterAddTensor engine branch.

    Stored in BackwardFn on the gather output node during forward.
    Retrieved by the backward engine when ScatterAddTensor fires —
    no extra channel needed in the return tuple.
    """

    var axis: Int
    var indices: IntArray
    var padding_idx: Optional[Int]
    var reduction: Reduction
@fieldwise_init
struct GatherBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    """padding_idx zeroing happens in engine's ScatterAddTensor branch
    by reading GatherArg.padding_idx.
    MEAN reduction: backward divides gradient by len(indices).
    """
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
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
        node's BackwardFn — the ScatterAddTensor engine branch reads
        padding_idx from there. GatherBackward scales by 1/n for MEAN,
        then signals the engine to use scatter-add semantics and clears
        its own gradbox via ZeroGrad.
        """
        var parent = output.ancestry().get(0)
        ref incoming_grad = output.gradients()

        # Fetch the erased BackwardFn once; extra_arg is the (deep-copied)
        # handle passed to the engine, bwd_arg is a typed ref into its payload.
        var extra_arg = output.ancestry().backward_fn()
        ref bwd_arg = extra_arg.get[GatherArg]()

        if bwd_arg.reduction.is_mean():
            var n = Scalar[Self.dtype](len(bwd_arg.indices))
            parent.update_grad(incoming_grad / n, ScatterAddTensor, extra_arg)
        else:
            parent.update_grad(incoming_grad, ScatterAddTensor, extra_arg)

        parent_ids.append(parent._id)
        output.gradients().zero_grad()


# SECTION 6 — Gather / EmbeddingBag forward (complete)


@fieldwise_init
struct Gather[dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE](
    Copyable, RegisterPassable
):
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
        sync: Bool = True,
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
            reduction:     How to reduce gathered rows (NONE/SUM/MEAN).
                           SUM/MEAN fuse gather+sum, output shape: (cols,).
            padding_idx:   Row index to keep zeroed.
            requires_grad: Override requires_grad. Defaults to self.requires_grad.
            sync:          Whether to synchronize the GPU operation.

        Returns:
            Contiguous tensor. Shape is (len(indices), ...) normally,
            or (cols,) when reduction is SUM or MEAN.

        Panics:
            - axis out of bounds after normalization.
            - indices is empty.
            - any index out of bounds after normalization.
        """
        var rank = self.shape().rank()

        # Normalize and validate axis
        var ax = Self._normalize_axis(axis, rank)

        if len(indices) == 0:
            panic("gather: indices cannot be empty")

        # Validate and normalize indices
        var ax_dim = self.shape()[ax]
        var normalized = Self._normalize_indices(indices, ax_dim, ax)

        # Copy / fused embedding-bag (CPU or GPU kernel)
        var is_fast_path = Self._is_fast_path(reduction, ax, rank)

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
                    self,
                    ax=ax,
                    normalized=normalized,
                    reduction=Reduction(2),
                    sync=sync,
                )
                out.requires_grad_(True)
                var bfa = BackwardFn(
                    GatherArg(ax, normalized, padding_idx, Reduction(2)),
                    GatherBackward[Self.dtype](),
                )
                out.add_ancestry(bfa^, self)
                out = Self._reduce_after[track_grad=True](
                    out, reduction, ax, sync=sync
                )
            else:
                out = Self._gather_copy(
                    self,
                    ax=ax,
                    normalized=normalized,
                    reduction=reduction,
                    sync=sync,
                )
                if grad_required:
                    out.requires_grad_(True)
                    var bfa = BackwardFn(
                        GatherArg(ax, normalized, padding_idx, reduction),
                        GatherBackward[Self.dtype](),
                    )
                    out.add_ancestry(bfa^, self)
        else:
            out = Self._gather_copy(
                self,
                ax=ax,
                normalized=normalized,
                reduction=reduction if is_fast_path else Reduction(2),
                sync=sync,
            )
            if not is_fast_path:
                out = Self._reduce_after[track_grad=False](
                    out, reduction, ax, sync=sync
                )

        return out^

    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        indices: Tensor[Self.index_dtype],
        axis: Int = 0,
        reduction: Reduction = Reduction(2),
        padding_idx: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
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
        var ax = Self._normalize_axis(axis, rank)

        if indices.numels() == 0:
            panic("gather: indices tensor cannot be empty")

        # Validate and normalize indices
        var ax_dim = self.shape()[ax]
        var n = indices.numels()
        var raw = IntArray.with_capacity(n)
        for k in range(n):
            raw.append(Int(indices.get(k)))
        var normalized = Self._normalize_indices(raw, ax_dim, ax)

        # Build output shape: (*indices.shape(), *self.shape()[ax+1:])
        var out_dims = IntArray()
        var idx_rank = indices.shape().rank()
        for d in range(idx_rank):
            out_dims.append(indices.shape()[d])
        var self_shape = self.shape()
        for d in range(ax + 1, rank):
            out_dims.append(self_shape[d])

        # GPU: route through IntArray forward + tracked reshape
        comptime if has_accelerator():
            if self.is_on_gpu():
                var flat = Self.forward[track_grad](
                    self,
                    normalized,
                    axis=ax,
                    reduction=reduction,
                    padding_idx=padding_idx,
                    requires_grad=requires_grad,
                    sync=sync,
                )
                return flat.reshape[track_grad](Shape(out_dims), sync=sync)

        # CPU: route through IntArray forward + tracked reshape
        # Same pattern as GPU: avoids scatter_add backward having to handle
        # multi-dimensional source gradients (GatherBackward sends flat indices
        # + axis=0 to Filler.scatter_add, which assumes source rank == target rank).
        var flat = Self.forward[track_grad](
            self,
            normalized,
            axis=ax,
            reduction=reduction,
            padding_idx=padding_idx,
            requires_grad=requires_grad,
            sync=sync,
        )
        return flat.reshape[track_grad](Shape(out_dims), sync=sync)

    @staticmethod
    def _normalize_axis(axis: Int, rank: Int) -> Int:
        """Normalize a (possibly negative) axis into `[0, rank)`, else panic."""
        var ax = axis if axis >= 0 else axis + rank
        if ax < 0 or ax >= rank:
            panic(
                "gather: axis ",
                String(axis),
                " out of bounds for rank ",
                String(rank),
            )
        return ax

    @staticmethod
    def _normalize_indices(var raw: IntArray, ax_dim: Int, ax: Int) -> IntArray:
        """Normalize indices in place into `[0, ax_dim)`, panicking on OOB.

        Negative indices are made positive by adding `ax_dim`. Mutates the
        owned array in place and returns it moved; the panic path reports
        the original (pre-normalization) value.
        """
        for k in range(len(raw)):
            var idx = raw[k]
            if idx < 0:
                idx += ax_dim
            if idx < 0 or idx >= ax_dim:
                panic(
                    "gather: index ",
                    String(raw[k]),
                    " out of bounds for axis ",
                    String(ax),
                    " with size ",
                    String(ax_dim),
                )
            raw[k] = idx
        return raw^

    @staticmethod
    def _is_fast_path(reduction: Reduction, ax: Int, rank: Int) -> Bool:
        """True when the fused single-pass embedding-bag kernel applies:
        rank==2, axis==0, and SUM or MEAN reduction."""
        return (
            (reduction.is_sum() or reduction.is_mean())
            and ax == 0
            and rank == 2
        )

    @staticmethod
    def _reduce_after[
        track_grad: Bool
    ](
        var gathered: Tensor[Self.dtype],
        reduction: Reduction,
        ax: Int,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Apply the sum/mean reduction after a standard gather (no-op for NONE)."""
        if reduction.is_sum():
            var r = gathered.sum[track_grad=track_grad](
                IntArray(ax), sync=sync
            )
            return r^
        elif reduction.is_mean():
            var r = gathered.mean[track_grad=track_grad](
                IntArray(ax), sync=sync
            )
            return r^
        return gathered^

    @staticmethod
    def _gather_copy(
        self: Tensor[Self.dtype],
        ax: Int,
        normalized: IntArray,
        reduction: Reduction,
        sync: Bool = True,
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
            sync:            Whether to synchronize the GPU operation.

        Returns:
            Fresh contiguous tensor on the same device as self.
        """
        ref shape = self.shape()
        var rank = shape.rank()

        # Fast path: rank==2, ax==0, SUM or MEAN
        if Self._is_fast_path(reduction, ax, rank):
            comptime if has_accelerator():
                if self.is_on_gpu():
                    try:
                        var result = GatherKernel[
                            Self.dtype, Self.index_dtype
                        ].gather_gpu(
                            self.buffer.layout(),
                            self.buffer.device_state.value(),
                            ax,
                            normalized,
                            reduction,
                            sync=sync,
                        )
                        var ndb = NDBuffer[Self.dtype].with_layout_device_state(
                            result[0], result[1]
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
            # Sort groups duplicate indices so consecutive equal rows collapse
            # into a single `count * row` accumulation. Multiplicity is
            # preserved (a repeated index contributes `count` times), so
            # token-embedding gather stays correct while skipping redundant
            # re-reads of repeated rows.
            var sorted = normalized.sorted()
            var i = 0
            var K = len(sorted)
            while i < K:
                var row = sorted[i]
                var count = 1
                while i + count < K and sorted[i + count] == row:
                    count += 1
                var row_offset = base_offset + row * cols
                if count == 1:
                    res_buffer += self_buffer[row_offset : row_offset + cols]
                else:
                    res_buffer += (
                        self_buffer[row_offset : row_offset + cols]
                        * Scalar[Self.dtype](count)
                    )
                i += count
            if reduction.is_mean():
                result /= Scalar[Self.dtype](len(normalized))
            return result^

        # General case: standard gather (CPU or GPU)
        comptime if has_accelerator():
            if self.is_on_gpu():
                try:
                    # Batch sync: if a follow-up sum/mean handles sync, skip
                    # gather_gpu's internal sync.
                    var has_followup = reduction.is_sum() or reduction.is_mean()
                    var result = GatherKernel[
                        Self.dtype, Self.index_dtype
                    ].gather_gpu(
                        self.buffer.layout(),
                        self.buffer.device_state.value(),
                        ax,
                        normalized,
                        Reduction(2),
                        sync=sync and not has_followup,
                    )
                    var ndb = NDBuffer[Self.dtype].with_layout_device_state(
                        result[0], result[1]
                    )
                    var gathered = Tensor[Self.dtype](ndb^, requires_grad=False)
                    gathered = Self._reduce_after[track_grad=False](
                        gathered, reduction, ax, sync=sync
                    )
                    return gathered^
                except e:
                    panic("gather_gpu failed: ", String(e))
                    return Tensor[Self.dtype].scalar(0)

        # Compute output shape: replace dim `ax` with len(normalized) —
        # rank stays the same.
        var out_rank = rank
        var out_shape_arr = IntArray.with_capacity(out_rank)
        for d in range(ax):
            out_shape_arr.append(shape[d])
        out_shape_arr.append(len(normalized))
        for d in range(ax + 1, rank):
            out_shape_arr.append(shape[d])

        var gathered = Tensor[Self.dtype].zeros(
            Shape(out_shape_arr), requires_grad=False, device=self.device()
        )
        var total = gathered.shape().num_elements()

        # Fast path: contiguous source — each (outer, index) slice maps to a
        # contiguous run of `inner` elements in both source and output, so
        # copy whole blocks (parallelized) instead of per-element coordinate
        # decomposition + Tensor get/set.
        if self.is_contiguous():
            var inner = 1
            for d in range(ax + 1, rank):
                inner *= shape[d]
            var outer = 1
            for d in range(ax):
                outer *= shape[d]
            var K = len(normalized)
            var src_seg = shape[ax] * inner
            var dst_seg = K * inner
            var self_off = self.offset()
            var self_in_ptr = self.data_ptr()
            var out_ptr = gathered.data_ptr()
            var n_blocks = outer * K
            var n_threads = num_physical_cores()

            def copy_block(b: Int) {imm}:
                var o = b // K
                var kk = b % K
                var src = self_off + o * src_seg + normalized[kk] * inner
                var dst = o * dst_seg + kk * inner
                unsafe_memcpy(
                    dest=out_ptr.unsafe_offset(dst),
                    src=self_in_ptr.unsafe_offset(src),
                    count=inner,
                )

            if n_blocks >= n_threads and total >= n_threads * 32768:
                parallelize(copy_block, n_blocks, n_threads)
            else:
                for b in range(n_blocks):
                    copy_block(b)
        else:
            # General path: coordinate-by-coordinate copy (any strides).
            # Reuse one coordinate buffer across all elements — no per-element
            # heap alloc and no O(rank) prepend-memmove in the hot loop.
            var coords = IntArray.with_capacity(out_rank)
            for _ in range(out_rank):
                coords.append(0)

            for flat in range(total):
                var rem = flat
                for d in range(out_rank - 1, -1, -1):
                    coords[d] = rem % gathered.shape()[d]
                    rem //= gathered.shape()[d]

                # Compute flat index into `normalized` from coords
                var flat_idx = coords[ax]
                var src_idx = normalized[flat_idx]

                # Map output coords to weight offset
                var src_offset = self.offset()
                for d in range(ax):
                    src_offset += coords[d] * self.strides()[d]
                src_offset += src_idx * self.strides()[ax]
                for k in range(rank - ax - 1):
                    src_offset += (
                        coords[ax + 1 + k] * self.strides()[ax + 1 + k]
                    )

                # `normalized` indices were range-validated at forward
                # index-normalization time, and `flat` counts within a fresh
                # `gathered` — iterator/storage addresses are in-range, so
                # skip the per-element min/max recomputation.
                gathered.storage_set[checked=False](
                    flat, self.storage_get[checked=False](src_offset)
                )

        gathered = Self._reduce_after[track_grad=False](
            gathered, reduction, ax, sync=sync
        )
        return gathered^
