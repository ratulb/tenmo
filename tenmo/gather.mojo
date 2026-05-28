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

from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys import has_accelerator
from tenmo.ndbuffer import NDBuffer
from tenmo.device import DeviceState
from tenmo.shapes import Shape
from tenmo.strides import Strides
from tenmo.device import GPU
from tenmo.buffers import Buffer
from tenmo.array import Array
from tenmo.intarray import IntArray
from tenmo.common_utils import panic
from tenmo.tensor import Tensor
from tenmo.gradbox import Gradbox
from tenmo.ancestry import Ancestor
from tenmo.backpropagation import BackwardFnArg, BACKWARD_GATHER, ArgumentType
from tenmo.mnemonics import AddTensor, ScatterAddTensor, ZeroGrad
from tenmo.shared import Reduction


# =============================================================================
# SECTION 1 — GPU Kernels
# =============================================================================


def gather_gpu_kernel[
    dtype: DType,
    rank: Int,
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: Array,
    in_strides: Array,
    in_offset: Int,
    indices_buffer: UnsafePointer[Int64, ImmutAnyOrigin],
    indices_len: Int,
    axis: Int,
    out_shape: Array,
    out_strides: Array,
    total_output: Int,
):
    """Generic gather kernel for rank-N tensors.

    One thread per output element — grid-stride loop for large outputs.
    No shared memory for indices: indices_len can exceed block size
    (e.g. 500 unique tokens per review vs block size 256).

    Thread work:
      1. Unravel flat output index → per-axis coordinates (comptime unrolled)
      2. Replace axis coordinate with indices_buffer[coord[axis]]
      3. Compute source flat index via fma(strides, coords, offset)
      4. Copy element to output

    Args:
        out_buffer:     Output device buffer (total_output elements).
        in_buffer:      Input device buffer.
        in_shape:       Input shape array (rank elements).
        in_strides:     Input strides array (rank elements).
        in_offset:      Input base offset.
        indices_buffer: Gather indices on device (indices_len elements).
        indices_len:    Number of indices — equals out_shape[axis].
        axis:           Axis to gather along.
        out_shape:      Output shape array (rank elements).
        out_strides:    Output strides array (rank elements).
        total_output:   Total number of output elements.
    """
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var gstride = Int(block_dim.x * grid_dim.x)
    var out_idx = gtid

    while out_idx < total_output:
        # ── Unravel flat output index → per-axis coordinates ──────────────────
        var out_coords = Array()
        out_coords.size = rank
        var rem = out_idx
        comptime for d in range(rank - 1, -1, -1):
            out_coords.storage[d] = rem % out_shape[d]
            rem //= out_shape[d]

        # ── Map axis coordinate through indices ───────────────────────────────
        var src_coords = out_coords
        var idx_val = indices_buffer[out_coords[axis]]
        if idx_val < 0:
            idx_val += Int64(in_shape[axis])
        src_coords.storage[axis] = Int(idx_val)

        # ── Compute flat indices and copy ─────────────────────────────────────
        var src_flat = in_strides.fma(src_coords, in_offset)
        var dst_flat = out_strides.fma(out_coords, 0)
        out_buffer[dst_flat] = in_buffer[src_flat]

        out_idx += gstride


def gather_rows_2d_kernel[
    dtype: DType
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_rows: Int,
    in_cols: Int,
    in_row_stride: Int,
    indices_buffer: UnsafePointer[Int64, ImmutAnyOrigin],
    out_rows: Int,
    out_row_stride: Int,
):
    """Optimised 2-D row-gather kernel (axis=0).

    Grid  : (out_rows,)  — one block per output row
    Block : (block_cols) — threads cover all columns, grid-stride within block

    Perfectly coalesced column access — each warp reads/writes contiguous
    memory. The hot path for NLP embedding lookup:
        weights_0_1 (vocab_size, hidden_size) gathered along axis=0.

    Args:
        out_buffer:     Output device buffer (out_rows × in_cols).
        in_buffer:      Input device buffer  (in_rows  × in_cols).
        in_rows:        Source row count (for negative index clamping).
        in_cols:        Column count — hidden_size in NLP use case.
        in_row_stride:  Source row stride (usually == in_cols).
        indices_buffer: Row indices to gather (out_rows elements).
        out_rows:       Number of output rows == len(indices).
        out_row_stride: Output row stride (usually == in_cols).
    """
    var row = Int(block_idx.x)
    var col = Int(thread_idx.x)

    if row >= out_rows:
        return

    var src_row = indices_buffer[row]
    if src_row < 0:
        src_row += Int64(in_rows)

    # Grid-stride within block to cover in_cols > block_dim
    var col_stride = Int(block_dim.x)
    var c = col
    while c < in_cols:
        out_buffer[row * out_row_stride + c] = in_buffer[
            src_row * Int64(in_row_stride) + Int64(c)
        ]
        c += col_stride


def embedding_bag_kernel[
    dtype: DType,
    mean: Bool,
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_rows: Int,
    in_cols: Int,
    in_row_stride: Int,
    indices_buffer: UnsafePointer[Int64, ImmutAnyOrigin],
    n_indices: Int,
):
    """Fused gather+sum kernel for 2-D embedding lookup (axis=0).

    Replaces gather(indices, axis=0).sum(axis=0) with a single kernel.
    When mean=True, divides the accumulated sum by n_indices.
    Output shape: (in_cols,) — one scalar per column, summed across all
    gathered rows. No intermediate (n_tokens, hidden_size) buffer needed.

    Grid  : (1,)       — single block
    Block : (block_cols) — one thread per column, grid-stride for cols > block

    Each thread owns one column and accumulates across all n_indices rows:
        out[c] = sum(in[indices[k] * in_row_stride + c] for k in range(n_indices))
        (divided by n_indices when mean=True)

    This is the hot path for bag-of-words NLP models:
        hidden = embedding_bag(weights, token_ids)
        # replaces: weights.gather(token_ids, axis=0).sum(axis=0)

    Memory traffic: reads n_indices × in_cols elements, writes in_cols elements.
    No atomics needed — each output element is owned by exactly one thread.

    Args:
        out_buffer:     Output device buffer (in_cols elements).
        in_buffer:      Input device buffer (in_rows × in_cols).
        in_rows:        Source row count (for negative index clamping).
        in_cols:        Column count — hidden_size in NLP use case.
        in_row_stride:  Source row stride (== in_cols for contiguous tensor).
        indices_buffer: Row indices to gather and sum (n_indices elements).
        n_indices:      Number of indices to sum over.
    """
    var col = Int(thread_idx.x)
    var col_stride = Int(block_dim.x)
    var c = col
    var divisor = Scalar[dtype](n_indices)
    while c < in_cols:
        var acc = Scalar[dtype](0)
        for k in range(n_indices):
            var src_row = indices_buffer[k]
            if src_row < 0:
                src_row += Int64(in_rows)
            acc += in_buffer[src_row * Int64(in_row_stride) + Int64(c)]
        comptime if mean:
            out_buffer[c] = acc / divisor
        else:
            out_buffer[c] = acc
        c += col_stride


# =============================================================================
# SECTION 2 — Launch helpers
# =============================================================================


def _gather_launch_config(total_elements: Int) -> Tuple[Int, Int]:
    """Returns (threads_per_block, num_blocks) for generic gather kernel."""
    var tpb = 256 if total_elements >= 128 else 128
    var blocks = min((total_elements + tpb - 1) // tpb, 4096)
    return (tpb, blocks)


def _gather_2d_block_cols(in_cols: Int) -> Int:
    """Nearest power-of-2 thread count for 2-D kernel, capped at 512."""
    if in_cols <= 32:
        return 32
    if in_cols <= 64:
        return 64
    if in_cols <= 128:
        return 128
    if in_cols <= 256:
        return 256
    return 512


def _launch_gather_generic[
    dtype: DType, rank: Int
](
    ctx: DeviceContext,
    out_dev: DeviceBuffer[dtype],
    in_dev: DeviceBuffer[dtype],
    in_shape: Array,
    in_strides: Array,
    in_offset: Int,
    idx_dev: DeviceBuffer[DType.int64],
    indices_len: Int,
    axis: Int,
    out_shape: Array,
    out_strides: Array,
    total_output: Int,
) raises:
    """Launch generic gather kernel for a specific comptime rank."""
    var (tpb, blocks) = _gather_launch_config(total_output)
    var compiled = ctx.compile_function[
        gather_gpu_kernel[dtype, rank],
        gather_gpu_kernel[dtype, rank],
    ]()
    ctx.enqueue_function(
        compiled,
        out_dev,
        in_dev,
        in_shape,
        in_strides,
        in_offset,
        idx_dev,
        indices_len,
        axis,
        out_shape,
        out_strides,
        total_output,
        grid_dim=blocks,
        block_dim=tpb,
    )


# =============================================================================
# SECTION 3 — Public GPU gather entry point
# =============================================================================


def gather_gpu[
    dtype: DType
](
    tensor: NDBuffer[dtype],
    axis: Int,
    indices: IntArray,
    reduction: Reduction = Reduction(0),
) raises -> NDBuffer[dtype]:
    """GPU gather with optional fused sum for embedding-bag use case.

    When reduction.is_sum() and the tensor is 2D with axis=0, uses
    embedding_bag_kernel to produce output shape (cols,) directly,
    avoiding the intermediate (n_tokens, cols) allocation.

    Args:
        tensor:   Input NDBuffer on GPU.
        axis:     Axis to gather along (must be 0 for reduction SUM/MEAN).
        indices:  Row indices to gather.
        reduction: How to reduce gathered rows (NONE=0, SUM=1, MEAN=2).
                   SUM/MEAN fuse gather+sum into one kernel.

    Returns:
        NDBuffer on GPU. Shape is (len(indices), ...) normally,
        or (cols,) when reduction is SUM or MEAN.
    """
    # Match DeviceState's internal storage type — bool is stored as uint8
    comptime datatype: DType = DType.uint8 if dtype == DType.bool else dtype

    var rank = tensor.shape.rank()
    if rank > 8:
        panic("gather_gpu: rank ", String(rank), " > 8 not supported")

    var n_indices = len(indices)

    ref ds = tensor.device_state.value()
    ref gpu = ds.get_gpu()
    var ctx = gpu[]

    # ── Build idx_dev — shared by all paths ───────────────────────────────────
    var idx_dev = ctx.enqueue_create_buffer[DType.int64](n_indices)
    with idx_dev.map_to_host() as host_idx:
        for k in range(n_indices):
            host_idx[k] = Int64(indices[k])

    var in_dev = ds.buffer

    # ── Fused embedding-bag path: gather + sum in one kernel ──────────────────
    # Conditions: caller requests it, rank==2, axis==0.
    # Output shape: (in_cols,) — the sum over gathered rows.
    if (reduction.is_sum() or reduction.is_mean()) and rank == 2 and axis == 0:
        var in_cols = tensor.shape[1]
        var block_cols = _gather_2d_block_cols(in_cols)
        var out_dev = ctx.enqueue_create_buffer[datatype](in_cols)
        if reduction.is_mean():
            var compiled = ctx.compile_function[
                embedding_bag_kernel[datatype, True],
                embedding_bag_kernel[datatype, True],
            ]()
            ctx.enqueue_function(
                compiled,
                out_dev,
                in_dev,
                tensor.shape[0],
                in_cols,
                tensor.strides[0],
                idx_dev,
                n_indices,
                grid_dim=1,
                block_dim=block_cols,
            )
        else:
            var compiled = ctx.compile_function[
                embedding_bag_kernel[datatype, False],
                embedding_bag_kernel[datatype, False],
            ]()
            ctx.enqueue_function(
                compiled,
                out_dev,
                in_dev,
                tensor.shape[0],
                in_cols,
                tensor.strides[0],
                idx_dev,
                n_indices,
                grid_dim=1,
                block_dim=block_cols,
            )
        ctx.synchronize()
        var out_shape = Shape(in_cols)
        var result_state = DeviceState[dtype].__init__[special=True](
            out_dev^, gpu
        )
        return NDBuffer[dtype].with_device_state(result_state^, out_shape)

    # ── Standard gather path ──────────────────────────────────────────────────
    var out_shape_arr = IntArray.with_capacity(rank)
    for d in range(rank):
        out_shape_arr.append(n_indices if d == axis else tensor.shape[d])
    var out_shape = Shape(out_shape_arr)
    var out_strides = Strides.default(out_shape)
    var total_output = out_shape.num_elements()

    var out_dev = ctx.enqueue_create_buffer[datatype](total_output)

    if rank == 2 and axis == 0 and tensor.shape[1] <= 512:
        var in_cols = tensor.shape[1]
        var block_cols = _gather_2d_block_cols(in_cols)
        var compiled = ctx.compile_function[
            gather_rows_2d_kernel[datatype],
            gather_rows_2d_kernel[datatype],
        ]()
        ctx.enqueue_function(
            compiled,
            out_dev,
            in_dev,
            tensor.shape[0],
            in_cols,
            tensor.strides[0],
            idx_dev,
            n_indices,
            out_strides[0],
            grid_dim=n_indices,
            block_dim=block_cols,
        )
    elif rank == 1:
        _launch_gather_generic[datatype, 1](
            ctx,
            out_dev,
            in_dev,
            tensor.shape.array(),
            tensor.strides.array(),
            tensor.offset,
            idx_dev,
            n_indices,
            axis,
            out_shape.array(),
            out_strides.array(),
            total_output,
        )
    elif rank == 2:
        _launch_gather_generic[datatype, 2](
            ctx,
            out_dev,
            in_dev,
            tensor.shape.array(),
            tensor.strides.array(),
            tensor.offset,
            idx_dev,
            n_indices,
            axis,
            out_shape.array(),
            out_strides.array(),
            total_output,
        )
    elif rank == 3:
        _launch_gather_generic[datatype, 3](
            ctx,
            out_dev,
            in_dev,
            tensor.shape.array(),
            tensor.strides.array(),
            tensor.offset,
            idx_dev,
            n_indices,
            axis,
            out_shape.array(),
            out_strides.array(),
            total_output,
        )
    elif rank == 4:
        _launch_gather_generic[datatype, 4](
            ctx,
            out_dev,
            in_dev,
            tensor.shape.array(),
            tensor.strides.array(),
            tensor.offset,
            idx_dev,
            n_indices,
            axis,
            out_shape.array(),
            out_strides.array(),
            total_output,
        )
    elif rank == 5:
        _launch_gather_generic[datatype, 5](
            ctx,
            out_dev,
            in_dev,
            tensor.shape.array(),
            tensor.strides.array(),
            tensor.offset,
            idx_dev,
            n_indices,
            axis,
            out_shape.array(),
            out_strides.array(),
            total_output,
        )
    elif rank == 6:
        _launch_gather_generic[datatype, 6](
            ctx,
            out_dev,
            in_dev,
            tensor.shape.array(),
            tensor.strides.array(),
            tensor.offset,
            idx_dev,
            n_indices,
            axis,
            out_shape.array(),
            out_strides.array(),
            total_output,
        )
    elif rank == 7:
        _launch_gather_generic[datatype, 7](
            ctx,
            out_dev,
            in_dev,
            tensor.shape.array(),
            tensor.strides.array(),
            tensor.offset,
            idx_dev,
            n_indices,
            axis,
            out_shape.array(),
            out_strides.array(),
            total_output,
        )
    else:
        _launch_gather_generic[datatype, 8](
            ctx,
            out_dev,
            in_dev,
            tensor.shape.array(),
            tensor.strides.array(),
            tensor.offset,
            idx_dev,
            n_indices,
            axis,
            out_shape.array(),
            out_strides.array(),
            total_output,
        )

    ctx.synchronize()
    var result_state = DeviceState[dtype].__init__[special=True](out_dev^, gpu)
    return NDBuffer[dtype].with_device_state(result_state^, out_shape)


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
        output: Ancestor[Self.dtype],
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
        ref incoming_grad = output.gradbox.unsafe_value()[]
        ref bwd_arg = output.ancestry().backward_fn_arg().get[GatherArg]()

        var extra_arg = output.ancestry().backward_fn_arg()

        if bwd_arg.reduction.is_mean():
            var n = Scalar[Self.dtype](len(bwd_arg.indices))
            parent.update_grad(incoming_grad / n, ScatterAddTensor, extra_arg)
        else:
            parent.update_grad(incoming_grad, ScatterAddTensor, extra_arg)

        parent_ids.append(parent._id)
        if not retain_graph:
            output.gradbox.unsafe_value()[].zero_grad()


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
                        var ndb = gather_gpu[Self.dtype](
                            self.buffer, ax, normalized, reduction
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
                    var ndb = gather_gpu[Self.dtype](
                        self.buffer, ax, normalized, Reduction(2)
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
