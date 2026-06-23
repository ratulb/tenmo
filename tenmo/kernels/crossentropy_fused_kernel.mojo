"""Fused CrossEntropy (class indices) GPU kernel + launcher.

Forward kernel computes softmax AND per-sample loss (with label smoothing
and ignore_index) in a single GPU pass — eliminating ~18 separate kernel
launches and the CPU-fallback onehot loop.

Grid: M blocks (one per row)
Block: power-of-2 ≤ min(C, MAX_BLOCK_SIZE) threads
Shared memory: MAX_BLOCK_SIZE × sizeof(dtype) — reused for max and sum reduction

Phases:
  1. Find max along C (tree reduction in shared memory)
  2. Compute exp(val - max) + sum(exp) + optional sum(logits) for label smoothing
  3. Normalize softmax, compute per-sample loss, atomic-accumulate scalar loss
"""

from std.gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from std.math import exp, log
from std.atomic import Atomic, Ordering
from std.memory import stack_allocation, AddressSpace
from std.sys import simd_width_of

from tenmo.ndbuffer import NDBuffer
from tenmo.device import DeviceState
from tenmo.common_utils import panic
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from tenmo.shared import Reduction
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE


comptime MAX_BLOCK_SIZE: Int = 256

# Initial value for max reduction — approximately -FLT_MAX
# Idle threads (tid >= C) contribute this value so active threads' values win.
comptime MAX_INIT: Float64 = -3.402823466e38


def fused_ce_class_indices_forward_kernel[
    dtype: DType,
    target_dtype: DType = DEFAULT_INDEX_DTYPE,
    max_block_size: Int = MAX_BLOCK_SIZE,
](
    logits: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    target: UnsafePointer[Scalar[target_dtype], ImmutAnyOrigin],
    softmax_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    per_sample_loss: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    scalar_loss: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    valid_count_out: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    M: Int,
    C: Int,
    ignore_index: Int,
    label_smoothing: Scalar[dtype],
) where dtype.is_floating_point():
    var row = block_idx.x
    if row >= M:
        return

    var tid = thread_idx.x
    var block = block_dim.x
    var base = row * C

    # ── Phase 1: Find max along C ──
    var local_max = Scalar[dtype](MAX_INIT)
    if tid < C:
        for c in range(tid, C, block):
            local_max = max(local_max, logits[base + c])

    # Tree-reduce max in shared memory
    var smem = stack_allocation[
        max_block_size,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()
    smem[tid] = local_max
    barrier()

    var stride = block // 2
    while stride > 0:
        if tid < stride:
            smem[tid] = max(smem[tid], smem[tid + stride])
        barrier()
        stride //= 2

    var max_val = smem[0]
    barrier()

    # ── Phase 2: Compute exp + sum_exp + optional sum_logits ──
    var local_sum_exp = Scalar[dtype](0)
    var local_sum_logits = Scalar[dtype](0)
    var has_ls = label_smoothing > Scalar[dtype](0)

    if tid < C:
        for c in range(tid, C, block):
            var val = logits[base + c]
            var e = exp(val - max_val)
            softmax_out[base + c] = e  # raw exp — normalized in Phase 3
            local_sum_exp += e
            if has_ls:
                local_sum_logits += val

    # Tree-reduce sum_exp
    smem[tid] = local_sum_exp
    barrier()

    stride = block // 2
    while stride > 0:
        if tid < stride:
            smem[tid] += smem[tid + stride]
        barrier()
        stride //= 2

    var sum_exp = smem[0]
    var log_sum_exp = log(sum_exp)
    barrier()

    # If label smoothing: tree-reduce sum_logits
    var sum_logits = Scalar[dtype](0)
    if has_ls:
        smem[tid] = local_sum_logits
        barrier()
        stride = block // 2
        while stride > 0:
            if tid < stride:
                smem[tid] += smem[tid + stride]
            barrier()
            stride //= 2
        sum_logits = smem[0]
        barrier()

    # ── Phase 3: Normalize softmax + compute loss ──
    var inv_sum_exp = Scalar[dtype](1) / sum_exp
    if tid < C:
        for c in range(tid, C, block):
            softmax_out[base + c] = softmax_out[base + c] * inv_sum_exp

    # Thread 0 computes per-sample loss and atomics
    if tid == 0:
        var tgt = target[row]
        var is_valid = tgt != Scalar[target_dtype](ignore_index)
        var loss = Scalar[dtype](0)

        if is_valid:
            var logit_tgt = logits[base + tgt.__int__()]
            var log_softmax_tgt = (logit_tgt - max_val) - log_sum_exp
            loss = -log_softmax_tgt

            if has_ls:
                var inv_C = Scalar[dtype](1) / Scalar[dtype](C)
                var mean_log_softmax = (
                    sum_logits * inv_C - max_val - log_sum_exp
                )
                loss = (
                    Scalar[dtype](1) - label_smoothing
                ) * loss - label_smoothing * mean_log_softmax

        per_sample_loss[row] = loss
        _ = Atomic.fetch_add(scalar_loss, loss)
        if is_valid:
            _ = Atomic.fetch_add(valid_count_out, Scalar[DType.int32](1))


def onehot_fill_kernel[
    dtype: DType,
    target_dtype: DType = DEFAULT_INDEX_DTYPE,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    indices: UnsafePointer[Scalar[target_dtype], ImmutAnyOrigin],
    M: Int,
    C: Int,
    ignore_index: Int,
):
    """Fill result[row * C + target[row]] = 1 for each valid row.

    One block per row — only thread 0 does work per block.
    Rows where target[row] == ignore_index are skipped (left as zeros).
    """
    var row = block_idx.x
    if row >= M:
        return
    var tgt = indices[row]
    if tgt == Scalar[target_dtype](ignore_index):
        return
    var c = tgt.__int__()
    if 0 <= c < C:
        result[row * C + c] = Scalar[dtype](1)


# ── Fused Backward Kernel ──────────────────────────────────────────────────────


def fused_ce_class_indices_backward_kernel[
    dtype: DType,
    target_dtype: DType = DEFAULT_INDEX_DTYPE,
    max_block_size: Int = MAX_BLOCK_SIZE,
](
    softmax: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    target: UnsafePointer[Scalar[target_dtype], ImmutAnyOrigin],
    upstream: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    grad_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    M: Int,
    C: Int,
    ignore_index: Int,
    label_smoothing: Scalar[dtype],
    reduction: Int,
    valid_count: Int,
) where dtype.is_floating_point():
    """One block per row. Computes the full CE class-indices backward in one pass.

    Formula:
      grad[m,c] = (softmax[m,c] - onehot[m,c] - ls_adjustment)
                  * ignore_mask[m] * upstream_scaled[m]

    reduction: 0=none, 1=sum, 2=mean
    - none: upstream is (M,) — each row reads its own upstream[row]
    - sum/mean: upstream is scalar — all rows read upstream[0]
      mean further divides by valid_count
    """
    var row = block_idx.x
    if row >= M:
        return

    var tid = thread_idx.x
    var block = block_dim.x
    var base = row * C

    # Get upstream value for this row
    var up_val: Scalar[dtype]
    if reduction == 0:  # none
        up_val = upstream[row]
    else:
        up_val = upstream[0]

    # Apply reduction scaling
    var scale: Scalar[dtype]
    if reduction == 2:  # mean
        var safe_count = valid_count if valid_count > 0 else 1
        scale = up_val / Scalar[dtype](safe_count)
    else:
        scale = up_val

    # Determine target class and validity for this row
    var tgt_scalar = target[row]
    var is_valid = tgt_scalar != Scalar[target_dtype](ignore_index)
    var tgt_class: Int
    if is_valid:
        tgt_class = tgt_scalar.__int__()
    else:
        tgt_class = -1

    var has_ls = label_smoothing > Scalar[dtype](0)
    var inv_C = Scalar[dtype](1) / Scalar[dtype](C)
    var ls_uniform = label_smoothing * inv_C

    if tid < C:
        for c in range(tid, C, block):
            var idx = base + c
            var g: Scalar[dtype]

            if is_valid:
                g = softmax[idx]
                if c == tgt_class:
                    # Subtract onehot
                    g = g - Scalar[dtype](1)
                    if has_ls:
                        # Restore ls (we subtracted 1 instead of 1-ls)
                        g = g + label_smoothing

                if has_ls:
                    # Subtract uniform smoothing term ls/C
                    g = g - ls_uniform
            else:
                g = Scalar[dtype](0)

            grad_out[idx] = g * scale


# ── Launcher ──────────────────────────────────────────────────────────────────


struct CrossEntropyFusedKernel[
    dtype: DType, target_dtype: DType = DEFAULT_INDEX_DTYPE
](ImplicitlyCopyable & Movable):
    @staticmethod
    def launch(
        logits_2d: NDBuffer[Self.dtype],
        target_1d: NDBuffer[Self.target_dtype],
        reduction: Reduction,
        ignore_index: Int,
        label_smoothing: Scalar[Self.dtype],
    ) raises -> Tuple[
        NDBuffer[Self.dtype],  # softmax_probs (M, C)
        NDBuffer[Self.dtype],  # per_sample_loss (M,)
        Scalar[Self.dtype],  # scalar_loss (sum of all per-sample losses)
        Int,  # valid_count (non-ignored rows)
    ] where Self.dtype.is_floating_point():
        debug_assert(logits_2d.is_on_gpu())
        debug_assert(target_1d.is_on_gpu())

        var M = logits_2d.shape[0]
        var C = logits_2d.shape[1]
        var numels = M * C

        ref device_state = logits_2d.device_state.value()
        var device_context = device_state.gpu[]

        # Make contiguous copies for kernel (no-op if already contiguous)
        var contig_logits = logits_2d.contiguous_device_state()
        var contig_target = target_1d.contiguous_device_state()

        # Allocate output buffers
        var softmax_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )
        var loss_buffer = device_context.enqueue_create_buffer[Self.dtype](M)
        var scalar_buffer = device_context.enqueue_create_buffer[Self.dtype](1)
        var valid_buffer = device_context.enqueue_create_buffer[DType.int32](1)
        scalar_buffer.enqueue_fill(0)
        valid_buffer.enqueue_fill(0)

        # Launch config: power-of-2 block size, one block per row
        var block_size = min(C, MAX_BLOCK_SIZE)
        var block_pow2 = 1
        while block_pow2 * 2 <= block_size:
            block_pow2 *= 2
        block_size = block_pow2
        var num_blocks = M

        # Compile and enqueue kernel
        var compiled = device_context.compile_function[
            fused_ce_class_indices_forward_kernel[
                dtype=Self.dtype,
                target_dtype=Self.target_dtype,
                max_block_size=MAX_BLOCK_SIZE,
            ],
            fused_ce_class_indices_forward_kernel[
                dtype=Self.dtype,
                target_dtype=Self.target_dtype,
                max_block_size=MAX_BLOCK_SIZE,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            contig_logits.device_buffer(),
            contig_target.device_buffer(),
            softmax_buffer,
            loss_buffer,
            scalar_buffer,
            valid_buffer,
            M,
            C,
            ignore_index,
            label_smoothing,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        # Sync to read back scalar results (replaces existing .item() synced path)
        device_context.synchronize()

        # Read back scalar_loss and valid_count from GPU
        var scalar_state = DeviceState[Self.dtype](
            scalar_buffer^, device_state.gpu
        )
        var valid_state = DeviceState[DType.int32](
            valid_buffer^, device_state.gpu
        )

        var scalar_val: Scalar[Self.dtype]
        var valid_cnt: Int
        with scalar_state.buffer.map_to_host() as host_scalar:
            scalar_val = rebind[Scalar[Self.dtype]](host_scalar[0])
        with valid_state.buffer.map_to_host() as host_valid:
            valid_cnt = host_valid[0].__int__()

        # Wrap results as NDBuffers
        var softmax_state = DeviceState[Self.dtype](
            softmax_buffer^, device_state.gpu
        )
        var loss_state = DeviceState[Self.dtype](loss_buffer^, device_state.gpu)

        var softmax_ndb = NDBuffer[Self.dtype].with_device_state(
            softmax_state^, logits_2d.shape
        )
        var loss_ndb = NDBuffer[Self.dtype].with_device_state(
            loss_state^, Shape(M)
        )

        return (softmax_ndb^, loss_ndb^, scalar_val, valid_cnt)

    @staticmethod
    def launch_onehot(
        target_1d: NDBuffer[Self.target_dtype],
        num_classes: Int,
        ignore_index: Int,
    ) raises -> NDBuffer[Self.dtype]:
        """Fill a (M, C) GPU buffer with onehot encoding of target_1d.

        One block per row. Each block sets element (row, target[row]) to 1.
        Rows where target[row] == ignore_index stay all zeros.
        """
        debug_assert(target_1d.is_on_gpu())
        var M = target_1d.shape[0]
        var C = num_classes

        ref device_state = target_1d.device_state.value()
        var ctx = device_state.gpu[]

        var contig_target = target_1d.contiguous_device_state()

        var result_buffer = ctx.enqueue_create_buffer[Self.dtype](M * C)
        result_buffer.enqueue_fill(0)

        var compiled = ctx.compile_function[
            onehot_fill_kernel[Self.dtype, Self.target_dtype],
            onehot_fill_kernel[Self.dtype, Self.target_dtype],
        ]()

        ctx.enqueue_function(
            compiled,
            result_buffer,
            contig_target.device_buffer(),
            M,
            C,
            ignore_index,
            grid_dim=M,
            block_dim=1,
        )

        var result_state = DeviceState[Self.dtype](
            result_buffer^, device_state.gpu
        )
        return NDBuffer[Self.dtype].with_device_state(
            result_state^, Shape(M, C)
        )

    @staticmethod
    def launch_backward(
        softmax_ndb: NDBuffer[Self.dtype],
        target_ndb: NDBuffer[Self.target_dtype],
        upstream_ndb: NDBuffer[Self.dtype],
        reduction: Reduction,
        valid_count: Int,
        M: Int,
        C: Int,
        ignore_index: Int,
        label_smoothing: Scalar[Self.dtype],
    ) raises -> NDBuffer[Self.dtype]:
        """Fused backward: onehot + smoothing + ignore_mask + scaling in one launch.

        Args:
            softmax_ndb: (M, C) — softmax probabilities on GPU
            target_ndb:  (M,) — class indices on GPU
            upstream_ndb: upstream gradient buffer
                For none reduction: shape (M,) — must be on GPU
                For sum/mean reduction: shape () — can be CPU, auto-transferred
            reduction: none/sum/mean enum
            valid_count: number of non-ignored rows (for mean scaling)

        Returns:
            (M, C) NDBuffer with the full gradient w.r.t. logits
        """
        debug_assert(softmax_ndb.is_on_gpu())
        debug_assert(target_ndb.is_on_gpu())

        ref device_state = softmax_ndb.device_state.value()
        var ctx = device_state.gpu[]

        # Ensure upstream is on GPU (mean/sum upstream is a CPU scalar)
        var upstream_gpu: NDBuffer[Self.dtype]
        if upstream_ndb.is_on_gpu():
            upstream_gpu = upstream_ndb
        else:
            # Transfer CPU scalar to GPU — single-element buffer
            var cpu_val = upstream_ndb.buffer[0]
            var ubuf = ctx.enqueue_create_buffer[Self.dtype](1)
            ubuf.enqueue_fill(cpu_val)
            var ust = DeviceState[Self.dtype](ubuf^, device_state.gpu)
            upstream_gpu = NDBuffer[Self.dtype].with_device_state(ust^, Shape())

        var numels = M * C
        var result_buffer = ctx.enqueue_create_buffer[Self.dtype](numels)

        # Launch config: power-of-2 block size, one block per row
        var block_size = min(C, MAX_BLOCK_SIZE)
        var block_pow2 = 1
        while block_pow2 * 2 <= block_size:
            block_pow2 *= 2
        block_size = block_pow2
        var num_blocks = M

        var compiled = ctx.compile_function[
            fused_ce_class_indices_backward_kernel[
                dtype=Self.dtype,
                target_dtype=Self.target_dtype,
                max_block_size=MAX_BLOCK_SIZE,
            ],
            fused_ce_class_indices_backward_kernel[
                dtype=Self.dtype,
                target_dtype=Self.target_dtype,
                max_block_size=MAX_BLOCK_SIZE,
            ],
        ]()

        # Convert reduction to integer for kernel dispatch
        var reduction_int: Int
        if reduction.is_none():
            reduction_int = 0
        elif reduction.is_sum():
            reduction_int = 1
        else:
            reduction_int = 2

        ctx.enqueue_function(
            compiled,
            softmax_ndb.contiguous_device_state().device_buffer(),
            target_ndb.contiguous_device_state().device_buffer(),
            upstream_gpu.contiguous_device_state().device_buffer(),
            result_buffer,
            M,
            C,
            ignore_index,
            label_smoothing,
            reduction_int,
            valid_count,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        var result_state = DeviceState[Self.dtype](
            result_buffer^, device_state.gpu
        )
        return NDBuffer[Self.dtype].with_device_state(
            result_state^, Shape(M, C)
        )
