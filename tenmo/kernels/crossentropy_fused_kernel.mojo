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

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from max.gpu import barrier
from std.math import exp, log
from std.atomic import Atomic
from std.memory import stack_allocation, AddressSpace

from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from ..gpu.device import DeviceState
from ..shared.panic import panic
from ..shared.shapes import Shape
from ..shared import Reduction
from ..shared.mnemonics import DEFAULT_INDEX_DTYPE


comptime MAX_BLOCK_SIZE: Int = 256

# Initial value for max reduction — approximately -FLT_MAX
# Idle threads (tid >= C) contribute this value so active threads' values win.
comptime MAX_INIT: Float64 = -3.402823466e38


def fused_ce_class_indices_forward_kernel[
    dtype: DType,
    target_dtype: DType = DEFAULT_INDEX_DTYPE,
    max_block_size: Int = MAX_BLOCK_SIZE,
](
    logits: Pointer[Scalar[dtype], ImmutAnyOrigin],
    target: Pointer[Scalar[target_dtype], ImmutAnyOrigin],
    softmax_out: Pointer[Scalar[dtype], MutAnyOrigin],
    per_sample_loss: Pointer[Scalar[dtype], MutAnyOrigin],
    scalar_loss: Pointer[Scalar[dtype], MutAnyOrigin],
    valid_count_out: Pointer[Scalar[DType.int32], MutAnyOrigin],
    M_: Int64,
    C_: Int64,
    ignore_index_: Int64,
    label_smoothing: Scalar[dtype],
) where dtype.is_floating_point():
    var M = Int(M_)
    var C = Int(C_)
    var ignore_index = Int(ignore_index_)
    var row = block_idx.x
    if row >= M:
        return

    var tid = thread_idx.x
    var block = block_dim.x
    var base = row * C

    # Phase 1: Find max along C
    var local_max = Scalar[dtype](MAX_INIT)
    if tid < C:
        for c in range(tid, C, block):
            local_max = max(local_max, logits[unsafe_offset=base + c])

    # Tree-reduce max in shared memory
    var smem = stack_allocation[
        max_block_size,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()
    smem[unsafe_offset=tid] = local_max
    barrier()

    var stride = block // 2
    while stride > 0:
        if tid < stride:
            smem[unsafe_offset=tid] = max(smem[unsafe_offset=tid], smem[unsafe_offset=tid + stride])
        barrier()
        stride //= 2

    var max_val = smem[unsafe_offset=0]
    barrier()

    # Phase 2: Compute exp + sum_exp + optional sum_logits
    var local_sum_exp = Scalar[dtype](0)
    var local_sum_logits = Scalar[dtype](0)
    var has_ls = label_smoothing > Scalar[dtype](0)

    if tid < C:
        for c in range(tid, C, block):
            var val = logits[unsafe_offset=base + c]
            var e = exp(val - max_val)
            softmax_out[unsafe_offset=base + c] = e  # raw exp — normalized in Phase 3
            local_sum_exp += e
            if has_ls:
                local_sum_logits += val

    # Tree-reduce sum_exp
    smem[unsafe_offset=tid] = local_sum_exp
    barrier()

    stride = block // 2
    while stride > 0:
        if tid < stride:
            smem[unsafe_offset=tid] += smem[unsafe_offset=tid + stride]
        barrier()
        stride //= 2

    var sum_exp = smem[unsafe_offset=0]
    var log_sum_exp = log(sum_exp)
    barrier()

    # If label smoothing: tree-reduce sum_logits
    var sum_logits = Scalar[dtype](0)
    if has_ls:
        smem[unsafe_offset=tid] = local_sum_logits
        barrier()
        stride = block // 2
        while stride > 0:
            if tid < stride:
                smem[unsafe_offset=tid] += smem[unsafe_offset=tid + stride]
            barrier()
            stride //= 2
        sum_logits = smem[unsafe_offset=0]
        barrier()

    # Phase 3: Normalize softmax + compute loss
    var inv_sum_exp = Scalar[dtype](1) / sum_exp
    if tid < C:
        for c in range(tid, C, block):
            softmax_out[unsafe_offset=base + c] = softmax_out[unsafe_offset=base + c] * inv_sum_exp

    # Thread 0 computes per-sample loss and atomics
    if tid == 0:
        var tgt = target[unsafe_offset=row]
        var is_valid = tgt != Scalar[target_dtype](ignore_index)
        var loss = Scalar[dtype](0)

        if is_valid:
            var logit_tgt = logits[unsafe_offset=base + tgt.__int__()]
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

        per_sample_loss[unsafe_offset=row] = loss
        _ = Atomic.fetch_add(scalar_loss, loss)
        if is_valid:
            _ = Atomic.fetch_add(valid_count_out, Scalar[DType.int32](1))


# Fused Backward Kernel


def fused_ce_class_indices_backward_kernel[
    dtype: DType,
    target_dtype: DType = DEFAULT_INDEX_DTYPE,
    max_block_size: Int = MAX_BLOCK_SIZE,
](
    softmax: Pointer[Scalar[dtype], ImmutAnyOrigin],
    target: Pointer[Scalar[target_dtype], ImmutAnyOrigin],
    upstream: Pointer[Scalar[dtype], ImmutAnyOrigin],
    grad_out: Pointer[Scalar[dtype], MutAnyOrigin],
    M_: Int64,
    C_: Int64,
    ignore_index_: Int64,
    label_smoothing: Scalar[dtype],
    reduction_: Int64,
    valid_count_: Int64,
) where dtype.is_floating_point():
    """One block per row. Computes the full CE class-indices backward in one pass.

    Formula:
      grad[m,c] = (softmax[m,c] - onehot[m,c] - ls_adjustment)
                  * ignore_mask[m] * upstream_scaled[m]

    reduction: 0=none, 1=sum, 2=mean
    - none: upstream is (M,)
    - sum/mean: upstream is scalar
      mean further divides by valid_count
    """
    var M = Int(M_)
    var C = Int(C_)
    var ignore_index = Int(ignore_index_)
    var reduction = Int(reduction_)
    var valid_count = Int(valid_count_)
    comptime if not dtype.is_floating_point():
        panic(
            "fused_ce_class_indices_backward_kernel requires a floating point"
            " dtype"
        )
    var row = block_idx.x
    if row >= M:
        return

    var tid = thread_idx.x
    var block = block_dim.x
    var base = row * C

    # Get upstream value for this row
    var up_val: Scalar[dtype]
    if reduction == 0:  # none
        up_val = upstream[unsafe_offset=row]
    else:
        up_val = upstream[unsafe_offset=0]

    # Apply reduction scaling
    var scale: Scalar[dtype]
    if reduction == 2:  # mean
        var safe_count = valid_count if valid_count > 0 else 1
        scale = up_val / Scalar[dtype](safe_count)
    else:
        scale = up_val

    # Determine target class and validity for this row
    var tgt_scalar = target[unsafe_offset=row]
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
                g = softmax[unsafe_offset=idx]
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

            grad_out[unsafe_offset=idx] = g * scale


# Launcher


struct CrossEntropyFusedKernel[
    dtype: DType, target_dtype: DType = DEFAULT_INDEX_DTYPE
](ImplicitlyCopyable):
    @staticmethod
    def launch(
        logits_2d_layout: Layout,
        logits_2d_device_state: DeviceState[Self.dtype],
        target_1d_layout: Layout,
        target_1d_device_state: DeviceState[Self.target_dtype],
        reduction: Reduction,
        ignore_index: Int,
        label_smoothing: Scalar[Self.dtype],
    ) raises -> Tuple[
        Tuple[Layout, DeviceState[Self.dtype]],  # softmax_probs (M, C)
        Tuple[Layout, DeviceState[Self.dtype]],  # per_sample_loss (M,)
        Scalar[Self.dtype],  # scalar_loss (sum of all per-sample losses)
        Int,  # valid_count (non-ignored rows)
    ] where Self.dtype.is_floating_point():

        var M = logits_2d_layout.shape[0]
        var C = logits_2d_layout.shape[1]
        var numels = M * C

        ref gpu = logits_2d_device_state.get_gpu()
        var device_context = gpu[]

        # Make contiguous copies for kernel (no-op if already contiguous)
        var contig_logits = materialize_contiguous(
            logits_2d_device_state, logits_2d_layout
        )
        var contig_target = materialize_contiguous(
            target_1d_device_state, target_1d_layout
        )

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
        ]()

        device_context.enqueue_function(
            compiled,
            contig_logits.device_buffer(),
            contig_target.device_buffer(),
            softmax_buffer,
            loss_buffer,
            scalar_buffer,
            valid_buffer,
            Int64(M),
            Int64(C),
            Int64(ignore_index),
            label_smoothing,
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        # Sync to read back scalar results (replaces existing .item() synced path)
        device_context.synchronize()

        # Read back scalar_loss and valid_count from GPU
        var scalar_state = DeviceState[Self.dtype](
            scalar_buffer^, gpu
        )
        var valid_state = DeviceState[DType.int32](
            valid_buffer^, gpu
        )

        var scalar_val: Scalar[Self.dtype]
        var valid_cnt: Int
        with scalar_state.buffer.map_to_host() as host_scalar:
            scalar_val = rebind[Scalar[Self.dtype]](host_scalar[0])
        with valid_state.buffer.map_to_host() as host_valid:
            valid_cnt = host_valid[0].__int__()

        # Wrap results as (Layout, Storage) pairs
        var softmax_state = DeviceState[Self.dtype](
            softmax_buffer^, gpu
        )
        var loss_state = DeviceState[Self.dtype](loss_buffer^, gpu)

        var softmax_pair = (
            Layout(logits_2d_layout.shape),
            softmax_state^,
        )
        var loss_pair = (
            Layout(Shape(M)),
            loss_state^,
        )

        return (softmax_pair, loss_pair, scalar_val, valid_cnt)

    @staticmethod
    def launch_backward(
        softmax_layout: Layout,
        softmax_device_state: DeviceState[Self.dtype],
        target_layout: Layout,
        target_device_state: DeviceState[Self.target_dtype],
        upstream_layout: Layout,
        upstream_device_state: DeviceState[Self.dtype],
        reduction: Reduction,
        valid_count: Int,
        M: Int,
        C: Int,
        ignore_index: Int,
        label_smoothing: Scalar[Self.dtype],
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        """Fused backward: onehot + smoothing + ignore_mask + scaling in one launch.

        Args:
            softmax_layout: Layout of the softmax probabilities.
            softmax_device_state: (M, C) softmax probabilities on GPU.
            target_layout: Layout of the target tensor.
            target_device_state: (M,) class indices on GPU.
            upstream_layout: Layout of the upstream gradient.
            upstream_device_state: Upstream gradient.
                For none reduction: shape (M,) — must be on GPU.
                For sum/mean reduction: shape () — can be CPU, auto-transferred.
            reduction:     None/sum/mean enum.
            valid_count:   Number of non-ignored rows (for mean scaling).
            M:             Number of rows.
            C:             Number of classes.
            ignore_index:  Class index to ignore in loss.
            label_smoothing: Label smoothing factor.

        Returns:
            ((M, C) Layout, DeviceState) pair with the full gradient w.r.t. logits.
        """

        ref gpu = softmax_device_state.get_gpu()
        var ctx = gpu[]

        var contig_softmax = materialize_contiguous(
            softmax_device_state, softmax_layout
        )
        var contig_target = materialize_contiguous(
            target_device_state, target_layout
        )

        # Ensure upstream is contiguous on GPU
        var contig_upstream = materialize_contiguous(
            upstream_device_state, upstream_layout
        )

        var numels = M * C
        var result_buffer = ctx.enqueue_create_buffer[Self.dtype](numels)

        # Launch config: power-of-2 block size, one block per row
        var block_size = min(C, MAX_BLOCK_SIZE)
        var block_pow2 = 1
        while block_pow2 * 2 <= block_size:
            block_pow2 *= 2
        block_size = block_pow2
        var num_blocks = M

        comptime if Self.dtype.is_floating_point():
            var compiled = ctx.compile_function[
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
                contig_softmax.device_buffer(),
                contig_target.device_buffer(),
                contig_upstream.device_buffer(),
                result_buffer,
                Int64(M),
                Int64(C),
                Int64(ignore_index),
                label_smoothing,
                Int64(reduction_int),
                Int64(valid_count),
                grid_dim=num_blocks,
                block_dim=block_size,
            )
        else:
            panic(
                "CrossEntropyFusedKernel.launch_backward: "
                "requires a floating point dtype"
            )

        var result_state = DeviceState[Self.dtype](
            result_buffer^, gpu
        )
        return (
            Layout(Shape(M, C)),
            result_state^,
        )
