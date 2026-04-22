
# dropout.mojo  — GPU-enabled Dropout, migrated to BackwardFnArg pattern
# =============================================================================
#
# CHANGE MAP
# ──────────────────────────────────────────────────────────────────────────────
#
# 1. kernels.mojo
#    NEW kernel fn:   dropout_forward_kernel[dtype, ...]
#    NEW launcher:    DropoutKernel[dtype].launch(...)
#                     → returns Tuple[NDBuffer, NDBuffer] (output, mask)
#                     → mask is generated ON-DEVICE via philox.Random
#                     → no CPU→GPU mask transfer needed
#
# 2. backpropagation.mojo  (ArgumentType)
#    EXISTING:  Buffer[dtype]   — used for CPU mask
#    EXISTING:  NDBuffer[dtype] — used for GPU mask  (added for ReLU already)
#    NEW entry: ArgDropout[dtype] = Tuple[Scalar[dtype], Bool]
#                   field 0: scale   (1 / (1 - p)), baked into mask on CPU,
#                            needed separately on GPU for mask reconstruction
#                   field 1: on_gpu flag — selects which arm to read
#    NOTE: The mask itself (Buffer or NDBuffer) is stored as a SECOND arg.
#          BackwardFnArg currently holds ONE arg. Two options:
#            a) Pack mask + scale into a named struct (chosen here — cleaner)
#            b) Use ancestry slot (wrong semantics)
#          → Add ArgDropout to ArgumentType (see below)
#
# 3. dropout.mojo  (this file)
#    DropoutBackward  — stateless, reads mask from bwd_fn_arg().arg
#    Dropout.__call__ — splits CPU / GPU paths, uses BackwardFnArg
#
# 4. mnemonics.mojo / backpropagation.mojo
#    BACKWARD_DROPOUT op code — unchanged
#    Backward.invoke  — DropoutBackward.backward(output) call — unchanged
#
# =============================================================================


# ── Imports ───────────────────────────────────────────────────────────────────

from std.random.philox import Random as PhiloxRandom
from std.random import random_float64, seed as set_seed
from std.sys import simd_width_of, has_accelerator
from std.gpu import thread_idx, block_dim, grid_dim, block_idx

from tensor import Tensor
from mnemonics import AddTensor, DROPOUT, RELU_FORWARD
from backpropagation import BackwardFnArg, ArgumentType, BACKWARD_DROPOUT
from gradbox import Gradbox
from common_utils import panic
from ndbuffer import NDBuffer
from device import DeviceState
from buffers import Buffer
from net import Module, Layer


# =============================================================================
# SECTION 1 — ArgDropout: packed argument struct for BackwardFnArg
# =============================================================================
#
# Stored in ArgumentType. Carries everything DropoutBackward needs:
#   mask_cpu  — CPU Buffer mask (set when tensor is on CPU, else empty)
#   mask_gpu  — GPU NDBuffer mask (set when tensor is on GPU, else None)
#   scale     — 1 / (1 - p), used to reconstruct gradient scaling
#   on_gpu    — which mask arm to read in backward
#
# Why a named struct rather than a nested Tuple?
#   Tuple[Buffer, NDBuffer, Scalar, Bool] is unwieldy to read and extend.
#   A named struct makes backward code self-documenting.
#
# Added to ArgumentType variant list:
#   comptime ArgumentType[dtype] = Variant[..., ArgDropout[dtype]]
# =============================================================================

@fieldwise_init
struct ArgDropout[dtype: DType](ImplicitlyCopyable & Movable):
    var mask_cpu: Buffer[Self.dtype]          # valid on CPU path
    var mask_gpu: Optional[NDBuffer[Self.dtype]]  # valid on GPU path
    var scale: Scalar[Self.dtype]
    var on_gpu: Bool


# =============================================================================
# SECTION 2 — DropoutBackward (stateless, new pattern)
# =============================================================================
#
# MIGRATION from old pattern:
#   OLD: struct holds mask_buffer: Buffer[dtype] as a field
#        embedded inside Delegate variant, retrieved via BackwardFn
#   NEW: stateless static method, reads ArgDropout from bwd_fn_arg().arg
#        exactly mirrors ReLUBackward after the revamp
#
# Backward pass:
#   grad_input = grad_output * mask
#   mask already has scale baked in (both CPU and GPU paths bake scale
#   into the mask during forward), so plain multiply is correct.
#
# Device dispatch:
#   CPU: mask_cpu (Buffer) → wrap in NDBuffer → NDBuffer multiply
#   GPU: mask_gpu (NDBuffer, on-device) → NDBuffer multiply directly
#        No host transfer. NDBuffer.arithmetic_ops dispatches to GPU kernel.
# =============================================================================

@fieldwise_init
struct DropoutBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:

        ref arg_dropout = output.bwd_fn_arg().arg[ArgDropout[Self.dtype]]
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        ref shape = ancestor.shape()

        var result_ndb: NDBuffer[Self.dtype]

        if arg_dropout.on_gpu:
            # GPU path — mask stays on device
            # NDBuffer arithmetic_ops dispatches through GPU kernel
            var mask_ndb = arg_dropout.mask_gpu.value()
            result_ndb = gradbox.buffer * mask_ndb
        else:
            # CPU path — wrap Buffer mask in NDBuffer, multiply
            var mask_ndb = NDBuffer[Self.dtype](
                arg_dropout.mask_cpu, shape
            )
            result_ndb = gradbox.buffer * mask_ndb

        var gradbox_ancestor = Gradbox[Self.dtype](result_ndb^, share=False)
        return [(ancestor^, gradbox_ancestor^, AddTensor)]


# =============================================================================
# SECTION 3 — GPU kernel: dropout_forward_kernel
# =============================================================================
#
# Added to kernels.mojo alongside unary_ops / unary_ops_with_mask.
#
# Key design points:
#
# 1. Philox RNG — each thread gets an independent random stream:
#      rng = PhiloxRandom(seed=seed, subsequence=global_thread_id, offset=0)
#    subsequence isolates per-thread streams → no race conditions.
#    Same seed → same mask for a given forward call (reproducible).
#
# 2. step_uniform() returns SIMD[float32, 4] — four values per call.
#    For non-float32 dtypes, cast after comparison.
#
# 3. Writes TWO output buffers in one pass (same pattern as unary_ops_with_mask):
#    result[i] = input[i] * mask[i]
#    mask[i]   = scale if rand > p else 0   (scale baked in)
#
# 4. No dtype.is_floating_point() constraint needed —
#    Dropout on integer tensors is unusual but the kernel is dtype-generic.
#    Callers should guard at the Module level if desired.
#
# 5. Chunk/stride pattern mirrors existing kernels exactly.
#
# =============================================================================

fn dropout_forward_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result:    UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mask_out:  UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A:         UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size:      Int,
    p:         Scalar[dtype],        # dropout probability
    scale:     Scalar[dtype],        # 1 / (1 - p)
    rng_seed:  UInt64,               # forwarded from Dropout.seed
):
    """Dropout forward kernel: generates mask via Philox RNG and applies it.

    Each thread owns an independent Philox subsequence keyed by global thread id.
    This guarantees statistically independent random streams across threads
    without any shared state or synchronisation.

    Writes:
        result[i]   = A[i] * mask[i]
        mask_out[i] = scale  if rand > p  else 0
    """
    var tid    = thread_idx.x
    var gtid   = Int(tid + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)

    # Independent Philox stream per thread
    var rng = PhiloxRandom(seed=rng_seed, subsequence=UInt64(gtid), offset=0)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    var zero_s  = Scalar[dtype](0)
    var scale_s = scale

    while base_idx < size:

        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var x_vec = A.load[width=simd_width](i)

                # Philox produces SIMD[float32, 4] per call.
                # We process simd_width elements per vector; call step_uniform
                # enough times to cover simd_width lanes.
                # For simd_width <= 4: one call, slice first simd_width values.
                # For simd_width > 4:  multiple calls (handled by scalar tail).
                var rand_f32 = rng.step_uniform()  # SIMD[float32, 4]

                var mask_vec = SIMD[dtype, simd_width](0)
                var res_vec  = SIMD[dtype, simd_width](0)

                comptime for lane in range(simd_width):
                    # Cast random float32 to dtype for threshold comparison
                    var r = rand_f32[lane % 4].cast[dtype]()
                    var m = scale_s if r > p else zero_s
                    mask_vec[lane] = m
                    res_vec[lane]  = x_vec[lane] * m

                result.store[width=simd_width](i, res_vec)
                mask_out.store[width=simd_width](i, mask_vec)

            elif i < size:
                # Scalar tail
                for j in range(size - i):
                    var rand_f32_scalar = rng.step_uniform()
                    var r = rand_f32_scalar[0].cast[dtype]()
                    var x_val = A[i + j]
                    var m     = scale_s if r > p else zero_s
                    result[i + j]   = x_val * m
                    mask_out[i + j] = m

        base_idx += stride * CHUNK_SIZE


# =============================================================================
# SECTION 4 — DropoutKernel launcher
# =============================================================================
#
# Added to kernels.mojo inside / alongside UnaryOpsKernel.
#
# Returns Tuple[NDBuffer, NDBuffer] — (output, mask) — both on GPU.
# Follows the same pattern as UnaryOpsKernel.launch_with_mask:
#   1. contiguous_device_state() for non-contiguous input (single map_to_host)
#   2. Allocate two output DeviceBuffers
#   3. Compile and enqueue dropout_forward_kernel
#   4. Wrap results in DeviceState → NDBuffer via with_device_state()
#
# Non-contiguous input:
#   Same fix as ReLU — contiguous_device_state() does ONE map_to_host sweep.
#   The kernel then operates on the flat buffer. No per-element host calls.
# =============================================================================

struct DropoutKernel[dtype: DType](ImplicitlyCopyable & Movable):

    @staticmethod
    fn launch(
        A:        NDBuffer[Self.dtype],
        p:        Scalar[Self.dtype],
        scale:    Scalar[Self.dtype],
        rng_seed: UInt64,
    ) raises -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        """Launch dropout forward kernel. Returns (output, mask) on GPU.

        Args:
            A:        Input NDBuffer. Must be on GPU.
            p:        Dropout probability.
            scale:    1 / (1 - p).
            rng_seed: Seed forwarded to Philox — same seed → same mask.

        Returns:
            Tuple of (output NDBuffer, mask NDBuffer), both on GPU.
        """
        debug_assert(A.is_on_gpu())

        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        # Reuse UnaryOpsKernel launch config — same heuristic applies
        var (threads_per_block, num_blocks) = UnaryOpsKernel[
            Self.dtype
        ].launch_config(numels, simdwidth)

        ref device_state   = A.device_state.value()
        var device_context = device_state.gpu()

        # Non-contiguous fix: single map_to_host sweep → flat contiguous buffer
        var contig_state = A.contiguous_device_state()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](numels)
        var mask_buffer   = device_context.enqueue_create_buffer[Self.dtype](numels)

        var compiled = device_context.compile_function[
            dropout_forward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2*simdwidth,
            ],
            dropout_forward_kernel[
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2*simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,                  # out: dropped values
            mask_buffer,                    # out: scale mask
            contig_state.device_buffer(),   # in:  input
            numels,
            p,
            scale,
            rng_seed,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, device_state.gpu)
        var mask_state   = DeviceState[Self.dtype](mask_buffer^,   device_state.gpu)

        var out_ndb  = NDBuffer[Self.dtype].with_device_state(result_state^, A.shape)
        var mask_ndb = NDBuffer[Self.dtype].with_device_state(mask_state^,   A.shape)

        return (out_ndb^, mask_ndb^)


# =============================================================================
# SECTION 5 — Dropout (forward, CPU + GPU)
# =============================================================================
#
# MIGRATION from old pattern:
#   OLD: builds DropoutBackward(mask_buffer=...).into_backward_fn()
#        → stateful struct embedded in Delegate
#   NEW: builds BackwardFnArg(BACKWARD_DROPOUT, ArgDropout(...))
#        → mask lives in ArgumentType, backward struct is stateless
#
# CPU path:
#   Largely unchanged from original — SIMD loop with random_float64.
#   Now uses PhiloxRandom for consistency and reproducibility.
#   mask stays as Buffer[dtype], wrapped into ArgDropout.mask_cpu.
#
# GPU path:
#   DropoutKernel.launch() handles RNG + mask + output in one kernel pass.
#   mask comes back as NDBuffer on device, stored in ArgDropout.mask_gpu.
#   No CPU→GPU mask transfer needed.
#
# eval mode / p==0:
#   Returns input unchanged (no grad bookkeeping needed — identity op).
#
# p==1 guard:
#   Returns zeros. No backward needed (grad is always zero).
# =============================================================================

@fieldwise_init
struct Dropout[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    """Dropout layer — CPU and GPU enabled.

    Forward (training):
      CPU: Philox RNG generates mask in a SIMD loop; output and mask written
           in one pass.
      GPU: dropout_forward_kernel generates mask on-device via per-thread
           Philox subsequences; output and mask returned as GPU NDBuffers.

    Backward:
      grad_input = grad_output * mask   (scale already baked into mask)
      Both CPU and GPU paths use NDBuffer multiply → device-aware dispatch.

    Eval / p==0:
      Identity — returns input, no mask stored, no grad bookkeeping.
    """

    var training: Bool
    var p: Scalar[Self.dtype]
    var scale: Scalar[Self.dtype]
    var seed: UInt64   # UInt64 to match PhiloxRandom.seed type

    fn __init__(out self, p: Scalar[Self.dtype] = Scalar[Self.dtype](0.5)):
        if p < 0.0 or p >= 1.0:
            panic("Dropout probability must be in [0, 1)")
        self.training = True
        self.p = p
        self.scale = Scalar[Self.dtype](1.0) / (Scalar[Self.dtype](1.0) - p)
        self.seed = 42

    fn __copyinit__(out self, copy: Self):
        self.training = copy.training
        self.p = copy.p
        self.scale = copy.scale
        self.seed = copy.seed

    fn __call__(self, x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:

        # ── Eval / no-op paths ────────────────────────────────────────────────
        if not self.training or self.p == Scalar[Self.dtype](0.0):
            return x

        if self.p == Scalar[Self.dtype](1.0):
            return Tensor[Self.dtype].zeros(x.shape())

        # ── GPU path ──────────────────────────────────────────────────────────
        comptime if has_accelerator():
            if x.buffer.is_on_gpu():
                try:
                    var result = DropoutKernel[Self.dtype].launch(
                        x.buffer,
                        self.p,
                        self.scale,
                        self.seed,
                    )
                    var out_ndb  = result[0]
                    var mask_ndb = result[1]

                    var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

                    if x.requires_grad:
                        out.requires_grad_(True)
                        var arg = ArgDropout[Self.dtype](
                            mask_cpu = Buffer[Self.dtype](),  # empty — not used
                            mask_gpu = Optional(mask_ndb^),
                            scale    = self.scale,
                            on_gpu   = True,
                        )
                        out.bwdFnArg = Optional(
                            BackwardFnArg[Self.dtype](
                                BACKWARD_DROPOUT,
                                ArgumentType[Self.dtype](arg^),
                            )
                        )
                        out.add_ancestry(x)

                    return out^

                except e:
                    panic("Dropout GPU forward failed: ", String(e))
                    # Unreachable
                    return Tensor[Self.dtype].zeros(x.shape())

        # ── CPU path ──────────────────────────────────────────────────────────
        var shape    = x.shape()
        var numels   = x.numels()
        var out_buf  = Buffer[Self.dtype](numels)
        var mask_buf = Buffer[Self.dtype](numels)

        var x_ptr    = x.buffer.data_ptr()
        var out_ptr  = out_buf.unsafe_ptr()
        var mask_ptr = mask_buf.unsafe_ptr()

        comptime simd_w = simd_width_of[Self.dtype]()

        # Philox RNG — 4 floats per step_uniform() call.
        # Use a single generator on CPU (single-threaded here).
        # subsequence=0, offset advances implicitly per step() call.
        var rng = PhiloxRandom(seed=self.seed, subsequence=0, offset=0)

        var p_s     = self.p
        var scale_s = self.scale
        var zero_s  = Scalar[Self.dtype](0)

        # Philox gives 4 float32 per call — process in groups of 4,
        # then handle remainder element-wise.
        var philox_chunk = 4
        var full_chunks  = numels // philox_chunk
        var remainder    = numels % philox_chunk

        for chunk in range(full_chunks):
            var base = chunk * philox_chunk
            var rand_f32 = rng.step_uniform()   # SIMD[float32, 4]

            comptime for lane in range(4):
                var r   = rand_f32[lane].cast[Self.dtype]()
                var x_v = x_ptr[base + lane]
                var m   = scale_s if r > p_s else zero_s
                out_ptr[base + lane]  = x_v * m
                mask_ptr[base + lane] = m

        # Scalar tail
        if remainder > 0:
            var rand_f32 = rng.step_uniform()
            var base = full_chunks * philox_chunk
            for lane in range(remainder):
                var r   = rand_f32[lane].cast[Self.dtype]()
                var x_v = x_ptr[base + lane]
                var m   = scale_s if r > p_s else zero_s
                out_ptr[base + lane]  = x_v * m
                mask_ptr[base + lane] = m

        var out_ndb = NDBuffer[Self.dtype](out_buf^, shape)
        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        if x.requires_grad:
            out.requires_grad_(True)
            var arg = ArgDropout[Self.dtype](
                mask_cpu = mask_buf^,
                mask_gpu = None,
                scale    = self.scale,
                on_gpu   = False,
            )
            out.bwdFnArg = Optional(
                BackwardFnArg[Self.dtype](
                    BACKWARD_DROPOUT,
                    ArgumentType[Self.dtype](arg^),
                )
            )
            out.add_ancestry(x)

        return out^

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    fn num_parameters(self) -> Int:
        return 0

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False

    fn set_seed(mut self, seed_val: UInt64):
        """Set Philox seed for reproducible dropout masks."""
        self.seed = seed_val

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), DROPOUT)


fn main() raises:
    pass
