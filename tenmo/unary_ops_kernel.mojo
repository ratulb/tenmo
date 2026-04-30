from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from .tensor import Tensor
from .ndbuffer import NDBuffer
from .device import DeviceState
from .common_utils import panic, Epsilon
from .shapes import Shape
from .mnemonics import (
    LOG,
    EXP,
    SQRT,
    TANH_FORWARD,
    NEGATE,
    SIGMOID_FORWARD,
    RELU_FORWARD,
    INVERT,
)
from std.math import log, exp, rsqrt

# Invert DType.bool


fn invert_bool[
    simd_width: Int = simd_width_of[DType.uint8](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
    A: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    size: Int,
):
    """Logical NOT for bool stored as uint8. 0 -> 1, 1 -> 0."""
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                # logical NOT: 0->1, anything else->0
                var vec_result = (
                    vec_a.eq(SIMD[DType.uint8, simd_width](0))
                ).cast[DType.uint8]()
                result.store[width=simd_width](i, vec_result)
            elif i < size:
                for j in range(size - i):
                    result[i + j] = UInt8(1) if A[i + j] == UInt8(0) else UInt8(
                        0
                    )
        base_idx += stride * CHUNK_SIZE


# ── Generic unary ops kernel (SQRT, NEGATE, ABS, RELU) ───────────────────────
# Works for any dtype — no floating point constraint needed.
# LOG, EXP, TANH, SIGMOID live in float_unary_ops below.


fn unary_ops[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
):
    """Generic unary ops kernel — SQRT, NEGATE, ABS, RELU.
    LOG, EXP, TANH, SIGMOID are handled by float_unary_ops.
    RELU = max(x, 0) — pure arithmetic, safe for any dtype.
    """
    var tid = thread_idx.x
    var gtid = Int(tid + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width]

                comptime if op_code == SQRT:
                    vec_result = SIMD[dtype, simd_width](1)/rsqrt(epsilon + vec_a)
                elif op_code == NEGATE:
                    vec_result = -vec_a
                elif op_code == INVERT:
                    vec_result = vec_a.__invert__()

                elif op_code == RELU_FORWARD:
                    vec_result = max(vec_a, SIMD[dtype, simd_width](0))
                else:  # ABS
                    vec_result = abs(vec_a)

                result.store[width=simd_width](i, vec_result)

            elif i < size:
                for j in range(size - i):
                    var val = A[i + j]
                    var res: Scalar[dtype]

                    comptime if op_code == SQRT:
                        res = Scalar[dtype](1)/rsqrt(epsilon + val)
                    elif op_code == NEGATE:
                        res = -val
                    elif op_code == INVERT:
                        res = val.__invert__()
                    elif op_code == RELU_FORWARD:
                        res = max(val, Scalar[dtype](0))
                    else:  # ABS
                        res = abs(val)

                    result[i + j] = res

        base_idx += stride * CHUNK_SIZE


# ── Floating point unary ops kernel (LOG, EXP, TANH, SIGMOID) ────────────────
# Single merged kernel — requires dtype.is_floating_point().
# Supported in Mojo 0.26.2+; previously crashed the compiler.
#
# tanh notes:
#   tanh() requires PTX ISA 7.0+ — implemented via exp() identity:
#   tanh(x) = (e^2x - 1) / (e^2x + 1)  — works on all PTX ISA versions.
#
# log notes:
#   epsilon-clamped: log(max(x, epsilon)) to avoid log(0) = -inf.
#   epsilon defaults differ by dtype:
#     float32 → 1e-7  (1e-12 flushes to 0.0 in float32 — silent breakage)
#     float64 → 1e-12


fn float_unary_ops[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
) where dtype.is_floating_point():
    """Floating point unary ops kernel — LOG, EXP, TANH, SIGMOID.
    Requires dtype.is_floating_point() — supported in Mojo 0.26.2+.
    """
    var tid = thread_idx.x
    var gtid = Int(tid + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var one = SIMD[dtype, simd_width](1.0)
                var vec_result: SIMD[dtype, simd_width]

                comptime if op_code == LOG:
                    vec_result = log(
                        max(vec_a, SIMD[dtype, simd_width](epsilon))
                    )
                elif op_code == EXP:
                    vec_result = exp(vec_a)
                elif op_code == TANH_FORWARD:
                    var e2x = exp(vec_a + vec_a)
                    vec_result = (e2x - one) / (e2x + one)
                else:  # SIGMOID_FORWARD
                    vec_result = one / (one + exp(-vec_a))

                result.store[width=simd_width](i, vec_result)

            elif i < size:
                for j in range(size - i):
                    var x = A[i + j]
                    var res: Scalar[dtype]

                    comptime if op_code == LOG:
                        res = log(max(x, epsilon))
                    elif op_code == EXP:
                        res = exp(x)
                    elif op_code == TANH_FORWARD:
                        var e2x = exp(x + x)
                        res = (e2x - 1.0) / (e2x + 1.0)
                    else:  # SIGMOID_FORWARD
                        res = 1.0 / (1.0 + exp(-x))

                    result[i + j] = res

        base_idx += stride * CHUNK_SIZE


# =============================================================================
#
# Writes two output buffers in a single kernel pass:
#   result  — the activated values  (ReLU: max(x, 0))
#   mask    — the gradient gate     (ReLU: 1.0 if x > 0 else 0.0)
#
# No floating-point constraint — ReLU is safe for any dtype.
# =============================================================================


fn unary_ops_with_mask[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
):
    """Single-pass kernel: compute activation output AND gradient mask.

    Both result and mask are written in one GPU pass — no second kernel needed.

    For RELU_FORWARD:
        result[i] = max(A[i], 0)
        mask[i]   = 1.0 if A[i] > 0 else 0.0

    Args:
        result: Output buffer for activated values.
        mask:   Output buffer for gradient mask.
        A:      Input buffer (contiguous, same device).
        size:   Total number of elements.
    """
    var tid = thread_idx.x
    var gtid = Int(tid + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    var zero_vec = SIMD[dtype, simd_width](0)
    var one_vec = SIMD[dtype, simd_width](1)
    var zero_s = Scalar[dtype](0)
    var one_s = Scalar[dtype](1)

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                # Full SIMD chunk
                var vec_a = A.load[width=simd_width](i)

                var vec_result: SIMD[dtype, simd_width]
                var vec_mask: SIMD[dtype, simd_width]

                comptime if op_code == RELU_FORWARD:
                    vec_result = max(vec_a, zero_vec)
                    # mask = 1 where input > 0, else 0
                    vec_mask = vec_a.gt(zero_vec).select(one_vec, zero_vec)

                # Extend here for other ops that need a mask (e.g. leaky ReLU)
                else:
                    vec_result = vec_a  # identity fallback
                    vec_mask = one_vec

                result.store[width=simd_width](i, vec_result)
                mask.store[width=simd_width](i, vec_mask)

            elif i < size:
                # Scalar tail
                for j in range(size - i):
                    var val = A[i + j]
                    var res: Scalar[dtype]
                    var msk: Scalar[dtype]

                    comptime if op_code == RELU_FORWARD:
                        res = max(val, zero_s)
                        msk = one_s if val > zero_s else zero_s
                    else:
                        res = val
                        msk = one_s

                    result[i + j] = res
                    mask[i + j] = msk

        base_idx += stride * CHUNK_SIZE


# ── UnaryOpsKernel launcher ───────────────────────────────────────────────────


struct UnaryOpsKernel[dtype: DType](ImplicitlyCopyable & Movable):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    fn launch[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](A: NDBuffer[Self.dtype]) raises -> NDBuffer[Self.dtype]:
        comptime epsilon_rebinded = rebind[Scalar[Self.datatype]](epsilon)
        comptime if op_code == INVERT:
            comptime assert (
                Self.dtype == DType.bool or Self.dtype.is_integral()
            ), "INVERT only valid for bool and integer types"
        debug_assert(A.is_on_gpu())
        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.datatype]()
        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )
        ref device_state = A.device_state.value()
        var device_context = device_state.gpu[]
        var contig_state = A.contiguous_device_state()
        var result_buffer = device_context.enqueue_create_buffer[Self.datatype](
            numels
        )
        comptime if op_code == LOG or op_code == EXP or op_code == TANH_FORWARD or op_code == SIGMOID_FORWARD:
            comptime if not Self.dtype.is_floating_point():
                panic(
                    "UnaryOpsKernel: LOG/EXP/TANH/SIGMOID require a floating"
                    " point dtype"
                )
            var compiled = device_context.compile_function[
                float_unary_ops[
                    op_code=op_code,
                    dtype=Self.datatype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                    epsilon=epsilon_rebinded,
                ],
                float_unary_ops[
                    op_code=op_code,
                    dtype=Self.datatype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                    epsilon=epsilon_rebinded,
                ],
            ]()
            device_context.enqueue_function(
                compiled,
                result_buffer,
                contig_state.device_buffer(),
                numels,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        elif op_code == INVERT and Self.dtype == DType.bool:
            var compiled = device_context.compile_function[
                invert_bool[simdwidth, 2 * simdwidth],
                invert_bool[simdwidth, 2 * simdwidth],
            ]()
            device_context.enqueue_function(
                compiled,
                result_buffer,
                contig_state.device_buffer(),
                numels,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        else:
            var compiled = device_context.compile_function[
                unary_ops[
                    op_code=op_code,
                    dtype=Self.datatype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                    epsilon=epsilon_rebinded,
                ],
                unary_ops[
                    op_code=op_code,
                    dtype=Self.datatype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                    epsilon=epsilon_rebinded,
                ],
            ]()
            device_context.enqueue_function(
                compiled,
                result_buffer,
                contig_state.device_buffer(),
                numels,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype].__init__[True](
            result_buffer^, device_state.gpu
        )
        return NDBuffer[Self.dtype].with_device_state(result_state^, A.shape)

    # ──launch_with_mask() ───────────────────────────────────────────────
    @staticmethod
    fn launch_with_mask[
        op_code: Int,
    ](A: NDBuffer[Self.dtype]) raises -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ]:
        """Launch unary op + mask kernel. Returns (output, mask) as GPU NDBuffers.

        Both buffers are written in a single GPU kernel pass.

        Non-contiguous input is handled via contiguous_device_state() —
        which performs ONE map_to_host copy (not one per element), then
        the kernel operates on the resulting flat buffer.

        Args:
            A: Input NDBuffer. Must be on GPU.

        Returns:
            Tuple of (output NDBuffer, mask NDBuffer), both contiguous on GPU.
        """
        debug_assert(A.is_on_gpu())

        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = A.device_state.value()
        var device_context = device_state.gpu[]

        # Non-contiguous: produce one contiguous GPU buffer in a single
        # map_to_host sweep — NOT one map_to_host call per index.
        var contig_state = A.contiguous_device_state()

        # Allocate both output buffers on the same device
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )
        var mask_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        var compiled = device_context.compile_function[
            unary_ops_with_mask[
                op_code=op_code,
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
            unary_ops_with_mask[
                op_code=op_code,
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        # Single kernel dispatch — writes result AND mask simultaneously
        device_context.enqueue_function(
            compiled,
            result_buffer,  # out: activated values
            mask_buffer,  # out: gradient mask
            contig_state.device_buffer(),  # in:  contiguous source
            numels,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](
            result_buffer^, device_state.gpu
        )
        var mask_state = DeviceState[Self.dtype](mask_buffer^, device_state.gpu)

        var out_ndb = NDBuffer[Self.dtype].with_device_state(
            result_state^, A.shape
        )
        var mask_ndb = NDBuffer[Self.dtype].with_device_state(
            mask_state^, A.shape
        )

        return (out_ndb^, mask_ndb^)

    @staticmethod
    fn launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
        threads_per_block: Int
        num_blocks: Int

        if numels < 4096:
            threads_per_block = 128
            num_blocks = (numels + 127) // 128
        elif numels < 65536:
            threads_per_block = 256
            num_blocks = (numels + 255) // 256
        else:
            threads_per_block = 256
            var total_chunks = (numels + (simdwidth * 2 * simdwidth - 1)) // (
                simdwidth * 2 * simdwidth
            )
            num_blocks = min((total_chunks + 255) // 256, 512)
        return threads_per_block, num_blocks
