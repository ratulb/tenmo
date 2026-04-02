from gpu import thread_idx, block_dim, grid_dim, block_idx
from sys import simd_width_of

from tenmo import Tensor
from ndbuffer import NDBuffer
from device import DeviceState
from common_utils import panic
from shapes import Shape
from mnemonics import (
    LOG,
    EXP,
    SQRT,
    TANH,
    NEGATE,
    SIGMOID,
    RELU,
)
from math import log, exp, sqrt


# ── Generic unary ops kernel (SQRT, NEGATE, ABS, RELU) ───────────────────────
# LOG, EXP, TANH, SIGMOID are handled by dedicated typed kernels because:
# - log(), exp() require where dtype.is_floating_point()
# - GPU compiler doesn't yet support that constraint on generic kernels
#   in current stable Mojo — even with hardcoded dtype, the runtime proof
#   fails. Dedicated typed kernels sidestep the constraint entirely.
# - tanh() additionally requires PTX ISA 7.0+ — implemented via exp() identity
# - sigmoid() is 1 / (1 + e^-x) — also requires exp()
#
# RELU = max(x, 0) — pure arithmetic, no float constraint — lives here.


fn unary_ops[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: UInt,
):
    """Generic unary ops kernel — SQRT, NEGATE, ABS, RELU.
    LOG, EXP, TANH, SIGMOID are handled by dedicated typed kernels.
    RELU = max(x, 0) — pure arithmetic, safe for any dtype.
    """
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width]

                @parameter
                if op_code == SQRT:
                    vec_result = sqrt(vec_a)
                elif op_code == NEGATE:
                    vec_result = -vec_a
                elif op_code == RELU:
                    vec_result = max(vec_a, SIMD[dtype, simd_width](0))
                else:  # ABS
                    vec_result = abs(vec_a)

                result.store[width=simd_width](i, vec_result)

            elif i < size:
                for j in range(size - i):
                    var val = A[i + j]
                    var res: Scalar[dtype]

                    @parameter
                    if op_code == SQRT:
                        res = sqrt(val)
                    elif op_code == NEGATE:
                        res = -val
                    elif op_code == RELU:
                        res = max(val, Scalar[dtype](0))
                    else:  # ABS
                        res = abs(val)

                    result[i + j] = res

        base_idx += stride * CHUNK_SIZE


# ── Dedicated exp kernels ─────────────────────────────────────────────────────


fn exp_op_f32[
    simd_width: Int = simd_width_of[DType.float32](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    A: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    size: UInt,
):
    """Dedicated float32 exp kernel."""
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                result.store[width=simd_width](i, exp(vec_a))

            elif i < size:
                for j in range(size - i):
                    result[i + j] = exp(A[i + j])

        base_idx += stride * CHUNK_SIZE


fn exp_op_f64[
    simd_width: Int = simd_width_of[DType.float64](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
    A: UnsafePointer[Scalar[DType.float64], ImmutAnyOrigin],
    size: UInt,
):
    """Dedicated float64 exp kernel."""
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                result.store[width=simd_width](i, exp(vec_a))

            elif i < size:
                for j in range(size - i):
                    result[i + j] = exp(A[i + j])

        base_idx += stride * CHUNK_SIZE


# ── Dedicated tanh kernels ────────────────────────────────────────────────────
# tanh() requires PTX ISA 7.0+ which is not generally available.
# Identity used: tanh(x) = (e^2x - 1) / (e^2x + 1)
# Only requires exp() — available on all PTX ISA versions.


fn tanh_op_f32[
    simd_width: Int = simd_width_of[DType.float32](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    A: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    size: UInt,
):
    """Dedicated float32 tanh kernel.
    Implemented as (e^2x - 1) / (e^2x + 1) to avoid PTX ISA 7.0 requirement.
    """
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var e2x = exp(vec_a + vec_a)
                var one = SIMD[DType.float32, simd_width](1.0)
                result.store[width=simd_width](i, (e2x - one) / (e2x + one))

            elif i < size:
                for j in range(size - i):
                    var x = A[i + j]
                    var e2x = exp(x + x)
                    result[i + j] = (e2x - 1.0) / (e2x + 1.0)

        base_idx += stride * CHUNK_SIZE


fn tanh_op_f64[
    simd_width: Int = simd_width_of[DType.float64](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
    A: UnsafePointer[Scalar[DType.float64], ImmutAnyOrigin],
    size: UInt,
):
    """Dedicated float64 tanh kernel.
    Implemented as (e^2x - 1) / (e^2x + 1) to avoid PTX ISA 7.0 requirement.
    """
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var e2x = exp(vec_a + vec_a)
                var one = SIMD[DType.float64, simd_width](1.0)
                result.store[width=simd_width](i, (e2x - one) / (e2x + one))

            elif i < size:
                for j in range(size - i):
                    var x = A[i + j]
                    var e2x = exp(x + x)
                    result[i + j] = (e2x - 1.0) / (e2x + 1.0)

        base_idx += stride * CHUNK_SIZE


# ── Dedicated sigmoid kernels ─────────────────────────────────────────────────
# sigmoid(x) = 1 / (1 + e^-x)
# Requires exp() — same is_floating_point() constraint — dedicated kernels.


fn sigmoid_op_f32[
    simd_width: Int = simd_width_of[DType.float32](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    A: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    size: UInt,
):
    """Dedicated float32 sigmoid kernel.
    Implemented as 1 / (1 + e^-x).
    """
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var one = SIMD[DType.float32, simd_width](1.0)
                result.store[width=simd_width](i, one / (one + exp(-vec_a)))

            elif i < size:
                for j in range(size - i):
                    var x = A[i + j]
                    result[i + j] = 1.0 / (1.0 + exp(-x))

        base_idx += stride * CHUNK_SIZE


fn sigmoid_op_f64[
    simd_width: Int = simd_width_of[DType.float64](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
    A: UnsafePointer[Scalar[DType.float64], ImmutAnyOrigin],
    size: UInt,
):
    """Dedicated float64 sigmoid kernel.
    Implemented as 1 / (1 + e^-x).
    """
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var one = SIMD[DType.float64, simd_width](1.0)
                result.store[width=simd_width](i, one / (one + exp(-vec_a)))

            elif i < size:
                for j in range(size - i):
                    var x = A[i + j]
                    result[i + j] = 1.0 / (1.0 + exp(-x))

        base_idx += stride * CHUNK_SIZE


# ── Dedicated log kernels ─────────────────────────────────────────────────────
# epsilon defaults differ by dtype:
#   float32 → 1e-7  (1e-12 flushes to 0.0 in float32 — silent breakage)
#   float64 → 1e-12


fn log_op_f32[
    simd_width: Int = simd_width_of[DType.float32](),
    simd_vectors_per_thread: Int = 2 * simd_width,
    epsilon: Scalar[DType.float32] = Scalar[DType.float32](1e-7),
](
    result: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    A: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    size: UInt,
):
    """Dedicated float32 log kernel with epsilon clamping.
    epsilon defaults to 1e-7 (safe for float32 precision floor).
    """
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var clamped = max(
                    vec_a, SIMD[DType.float32, simd_width](epsilon)
                )
                result.store[width=simd_width](i, log(clamped))

            elif i < size:
                for j in range(size - i):
                    var val = A[i + j]
                    var clamped = max(val, epsilon)
                    result[i + j] = log(clamped)

        base_idx += stride * CHUNK_SIZE


fn log_op_f64[
    simd_width: Int = simd_width_of[DType.float64](),
    simd_vectors_per_thread: Int = 2 * simd_width,
    epsilon: Scalar[DType.float64] = Scalar[DType.float64](1e-12),
](
    result: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
    A: UnsafePointer[Scalar[DType.float64], ImmutAnyOrigin],
    size: UInt,
):
    """Dedicated float64 log kernel with epsilon clamping.
    epsilon defaults to 1e-12 (safe for float64 precision).
    """
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var clamped = max(
                    vec_a, SIMD[DType.float64, simd_width](epsilon)
                )
                result.store[width=simd_width](i, log(clamped))

            elif i < size:
                for j in range(size - i):
                    var val = A[i + j]
                    var clamped = max(val, epsilon)
                    result[i + j] = log(clamped)

        base_idx += stride * CHUNK_SIZE


# ── UnaryOpsKernel launcher ───────────────────────────────────────────────────


struct UnaryOpsKernel[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    fn launch[
        op_code: Int,
        epsilon: Scalar[Self.dtype] = 1e-7 if Self.dtype
        == DType.float32 else 1e-12 if Self.dtype.is_floating_point() else Scalar[
            Self.dtype
        ](
            0
        ),
    ](A: NDBuffer[Self.dtype]) raises -> NDBuffer[Self.dtype]:
        """
        Core launch — takes NDBuffer, returns NDBuffer.
        Caller must ensure A is on GPU.
        Result is contiguous with zero offset.

        Dispatch:
          LOG              → log_op_f32 / log_op_f64     (epsilon-clamped)
          EXP              → exp_op_f32 / exp_op_f64
          TANH             → tanh_op_f32 / tanh_op_f64   (via exp identity)
          SIGMOID          → sigmoid_op_f32 / sigmoid_op_f64 (via exp)
          SQRT, NEGATE,
          ABS, RELU        → unary_ops                   (any dtype)
        """
        debug_assert(A.is_on_gpu())

        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = A.device_state.value()
        var device_context = device_state.gpu()

        # Ensure contiguous GPU input
        var contig_state = A.contiguous_device_state()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        @parameter
        if op_code == LOG:

            @parameter
            if Self.dtype == DType.float32:
                var compiled = device_context.compile_function[
                    log_op_f32[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                        epsilon = epsilon.cast[DType.float32](),
                    ],
                    log_op_f32[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                        epsilon = epsilon.cast[DType.float32](),
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    result_buffer,
                    contig_state.device_buffer(),
                    UInt(numels),
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            elif Self.dtype == DType.float64:
                var compiled = device_context.compile_function[
                    log_op_f64[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                        epsilon = epsilon.cast[DType.float64](),
                    ],
                    log_op_f64[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                        epsilon = epsilon.cast[DType.float64](),
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    result_buffer,
                    contig_state.device_buffer(),
                    UInt(numels),
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            else:
                panic(
                    "UnaryOpsKernel: LOG only supported for float32 and float64"
                )
        elif op_code == EXP:

            @parameter
            if Self.dtype == DType.float32:
                var compiled = device_context.compile_function[
                    exp_op_f32[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                    ],
                    exp_op_f32[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    result_buffer,
                    contig_state.device_buffer(),
                    UInt(numels),
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            elif Self.dtype == DType.float64:
                var compiled = device_context.compile_function[
                    exp_op_f64[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                    ],
                    exp_op_f64[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    result_buffer,
                    contig_state.device_buffer(),
                    UInt(numels),
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            else:
                panic(
                    "UnaryOpsKernel: EXP only supported for float32 and float64"
                )
        elif op_code == TANH:

            @parameter
            if Self.dtype == DType.float32:
                var compiled = device_context.compile_function[
                    tanh_op_f32[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                    ],
                    tanh_op_f32[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    result_buffer,
                    contig_state.device_buffer(),
                    UInt(numels),
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            elif Self.dtype == DType.float64:
                var compiled = device_context.compile_function[
                    tanh_op_f64[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                    ],
                    tanh_op_f64[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    result_buffer,
                    contig_state.device_buffer(),
                    UInt(numels),
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            else:
                panic(
                    "UnaryOpsKernel: TANH only supported for float32 and"
                    " float64"
                )
        elif op_code == SIGMOID:

            @parameter
            if Self.dtype == DType.float32:
                var compiled = device_context.compile_function[
                    sigmoid_op_f32[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                    ],
                    sigmoid_op_f32[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    result_buffer,
                    contig_state.device_buffer(),
                    UInt(numels),
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            elif Self.dtype == DType.float64:
                var compiled = device_context.compile_function[
                    sigmoid_op_f64[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                    ],
                    sigmoid_op_f64[
                        simd_width=simdwidth,
                        simd_vectors_per_thread = 2 * simdwidth,
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    result_buffer,
                    contig_state.device_buffer(),
                    UInt(numels),
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            else:
                panic(
                    "UnaryOpsKernel: SIGMOID only supported for float32 and"
                    " float64"
                )
        else:
            # Generic unary_ops kernel for SQRT, NEGATE, ABS, RELU
            var compiled = device_context.compile_function[
                unary_ops[
                    op_code=op_code,
                    dtype = Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread = 2 * simdwidth,
                ],
                unary_ops[
                    op_code=op_code,
                    dtype = Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread = 2 * simdwidth,
                ],
            ]()
            device_context.enqueue_function(
                compiled,
                result_buffer,
                contig_state.device_buffer(),
                UInt(numels),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](
            result_buffer^, device_state.gpu
        )
        return NDBuffer[Self.dtype].with_device_state(result_state^, A.shape)

    @staticmethod
    fn launch[
        op_code: Int,
        epsilon: Scalar[Self.dtype] = 1e-7 if Self.dtype
        == DType.float32 else 1e-12 if Self.dtype.is_floating_point() else Scalar[
            Self.dtype
        ](
            0
        ),
    ](A: Tensor[Self.dtype]) raises -> Tensor[Self.dtype]:
        """
        Convenience overload — takes Tensor, returns Tensor.
        Delegates to NDBuffer overload.
        """
        debug_assert(A.is_on_gpu())
        var result_ndb = Self.launch[op_code, epsilon](A.buffer)
        return Tensor[Self.dtype](result_ndb^, requires_grad=False)

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
            num_blocks = min(
                (total_chunks + 255) // 256, 512
            )  # Cap at 512 blocks
        return threads_per_block, num_blocks


from testing import assert_true
from common_utils import now


fn main() raises:
    var SIZE = 65536
    comptime dtype = DType.float32

    # Test EXP
    var tensor_A = Tensor[dtype].ones(SIZE) * 2
    var tensor_a = tensor_A.to_gpu()
    var start = now()
    var expect = tensor_A.exp()
    print("CPU exp took: ", (now() - start) * 1000, "ms")
    start = now()
    var result = tensor_a.exp()
    print("GPU exp took: ", (now() - start) * 1000, "ms")
    assert_true(result.to_cpu().all_close(expect))

    # Test TANH
    tensor_A = Tensor[dtype].ones(SIZE) * 2
    var tensor_a_tanh = tensor_A.to_gpu()
    expect = tensor_A.tanh()
    start = now()
    result = UnaryOpsKernel[dtype].launch[TANH](tensor_a_tanh)
    print("GPU tanh took: ", (now() - start) * 1000, "ms")
    assert_true(result.to_cpu().all_close(expect))

    # Test SIGMOID
    tensor_A = Tensor[dtype].ones(SIZE) * 2
    var tensor_a_sigmoid = tensor_A.to_gpu()
    expect = tensor_A.sigmoid()
    start = now()
    result = UnaryOpsKernel[dtype].launch[SIGMOID](tensor_a_sigmoid)
    print("GPU sigmoid took: ", (now() - start) * 1000, "ms")
    assert_true(result.to_cpu().all_close(expect))

    # Test RELU
    tensor_A = Tensor[dtype].randn(SIZE)  # mix of positive and negative
    var tensor_a_relu = tensor_A.to_gpu()
    expect = tensor_A.relu()
    start = now()
    result = UnaryOpsKernel[dtype].launch[RELU](tensor_a_relu)
    print("GPU relu took: ", (now() - start) * 1000, "ms")
    assert_true(result.to_cpu().all_close(expect))

    # Test LOG via NDBuffer overload with default epsilon
    tensor_A = Tensor[dtype].ones(SIZE) * 2
    var ndb_a = tensor_A.to_gpu().buffer
    expect = tensor_A.log()
    start = now()
    var result_ndb = UnaryOpsKernel[dtype].launch[LOG](ndb_a)
    print(
        "GPU log NDBuffer (default epsilon) took: ",
        (now() - start) * 1000,
        "ms",
    )
    assert_true(
        Tensor[dtype](result_ndb^, requires_grad=False)
        .to_cpu()
        .all_close(expect)
    )

    # Test LOG via Tensor overload with default epsilon
    tensor_A = Tensor[dtype].ones(SIZE) * 2
    var tensor_a_log = tensor_A.to_gpu()
    expect = tensor_A.log()
    start = now()
    result = UnaryOpsKernel[dtype].launch[LOG](tensor_a_log)
    print(
        "GPU log Tensor (default epsilon) took: ", (now() - start) * 1000, "ms"
    )
    assert_true(result.to_cpu().all_close(expect))

    # Test LOG via Tensor overload with custom epsilon
    tensor_A = Tensor[dtype].ones(SIZE) * 2
    tensor_a_log = tensor_A.to_gpu()
    start = now()
    result = UnaryOpsKernel[dtype].launch[LOG, Scalar[dtype](1e-7)](
        tensor_a_log
    )
    print(
        "GPU log Tensor (custom epsilon) took: ", (now() - start) * 1000, "ms"
    )
    assert_true(result.to_cpu().all_close(expect))

    # Test NEGATE
    tensor_A = Tensor[dtype].ones(SIZE) * 2
    expect = -tensor_A
    var tensor_a_neg = tensor_A.to_gpu()
    result = -tensor_a_neg
    assert_true(result.to_cpu().all_close(expect))

    print("Launch success")
