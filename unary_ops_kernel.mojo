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
)
from math import log, exp, sqrt, tanh


# ── Generic unary ops kernel (no LOG) ────────────────────────────────────────


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
    """Generic unary ops kernel — EXP, SQRT, TANH, NEGATE, ABS.
    LOG is handled by dedicated log_op_f32 / log_op_f64 kernels.
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
                if op_code == EXP:
                    vec_result = exp(vec_a)
                elif op_code == SQRT:
                    vec_result = sqrt(vec_a)
                    _ = """elif op_code == TANH:
                    vec_result = tanh(vec_a)"""
                elif op_code == NEGATE:
                    vec_result = -vec_a
                else:  # ABS
                    vec_result = abs(vec_a)

                result.store[width=simd_width](i, vec_result)

            elif i < size:
                for j in range(size - i):
                    var val = A[i + j]
                    var res: Scalar[dtype]

                    @parameter
                    if op_code == EXP:
                        res = exp(val)
                    elif op_code == SQRT:
                        res = sqrt(val)
                        _ = """elif op_code == TANH:
                        res = tanh(val)"""
                    elif op_code == NEGATE:
                        res = -val
                    else:  # ABS
                        res = abs(val)

                    result[i + j] = res

        base_idx += stride * CHUNK_SIZE


# ── Dedicated log kernels ─────────────────────────────────────────────────────
# LOG is separated from unary_ops because:
# - log() requires where dtype.is_floating_point() which GPU compiler doesn't
#   yet support on generic kernels in current stable Mojo
# - Hardcoding dtype sidesteps the constraint entirely


fn log_op_f32[
    simd_width: Int = simd_width_of[DType.float32](),
    simd_vectors_per_thread: Int = 2 * simd_width,
    epsilon: Scalar[DType.float32] = Scalar[DType.float32](1e-12),
](
    result: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    A: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    size: UInt,
):
    """Dedicated float32 log kernel with epsilon clamping."""
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
    """Dedicated float64 log kernel with epsilon clamping."""
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
        epsilon: Scalar[
            Self.dtype
        ] = 1e-12 if Self.dtype.is_floating_point() else Scalar[Self.dtype](0),
    ](A: NDBuffer[Self.dtype]) raises -> NDBuffer[Self.dtype]:
        """
        Core launch — takes NDBuffer, returns NDBuffer.
        Caller must ensure A is on GPU.
        Result is contiguous with zero offset.
        LOG dispatches to dedicated f32/f64 kernels.
        All others dispatch to generic unary_ops kernel.
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
            # Dispatch to dedicated typed log kernels
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
        else:
            # Generic unary_ops kernel for EXP, SQRT, TANH, NEGATE, ABS
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
        epsilon: Scalar[
            Self.dtype
        ] = 1e-12 if Self.dtype.is_floating_point() else Scalar[Self.dtype](0),
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
    var SIZE = 65536 * 1000
    comptime dtype = DType.float32

    # Test EXP via Tensor overload
    var tensor_A = Tensor[dtype].ones(SIZE) * 2
    var tensor_a = tensor_A.to_gpu()
    var start = now()
    var expect = tensor_A.exp()
    print("CPU exp took: ", (now() - start) * 1000, "ms")
    start = now()
    var result = tensor_a.exp()
    print("GPU exp took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))

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
        Tensor[dtype](result_ndb^, requires_grad=False).all_close(expect)
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
    assert_true(result.all_close(expect))

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
    assert_true(result.all_close(expect))

    # Test NEGATE
    tensor_A = Tensor[dtype].ones(SIZE) * 2
    expect = -tensor_A
    var tensor_a_neg = tensor_A.to_gpu()
    result = -tensor_a_neg
    assert_true(result.all_close(expect))

    print("Launch success")
