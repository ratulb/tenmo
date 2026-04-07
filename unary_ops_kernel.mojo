from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from tenmo import Tensor
from ndbuffer import NDBuffer
from device import DeviceState
from common_utils import panic, Epsilon
from shapes import Shape
from mnemonics import (
    LOG,
    EXP,
    SQRT,
    TANH_FORWARD,
    NEGATE,
    SIGMOID_FORWARD,
    RELU_FORWARD,
)
from std.math import log, exp, sqrt


# ── Generic unary ops kernel (SQRT, NEGATE, ABS, RELU) ───────────────────────
# Works for any dtype — no floating point constraint needed.
# LOG, EXP, TANH, SIGMOID live in float_unary_ops below.


fn unary_ops[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
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
                    vec_result = sqrt(vec_a)
                elif op_code == NEGATE:
                    vec_result = -vec_a
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
                        res = sqrt(val)
                    elif op_code == NEGATE:
                        res = -val
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
                    vec_result = log(max(vec_a, SIMD[dtype, simd_width](epsilon)))
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



# ── UnaryOpsKernel launcher ───────────────────────────────────────────────────


struct UnaryOpsKernel[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    fn launch[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](A: NDBuffer[Self.dtype]) raises -> NDBuffer[Self.dtype]:
        """
        Core launch — takes NDBuffer, returns NDBuffer.
        Caller must ensure A is on GPU.
        Result is contiguous with zero offset.

        Dispatch:
          LOG, EXP,
          TANH_FORWARD,
          SIGMOID_FORWARD  → float_unary_ops  (dtype.is_floating_point())
          SQRT, NEGATE,
          ABS, RELU        → unary_ops        (any dtype)
        """
        debug_assert(A.is_on_gpu())

        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = A.device_state.value()
        var device_context = device_state.gpu()

        var contig_state = A.contiguous_device_state()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
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
                    dtype=Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread = 2 * simdwidth,
                    epsilon=epsilon,
                ],
                float_unary_ops[
                    op_code=op_code,
                    dtype=Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread = 2 * simdwidth,
                    epsilon=epsilon,
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
        else:
            var compiled = device_context.compile_function[
                unary_ops[
                    op_code=op_code,
                    dtype=Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread = 2 * simdwidth,
                ],
                unary_ops[
                    op_code=op_code,
                    dtype=Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread = 2 * simdwidth,
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

        var result_state = DeviceState[Self.dtype](
            result_buffer^, device_state.gpu
        )
        return NDBuffer[Self.dtype].with_device_state(result_state^, A.shape)

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
            )
        return threads_per_block, num_blocks


from std.testing import assert_true
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
    var result = Tensor[dtype](tensor_a.buffer.exp())
    print("GPU exp took: ", (now() - start) * 1000, "ms")
    assert_true(result.to_cpu().all_close(expect))

    # Test TANH
    tensor_A = Tensor[dtype].ones(SIZE) * 2
    var tensor_a_tanh = tensor_A.to_gpu()
    expect = tensor_A.tanh()
    start = now()
    result = Tensor[dtype](
        UnaryOpsKernel[dtype].launch[TANH_FORWARD](tensor_a_tanh.buffer)
    )
    print("GPU tanh took: ", (now() - start) * 1000, "ms")
    assert_true(result.to_cpu().all_close(expect))

    # Test SIGMOID
    tensor_A = Tensor[dtype].ones(SIZE) * 2
    var tensor_a_sigmoid = tensor_A.to_gpu()
    expect = tensor_A.sigmoid()
    start = now()
    result = Tensor[dtype](
        UnaryOpsKernel[dtype].launch[SIGMOID_FORWARD](tensor_a_sigmoid.buffer)
    )
    print("GPU sigmoid took: ", (now() - start) * 1000, "ms")
    assert_true(result.to_cpu().all_close(expect))

    # Test RELU
    tensor_A = Tensor[dtype].randn(SIZE)
    var tensor_a_relu = tensor_A.to_gpu()
    expect = tensor_A.relu()
    start = now()
    result = Tensor[dtype](
        UnaryOpsKernel[dtype].launch[RELU_FORWARD](tensor_a_relu.buffer)
    )
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
    result = Tensor[dtype](
        UnaryOpsKernel[dtype].launch[LOG](tensor_a_log.buffer)
    )
    print(
        "GPU log Tensor (default epsilon) took: ", (now() - start) * 1000, "ms"
    )
    assert_true(result.to_cpu().all_close(expect))

    # Test LOG via Tensor overload with custom epsilon
    tensor_A = Tensor[dtype].ones(SIZE) * 2
    tensor_a_log = tensor_A.to_gpu()
    start = now()
    result = Tensor[dtype](
        UnaryOpsKernel[dtype].launch[LOG, Scalar[dtype](1e-7)](
            tensor_a_log.buffer
        )
    )
    print(
        "GPU log Tensor (custom epsilon) took: ", (now() - start) * 1000, "ms"
    )
    assert_true(result.to_cpu().all_close(expect))

    # Test NEGATE
    tensor_A = Tensor[dtype].ones(SIZE) * 2
    expect = -tensor_A
    var tensor_a_neg = tensor_A.to_gpu()
    result = Tensor[dtype](-tensor_a_neg.buffer)
    assert_true(result.to_cpu().all_close(expect))

    print("Launch success")
