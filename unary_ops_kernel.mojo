from gpu import thread_idx, block_dim, grid_dim, block_idx
from sys import simd_width_of

from tenmo import Tensor
from common_utils import panic
from shapes import Shape
from mnemonics import (
    LOG,
    EXP,
    SQRT,
    TANH,
    NEGATE,
)
from math import log10, exp, sqrt, tanh

comptime log_factor = 2.302585092


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
                if op_code == LOG:
                    vec_result = log10(vec_a) * Scalar[dtype](log_factor)
                elif op_code == EXP:
                    vec_result = exp(vec_a)
                elif op_code == SQRT:
                    vec_result = sqrt(vec_a)
                elif op_code == TANH:
                    vec_result = tanh(vec_a)
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
                    if op_code == LOG:
                        res = log10(val) * Scalar[dtype](log_factor)
                    elif op_code == EXP:
                        res = exp(val)
                    elif op_code == SQRT:
                        res = sqrt(val)
                    elif op_code == TANH:
                        res = tanh(val)
                    elif op_code == NEGATE:
                        res = -val
                    else:  # ABS
                        res = abs(val)

                    result[i + j] = res

        base_idx += stride * CHUNK_SIZE


struct UnaryOpsKernel[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    fn launch[
        op_code: Int,
    ](A: Tensor[Self.dtype]) raises -> Tensor[Self.dtype]:
        debug_assert(A.is_on_gpu())

        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )

        ref device_state = A.buffer.device_state.value()
        var device_context = device_state.gpu()

        var compiled_func = device_context.compile_function[
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

        ref A_buffer = device_state.device_buffer()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            UInt(numels),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        return Tensor[Self.dtype].from_device_buffer(result_buffer, A.shape())

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
    var tensor_A = Tensor[dtype].ones(SIZE) * 2
    var tensor_a = tensor_A.to_gpu()
    var start = now()
    var expect = tensor_A.exp()
    print("CPU mul took: ", (now() - start) * 1000, "ms")
    # First test
    start = now()
    var result = tensor_a.exp()
    print("GPU mul took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))

    # Second test
    tensor_A = Tensor[dtype].ones(SIZE) * 2
    expect = - tensor_A
    tensor_a = tensor_A.to_gpu()
    result = - tensor_a
    assert_true(result.all_close(expect))

    print("Launch success")
