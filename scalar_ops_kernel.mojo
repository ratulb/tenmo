from gpu import thread_idx, block_dim, grid_dim, block_idx
from sys import simd_width_of

from tenmo import Tensor
from common_utils import panic
from shapes import Shape
from mnemonics import Multiply, Add, Subtract, Divide, ReverseSubtract
from device import DeviceState
from ndbuffer import NDBuffer

# Kernel template for various arithmetic ops involving ND Tensor and a single scalar
# Simplification - views becomes contiguous when copied to device and offset becomes 0


fn scalar_ops[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2
    * simd_width,  # Each thread processes twice simd size elements
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    scalar: Scalar[dtype],
    size: UInt,
):
    """
    Element-wise scalar operations.

    - Each thread processes multiple items (better ILP)
    - SIMD vectorization within each item
    - Loop unrolling
    - Minimal divergence

    """

    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    # ===================================================================
    # Each thread processes CHUNK_SIZE elements
    # ===================================================================
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        # Process simd_vectors_per_thread vectors per thread
        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            # Bounds check for this vector
            if i + simd_width <= size:
                # Full vector load
                var vec_a = A.load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width]

                @parameter
                if op_code == Add:
                    vec_result = vec_a + scalar
                elif op_code == Subtract:
                    vec_result = vec_a - scalar
                elif op_code == ReverseSubtract:
                    vec_result = scalar - vec_a
                elif op_code == Multiply:
                    vec_result = vec_a * scalar
                elif op_code == Divide:
                    vec_result = vec_a / scalar
                else:
                    vec_result = scalar / vec_a

                result.store[width=simd_width](i, vec_result)
            elif i < size:
                # Partial vector (tail handling)
                for j in range(size - i):
                    var val = A[i + j]
                    var res: Scalar[dtype]

                    @parameter
                    if op_code == Add:
                        res = val + scalar
                    elif op_code == Subtract:
                        res = val - scalar
                    elif op_code == ReverseSubtract:
                        res = scalar - val
                    elif op_code == Multiply:
                        res = val * scalar
                    elif op_code == Divide:
                        res = val / scalar
                    else:
                        res = scalar / val

                    result[i + j] = res

        base_idx += stride * CHUNK_SIZE


struct ScalarOpsKernel[dtype: DType = DType.float32](
    ImplicitlyCopyable & Movable
):
    @staticmethod
    fn launch[
        op_code: Int,
    ](A: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) raises -> Tensor[
        Self.dtype
    ]:
        debug_assert(
            A.is_on_gpu(), "ScalarOpsKernel -> launch: Tensors must be on GPU"
        )
        if op_code == Divide and scalar == Scalar[Self.dtype](0):
            raise Error("Divide by zero")

        var numels = A.numels()

        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )
        ref A_device_state = A.buffer.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()

        var compiled_func = device_context.compile_function[
            scalar_ops[
                op_code=op_code,
                dtype = Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread = 2 * simdwidth,
            ],
            scalar_ops[
                op_code=op_code,
                dtype = Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread = 2 * simdwidth,
            ],
        ]()

        ref A_buffer = A_device_state.device_buffer()
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )
        var start = now()
        print("Writing to buffer took: ", (now() - start) * 1000, "ms")

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            scalar,
            UInt(numels),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()
        start = now()
        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        var ndb = NDBuffer[Self.dtype].with_device_state(
            device_state^, A.shape()
        )
        var out = Tensor[Self.dtype](ndb^, requires_grad=A.requires_grad)

        _ = """var out = Tensor[Self.dtype].from_device_buffer(
            result_buffer, A.shape(), requires_grad=A.requires_grad
        )"""
        print("Reading from buffer took: ", (now() - start) * 1000, "ms")
        return out^

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
    var SIZE = 65536 * 10
    comptime dtype = DType.float32
    var tensor_A = Tensor[dtype].ones(SIZE, requires_grad=True)
    var tensor_a = tensor_A.to_gpu()
    var start = now()
    var expect = (tensor_A * 42) + 2
    print("CPU mul took: ", (now() - start) * 1000, "ms")
    # First test
    start = now()
    var result = (tensor_a * 42) + 2
    result = result.to_cpu()
    print("Result\n")
    result.print()
    print("GPU mul took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))

    # Second test
    tensor_A = Tensor[dtype].rand(SIZE // 2, 2)
    var reshaped = tensor_A.reshape(2, SIZE // 2)
    start = now()
    expect = reshaped * 1919

    print("CPU mul took: ", (now() - start) * 1000, "ms")

    print("CPU mul took: ", (now() - start) * 1000, "ms")
    start = now()
    tensor_a = reshaped.to_gpu()
    result = tensor_a * 1919

    result = result.to_cpu()
    print("GPU mul took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))
    start = now()
    expect = reshaped / 89

    print("CPU div took: ", (now() - start) * 1000, "ms")
    start = now()
    result = tensor_a / 89

    result = result.to_cpu()
    print("GPU div took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))
    start = now()
    expect = reshaped - 999

    print("CPU subtract took: ", (now() - start) * 1000, "ms")
    start = now()
    result = tensor_a - 999

    result = result.to_cpu()
    print("GPU subtract took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))

    start = now()
    expect = 999 - reshaped

    print("CPU reverse subtract took: ", (now() - start) * 1000, "ms")
    start = now()
    result = 999 - tensor_a

    result = result.to_cpu()
    print("GPU reverse subtract took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))

    start = now()
    expect = 999 / reshaped

    print("CPU reverse divide took: ", (now() - start) * 1000, "ms")
    start = now()
    result = 999 / tensor_a

    result = result.to_cpu()
    print("GPU reverse divide took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))

    print("Launch success")
