from gpu import thread_idx, block_dim, grid_dim, block_idx
from gpu.host import DeviceContext
from sys import simd_width_of

from tenmo import Tensor
from common_utils import panic
from shapes import Shape
from mnemonics import Multiply, Add, Subtract, Divide, ReverseSubtract


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


fn launch[
    op_code: Int,
    dtype: DType = DType.float32,
](A: Tensor[dtype], scalar: Scalar[dtype]) raises -> Tensor[dtype]:
    if op_code == Divide and scalar == Scalar[dtype](0):
        raise Error("Divide by zero")

    var numels = A.numels()

    var threads_per_block: Int
    var num_blocks: Int

    if numels < 10_000:
        # Small arrays: Use fewer threads to avoid overhead
        threads_per_block = 128
        num_blocks = min(4, (numels + 127) // 128)
    elif numels < 1_000_000:
        # Medium arrays: Standard configuration
        threads_per_block = 256
        num_blocks = min(128, (numels + 255) // 256)
    else:
        # Large arrays: Maximize occupancy
        threads_per_block = 512
        # Cap at reasonable number to avoid diminishing returns
        num_blocks = min(1024, (numels + 511) // 512)

    # ===================================================================
    # Launch kernel
    # ===================================================================
    var ctx = DeviceContext()

    var compiled_func = ctx.compile_function[
        scalar_ops[op_code=op_code, dtype=dtype, simd_width=4],
        scalar_ops[op_code=op_code, dtype=dtype, simd_width=4],
    ]()

    var A_buffer = ctx.enqueue_create_buffer[dtype](numels)
    var result_buffer = ctx.enqueue_create_buffer[dtype](numels)

    # No need to zero-fill result_buffer - we write all elements
    A.write_to_device_buffer(A_buffer)

    ctx.enqueue_function(
        compiled_func,
        result_buffer,
        A_buffer,
        scalar,
        UInt(numels),
        grid_dim=num_blocks,
        block_dim=threads_per_block,
    )

    ctx.synchronize()

    return Tensor[dtype].from_device_buffer(result_buffer, A.shape())


from testing import assert_true
from common_utils import now


fn main() raises:
    var SIZE = 65536 * 10
    comptime dtype = DType.float32
    var tensor_a = Tensor[dtype].ones(SIZE)
    var start = now()
    var expect = tensor_a * 42
    print("CPU mul took: ", (now() - start) * 1000, "ms")
    # First test
    start = now()
    var result = launch[op_code=Multiply, dtype=dtype](tensor_a, 42)
    print("GPU mul took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))

    # Second test
    tensor_a = Tensor[dtype].rand(SIZE // 2, 2)
    var reshaped = tensor_a.reshape(2, SIZE // 2)
    start = now()
    expect = reshaped * 1919

    print("CPU mul took: ", (now() - start) * 1000, "ms")
    start = now()
    result = launch[op_code=Multiply, dtype=dtype](reshaped, 1919)

    print("GPU mul took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))
    start = now()
    expect = reshaped / 89

    print("CPU div took: ", (now() - start) * 1000, "ms")
    start = now()
    result = launch[op_code=Divide, dtype=dtype](reshaped, 89)

    print("GPU div took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))
    start = now()
    expect = reshaped - 999

    print("CPU subtract took: ", (now() - start) * 1000, "ms")
    start = now()
    result = launch[op_code=Subtract, dtype=dtype](reshaped, 999)

    print("GPU subtract took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))

    print("Launch success")
