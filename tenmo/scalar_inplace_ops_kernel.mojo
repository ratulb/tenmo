from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from .tensor import Tensor
from .common_utils import panic
from .shapes import Shape
from .mnemonics import Multiply, Add, Subtract, Divide, ReverseSubtract
from .device import DeviceState
from .ndbuffer import NDBuffer

# Kernel template for various arithmetic ops involving ND Tensor and a single scalar
# Simplification - views becomes contiguous when copied to device and offset becomes 0


fn inplace_scalar_ops[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2
    * simd_width,  # Each thread processes twice simd size elements
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    scalar: Scalar[dtype],
    size: Int,
):
    """
    Element-wise scalar operations.

    - Each thread processes multiple items (better ILP)
    - SIMD vectorization within each item
    - Loop unrolling
    - Minimal divergence

    """

    var tid = thread_idx.x
    var gtid = Int(tid + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    # ===================================================================
    # Each thread processes CHUNK_SIZE elements
    # ===================================================================
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        # Process simd_vectors_per_thread vectors per thread
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            # Bounds check for this vector
            if i + simd_width <= size:
                # Full vector load
                var vec_a = A.load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width]

                comptime if op_code == Add:
                    vec_result = vec_a + scalar
                elif op_code == Subtract:
                    vec_result = vec_a - scalar
                elif op_code == Multiply:
                    vec_result = vec_a * scalar
                else:  # op_code == Divide:
                    vec_result = vec_a / scalar

                A.store[width=simd_width](i, vec_result)
            elif i < size:
                # Partial vector (tail handling)
                for j in range(size - i):
                    var val = A[i + j]
                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = val + scalar
                    elif op_code == Subtract:
                        res = val - scalar
                    elif op_code == Multiply:
                        res = val * scalar
                    else:  # op_code == Divide:
                        res = val / scalar

                    A[i + j] = res

        base_idx += stride * CHUNK_SIZE


struct InplaceScalarOperations[dtype: DType = DType.float32](
    ImplicitlyCopyable & Movable
):
    @staticmethod
    fn launch[
        op_code: Int,
    ](A: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]) raises:
        var numels = A.numels()

        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )
        ref A_device_state = A.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()

        var compiled_func = device_context.compile_function[
            inplace_scalar_ops[
                op_code=op_code,
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
            inplace_scalar_ops[
                op_code=op_code,
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        ref A_buffer = A_device_state.device_buffer()

        device_context.enqueue_function(
            compiled_func,
            # result_buffer,
            A_buffer,
            scalar,
            numels,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()
        # var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        # var out = NDBuffer[Self.dtype].with_device_state(device_state^, A.shape)

        # return out^

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

