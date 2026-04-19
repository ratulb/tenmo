from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from tenmo import Tensor
from common_utils import panic
from shapes import Shape
from mnemonics import (
    Multiply,
    Add,
    Subtract,
    Divide,
    ReverseSubtract,
    MAX,
    MIN,
    POW,
)
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
                elif op_code == ReverseSubtract:
                    vec_result = scalar - vec_a
                elif op_code == Multiply:
                    vec_result = vec_a * scalar
                elif op_code == Divide:
                    vec_result = vec_a / scalar
                elif op_code == MAX:
                    vec_result = max(vec_a, SIMD[dtype, simd_width](scalar))
                elif op_code == MIN:
                    vec_result = min(vec_a, SIMD[dtype, simd_width](scalar))

                else:
                    vec_result = scalar / vec_a

                result.store[width=simd_width](i, vec_result)
            elif i < size:
                # Partial vector (tail handling)
                for j in range(size - i):
                    var val = A[i + j]
                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = val + scalar
                    elif op_code == Subtract:
                        res = val - scalar
                    elif op_code == ReverseSubtract:
                        res = scalar - val
                    elif op_code == Multiply:
                        res = val * scalar
                    elif op_code == Divide:
                        res = val / scalar
                    elif op_code == MAX:
                        res = val if val > scalar else scalar
                    elif op_code == MIN:
                        res = val if val < scalar else scalar

                    else:
                        res = scalar / val

                    result[i + j] = res

        base_idx += stride * CHUNK_SIZE


# ── Dedicated pow kernels ─────────────────────────────────────────────────────
# POW is separated from scalar_ops because:
# - pow() with float exponent needs Powable trait constraint
# - GPU compiler doesn't yet support generic dtype constraints on kernels
# - Hardcoding dtype sidesteps the constraint entirely


fn pow_op_f32[
    simd_width: Int = simd_width_of[DType.float32](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    A: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    exponent: Scalar[DType.float32],
    size: Int,
):
    """Dedicated float32 pow kernel — x ** exponent elementwise."""
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
                var vec_exp = SIMD[DType.float32, simd_width](exponent)
                result.store[width=simd_width](i, pow(vec_a, vec_exp))

            elif i < size:
                for j in range(size - i):
                    result[i + j] = pow(A[i + j], exponent)

        base_idx += stride * CHUNK_SIZE


fn pow_op_f64[
    simd_width: Int = simd_width_of[DType.float64](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[DType.float64], MutAnyOrigin],
    A: UnsafePointer[Scalar[DType.float64], ImmutAnyOrigin],
    exponent: Scalar[DType.float64],
    size: Int,
):
    """Dedicated float64 pow kernel — x ** exponent elementwise."""
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
                var vec_exp = SIMD[DType.float64, simd_width](exponent)
                result.store[width=simd_width](i, pow(vec_a, vec_exp))

            elif i < size:
                for j in range(size - i):
                    result[i + j] = pow(A[i + j], exponent)

        base_idx += stride * CHUNK_SIZE


struct ScalarOperations[dtype: DType = DType.float32](
    ImplicitlyCopyable & Movable
):
    @staticmethod
    fn launch[
        op_code: Int,
    ](A: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]) raises -> NDBuffer[
        Self.dtype
    ]:
        var numels = A.numels()

        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )
        ref A_device_state = A.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()

        var compiled_func = device_context.compile_function[
            scalar_ops[
                op_code=op_code,
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
            scalar_ops[
                op_code=op_code,
                dtype=Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        ref A_buffer = A_device_state.device_buffer()
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            scalar,
            numels,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()
        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        var out = NDBuffer[Self.dtype].with_device_state(device_state^, A.shape)

        return out^

    @staticmethod
    fn launch_pow(
        A: NDBuffer[Self.dtype], exponent: Scalar[Self.dtype]
    ) raises -> NDBuffer[Self.dtype]:
        """
        Dedicated POW launcher — dispatches to typed f32/f64 kernels.
        POW only supported for float32 and float64.
        """
        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.dtype]()
        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )
        ref A_device_state = A.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()

        # Ensure contiguous GPU input
        var contig_state = A.contiguous_device_state()
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )

        comptime if Self.dtype == DType.float32:
            var compiled = device_context.compile_function[
                pow_op_f32[
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
                pow_op_f32[
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
            ]()
            device_context.enqueue_function(
                compiled,
                result_buffer,
                contig_state.device_buffer(),
                exponent.cast[DType.float32](),
                numels,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        elif Self.dtype == DType.float64:
            var compiled = device_context.compile_function[
                pow_op_f64[
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
                pow_op_f64[
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
            ]()
            device_context.enqueue_function(
                compiled,
                result_buffer,
                contig_state.device_buffer(),
                exponent.cast[DType.float64](),
                numels,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        else:
            panic(
                "ScalarOperations: POW only supported for float32 and float64"
            )

        device_context.synchronize()
        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(device_state^, A.shape)

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


from std.testing import assert_true
from common_utils import now
from std.sys import has_accelerator


fn main() raises:
    var SIZE = 65536 * 10
    comptime dtype = DType.float32
    # Test CPU pow forward
    var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var b_cpu = a_cpu**2.0
    assert_true(b_cpu.all_close(Tensor[dtype].d1([4.0, 9.0, 16.0])))

    # Test CPU pow backward — ∂(x²)/∂x = 2x
    var loss_cpu = b_cpu.sum()
    loss_cpu.backward()
    assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([4.0, 6.0, 8.0])))

    comptime if has_accelerator():
        # Test GPU pow forward
        var a_gpu = (
            Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True).to_gpu()
        )
        var b_gpu = a_gpu**2.0
        assert_true(
            b_gpu.to_cpu().all_close(Tensor[dtype].d1([4.0, 9.0, 16.0]))
        )

        # Test GPU pow backward
        var loss_gpu = b_gpu.sum()
        loss_gpu.backward()
        assert_true(
            a_gpu.grad().to_cpu().all_close(Tensor[dtype].d1([4.0, 6.0, 8.0]))
        )

    print("Exponentiator tests passed!")
    var tensor_A = Tensor[dtype].ones(SIZE, requires_grad=True)
    var tensor_a = tensor_A.to_gpu()
    var start = now()
    # var expect = (tensor_A * 42) + 2
    var expect = tensor_A + 2
    print("CPU mul took: ", (now() - start) * 1000, "ms")
    # First test
    start = now()
    # var result = (tensor_a * 42) + 2
    var result = tensor_a + 2
    result = result.to_cpu()
    print("GPU mul took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))
    result.backward()
    tensor_A.grad().print()
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
