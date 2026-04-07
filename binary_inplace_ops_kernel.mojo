from std.sys import simd_width_of
from std.gpu import thread_idx, block_idx, block_dim, grid_dim
from std.memory import stack_allocation, AddressSpace
from mnemonics import Add, Multiply, Subtract, Divide
from tenmo import Tensor
from strides import Strides
from broadcasthelper import ShapeBroadcaster
from device import DeviceState
from array import Array
from ndbuffer import NDBuffer


fn arithmetic_ops_both_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    A_offset: Int,
    B_offset: Int,
    size: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](A_offset + i)
                var vec_b = B.load[width=simd_width](B_offset + i)
                var vec_result: SIMD[dtype, simd_width]

                comptime if op_code == Add:
                    vec_result = vec_a + vec_b
                elif op_code == Subtract:
                    vec_result = vec_a - vec_b
                elif op_code == Multiply:
                    vec_result = vec_a * vec_b
                else:
                    vec_result = vec_a / vec_b

                A.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var idx = i + j
                    var res: Scalar[dtype] = 0

                    comptime if op_code == Add:
                        res = A[A_offset + idx] + B[B_offset + idx]
                    elif op_code == Subtract:
                        res = A[A_offset + idx] - B[B_offset + idx]
                    elif op_code == Multiply:
                        res = A[A_offset + idx] * B[B_offset + idx]
                    else:
                        res = A[A_offset + idx] / B[B_offset + idx]

                    A[idx] = res

        base_idx += grid_stride * CHUNK_SIZE


fn arithmetic_ops_A_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    A_offset: Int,
    result_shape: Array,
    B_strides: Array,
    B_offset: Int,
    size: Int,
    rank: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](A_offset + i)
                var vec_result: SIMD[dtype, simd_width] = 0

                comptime for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx
                    var b_idx = B_offset

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    comptime if op_code == Add:
                        vec_result[lane] = vec_a[lane] + B[b_idx]
                    elif op_code == Subtract:
                        vec_result[lane] = vec_a[lane] - B[b_idx]
                    elif op_code == Multiply:
                        vec_result[lane] = vec_a[lane] * B[b_idx]
                    else:
                        vec_result[lane] = vec_a[lane] / B[b_idx]

                A.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var b_idx = B_offset

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    var res: Scalar[dtype] = 0

                    comptime if op_code == Add:
                        res = A[A_offset + linear_idx] + B[b_idx]
                    elif op_code == Subtract:
                        res = A[A_offset + linear_idx] - B[b_idx]
                    elif op_code == Multiply:
                        res = A[A_offset + linear_idx] * B[b_idx]
                    else:
                        res = A[A_offset + linear_idx] / B[b_idx]

                    A[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


fn arithmetic_ops_B_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: Array,
    A_strides: Array,
    A_offset: Int,
    B_offset: Int,
    size: Int,
    rank: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)  # total threads in grid

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    # Grid-stride loop over CHUNK_SIZE blocks
    var base_idx = gtid * CHUNK_SIZE
    while base_idx < size:

        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                # B: Vectorized load (contiguous)
                var vec_b = B.load[width=simd_width](B_offset + i)
                var vec_result: SIMD[dtype, simd_width] = 0

                comptime for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx

                    # Coordinate decomposition
                    var a_idx = A_offset
                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        remaining //= result_shape[dim]

                    comptime if op_code == Add:
                        vec_result[lane] = A[a_idx] + vec_b[lane]
                    elif op_code == Subtract:
                        vec_result[lane] = A[a_idx] - vec_b[lane]
                    elif op_code == Multiply:
                        vec_result[lane] = A[a_idx] * vec_b[lane]
                    else:
                        vec_result[lane] = A[a_idx] / vec_b[lane]

                A.store[width=simd_width](i, vec_result)

            else:
                # Scalar tail
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var a_idx = A_offset

                    for dim in range(Int(rank) - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        remaining //= result_shape[dim]

                    var res: Scalar[dtype] = 0

                    comptime if op_code == Add:
                        res = A[a_idx] + B[B_offset + linear_idx]
                    elif op_code == Subtract:
                        res = A[a_idx] - B[B_offset + linear_idx]
                    elif op_code == Multiply:
                        res = A[a_idx] * B[B_offset + linear_idx]
                    else:
                        res = A[a_idx] / B[B_offset + linear_idx]

                    A[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


fn arithmetic_ops_both_strided[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: Array,
    A_strides: Array,
    B_strides: Array,
    A_offset: Int,
    B_offset: Int,
    size: Int,
    rank: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_result: SIMD[dtype, simd_width] = 0

                comptime for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx
                    var a_idx = A_offset
                    var b_idx = B_offset

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    comptime if op_code == Add:
                        vec_result[lane] = A[a_idx] + B[b_idx]
                    elif op_code == Subtract:
                        vec_result[lane] = A[a_idx] - B[b_idx]
                    elif op_code == Multiply:
                        vec_result[lane] = A[a_idx] * B[b_idx]
                    else:
                        vec_result[lane] = A[a_idx] / B[b_idx]

                A.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var a_idx = A_offset
                    var b_idx = B_offset

                    for dim in range(Int(rank) - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    var res: Scalar[dtype] = 0

                    comptime if op_code == Add:
                        res = A[a_idx] + B[b_idx]
                    elif op_code == Subtract:
                        res = A[a_idx] - B[b_idx]
                    elif op_code == Multiply:
                        res = A[a_idx] * B[b_idx]
                    else:
                        res = A[a_idx] / B[b_idx]

                    A[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


@fieldwise_init
struct BinaryInplaceOperations[dtype: DType](
    RegisterPassable, ImplicitlyCopyable
):
    @staticmethod
    fn launch[
        op_code: Int,
    ](A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]) raises:
        comptime simdwidth = simd_width_of[Self.dtype]()
        var A_shape = A.shape
        var B_shape = B.shape

        var broadcast_shape = ShapeBroadcaster.broadcast_shape(A_shape, B_shape)
        var output_size = broadcast_shape.product()

        # Launch configuration
        var (threads_per_block, num_blocks) = Self.launch_config(output_size)

        # Check if broadcasting is needed
        var needs_broadcasting = (
            A_shape != broadcast_shape or B_shape != broadcast_shape
        )

        ref A_device_state = A.device_state.value()
        ref B_device_state = B.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()

        ref A_buffer = A_device_state.device_buffer()
        ref B_buffer = B_device_state.device_buffer()

        # ================================================================
        # PATH 1: Both contiguous, same shape, no broadcasting
        # ================================================================
        if (
            A_shape == B_shape
            and A.is_contiguous()
            and B.is_contiguous()
            and not needs_broadcasting
        ):
            var compiled_func = device_context.compile_function[
                arithmetic_ops_both_contiguous[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
                arithmetic_ops_both_contiguous[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
            ]()

            device_context.enqueue_function(
                compiled_func,
                A_buffer,
                B_buffer,
                0,
                0,
                output_size,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

            device_context.synchronize()
            return

        # Prepare for strided kernels
        var rank = broadcast_shape.rank()
        var A_broadcast_strides = ShapeBroadcaster.broadcast_strides(
            A_shape, Strides.default(A_shape), broadcast_shape
        )
        var B_broadcast_strides = ShapeBroadcaster.broadcast_strides(
            B_shape, Strides.default(B_shape), broadcast_shape
        )

        var A_is_contiguous = A.is_contiguous()
        var B_is_contiguous = B.is_contiguous()

        # ================================================================
        # PATH 2: A contiguous, B strided
        # ================================================================
        if A_is_contiguous and not B_is_contiguous:
            var compiled_func = device_context.compile_function[
                arithmetic_ops_A_contiguous[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
                arithmetic_ops_A_contiguous[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
            ]()

            device_context.enqueue_function(
                compiled_func,
                A_buffer,
                B_buffer,
                0,
                broadcast_shape.array(),
                B_broadcast_strides.array(),
                0,
                output_size,
                rank,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

            device_context.synchronize()
        # ================================================================
        # PATH 3: A strided, B contiguous
        # ================================================================
        elif not A_is_contiguous and B_is_contiguous:
            var compiled_func = device_context.compile_function[
                arithmetic_ops_B_contiguous[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
                arithmetic_ops_B_contiguous[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
            ]()

            device_context.enqueue_function(
                compiled_func,
                A_buffer,
                B_buffer,
                broadcast_shape.array(),
                A_broadcast_strides.array(),
                0,
                0,
                output_size,
                rank,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

            device_context.synchronize()
        # ================================================================
        # PATH 4: Both strided (or broadcasting)
        # ================================================================

        else:
            var compiled_func = device_context.compile_function[
                arithmetic_ops_both_strided[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
                arithmetic_ops_both_strided[
                    op_code, Self.dtype, simdwidth, 2 * simdwidth
                ],
            ]()

            device_context.enqueue_function(
                compiled_func,
                A_buffer,
                B_buffer,
                broadcast_shape.array(),
                A_broadcast_strides.array(),
                B_broadcast_strides.array(),
                0,
                0,
                output_size,
                rank,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

            device_context.synchronize()

    @staticmethod
    fn launch_config(output_size: Int) -> Tuple[Int, Int]:
        """Launch configuration."""
        var threads_per_block: Int
        var num_blocks: Int

        if output_size < 4096:
            threads_per_block = 128
            num_blocks = (output_size + 127) // 128
        elif output_size < 65536:
            threads_per_block = 256
            num_blocks = min((output_size + 255) // 256, 128)
        else:
            threads_per_block = 512
            num_blocks = min((output_size + 511) // 512, 512)

        return num_blocks, threads_per_block


fn main() raises:
    pass
