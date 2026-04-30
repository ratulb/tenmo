from std.sys import simd_width_of
from std.gpu import thread_idx, block_idx, block_dim, grid_dim
from .mnemonics import (
    Add,
    Multiply,
    Subtract,
    Divide,
    max_rank,
    SIGMOID_BACKWARD,
    TANH_BACKWARD,
    LOG_BACKWARD,
    SQRT_BACKWARD,
)
from .strides import Strides
from .broadcasthelper import ShapeBroadcaster
from .device import DeviceState
from .array import Array
from .ndbuffer import NDBuffer
from .common_utils import One, Epsilon
from std.math import rsqrt


fn arithmetic_ops_both_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)
    var one: SIMD[dtype, simd_width]
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var vec_b = B.load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width]

                comptime if dtype.is_floating_point():
                    one = SIMD[dtype, simd_width](1.0)
                else:
                    one = SIMD[dtype, simd_width](1)

                comptime if op_code == Add:
                    vec_result = vec_a + vec_b
                elif op_code == Subtract:
                    vec_result = vec_a - vec_b
                elif op_code == Multiply:
                    vec_result = vec_a * vec_b
                elif op_code == Divide:
                    vec_result = vec_a / vec_b
                elif op_code == SIGMOID_BACKWARD:
                    vec_result = vec_b * vec_a * (one - vec_a)
                elif op_code == TANH_BACKWARD:
                    vec_result = vec_b * (one - vec_a * vec_a)
                elif op_code == SQRT_BACKWARD:
                    vec_result = vec_b * (
                        SIMD[dtype, simd_width](0.5)
                        #* rsqrt(vec_a + epsilon)
                        * rsqrt(max(vec_a, SIMD[dtype, simd_width](0)))
                        # / (epsilon + SIMD[dtype, simd_width](2) * sqrt(vec_a))
                    )

                else:  # Log backward
                    vec_result = vec_b / max(vec_a, epsilon)
                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var idx = i + j
                    var a = A[idx]
                    var b = B[idx]
                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = a + b
                    elif op_code == Subtract:
                        res = a - b
                    elif op_code == Multiply:
                        res = a * b
                    elif op_code == Divide:
                        res = a / b
                    elif op_code == SIGMOID_BACKWARD:
                        res = b * a * (One[dtype].value() - a)
                    elif op_code == TANH_BACKWARD:
                        res = b * (One[dtype].value() - a * a)
                    elif op_code == SQRT_BACKWARD:
                        res = b * (
                            Scalar[dtype](0.5)
                            #* rsqrt(a + epsilon)
                            * rsqrt(max(a, Scalar[dtype](0)))
                            # / (epsilon + Scalar[dtype](2) * sqrt(a))
                        )

                    else:
                        res = b / max(a, epsilon)
                    result[idx] = res

        base_idx += grid_stride * CHUNK_SIZE


fn arithmetic_ops_A_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: Array,
    B_strides: Array,
    size: Int,
    rank: Int,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
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
                var vec_a = A.load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width] = 0

                comptime for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx
                    var b_idx = 0

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
                    elif op_code == Divide:
                        vec_result[lane] = vec_a[lane] / B[b_idx]
                    elif op_code == SIGMOID_BACKWARD:
                        vec_result[lane] = (
                            B[b_idx]
                            * vec_a[lane]
                            * (One[dtype].value() - vec_a[lane])
                        )
                    elif op_code == TANH_BACKWARD:
                        vec_result[lane] = B[b_idx] * (
                            One[dtype].value() - vec_a[lane] * vec_a[lane]
                        )
                    elif op_code == SQRT_BACKWARD:
                        vec_result[lane] = B[b_idx] * (
                            Scalar[dtype](0.5)
                            #* rsqrt(vec_a[lane] + epsilon)
                            * rsqrt(max(vec_a[lane], Scalar[dtype](0)))
                            # / (epsilon + Scalar[dtype](2) * sqrt(vec_a[lane]))
                        )

                    else:  # Log backward
                        vec_result[lane] = B[b_idx] / max(vec_a[lane], epsilon)

                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var b_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    var a = A[linear_idx]
                    var b = B[b_idx]
                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = a + b
                    elif op_code == Subtract:
                        res = a - b
                    elif op_code == Multiply:
                        res = a * b
                    elif op_code == Divide:
                        res = a / b
                    elif op_code == SIGMOID_BACKWARD:
                        res = b * a * (One[dtype].value() - a)
                    elif op_code == TANH_BACKWARD:
                        res = b * (One[dtype].value() - a * a)
                    elif op_code == SQRT_BACKWARD:
                        res = b * (
                            Scalar[dtype](0.5)
                            #* rsqrt(a + epsilon)
                            * rsqrt(max(a, Scalar[dtype](0)))
                            # / (epsilon + Scalar[dtype](2) * sqrt(a))
                        )

                    else:  # Log backward
                        res = b / max(a, epsilon)

                    result[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


fn arithmetic_ops_B_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: Array,
    A_strides: Array,
    size: Int,
    rank: Int,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
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
                var vec_b = B.load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width] = 0

                comptime for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx
                    var a_idx = 0

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
                    elif op_code == Divide:
                        vec_result[lane] = A[a_idx] / vec_b[lane]
                    elif op_code == SIGMOID_BACKWARD:
                        vec_result[lane] = (
                            vec_b[lane]
                            * A[a_idx]
                            * (One[dtype].value() - A[a_idx])
                        )
                    elif op_code == TANH_BACKWARD:
                        vec_result[lane] = vec_b[lane] * (
                            One[dtype].value() - A[a_idx] * A[a_idx]
                        )
                    elif op_code == SQRT_BACKWARD:
                        vec_result[lane] = vec_b[lane] * (
                            Scalar[dtype](0.5)
                            #* rsqrt(A[a_idx] + epsilon)
                            * rsqrt(max(A[a_idx], Scalar[dtype](0)))
                            # / (epsilon + Scalar[dtype](2) * sqrt(A[a_idx]))
                        )

                    else:  # Log backward
                        vec_result[lane] = vec_b[lane] / max(A[a_idx], epsilon)

                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var a_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        remaining //= result_shape[dim]

                    var a = A[a_idx]
                    var b = B[linear_idx]
                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = a + b
                    elif op_code == Subtract:
                        res = a - b
                    elif op_code == Multiply:
                        res = a * b
                    elif op_code == Divide:
                        res = a / b
                    elif op_code == SIGMOID_BACKWARD:
                        res = b * a * (One[dtype].value() - a)
                    elif op_code == TANH_BACKWARD:
                        res = b * (One[dtype].value() - a * a)
                    elif op_code == SQRT_BACKWARD:
                        res = b * (
                            Scalar[dtype](0.5)
                            #* rsqrt(a + epsilon)
                            * rsqrt(max(a, Scalar[dtype](0)))
                            # / (epsilon + Scalar[dtype](2) * sqrt(a))
                        )

                    else:
                        res = b / max(a, epsilon)

                    result[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


fn arithmetic_ops_both_strided[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: Array,
    A_strides: Array,
    B_strides: Array,
    size: Int,
    rank: Int,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
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
                    var a_idx = 0
                    var b_idx = 0

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
                    elif op_code == Divide:
                        vec_result[lane] = A[a_idx] / B[b_idx]
                    elif op_code == SIGMOID_BACKWARD:
                        vec_result[lane] = (
                            B[b_idx]
                            * A[a_idx]
                            * (One[dtype].value() - A[a_idx])
                        )
                    elif op_code == TANH_BACKWARD:
                        vec_result[lane] = B[b_idx] * (
                            One[dtype].value() - A[a_idx] * A[a_idx]
                        )
                    elif op_code == SQRT_BACKWARD:
                        _ = """vec_result[lane] = B[b_idx] * (
                            Scalar[dtype](1)
                            / (epsilon + Scalar[dtype](2) * sqrt(A[a_idx]))
                        )"""
                        vec_result[lane] = B[b_idx] * (
                            #Scalar[dtype](0.5) * rsqrt(A[a_idx] + epsilon)
                            Scalar[dtype](0.5) * rsqrt(max(A[a_idx], Scalar[dtype](0)))
                        )

                    else:  # Log backward
                        vec_result[lane] = B[b_idx] / max(A[a_idx], epsilon)

                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var a_idx = 0
                    var b_idx = 0

                    for dim in range(rank - 1, -1, -1):
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    var a = A[a_idx]
                    var b = B[b_idx]
                    var res: Scalar[dtype]

                    comptime if op_code == Add:
                        res = a + b
                    elif op_code == Subtract:
                        res = a - b
                    elif op_code == Multiply:
                        res = a * b
                    elif op_code == Divide:
                        res = a / b
                    elif op_code == SIGMOID_BACKWARD:
                        res = b * a * (One[dtype].value() - a)
                    elif op_code == TANH_BACKWARD:
                        res = b * (One[dtype].value() - a * a)
                    elif op_code == SQRT_BACKWARD:
                        res = b * (
                            Scalar[dtype](0.5)
                            #* rsqrt(a + epsilon)
                            * rsqrt(max(a, Scalar[dtype](0)))
                            # / (epsilon + Scalar[dtype](2) * sqrt(a))
                        )

                    else:  # Log backward
                        res = b / max(a, epsilon)

                    result[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


@fieldwise_init
struct BinaryOperations[dtype: DType = DType.float32](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    fn launch[
        op_code: Int,
    ](
        A: NDBuffer[Self.dtype],
        B: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) raises -> NDBuffer[Self.dtype]:
        comptime simdwidth = simd_width_of[Self.dtype]()
        var A_shape = A.shape
        var B_shape = B.shape

        var broadcast_shape = ShapeBroadcaster.broadcast_shape(A_shape, B_shape)
        var output_size = broadcast_shape.product()
        var rank = broadcast_shape.rank()

        var (threads_per_block, num_blocks) = Self.launch_config(output_size)

        var needs_broadcasting = (
            A_shape != broadcast_shape or B_shape != broadcast_shape
        )

        ref A_device_state = A.device_state.value()
        ref B_device_state = B.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            output_size
        )

        ref A_buffer = A_device_state.device_buffer()
        ref B_buffer = B_device_state.device_buffer()

        var A_is_contiguous = A.is_contiguous()
        var B_is_contiguous = B.is_contiguous()
        # ================================================================
        # PATH 1: Both contiguous, same shape, no broadcasting
        # ================================================================
        if (
            A_shape == B_shape
            and A_is_contiguous
            and B_is_contiguous
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
                result_buffer,
                A_buffer,
                B_buffer,
                output_size,
                epsilon,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
            device_context.synchronize()
            var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
            return NDBuffer[Self.dtype].with_device_state(
                device_state^, broadcast_shape
            )

        # Broadcast strides needed for all remaining paths
        var A_broadcast_strides = ShapeBroadcaster.broadcast_strides(
            A_shape, Strides.default(A_shape), broadcast_shape
        )
        var B_broadcast_strides = ShapeBroadcaster.broadcast_strides(
            B_shape, Strides.default(B_shape), broadcast_shape
        )

        # ================================================================
        # PATH 2: A contiguous and not broadcast-expanded, B strided/broadcast
        # FIX: guard A_shape == broadcast_shape so A's linear index is valid
        # ================================================================
        if (
            A_is_contiguous
            and A_shape == broadcast_shape
            and not B_is_contiguous
        ):
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
                result_buffer,
                A_buffer,
                B_buffer,
                broadcast_shape.array(),
                B_broadcast_strides.array(),
                output_size,
                rank,
                epsilon,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
            device_context.synchronize()
            var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
            return NDBuffer[Self.dtype].with_device_state(
                device_state^, broadcast_shape
            )

        # ================================================================
        # PATH 3: B contiguous and not broadcast-expanded, A strided/broadcast
        # FIX: guard B_shape == broadcast_shape so B's linear index is valid
        # ================================================================
        if (
            B_is_contiguous
            and B_shape == broadcast_shape
            and not A_is_contiguous
        ):
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
                result_buffer,
                A_buffer,
                B_buffer,
                broadcast_shape.array(),
                A_broadcast_strides.array(),
                output_size,
                rank,
                epsilon,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
            device_context.synchronize()
            var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
            return NDBuffer[Self.dtype].with_device_state(
                device_state^, broadcast_shape
            )

        # ================================================================
        # PATH 4: Both strided, or either operand needs broadcasting.
        # Handles all remaining cases correctly via stride decomposition.
        # ================================================================
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
            result_buffer,
            A_buffer,
            B_buffer,
            broadcast_shape.array(),
            A_broadcast_strides.array(),
            B_broadcast_strides.array(),
            output_size,
            rank,
            epsilon,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        device_context.synchronize()
        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(
            device_state^, broadcast_shape
        )

    @staticmethod
    fn launch_config(output_size: Int) -> Tuple[Int, Int]:
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
