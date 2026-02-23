from sys import simd_width_of
from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from memory import stack_allocation, AddressSpace
from mnemonics import Add, Multiply, Subtract, Divide
from tenmo import Tensor
from shapes import Shape
from strides import Strides
from broadcasthelper import ShapeBroadcaster
from device import GPU
from array import Array

comptime MAX_RANK = 8


fn arithmetic_ops_both_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
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

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](A_offset + i)
                var vec_b = B.load[width=simd_width](B_offset + i)
                var vec_result: SIMD[dtype, simd_width] = 0

                @parameter
                if op_code == Add:
                    vec_result = vec_a + vec_b
                elif op_code == Subtract:
                    vec_result = vec_a - vec_b
                elif op_code == Multiply:
                    vec_result = vec_a * vec_b
                else:
                    vec_result = vec_a / vec_b

                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var idx = i + j
                    var res: Scalar[dtype] = 0

                    @parameter
                    if op_code == Add:
                        res = A[A_offset + idx] + B[B_offset + idx]
                    elif op_code == Subtract:
                        res = A[A_offset + idx] - B[B_offset + idx]
                    elif op_code == Multiply:
                        res = A[A_offset + idx] * B[B_offset + idx]
                    else:
                        res = A[A_offset + idx] / B[B_offset + idx]

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
    A_offset: Int,
    # result_shape: UnsafePointer[Int64, ImmutAnyOrigin],
    result_shape: Array,
    # B_strides: UnsafePointer[Int64, ImmutAnyOrigin],
    B_strides: Array,
    B_offset: Int,
    size: Int,
    rank: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    _ = """var shape_local = stack_allocation[
        MAX_RANK, Int, address_space = AddressSpace.SHARED
    ]()
    var strides_B_local = stack_allocation[
        MAX_RANK, Int, address_space = AddressSpace.SHARED
    ]()

    for i in range(rank):
        shape_local[i] = Int(result_shape[i + 2])
        strides_B_local[i] = Int(B_strides[i + 2])"""

    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](A_offset + i)
                var vec_result: SIMD[dtype, simd_width] = 0

                @parameter
                for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx
                    var b_idx = B_offset

                    for dim in range(rank - 1, -1, -1):
                        # var coord = remaining % shape_local[dim]
                        var coord = remaining % result_shape[dim]
                        # b_idx += coord * strides_B_local[dim]
                        b_idx += coord * B_strides[dim]
                        # remaining //= shape_local[dim]
                        remaining //= result_shape[dim]

                    @parameter
                    if op_code == Add:
                        vec_result[lane] = vec_a[lane] + B[b_idx]
                    elif op_code == Subtract:
                        vec_result[lane] = vec_a[lane] - B[b_idx]
                    elif op_code == Multiply:
                        vec_result[lane] = vec_a[lane] * B[b_idx]
                    else:
                        vec_result[lane] = vec_a[lane] / B[b_idx]

                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var b_idx = B_offset

                    for dim in range(rank - 1, -1, -1):
                        # var coord = remaining % shape_local[dim]
                        var coord = remaining % result_shape[dim]
                        # b_idx += coord * strides_B_local[dim]
                        b_idx += coord * B_strides[dim]
                        # remaining //= shape_local[dim]
                        remaining //= result_shape[dim]

                    var res: Scalar[dtype] = 0

                    @parameter
                    if op_code == Add:
                        res = A[A_offset + linear_idx] + B[b_idx]
                    elif op_code == Subtract:
                        res = A[A_offset + linear_idx] - B[b_idx]
                    elif op_code == Multiply:
                        res = A[A_offset + linear_idx] * B[b_idx]
                    else:
                        res = A[A_offset + linear_idx] / B[b_idx]

                    result[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE

        _ = """fn arithmetic_ops_B_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    result_shape: UnsafePointer[Int64, ImmutAnyOrigin],
    A_strides: UnsafePointer[Int64, ImmutAnyOrigin],
    A_offset: Int,
    B_offset: Int,
    size: Int,
    rank: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    var shape_local = stack_allocation[
        MAX_RANK, Int, address_space = AddressSpace.SHARED
    ]()
    var strides_A_local = stack_allocation[
        MAX_RANK, Int, address_space = AddressSpace.SHARED
    ]()
    var coords = stack_allocation[
        MAX_RANK, Int, address_space = AddressSpace.SHARED
    ]()

    for i in range(rank):
        shape_local[i] = Int(result_shape[i + 2])
        strides_A_local[i] = Int(A_strides[i + 2])

    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        # ---- compute starting coords once ----
        var tmp = base_idx
        var a_idx = A_offset

        for dim in range(rank - 1, -1, -1):
            coords[dim] = tmp % shape_local[dim]
            tmp //= shape_local[dim]
            a_idx += coords[dim] * strides_A_local[dim]

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_b = B.load[width=simd_width](B_offset + i)
                var vec_result: SIMD[dtype, simd_width] = 0

                @parameter
                for lane in range(simd_width):

                    @parameter
                    if op_code == Add:
                        vec_result[lane] = A[a_idx] + vec_b[lane]
                    elif op_code == Subtract:
                        vec_result[lane] = A[a_idx] - vec_b[lane]
                    elif op_code == Multiply:
                        vec_result[lane] = A[a_idx] * vec_b[lane]
                    else:
                        vec_result[lane] = A[a_idx] / vec_b[lane]

                    # increment odometer
                    for dim in range(rank - 1, -1, -1):
                        coords[dim] += 1
                        a_idx += strides_A_local[dim]

                        if coords[dim] < shape_local[dim]:
                            break
                        else:
                            coords[dim] = 0
                            a_idx -= strides_A_local[dim] * shape_local[dim]

                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):

                    @parameter
                    if op_code == Add:
                        result[i + j] = A[a_idx] + B[B_offset + i + j]
                    elif op_code == Subtract:
                        result[i + j] = A[a_idx] - B[B_offset + i + j]
                    elif op_code == Multiply:
                        result[i + j] = A[a_idx] * B[B_offset + i + j]
                    else:
                        result[i + j] = A[a_idx] / B[B_offset + i + j]

                    # increment odometer
                    for dim in range(rank - 1, -1, -1):
                        coords[dim] += 1
                        a_idx += strides_A_local[dim]

                        if coords[dim] < shape_local[dim]:
                            break
                        else:
                            coords[dim] = 0
                            a_idx -= strides_A_local[dim] * shape_local[dim]

                break

        base_idx += grid_stride * CHUNK_SIZE"""


fn arithmetic_ops_B_contiguous[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    # result_shape: UnsafePointer[Int64, ImmutAnyOrigin],
    result_shape: Array,
    # A_strides: UnsafePointer[Int64, ImmutAnyOrigin],
    A_strides: Array,
    A_offset: Int,
    B_offset: Int,
    size: Int,
    rank: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)  # total threads in grid

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    # Read metadata into stack once per thread
    _ = """var coords = stack_allocation[
        MAX_RANK, Int, address_space = AddressSpace.SHARED
    ]()
    var shape_local = stack_allocation[
        MAX_RANK, Int, address_space = AddressSpace.SHARED
    ]()
    var strides_A_local = stack_allocation[
        MAX_RANK, Int, address_space = AddressSpace.SHARED
    ]()

    for i in range(rank):
        shape_local[i] = Int(result_shape[i + 2])
        strides_A_local[i] = Int(A_strides[i + 2])"""

    # Grid-stride loop over CHUNK_SIZE blocks
    var base_idx = gtid * CHUNK_SIZE
    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break  # @parameter for allows break unlike return-mid-kernel

            if i + simd_width <= size:
                # B: Vectorized load (contiguous)
                var vec_b = B.load[width=simd_width](B_offset + i)
                var vec_result: SIMD[dtype, simd_width] = 0

                @parameter
                for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx

                    # Coordinate decomposition
                    var a_idx = A_offset
                    for dim in range(rank - 1, -1, -1):
                        # var coord = remaining % shape_local[dim]
                        var coord = remaining % result_shape[dim]
                        # a_idx += coord * strides_A_local[dim]
                        a_idx += coord * A_strides[dim]
                        # remaining //= shape_local[dim]
                        remaining //= result_shape[dim]

                    @parameter
                    if op_code == Add:
                        vec_result[lane] = A[a_idx] + vec_b[lane]
                    elif op_code == Subtract:
                        vec_result[lane] = A[a_idx] - vec_b[lane]
                    elif op_code == Multiply:
                        vec_result[lane] = A[a_idx] * vec_b[lane]
                    else:
                        vec_result[lane] = A[a_idx] / vec_b[lane]

                result.store[width=simd_width](i, vec_result)

            else:
                # Scalar tail
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var a_idx = A_offset

                    for dim in range(Int(rank) - 1, -1, -1):
                        # var coord = remaining % shape_local[dim]
                        var coord = remaining % result_shape[dim]
                        # a_idx += coord * strides_A_local[dim]
                        a_idx += coord * A_strides[dim]
                        # remaining //= shape_local[dim]
                        remaining //= result_shape[dim]

                    var res: Scalar[dtype] = 0

                    @parameter
                    if op_code == Add:
                        res = A[a_idx] + B[B_offset + linear_idx]
                    elif op_code == Subtract:
                        res = A[a_idx] - B[B_offset + linear_idx]
                    elif op_code == Multiply:
                        res = A[a_idx] * B[B_offset + linear_idx]
                    else:
                        res = A[a_idx] / B[B_offset + linear_idx]

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
    # result_shape: UnsafePointer[Int64, ImmutAnyOrigin],
    result_shape: Array,
    # A_strides: UnsafePointer[Int64, ImmutAnyOrigin],
    A_strides: Array,
    # B_strides: UnsafePointer[Int64, ImmutAnyOrigin],
    B_strides: Array,
    A_offset: Int,
    B_offset: Int,
    size: Int,
    rank: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    _ = """var shape_local = stack_allocation[
        MAX_RANK, Int, address_space = AddressSpace.SHARED
    ]()
    var strides_A_local = stack_allocation[
        MAX_RANK, Int, address_space = AddressSpace.SHARED
    ]()
    var strides_B_local = stack_allocation[
        MAX_RANK, Int, address_space = AddressSpace.SHARED
    ]()

    for i in range(rank):
        shape_local[i] = Int(result_shape[i + 2])
        strides_A_local[i] = Int(A_strides[i + 2])
        strides_B_local[i] = Int(B_strides[i + 2])"""

    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_result: SIMD[dtype, simd_width] = 0

                @parameter
                for lane in range(simd_width):
                    var linear_idx = i + lane
                    var remaining = linear_idx
                    var a_idx = A_offset
                    var b_idx = B_offset

                    for dim in range(rank - 1, -1, -1):
                        # var coord = remaining % shape_local[dim]
                        var coord = remaining % result_shape[dim]
                        # a_idx += coord * strides_A_local[dim]
                        a_idx += coord * A_strides[dim]
                        # b_idx += coord * strides_B_local[dim]
                        b_idx += coord * B_strides[dim]
                        # remaining //= shape_local[dim]
                        remaining //= result_shape[dim]

                    @parameter
                    if op_code == Add:
                        vec_result[lane] = A[a_idx] + B[b_idx]
                    elif op_code == Subtract:
                        vec_result[lane] = A[a_idx] - B[b_idx]
                    elif op_code == Multiply:
                        vec_result[lane] = A[a_idx] * B[b_idx]
                    else:
                        vec_result[lane] = A[a_idx] / B[b_idx]

                result.store[width=simd_width](i, vec_result)

            else:
                for j in range(size - i):
                    var linear_idx = i + j
                    var remaining = linear_idx
                    var a_idx = A_offset
                    var b_idx = B_offset

                    for dim in range(Int(rank) - 1, -1, -1):
                        _ = """var coord = remaining % shape_local[dim]
                        a_idx += coord * strides_A_local[dim]
                        b_idx += coord * strides_B_local[dim]
                        remaining //= shape_local[dim]"""
                        var coord = remaining % result_shape[dim]
                        a_idx += coord * A_strides[dim]
                        b_idx += coord * B_strides[dim]
                        remaining //= result_shape[dim]

                    var res: Scalar[dtype] = 0

                    @parameter
                    if op_code == Add:
                        res = A[a_idx] + B[b_idx]
                    elif op_code == Subtract:
                        res = A[a_idx] - B[b_idx]
                    elif op_code == Multiply:
                        res = A[a_idx] * B[b_idx]
                    else:
                        res = A[a_idx] / B[b_idx]

                    result[linear_idx] = res

        base_idx += grid_stride * CHUNK_SIZE


fn launch[
    op_code: Int,
    dtype: DType = DType.float32,
](A: Tensor[dtype], B: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Select the optimal kernel based on memory layout.
    """

    if not A.broadcastable(B):
        raise Error("Shape mismatch")

    # var ctx = DeviceContext()
    var gpu = GPU()
    var ctx = gpu()
    comptime width_of_simd = simd_width_of[dtype]()

    var broadcast_shape = ShapeBroadcaster.broadcast_shape(A.shape(), B.shape())
    var output_size = broadcast_shape.product()

    # Launch configuration
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

    # Check if broadcasting is needed
    var needs_broadcasting = (
        A.shape() != broadcast_shape or B.shape() != broadcast_shape
    )

    # ================================================================
    # PATH 1: Both contiguous, same shape, no broadcasting
    # ================================================================
    if (
        A.shape() == B.shape()
        and A.is_contiguous()
        and B.is_contiguous()
        and not needs_broadcasting
    ):
        print("[GPU] Using Kernel 1: Both contiguous")

        var compiled_func = ctx.compile_function[
            arithmetic_ops_both_contiguous[
                op_code, dtype, width_of_simd, 2 * width_of_simd
            ],
            arithmetic_ops_both_contiguous[
                op_code, dtype, width_of_simd, 2 * width_of_simd
            ],
        ]()

        var A_buffer = ctx.enqueue_create_buffer[dtype](A.numels())
        var B_buffer = ctx.enqueue_create_buffer[dtype](B.numels())

        var result_buffer = ctx.enqueue_create_buffer[dtype](output_size)
        var start_data_write = now()

        ctx.enqueue_copy(A_buffer, A.data_ptr() + A.offset())
        ctx.enqueue_copy(B_buffer, B.data_ptr() + B.offset())

        print("Data transfer time: ", (now() - start_data_write) * 1000, "ms")
        var start_time = now()
        ctx.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            B_buffer,
            0,  # A.offset(),
            0,  # B.offset(),
            output_size,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        ctx.synchronize()
        print("GPU execution time: ", (now() - start_time) * 1000, "ms")
        var data_retrieve_time = now()
        var out = Tensor[dtype].from_device_buffer(
            result_buffer, broadcast_shape
        )
        print(
            "Data transfer back took: ",
            (now() - data_retrieve_time) * 1000,
            "ms",
        )
        return out^

    # Prepare for strided kernels
    var rank = broadcast_shape.rank()
    var A_broadcast_strides = ShapeBroadcaster.broadcast_strides(
        A.shape(), Strides.default(A.shape()), broadcast_shape
    )
    var B_broadcast_strides = ShapeBroadcaster.broadcast_strides(
        B.shape(), Strides.default(B.shape()), broadcast_shape
    )

    var A_is_contiguous = A.is_contiguous()
    var B_is_contiguous = B.is_contiguous()

    # ================================================================
    # PATH 2: A contiguous, B strided
    # ================================================================
    if A_is_contiguous and not B_is_contiguous:
        print("[GPU] Using Kernel 2: A contiguous, B strided")

        var compiled_func = ctx.compile_function[
            arithmetic_ops_A_contiguous[
                op_code, dtype, width_of_simd, 2 * width_of_simd
            ],
            arithmetic_ops_A_contiguous[
                op_code, dtype, width_of_simd, 2 * width_of_simd
            ],
        ]()

        var A_buffer = ctx.enqueue_create_buffer[dtype](A.numels())
        var B_buffer = ctx.enqueue_create_buffer[dtype](B.numels())
        var result_buffer = ctx.enqueue_create_buffer[dtype](output_size)

        _ = """var result_shape_buffer = ctx.enqueue_create_buffer[DType.int64](
            #broadcast_shape.numels()
            output_size
        )
        var B_strides_buffer = ctx.enqueue_create_buffer[DType.int64](
            B_broadcast_strides.write_length()
        )"""

        ctx.enqueue_copy(A_buffer, A.data_ptr() + A.offset())
        B.write_to(B_buffer)

        # broadcast_shape.write_to_device_buffer(result_shape_buffer)
        # B_broadcast_strides.write_to_device_buffer(B_strides_buffer)

        ctx.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            B_buffer,
            0,  # A.offset(),
            broadcast_shape.array(),
            B.strides().array(),
            0,  # B.offset(),
            output_size,
            rank,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        ctx.synchronize()
        return Tensor[dtype].from_device_buffer(result_buffer, broadcast_shape)

    # ================================================================
    # PATH 3: A strided, B contiguous
    # ================================================================
    if not A_is_contiguous and B_is_contiguous:
        print("[GPU] Using Kernel 3: A strided, B contiguous (MEDIUM-FAST)")

        var compiled_func = ctx.compile_function[
            arithmetic_ops_B_contiguous[
                op_code, dtype, width_of_simd, 2 * width_of_simd
            ],
            arithmetic_ops_B_contiguous[
                op_code, dtype, width_of_simd, 2 * width_of_simd
            ],
        ]()

        var A_buffer = ctx.enqueue_create_buffer[dtype](A.numels())
        var B_buffer = ctx.enqueue_create_buffer[dtype](B.numels())
        var result_buffer = ctx.enqueue_create_buffer[dtype](output_size)

        _ = """var result_shape_buffer = ctx.enqueue_create_buffer[DType.int64](
            broadcast_shape.write_length()
        )
        var A_strides_buffer = ctx.enqueue_create_buffer[DType.int64](
            A_broadcast_strides.write_length()
        )"""

        A.write_to(A_buffer)
        ctx.enqueue_copy(B_buffer, B.data_ptr() + B.offset())

        # broadcast_shape.write_to_device_buffer(result_shape_buffer)
        # A_broadcast_strides.write_to_device_buffer(A_strides_buffer)

        ctx.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            B_buffer,
            broadcast_shape.array(),
            A.strides().array(),
            0,  # A.offset(),
            0,  # B.offset(),
            output_size,
            rank,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        ctx.synchronize()
        return Tensor[dtype].from_device_buffer(result_buffer, broadcast_shape)

    # ================================================================
    # PATH 4: Both strided (or broadcasting)
    # ================================================================

    print("[GPU] Using Kernel 4: Both strided")

    var compiled_func = ctx.compile_function[
        arithmetic_ops_both_strided[
            op_code, dtype, width_of_simd, 2 * width_of_simd
        ],
        arithmetic_ops_both_strided[
            op_code, dtype, width_of_simd, 2 * width_of_simd
        ],
    ]()

    var A_buffer = ctx.enqueue_create_buffer[dtype](A.numels())
    var B_buffer = ctx.enqueue_create_buffer[dtype](B.numels())

    var result_buffer = ctx.enqueue_create_buffer[dtype](output_size)

    _ = """var result_shape_buffer = ctx.enqueue_create_buffer[DType.int64](
        broadcast_shape.write_length()
    )
    var A_strides_buffer = ctx.enqueue_create_buffer[DType.int64](
        A_broadcast_strides.write_length()
    )
    var B_strides_buffer = ctx.enqueue_create_buffer[DType.int64](
        B_broadcast_strides.write_length()
    )"""

    A.write_to(A_buffer)
    B.write_to(B_buffer)

    _ = """broadcast_shape.write_to_device_buffer(result_shape_buffer)
    A_broadcast_strides.write_to_device_buffer(A_strides_buffer)
    B_broadcast_strides.write_to_device_buffer(B_strides_buffer)"""

    ctx.enqueue_function(
        compiled_func,
        result_buffer,
        A_buffer,
        B_buffer,
        broadcast_shape.array(),
        A.strides().array(),
        B.strides().array(),
        0,  # A.offset(),
        0,  # B.offset(),
        output_size,
        rank,
        grid_dim=num_blocks,
        block_dim=threads_per_block,
    )

    ctx.synchronize()
    return Tensor[dtype].from_device_buffer(result_buffer, broadcast_shape)


fn main() raises:
    print("=" * 60)
    print("Production Tensor-Tensor Arithmetic Tests")
    print("With Offset Support")
    print("=" * 60)

    # Original tests
    test_contiguous_same_shape()
    test_non_contiguous()
    test_broadcasting()
    test_scalar_broadcast()
    test_complex_broadcasting()
    test_large_arrays()

    # Offset-specific tests
    test_contiguous_view_with_offset()
    test_all_offset_scenarios()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED (Including Offset Tests)")
    print("=" * 60)


from common_utils import now
from testing import assert_true


fn test_contiguous_same_shape() raises:
    """Test fast path: contiguous, same shape."""
    print("=== Test 1: Contiguous Same Shape ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(1000, 10000)
    var b = Tensor[dtype].rand(1000, 10000)

    var start = now()
    var gpu_result = launch[Multiply, dtype](a, b)
    var gpu_time = (now() - start) * 1000

    start = now()
    var cpu_result = a * b
    var cpu_time = (now() - start) * 1000

    assert_true(gpu_result.all_close(cpu_result))
    print("  GPU:", gpu_time, "ms")
    print("  CPU:", cpu_time, "ms")
    print("  Passed")


fn test_broadcasting() raises:
    """Test broadcasting path."""
    print("\n=== Test 2: Broadcasting ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(3, 1, 4)  # [3, 1, 4]
    var b = Tensor[dtype].rand(1, 2, 4)  # [1, 2, 4]

    var gpu_result = launch[Add, dtype](a, b)  # Result: [3, 2, 4]
    var cpu_result = a + b

    assert_true(gpu_result.shape() == Shape(3, 2, 4))
    assert_true(gpu_result.all_close(cpu_result))
    print("  Shape:", gpu_result.shape())
    print("  Passed")


fn test_scalar_broadcast() raises:
    """Test scalar broadcasting."""
    print("\n=== Test 3: Scalar Broadcast ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(100, 100)
    var b = Tensor[dtype].ones(1, 1) * 42  # Broadcasts to [100, 100]

    var gpu_result = launch[Multiply, dtype](a, b)
    var cpu_result = a * b

    assert_true(gpu_result.all_close(cpu_result))
    print("  Passed")


fn test_non_contiguous() raises:
    """Test non-contiguous tensors (views/transposes)."""
    print("\n=== Test 4: Non-Contiguous (Transpose) ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(30, 20)
    var b = Tensor[dtype].rand(20, 30)
    var a_t = a.transpose(1, 0)  # Non-contiguous view [20, 10]
    var gpu_result = launch[Add, dtype](a_t, b)
    var cpu_result = a_t + b
    assert_true(gpu_result.all_close(cpu_result))
    print("  Passed")


fn test_complex_broadcasting() raises:
    """Test complex multi-dimensional broadcasting."""
    print("\n=== Test 5: Complex Broadcasting ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(1, 5, 1, 7)  # [1, 5, 1, 7]
    var b = Tensor[dtype].rand(3, 1, 4, 1)  # [3, 1, 4, 1]

    var gpu_result = launch[Multiply, dtype](a, b)  # Result: [3, 5, 4, 7]
    var cpu_result = a * b

    assert_true(gpu_result.shape() == Shape(3, 5, 4, 7))
    assert_true(gpu_result.all_close(cpu_result))
    print("  Shape:", gpu_result.shape())
    print("  Passed")


fn test_large_arrays() raises:
    """Stress test with large arrays."""
    print("\n=== Test 6: Large Arrays ===")

    comptime dtype = DType.float32
    var size = 10_000_000  # 10M elements
    var a = Tensor[dtype].rand(size)
    var b = Tensor[dtype].rand(size)

    var start = now()
    var gpu_result = launch[Add, dtype](a, b)
    var gpu_time = (now() - start) * 1000

    start = now()
    var cpu_result = a + b
    var cpu_time = (now() - start) * 1000

    assert_true(gpu_result.all_close(cpu_result))
    print("  Size:", size, "elements")
    print("  GPU:", gpu_time, "ms")
    print("  CPU:", cpu_time, "ms")
    print("  Speedup:", cpu_time / gpu_time, "x")
    print("  Passed")


fn test_contiguous_view_with_offset() raises:
    """Test contiguous view/slice with non-zero offset."""
    print("\n=== Test: Contiguous View with Offset ===")

    comptime dtype = DType.float32

    # Create larger tensors
    var a_full = Tensor[dtype].rand(1000)
    var b_full = Tensor[dtype].rand(1000)

    # Create contiguous slices with offsets
    # e.g., a_full[100:600] is contiguous but has offset 100
    var a = a_full[100:600]  # Contiguous, offset=100
    var b = b_full[200:700]  # Contiguous, offset=200

    # Verify they're contiguous
    assert_true(a.is_contiguous(), "A should be contiguous")
    assert_true(b.is_contiguous(), "B should be contiguous")
    assert_true(a.buffer.offset == 100, "A offset should be 100")
    assert_true(b.buffer.offset == 200, "B offset should be 200")

    # GPU computation
    var gpu_result = launch[Multiply, dtype](a, b)

    # CPU reference
    var cpu_result = a * b

    # Verify
    assert_true(gpu_result.all_close(cpu_result))
    print("  A offset:", a.buffer.offset)
    print("  B offset:", b.buffer.offset)
    print("  Result shape:", gpu_result.shape())
    print("  Passed - Offsets handled correctly!")


fn test_all_offset_scenarios() raises:
    """Comprehensive offset testing."""
    print("\n=== Test: All Offset Scenarios ===")

    comptime dtype = DType.float32

    # Scenario 1: Both have offsets
    print("  Scenario 1: Both tensors have offsets")
    var a1 = Tensor[dtype].rand(1000)
    a1 = a1[100:600]
    var b1 = Tensor[dtype].rand(1000)
    b1 = b1[50:550]
    var result1 = launch[Add, dtype](a1, b1)
    assert_true(result1.all_close(a1 + b1))
    print("    Passed")

    # Scenario 2: Only A has offset
    print("  Scenario 2: Only A has offset")
    var a2 = Tensor[dtype].rand(1000)
    a2 = a2[100:600]
    var b2 = Tensor[dtype].rand(500)  # No offset
    var result2 = launch[Subtract, dtype](a2, b2)
    assert_true(result2.all_close(a2 - b2))
    print("    Passed")

    # Scenario 3: Only B has offset
    print("  Scenario 3: Only B has offset")
    var a3 = Tensor[dtype].rand(500)  # No offset
    var b3 = Tensor[dtype].rand(1000)
    b3 = b3[200:700]
    var result3 = launch[Multiply, dtype](a3, b3)
    assert_true(result3.all_close(a3 * b3))
    print("    Passed")

    # Scenario 4: Neither has offset (original case)
    print("  Scenario 4: No offsets (baseline)")
    var a4 = Tensor[dtype].rand(500)
    var b4 = Tensor[dtype].rand(500)
    var result4 = launch[Divide, dtype](a4, b4)
    assert_true(result4.all_close(a4 / b4))
    print("    Passed")

    print("All offset scenarios passed!")
