from sys import simd_width_of
from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from memory import stack_allocation, AddressSpace
from mnemonics import Add, Multiply, Subtract, Divide
from tenmo import Tensor
from shapes import Shape
from strides import Strides
from broadcasthelper import ShapeBroadcaster


# Kernel 1: Both contiguous
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
    size: UInt,
):
    """
    FASTEST: Both inputs contiguous.
    Full SIMD vectorization on loads, compute, and stores.
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
                # TRUE SIMD: Vectorized loads
                var vec_a = A.load[width=simd_width](A_offset + i)
                var vec_b = B.load[width=simd_width](B_offset + i)
                var vec_result: SIMD[dtype, simd_width]

                # TRUE SIMD: Vectorized arithmetic
                @parameter
                if op_code == Add:
                    vec_result = vec_a + vec_b
                elif op_code == Subtract:
                    vec_result = vec_a - vec_b
                elif op_code == Multiply:
                    vec_result = vec_a * vec_b
                else:  # op_code == Divide:
                    vec_result = vec_a / vec_b

                # TRUE SIMD: Vectorized store
                result.store[width=simd_width](i, vec_result)

            elif i < size:
                # Scalar tail
                for j in range(size - i):
                    var idx = i + j
                    var a_val = A[A_offset + idx]
                    var b_val = B[B_offset + idx]
                    var res: Scalar[dtype]

                    @parameter
                    if op_code == Add:
                        res = a_val + b_val
                    elif op_code == Subtract:
                        res = a_val - b_val
                    elif op_code == Multiply:
                        res = a_val * b_val
                    else:  # op_code == Divide:
                        res = a_val / b_val

                    result[idx] = res

        base_idx += stride * CHUNK_SIZE


# Kernel 2: A contiguous, B strides
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
    result_shape: UnsafePointer[Int64, ImmutAnyOrigin],
    B_strides: UnsafePointer[Int64, ImmutAnyOrigin],
    B_offset: Int,
    size: UInt,
    rank: UInt,
):
    """
    MEDIUM-FAST: A contiguous (vectorized load), B strided (scalar loads).
    """

    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    # Read metadata into stack
    comptime MAX_RANK = 8
    var coords = stack_allocation[MAX_RANK, Int]()
    var shape_local = stack_allocation[MAX_RANK, Int]()
    var strides_B_local = stack_allocation[MAX_RANK, Int]()

    for i in range(rank):
        shape_local[Int(i)] = Int(result_shape[Int(i) + 2])
        strides_B_local[Int(i)] = Int(B_strides[Int(i) + 2])

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                # A: Vectorized load (contiguous)
                var vec_a = A.load[width=simd_width](A_offset + i)
                var vec_result: SIMD[dtype, simd_width] = 0

                # B: Scalar loads (strided)
                @parameter
                for lane in range(simd_width):
                    var linear_idx = i + lane

                    # Calculate coords
                    var remaining = Int(linear_idx)
                    for dim in range(Int(rank) - 1, -1, -1):
                        coords[dim] = remaining % shape_local[dim]
                        remaining //= shape_local[dim]

                    # Calculate B offset
                    var b_idx = B_offset
                    for dim in range(rank):
                        b_idx += coords[Int(dim)] * strides_B_local[Int(dim)]

                    var a_val = vec_a[lane]  # From vectorized load
                    var b_val = B[b_idx]  # Scalar load

                    @parameter
                    if op_code == Add:
                        vec_result[lane] = a_val + b_val
                    elif op_code == Subtract:
                        vec_result[lane] = a_val - b_val
                    elif op_code == Multiply:
                        vec_result[lane] = a_val * b_val
                    else:  # op_code == Divide:
                        vec_result[lane] = a_val / b_val

                # Vectorized store
                result.store[width=simd_width](i, vec_result)

            elif i < size:
                # Scalar tail
                for j in range(size - i):
                    var linear_idx = i + j

                    var remaining = Int(linear_idx)
                    for dim in range(Int(rank) - 1, -1, -1):
                        coords[dim] = remaining % shape_local[dim]
                        remaining //= shape_local[dim]

                    var b_idx = B_offset
                    for dim in range(rank):
                        b_idx += coords[Int(dim)] * strides_B_local[Int(dim)]

                    var a_val = A[A_offset + linear_idx]
                    var b_val = B[b_idx]
                    var res: Scalar[dtype]

                    @parameter
                    if op_code == Add:
                        res = a_val + b_val
                    elif op_code == Subtract:
                        res = a_val - b_val
                    elif op_code == Multiply:
                        res = a_val * b_val
                    else:  # op_code == Divide:
                        res = a_val / b_val

                    result[linear_idx] = res

        base_idx += stride * CHUNK_SIZE


# Kernel 3: A Strided, B Contiguous
fn arithmetic_ops_B_contiguous[
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
    size: UInt,
    rank: UInt,
):
    """
    MEDIUM-FAST: A strided (scalar loads), B contiguous (vectorized load).
    """

    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    # Read metadata into stack
    comptime MAX_RANK = 8
    var coords = stack_allocation[MAX_RANK, Int]()
    var shape_local = stack_allocation[MAX_RANK, Int]()
    var strides_A_local = stack_allocation[MAX_RANK, Int]()

    for i in range(rank):
        shape_local[Int(i)] = Int(result_shape[Int(i) + 2])
        strides_A_local[Int(i)] = Int(A_strides[Int(i) + 2])

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                # B: Vectorized load (contiguous)
                var vec_b = B.load[width=simd_width](B_offset + i)
                var vec_result: SIMD[dtype, simd_width] = 0

                # A: Scalar loads (strided)
                @parameter
                for lane in range(simd_width):
                    var linear_idx = i + lane

                    # Calculate coords
                    var remaining = Int(linear_idx)
                    for dim in range(Int(rank) - 1, -1, -1):
                        coords[dim] = remaining % shape_local[dim]
                        remaining //= shape_local[dim]

                    # Calculate A offset
                    var a_idx = A_offset
                    for dim in range(rank):
                        a_idx += coords[Int(dim)] * strides_A_local[Int(dim)]

                    var a_val = A[a_idx]  # Scalar load
                    var b_val = vec_b[lane]  # From vectorized load

                    @parameter
                    if op_code == Add:
                        vec_result[lane] = a_val + b_val
                    elif op_code == Subtract:
                        vec_result[lane] = a_val - b_val
                    elif op_code == Multiply:
                        vec_result[lane] = a_val * b_val
                    else:  # op_code == Divide:
                        vec_result[lane] = a_val / b_val

                # Vectorized store
                result.store[width=simd_width](i, vec_result)

            elif i < size:
                # Scalar tail
                for j in range(size - i):
                    var linear_idx = i + j

                    var remaining = Int(linear_idx)
                    for dim in range(Int(rank) - 1, -1, -1):
                        coords[dim] = remaining % shape_local[dim]
                        remaining //= shape_local[dim]

                    var a_idx = A_offset
                    for dim in range(rank):
                        a_idx += coords[Int(dim)] * strides_A_local[Int(dim)]

                    var a_val = A[a_idx]
                    var b_val = B[B_offset + linear_idx]
                    var res: Scalar[dtype]

                    @parameter
                    if op_code == Add:
                        res = a_val + b_val
                    elif op_code == Subtract:
                        res = a_val - b_val
                    elif op_code == Multiply:
                        res = a_val * b_val
                    else:  # op_code == Divide:
                        res = a_val / b_val

                    result[linear_idx] = res

        base_idx += stride * CHUNK_SIZE


# Kernel 4: Both Strided (Slowest)
fn arithmetic_ops_both_strided[
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
    B_strides: UnsafePointer[Int64, ImmutAnyOrigin],
    A_offset: Int,
    B_offset: Int,
    size: UInt,
    rank: UInt,
):
    """
    SLOWEST: Both inputs strided (both scalar loads).
    Only vectorized store, no vectorized loads.
    """

    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    # Read metadata into stack
    comptime MAX_RANK = 8
    var coords = stack_allocation[MAX_RANK, Int]()
    var shape_local = stack_allocation[MAX_RANK, Int]()
    var strides_A_local = stack_allocation[MAX_RANK, Int]()
    var strides_B_local = stack_allocation[MAX_RANK, Int]()

    for i in range(rank):
        shape_local[Int(i)] = Int(result_shape[Int(i)])
        strides_A_local[Int(i)] = Int(A_strides[Int(i)])
        strides_B_local[Int(i)] = Int(B_strides[Int(i)])

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_result: SIMD[dtype, simd_width] = 0

                # Both A and B: Scalar loads (strided)
                @parameter
                for lane in range(simd_width):
                    var linear_idx = i + lane

                    # Calculate coords
                    var remaining = Int(linear_idx)
                    for dim in range(Int(rank) - 1, -1, -1):
                        coords[dim] = remaining % shape_local[dim]
                        remaining //= shape_local[dim]

                    # Calculate A offset
                    var a_idx = A_offset
                    for dim in range(rank):
                        a_idx += coords[Int(dim)] * strides_A_local[Int(dim)]

                    # Calculate B offset
                    var b_idx = B_offset
                    for dim in range(rank):
                        b_idx += coords[Int(dim)] * strides_B_local[Int(dim)]

                    var a_val = A[a_idx]  # Scalar load
                    var b_val = B[b_idx]  # Scalar load

                    @parameter
                    if op_code == Add:
                        vec_result[lane] = a_val + b_val
                    elif op_code == Subtract:
                        vec_result[lane] = a_val - b_val
                    elif op_code == Multiply:
                        vec_result[lane] = a_val * b_val
                    elif op_code == Divide:
                        vec_result[lane] = a_val / b_val

                # Vectorized store
                result.store[width=simd_width](i, vec_result)

            elif i < size:
                # Scalar tail
                for j in range(size - i):
                    var linear_idx = i + j

                    var remaining = Int(linear_idx)
                    for dim in range(Int(rank) - 1, -1, -1):
                        coords[dim] = remaining % shape_local[dim]
                        remaining //= shape_local[dim]

                    var a_idx = A_offset
                    var b_idx = B_offset
                    for dim in range(rank):
                        a_idx += coords[Int(dim)] * strides_A_local[Int(dim)]
                        b_idx += coords[Int(dim)] * strides_B_local[Int(dim)]

                    var a_val = A[a_idx]
                    var b_val = B[b_idx]
                    var res: Scalar[dtype] = 0

                    @parameter
                    if op_code == Add:
                        res = a_val + b_val
                    elif op_code == Subtract:
                        res = a_val - b_val
                    elif op_code == Multiply:
                        res = a_val * b_val
                    elif op_code == Divide:
                        res = a_val / b_val

                    result[linear_idx] = res

        base_idx += stride * CHUNK_SIZE


fn launch[
    op_code: Int,
    dtype: DType = DType.float32,
](A: Tensor[dtype], B: Tensor[dtype]) raises -> Tensor[dtype]:
    """
    Select the optimal kernel based on memory layout.
    """

    if not A.broadcastable(B):
        raise Error("Shape mismatch")

    var ctx = DeviceContext()
    comptime optimal_simd = simd_width_of[dtype]()

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
        print("[GPU] Using Kernel 1: Both contiguous (FASTEST)")

        var compiled_func = ctx.compile_function[
            arithmetic_ops_both_contiguous[
                op_code, dtype, optimal_simd, 2 * optimal_simd
            ],
            arithmetic_ops_both_contiguous[
                op_code, dtype, optimal_simd, 2 * optimal_simd
            ],
        ]()

        var A_buffer = ctx.enqueue_create_buffer[dtype](A.numels())
        var B_buffer = ctx.enqueue_create_buffer[dtype](B.numels())
        var result_buffer = ctx.enqueue_create_buffer[dtype](output_size)

        A.write_to_device_buffer(A_buffer)
        B.write_to_device_buffer(B_buffer)

        ctx.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            B_buffer,
            0,  # A.offset(),
            0,  # B.offset(),
            UInt(output_size),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        ctx.synchronize()
        return Tensor[dtype].from_device_buffer(result_buffer, broadcast_shape)

    # Prepare for strided kernels
    var rank = broadcast_shape.rank()
    var A_broadcast_strides = ShapeBroadcaster.broadcast_strides(
        A.shape(), A.strides(), broadcast_shape
    )
    var B_broadcast_strides = ShapeBroadcaster.broadcast_strides(
        B.shape(), B.strides(), broadcast_shape
    )
    # var A_is_contiguous = A.is_contiguous() and A.shape() == broadcast_shape
    #var B_is_contiguous = B.is_contiguous() and B.shape() == broadcast_shape

    var A_is_contiguous = A.is_contiguous()
    var B_is_contiguous = B.is_contiguous()

    # ================================================================
    # PATH 2: A contiguous, B strided
    # ================================================================
    if A_is_contiguous and not B_is_contiguous:
        print("[GPU] Using Kernel 2: A contiguous, B strided (MEDIUM-FAST)")

        var compiled_func = ctx.compile_function[
            arithmetic_ops_A_contiguous[
                op_code, dtype, optimal_simd, 2 * optimal_simd
            ],
            arithmetic_ops_A_contiguous[
                op_code, dtype, optimal_simd, 2 * optimal_simd
            ],
        ]()

        var A_buffer = ctx.enqueue_create_buffer[dtype](A.numels())
        var B_buffer = ctx.enqueue_create_buffer[dtype](B.numels())
        var result_buffer = ctx.enqueue_create_buffer[dtype](output_size)

        var result_shape_buffer = ctx.enqueue_create_buffer[DType.int64](
            broadcast_shape.write_length()
        )
        var B_strides_buffer = ctx.enqueue_create_buffer[DType.int64](
            B_broadcast_strides.write_length()
        )

        A.write_to_device_buffer(A_buffer)
        B.write_to_device_buffer(B_buffer)
        broadcast_shape.write_to_device_buffer(result_shape_buffer)
        B_broadcast_strides.write_to_device_buffer(B_strides_buffer)

        ctx.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            B_buffer,
            0,  # A.offset(),
            result_shape_buffer,
            B_strides_buffer,
            0,  # B.offset(),
            UInt(output_size),
            UInt(rank),
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
                op_code, dtype, optimal_simd, 2 * optimal_simd
            ],
            arithmetic_ops_B_contiguous[
                op_code, dtype, optimal_simd, 2 * optimal_simd
            ],
        ]()

        var A_buffer = ctx.enqueue_create_buffer[dtype](A.numels())
        var B_buffer = ctx.enqueue_create_buffer[dtype](B.numels())
        var result_buffer = ctx.enqueue_create_buffer[dtype](output_size)

        var result_shape_buffer = ctx.enqueue_create_buffer[DType.int64](
            broadcast_shape.write_length()
        )
        var A_strides_buffer = ctx.enqueue_create_buffer[DType.int64](
            A_broadcast_strides.write_length()
        )

        A.write_to_device_buffer(A_buffer)
        B.write_to_device_buffer(B_buffer)
        broadcast_shape.write_to_device_buffer(result_shape_buffer)
        A_broadcast_strides.write_to_device_buffer(A_strides_buffer)

        ctx.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            B_buffer,
            result_shape_buffer,
            A_strides_buffer,
            0,  # A.offset(),
            0,  # B.offset(),
            UInt(output_size),
            UInt(rank),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        ctx.synchronize()
        return Tensor[dtype].from_device_buffer(result_buffer, broadcast_shape)

    # ================================================================
    # PATH 4: Both strided (or broadcasting)
    # ================================================================

    print("[GPU] Using Kernel 4: Both strided (SLOWEST)")

    var compiled_func = ctx.compile_function[
        arithmetic_ops_both_strided[
            op_code, dtype, optimal_simd, 2 * optimal_simd
        ],
        arithmetic_ops_both_strided[
            op_code, dtype, optimal_simd, 2 * optimal_simd
        ],
    ]()

    var A_buffer = ctx.enqueue_create_buffer[dtype](A.numels())
    var B_buffer = ctx.enqueue_create_buffer[dtype](B.numels())
    var result_buffer = ctx.enqueue_create_buffer[dtype](output_size)

    var result_shape_buffer = ctx.enqueue_create_buffer[DType.int64](
        broadcast_shape.write_length()
    )
    var A_strides_buffer = ctx.enqueue_create_buffer[DType.int64](
        A_broadcast_strides.write_length()
    )
    var B_strides_buffer = ctx.enqueue_create_buffer[DType.int64](
        B_broadcast_strides.write_length()
    )

    A.write_to_device_buffer(A_buffer)
    B.write_to_device_buffer(B_buffer)
    broadcast_shape.write_to_device_buffer(result_shape_buffer)
    A_broadcast_strides.write_to_device_buffer(A_strides_buffer)
    B_broadcast_strides.write_to_device_buffer(B_strides_buffer)

    ctx.enqueue_function(
        compiled_func,
        result_buffer,
        A_buffer,
        B_buffer,
        result_shape_buffer,
        A_strides_buffer,
        B_strides_buffer,
        0,  # A.offset(),
        0,  # B.offset(),
        UInt(output_size),
        UInt(rank),
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
    # test_contiguous_same_shape()
    test_non_contiguous()
    _ = """test_broadcasting()
    test_scalar_broadcast()
    test_complex_broadcasting()
    test_large_arrays()

    # ✅ NEW: Offset-specific tests
    test_contiguous_view_with_offset()
    test_all_offset_scenarios()

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED (Including Offset Tests)")
    print("="*60)"""


from common_utils import now
from testing import assert_true


fn test_contiguous_same_shape() raises:
    """Test fast path: contiguous, same shape."""
    print("=== Test 1: Contiguous Same Shape ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(1000, 1000)
    var b = Tensor[dtype].rand(1000, 1000)

    var start = now()
    var gpu_result = launch[Multiply, dtype](a, b)
    var gpu_time = (now() - start) * 1000

    start = now()
    var cpu_result = a * b
    var cpu_time = (now() - start) * 1000

    assert_true(gpu_result.all_close(cpu_result))
    print("  GPU:", gpu_time, "ms")
    print("  CPU:", cpu_time, "ms")
    print("  ✓ Passed")


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
    print("  ✓ Passed")


fn test_scalar_broadcast() raises:
    """Test scalar broadcasting."""
    print("\n=== Test 3: Scalar Broadcast ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(100, 100)
    var b = Tensor[dtype].ones(1, 1) * 42  # Broadcasts to [100, 100]

    var gpu_result = launch[Multiply, dtype](a, b)
    var cpu_result = a * b

    assert_true(gpu_result.all_close(cpu_result))
    print("  ✓ Passed")


fn test_non_contiguous() raises:
    """Test non-contiguous tensors (views/transposes)."""
    print("\n=== Test 4: Non-Contiguous (Transpose) ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(3, 2)
    var b = Tensor[dtype].rand(2, 3)
    a.print()
    b.print()
    var a_t = a.transpose(1, 0)  # Non-contiguous view [20, 10]
    a_t.print()
    var gpu_result = launch[Add, dtype](a_t, b)
    var cpu_result = a_t + b
    gpu_result.print()
    print()
    cpu_result.print()
    assert_true(gpu_result.all_close(cpu_result))
    print("  ✓ Passed")


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
    print("  ✓ Passed")


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
    print("  ✓ Passed")


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
    print("  ✓ Passed - Offsets handled correctly!")


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
    print("    ✓ Passed")

    # Scenario 2: Only A has offset
    print("  Scenario 2: Only A has offset")
    var a2 = Tensor[dtype].rand(1000)
    a2 = a2[100:600]
    var b2 = Tensor[dtype].rand(500)  # No offset
    var result2 = launch[Subtract, dtype](a2, b2)
    assert_true(result2.all_close(a2 - b2))
    print("    ✓ Passed")

    # Scenario 3: Only B has offset
    print("  Scenario 3: Only B has offset")
    var a3 = Tensor[dtype].rand(500)  # No offset
    var b3 = Tensor[dtype].rand(1000)
    b3 = b3[200:700]
    var result3 = launch[Multiply, dtype](a3, b3)
    assert_true(result3.all_close(a3 * b3))
    print("    ✓ Passed")

    # Scenario 4: Neither has offset (original case)
    print("  Scenario 4: No offsets (baseline)")
    var a4 = Tensor[dtype].rand(500)
    var b4 = Tensor[dtype].rand(500)
    var result4 = launch[Divide, dtype](a4, b4)
    assert_true(result4.all_close(a4 / b4))
    print("    ✓ Passed")

    print("  ✅ All offset scenarios passed!")
