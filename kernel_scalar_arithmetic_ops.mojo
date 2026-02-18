from memory import AddressSpace, stack_allocation
from gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from gpu.host import DeviceContext
from os.atomic import Atomic, Consistency

from gpu.primitives.id import lane_id, warp_id
from gpu.primitives.warp import shuffle_down
from gpu.globals import WARP_SIZE


from tenmo import Tensor
from mnemonics import dot
from testing import assert_true
from common_utils import panic
from shapes import Shape

from mnemonics import Multiply, Add, Subtract, Divide, ReverseSubtract


fn dot_product_warp[
    dtype: DType, BLOCK_SIZE: Int = 256
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: UInt,
):
    """
    Warp-optimized dot product kernel.

    Uses warp shuffle for efficient reduction:
    - Grid-stride loop for workload distribution
    - Warp-level shuffle reduction (faster than shared memory)
    - Single atomic write per block

    Performance: ~1.5-2× faster than tree reduction.
    """

    # comptime BLOCK_SIZE = 256
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    var warp_sums = stack_allocation[
        NUM_WARPS, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()

    # Thread and block identification
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x

    # =================================================================
    # Phase 1: Grid-Stride Accumulation
    # Each thread processes multiple elements with stride = total_threads
    # =================================================================
    var accum = Scalar[dtype](0)
    var i = gtid
    while i < size:
        accum += a[i] * b[i]
        i += block_dim.x * grid_dim.x

    # =================================================================
    # Phase 2: Warp-Level Reduction (Shuffle)
    # Reduces WARP_SIZE values to 1 using shuffle instructions
    # Complexity: O(log₂ WARP_SIZE) = 5 iterations for 32-thread warp
    # =================================================================
    var lane = lane_id()  # 0 to WARP_SIZE-1
    var warp = warp_id()  # 0 to NUM_WARPS-1

    var offset = UInt32(WARP_SIZE // 2)
    while offset > 0:
        accum += shuffle_down(accum, offset)
        offset //= 2

    # =================================================================
    # Phase 3: Store Warp Results
    # First thread of each warp writes to shared memory
    # =================================================================
    if lane == 0:
        warp_sums[warp] = accum

    barrier()  # Ensure all warps have written

    # =================================================================
    # Phase 4: Final Reduction (First Warp Only)
    # Reduces NUM_WARPS values to 1 using shuffle
    # Complexity: O(log₂ NUM_WARPS) = 3 iterations for 8 warps
    # =================================================================
    if warp == 0:
        # Each thread in first warp loads one warp sum
        accum = warp_sums[lane] if lane < NUM_WARPS else Scalar[dtype](0)

        # Shuffle reduction
        offset = UInt32(NUM_WARPS // 2)
        while offset > 0:
            accum += shuffle_down(accum, offset)
            offset //= 2

        # Single atomic write per block
        if lane == 0:
            _ = Atomic.fetch_add[ordering = Consistency.MONOTONIC](
                result, accum
            )


# Kernel template for various arithmetic ops involving ND Tensor and a single scalar
# Simplification - views becomes contiguous when copied to device and offset becomes 0
fn scalar_ops[
    op_code: Int,
    dtype: DType,
    block_size: Int = 512,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    scalar: Scalar[dtype],
    size: UInt,
):
    var block_shared_memory = stack_allocation[
        block_size, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    if gtid < size:
        block_shared_memory[tid] = A[gtid]

        @parameter
        if op_code == Add:
            block_shared_memory[tid] += scalar

        elif op_code == Subtract:
            block_shared_memory[tid] -= scalar

        elif op_code == ReverseSubtract:
            block_shared_memory[tid] = scalar - block_shared_memory[tid]

        elif op_code == Multiply:
            block_shared_memory[tid] *= scalar

        elif op_code == Divide:
            block_shared_memory[tid] /= scalar

        else:  # Reverse divide
            block_shared_memory[tid] = scalar / block_shared_memory[tid]

        result[gtid] = block_shared_memory[tid]


fn launch[
    op_code: Int,
    dtype: DType = DType.float32,
    threads_per_block: Int = 512,
](A: Tensor[dtype], scalar: Scalar[dtype]) raises -> Tensor[dtype]:
    constrained[
        threads_per_block <= 512,
        "Threads per block should be <= 512",
    ]()

    if scalar == Scalar[dtype](0):
        raise Error("Divide by zero")

    var numels = A.numels()
    var num_computed_blocks = (
        numels + threads_per_block - 1
    ) // threads_per_block
    var ctx = DeviceContext()
    _ = """var compiled_func = ctx.compile_function[
        dot_product_warp[dtype, BLOCK_SIZE=threads_per_block],
        dot_product_warp[dtype, BLOCK_SIZE=threads_per_block],
    ]()"""

    var compiled_func = ctx.compile_function[
        scalar_ops[op_code=op_code, dtype=dtype, block_size=threads_per_block],
        scalar_ops[op_code=op_code, dtype=dtype, block_size=threads_per_block],
    ]()

    var A_buffer = ctx.enqueue_create_buffer[dtype](numels)
    var result_buffer = ctx.enqueue_create_buffer[dtype](numels)
    result_buffer.enqueue_fill(0)
    # Write tensor data to device buffer
    A.write_to_device_buffer(A_buffer)
    ctx.enqueue_function(
        compiled_func,
        result_buffer,
        A_buffer,
        scalar,
        UInt(numels),
        grid_dim=num_computed_blocks,
        block_dim=threads_per_block,
    )

    ctx.synchronize()

    return Tensor[dtype].from_device_buffer(result_buffer, A.shape())


fn main() raises:
    var SIZE = 65536
    comptime dtype = DType.float32
    var tensor_a = Tensor[dtype].ones(SIZE)
    var expect = tensor_a * 42
    # First test
    var result = launch[op_code=Multiply, threads_per_block=512](tensor_a, 42)
    assert_true(result.all_close(expect))

    # Second test
    tensor_a = Tensor[dtype].rand(SIZE // 2, 2)
    var reshaped = tensor_a.reshape(2, SIZE // 2)
    expect = reshaped * 1919

    result = launch[op_code=Multiply, threads_per_block=512](reshaped, 1919)
    assert_true(result.all_close(expect))

    expect = reshaped / 89

    result = launch[op_code=Divide, threads_per_block=768](reshaped, 89)
    assert_true(result.all_close(expect))

    expect = reshaped - 999

    result = launch[op_code=Subtract, threads_per_block=768](reshaped, 999)
    assert_true(result.all_close(expect))

    print("Launch success")
