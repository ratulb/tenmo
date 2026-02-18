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
        # accum += shuffle_down[dtype, 1](accum, offset)
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
            # accum += shuffle_down[dtype, 1](accum, offset)
            accum += shuffle_down(accum, offset)
            offset //= 2

        # Single atomic write per block
        if lane == 0:
            _ = Atomic.fetch_add[ordering = Consistency.MONOTONIC](
                result, accum
            )


# Kernel
fn dot_product[
    dtype: DType,
    BLOCK_SIZE: Int = 512,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: UInt,
):
    var block_shared_memory = stack_allocation[
        BLOCK_SIZE, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()
    var cache_index = thread_idx.x
    var gtid = cache_index + block_dim.x * block_idx.x
    var accum: Scalar[dtype] = 0
    # Grid-stride loop
    for i in range(gtid, size, block_dim.x * grid_dim.x):
        accum += a[i] * b[i]

    block_shared_memory[cache_index] = accum
    barrier()

    var stride = UInt(block_dim.x // 2)

    while stride > 0:
        if cache_index < stride:
            block_shared_memory[cache_index] += block_shared_memory[
                cache_index + stride
            ]
        barrier()
        stride //= 2

    # only thread 0 of each block writes the final result
    if cache_index == 0:
        _ = Atomic.fetch_add(result, block_shared_memory[0])


fn launch[
    dtype: DType = DType.float32,
    num_blocks: Int = 1,
    threads_per_block: Int = 256,
](A: Tensor[dtype], B: Tensor[dtype]) raises -> Tensor[dtype]:
    constrained[
        threads_per_block <= 512,
        "Threads per block should be <= 512",
    ]()
    var rank = A.rank()
    if rank != 1 or B.rank() != 1:
        panic(
            "Dot product expects 1D tensors. Found",
            "A rank: ",
            rank.__str__(),
            "and B rank: ",
            B.rank().__str__(),
        )
    var numels = A.numels()
    if numels != B.numels():
        panic(
            "Tensor lengths do not match.",
            "A length: ",
            numels.__str__(),
            "B length: ",
            B.numels().__str__(),
        )

    var ctx = DeviceContext()
    _="""var compiled_func = ctx.compile_function[
        dot_product_warp[dtype, BLOCK_SIZE=threads_per_block],
        dot_product_warp[dtype, BLOCK_SIZE=threads_per_block],
    ]()"""

    var compiled_func = ctx.compile_function[
        dot_product[dtype, BLOCK_SIZE=threads_per_block],
        dot_product[dtype, BLOCK_SIZE=threads_per_block],
    ]()

    var A_buffer = ctx.enqueue_create_buffer[dtype](numels)
    var B_buffer = ctx.enqueue_create_buffer[dtype](numels)
    var result_buffer = ctx.enqueue_create_buffer[dtype](1)
    result_buffer.enqueue_fill(0)
    A.write_to_device_buffer(A_buffer)
    B.write_to_device_buffer(B_buffer)
    ctx.enqueue_function(
        compiled_func,
        result_buffer,
        A_buffer,
        B_buffer,
        UInt(numels),
        grid_dim=num_blocks,
        block_dim=threads_per_block,
    )

    ctx.synchronize()

    return Tensor[dtype].from_device_buffer(result_buffer, Shape())


fn main() raises:
    var SIZE = 65536
    # var threads_per_block = 512
    # var num_blocks = (SIZE + threads_per_block -1 ) // threads_per_block
    comptime dtype = DType.float32
    var tensor_a = Tensor[dtype].ones(SIZE)
    var tensor_b = Tensor[dtype].ones(SIZE)
    var expect = tensor_a.matmul[mode=dot](tensor_b)

    var result = launch[num_blocks=129, threads_per_block=512](
        tensor_a, tensor_b
    )
    print(expect.item(), result.item())
    assert_true(result.all_close(expect))

    SIZE = 70000
    tensor_a = Tensor[dtype].rand(SIZE)
    tensor_b = Tensor[dtype].rand(SIZE)
    expect = tensor_a.matmul[mode=dot](tensor_b)
    result = launch[num_blocks=138, threads_per_block=512](tensor_a, tensor_b)
    print(expect.item(), result.item())
    assert_true(result.all_close(expect))
    print("Launch success")
