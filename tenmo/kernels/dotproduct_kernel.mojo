# =============================================================================
# dotproduct_kernel.mojo — GPU dot product kernels
# =============================================================================

from std.memory import stack_allocation, AddressSpace
from std.gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from std.atomic import Atomic, Ordering
from std.sys import simd_width_of
from std.gpu.primitives.id import lane_id, warp_id
from std.gpu.primitives.warp import shuffle_down
from std.gpu.globals import WARP_SIZE
from tenmo.device import DeviceState
from tenmo.shapes import Shape
from tenmo.ndbuffer import NDBuffer
from tenmo.tensor import Tensor
from tenmo.common_utils import panic


def dot_product_32[
    dtype: DType, BLOCK_SIZE: Int = 512
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
):
    """
    Warp-optimized dot product kernel.

    Uses warp shuffle for efficient reduction:
    - Grid-stride loop for workload distribution
    - Warp-level shuffle reduction (faster than shared memory)
    - Single atomic write per block

    Performance: ~1.5-2× faster than tree reduction.
    """

    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    var warp_sums = stack_allocation[
        NUM_WARPS, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x

    var accum = Scalar[dtype](0)
    var i = gtid
    while i < size:
        accum += a[i] * b[i]
        i += block_dim.x * grid_dim.x

    var lane = lane_id()
    var warp = warp_id()

    var offset = WARP_SIZE // 2
    while offset > 0:
        accum += shuffle_down(accum, UInt32(offset))
        offset //= 2

    if lane == 0:
        warp_sums[warp] = accum

    barrier()

    if warp == 0:
        accum = warp_sums[lane] if lane < NUM_WARPS else Scalar[dtype](0)

        offset = NUM_WARPS // 2
        while offset > 0:
            accum += shuffle_down(accum, UInt32(offset))
            offset //= 2

        if lane == 0:
            _ = Atomic.fetch_add(result, accum)


def dot_product_64[
    dtype: DType,
    BLOCK_SIZE: Int = 512,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
):
    var block_shared_memory = stack_allocation[
        BLOCK_SIZE, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var cache_index = thread_idx.x
    var gtid = cache_index + block_dim.x * block_idx.x
    var accum: Scalar[dtype] = 0
    for i in range(gtid, size, block_dim.x * grid_dim.x):
        accum += a[i] * b[i]

    block_shared_memory[cache_index] = accum
    barrier()

    var stride = block_dim.x // 2

    while stride > 0:
        if cache_index < stride:
            block_shared_memory[cache_index] += block_shared_memory[
                cache_index + stride
            ]
        barrier()
        stride //= 2

    if cache_index == 0:
        _ = Atomic.fetch_add(result, block_shared_memory[0])


struct DotproductKernel[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def launch[
        num_blocks: Int = 1,
        threads_per_block: Int = 512,
        suppress_validation: Bool = False,
    ](
        A: Tensor[Self.dtype], B: Tensor[Self.dtype], sync: Bool = False
    ) raises -> Tensor[Self.dtype]:
        comptime assert (
            threads_per_block <= 512
        ), "Threads per block should be <= 512"
        var rank = A.rank()
        var numels = A.numels()

        comptime if not suppress_validation:
            if rank != 1 or B.rank() != 1:
                panic(
                    "Dot product expects 1D tensors. Found",
                    "A rank: ",
                    String(rank),
                    "and B rank: ",
                    String(B.rank()),
                )
            if numels != B.numels():
                panic(
                    "Tensor lengths do not match.",
                    "A length: ",
                    String(numels),
                    "B length: ",
                    String(B.numels()),
                )

        ref A_device_state = A.buffer.device_state.value()
        ref B_device_state = B.buffer.device_state.value()

        var device_context = A_device_state.gpu[]

        ref A_buffer = A_device_state.device_buffer()
        ref B_buffer = B_device_state.device_buffer()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](1)
        result_buffer.enqueue_fill(0)

        comptime use_32_kernel = True if simd_width_of[
            Self.dtype
        ]() > 8 else False

        comptime if use_32_kernel:
            var compiled_func = device_context.compile_function[
                dot_product_32[Self.dtype, BLOCK_SIZE=threads_per_block],
            ]()

            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                B_buffer,
                numels,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        else:
            var compiled_func = device_context.compile_function[
                dot_product_64[Self.dtype, BLOCK_SIZE=threads_per_block],
            ]()

            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                B_buffer,
                numels,
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        if sync:
            device_context.synchronize()
        var device_state = DeviceState[Self.dtype](
            result_buffer^, A_device_state.get_gpu()
        )
        var ndb = NDBuffer[Self.dtype].with_device_state(device_state^, Shape())
        return Tensor[Self.dtype](ndb^, requires_grad=False)
