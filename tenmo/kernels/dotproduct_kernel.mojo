# dotproduct_kernel.mojo — GPU dot product kernels

from std.memory import stack_allocation, AddressSpace
from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from max.gpu import barrier
from std.atomic import Atomic
from std.sys import simd_width_of
from max.gpu.primitives.id import lane_id, warp_id
from max.gpu.primitives.warp import shuffle_down
from max.gpu.globals import WARP_SIZE
from ..gpu.device import DeviceState
from ..shared.shapes import Shape
from ..shared.layout import Layout
from ..shared.panic import panic


def dot_product_32[
    dtype: DType, BLOCK_SIZE: Int = 512
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], ImmutAnyOrigin],
    b: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
):
    """
        Warp-optimized dot product kernel.

    Uses warp shuffle for efficient reduction:
    - Grid-stride loop for workload distribution
    - Warp-level shuffle reduction (faster than shared memory)
    - Single atomic write per block

    Performance: ~1.5-2× faster than tree reduction.
    """

    var size = Int(size_)
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    var warp_sums = stack_allocation[
        NUM_WARPS, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x

    var accum = Scalar[dtype](0)
    var i = gtid
    while i < size:
        accum += a[unsafe_offset=i] * b[unsafe_offset=i]
        i += block_dim.x * grid_dim.x

    var lane = lane_id()
    var warp = warp_id()

    var offset = WARP_SIZE // 2
    while offset > 0:
        accum += shuffle_down(accum, UInt32(offset))
        offset //= 2

    if lane == 0:
        warp_sums[unsafe_offset=warp] = accum

    barrier()

    if warp == 0:
        accum = warp_sums[unsafe_offset=lane] if lane < NUM_WARPS else Scalar[dtype](0)

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
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], ImmutAnyOrigin],
    b: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
):
    var size = Int(size_)
    var block_shared_memory = stack_allocation[
        BLOCK_SIZE, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var cache_index = thread_idx.x
    var gtid = cache_index + block_dim.x * block_idx.x
    var accum: Scalar[dtype] = 0
    for i in range(gtid, size, block_dim.x * grid_dim.x):
        accum += a[unsafe_offset=i] * b[unsafe_offset=i]

    block_shared_memory[unsafe_offset=cache_index] = accum
    barrier()

    var stride = block_dim.x // 2

    while stride > 0:
        if cache_index < stride:
            block_shared_memory[unsafe_offset=cache_index] += block_shared_memory[unsafe_offset=
                cache_index + stride
            ]
        barrier()
        stride //= 2

    if cache_index == 0:
        _ = Atomic.fetch_add(result, block_shared_memory[unsafe_offset=0])


struct DotProductKernel[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    def launch[
        num_blocks: Int = 1,
        threads_per_block: Int = 512,
        suppress_validation: Bool = False,
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        B_layout: Layout,
        B_device_state: DeviceState[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        comptime assert (
            threads_per_block <= 512
        ), "Threads per block should be <= 512"
        var rank = A_layout.rank()
        var numels = A_layout.numel()

        comptime if not suppress_validation:
            if rank != 1 or B_layout.rank() != 1:
                panic(
                    "Dot product expects 1D tensors. Found",
                    "A rank: ",
                    String(rank),
                    "and B rank: ",
                    String(B_layout.rank()),
                )
            if numels != B_layout.numel():
                panic(
                    "Tensor lengths do not match.",
                    "A length: ",
                    String(numels),
                    "B length: ",
                    String(B_layout.numel()),
                )


        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

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
                Int64(numels),
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
                Int64(numels),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        if sync:
            device_context.synchronize()
        var device_state = DeviceState[Self.dtype](
            result_buffer^, A_device_state.get_gpu()
        )
        return (
            Layout(Shape()),
            device_state^,
        )
