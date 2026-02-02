from memory import AddressSpace, stack_allocation
from gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from gpu.host import DeviceContext

comptime BLOCK_SIZE = 256


fn dot_product[
    dtype: DType
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: UInt,
):
    var cache = stack_allocation[
        BLOCK_SIZE, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()
    var cache_index = thread_idx.x
    var gtid = cache_index + block_dim.x * block_idx.x
    var accum: Scalar[dtype] = 0
    # Grid-stride loop
    for i in range(gtid, size, block_dim.x * grid_dim.x):
        accum += a[i] * b[i]

    cache[cache_index] = accum
    barrier()

    var stride = UInt(BLOCK_SIZE // 2)

    while stride > 0:
        if cache_index < stride:
            cache[cache_index] += cache[cache_index + stride]
        barrier()
        stride //= 2

    # only thread 0 writes the final result
    if cache_index == 0:
        result[0] = cache[0]


from tenmo import Tensor
from memory import memcpy
from operators import dot
from testing import assert_equal, assert_almost_equal


fn main() raises:
    var SIZE = 8
    comptime dtype = DType.float32
    with DeviceContext() as ctx:
        var result = ctx.enqueue_create_buffer[dtype](1)
        result.enqueue_fill(0)
        var a = ctx.enqueue_create_buffer[dtype](SIZE)
        var b = ctx.enqueue_create_buffer[dtype](SIZE)
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = i
                b_host[i] = i
        var BLOCKS_PER_GRID = 1

        ctx.enqueue_function[dot_product[dtype], dot_product[dtype]](
            result,
            a,
            b,
            UInt(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=BLOCK_SIZE,
        )

        expected = ctx.enqueue_create_host_buffer[dtype](1)

        ctx.synchronize()

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                expected[0] += a_host[i] * b_host[i]

        with result.map_to_host() as out_host:
            assert_equal(out_host[0], expected[0])

        var tensor_a = Tensor[dtype].rand(128)
        var tensor_b = Tensor[dtype].rand(128)
        result = ctx.enqueue_create_buffer[dtype](1)
        SIZE = 512
        a = ctx.enqueue_create_buffer[dtype](SIZE)
        b = ctx.enqueue_create_buffer[dtype](SIZE)
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            memcpy(
                dest=a_host.unsafe_ptr(),
                src=tensor_a.buffer.buffer.data,
                count=SIZE,
            )
            memcpy(
                dest=b_host.unsafe_ptr(),
                src=tensor_b.buffer.buffer.data,
                count=SIZE,
            )
        ctx.enqueue_function[dot_product[dtype], dot_product[dtype]](
            result,
            a,
            b,
            UInt(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=BLOCK_SIZE,
        )

        ctx.synchronize()
        var dot_result = tensor_a.matmul[mode=dot](tensor_b)
        with result.map_to_host() as out_host:
            assert_almost_equal(out_host[0], dot_result.item())
