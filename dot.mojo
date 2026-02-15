from memory import AddressSpace, stack_allocation
from gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from gpu.host import DeviceContext
from os.atomic import Atomic, Consistency

from tenmo import Tensor
from mnemonics import dot
from testing import assert_true
from common_utils import panic


fn dot_product[
    dtype: DType,
    shared_mem_size: Int = 256,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: UInt,
):
    var cache = stack_allocation[
        shared_mem_size, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()
    var cache_index = thread_idx.x
    var gtid = cache_index + block_dim.x * block_idx.x
    var accum: Scalar[dtype] = 0
    # Grid-stride loop
    for i in range(gtid, size, block_dim.x * grid_dim.x):
        accum += a[i] * b[i]

    cache[cache_index] = accum
    barrier()

    var stride = UInt(block_dim.x // 2)

    while stride > 0:
        if cache_index < stride:
            cache[cache_index] += cache[cache_index + stride]
        barrier()
        stride //= 2

    # only thread 0 writes the final result
    if cache_index == 0:
        # result[0] = cache[0]
        # _= Atomic.fetch_add[ordering = Consistency.MONOTONIC](result, cache[0])
        _ = Atomic.fetch_add[ordering = Consistency.SEQUENTIAL](
            result, cache[0]
        )


fn launch[
    dtype: DType = DType.float32,
    shared_mem_size: Int = 256,
    num_blocks: Int = 1,
    threads_per_block: Int = 512,
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    var length = a.numels()
    if a.rank() != 1 or b.rank() != 1 or length != b.numels():
        panic("Either tensors are not 1D or or tensors length do not match")

    var ctx = DeviceContext()
    _="""var compiled_func = ctx.compile_function[
        dot_product[dtype, shared_mem_size],
        dot_product[dtype, shared_mem_size],
    ]()"""

    var a_buffer = ctx.enqueue_create_buffer[dtype](length)
    var b_buffer = ctx.enqueue_create_buffer[dtype](length)
    var result_buffer = ctx.enqueue_create_buffer[dtype](1)
    result_buffer.enqueue_fill(0)
    a.write_to_device_buffer(a_buffer)
    b.write_to_device_buffer(b_buffer)
    ctx.enqueue_function[dot_product[dtype, shared_mem_size], dot_product[dtype, shared_mem_size]](
        result_buffer,
        a_buffer,
        b_buffer,
        UInt(length),
        grid_dim=num_blocks,
        block_dim=threads_per_block,
    )
    ctx.synchronize()

    return Tensor[dtype].from_device_buffer(result_buffer)


fn main() raises:
    var SIZE = 65536
    comptime dtype = DType.float32
    var tensor_a = Tensor[dtype].ones(SIZE)
    var tensor_b = Tensor[dtype].ones(SIZE)
    var expect = tensor_a.matmul[mode=dot](tensor_b)
    var result = launch(tensor_a, tensor_b)
    print(expect.item(), result.item())
    assert_true(result.all_close(expect))

    SIZE = 70000
    tensor_a = Tensor[dtype].rand(SIZE)
    tensor_b = Tensor[dtype].rand(SIZE)
    expect = tensor_a.matmul[mode=dot](tensor_b)
    result = launch(tensor_a, tensor_b)
    print(expect.item(), result.item())
    assert_true(result.all_close(expect))
    print("Launch success")
