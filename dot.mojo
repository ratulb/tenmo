from memory import AddressSpace, stack_allocation
from gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from gpu.host import DeviceContext,  DeviceAttribute
from os.atomic import Atomic, Consistency

from tenmo import Tensor
from mnemonics import dot
from testing import assert_true
from common_utils import panic
from shapes import Shape

# Kernel
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

    # only thread 0 of each block writes the final result
    if cache_index == 0:
        _ = Atomic.fetch_add(result, cache[0])


fn launch[
    dtype: DType = DType.float32,
    shared_mem_size: Int = 256,
    num_blocks: Int = 1,
    threads_per_block: Int = shared_mem_size,
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Tensor[dtype]:
    constrained[
        shared_mem_size == threads_per_block,
        "shared memory size should be equal to threads per block",
    ]()
    var length = a.numels()
    if a.rank() != 1 or b.rank() != 1 or length != b.numels():
        panic("Either tensors are not 1D or or tensors length do not match")

    var ctx = DeviceContext()
    var compiled_func = ctx.compile_function[
        dot_product[dtype, shared_mem_size],
        dot_product[dtype, shared_mem_size],
    ]()

    var a_buffer = ctx.enqueue_create_buffer[dtype](length)
    var b_buffer = ctx.enqueue_create_buffer[dtype](length)
    var result_buffer = ctx.enqueue_create_buffer[dtype](1)
    result_buffer.enqueue_fill(0)
    a.write_to_device_buffer(a_buffer)
    b.write_to_device_buffer(b_buffer)
    ctx.enqueue_function(
        compiled_func,
        result_buffer,
        a_buffer,
        b_buffer,
        UInt(length),
        grid_dim=num_blocks,
        block_dim=threads_per_block,
    )

    ctx.synchronize()
    print(ctx.name(), ctx.api(), ctx.id())
    var attr = DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR
    var max_blocks = ctx.get_attribute(attr)
    print("mx blocks: ", max_blocks)
    var attr2 = DeviceAttribute.CLOCK_RATE
    var clock_rate = ctx.get_attribute(attr2)
    print("Clock rate: ", clock_rate)
    var attr3 = DeviceAttribute.MAX_BLOCK_DIM_X
    var max_thread_x = ctx.get_attribute(attr3)
    print("max thread_x : ", max_thread_x)
    var attr4 = DeviceAttribute.MAX_BLOCK_DIM_Y
    var max_thread_y = ctx.get_attribute(attr4)
    print("max thread_y : ", max_thread_y)
    var attr5 = DeviceAttribute.MAX_BLOCK_DIM_Z
    var max_thread_z = ctx.get_attribute(attr5)
    print("max thread_z : ", max_thread_z)



    return Tensor[dtype].from_device_buffer(result_buffer, Shape())


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
