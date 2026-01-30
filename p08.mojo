from memory import stack_allocation, AddressSpace
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from math import iota
from testing import assert_true

comptime TPB = 4
comptime SIZE = 8
comptime BLOCKS_PER_GRID = (4, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32


fn add_10_shared(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: Int,
):
    var shared = stack_allocation[
        TPB, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()

    var gtid = (
        block_idx.x * block_dim.x + thread_idx.x
    )  # Globally unique thread id
    var local_tid = thread_idx.x  # Block local thread id
    print("Global thread id: ", gtid, "local thread id: ", local_tid)
    print("Thread block size: ", block_dim.x * block_dim.y * block_dim.z)
    # Copy data from global memory to shared memory
    if gtid < UInt(size):
        shared[local_tid] = a[gtid]
    barrier()
    if gtid < UInt(size):
        output[gtid] = shared[local_tid] + 10


fn main() raises:
    with DeviceContext() as ctx:
        var out_buff_d = ctx.enqueue_create_buffer[dtype](SIZE)
        var host_buff = ctx.enqueue_create_host_buffer[dtype](SIZE)
        out_buff_d.enqueue_fill(0)
        var a_buff_d = ctx.enqueue_create_buffer[dtype](SIZE)
        with a_buff_d.map_to_host() as a_buff_h:
            iota(a_buff_h.unsafe_ptr(), SIZE)
        print(a_buff_d)
        iota(host_buff.unsafe_ptr(), SIZE)
        ctx.enqueue_function[add_10_shared, add_10_shared](
            out_buff_d,
            a_buff_d,
            SIZE,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )
        ctx.synchronize()
        with out_buff_d.map_to_host() as out_buff_h:
            for i in range(SIZE):
                assert_true(out_buff_h[i] == host_buff[i] + 10)
