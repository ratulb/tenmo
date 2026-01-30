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
    size: UInt,
):
    var shared = stack_allocation[
        TPB, Scalar[dtype], address_space = AddressSpace.SHARED
    ]()

    var gtid = (
        block_idx.x * block_dim.x + thread_idx.x
    )  # Globally unique thread id
    var local_tid = thread_idx.x  # Block local thread id


fn main() raises:
    with DeviceContext() as ctx:
        var out_buff_d = ctx.enqueue_create_buffer[dtype](SIZE)
        out_buff_d.enqueue_fill(0)
        var a_buff_d = ctx.enqueue_create_buffer[dtype](SIZE)
        iota(a_buff_d.unsafe_ptr, SIZE)
        print(a_buff_d)
