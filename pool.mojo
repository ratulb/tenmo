from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from memory import AddressSpace
from layout import Layout, LayoutTensor
from testing import assert_true

comptime SIZE = 10
comptime dtype = DType.float32
comptime layout = Layout.row_major(SIZE)
comptime Matrix = LayoutTensor[dtype, layout, _]


fn pooling(matrix_in: Matrix[ImmutAnyOrigin], matrix_out: Matrix[MutAnyOrigin]):
    var bti = thread_idx.x  # Block thread index
    var gti = block_idx.x * block_dim.x + bti  # Global thread index
    var bsm = LayoutTensor[
        dtype, layout, MutAnyOrigin, address_space = AddressSpace.SHARED
    ].stack_allocation()  # Block shared memory
    var size = UInt(SIZE)
    if gti < size:
        bsm[bti] = matrix_in[gti]
    barrier()  # Make sure each block thread has copied own slot data from global memory
    print(bsm)
    if gti < size:
        if gti == 0:
            matrix_out[0] = bsm[0]
        elif gti == 1:
            matrix_out[1] = bsm[0] + bsm[1]
        else:
            matrix_out[gti] = bsm[bti - 2] + bsm[bti - 1] + bsm[bti]


fn main() raises:
    var ctx = DeviceContext()
    var dev_buff_in = ctx.enqueue_create_buffer[dtype](SIZE)
    var dev_buff_out = ctx.enqueue_create_buffer[dtype](SIZE)
    dev_buff_out.enqueue_fill(0)
    with dev_buff_in.map_to_host() as dev_buff_host:
        for i in range(SIZE):
            dev_buff_host[i] = i
        # print(dev_buff_host)
    var matrix_in = Matrix[ImmutAnyOrigin](dev_buff_in)
    var matrix_out = Matrix[MutAnyOrigin](dev_buff_out)
    ctx.enqueue_function[pooling, pooling](
        matrix_in, matrix_out, grid_dim=1, block_dim=8
    )
    ctx.synchronize()
    with dev_buff_out.map_to_host() as host_buff:
        # print(host_buff)
        pass
