from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import block_idx, thread_idx, block_dim, barrier
from layout import Layout, LayoutTensor
from tenmo import Tensor
from memory import memcpy, memset_zero, AddressSpace
from testing import assert_true

comptime dtype = DType.float32

comptime blocks = 3
comptime threads = 5
comptime rows = blocks
comptime cols = threads

comptime num_elems = blocks * threads

comptime layout_A = Layout.row_major(blocks, threads)
comptime layout_B = Layout.row_major(threads, blocks)
comptime layout_C = Layout.row_major(blocks, blocks)

comptime Matrix_A = LayoutTensor[dtype, layout_A, MutAnyOrigin]
comptime Matrix_B = LayoutTensor[dtype, layout_B, MutAnyOrigin]
comptime Matrix_C = LayoutTensor[dtype, layout_C, MutAnyOrigin]

comptime ROW = LayoutTensor[
    dtype,
    Layout.row_major(1, cols),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
]
comptime COL = LayoutTensor[
    dtype,
    Layout.row_major(rows, 1),
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
]


fn matmul_2d(A: Matrix_A, B: Matrix_B, C: Matrix_C):
    var output_row = Int(block_idx.x)
    var output_col = Int(thread_idx.x)
    var tid = thread_idx.x
    var dest_row = ROW.null()
    var dest_col = COL.null()
    if C.dim(0) > output_row and C.dim(1) > output_col:
        if tid == 0:
            var row = A.tile[1, cols](output_row, 0)
            dest_row = ROW.stack_allocation()
            dest_row.copy_from(row)
            var col = B.tile[rows, 1](0, output_col)
            dest_col = COL.stack_allocation()
            dest_col.copy_from(col)
        barrier()
        var accum = Scalar[dtype](0)
        for i in range(A.dim(1)):
            accum += (dest_row[0, i] * dest_col[i, 0])[0]
        C[output_row, output_col] = accum


fn main() raises:
    var ctx = DeviceContext()

    var input_buffer_A = ctx.enqueue_create_buffer[dtype](num_elems)
    var input_buffer_B = ctx.enqueue_create_buffer[dtype](num_elems)
    var output_buffer_C = ctx.enqueue_create_buffer[dtype](blocks * blocks)


    var A_src = Tensor[dtype].rand(blocks, threads)
    var B_src = Tensor[dtype].rand(threads, blocks)
    with input_buffer_A.map_to_host() as A_host_buffer:
        memcpy(
            dest=A_host_buffer.unsafe_ptr(),
            src=A_src.data_ptr(),
            count=A_src.numels(),
        )
    with input_buffer_B.map_to_host() as B_host_buffer:
        memcpy(
            dest=B_host_buffer.unsafe_ptr(),
            src=B_src.data_ptr(),
            count=B_src.numels(),
        )
    var A = Matrix_A(input_buffer_A)
    var B = Matrix_B(input_buffer_B)
    var C = Matrix_C(output_buffer_C)
    with output_buffer_C.map_to_host() as output:
        memset_zero(output.unsafe_ptr(), count=blocks * blocks)
    ctx.enqueue_function[matmul_2d, matmul_2d](
        A, B, C, grid_dim=blocks, block_dim=threads
    )

    ctx.synchronize()
    var expect = A_src.matmul[track_grad=False](B_src)
    var actual = Tensor[dtype].zeros(blocks, blocks)

    with output_buffer_C.map_to_host() as C_host_buff:
        memcpy(
            dest=actual.data_ptr(),
            src=C_host_buff.unsafe_ptr(),
            count=len(C_host_buff),
        )

    assert_true(actual.all_close(expect))

