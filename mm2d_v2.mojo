from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import block_idx, thread_idx
from layout import Layout, LayoutTensor
from tenmo import Tensor
from memory import memcpy, memset_zero
from testing import assert_true
from common_utils import now
from collections import InlineArray

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


fn matmul_2d(A: Matrix_A, B: Matrix_B, C: Matrix_C):
    var output_row = Int(block_idx.x)
    var output_col = Int(thread_idx.x)

    if C.dim(0) > output_row and C.dim(1) > output_col:
        var accum = Scalar[dtype](0)
        var row = A.tile[1, cols](output_row, 0)
        var dest_row = LayoutTensor[
            dtype,
            Layout.row_major(1, cols),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()
        dest_row.copy_from(row)
        var col = B.tile[rows, 1](0, output_col)
        var dest_col = LayoutTensor[
            dtype,
            Layout.row_major(rows, 1),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ].stack_allocation()
        dest_col.copy_from(col)

        for i in range(A.dim(1)):
            accum += (dest_row[0, i] * dest_col[i, 0])[0]
        C[output_row, output_col] = accum


fn main() raises:
    var ctx = DeviceContext()

    var input_buffer_A = ctx.enqueue_create_buffer[dtype](num_elems)
    var input_buffer_B = ctx.enqueue_create_buffer[dtype](num_elems)
    var output_buffer_C = ctx.enqueue_create_buffer[dtype](blocks * blocks)

    # output_buffer_C.enqueue_fill(0)

    var A_src = Tensor[dtype].rand(blocks, threads)
    var B_src = Tensor[dtype].rand(threads, blocks)
    with input_buffer_A.map_to_host() as A_host_buffer:
        memcpy(
            dest=A_host_buffer.unsafe_ptr(),
            src=A_src.buffer.buffer.data,
            count=A_src.numels(),
        )
    with input_buffer_B.map_to_host() as B_host_buffer:
        memcpy(
            dest=B_host_buffer.unsafe_ptr(),
            src=B_src.buffer.buffer.data,
            count=B_src.numels(),
        )
    var A = Matrix_A(input_buffer_A)
    var B = Matrix_B(input_buffer_B)
    var C = Matrix_C(output_buffer_C)
    var start = now()
    for _ in range(100):
        with output_buffer_C.map_to_host() as output:
            memset_zero(output.unsafe_ptr(), count=blocks * blocks)
        ctx.enqueue_function[matmul_2d, matmul_2d](
            A, B, C, grid_dim=blocks, block_dim=threads
        )

    ctx.synchronize()
    print("GPU matmul time: ", (now() - start) * 1000, "ms")
    var expect: Tensor[dtype] = Tensor[dtype].scalar(0)
    start = now()
    for _ in range(100):
        expect = A_src.matmul[track_grad=False](B_src)
    print("CPU matmul time: ", (now() - start) * 1000, "ms")
    var actual = Tensor[dtype].zeros(blocks, blocks)

    with output_buffer_C.map_to_host() as C_host_buff:
        memcpy(
            dest=actual.buffer.buffer.data,
            src=C_host_buff.unsafe_ptr(),
            count=len(C_host_buff),
        )

    assert_true(actual.all_close(expect))
    # actual.print()

    # print("\nExpected\n")
    # expect.print()
    _ = """var array = InlineArray[Float32, num_elems](
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    )
    var lo_tensor = Matrix_A(array)
    print(lo_tensor)
    var src_row0 = lo_tensor.tile[1, cols](0, 0)
    var src_col0 = lo_tensor.tile[rows, 1](0, 0)
    print()
    # print(row0)
    # print(col0)
    var dst_row0 = LayoutTensor[
        dtype,
        Layout.row_major(1, cols),
        MutAnyOrigin,
        # address_space = AddressSpace.SHARED,
        # address_space = AddressSpace.GENERIC,
    ].stack_allocation()
    dst_row0.copy_from(src_row0)
    print(dst_row0)
    var dst_col0 = LayoutTensor[
        dtype,
        Layout.row_major(rows, 1),
        MutAnyOrigin,
        # address_space = AddressSpace.SHARED,
        # address_space = AddressSpace.GENERIC,
    ].stack_allocation()
    dst_col0.copy_from(src_col0)
    print(dst_col0)"""
