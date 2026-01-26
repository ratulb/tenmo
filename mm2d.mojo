from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import block_idx, thread_idx
from layout import Layout, LayoutTensor
from tenmo import Tensor
from memory import memcpy
from testing import assert_true


comptime dtype = DType.float32

comptime blocks = 4
comptime threads = 5
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
        for col in range(A.dim(1)):
            var a_val = A[output_row, col]
            var b_val = B[col, output_col]
            accum += a_val[0] * b_val[0]
        C[output_row, output_col] = accum


fn main() raises:
    var ctx = DeviceContext()

    var input_buffer_A = ctx.enqueue_create_buffer[dtype](num_elems)
    var input_buffer_B = ctx.enqueue_create_buffer[dtype](num_elems)
    var output_buffer_C = ctx.enqueue_create_buffer[dtype](blocks * blocks)

    output_buffer_C.enqueue_fill(0)

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

    ctx.enqueue_function[matmul_2d, matmul_2d](
        A, B, C, grid_dim=blocks, block_dim=threads
    )

    ctx.synchronize()

    var expect = A_src.matmul[track_grad=False](B_src)
    var actual = Tensor[dtype].zeros(blocks, blocks)

    with output_buffer_C.map_to_host() as C_host_buff:
        memcpy(
            dest=actual.buffer.buffer.data,
            src=C_host_buff.unsafe_ptr(),
            count=len(C_host_buff),
        )

    assert_true(actual == expect)
    actual.print()

    print("\nExpected\n")
    expect.print()
