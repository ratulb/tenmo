from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import block_idx, thread_idx
from layout import Layout, LayoutTensor
from tenmo import Tensor
from memory import memcpy, memset_zero
from testing import assert_true
from common_utils import now
from shapes import Shape

comptime dtype = DType.float32

comptime blocks = 20
comptime threads = 50
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

    var A_src = Tensor[dtype].rand(blocks, threads)
    var B_src = Tensor[dtype].rand(threads, blocks)
    A_src.write_to_device_buffer(input_buffer_A)
    B_src.write_to_device_buffer(input_buffer_B)

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

    var actual = Tensor[dtype].from_device_buffer(output_buffer_C, Shape(blocks, blocks))
    assert_true(actual.all_close(expect))
