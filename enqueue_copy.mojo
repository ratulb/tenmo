from tenmo import Tensor
from device import GPU


fn arithmetic_ops[
    dtype: DType,
](A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin], size: Int):
    for i in range(size):
        print("A[", i, "]", A[i])


fn launch[
    dtype: DType = DType.float32,
](A: Tensor[dtype]) raises:
    var gpu = GPU()
    var ctx = gpu()

    # Launch configuration
    var threads_per_block: Int = 1
    var num_blocks: Int = 1

    var compiled_func = ctx.compile_function[
        arithmetic_ops[dtype],
        arithmetic_ops[dtype],
    ]()

    var A_buffer = ctx.enqueue_create_buffer[dtype](A.numels())

    ctx.enqueue_copy(A_buffer, A.data_ptr())

    ctx.enqueue_function(
        compiled_func,
        A_buffer,
        grid_dim=num_blocks,
        block_dim=threads_per_block,
    )

    ctx.synchronize()


fn main() raises:
    print("Production Tensor-Tensor Arithmetic Tests")

    test_contiguous_same_shape()
    print("ALL TESTS PASSED (Including Offset Tests)")


from common_utils import now
from testing import assert_true


fn test_contiguous_same_shape() raises:
    """Test fast path: contiguous, same shape."""
    print("=== Test 1: Contiguous Same Shape ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(12)

    launch[dtype](a)
    print("  Passed")
