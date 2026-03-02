from memory import AddressSpace, stack_allocation, memset
from gpu import thread_idx, block_dim, grid_dim, block_idx, barrier

from gpu.host import DeviceContext


from gpu.primitives.id import lane_id, warp_id
from gpu.primitives.warp import shuffle_down
from gpu.globals import WARP_SIZE

from tenmo import Tensor
from testing import assert_true
from common_utils import panic
from shapes import Shape
from strides import Strides
from intarray import IntArray
from indexhelper import IndexIterator
from array import Array


fn pass_to_device[
    dtype: DType
](A: UnsafePointer[Scalar[dtype], MutAnyOrigin], shape: Array,):
    var size = len(shape)
    print("The size: ", size, shape[0])
    for i in range(shape[0]):
        (A + i)[] = (A + i)[] * 42
        print("Here: ", A[i])	

from device import GPU


fn launch() raises:
    var gpu = GPU()
    var ctx = gpu()
    comptime dtype = DType.float32
    var compiled_func = ctx.compile_function[
        pass_to_device[dtype],
        pass_to_device[dtype],
    ]()

    var A = Tensor[dtype].arange(7)
    ref A_gpu = A.to_gpu(gpu)
    ref A_device_buffer = A_gpu.buffer.get_device_state().buffer

    ctx.enqueue_function(
        compiled_func,
        A_device_buffer,
        Array(7),
        grid_dim=1,
        block_dim=1,
    )

    gpu().synchronize()
    A_cpu = A_gpu.to_cpu()
    ctx.synchronize()
    A.print()
    A_cpu.print()

    print("Post synchonize")


fn main() raises:
    launch()
    print("Launch success")
    comptime dtype = DType.float32
    var ctx = DeviceContext()
    var buffer = ctx.enqueue_create_buffer[dtype](5)
    buffer.enqueue_fill(42)
    print("The buffer: ", buffer)
    with buffer.map_to_host() as h_buffer:
        h_buffer[0] = h_buffer[0] * 33

        print("The buffer: ", buffer, h_buffer[0], h_buffer)

    print("The buffe ****r: ", buffer)
