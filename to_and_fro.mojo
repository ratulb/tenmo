from memory import AddressSpace, stack_allocation, memset
from gpu import thread_idx, block_dim, grid_dim, block_idx, barrier

# from gpu.host import DeviceContext


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
from buffers import Buffer

fn pass_to_device[
    dtype: DType
](A: UnsafePointer[Scalar[dtype], MutAnyOrigin], shape: Array):
    for i in range(9):
        (A + i)[] = (A + i)[] * 99
    #print("The incoming shape is: ", shape)
    #in_kernel = Buffer[dtype](9,A,copy=True)
    #print(in_kernel)

from device import GPU


fn launch() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].arange(12)
    var B = A[3:]
    var C = B.reshape(3, 3)
    C.to_gpu()
    ctx = C.buffer.device.value().kind[GPU]()
    var device_buffer = C.buffer.device_buffer.value()
    var compiled_func = ctx.compile_function[
        pass_to_device[dtype],
        pass_to_device[dtype],
    ]()

    ctx.enqueue_function(
        compiled_func,
        device_buffer,
        C.shape().array(),
        grid_dim=1,
        block_dim=1,
    )

    ctx.synchronize()
    C.to_cpu()
    A.print()
    B.print()
    C.print()
    #print("Before syncing\n")
    #C.to_cpu()
    #C.print()
    #with C.buffer.device_buffer.value().map_to_host() as dv:
    #    print("The good dv is: ", dv)

    print("Post synchonize")


fn main() raises:
    launch()
    print("Launch success")
