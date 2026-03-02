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


fn pass_to_device[
    dtype: DType
](A: UnsafePointer[Scalar[dtype], MutAnyOrigin], shape: Array,):
    var size = len(shape)
    for i in range(size):
        (A + i)[] = (A + i)[] * 42


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
    var A_gpu = A.to_gpu(gpu)
    ref A_device_buffer = A.buffer.get_device_state().buffer

    ctx.enqueue_function(
        compiled_func,
        A_device_buffer,
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
