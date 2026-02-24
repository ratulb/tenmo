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


fn pass_to_device[
    dtype: DType
](A: UnsafePointer[Scalar[dtype], MutAnyOrigin],):
    for i in range(5):
        (A + i)[] = i + 99


from device import GPU


fn launch() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].arange(12)
    A.to_gpu()
    ctx = A.buffer.device.value().kind[GPU]()
    ref device_buffer = A.buffer.device_buffer.value()
    var compiled_func = ctx.compile_function[
        pass_to_device[dtype],
        pass_to_device[dtype],
    ]()

    ctx.enqueue_function(
        compiled_func,
        device_buffer,
        grid_dim=1,
        block_dim=1,
    )

    ctx.synchronize()
    _ = A.to_cpu()
    A.print()

    print("Post synchonize")


fn main() raises:
    launch()
    print("Launch success")
