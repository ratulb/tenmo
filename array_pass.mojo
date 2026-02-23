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


fn pass_to_device(array: Array):
    print("This guy came inside", array)


from device import GPU


fn launch() raises:
    var gpu = GPU()
    var ctx = gpu()
    var compiled_func = ctx.compile_function[pass_to_device, pass_to_device]()

    ctx.enqueue_function(
        compiled_func,
        Array(1, 2, 3),
        grid_dim=1,
        block_dim=1,
    )

    ctx.synchronize()

    print("Post synchonize")


fn main() raises:
    launch()
    print("Launch success")
