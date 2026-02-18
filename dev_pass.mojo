from memory import AddressSpace, stack_allocation
from gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from gpu.host import DeviceContext

from gpu.primitives.id import lane_id, warp_id
from gpu.primitives.warp import shuffle_down
from gpu.globals import WARP_SIZE


from testing import assert_true
from common_utils import panic
from shapes import Shape


fn pass_to_device(shape: Shape):
    print("Shape: ", shape)

fn launch() raises:
    var ctx = DeviceContext()

    var compiled_func = ctx.compile_function[
        pass_to_device,
        pass_to_device,
    ]()
    shape = Shape(10, 3, 15)
    ctx.enqueue_function(
        compiled_func,
        shape,
        grid_dim=1,
        block_dim=1,
    )

    ctx.synchronize()


fn main() raises:
    launch()
    print("Launch success")
