from memory import AddressSpace, stack_allocation
from gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from gpu.host import DeviceContext

from gpu.primitives.id import lane_id, warp_id
from gpu.primitives.warp import shuffle_down
from gpu.globals import WARP_SIZE


from testing import assert_true
from common_utils import panic
from shapes import Shape
from strides import Strides
from intarray import IntArray

fn pass_to_device[dtype: DType](shape: UnsafePointer[Scalar[DType.int64], MutAnyOrigin], strides: UnsafePointer[Int64, MutAnyOrigin], intarray: UnsafePointer[Int64, MutAnyOrigin], offset: Int = 2):
    print("Shape: ", Shape.read_from(shape))
    print("Strides: ", Strides.read_from(strides))
    print("IntArray: ", IntArray.read_from(intarray))
    print("Offset: ", offset)

fn launch() raises:
    var ctx = DeviceContext()

    comptime dtype = DType.int64
    var compiled_func = ctx.compile_function[
        pass_to_device[dtype],
        pass_to_device[dtype],
    ]()

    var shape = Shape(10, 3, 15)
    var strides = Strides(45, 15, 1)
    var intarray  = IntArray(5, 5, 10, 100)
    var offset = 3
    var shape_buffer = ctx.enqueue_create_buffer[dtype](5)
    var strides_buffer = ctx.enqueue_create_buffer[dtype](5)
    var intarray_buffer = ctx.enqueue_create_buffer[dtype](intarray.get_buffer_write_length())

    shape.write_to_device_buffer(shape_buffer)
    strides.write_to_device_buffer(strides_buffer)
    intarray.write_to_device_buffer(intarray_buffer)

    ctx.enqueue_function(
        compiled_func,
        shape_buffer,
        strides_buffer,
        intarray_buffer,
        offset,
        grid_dim=1,
        block_dim=1,
    )

    ctx.synchronize()


fn main() raises:
    launch()
    print("Launch success")
