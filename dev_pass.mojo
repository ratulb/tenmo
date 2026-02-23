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
](
    A: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    shape: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    strides: UnsafePointer[Int64, MutAnyOrigin],
    offset: Int,
    rank: Int,
):
    print("Shape: ", Shape.read_from(shape))
    print("Strides: ", Strides.read_from(strides))
    print("Offset: ", offset)
    print("Rank: ", rank)
    #memset(A, 42, 5)
    for i in range(5):
        (A + i)[] = i + 99
    ######## Shape iteration ###########
    var count = 0
    for coord in Shape.read_from(shape):
        print("Coord: ", coord)
        count += 1
        if count == 5:
            break
    count = 0
    var index_iterator = IndexIterator(
        Pointer(to=Shape.read_from(shape)),
        Pointer(to=Strides.read_from(strides)),
        rank,
    )
    for idx in index_iterator:
        print("Index: ", idx)


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
    var (A_device_buffer, shape_buffer, strides_buffer, offset, rank) = A.to_gpu(gpu).value()
    _="""var shape = Shape(1, 3, 2)
    var strides = Strides(6, 2, 1)
    var intarray = IntArray(5, 5, 10, 100)
    var offset = 3

    # .write_length() - better to use
    var shape_buffer = ctx.enqueue_create_buffer[DType.int64](
        5
    )  # 5 because size and capacity needs to be acoounted for
    var strides_buffer = ctx.enqueue_create_buffer[DType.int64](
        5
    )  # can use .write_length()
    var intarray_buffer = ctx.enqueue_create_buffer[DType.int64](
        intarray.write_length()
    )

    shape.write_to_device_buffer(shape_buffer)
    strides.write_to_device_buffer(strides_buffer)
    intarray.write_to_device_buffer(intarray_buffer)"""

    ctx.enqueue_function(
        compiled_func,
        A_device_buffer,
        shape_buffer,
        strides_buffer,
        offset,
        rank,
        grid_dim=1,
        block_dim=1,
    )

    gpu().synchronize()
    _ = A.to_cpu()
    ctx.synchronize()
    A.print()

    print("Post synchonize")


fn main() raises:
    launch()
    print("Launch success")
