from gpu.host import DeviceContext, DeviceAttribute


fn main() raises:
    var ctx = DeviceContext()

    print("\n\n===========Accelerator details============\n\n")

    print(ctx.name(), ctx.api(), ctx.id())

    var MAX_BLOCKS_PER_MULTIPROCESSOR = ctx.get_attribute(
        DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR
    )
    print("MAX_BLOCKS_PER_MULTIPROCESSOR: ", MAX_BLOCKS_PER_MULTIPROCESSOR)

    var CLOCK_RATE = ctx.get_attribute(DeviceAttribute.CLOCK_RATE)
    print("CLOCK_RATE: ", CLOCK_RATE)

    var MAX_BLOCK_DIM_X = ctx.get_attribute(DeviceAttribute.MAX_BLOCK_DIM_X)
    print("MAX_BLOCK_DIM_X: ", MAX_BLOCK_DIM_X)

    var MAX_BLOCK_DIM_Y = ctx.get_attribute(DeviceAttribute.MAX_BLOCK_DIM_Y)
    print("MAX_BLOCK_DIM_Y: ", MAX_BLOCK_DIM_Y)

    var MAX_BLOCK_DIM_Z = ctx.get_attribute(DeviceAttribute.MAX_BLOCK_DIM_Z)
    print("MAX_BLOCK_DIM_Z: ", MAX_BLOCK_DIM_Z)

    var MAX_GRID_DIM_X = ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X)
    print("MAX_GRID_DIM_X: ", MAX_GRID_DIM_X)

    var MAX_GRID_DIM_Y = ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_Y)
    print("MAX_GRID_DIM_Y: ", MAX_GRID_DIM_Y)

    var MAX_GRID_DIM_Z = ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_Z)
    print("MAX_GRID_DIM_Z: ", MAX_GRID_DIM_Z)

    var MAX_REGISTERS_PER_BLOCK = ctx.get_attribute(
        DeviceAttribute.MAX_REGISTERS_PER_BLOCK
    )
    print("MAX_REGISTERS_PER_BLOCK : ", MAX_REGISTERS_PER_BLOCK)

    var MAX_REGISTERS_PER_MULTIPROCESSOR = ctx.get_attribute(
        DeviceAttribute.MAX_REGISTERS_PER_MULTIPROCESSOR
    )
    print(
        "MAX_REGISTERS_PER_MULTIPROCESSOR : ", MAX_REGISTERS_PER_MULTIPROCESSOR
    )

    var MAX_SHARED_MEMORY_PER_BLOCK = ctx.get_attribute(
        DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK
    )
    print("MAX_SHARED_MEMORY_PER_BLOCK : ", MAX_SHARED_MEMORY_PER_BLOCK)

    var MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = ctx.get_attribute(
        DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    )
    print(
        "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN : ",
        MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    )

    var MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = ctx.get_attribute(
        DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
    )
    print(
        "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR : ",
        MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
    )

    var MAX_THREADS_PER_BLOCK = ctx.get_attribute(
        DeviceAttribute.MAX_THREADS_PER_BLOCK
    )
    print("MAX_THREADS_PER_BLOCK : ", MAX_THREADS_PER_BLOCK)

    var MAX_THREADS_PER_MULTIPROCESSOR = ctx.get_attribute(
        DeviceAttribute.MAX_THREADS_PER_MULTIPROCESSOR
    )
    print("MAX_THREADS_PER_MULTIPROCESSOR : ", MAX_THREADS_PER_MULTIPROCESSOR)

    var MULTIPROCESSOR_COUNT = ctx.get_attribute(
        DeviceAttribute.MULTIPROCESSOR_COUNT
    )
    print("MULTIPROCESSOR_COUNT : ", MULTIPROCESSOR_COUNT)

    var WARP_SIZE = ctx.get_attribute(DeviceAttribute.WARP_SIZE)
    print("WARP_SIZE : ", WARP_SIZE)

    var COOPERATIVE_LAUNCH = ctx.get_attribute(
        DeviceAttribute.COOPERATIVE_LAUNCH
    )
    print("COOPERATIVE_LAUNCH : ", COOPERATIVE_LAUNCH)

    var COMPUTE_CAPABILITY_MAJOR = ctx.get_attribute(
        DeviceAttribute.COMPUTE_CAPABILITY_MAJOR
    )
    print("COMPUTE_CAPABILITY_MAJOR : ", COMPUTE_CAPABILITY_MAJOR)

    var COMPUTE_CAPABILITY_MINOR = ctx.get_attribute(
        DeviceAttribute.COMPUTE_CAPABILITY_MINOR
    )
    print("COMPUTE_CAPABILITY_MINOR : ", COMPUTE_CAPABILITY_MINOR)
