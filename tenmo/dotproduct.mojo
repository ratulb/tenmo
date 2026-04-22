from .tensor import Tensor
from .common_utils import panic
from .backpropagation import BackwardFnArg, BACKWARD_DOT
from .mnemonics import AddTensor
from .gradbox import Gradbox
from .shapes import Shape
from std.memory import stack_allocation, AddressSpace
from std.gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from std.os.atomic import Atomic, Consistency
from std.sys import has_accelerator
from std.gpu.primitives.id import lane_id, warp_id
from std.gpu.primitives.warp import shuffle_down
from std.gpu.globals import WARP_SIZE
from std.sys import simd_width_of
from .ancestry import Ancestor


@fieldwise_init
struct DotBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var scalar_grad_value = gradbox.item()  # Scalar
        var grad_shares: List[
            Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]
        ] = []
        var tensor_lhs_ref = output.ancestry().get(0)
        var tensor_rhs_ref = output.ancestry().get(1)

        var tensor_lhs = Tensor[Self.dtype](
            tensor_lhs_ref.buffer(), requires_grad=tensor_lhs_ref.requires_grad
        )
        var tensor_rhs = Tensor[Self.dtype](
            tensor_rhs_ref.buffer(), requires_grad=tensor_rhs_ref.requires_grad
        )

        if tensor_lhs.requires_grad:
            var grad_tensor = tensor_rhs.__mul__[track_grad=False](
                scalar_grad_value
            )
            var gradbox_lhs = grad_tensor.as_gradbox(
                share=False, contiguous=False
            )
            grad_shares.append((tensor_lhs_ref, gradbox_lhs^, AddTensor))

        if tensor_rhs.requires_grad:
            var grad_tensor = tensor_lhs.__mul__[track_grad=False](
                scalar_grad_value
            )
            var gradbox_rhs = grad_tensor.as_gradbox(
                share=False, contiguous=False
            )

            grad_shares.append((tensor_rhs_ref^, gradbox_rhs^, AddTensor))

        return grad_shares^


@fieldwise_init
struct Dot[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](lhs: Tensor[Self.dtype], rhs: Tensor[Self.dtype],) -> Tensor[Self.dtype]:
        rank_lhs = lhs.rank()
        rank_rhs = rhs.rank()
        if not rank_lhs == rank_rhs and not rank_lhs <= 1:
            panic("Tensor → dot: not supported for rank > 1")
        var numels_lhs = lhs.numels()
        var numels_rhs = rhs.numels()
        if not numels_lhs == numels_rhs:
            panic(
                "Tensor → dot: size does not match",
                String(numels_lhs),
                String(numels_rhs),
            )

        var out: Tensor[Self.dtype]

        comptime if has_accelerator():
            if lhs.is_on_gpu() and rhs.is_on_gpu():
                try:
                    out = DotproductKernel[Self.dtype].launch[
                        suppress_validation=True
                    ](lhs, rhs)

                except e:
                    print(e)
                    panic("Dot - GPU operation failed")
                    # Not reachable
                    out = Tensor[Self.dtype].scalar(0)
            else:
                out = Tensor[Self.dtype].scalar(
                    lhs.buffer.contiguous_buffer().dot(
                        rhs.buffer.contiguous_buffer()
                    ),
                    requires_grad=False,
                )
        else:
            out = Tensor[Self.dtype].scalar(
                lhs.buffer.contiguous_buffer().dot(
                    rhs.buffer.contiguous_buffer()
                ),
                requires_grad=False,
            )

        comptime if track_grad:
            grad_required = lhs.requires_grad or rhs.requires_grad

            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_DOT
                )

                out.add_ancestry(backwardFnArg^, lhs, rhs)

        return out^


fn dot_product_32[
    dtype: DType, BLOCK_SIZE: Int = 512
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: UInt,
):
    """
    Warp-optimized dot product kernel.

    Uses warp shuffle for efficient reduction:
    - Grid-stride loop for workload distribution
    - Warp-level shuffle reduction (faster than shared memory)
    - Single atomic write per block

    Performance: ~1.5-2× faster than tree reduction.
    """

    # comptime BLOCK_SIZE = 256
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    var warp_sums = stack_allocation[
        NUM_WARPS, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()

    # Thread and block identification
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x

    # =================================================================
    # Phase 1: Grid-Stride Accumulation
    # Each thread processes multiple elements with stride = total_threads
    # =================================================================
    var accum = Scalar[dtype](0)
    var i = gtid
    while i < size:
        accum += a[i] * b[i]
        i += block_dim.x * grid_dim.x

    # =================================================================
    # Phase 2: Warp-Level Reduction (Shuffle)
    # Reduces WARP_SIZE values to 1 using shuffle instructions
    # Complexity: O(log₂ WARP_SIZE) = 5 iterations for 32-thread warp
    # =================================================================
    var lane = lane_id()  # 0 to WARP_SIZE-1
    var warp = warp_id()  # 0 to NUM_WARPS-1

    var offset = UInt32(WARP_SIZE // 2)
    while offset > 0:
        # accum += shuffle_down[dtype, 1](accum, offset)
        accum += shuffle_down(accum, offset)
        offset //= 2

    # =================================================================
    # Phase 3: Store Warp Results
    # First thread of each warp writes to shared memory
    # =================================================================
    if lane == 0:
        warp_sums[warp] = accum

    barrier()  # Ensure all warps have written

    # =================================================================
    # Phase 4: Final Reduction (First Warp Only)
    # Reduces NUM_WARPS values to 1 using shuffle
    # Complexity: O(log₂ NUM_WARPS) = 3 iterations for 8 warps
    # =================================================================
    if warp == 0:
        # Each thread in first warp loads one warp sum
        accum = warp_sums[lane] if lane < UInt(NUM_WARPS) else Scalar[dtype](0)

        # Shuffle reduction
        offset = UInt32(NUM_WARPS // 2)
        while offset > 0:
            # accum += shuffle_down[dtype, 1](accum, offset)
            accum += shuffle_down(accum, offset)
            offset //= 2

        # Single atomic write per block
        if lane == 0:
            _ = Atomic.fetch_add(result, accum)


fn dot_product_64[
    dtype: DType,
    BLOCK_SIZE: Int = 512,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: UInt,
):
    var block_shared_memory = stack_allocation[
        BLOCK_SIZE, Scalar[dtype], address_space=AddressSpace.SHARED
    ]()
    var cache_index = thread_idx.x
    var gtid = cache_index + block_dim.x * block_idx.x
    var accum: Scalar[dtype] = 0
    # Grid-stride loop
    for i in range(gtid, size, block_dim.x * grid_dim.x):
        accum += a[i] * b[i]

    block_shared_memory[cache_index] = accum
    barrier()

    var stride = UInt(block_dim.x // 2)

    while stride > 0:
        if cache_index < stride:
            block_shared_memory[cache_index] += block_shared_memory[
                cache_index + stride
            ]
        barrier()
        stride //= 2

    # only thread 0 of each block writes the final result
    if cache_index == 0:
        _ = Atomic.fetch_add(result, block_shared_memory[0])


struct DotproductKernel[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    fn launch[
        num_blocks: Int = 1,
        threads_per_block: Int = 512,
        suppress_validation: Bool = False,
    ](A: Tensor[Self.dtype], B: Tensor[Self.dtype]) raises -> Tensor[
        Self.dtype
    ]:
        comptime assert (
            threads_per_block <= 512
        ), "Threads per block should be <= 512"
        var rank = A.rank()
        var numels = A.numels()

        comptime if not suppress_validation:
            if rank != 1 or B.rank() != 1:
                panic(
                    "Dot product expects 1D tensors. Found",
                    "A rank: ",
                    String(rank),
                    "and B rank: ",
                    String(B.rank()),
                )
            if numels != B.numels():
                panic(
                    "Tensor lengths do not match.",
                    "A length: ",
                    String(numels),
                    "B length: ",
                    String(B.numels()),
                )

        ref A_device_state = A.buffer.device_state.value()
        ref B_device_state = B.buffer.device_state.value()

        var device_context = A_device_state.gpu()

        ref A_buffer = A_device_state.device_buffer()
        ref B_buffer = B_device_state.device_buffer()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](1)
        result_buffer.enqueue_fill(0)

        # Warp possibly would not work for DType.float64 in NVIDIA GPU
        comptime use_32_kernel = True if simd_width_of[
            Self.dtype
        ]() > 8 else False

        comptime if use_32_kernel:
            var compiled_func = device_context.compile_function[
                dot_product_32[Self.dtype, BLOCK_SIZE=threads_per_block],
                dot_product_32[Self.dtype, BLOCK_SIZE=threads_per_block],
            ]()

            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                B_buffer,
                UInt(numels),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        else:
            var compiled_func = device_context.compile_function[
                dot_product_64[Self.dtype, BLOCK_SIZE=threads_per_block],
                dot_product_64[Self.dtype, BLOCK_SIZE=threads_per_block],
            ]()

            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                B_buffer,
                UInt(numels),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        device_context.synchronize()

        return Tensor[Self.dtype].from_device_buffer(result_buffer, Shape())

