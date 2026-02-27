from tenmo import Tensor
from common_utils import panic
from backpropagation import Delegate, BackwardFn, BACKWARD_DOT
from mnemonics import AddTensor
from gradbox import Gradbox
from shapes import Shape
from memory import stack_allocation, AddressSpace
from gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from os.atomic import Atomic, Consistency
from sys import has_accelerator
from gpu.primitives.id import lane_id, warp_id
from gpu.primitives.warp import shuffle_down
from gpu.globals import WARP_SIZE

# from gpu.host import DeviceContext


@fieldwise_init
@register_passable
struct DotBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_DOT

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var scalar_grad_value = gradbox.item()  # Scalar
        var grad_shares: List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ] = []
        var tensor_lhs = output.ancestry().get(0)
        var tensor_rhs = output.ancestry().get(1)

        if tensor_lhs.requires_grad:
            var grad_tensor = tensor_rhs.__mul__[track_grad=False](
                scalar_grad_value
            )
            var gradbox_lhs = grad_tensor.as_gradbox(
                share=False, contiguous=False
            )
            grad_shares.append((tensor_lhs, gradbox_lhs^, AddTensor))

        if tensor_rhs.requires_grad:
            var grad_tensor = tensor_lhs.__mul__[track_grad=False](
                scalar_grad_value
            )
            var gradbox_rhs = grad_tensor.as_gradbox(
                share=False, contiguous=False
            )

            grad_shares.append((tensor_rhs^, gradbox_rhs^, AddTensor))

        return grad_shares^


@fieldwise_init
@register_passable
struct Dot[dtype: DType](ImplicitlyCopyable & Movable):
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
                numels_lhs.__str__(),
                numels_rhs.__str__(),
            )

        var out: Tensor[Self.dtype]

        @parameter
        if has_accelerator():
            if lhs.is_on_gpu() and rhs.is_on_gpu():
                try:
                    start = now()
                    out = DotproductKernel[Self.dtype].launch[
                        suppress_validation=True
                    ](lhs, rhs)

                    print("GPU dot took: ", (now() - start) * 1000, "ms")
                except e:
                    print(e)
                    print("Dot - GPU operation failed. Failling back on CPU")

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
        else:
            out = Tensor[Self.dtype].scalar(
                lhs.buffer.contiguous_buffer().dot(
                    rhs.buffer.contiguous_buffer()
                ),
                requires_grad=False,
            )

        @parameter
        if track_grad:
            grad_required = lhs.requires_grad or rhs.requires_grad

            if grad_required:
                out.requires_grad_(True)
                backward_fn = DotBackward[Self.dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(lhs, rhs)

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
        NUM_WARPS, Scalar[dtype], address_space = AddressSpace.SHARED
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
        accum = warp_sums[lane] if lane < NUM_WARPS else Scalar[dtype](0)

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
        BLOCK_SIZE, Scalar[dtype], address_space = AddressSpace.SHARED
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
        constrained[
            threads_per_block <= 512,
            "Threads per block should be <= 512",
        ]()
        var rank = A.rank()
        var numels = A.numels()

        @parameter
        if not suppress_validation:
            if rank != 1 or B.rank() != 1:
                panic(
                    "Dot product expects 1D tensors. Found",
                    "A rank: ",
                    rank.__str__(),
                    "and B rank: ",
                    B.rank().__str__(),
                )
            if numels != B.numels():
                panic(
                    "Tensor lengths do not match.",
                    "A length: ",
                    numels.__str__(),
                    "B length: ",
                    B.numels().__str__(),
                )

        ref A_device_state = A.buffer.device_state.value()
        ref B_device_state = B.buffer.device_state.value()

        var device_context = A_device_state.gpu()

        var compiled_func = device_context.compile_function[
            dot_product_64[Self.dtype, BLOCK_SIZE=threads_per_block],
            dot_product_64[Self.dtype, BLOCK_SIZE=threads_per_block],
        ]()

        ref A_buffer = A_device_state.device_buffer()
        ref B_buffer = B_device_state.device_buffer()

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](1)
        result_buffer.enqueue_fill(0)

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


from mnemonics import dot
from testing import assert_true
from common_utils import now


fn main() raises:
    var SIZE = 2 << 22
    comptime dtype = DType.float32
    var tensor_a = Tensor[dtype].ones(SIZE)
    var tensor_b = Tensor[dtype].ones(SIZE)
    start = now()
    var expect = tensor_a.matmul[mode=dot](tensor_b)
    print("CPU dot took: ", (now() - start) * 1000, "ms")

    var tensor_A = tensor_a.to_gpu()
    var tensor_B = tensor_b.to_gpu()

    var result = tensor_A.dot(tensor_B)

    print(expect.item(), result.item())
    assert_true(result.all_close(expect))

    assert_true(tensor_A.to_cpu() == tensor_a)
    assert_true(tensor_B.to_cpu() == tensor_b)

    SIZE = 2 << 24
    comptime dtype2 = DType.float64
    tensor_a2 = Tensor[dtype2].rand(SIZE)
    tensor_b2 = Tensor[dtype2].rand(SIZE)
    start = now()
    expect2 = tensor_a2.matmul[mode=dot](tensor_b2)
    print("CPU dot took: ", (now() - start) * 1000, "ms")

    tensor_A2 = tensor_a2.to_gpu()
    tensor_B2 = tensor_b2.to_gpu()

    result2 = tensor_A2.dot(tensor_B2)

    print(expect2.item(), result2.item())
    assert_true(result2.all_close(expect2))

    assert_true(tensor_A2.to_cpu() == tensor_a2)
    assert_true(tensor_B2.to_cpu() == tensor_b2)

    print("Launch success")
