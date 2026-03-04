from sys import simd_width_of
from gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from os.atomic import Atomic, Consistency
from memory import AddressSpace, stack_allocation
from utils.numerics import isnan, isinf

from mnemonics import (
    Equal,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
)
from device import DeviceState
from ndbuffer import NDBuffer


fn atomic_and[
    address_space: AddressSpace,
    //,  # Infer address space - we are using this fn for GENERIC/GLOBAL and SHARED address space
    ordering: Consistency = Consistency.SEQUENTIAL,
](
    ptr: UnsafePointer[
        Scalar[DType.uint8], MutAnyOrigin, address_space=address_space
    ],
    mask: Scalar[DType.uint8],
) -> Scalar[DType.uint8]:
    """Atomically performs *ptr &= mask, returning the OLD value.

    Uses a CAS (Compare-And-Swap) loop:
      1. Load the current value as `expected`.
      2. Compute `desired = expected & mask`.
      3. Try compare_exchange(ptr, expected, desired).
         - SUCCESS → ptr was still `expected`, now holds `desired`. Done.
         - FAILURE → another thread changed ptr; `expected` is refreshed
                     automatically by compare_exchange. Retry.
    """
    var expected = ptr[]  # initial load (non-atomic, CAS will validate)

    while True:
        var desired = expected & mask

        # If ptr == expected  →  write desired, return True
        # If ptr != expected  →  update expected with *ptr, return False
        if Atomic.compare_exchange[
            failure_ordering=ordering,
            success_ordering=ordering,
        ](ptr, expected, desired):
            return expected  # return the OLD value (before the AND)
        # `expected` has been refreshed by compare_exchange — just retry


fn all_close[
    dtype: DType,
    rtol: Scalar[dtype] = 1e-5,
    atol: Scalar[dtype] = 1e-8,
    treat_nan_equal: Bool = True,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2,
](
    result: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
):
    constrained[dtype.is_floating_point(), "all_close requires a float dtype"]()

    # ── Grid-stride setup ─────────────────────────────────────────────────────
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    # ── One shared flag per block, lives for the entire kernel launch ─────────
    var block_result = stack_allocation[
        1, Scalar[DType.uint8], address_space = AddressSpace.SHARED
    ]()

    if thread_idx.x == 0:
        block_result[] = 1
    barrier()

    # ── Grid-stride loop ──────────────────────────────────────────────────────
    # FIX 1: break (not return) so every thread reaches the barrier() below.
    while base_idx < size:
        if block_result[] == 0:
            break  # ← was `return` — divergent exit before barrier = deadlock

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            # ── SIMD path ─────────────────────────────────────────────────────
            if i + simd_width <= size:
                var va = A.load[width=simd_width](i)
                var vb = B.load[width=simd_width](i)

                var has_special = (
                    isnan(va).reduce_or()
                    or isnan(vb).reduce_or()
                    or isinf(va).reduce_or()
                    or isinf(vb).reduce_or()
                )

                if has_special:
                    # FIX 2: restore the full three-way if/elif/else.
                    # The previous `else` branch used `==` for finite values
                    # that happened to share a vector with a NaN/Inf lane.
                    for k in range(simd_width):
                        var a_val = va[k]
                        var b_val = vb[k]
                        var lane_ok: Bool

                        if isnan(a_val) or isnan(b_val):
                            lane_ok = (
                                treat_nan_equal
                                and isnan(a_val)
                                and isnan(b_val)
                            )
                        elif isinf(a_val) or isinf(b_val):
                            lane_ok = a_val == b_val
                        else:  # ← finite: must use tolerance, not ==
                            lane_ok = abs(a_val - b_val) <= atol + rtol * abs(
                                b_val
                            )

                        if not lane_ok:
                            _ = atomic_and(block_result, UInt8(0))
                            break
                else:
                    # Pure finite SIMD — mirrors Buffer.all_close
                    var diff = abs(va - vb)
                    var tolerance = atol + rtol * abs(vb)
                    if not diff.le(tolerance).reduce_and():
                        _ = atomic_and(block_result, UInt8(0))

            # ── Scalar tail ───────────────────────────────────────────────────
            else:
                for j in range(size - i):
                    var idx = i + j
                    var a_val = (A + idx)[]
                    var b_val = (B + idx)[]
                    var local_ok: Bool

                    if isnan(a_val) or isnan(b_val):
                        local_ok = (
                            treat_nan_equal and isnan(a_val) and isnan(b_val)
                        )
                    elif isinf(a_val) or isinf(b_val):
                        local_ok = a_val == b_val
                    else:
                        local_ok = abs(a_val - b_val) <= atol + rtol * abs(
                            b_val
                        )

                    if not local_ok:
                        _ = atomic_and(block_result, UInt8(0))
                        break

        base_idx += grid_stride * CHUNK_SIZE

    # ── FIX 3: single barrier + single atomic flush, once, after all strides ──
    # Removed the per-stride double-barrier + global flush. Cross-block polling
    # has no safe observation point and costs more than it saves.
    barrier()
    if thread_idx.x == 0:
        _ = atomic_and(result, block_result[])


@fieldwise_init
@register_passable
struct AllClose[dtype: DType = DType.float32](ImplicitlyCopyable & Movable):
    @staticmethod
    fn launch[
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
        treat_nan_equal: Bool = True,
        simd_width: Int = simd_width_of[Self.dtype](),
        simd_vectors_per_thread: Int = 2,
    ](A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]) raises -> Bool:
        comptime simdwidth = simd_width_of[Self.dtype]()

        # Launch configuration
        var (threads_per_block, num_blocks) = Self.launch_config(A.numels())

        ref A_device_state = A.device_state.value()
        ref B_device_state = B.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()
        var result_buffer = device_context.enqueue_create_buffer[DType.uint8](1)

        ref A_buffer = A_device_state.device_buffer()
        ref B_buffer = B_device_state.device_buffer()

        var compiled_func = device_context.compile_function[
            all_close[
                Self.dtype,
                rtol,
                atol,
                treat_nan_equal,
                simdwidth,
                2 * simdwidth,
            ],
            all_close[
                Self.dtype,
                rtol,
                atol,
                treat_nan_equal,
                simdwidth,
                2 * simdwidth,
            ],
        ]()

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            B_buffer,
            A.numels(),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()
        var all_close_result: Bool
        with result_buffer.map_to_host() as host_buffer:
            all_close_result = True if host_buffer[0] == 1 else False
        return all_close_result

    @staticmethod
    fn launch_config(output_size: Int) -> Tuple[Int, Int]:
        """Launch configuration."""
        var threads_per_block: Int
        var num_blocks: Int

        if output_size < 4096:
            threads_per_block = 128
            num_blocks = (output_size + 127) // 128
        elif output_size < 65536:
            threads_per_block = 256
            num_blocks = min((output_size + 255) // 256, 128)
        else:
            threads_per_block = 512
            num_blocks = min((output_size + 511) // 512, 512)

        return num_blocks, threads_per_block


fn compare[
    op_code: Int,
    dtype: DType,
    simd_width: Int = 1 if dtype == DType.bool else simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    A_offset: Int,
    B_offset: Int,
    size: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](A_offset + i)
                var vec_b = B.load[width=simd_width](B_offset + i)
                var vec_result: SIMD[DType.bool, simd_width]

                @parameter
                if op_code == Equal:
                    vec_result = vec_a.eq(vec_b)
                elif op_code == NotEqual:
                    vec_result = vec_a.ne(vec_b)
                elif op_code == GreaterThan:
                    vec_result = vec_a.gt(vec_b)
                elif op_code == GreaterThanEqual:
                    vec_result = vec_a.ge(vec_b)
                elif op_code == LessThan:
                    vec_result = vec_a.lt(vec_b)
                else:  # LessThanEqual
                    vec_result = vec_a.le(vec_b)

                # Store results element-by-element (required for bool bit-packing)
                # result.store[width=simd_width](i, vec_result)
                for idx in range(simd_width):
                    (result + i + idx)[] = vec_result[idx]

            else:
                for j in range(size - i):
                    var idx = i + j
                    var res: Scalar[DType.bool]

                    @parameter
                    if op_code == Equal:
                        res = A[A_offset + idx] == B[B_offset + idx]
                    elif op_code == NotEqual:
                        res = A[A_offset + idx] != B[B_offset + idx]
                    elif op_code == GreaterThan:
                        res = A[A_offset + idx] > B[B_offset + idx]
                    elif op_code == GreaterThanEqual:
                        res = A[A_offset + idx] >= B[B_offset + idx]
                    elif op_code == LessThan:
                        res = A[A_offset + idx] < B[B_offset + idx]
                    else:  # LessThanEqual
                        res = A[A_offset + idx] <= B[B_offset + idx]

                    result[idx] = res

        base_idx += grid_stride * CHUNK_SIZE


@fieldwise_init
@register_passable
struct Compare[dtype: DType = DType.float32](ImplicitlyCopyable & Movable):
    @staticmethod
    fn launch[
        op_code: Int,
    ](A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]) raises -> NDBuffer[
        DType.bool
    ]:
        comptime simdwidth = simd_width_of[Self.dtype]()
        var output_shape = A.shape

        var output_size = output_shape.num_elements()

        # Launch configuration
        var (threads_per_block, num_blocks) = Self.launch_config(output_size)

        ref A_device_state = A.device_state.value()
        ref B_device_state = B.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()
        var result_buffer = device_context.enqueue_create_buffer[DType.bool](
            output_size
        )

        ref A_buffer = A_device_state.device_buffer()
        ref B_buffer = B_device_state.device_buffer()

        var compiled_func = device_context.compile_function[
            compare[op_code, Self.dtype, simdwidth, 2 * simdwidth],
            compare[op_code, Self.dtype, simdwidth, 2 * simdwidth],
        ]()

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            B_buffer,
            0,  # A.offset(),
            0,  # B.offset(),
            output_size,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()
        var device_state = DeviceState[DType.bool](result_buffer^, gpu)
        var out = NDBuffer[DType.bool].with_device_state(
            device_state^, output_shape
        )

        return out^

    @staticmethod
    fn launch_config(output_size: Int) -> Tuple[Int, Int]:
        """Launch configuration."""
        var threads_per_block: Int
        var num_blocks: Int

        if output_size < 4096:
            threads_per_block = 128
            num_blocks = (output_size + 127) // 128
        elif output_size < 65536:
            threads_per_block = 256
            num_blocks = min((output_size + 255) // 256, 128)
        else:
            threads_per_block = 512
            num_blocks = min((output_size + 511) // 512, 512)

        return num_blocks, threads_per_block


fn compare_scalar[
    op_code: Int,
    dtype: DType,
    simd_width: Int = 1 if dtype == DType.bool else simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2
    * simd_width,  # Each thread processes twice simd size elements
](
    result: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    scalar: Scalar[dtype],
    size: UInt,
):
    """
    Element-wise scalar compare operations.

    - Each thread processes multiple items (better ILP)
    - SIMD vectorization within each item
    - Loop unrolling
    - Minimal divergence

    """

    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width

    # ===================================================================
    # Each thread processes CHUNK_SIZE elements
    # ===================================================================
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        # Process simd_vectors_per_thread vectors per thread
        @parameter
        for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            # Bounds check for this vector
            if i + simd_width <= size:
                # Full vector load
                var vec_a = A.load[width=simd_width](i)
                var vec_result: SIMD[DType.bool, simd_width]

                @parameter
                if op_code == Equal:
                    vec_result = vec_a.eq(scalar)
                elif op_code == NotEqual:
                    vec_result = vec_a.ne(scalar)
                elif op_code == GreaterThan:
                    vec_result = vec_a.gt(scalar)
                elif op_code == GreaterThanEqual:
                    vec_result = vec_a.ge(scalar)
                elif op_code == LessThan:
                    vec_result = vec_a.lt(scalar)
                else:  # LessThanEqual
                    vec_result = vec_a.le(scalar)

                # result.store[width=simd_width](i, vec_result)
                # Store results element-by-element (required for bool bit-packing)
                for idx in range(simd_width):
                    (result + i + idx)[] = vec_result[idx]

            elif i < size:
                # Partial vector (tail handling)
                for j in range(size - i):
                    var val = A[i + j]
                    var res: Scalar[DType.bool]

                    @parameter
                    if op_code == Equal:
                        res = val == scalar
                    elif op_code == NotEqual:
                        res = val != scalar
                    elif op_code == GreaterThan:
                        res = val > scalar
                    elif op_code == GreaterThanEqual:
                        res = val >= scalar
                    elif op_code == LessThan:
                        res = val < scalar
                    else:  # LessThanEqual
                        res = val <= scalar

                    result[i + j] = res

        base_idx += stride * CHUNK_SIZE


struct CompareScalar[dtype: DType = DType.float32](
    ImplicitlyCopyable & Movable
):
    @staticmethod
    fn launch[
        op_code: Int,
    ](A: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]) raises -> NDBuffer[
        DType.bool
    ]:
        var numels = A.numels()

        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(
            numels, simdwidth
        )
        ref A_device_state = A.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()

        var compiled_func = device_context.compile_function[
            compare_scalar[
                op_code=op_code,
                dtype = Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread = 2 * simdwidth,
            ],
            compare_scalar[
                op_code=op_code,
                dtype = Self.dtype,
                simd_width=simdwidth,
                simd_vectors_per_thread = 2 * simdwidth,
            ],
        ]()

        ref A_buffer = A_device_state.device_buffer()
        var result_buffer = device_context.enqueue_create_buffer[DType.bool](
            numels
        )

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            scalar,
            UInt(numels),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()
        var device_state = DeviceState[DType.bool](result_buffer^, gpu)
        var out = NDBuffer[DType.bool].with_device_state(device_state^, A.shape)

        return out^

    @staticmethod
    fn launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
        threads_per_block: Int
        num_blocks: Int

        if numels < 4096:
            threads_per_block = 128
            num_blocks = (numels + 127) // 128
        elif numels < 65536:
            threads_per_block = 256
            num_blocks = (numels + 255) // 256
        else:
            threads_per_block = 256
            var total_chunks = (numels + (simdwidth * 2 * simdwidth - 1)) // (
                simdwidth * 2 * simdwidth
            )
            num_blocks = min(
                (total_chunks + 255) // 256, 512
            )  # Cap at 512 blocks
        return threads_per_block, num_blocks


from tenmo import Tensor


fn main() raises:
    print("=" * 60)
    print("Production Tensor-Tensor Arithmetic Tests")
    print("With Offset Support")
    print("=" * 60)

    # Original tests
    test_to_gpu_and_back()
    test_contiguous_same_shape()
    test_non_contiguous()
    test_broadcasting()
    test_scalar_broadcast()
    test_complex_broadcasting()
    test_large_arrays()

    # Offset-specific tests
    test_contiguous_view_with_offset()
    test_all_offset_scenarios()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED (Including Offset Tests)")
    print("=" * 60)


from common_utils import now
from testing import assert_true
from shapes import Shape


fn test_to_gpu_and_back() raises:
    """Test cpu->gpu -> cpu."""

    comptime dtype = DType.float32
    var a = Tensor[dtype].ones(10)
    a_gpu = a.to_gpu()
    back_cpu = a_gpu.to_cpu()

    back_cpu.print()

    assert_true(back_cpu.all_close(a))
    print("  Passed")


fn test_contiguous_same_shape() raises:
    """Test fast path: contiguous, same shape."""
    print("=== Test 1: Contiguous Same Shape ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(1000, 10000)
    var b = Tensor[dtype].rand(1000, 10000)

    ag = a.to_gpu()
    bg = b.to_gpu()

    var start = now()
    var gpu_result = ag * bg
    var gpu_time = (now() - start) * 1000
    start = now()
    var cpu_result = a * b
    var cpu_time = (now() - start) * 1000

    assert_true(gpu_result.to_cpu().all_close(cpu_result))
    print("  GPU:", gpu_time, "ms")
    print("  CPU:", cpu_time, "ms")
    print("  Passed")


fn test_broadcasting() raises:
    """Test broadcasting path."""
    print("\n=== Test 2: Broadcasting ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(3, 1, 4)  # [3, 1, 4]
    var b = Tensor[dtype].rand(1, 2, 4)  # [1, 2, 4]

    ag = a.to_gpu()
    bg = b.to_gpu()

    var gpu_result = ag + bg
    var cpu_result = a + b
    assert_true(gpu_result.shape() == Shape(3, 2, 4))
    assert_true(gpu_result.to_cpu().all_close(cpu_result))
    cpu_ag = ag.to_cpu()
    cpu_bg = bg.to_cpu()

    assert_true(cpu_ag == a)
    assert_true(cpu_bg == b)
    print("  Shape:", gpu_result.shape())
    print("  Passed")


fn test_scalar_broadcast() raises:
    """Test scalar broadcasting."""
    print("\n=== Test 3: Scalar Broadcast ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(100, 100)
    var b = Tensor[dtype].ones(1, 1) * 42  # Broadcasts to [100, 100]

    ag = a.to_gpu()
    bg = b.to_gpu()

    var gpu_result = ag * bg
    var cpu_result = a * b
    cpu_ag = ag.to_cpu()
    cpu_bg = bg.to_cpu()

    assert_true(cpu_ag == a)
    assert_true(cpu_bg == b)

    assert_true(gpu_result.to_cpu().all_close(cpu_result))
    print("  Passed")


fn test_non_contiguous() raises:
    """Test non-contiguous tensors (views/transposes)."""
    print("\n=== Test 4: Non-Contiguous (Transpose) ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(3000, 2000)
    var b = Tensor[dtype].rand(2000, 3000)
    var a_t = a.transpose(1, 0)  # Non-contiguous view [2000, 3000]

    a_tg = a_t.to_gpu()
    bg = b.to_gpu()

    var gpu_result = a_tg + bg

    var cpu_result = a_t + b
    assert_true(gpu_result.to_cpu().all_close(cpu_result))
    cpu_ag = a_tg.to_cpu()
    cpu_bg = bg.to_cpu()

    assert_true(cpu_ag == a_t)
    assert_true(cpu_bg == b)

    print("  Passed")


fn test_complex_broadcasting() raises:
    """Test complex multi-dimensional broadcasting."""
    print("\n=== Test 5: Complex Broadcasting ===")

    comptime dtype = DType.float32
    var a = Tensor[dtype].rand(1, 5, 1, 7)  # [1, 5, 1, 7]
    var b = Tensor[dtype].rand(3, 1, 4, 1)  # [3, 1, 4, 1]

    ag = a.to_gpu()
    bg = b.to_gpu()

    var gpu_result = ag * bg

    var cpu_result = a * b

    assert_true(gpu_result.shape() == Shape(3, 5, 4, 7))
    assert_true(gpu_result.to_cpu().all_close(cpu_result))
    print("  Shape:", gpu_result.shape())
    print("  Passed")


fn test_large_arrays() raises:
    """Stress test with large arrays."""
    print("\n=== Test 6: Large Arrays ===")

    comptime dtype = DType.float32
    var size = 10_000_000  # 10M elements
    var a = Tensor[dtype].rand(size)
    var b = Tensor[dtype].rand(size)

    ag = a.to_gpu()
    bg = b.to_gpu()

    var start = now()
    var gpu_result = ag + bg
    var gpu_time = (now() - start) * 1000

    start = now()
    var cpu_result = a + b
    var cpu_time = (now() - start) * 1000

    assert_true(gpu_result.to_cpu().all_close(cpu_result))
    print("  Size:", size, "elements")
    print("  GPU:", gpu_time, "ms")
    print("  CPU:", cpu_time, "ms")
    print("  Speedup:", cpu_time / gpu_time, "x")
    print("  Passed")


fn test_contiguous_view_with_offset() raises:
    """Test contiguous view/slice with non-zero offset."""
    print("\n=== Test: Contiguous View with Offset ===")

    comptime dtype = DType.float32

    # Create larger tensors
    var a_full = Tensor[dtype].rand(1000)
    var b_full = Tensor[dtype].rand(1000)

    # Create contiguous slices with offsets
    # e.g., a_full[100:600] is contiguous but has offset 100
    var a = a_full[100:600]  # Contiguous, offset=100
    var b = b_full[200:700]  # Contiguous, offset=200

    # Verify they're contiguous
    assert_true(a.is_contiguous(), "A should be contiguous")
    assert_true(b.is_contiguous(), "B should be contiguous")
    assert_true(a.buffer.offset == 100, "A offset should be 100")
    assert_true(b.buffer.offset == 200, "B offset should be 200")

    ag = a.to_gpu()
    bg = b.to_gpu()

    # GPU computation
    var gpu_result = ag * bg

    # CPU reference
    var cpu_result = a * b

    # Verify
    assert_true(gpu_result.to_cpu().all_close(cpu_result))
    print("  A offset:", a.buffer.offset)
    print("  B offset:", b.buffer.offset)
    print("  Result shape:", gpu_result.shape())
    print("  Passed - Offsets handled correctly!")


fn test_all_offset_scenarios() raises:
    """Comprehensive offset testing."""
    print("\n=== Test: All Offset Scenarios ===")

    comptime dtype = DType.float32

    # Scenario 1: Both have offsets
    print("  Scenario 1: Both tensors have offsets")
    var a1 = Tensor[dtype].rand(1000)
    a1 = a1[100:600]
    var b1 = Tensor[dtype].rand(1000)
    b1 = b1[50:550]

    a1g = a1.to_gpu()
    b1g = b1.to_gpu()

    var result1 = a1g + b1g

    assert_true(result1.to_cpu().all_close(a1 + b1))
    print("    Passed")

    # Scenario 2: Only A has offset
    print("  Scenario 2: Only A has offset")
    var a2 = Tensor[dtype].rand(1000)
    a2 = a2[100:600]
    var b2 = Tensor[dtype].rand(500)  # No offset

    a2g = a2.to_gpu()
    b2g = b2.to_gpu()

    var result2 = a2g - b2g

    assert_true(result2.to_cpu().all_close(a2 - b2))
    print("    Passed")

    # Scenario 3: Only B has offset
    print("  Scenario 3: Only B has offset")
    var a3 = Tensor[dtype].rand(500)  # No offset
    var b3 = Tensor[dtype].rand(1000)
    b3 = b3[200:700]

    a3g = a3.to_gpu()
    b3g = b3.to_gpu()

    var result3 = a3g * b3g

    assert_true(result3.to_cpu().all_close(a3 * b3))
    print("    Passed")

    # Scenario 4: Neither has offset (original case)
    print("  Scenario 4: No offsets (baseline)")
    var a4 = Tensor[dtype].rand(500)
    var b4 = Tensor[dtype].rand(500)

    a4g = a4.to_gpu()
    b4g = b4.to_gpu()

    var result4 = a4g / b4g

    assert_true(result4.to_cpu().all_close(a4 / b4))
    print("    Passed")

    print("All offset scenarios passed!")


from testing import assert_true
from common_utils import now


fn main_1() raises:
    var SIZE = 65536 * 10
    comptime dtype = DType.float32
    var tensor_A = Tensor[dtype].ones(SIZE, requires_grad=True)
    var tensor_a = tensor_A.to_gpu()
    var start = now()
    # var expect = (tensor_A * 42) + 2
    var expect = tensor_A + 2
    print("CPU mul took: ", (now() - start) * 1000, "ms")
    # First test
    start = now()
    # var result = (tensor_a * 42) + 2
    var result = tensor_a + 2
    result = result.to_cpu()
    print("GPU mul took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))
    result.backward()
    tensor_A.grad().print()
    # Second test
    tensor_A = Tensor[dtype].rand(SIZE // 2, 2)
    var reshaped = tensor_A.reshape(2, SIZE // 2)
    start = now()
    expect = reshaped * 1919

    print("CPU mul took: ", (now() - start) * 1000, "ms")

    print("CPU mul took: ", (now() - start) * 1000, "ms")
    start = now()
    tensor_a = reshaped.to_gpu()
    result = tensor_a * 1919

    result = result.to_cpu()
    print("GPU mul took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))
    start = now()
    expect = reshaped / 89

    print("CPU div took: ", (now() - start) * 1000, "ms")
    start = now()
    result = tensor_a / 89

    result = result.to_cpu()
    print("GPU div took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))
    start = now()
    expect = reshaped - 999

    print("CPU subtract took: ", (now() - start) * 1000, "ms")
    start = now()
    result = tensor_a - 999

    result = result.to_cpu()
    print("GPU subtract took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))

    start = now()
    expect = 999 - reshaped

    print("CPU reverse subtract took: ", (now() - start) * 1000, "ms")
    start = now()
    result = 999 - tensor_a

    result = result.to_cpu()
    print("GPU reverse subtract took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))

    start = now()
    expect = 999 / reshaped

    print("CPU reverse divide took: ", (now() - start) * 1000, "ms")
    start = now()
    result = 999 / tensor_a

    result = result.to_cpu()
    print("GPU reverse divide took: ", (now() - start) * 1000, "ms")
    assert_true(result.all_close(expect))

    print("Launch success")
