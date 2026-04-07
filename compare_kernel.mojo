from std.sys import simd_width_of
from std.gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from std.os.atomic import Atomic, Consistency
from std.memory import AddressSpace, stack_allocation
from std.utils.numerics import isnan, isinf

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
    //,
    ordering: Consistency = Consistency.SEQUENTIAL,
](
    ptr: UnsafePointer[
        Scalar[DType.uint8], MutAnyOrigin, address_space=address_space
    ],
    mask: Scalar[DType.uint8],
) -> Scalar[DType.uint8]:
    var expected = ptr[]
    while True:
        var desired = expected & mask
        if Atomic.compare_exchange[
            failure_ordering=ordering,
            success_ordering=ordering,
        ](ptr, expected, desired):
            return expected


# ── all_close kernel — unchanged ──────────────────────────────────────────────


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
    comptime assert dtype.is_floating_point(), "all_close requires a float dtype"

    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    var block_result = stack_allocation[
        1, Scalar[DType.uint8], address_space = AddressSpace.SHARED
    ]()

    if thread_idx.x == 0:
        block_result[] = 1
    barrier()

    while base_idx < size:
        if block_result[] == 0:
            break

        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

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
                        else:
                            lane_ok = abs(a_val - b_val) <= atol + rtol * abs(
                                b_val
                            )

                        if not lane_ok:
                            _ = atomic_and(block_result, UInt8(0))
                            break
                else:
                    var diff = abs(va - vb)
                    var tolerance = atol + rtol * abs(vb)
                    if not diff.le(tolerance).reduce_and():
                        _ = atomic_and(block_result, UInt8(0))
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

    barrier()
    if thread_idx.x == 0:
        _ = atomic_and(result, block_result[])


@fieldwise_init
struct AllClose[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn launch[
        rtol: Scalar[Self.dtype] = 1e-5,
        atol: Scalar[Self.dtype] = 1e-8,
        treat_nan_equal: Bool = True,
        simd_width: Int = simd_width_of[Self.dtype](),
        simd_vectors_per_thread: Int = 2,
    ](A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]) raises -> Bool:
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (threads_per_block, num_blocks) = Self.launch_config(A.numels())

        ref A_device_state = A.device_state.value()
        ref B_device_state = B.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()
        var result_buffer = device_context.enqueue_create_buffer[DType.uint8](1)
        result_buffer.enqueue_fill(1)

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


# ── compare kernel ────────────────────────────────────────────────────────────
# Kernel output pointer is DType.uint8 — enqueue_create_buffer[DType.bool]
# is not supported by Mojo GPU runtime.
# DeviceState[DType.bool] internally uses DeviceBuffer[DType.uint8] via the
# bool→uint8 mapping in DeviceState, so we can safely wrap the uint8
# DeviceBuffer in DeviceState[DType.bool] and return GPU NDBuffer[DType.bool].


fn compare[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
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

        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i >= size:
                break

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](A_offset + i)
                var vec_b = B.load[width=simd_width](B_offset + i)
                var vec_result: SIMD[DType.bool, simd_width]

                comptime if op_code == Equal:
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

                # Write uint8 0/1 — element by element
                # bool bit-packing requires scalar writes
                for idx in range(simd_width):
                    (result + i + idx)[] = UInt8(1) if vec_result[
                        idx
                    ] else UInt8(0)

            else:
                for j in range(size - i):
                    var idx = i + j
                    var res: Scalar[DType.bool]

                    comptime if op_code == Equal:
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

                    (result + idx)[] = UInt8(1) if res else UInt8(0)

        base_idx += grid_stride * CHUNK_SIZE


@fieldwise_init
struct Compare[dtype: DType = DType.float32](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn launch[
        op_code: Int,
    ](A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]) raises -> NDBuffer[
        DType.bool
    ]:
        comptime simdwidth = simd_width_of[Self.dtype]()
        var output_shape = A.shape
        var output_size = output_shape.num_elements()

        var (threads_per_block, num_blocks) = Self.launch_config(output_size)

        ref A_device_state = A.device_state.value()
        ref B_device_state = B.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()

        # enqueue_create_buffer[DType.bool] not supported on GPU —
        # use uint8 instead. DeviceState[DType.bool] wraps uint8 internally.
        var result_buffer = device_context.enqueue_create_buffer[DType.uint8](
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
            0,
            0,
            output_size,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        # Wrap uint8 DeviceBuffer in DeviceState[DType.bool].
        # DeviceState[DType.bool].dtype == DType.uint8 internally —
        # so DeviceBuffer[DType.uint8] is a valid internal buffer for it.
        var device_state = DeviceState[DType.bool].__init__[True](
            result_buffer^, gpu
        )
        return NDBuffer[DType.bool].with_device_state(
            device_state^, output_shape
        )

    @staticmethod
    fn launch_config(output_size: Int) -> Tuple[Int, Int]:
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


# ── compare_scalar kernel ─────────────────────────────────────────────────────
# Same pattern as compare — uint8 output, wrapped in DeviceState[DType.bool]


fn compare_scalar[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin],
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    scalar: Scalar[dtype],
    size: Int,
):
    var tid = thread_idx.x
    var gtid = Int(tid + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:

        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.load[width=simd_width](i)
                var vec_result: SIMD[DType.bool, simd_width]

                comptime if op_code == Equal:
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

                # Write uint8 0/1 — element by element
                for idx in range(simd_width):
                    (result + i + idx)[] = UInt8(1) if vec_result[
                        idx
                    ] else UInt8(0)

            elif i < size:
                for j in range(size - i):
                    var val = A[i + j]
                    var res: Scalar[DType.bool]

                    comptime if op_code == Equal:
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

                    (result + i + j)[] = UInt8(1) if res else UInt8(0)

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

        # enqueue_create_buffer[DType.bool] not supported on GPU —
        # use uint8 instead. DeviceState[DType.bool] wraps uint8 internally.
        var result_buffer = device_context.enqueue_create_buffer[DType.uint8](
            numels
        )

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            scalar,
            numels,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        # Wrap uint8 DeviceBuffer in DeviceState[DType.bool].
        # DeviceState[DType.bool].dtype == DType.uint8 internally —
        # so DeviceBuffer[DType.uint8] is a valid internal buffer for it.
        var device_state = DeviceState[DType.bool].__init__[True](
            result_buffer^, gpu
        )
        return NDBuffer[DType.bool].with_device_state(device_state^, A.shape)

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


fn main() raises:
    print("passes")
