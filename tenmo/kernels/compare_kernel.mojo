from std.sys import simd_width_of
from max.gpu import thread_idx, block_idx, block_dim, grid_dim
from max.gpu import barrier
from std.atomic import Atomic, Ordering
from std.memory import AddressSpace, stack_allocation
from std.utils.numerics import isnan, isinf
from ..shared.constants import CloseTol

from ..shared.mnemonics import (
    Equal,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
)
from ..gpu.device import DeviceState
from ..shared.layout import Layout
from .kernel_helpers import elementwise_launch_config


def atomic_and[
    address_space: AddressSpace,
    //,
    ordering: Ordering = Ordering.SEQUENTIAL,
](
    ptr: Pointer[
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


@always_inline
def _abs[
    dtype: DType,
    width: Int,
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    # |x| without the abs() builtin, which lowers to the llvm.nvvm.fabs
    # intrinsic that this toolchain's NVPTX backend cannot select
    # ("LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.fabs").
    # max(x, -x) is equivalent: all_close handles NaN/Inf lanes separately,
    # so abs()'s NaN propagation is never needed in these paths.
    return max(x, -x)


@always_inline
def _abs[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    return max(x, -x)


def all_close[
    dtype: DType,
    rtol: Scalar[dtype] = CloseTol[dtype].rtol(),
    atol: Scalar[dtype] = CloseTol[dtype].atol(),
    treat_nan_equal: Bool = True,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2,
](
    result: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    B: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
):
    var size = Int(size_)
    comptime assert (
        dtype.is_floating_point()
    ), "all_close requires a float dtype"

    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var grid_stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    var block_result = stack_allocation[
        1, Scalar[DType.uint8], address_space=AddressSpace.SHARED
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
                var va = A.unsafe_load[width=simd_width](i)
                var vb = B.unsafe_load[width=simd_width](i)

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
                            lane_ok = _abs(a_val - b_val) <= atol + rtol * _abs(
                                b_val
                            )

                        if not lane_ok:
                            _ = atomic_and(
                                block_result.as_unsafe_any_origin(), UInt8(0)
                            )
                            break
                else:
                    var diff = _abs(va - vb)
                    var tolerance = atol + rtol * _abs(vb)
                    if not diff.le(tolerance).reduce_and():
                        _ = atomic_and(
                            block_result.as_unsafe_any_origin(), UInt8(0)
                        )
            else:
                for j in range(size - i):
                    var idx = i + j
                    var a_val = (A.unsafe_offset(idx))[]
                    var b_val = (B.unsafe_offset(idx))[]
                    var local_ok: Bool

                    if isnan(a_val) or isnan(b_val):
                        local_ok = (
                            treat_nan_equal and isnan(a_val) and isnan(b_val)
                        )
                    elif isinf(a_val) or isinf(b_val):
                        local_ok = a_val == b_val
                    else:
                        local_ok = _abs(a_val - b_val) <= atol + rtol * _abs(
                            b_val
                        )

                    if not local_ok:
                        _ = atomic_and(
                            block_result.as_unsafe_any_origin(), UInt8(0)
                        )
                        break

        base_idx += grid_stride * CHUNK_SIZE

    barrier()
    if thread_idx.x == 0:
        _ = atomic_and(result, block_result[])


@fieldwise_init
struct AllClose[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def launch[
        rtol: Scalar[Self.dtype] = CloseTol[Self.dtype].rtol(),
        atol: Scalar[Self.dtype] = CloseTol[Self.dtype].atol(),
        treat_nan_equal: Bool = True,
        simd_width: Int = simd_width_of[Self.dtype](),
        simd_vectors_per_thread: Int = 2,
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        B_layout: Layout,
        B_device_state: DeviceState[Self.dtype],
        sync: Bool = False,
    ) raises -> Bool:
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            A_layout.numel(), simdwidth
        )

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]
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
        ]()

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            B_buffer,
            Int64(A_layout.numel()),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()
        var all_close_result: Bool
        with result_buffer.map_to_host() as host_buffer:
            all_close_result = True if host_buffer[0] == 1 else False
        return all_close_result

    @staticmethod
    def launch_config(output_size: Int, simdwidth: Int) -> Tuple[Int, Int]:
        return elementwise_launch_config(output_size, simdwidth)


def compare[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    B: Pointer[Scalar[dtype], ImmutAnyOrigin],
    A_offset_: Int64,
    B_offset_: Int64,
    size_: Int64,
):
    """Compare kernel.
    Kernel output pointer is DType.uint8 — enqueue_create_buffer[DType.bool]
    is not supported by Mojo GPU runtime.
    DeviceState[DType.bool] internally uses DeviceBuffer[DType.uint8] via the
    bool→uint8 mapping in DeviceState, so we can safely wrap the uint8
    DeviceBuffer in DeviceState[DType.bool] and return GPU NDBuffer[DType.bool].
    """
    var A_offset = Int(A_offset_)
    var B_offset = Int(B_offset_)
    var size = Int(size_)
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
                var vec_a = A.unsafe_load[width=simd_width](A_offset + i)
                var vec_b = B.unsafe_load[width=simd_width](B_offset + i)
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
                    (result.unsafe_offset(i + idx))[] = UInt8(1) if vec_result[
                        idx
                    ] else UInt8(0)

            else:
                for j in range(size - i):
                    var idx = i + j
                    var res: Scalar[DType.bool]

                    comptime if op_code == Equal:
                        res = (
                            A[unsafe_offset=A_offset + idx]
                            == B[unsafe_offset=B_offset + idx]
                        )
                    elif op_code == NotEqual:
                        res = (
                            A[unsafe_offset=A_offset + idx]
                            != B[unsafe_offset=B_offset + idx]
                        )
                    elif op_code == GreaterThan:
                        res = (
                            A[unsafe_offset=A_offset + idx]
                            > B[unsafe_offset=B_offset + idx]
                        )
                    elif op_code == GreaterThanEqual:
                        res = (
                            A[unsafe_offset=A_offset + idx]
                            >= B[unsafe_offset=B_offset + idx]
                        )
                    elif op_code == LessThan:
                        res = (
                            A[unsafe_offset=A_offset + idx]
                            < B[unsafe_offset=B_offset + idx]
                        )
                    else:  # LessThanEqual
                        res = (
                            A[unsafe_offset=A_offset + idx]
                            <= B[unsafe_offset=B_offset + idx]
                        )

                    (result.unsafe_offset(idx))[] = UInt8(1) if res else UInt8(
                        0
                    )

        base_idx += grid_stride * CHUNK_SIZE


@fieldwise_init
struct Compare[dtype: DType = DType.float32](
    ImplicitlyCopyable, RegisterPassable
):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def launch[
        op_code: Int,
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        B_layout: Layout,
        B_device_state: DeviceState[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[DType.bool]]:
        comptime simdwidth = simd_width_of[Self.datatype]()
        var output_shape = A_layout.shape
        var output_size = output_shape.num_elements()

        var (num_blocks, threads_per_block) = Self.launch_config(
            output_size, simdwidth
        )

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        var result_buffer = device_context.enqueue_create_buffer[DType.uint8](
            output_size
        )

        ref A_buffer = A_device_state.device_buffer()
        ref B_buffer = B_device_state.device_buffer()

        var compiled_func = device_context.compile_function[
            compare[op_code, Self.datatype, simdwidth, 2 * simdwidth],
        ]()

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buffer,
            B_buffer,
            Int64(0),
            Int64(0),
            Int64(output_size),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        # bool-as-uint8: skip sub-buffer creation (buffer dtype != Self.dtype when bool)
        var device_state = DeviceState[DType.bool].__init__[True](
            result_buffer^, gpu
        )
        return (Layout(output_shape), device_state^)

    @staticmethod
    def launch_config(output_size: Int, simdwidth: Int) -> Tuple[Int, Int]:
        return elementwise_launch_config(output_size, simdwidth)


def compare_scalar[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    scalar: Scalar[dtype],
    size_: Int64,
):
    """compare_scalar kernel
    Same pattern as compare — uint8 output, wrapped in DeviceState[DType.bool]
    """
    var size = Int(size_)
    var tid = thread_idx.x
    var gtid = Int(tid + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.unsafe_load[width=simd_width](i)
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
                    (result.unsafe_offset(i + idx))[] = UInt8(1) if vec_result[
                        idx
                    ] else UInt8(0)

            elif i < size:
                for j in range(size - i):
                    var val = A[unsafe_offset=i + j]
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

                    (result.unsafe_offset(i + j))[] = UInt8(
                        1
                    ) if res else UInt8(0)

        base_idx += stride * CHUNK_SIZE


struct CompareScalar[dtype: DType = DType.float32](ImplicitlyCopyable):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def launch[
        op_code: Int,
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        scalar: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[DType.bool]]:
        var numels = A_layout.numel()
        comptime simdwidth = simd_width_of[Self.datatype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        var compiled_func = device_context.compile_function[
            compare_scalar[
                op_code=op_code,
                dtype=Self.datatype,
                simd_width=simdwidth,
                simd_vectors_per_thread=2 * simdwidth,
            ],
        ]()

        ref A_buffer = A_device_state.device_buffer()

        var result_buffer = device_context.enqueue_create_buffer[DType.uint8](
            numels
        )

        comptime if Self.dtype == DType.bool:
            var storage_scalar = rebind[Scalar[Self.datatype]](
                UInt8(1) if scalar.cast[DType.bool]() else UInt8(0)
            )
            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                storage_scalar,
                Int64(numels),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )
        else:
            device_context.enqueue_function(
                compiled_func,
                result_buffer,
                A_buffer,
                scalar,
                Int64(numels),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        if sync:
            device_context.synchronize()

        # bool-as-uint8: skip sub-buffer creation (buffer dtype != Self.dtype when bool)
        var device_state = DeviceState[DType.bool].__init__[True](
            result_buffer^, gpu
        )
        return (Layout(A_layout.shape), device_state^)

    @staticmethod
    def launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
        return elementwise_launch_config(numels, simdwidth)
