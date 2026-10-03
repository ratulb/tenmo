from std.sys import simd_width_of
from max.gpu import thread_idx, block_dim, grid_dim, block_idx

from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from .kernel_helpers import elementwise_launch_config
from ..gpu.device import DeviceState
from ..shared.panic import panic
from ..kernels.unary_device import (
    invert_bool,
    unary_ops,
    float_unary_ops,
    float_unary_ops_with_mask,
)
from ..shared.constants import Epsilon
from std.math import exp2
from ..shared.mnemonics import (
    LOG,
    EXP,
    SQRT,
    TANH_FORWARD,
    NEGATE,
    SIGMOID_FORWARD,
    RELU_FORWARD,
    GELU_FORWARD,
    INVERT,
    ROUND,
    FLOOR,
)


def unary_ops_with_mask[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    mask: Pointer[Scalar[dtype], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
):
    """Single-pass kernel: compute activation output AND gradient mask.

    Both result and mask are written in one GPU pass — no second kernel needed.

    For RELU_FORWARD:
        result[i] = max(A[i], 0)
        mask[i]   = 1.0 if A[i] > 0 else 0.0

    Args:
        result: Output buffer for activated values.
        mask:   Output buffer for gradient mask.
        A:      Input buffer (contiguous, same device).
        size_:  Total number of elements.

    Writes two output buffers in a single kernel pass:
      result  — the activated values  (ReLU: max(x, 0))
      mask    — the gradient gate     (ReLU: 1.0 if x > 0 else 0.0)
    No floating-point constraint — ReLU is safe for any dtype.
    """
    var size = Int(size_)
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    var zero_vec = SIMD[dtype, simd_width](0)
    var one_vec = SIMD[dtype, simd_width](1)
    var zero_s = Scalar[dtype](0)
    var one_s = Scalar[dtype](1)

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                # Full SIMD chunk
                var vec_a = A.unsafe_load[width=simd_width](i)

                var vec_result: SIMD[dtype, simd_width]
                var vec_mask: SIMD[dtype, simd_width]

                comptime if op_code == RELU_FORWARD:
                    vec_result = max(vec_a, zero_vec)
                    # mask = 1 where input > 0, else 0
                    vec_mask = vec_a.gt(zero_vec).select(one_vec, zero_vec)

                # Extend here for other ops that need a mask (e.g. leaky ReLU)
                else:
                    vec_result = vec_a  # identity fallback
                    vec_mask = one_vec

                result.unsafe_store[width=simd_width](i, vec_result)
                mask.unsafe_store[width=simd_width](i, vec_mask)

            elif i < size:
                # Scalar tail
                for j in range(size - i):
                    var val = A[unsafe_offset=i + j]
                    var res: Scalar[dtype]
                    var msk: Scalar[dtype]

                    comptime if op_code == RELU_FORWARD:
                        res = max(val, zero_s)
                        msk = one_s if val > zero_s else zero_s
                    else:
                        res = val
                        msk = one_s

                    result[unsafe_offset=i + j] = res
                    mask[unsafe_offset=i + j] = msk

        base_idx += stride * CHUNK_SIZE


# UnaryKernel launcher


struct UnaryKernel[dtype: DType](ImplicitlyCopyable):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def launch[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        comptime epsilon_for_float = rebind[Scalar[Self.dtype]](epsilon)
        comptime if op_code == INVERT:
            comptime assert (
                Self.dtype == DType.bool or Self.dtype.is_integral()
            ), "INVERT only valid for bool and integer types"
        var numels = A_layout.numel()
        comptime simdwidth = simd_width_of[Self.datatype]()
        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]
        var contig_state = materialize_contiguous(
            A_device_state, A_layout, sync=sync
        )
        var result_buffer = device_context.enqueue_create_buffer[Self.datatype](
            numels
        )
        comptime if op_code == LOG or op_code == EXP or op_code == TANH_FORWARD or op_code == SIGMOID_FORWARD or op_code == ROUND or op_code == FLOOR:
            comptime if Self.dtype.is_floating_point():
                var compiled = device_context.compile_function[
                    float_unary_ops[
                        op_code=op_code,
                        dtype=Self.dtype,
                        simd_width=simdwidth,
                        simd_vectors_per_thread=2 * simdwidth,
                        epsilon=epsilon_for_float,
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    result_buffer,
                    contig_state.device_buffer(),
                    Int64(numels),
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            else:
                panic(
                    "UnaryKernel: LOG/EXP/TANH/SIGMOID require a floating"
                    " point dtype"
                )
        elif op_code == INVERT and Self.dtype == DType.bool:
            var compiled = device_context.compile_function[
                invert_bool[simdwidth, 2 * simdwidth],
            ]()
            device_context.enqueue_function(
                compiled,
                result_buffer,
                contig_state.device_buffer(),
                Int64(numels),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        else:
            var compiled = device_context.compile_function[
                unary_ops[
                    op_code=op_code,
                    dtype=Self.datatype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
            ]()
            device_context.enqueue_function(
                compiled,
                result_buffer,
                contig_state.device_buffer(),
                Int64(numels),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        if sync:
            device_context.synchronize()

        # bool-as-uint8: skip sub-buffer creation (buffer dtype != Self.dtype when bool)
        var result_state = DeviceState[Self.dtype].__init__[True](
            result_buffer^, gpu
        )
        return (Layout(A_layout.shape), result_state^)

    # launch_with_mask()
    @staticmethod
    def launch_with_mask[
        op_code: Int,
    ](
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        sync: Bool = False,
    ) raises -> Tuple[
        Layout, DeviceState[Self.dtype], Layout, DeviceState[Self.dtype]
    ]:
        """Launch unary op + mask kernel. Returns (out layout, out storage,
        mask layout, mask storage) as GPU (Layout, Storage) pairs.

        Both buffers are written in a single GPU kernel pass.

        Non-contiguous input is handled via materialize_contiguous — which
        performs ONE map_to_host copy (not one per element), then the kernel
        operates on the resulting flat buffer.

        Args:
            A_layout:  Input layout. Must be on GPU.
            A_device_state: Input storage. Must be on GPU.
            sync: Whether to sync GPU after operation.

        Returns:
            Tuple of (out Layout, out Storage, mask Layout, mask Storage),
            both contiguous on GPU.
        """

        comptime if op_code == GELU_FORWARD:
            comptime assert (
                Self.dtype.is_floating_point()
            ), "GELU_FORWARD requires a floating point dtype"

        var numels = A_layout.numel()
        comptime simdwidth = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = Self.launch_config(
            numels, simdwidth
        )

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        # Non-contiguous: produce one contiguous GPU buffer in a single
        # map_to_host sweep — NOT one map_to_host call per index.
        var contig_state = materialize_contiguous(
            A_device_state, A_layout, sync=sync
        )

        # Allocate both output buffers on the same device
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )
        var mask_buffer = device_context.enqueue_create_buffer[Self.dtype](
            numels
        )
        comptime if op_code == GELU_FORWARD:
            comptime if Self.dtype.is_floating_point():
                var compiled = device_context.compile_function[
                    float_unary_ops_with_mask[
                        op_code=op_code,
                        dtype=Self.dtype,
                        simd_width=simdwidth,
                        simd_vectors_per_thread=2 * simdwidth,
                    ],
                ]()
                device_context.enqueue_function(
                    compiled,
                    result_buffer,
                    mask_buffer,
                    contig_state.device_buffer(),
                    Int64(numels),
                    grid_dim=num_blocks,
                    block_dim=threads_per_block,
                )
            else:
                panic(
                    "UnaryKernel.launch_with_mask: GELU_FORWARD requires a"
                    " floating point dtype"
                )
        else:
            var compiled = device_context.compile_function[
                unary_ops_with_mask[
                    op_code=op_code,
                    dtype=Self.dtype,
                    simd_width=simdwidth,
                    simd_vectors_per_thread=2 * simdwidth,
                ],
            ]()

            # Single kernel dispatch — writes result AND mask simultaneously
            device_context.enqueue_function(
                compiled,
                result_buffer,  # out: activated values
                mask_buffer,  # out: gradient mask
                contig_state.device_buffer(),  # in:  contiguous source
                Int64(numels),
                grid_dim=num_blocks,
                block_dim=threads_per_block,
            )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        var mask_state = DeviceState[Self.dtype](mask_buffer^, gpu)

        var out_layout = Layout(A_layout.shape)

        return (out_layout, result_state, out_layout, mask_state^)

    @staticmethod
    def launch_config(numels: Int, simdwidth: Int) -> Tuple[Int, Int]:
        return elementwise_launch_config(numels, simdwidth)
