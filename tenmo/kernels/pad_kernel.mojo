# pad_kernel.mojo — GPU constant-padding kernel
#
# Both forward and backward use the same kernel body. A comptime bool
# `forward` controls the copy direction:
#
#   forward=True  (pad):   dst[out_flat] = src[flat]
#   forward=False (unpad): dst[flat]     = src[out_flat]
#
# where:
#
#   flat     = linear index in the src (contiguous) buffer
#   out_flat = flat index in the dst buffer after applying the padding offset
#
# Coordinate decomposition of flat uses src_shape (row-major).  Coordinate
# reconstruction uses dst_strides and the per-dimension before-padding
# amounts stored in `pad_before`.
#
#   out_flat = Σ (coord[d] + pad_before[d]) × dst_stride[d]
#
# This handles any number of dimensions because RankArray (DevicePassable)
# carries the runtime size alongside the fixed-capacity storage.

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from ..gpu.device import GPU, DeviceState
from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from ..shared.shapes import Shape
from ..shared.array import RankArray
from ..shared.panic import panic
from .kernel_helpers import elementwise_launch_config


def pad_constant_kernel[
    dtype: DType,
    forward: Bool,
](
    src: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    numels_: Int64,
    ndim_: Int64,
    src_shape: RankArray,
    dst_strides: RankArray,
    pad_before: RankArray,
):
    """
        GPU kernel for constant padding forward/backward.

    Args:
        src:  Source buffer pointer (contiguous input or padded grad_output).
        dst:  Destination buffer pointer.
        numels_: Number of elements in the source (the smaller tensor).
        ndim_:  Number of tensor dimensions.
        src_shape:    Shape of the source tensor (contiguous, row-major).
        dst_strides:  Strides of the destination tensor.
        pad_before:   Before-padding for each dimension.
    """
    var numels = Int(numels_)
    var ndim = Int(ndim_)
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var gstride = Int(block_dim.x * grid_dim.x)

    var flat = gtid
    while flat < numels:
        var remaining = flat
        var out_flat = 0
        for d in range(ndim - 1, -1, -1):
            var coord = remaining % src_shape[d]
            remaining //= src_shape[d]
            out_flat += (coord + pad_before[d]) * dst_strides[d]

        comptime if forward:
            dst[unsafe_offset=out_flat] = src[unsafe_offset=flat]
        else:
            dst[unsafe_offset=flat] = src[unsafe_offset=out_flat]

        flat += gstride


@fieldwise_init
struct PadKernel[dtype: DType](ImplicitlyCopyable):
    """
        GPU constant-padding kernel launcher.

    Provides `launch_forward` (pad) and `launch_backward` (unpad) for
    GPU-resident constant-mode padding and its backward pass.

    The source tensor is made contiguous before launching; its temporary
    DeviceBuffer is kept alive until the kernel synchronises.
    """

    @staticmethod
    def _launch[
        forward: Bool,
    ](
        src_layout: Layout,
        src_device_state: DeviceState[Self.dtype],
        dst_layout: Layout,
        dst_device_state: DeviceState[Self.dtype],
        pad: List[Tuple[Int, Int]],
        sync: Bool = False,
    ) raises -> None:
        """
                Internal launch helper.

        Makes src contiguous, enqueues the copy kernel, synchronises
        (when `sync` is True).
        dst must already contain the pad value in its padded regions.
        """
        var ndim = src_layout.rank()
        var numels: Int
        var src_shape = RankArray()
        var dst_strides = RankArray()
        var pad_before = RankArray()

        comptime if forward:
            numels = src_layout.numel()
            for d in range(ndim):
                src_shape.append(src_layout.shape[d])
                dst_strides.append(dst_layout.strides[d])
                pad_before.append(pad[d][0])
        else:
            numels = dst_layout.numel()
            for d in range(ndim):
                src_shape.append(dst_layout.shape[d])
                dst_strides.append(src_layout.strides[d])
                pad_before.append(pad[d][0])

        ref dst_state = dst_device_state
        ref gpu = dst_state.get_gpu()
        var device_context = gpu[]

        comptime simdwidth = 1
        var (num_blocks, threads_per_block) = elementwise_launch_config(
            numels, simdwidth
        )

        # Materialise contiguous src
        var contig_src_state = materialize_contiguous(
            src_device_state, src_layout
        )

        var compiled = device_context.compile_function[
            pad_constant_kernel[Self.dtype, forward],
        ]()

        device_context.enqueue_function(
            compiled,
            contig_src_state.device_buffer(),
            dst_state.device_buffer(),
            Int64(numels),
            Int64(ndim),
            src_shape,
            dst_strides,
            pad_before,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

    @staticmethod
    def launch_forward(
        src_layout: Layout,
        src_device_state: DeviceState[Self.dtype],
        dst_layout: Layout,
        dst_device_state: DeviceState[Self.dtype],
        pad: List[Tuple[Int, Int]],
        sync: Bool = False,
    ) raises -> None:
        """Forward pad: copy src elements into dst at padded positions.

        dst[out_flat] = src[flat]  for all flat in [0, src.numels())
        """
        PadKernel[Self.dtype]._launch[True](
            src_layout,
            src_device_state,
            dst_layout,
            dst_device_state,
            pad,
            sync=sync,
        )

    @staticmethod
    def launch_backward(
        grad_output_layout: Layout,
        grad_output_device_state: DeviceState[Self.dtype],
        grad_parent_layout: Layout,
        grad_parent_device_state: DeviceState[Self.dtype],
        pad: List[Tuple[Int, Int]],
        sync: Bool = False,
    ) raises -> None:
        """Backward unpad: extract center region from grad_output.

        grad_parent[flat] = grad_output[out_flat]  for all flat.
        """
        PadKernel[Self.dtype]._launch[False](
            grad_output_layout,
            grad_output_device_state,
            grad_parent_layout,
            grad_parent_device_state,
            pad,
            sync=sync,
        )
