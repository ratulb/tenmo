# =============================================================================
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
# This handles any number of dimensions because Array (DevicePassable)
# carries the runtime size alongside the fixed-capacity storage.
# =============================================================================

from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import has_accelerator
from tenmo.ndbuffer import NDBuffer
from tenmo.device import GPU, DeviceState
from tenmo.shapes import Shape
from tenmo.array import Array
from tenmo.common_utils import panic
from .kernel_helpers import elementwise_launch_config


def pad_constant_kernel[
    dtype: DType,
    forward: Bool,
](
    src: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    numels: Int,
    ndim: Int,
    src_shape: Array,
    dst_strides: Array,
    pad_before: Array,
):
    """
    GPU kernel for constant padding forward/backward.

    Args:
        src:  Source buffer pointer (contiguous input or padded grad_output).
        dst:  Destination buffer pointer.
        numels: Number of elements in the source (the smaller tensor).
        ndim:  Number of tensor dimensions.
        src_shape:    Shape of the source tensor (contiguous, row-major).
        dst_strides:  Strides of the destination tensor.
        pad_before:   Before-padding for each dimension.
    """
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
            dst[out_flat] = src[flat]
        else:
            dst[flat] = src[out_flat]

        flat += gstride


@fieldwise_init
struct PadConstantGpuKernel[dtype: DType](
    ImplicitlyCopyable & Movable
):
    """
    GPU constant-padding kernel launcher.

    Provides ``launch_forward`` (pad) and ``launch_backward`` (unpad) for
    GPU-resident constant-mode padding and its backward pass.

    The source tensor is made contiguous before launching; its temporary
    DeviceBuffer is kept alive until the kernel synchronises.
    """

    @staticmethod
    def _launch[
        forward: Bool,
    ](
        src_ndb: NDBuffer[Self.dtype],
        dst_ndb: NDBuffer[Self.dtype],
        pad: List[Tuple[Int, Int]],
    ) raises -> None:
        """
        Internal launch helper.

        Makes src contiguous, enqueues the copy kernel, synchronises.
        dst must already contain the pad value in its padded regions.
        """
        debug_assert(src_ndb.is_on_gpu(), "PadConstantGpuKernel requires GPU src")
        debug_assert(dst_ndb.is_on_gpu(), "PadConstantGpuKernel requires GPU dst")

        var ndim = src_ndb.rank()
        var numels = src_ndb.numels()

        # Build Array objects for the GPU kernel
        var src_shape = Array()
        var dst_strides = Array()
        var pad_before = Array()
        for d in range(ndim):
            src_shape.append(src_ndb.shape[d])
            dst_strides.append(dst_ndb.strides[d])
            pad_before.append(pad[d][0])

        ref dst_state = dst_ndb.device_state.value()
        ref gpu = dst_state.get_gpu()
        var device_context = gpu[]

        comptime simdwidth = 1
        var (num_blocks, threads_per_block) = elementwise_launch_config(
            numels, simdwidth
        )

        # Materialise contiguous src
        var contig_src_state = src_ndb.contiguous_device_state()

        var compiled = device_context.compile_function[
            pad_constant_kernel[Self.dtype, forward],
            pad_constant_kernel[Self.dtype, forward],
        ]()

        device_context.enqueue_function(
            compiled,
            contig_src_state.device_buffer(),
            dst_state.device_buffer(),
            numels,
            ndim,
            src_shape,
            dst_strides,
            pad_before,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

    @staticmethod
    def launch_forward(
        src: NDBuffer[Self.dtype],
        dst: NDBuffer[Self.dtype],
        pad: List[Tuple[Int, Int]],
    ) raises -> None:
        """Forward pad: copy src elements into dst at padded positions.

        dst[out_flat] = src[flat]  for all flat in [0, src.numels())
        """
        PadConstantGpuKernel[Self.dtype]._launch[True](src, dst, pad)

    @staticmethod
    def launch_backward(
        grad_output: NDBuffer[Self.dtype],
        grad_parent: NDBuffer[Self.dtype],
        pad: List[Tuple[Int, Int]],
    ) raises -> None:
        """Backward unpad: extract center region from grad_output.

        grad_parent[flat] = grad_output[out_flat]  for all flat.
        """
        PadConstantGpuKernel[Self.dtype]._launch[False](grad_output, grad_parent, pad)
