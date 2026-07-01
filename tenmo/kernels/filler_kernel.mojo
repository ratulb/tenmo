# =============================================================================
# filler_kernel.mojo — GPU fill and scatter-add kernels
# =============================================================================

from std.gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from std.atomic import Atomic
from tenmo.ndbuffer import NDBuffer
from tenmo.device import DeviceState
from tenmo.shapes import Shape
from tenmo.strides import Strides
from tenmo.intarray import IntArray
from tenmo.broadcasthelper import ShapeBroadcaster
from tenmo.indexhelper import IndexIterator
from tenmo.common_utils import panic
from std.sys import simd_width_of, has_accelerator
from .kernel_helpers import elementwise_launch_config


def fill_scalar_kernel[
    dtype: DType
](
    target: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    value: Scalar[dtype],
    size: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    var i = gtid
    while i < size:
        target[i] = value
        i += stride


def fill_from_buffer_kernel[
    dtype: DType
](
    target: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    source: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    target_offset: Int,
    source_offset: Int,
    size: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)
    var i = gtid
    while i < size:
        target[target_offset + i] = source[source_offset + i]
        i += stride


def scatter_add_rows_kernel[
    dtype: DType
](
    target: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    source: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    indices: UnsafePointer[Int32, ImmutAnyOrigin],
    n_indices: Int,
    row_width: Int,
):
    var row = Int(block_idx.x)
    var col = Int(thread_idx.x)

    if row >= n_indices or col >= row_width:
        return

    var target_row = Int(indices[row])
    var target_idx = target_row * row_width + col
    var source_idx = row * row_width + col

    _ = Atomic.fetch_add(target + target_idx, source[source_idx])


def scatter_add_rows_strided_kernel[
    dtype: DType
](
    target: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    source: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    indices: UnsafePointer[Int32, ImmutAnyOrigin],
    target_stride0: Int,
    target_stride1: Int,
    source_stride0: Int,
    source_stride1: Int,
    target_offset: Int,
    source_offset: Int,
    n_indices: Int,
    row_width: Int,
):
    var row = Int(block_idx.x)
    var col = Int(thread_idx.x)

    if row >= n_indices or col >= row_width:
        return

    var target_row = Int(indices[row])
    var target_idx = (
        target_offset + target_row * target_stride0 + col * target_stride1
    )
    var source_idx = source_offset + row * source_stride0 + col * source_stride1

    _ = Atomic.fetch_add(target + target_idx, source[source_idx])


def scatter_add_broadcast_kernel[
    dtype: DType
](
    target: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    source: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    indices: UnsafePointer[Int32, ImmutAnyOrigin],
    n_indices: Int,
    row_width: Int,
):
    var row = Int(block_idx.x)
    var col = Int(thread_idx.x)
    if row >= n_indices or col >= row_width:
        return
    var target_row = Int(indices[row])
    _ = Atomic.fetch_add(target + target_row * row_width + col, source[col])


def scatter_add_nd_kernel[
    dtype: DType
](
    target: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    source: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    indices: UnsafePointer[Int32, ImmutAnyOrigin],
    n_indices: Int,
    axis: Int,
    rank: Int,
    slice_volume: Int,
    target_shape: UnsafePointer[Int32, ImmutAnyOrigin],
    target_strides: UnsafePointer[Int32, ImmutAnyOrigin],
    source_strides: UnsafePointer[Int32, ImmutAnyOrigin],
    target_offset: Int,
    source_offset: Int,
    is_broadcast: Int,
):
    """N-dimensional scatter-add GPU kernel for any axis.

    Each block handles one index k. Each thread handles one element
    within the slice orthogonal to the given axis. Decomposes the
    flat element index into non-axis coordinates (same logic as CPU
    _scatter_add_cpu) to compute correct strided offsets for any rank.
    """
    var row = Int(block_idx.x)
    var col = Int(thread_idx.x)

    if row >= n_indices or col >= slice_volume:
        return

    var tgt_idx = Int(indices[row])
    var rem = col

    var dst_off = target_offset + tgt_idx * Int(target_strides[axis])
    var src_off = source_offset
    if not is_broadcast:
        src_off += row * Int(source_strides[axis])

    var d = rank - 1
    while d >= 0:
        if d != axis:
            var dim_size = Int(target_shape[d])
            var cd = rem % dim_size
            rem //= dim_size
            dst_off += cd * Int(target_strides[d])
            if not is_broadcast:
                src_off += cd * Int(source_strides[d])
        d -= 1

    if is_broadcast:
        src_off = source_offset + col

    _ = Atomic.fetch_add(target + dst_off, source[src_off])


@fieldwise_init
struct FillerGpu[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    @staticmethod
    def _fill_scalar_gpu(
        target: NDBuffer[Self.dtype],
        value: Scalar[Self.dtype],
        shape: Shape,
        strides: Strides,
        absolute_offset: Int,
        sync: Bool = False,
    ) raises:
        comptime if has_accelerator():
            ref device_state = target.device_state.value()
            ref gpu = device_state.get_gpu()
            var ctx = gpu[]
            var size = shape.num_elements()

            if strides.is_contiguous(shape):
                comptime simdwidth = simd_width_of[Self.dtype]()
                var (blocks, tpb) = elementwise_launch_config(size, simdwidth)
                var compiled = ctx.compile_function[
                    fill_scalar_kernel[Self.dtype],
                ]()
                ctx.enqueue_function(
                    compiled,
                    device_state.device_buffer(),
                    value,
                    size,
                    grid_dim=blocks,
                    block_dim=tpb,
                )
                if sync:
                    ctx.synchronize()
            else:
                var index_iterator = IndexIterator(
                    shape=Pointer(to=shape),
                    strides=Pointer(to=strides),
                    start_offset=absolute_offset,
                )
                for idx in index_iterator:
                    target.set(idx, value)

    @staticmethod
    def _fill_buffer_gpu(
        target: NDBuffer[Self.dtype],
        source: NDBuffer[Self.dtype],
        shape: Shape,
        strides: Strides,
        absolute_offset: Int,
        sync: Bool = False,
    ) raises:
        comptime if has_accelerator():
            ref t_state = target.device_state.value()
            ref gpu = t_state.get_gpu()
            var ctx = gpu[]
            var size = shape.num_elements()

            if (
                shape == source.shape
                and source.is_contiguous()
                and strides.is_contiguous(shape)
            ):
                comptime simdwidth = simd_width_of[Self.dtype]()
                var (blocks, tpb) = elementwise_launch_config(size, simdwidth)
                ref s_state = source.device_state.value()
                var compiled = ctx.compile_function[
                    fill_from_buffer_kernel[Self.dtype],
                ]()
                ctx.enqueue_function(
                    compiled,
                    t_state.device_buffer(),
                    s_state.device_buffer(),
                    absolute_offset,
                    source.offset,
                    size,
                    grid_dim=blocks,
                    block_dim=tpb,
                )
                if sync:
                    ctx.synchronize()
            else:
                if shape == source.shape:
                    var src_offset = source.offset
                    var dest_iter = IndexIterator(
                        shape=Pointer(to=shape),
                        strides=Pointer(to=strides),
                        start_offset=absolute_offset,
                    )
                    for dst_idx in dest_iter:
                        target.set(dst_idx, source.get(src_offset))
                        src_offset += 1
                else:
                    var mask = ShapeBroadcaster.broadcast_mask(
                        source.shape, shape
                    )
                    var index_iterator = IndexIterator(
                        shape=Pointer(to=shape),
                        strides=Pointer(to=strides),
                        start_offset=absolute_offset,
                    )
                    var coord_iterator = shape.__iter__()
                    for dst_idx in index_iterator:
                        try:
                            var coord = coord_iterator.__next__()
                            var source_coord = ShapeBroadcaster.translate_index(
                                source.shape, coord, mask, shape
                            )
                            target.set(dst_idx, source[source_coord])
                        except e:
                            print(e)
                            panic(
                                "Filler -> _fill_buffer_gpu: raised"
                                " StopIteration"
                            )

    @staticmethod
    def _scatter_add_gpu(
        target: NDBuffer[Self.dtype],
        source: NDBuffer[Self.dtype],
        indices: IntArray,
        n_indices: Int,
        row_width: Int,
        sync: Bool = False,
    ) raises:
        comptime if has_accelerator():
            ref t_state = target.device_state.value()
            ref s_state = source.device_state.value()
            ref gpu = t_state.get_gpu()
            var ctx = gpu[]

            var idx_buf = ctx.enqueue_create_buffer[DType.int32](n_indices)
            with idx_buf.map_to_host() as host_idx:
                for k in range(n_indices):
                    host_idx[k] = Int32(indices[k])

            var tpb = min(row_width, 512)
            var blocks = n_indices

            if source.shape.rank() == 1:
                var compiled = ctx.compile_function[
                    scatter_add_broadcast_kernel[Self.dtype],
                ]()
                ctx.enqueue_function(
                    compiled,
                    t_state.device_buffer(),
                    s_state.device_buffer(),
                    idx_buf,
                    n_indices,
                    row_width,
                    grid_dim=blocks,
                    block_dim=tpb,
                )
            else:
                var compiled = ctx.compile_function[
                    scatter_add_rows_kernel[Self.dtype],
                ]()
                ctx.enqueue_function(
                    compiled,
                    t_state.device_buffer(),
                    s_state.device_buffer(),
                    idx_buf,
                    n_indices,
                    row_width,
                    grid_dim=blocks,
                    block_dim=tpb,
                )

            if sync:
                ctx.synchronize()

    @staticmethod
    def _scatter_add_nd_gpu(
        target: NDBuffer[Self.dtype],
        source: NDBuffer[Self.dtype],
        indices: IntArray,
        n_indices: Int,
        slice_volume: Int,
        axis: Int,
        sync: Bool = False,
    ) raises:
        """N-dimensional scatter-add GPU dispatch for any axis.

        Copies indices, shape, and strides to GPU device buffers,
        then launches scatter_add_nd_kernel with one block per index.
        """
        comptime if has_accelerator():
            ref t_state = target.device_state.value()
            ref s_state = source.device_state.value()
            ref gpu = t_state.get_gpu()
            var ctx = gpu[]

            var idx_buf = ctx.enqueue_create_buffer[DType.int32](n_indices)
            with idx_buf.map_to_host() as host_idx:
                for k in range(n_indices):
                    host_idx[k] = Int32(indices[k])

            var rank = target.shape.rank()
            var shape_buf = ctx.enqueue_create_buffer[DType.int32](rank)
            var t_stride_buf = ctx.enqueue_create_buffer[DType.int32](rank)
            var s_stride_buf = ctx.enqueue_create_buffer[DType.int32](rank)

            with shape_buf.map_to_host() as h:
                for d in range(rank):
                    h[d] = Int32(target.shape[d])
            with t_stride_buf.map_to_host() as h:
                for d in range(rank):
                    h[d] = Int32(target.strides[d])
            with s_stride_buf.map_to_host() as h:
                for d in range(rank):
                    h[d] = Int32(source.strides[d])

            var tpb = min(slice_volume, 512)
            var blocks = n_indices
            var is_broadcast = Int(1 if source.shape.rank() == 1 else 0)

            var compiled = ctx.compile_function[
                scatter_add_nd_kernel[Self.dtype],
            ]()
            ctx.enqueue_function(
                compiled,
                t_state.device_buffer(),
                s_state.device_buffer(),
                idx_buf,
                n_indices,
                axis,
                rank,
                slice_volume,
                shape_buf,
                t_stride_buf,
                s_stride_buf,
                target.offset,
                source.offset,
                is_broadcast,
                grid_dim=blocks,
                block_dim=tpb,
            )

            if sync:
                ctx.synchronize()
