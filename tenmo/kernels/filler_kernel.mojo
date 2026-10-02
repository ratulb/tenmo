# filler_kernel.mojo — GPU fill and scatter-add kernels

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from max.gpu import barrier
from std.atomic import Atomic
from std.sys import simd_width_of, has_accelerator
from ..shared.layout import Layout
from ..gpu.device import DeviceState
from ..shared.shapes import Shape
from ..shared.strides import Strides
from ..shared.intarray import IntArray
from ..shared.broadcasthelper import ShapeBroadcaster
from ..shared.indexhelper import IndexIterator
from ..shared.panic import panic
from .kernel_helpers import elementwise_launch_config


def fill_scalar_kernel[
    dtype: DType
](
    target: Pointer[Scalar[dtype], MutAnyOrigin],
    value: Scalar[dtype],
    size_: Int64,
):
    var size = Int(size_)
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    var i = gtid
    while i < size:
        target[unsafe_offset=i] = value
        i += stride


def fill_from_buffer_kernel[
    dtype: DType
](
    target: Pointer[Scalar[dtype], MutAnyOrigin],
    source: Pointer[Scalar[dtype], ImmutAnyOrigin],
    target_offset_: Int64,
    source_offset_: Int64,
    size_: Int64,
):
    var target_offset = Int(target_offset_)
    var source_offset = Int(source_offset_)
    var size = Int(size_)
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    var i = gtid
    while i < size:
        target[unsafe_offset=target_offset + i] = source[
            unsafe_offset=source_offset + i
        ]
        i += stride


def scatter_add_rows_kernel[
    dtype: DType
](
    target: Pointer[Scalar[dtype], MutAnyOrigin],
    source: Pointer[Scalar[dtype], ImmutAnyOrigin],
    indices: Pointer[Int32, ImmutAnyOrigin],
    n_indices_: Int64,
    row_width_: Int64,
):
    var n_indices = Int(n_indices_)
    var row_width = Int(row_width_)
    var row = block_idx.x
    var col = thread_idx.x

    if row >= n_indices or col >= row_width:
        return

    var target_row = Int(indices[unsafe_offset=row])
    var target_idx = target_row * row_width + col
    var source_idx = row * row_width + col

    _ = Atomic.fetch_add(
        target.unsafe_offset(target_idx), source[unsafe_offset=source_idx]
    )


def scatter_add_rows_strided_kernel[
    dtype: DType
](
    target: Pointer[Scalar[dtype], MutAnyOrigin],
    source: Pointer[Scalar[dtype], ImmutAnyOrigin],
    indices: Pointer[Int32, ImmutAnyOrigin],
    target_stride0_: Int64,
    target_stride1_: Int64,
    source_stride0_: Int64,
    source_stride1_: Int64,
    target_offset_: Int64,
    source_offset_: Int64,
    n_indices_: Int64,
    row_width_: Int64,
):
    var target_stride0 = Int(target_stride0_)
    var target_stride1 = Int(target_stride1_)
    var source_stride0 = Int(source_stride0_)
    var source_stride1 = Int(source_stride1_)
    var target_offset = Int(target_offset_)
    var source_offset = Int(source_offset_)
    var n_indices = Int(n_indices_)
    var row_width = Int(row_width_)
    var row = block_idx.x
    var col = thread_idx.x

    if row >= n_indices or col >= row_width:
        return

    var target_row = Int(indices[unsafe_offset=row])
    var target_idx = (
        target_offset + target_row * target_stride0 + col * target_stride1
    )
    var source_idx = source_offset + row * source_stride0 + col * source_stride1

    _ = Atomic.fetch_add(
        target.unsafe_offset(target_idx), source[unsafe_offset=source_idx]
    )


def scatter_add_broadcast_kernel[
    dtype: DType
](
    target: Pointer[Scalar[dtype], MutAnyOrigin],
    source: Pointer[Scalar[dtype], ImmutAnyOrigin],
    indices: Pointer[Int32, ImmutAnyOrigin],
    n_indices_: Int64,
    row_width_: Int64,
):
    var n_indices = Int(n_indices_)
    var row_width = Int(row_width_)
    var row = block_idx.x
    var col = thread_idx.x
    if row >= n_indices or col >= row_width:
        return
    var target_row = Int(indices[unsafe_offset=row])
    _ = Atomic.fetch_add(
        target.unsafe_offset(target_row * row_width + col),
        source[unsafe_offset=col],
    )


def scatter_add_nd_kernel[
    dtype: DType
](
    target: Pointer[Scalar[dtype], MutAnyOrigin],
    source: Pointer[Scalar[dtype], ImmutAnyOrigin],
    indices: Pointer[Int32, ImmutAnyOrigin],
    n_indices_: Int64,
    axis_: Int64,
    rank_: Int64,
    slice_volume_: Int64,
    target_shape: Pointer[Int32, ImmutAnyOrigin],
    target_strides: Pointer[Int32, ImmutAnyOrigin],
    source_strides: Pointer[Int32, ImmutAnyOrigin],
    target_offset_: Int64,
    source_offset_: Int64,
    is_broadcast_: Int64,
):
    """N-dimensional scatter-add GPU kernel for any axis.

    Each block handles one index k. Each thread handles one element
    within the slice orthogonal to the given axis. Decomposes the
    flat element index into non-axis coordinates (same logic as CPU
    _scatter_add_cpu) to compute correct strided offsets for any rank.
    """
    var n_indices = Int(n_indices_)
    var axis = Int(axis_)
    var rank = Int(rank_)
    var slice_volume = Int(slice_volume_)
    var target_offset = Int(target_offset_)
    var source_offset = Int(source_offset_)
    var is_broadcast = Int(is_broadcast_)
    var row = block_idx.x
    var col = thread_idx.x

    if row >= n_indices or col >= slice_volume:
        return

    var tgt_idx = Int(indices[unsafe_offset=row])
    var rem = col

    var dst_off = target_offset + tgt_idx * Int(
        target_strides[unsafe_offset=axis]
    )
    var src_off = source_offset
    if not is_broadcast:
        src_off += row * Int(source_strides[unsafe_offset=axis])

    var d = rank - 1
    while d >= 0:
        if d != axis:
            var dim_size = Int(target_shape[unsafe_offset=d])
            var cd = rem % dim_size
            rem //= dim_size
            dst_off += cd * Int(target_strides[unsafe_offset=d])
            if not is_broadcast:
                src_off += cd * Int(source_strides[unsafe_offset=d])
        d -= 1

    if is_broadcast:
        src_off = source_offset + col

    _ = Atomic.fetch_add(
        target.unsafe_offset(dst_off), source[unsafe_offset=src_off]
    )


@fieldwise_init
struct FillerKernel[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    # Internal storage dtype: bool → uint8, everything else → dtype
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def _fill_scalar_gpu(
        target_layout: Layout,
        target_device_state: DeviceState[Self.dtype],
        value: Scalar[Self.dtype],
        shape: Shape,
        strides: Strides,
        absolute_offset: Int,
        sync: Bool = False,
    ) raises:
        comptime if has_accelerator():
            ref device_state = target_device_state
            ref gpu = device_state.get_gpu()
            var ctx = gpu[]
            var size = shape.num_elements()

            if strides.is_contiguous(shape):
                comptime simdwidth = simd_width_of[Self.datatype]()
                var (blocks, tpb) = elementwise_launch_config(size, simdwidth)
                var compiled = ctx.compile_function[
                    fill_scalar_kernel[Self.datatype],
                ]()
                comptime if Self.dtype == DType.bool:
                    var storage_value = rebind[Scalar[Self.datatype]](
                        UInt8(1) if value.cast[DType.bool]() else UInt8(0)
                    )
                    ctx.enqueue_function(
                        compiled,
                        device_state.device_buffer(),
                        storage_value,
                        Int64(size),
                        grid_dim=blocks,
                        block_dim=tpb,
                    )
                else:
                    ctx.enqueue_function(
                        compiled,
                        device_state.device_buffer(),
                        value,
                        Int64(size),
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
                    device_state[idx] = value

    @staticmethod
    def _fill_buffer_gpu(
        target_layout: Layout,
        target_device_state: DeviceState[Self.dtype],
        source_layout: Layout,
        source_device_state: DeviceState[Self.dtype],
        shape: Shape,
        strides: Strides,
        absolute_offset: Int,
        sync: Bool = False,
    ) raises:
        comptime if has_accelerator():
            ref t_state = target_device_state
            ref s_state = source_device_state
            ref gpu = t_state.get_gpu()
            var ctx = gpu[]
            var size = shape.num_elements()

            if (
                shape == source_layout.shape
                and source_layout.is_contiguous()
                and strides.is_contiguous(shape)
            ):
                comptime simdwidth = simd_width_of[Self.datatype]()
                var (blocks, tpb) = elementwise_launch_config(size, simdwidth)
                var compiled = ctx.compile_function[
                    fill_from_buffer_kernel[Self.datatype],
                ]()
                ctx.enqueue_function(
                    compiled,
                    t_state.device_buffer(),
                    s_state.device_buffer(),
                    Int64(absolute_offset),
                    Int64(source_layout.offset),
                    Int64(size),
                    grid_dim=blocks,
                    block_dim=tpb,
                )
                if sync:
                    ctx.synchronize()
            else:
                if shape == source_layout.shape:
                    var src_offset = source_layout.offset
                    var dest_iter = IndexIterator(
                        shape=Pointer(to=shape),
                        strides=Pointer(to=strides),
                        start_offset=absolute_offset,
                    )
                    for dst_idx in dest_iter:
                        t_state[dst_idx] = s_state[src_offset]
                        src_offset += 1
                else:
                    var mask = ShapeBroadcaster.broadcast_mask(
                        source_layout.shape, shape
                    )
                    var index_iterator = IndexIterator(
                        shape=Pointer(to=shape),
                        strides=Pointer(to=strides),
                        start_offset=absolute_offset,
                    )
                    var coord_iterator = shape.__iter__()
                    for dst_idx in index_iterator:
                        var source_flat = 0
                        try:
                            var coord = coord_iterator.__next__()
                            var source_coord = ShapeBroadcaster.translate_index(
                                source_layout.shape, coord, mask, shape
                            )
                            source_flat = source_layout.offset
                            for d in range(source_layout.shape.rank()):
                                source_flat += (
                                    source_coord[d] * source_layout.strides[d]
                                )
                        except e:
                            print(e)
                            panic(
                                "Filler -> _fill_buffer_gpu: raised"
                                " StopIteration"
                            )
                        t_state[dst_idx] = s_state[source_flat]

    @staticmethod
    def _scatter_add_gpu(
        target_layout: Layout,
        target_device_state: DeviceState[Self.dtype],
        source_layout: Layout,
        source_device_state: DeviceState[Self.dtype],
        indices: IntArray,
        n_indices: Int,
        row_width: Int,
        sync: Bool = False,
    ) raises:
        comptime if has_accelerator():
            ref t_state = target_device_state
            ref s_state = source_device_state
            ref gpu = t_state.get_gpu()
            var ctx = gpu[]

            var idx_buf = ctx.enqueue_create_buffer[DType.int32](n_indices)
            with idx_buf.map_to_host() as host_idx:
                for k in range(n_indices):
                    host_idx[k] = Int32(indices[k])

            var tpb = min(row_width, 512)
            var blocks = n_indices

            if source_layout.shape.rank() == 1:
                var compiled = ctx.compile_function[
                    scatter_add_broadcast_kernel[Self.datatype],
                ]()
                ctx.enqueue_function(
                    compiled,
                    t_state.device_buffer(),
                    s_state.device_buffer(),
                    idx_buf,
                    Int64(n_indices),
                    Int64(row_width),
                    grid_dim=blocks,
                    block_dim=tpb,
                )
            else:
                var compiled = ctx.compile_function[
                    scatter_add_rows_kernel[Self.datatype],
                ]()
                ctx.enqueue_function(
                    compiled,
                    t_state.device_buffer(),
                    s_state.device_buffer(),
                    idx_buf,
                    Int64(n_indices),
                    Int64(row_width),
                    grid_dim=blocks,
                    block_dim=tpb,
                )

            if sync:
                ctx.synchronize()

    @staticmethod
    def _scatter_add_nd_gpu(
        target_layout: Layout,
        target_device_state: DeviceState[Self.dtype],
        source_layout: Layout,
        source_device_state: DeviceState[Self.dtype],
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
            ref t_state = target_device_state
            ref s_state = source_device_state
            ref gpu = t_state.get_gpu()
            var ctx = gpu[]

            var idx_buf = ctx.enqueue_create_buffer[DType.int32](n_indices)
            with idx_buf.map_to_host() as host_idx:
                for k in range(n_indices):
                    host_idx[k] = Int32(indices[k])

            var rank = target_layout.shape.rank()
            var shape_buf = ctx.enqueue_create_buffer[DType.int32](rank)
            var t_stride_buf = ctx.enqueue_create_buffer[DType.int32](rank)
            var s_stride_buf = ctx.enqueue_create_buffer[DType.int32](rank)

            with shape_buf.map_to_host() as h:
                for d in range(rank):
                    h[d] = Int32(target_layout.shape[d])
            with t_stride_buf.map_to_host() as h:
                for d in range(rank):
                    h[d] = Int32(target_layout.strides[d])
            with s_stride_buf.map_to_host() as h:
                for d in range(rank):
                    h[d] = Int32(source_layout.strides[d])

            var tpb = min(slice_volume, 512)
            var blocks = n_indices
            var is_broadcast = Int(
                1 if source_layout.shape.rank() == 1 else 0
            )

            var compiled = ctx.compile_function[
                scatter_add_nd_kernel[Self.datatype],
            ]()
            ctx.enqueue_function(
                compiled,
                t_state.device_buffer(),
                s_state.device_buffer(),
                idx_buf,
                Int64(n_indices),
                Int64(axis),
                Int64(rank),
                Int64(slice_volume),
                shape_buf,
                t_stride_buf,
                s_stride_buf,
                Int64(target_layout.offset),
                Int64(source_layout.offset),
                Int64(is_broadcast),
                grid_dim=blocks,
                block_dim=tpb,
            )

            if sync:
                ctx.synchronize()
