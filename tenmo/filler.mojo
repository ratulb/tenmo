from tenmo.tensor import Tensor
from tenmo.common_utils import Idx, panic
from tenmo.validators import Validator
from tenmo.broadcasthelper import ShapeBroadcaster
from tenmo.indexhelper import IndexIterator
from tenmo.ndbuffer import NDBuffer
from tenmo.device import DeviceState
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from tenmo.strides import Strides
from std.memory import memcpy, AddressSpace, stack_allocation
from std.gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from std.os.atomic import Atomic
from std.sys import has_accelerator, simd_width_of


# =============================================================================
# GPU kernels
# =============================================================================


def fill_scalar_kernel[
    dtype: DType
](
    target: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    value: Scalar[dtype],
    size: Int,
):
    """Fill contiguous target buffer with a scalar value.
    Grid-stride loop — each thread handles multiple elements.
    """
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
    """Copy contiguous source into contiguous target region.
    Grid-stride loop.
    """
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
    """Scatter-add rows from source into target using indices.

    Forward:  out[k]               = target[indices[k], :]   (gather)
    Backward: target[indices[k], :] += source[k, :]          (scatter-add)

    Grid  : (n_indices,) — one block per gathered row
    Block : (min(row_width, 512),) — one thread per column

    Atomic add handles repeated indices (same token appearing
    multiple times in a review) without data races.
    """
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
    """Stride-aware scatter-add for non-contiguous tensors.

    Uses explicit strides so both source and target can have
    arbitrary memory layouts. Atomic add for correctness on
    repeated indices.
    """
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


# =============================================================================
# GPU launcher helpers
# =============================================================================


def _gpu_launch_config(size: Int) -> Tuple[Int, Int]:
    """Returns (threads_per_block, num_blocks) for a flat kernel over size elements.
    """
    var tpb = 256 if size >= 256 else size
    var blocks = (size + tpb - 1) // tpb
    return (tpb, blocks)


# =============================================================================
# Filler
# =============================================================================


@fieldwise_init
struct Filler[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    """Element-wise fill and copy for NDBuffer — CPU and GPU capable.

    CPU path: fast memcpy for contiguous cases, strided iterator otherwise.
    GPU path: dedicated kernels for contiguous cases;
              scatter_add_rows(_strided) for ScatterAddTensor backward.

    All public entry points dispatch on device automatically — callers
    do not need to check is_on_gpu().
    """

    # =========================================================================
    # Public API — scalar fill
    # =========================================================================

    @always_inline
    @staticmethod
    def fill(
        target: NDBuffer[Self.dtype],
        value: Scalar[Self.dtype],
        indices: VariadicList[Idx, _],
    ):
        try:
            var (
                shape,
                strides,
                offset,
            ) = Validator.validate_and_compute_advanced_indexing_metadata(
                target.shape, target.strides, indices
            )
            var absolute_offset = target.offset + offset

            comptime if has_accelerator():
                if target.is_on_gpu():
                    Self._fill_scalar_gpu(
                        target, value, shape, strides, absolute_offset
                    )
                    return
            Self._fill_scalar_cpu(
                target, value, shape, strides, absolute_offset
            )
        except e:
            print(e)
            panic("Filler fill(scalar) error")

    # =========================================================================
    # Public API — buffer-to-buffer fill (general)
    # =========================================================================

    @always_inline
    @staticmethod
    def fill(
        target: NDBuffer[Self.dtype],
        source: NDBuffer[Self.dtype],
        indices: VariadicList[Idx, _],
    ):
        try:
            var (
                shape,
                strides,
                offset,
            ) = Validator.validate_and_compute_advanced_indexing_metadata(
                target.shape, target.strides, indices
            )
            ref source_shape = source.shape
            if not ShapeBroadcaster.broadcastable(source_shape, shape):
                panic(
                    "Filler → fill: input buffer not broadcastable to shape",
                    shape.__str__(),
                )
            var absolute_offset = target.offset + offset

            comptime if has_accelerator():
                if target.is_on_gpu():
                    Self._fill_buffer_gpu(
                        target, source, shape, strides, absolute_offset
                    )
                    return
            Self._fill_buffer_cpu(
                target, source, shape, strides, absolute_offset
            )
        except e:
            print(e)
            panic("Filler fill(scalar) error")

    # =========================================================================
    # Public API — scatter-add rows (GatherBackward / ScatterAddTensor==op_code)
    # =========================================================================

    @always_inline
    @staticmethod
    def scatter_add(
        target: NDBuffer[Self.dtype],  # (vocab_size, hidden_size) gradbox
        source: NDBuffer[Self.dtype],  # (n_indices,  hidden_size) incoming grad
        indices: IntArray,  # which rows of target to scatter into
        axis: Int = 0,
    ):
        """Scatter-add source rows into target rows at the given indices.

        target[indices[k], :] += source[k, :]  for k in range(n_indices)

        Uses atomic add — safe for repeated indices (same token appearing
        multiple times in a review).

        Dispatches to GPU kernel when target is on GPU, CPU loop otherwise.
        """
        try:
            var n_indices = len(indices)
            var row_width = (
                source.shape[1] if source.shape.rank() > 1 else source.numels()
            )

            comptime if has_accelerator():
                if target.is_on_gpu():
                    Self._scatter_add_gpu(
                        target, source, indices, n_indices, row_width
                    )
                    return
            Self._scatter_add_cpu(target, source, indices, n_indices, row_width)
        except e:
            print(e)
            panic("Error in Filler scatter_add")

    # =========================================================================
    # CPU implementations
    # =========================================================================

    @staticmethod
    def _fill_scalar_cpu(
        target: NDBuffer[Self.dtype],
        value: Scalar[Self.dtype],
        shape: Shape,
        strides: Strides,
        absolute_offset: Int,
    ):
        if strides.is_contiguous(shape):
            # Fast path — sequential write
            ref buffer = target.data_buffer()
            var end = absolute_offset + shape.num_elements()
            for idx in range(absolute_offset, end):
                buffer[idx] = value
        else:
            ref buffer = target.data_buffer()
            var index_iterator = IndexIterator(
                shape=Pointer(to=shape),
                strides=Pointer(to=strides),
                start_offset=absolute_offset,
            )
            for idx in index_iterator:
                buffer[idx] = value

    @staticmethod
    def _fill_buffer_cpu(
        target: NDBuffer[Self.dtype],
        source: NDBuffer[Self.dtype],
        shape: Shape,
        strides: Strides,
        absolute_offset: Int,
    ):
        ref source_shape = source.shape

        if shape == source_shape:
            if source.is_contiguous() and strides.is_contiguous(shape):
                # Both contiguous — memcpy fast path
                dest = (
                    target.data_ptr()
                    .unsafe_mut_cast[True]()
                    .unsafe_origin_cast[MutAnyOrigin]()
                )
                src = source.data_ptr()
                memcpy(
                    dest=dest + absolute_offset,
                    src=src + source.offset,
                    count=shape.num_elements(),
                )

            elif source.is_contiguous() and not strides.is_contiguous(shape):
                # Source contiguous, target strided
                var src_offset = source.offset
                var index_iterator = IndexIterator(
                    shape=Pointer(to=shape),
                    strides=Pointer(to=strides),
                    start_offset=absolute_offset,
                )
                ref src_buf = source.data_buffer()
                ref dest_buf = target.data_buffer()
                for dst_idx in index_iterator:
                    dest_buf[dst_idx] = src_buf[src_offset]
                    src_offset += 1

            elif not source.is_contiguous() and strides.is_contiguous(shape):
                # Source strided, target contiguous
                ref src_buf = source.data_buffer()
                ref dest_buf = target.data_buffer()
                var dst_offset = absolute_offset
                for src_idx in source.index_iterator():
                    dest_buf[dst_offset] = src_buf[src_idx]
                    dst_offset += 1

            else:
                # Both strided
                ref src_buf = source.data_buffer()
                ref dest_buf = target.data_buffer()
                var src_iter = source.index_iterator()
                var dest_iter = IndexIterator(
                    shape=Pointer(to=shape),
                    strides=Pointer(to=strides),
                    start_offset=absolute_offset,
                )
                while src_iter.__has_next__():
                    var src_idx = -1
                    var dst_idx = -1
                    try:
                        src_idx = src_iter.__next__()
                        dst_idx = dest_iter.__next__()
                    except e:
                        print(e)
                        panic("Raised StopIteration in Filler -> fill")
                    dest_buf[dst_idx] = src_buf[src_idx]

        else:
            # Shapes differ but broadcastable
            var broadcast_shape = ShapeBroadcaster.broadcast_shape[
                validated=True
            ](source_shape, shape)
            if broadcast_shape != shape:
                panic(
                    "Filler → fill: broadcast shape",
                    broadcast_shape.__str__(),
                    "is not equal to selected slice shape",
                    shape.__str__(),
                )
            var mask = ShapeBroadcaster.broadcast_mask(source_shape, shape)
            var index_iterator = IndexIterator(
                shape=Pointer(to=shape),
                strides=Pointer(to=strides),
                start_offset=absolute_offset,
            )
            var coord_iterator = shape.__iter__()
            ref dest_buf = target.data_buffer()
            for dst_idx in index_iterator:
                try:
                    var coord = coord_iterator.__next__()
                    var source_coord = ShapeBroadcaster.translate_index(
                        source_shape, coord, mask, shape
                    )
                    dest_buf[dst_idx] = source[source_coord]
                except e:
                    print(e)
                    panic("Filler -> fill: raised StopIteration error")

    @staticmethod
    def _scatter_add_cpu(
        target: NDBuffer[Self.dtype],
        source: NDBuffer[Self.dtype],
        indices: IntArray,
        n_indices: Int,
        row_width: Int,
    ):
        """CPU scatter-add — row-by-row loop, no atomics needed (single thread).
        """
        for k in range(n_indices):
            var target_row = indices[k]
            for col in range(row_width):
                var src_val = source.get(source.offset + k * row_width + col)
                var dst_idx = target.offset + target_row * row_width + col
                var dst_val = target.get(dst_idx)
                target.set(dst_idx, dst_val + src_val)

    # =========================================================================
    # GPU implementations
    # =========================================================================

    @staticmethod
    def _fill_scalar_gpu(
        target: NDBuffer[Self.dtype],
        value: Scalar[Self.dtype],
        shape: Shape,
        strides: Strides,
        absolute_offset: Int,
    ) raises:
        """GPU scalar fill — kernel for contiguous, get/set fallback for strided.
        """
        comptime if has_accelerator():
            ref device_state = target.device_state.value()
            ref gpu = device_state.get_gpu()
            var ctx = gpu[]
            var size = shape.num_elements()

            if strides.is_contiguous(shape):
                # Fast path — launch fill kernel
                var (tpb, blocks) = _gpu_launch_config(size)
                var compiled = ctx.compile_function[
                    fill_scalar_kernel[Self.dtype],
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
                ctx.synchronize()
            else:
                # Strided fallback — get/set (correct, slower)
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
    ) raises:
        """GPU buffer fill — kernel for contiguous same-shape case,
        get/set fallback for strided or broadcast cases.
        """
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
                # Both contiguous and same shape — kernel copy
                var (tpb, blocks) = _gpu_launch_config(size)
                ref s_state = source.device_state.value()
                var compiled = ctx.compile_function[
                    fill_from_buffer_kernel[Self.dtype],
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
                ctx.synchronize()
            else:
                # Strided or broadcast — get/set fallback
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
                    # Broadcast case — get/set
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
    ) raises:
        """GPU scatter-add using atomic add kernel.

        Selects strided or contiguous kernel based on source/target layout.
        Atomic add ensures correctness when the same token_id appears more
        than once in a review's token_ids list.
        """
        comptime if has_accelerator():
            ref t_state = target.device_state.value()
            ref s_state = source.device_state.value()
            ref gpu = t_state.get_gpu()
            var ctx = gpu[]

            # Build Int32 indices buffer on GPU
            var idx_buf = ctx.enqueue_create_buffer[DType.int32](n_indices)
            with idx_buf.map_to_host() as host_idx:
                for k in range(n_indices):
                    host_idx[k] = Int32(indices[k])

            # One block per source row, one thread per column
            # Cap threads at 512 (hardware limit in our kernels)
            var tpb = min(row_width, 512)
            var blocks = n_indices

            if source.is_contiguous() and target.strides[1] == 1:
                # Fast path — contiguous kernel
                var compiled = ctx.compile_function[
                    scatter_add_rows_kernel[Self.dtype],
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
            else:
                # Strided path
                var compiled = ctx.compile_function[
                    scatter_add_rows_strided_kernel[Self.dtype],
                    scatter_add_rows_strided_kernel[Self.dtype],
                ]()
                ctx.enqueue_function(
                    compiled,
                    t_state.device_buffer(),
                    s_state.device_buffer(),
                    idx_buf,
                    target.strides[0],
                    target.strides[1] if target.shape.rank() > 1 else 1,
                    source.strides[0] if source.shape.rank() > 1 else row_width,
                    source.strides[1] if source.shape.rank() > 1 else 1,
                    target.offset,
                    source.offset,
                    n_indices,
                    row_width,
                    grid_dim=blocks,
                    block_dim=tpb,
                )

            ctx.synchronize()
