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
from std.sys import has_accelerator, simd_width_of
from .kernels.filler_kernel import FillerGpu


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
        sync: Bool = True,
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
                    FillerGpu[Self.dtype]._fill_scalar_gpu(
                        target, value, shape, strides, absolute_offset, sync=sync
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
        sync: Bool = True,
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
                    FillerGpu[Self.dtype]._fill_buffer_gpu(
                        target, source, shape, strides, absolute_offset, sync=sync
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
        target: NDBuffer[Self.dtype],  # gradbox
        source: NDBuffer[Self.dtype],  # incoming grad
        indices: IntArray,
        axis: Int = 0,
        sync: Bool = True,
    ):
        """Scatter-add source into target at given indices along axis.

        For axis=0: target[indices[k], ...] += source[k, ...]
        For axis=1: target[:, indices[k], ...] += source[:, k, ...]
        (broadcasts source when source.shape.rank() == 1)

        Uses atomic add — safe for repeated indices.
        Dispatches to GPU kernel when target is on GPU, CPU loop otherwise.
        """
        try:
            var n_indices = len(indices)
            # Slice volume = number of target elements orthogonal to axis
            var tgt_ax_size = target.shape[axis]
            var slice_volume = target.numels() // tgt_ax_size

            comptime if has_accelerator():
                if target.is_on_gpu():
                    if axis == 0:
                        FillerGpu[Self.dtype]._scatter_add_gpu(
                            target, source, indices, n_indices, slice_volume, sync=sync
                        )
                    else:
                        FillerGpu[Self.dtype]._scatter_add_nd_gpu(
                            target, source, indices, n_indices, slice_volume, axis, sync=sync
                        )
                    return
            Self._scatter_add_cpu(
                target, source, indices, n_indices, slice_volume, axis
            )
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
            var broadcast_shape = ShapeBroadcaster.broadcast_shape(
                source_shape, shape
            )
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
        slice_volume: Int,
        axis: Int,
    ):
        """CPU scatter-add — iterates over all elements in each slice along axis.
        Correct for any rank and axis (not just 2D axis=0).
        """
        var rank = target.shape.rank()
        var is_broadcast = source.shape.rank() == 1

        for k in range(n_indices):
            var tgt_idx = indices[k]
            for elem in range(slice_volume):
                # Decompose elem into coordinates for non-axis dims
                var rem = elem
                var dst_off = target.offset + tgt_idx * target.strides[axis]
                var src_off: Int = source.offset
                if not is_broadcast:
                    src_off += k * source.strides[axis]

                for d in range(rank - 1, -1, -1):
                    if d == axis:
                        continue
                    var dim_size = target.shape[d]
                    var cd = rem % dim_size
                    rem //= dim_size
                    dst_off += cd * target.strides[d]
                    if not is_broadcast:
                        src_off += cd * source.strides[d]

                if is_broadcast:
                    src_off = source.offset + elem

                target.set(
                    dst_off,
                    target.get(dst_off) + source.get(src_off),
                )
