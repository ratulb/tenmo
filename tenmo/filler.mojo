from .shared.indexhelper import Idx
from .shared.panic import panic
from .validators import Validator
from .shared.broadcasthelper import ShapeBroadcaster
from .shared.indexhelper import IndexIterator
from .ndbuffer import NDBuffer
from .shared.shapes import Shape
from .shared.intarray import IntArray
from .shared.strides import Strides
from std.memory import unsafe_memcpy
from std.sys import has_accelerator, simd_width_of
from std.sys.info import num_physical_cores
from max.algorithm import parallelize
from .kernels.filler_kernel import FillerKernel


# Filler


@fieldwise_init
struct Filler[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    """Element-wise fill and copy for NDBuffer — CPU and GPU capable.

    CPU path: fast unsafe_memcpy for contiguous cases, strided iterator otherwise.
    GPU path: dedicated kernels for contiguous cases;
              scatter_add_rows(_strided) for ScatterAddTensor backward.

    All public entry points dispatch on device automatically — callers
    do not need to check is_on_gpu().
    """

    # Public API — scalar fill

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
                    FillerKernel[Self.dtype]._fill_scalar_gpu(
                        target.layout(),
                        target.device_state.value(),
                        value,
                        shape,
                        strides,
                        absolute_offset,
                        sync=sync,
                    )
                    return
            Self._fill_scalar_cpu(
                target, value, shape, strides, absolute_offset
            )
        except e:
            print(e)
            panic("Filler fill(scalar) error")

    # Public API — buffer-to-buffer fill (general)

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
                    FillerKernel[Self.dtype]._fill_buffer_gpu(
                        target.layout(),
                        target.device_state.value(),
                        source.layout(),
                        source.device_state.value(),
                        shape,
                        strides,
                        absolute_offset,
                        sync=sync,
                    )
                    return
            Self._fill_buffer_cpu(
                target, source, shape, strides, absolute_offset
            )
        except e:
            print(e)
            panic("Filler fill(scalar) error")

    # Public API — List[Idx]-based fills (Python bindings)

    @always_inline
    @staticmethod
    def fill_list(
        target: NDBuffer[Self.dtype],
        value: Scalar[Self.dtype],
        indices: List[Idx],
        sync: Bool = True,
    ):
        """List[Idx]-based scalar fill — non-variadic twin of fill(scalar)."""
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
                    FillerKernel[Self.dtype]._fill_scalar_gpu(
                        target.layout(),
                        target.device_state.value(),
                        value,
                        shape,
                        strides,
                        absolute_offset,
                        sync=sync,
                    )
                    return
            Self._fill_scalar_cpu(
                target, value, shape, strides, absolute_offset
            )
        except e:
            print(e)
            panic("Filler fill(scalar) error")

    @always_inline
    @staticmethod
    def fill_list(
        target: NDBuffer[Self.dtype],
        source: NDBuffer[Self.dtype],
        indices: List[Idx],
        sync: Bool = True,
    ):
        """List[Idx]-based buffer fill — non-variadic twin of fill(buffer)."""
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
                    FillerKernel[Self.dtype]._fill_buffer_gpu(
                        target.layout(),
                        target.device_state.value(),
                        source.layout(),
                        source.device_state.value(),
                        shape,
                        strides,
                        absolute_offset,
                        sync=sync,
                    )
                    return
            Self._fill_buffer_cpu(
                target, source, shape, strides, absolute_offset
            )
        except e:
            print(e)
            panic("Filler fill(buffer) error")

    # Public API — scatter-add rows (GatherBackward / ScatterAddTensor==op_code)

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
                        FillerKernel[Self.dtype]._scatter_add_gpu(
                            target.layout(),
                            target.device_state.value(),
                            source.layout(),
                            source.device_state.value(),
                            indices,
                            n_indices,
                            slice_volume,
                            sync=sync,
                        )
                    else:
                        FillerKernel[Self.dtype]._scatter_add_nd_gpu(
                            target.layout(),
                            target.device_state.value(),
                            source.layout(),
                            source.device_state.value(),
                            indices,
                            n_indices,
                            slice_volume,
                            axis,
                            sync=sync,
                        )
                    return
            Self._scatter_add_cpu(
                target, source, indices, n_indices, slice_volume, axis
            )
        except e:
            print(e)
            panic("Error in Filler scatter_add")

    # CPU implementations

    @staticmethod
    def _fill_scalar_cpu(
        target: NDBuffer[Self.dtype],
        value: Scalar[Self.dtype],
        shape: Shape,
        strides: Strides,
        absolute_offset: Int,
    ):
        if strides.is_contiguous(shape):
            # Fast path — SIMD vectorized fill via Buffer.fill
            ref buffer = target.data_buffer()
            var end = absolute_offset + shape.num_elements()
            buffer.fill(value, absolute_offset, end)
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
                # Both contiguous — unsafe_memcpy fast path
                var dest = (
                    target.data_ptr()
                    .unsafe_mut_cast[True]()
                    .unsafe_origin_cast[MutAnyOrigin]()
                )
                var src = source.data_ptr()
                unsafe_memcpy(
                    dest=dest.unsafe_offset(absolute_offset),
                    src=src.unsafe_offset(source.offset),
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
                    var src_idx = src_iter.peek()
                    var dst_idx = dest_iter.peek()
                    dest_buf[dst_idx] = src_buf[src_idx]
                    src_iter.skip(1)
                    dest_iter.skip(1)

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
            # Fast path: contiguous target slice + contiguous source with a
            # 1:1 last dim — every output row is a contiguous run, so
            # bulk-memcpy per row (parallelized). A broadcasting source last
            # dim becomes a per-row SIMD splat. Falls through to the generic
            # coord loop otherwise.
            var rank = shape.rank()
            var src_rank = source_shape.rank()
            var last = shape[rank - 1] if rank > 0 else 0
            var src_last_size = (
                source_shape[src_rank - 1] if src_rank > 0 else 1
            )
            if (
                rank > 0
                and last > 0
                and strides.is_contiguous(shape)
                and source.is_contiguous()
            ):
                var outer = shape.num_elements() // last
                var eff = List[Int]()
                var run = 1
                for d in range(rank - 1, -1, -1):
                    var sd = d - (rank - src_rank)
                    if sd < 0 or source_shape[sd] == 1:
                        eff.append(0)
                    else:
                        eff.append(run)
                    if sd >= 0:
                        run *= source_shape[sd]
                var src_off_base = source.offset
                ref src_buf = source.data_buffer()
                var sptr = src_buf.unsafe_ptr()
                ref dest_buf = target.data_buffer()
                var dptr = dest_buf.unsafe_ptr()
                var n_threads = num_physical_cores()
                comptime sw = simd_width_of[Self.dtype]()

                def fill_row(r: Int) {imm}:
                    var rem = r
                    var src_off = src_off_base
                    for d in range(rank - 2, -1, -1):
                        var sz = shape[d]
                        var c = rem % sz
                        rem //= sz
                        src_off += c * eff[rank - 1 - d]
                    var dst = absolute_offset + r * last
                    if src_last_size == last:
                        unsafe_memcpy(
                            dest=dptr.unsafe_offset(dst),
                            src=sptr.unsafe_offset(src_off),
                            count=last,
                        )
                    else:
                        var v = sptr[unsafe_offset=src_off]
                        var vec = SIMD[Self.dtype, sw](v)
                        var c = 0
                        while c + sw <= last:
                            dptr.unsafe_store[width=sw](dst + c, vec)
                            c += sw
                        while c < last:
                            dptr[unsafe_offset=dst + c] = v
                            c += 1

                if (
                    outer >= n_threads
                    and shape.num_elements() >= n_threads * 32768
                ):
                    parallelize(fill_row, outer, n_threads)
                else:
                    for r in range(outer):
                        fill_row(r)
                return
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

        Fast path (transformer embedding / wte backward):
            2D target, axis=0, rank-2 contiguous non-broadcast source.
        Each target row is owned by exactly one thread (`row % n_threads`),
        so threads never race on the target buffer — no atomics, and the
        row accumulation is vectorized over the contiguous inner dim.
        """
        var rank = target.shape.rank()
        var is_broadcast = source.shape.rank() == 1

        if (
            not is_broadcast
            and rank == 2
            and axis == 0
            and source.shape.rank() == 2
            and source.is_contiguous()
            and target.is_contiguous()
        ):
            Self._scatter_add_2d_axis0_cpu(
                target, source, indices, n_indices, slice_volume
            )
            return

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

                # Offsets are in-range by construction: indices arrive
                # normalized to [0, tgt_ax_size) (gather forward panics on
                # OOB at index-normalization time), and elem decomposes
                # within one slice volume — so the min/max recomputation
                # and branches of checked access are pure overhead here.
                target.storage_set[checked=False](
                    dst_off,
                    target.storage_get[checked=False](dst_off)
                    + source.storage_get[checked=False](src_off),
                )

    @staticmethod
    def _scatter_add_2d_axis0_cpu(
        target: NDBuffer[Self.dtype],
        source: NDBuffer[Self.dtype],
        indices: IntArray,
        n_indices: Int,
        row_width: Int,
    ):
        """Vectorized scatter-add for the 2D axis=0 embedding case.

        target[n_rows, row_width]  <- target[indices[k], :] += source[k, :]
        Disjoint-row thread partition: thread t owns all target rows r where
        r % n_threads == t. Duplicate indices are handled (all source rows
        mapping to the same target row funnel to the same thread). The target
        gradbox may already hold contributions from earlier children — the
        load/add/store accumulates, never overwrites.
        """
        # Fast path assumes contiguous target & source (gradboxes are always
        # contiguous; source is the flattened (B*T, embd) incoming grad).
        var tptr = target.data_ptr().unsafe_offset(target.offset)
        var sptr = source.data_ptr().unsafe_offset(source.offset)
        comptime simd: Int = simd_width_of[Self.dtype]()

        def add_row(tr: Int, src_k: Int) {imm}:
            var top = tr * row_width
            var spo = src_k * row_width
            var c = 0
            while c + simd <= row_width:
                var acc = tptr.unsafe_load[width=simd](top + c)
                var srcv = sptr.unsafe_load[width=simd](spo + c)
                tptr.unsafe_store[width=simd](top + c, acc + srcv)
                c += simd
            while c < row_width:
                var to = top + c
                tptr[unsafe_offset=to] = (
                    tptr[unsafe_offset=to] + sptr[unsafe_offset=spo + c]
                )
                c += 1

        # Single-threaded path for trivially small workloads — parallelize()
        # would only add scheduling overhead for a handful of rows.
        if n_indices < 2 or num_physical_cores() < 2:
            for k in range(n_indices):
                add_row(indices[k], k)
            return

        var n_threads = num_physical_cores()

        def worker(t: Int) {imm}:
            # Thread t owns every target row with r % n_threads == t.
            for k in range(n_indices):
                var tr = indices[k]
                if t == tr % n_threads:
                    add_row(tr, k)

        parallelize(worker, n_threads, n_threads)
