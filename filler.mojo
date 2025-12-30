from tenmo import Tensor
from common_utils import Idx, panic
from validators import Validator
from broadcasthelper import ShapeBroadcaster
from indexhelper import IndexIterator
from memory import memcpy
from ndbuffer import NDBuffer


@fieldwise_init
@register_passable
struct Filler[dtype: DType](ImplicitlyCopyable & Movable):
    @always_inline
    @staticmethod
    fn fill(
        target: NDBuffer[Self.dtype],
        value: Scalar[Self.dtype],
        indices: VariadicListMem[Idx],
    ):
        # Compute view metadata
        shape, strides, offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                target.shape, target.strides, indices
            )
        )
        var absolute_offset = target.offset + offset
        ref buffer = target.data_buffer()
        if strides.is_contiguous(shape):
            buffer.fill(
                value,
                start_index=absolute_offset,
                end_index=absolute_offset + shape.num_elements(),
            )
        else:
            var index_iterator = IndexIterator(
                shape=Pointer(to=shape),
                strides=Pointer(to=strides),
                start_offset=absolute_offset,
            )
            for index in index_iterator:
                buffer[index] = value

    @always_inline
    @staticmethod
    fn fill(
        target: NDBuffer[Self.dtype],
        source: NDBuffer[Self.dtype],
        indices: VariadicListMem[Idx],
    ):
        shape, strides, offset = (
            Validator.validate_and_compute_advanced_indexing_metadata(
                target.shape, target.strides, indices
            )
        )
        ref source_shape = source.shape
        if not ShapeBroadcaster.broadcastable(source_shape, shape):
            panic(
                "Filler → fill: input buffer not broadcastable to shape",
                shape.__str__(),
            )
        var absolute_offset = target.offset + offset
        if shape == source_shape:
            if source.is_contiguous() and strides.is_contiguous(shape):
                dest = target.buffer.data
                src = source.buffer.data
                memcpy(
                    dest=dest + absolute_offset,
                    src=src + source.offset,
                    count=shape.num_elements(),
                )

            elif source.is_contiguous() and not strides.is_contiguous(shape):
                var src_offset = source.offset
                var index_iterator = IndexIterator(
                    shape=Pointer(to=shape),
                    strides=Pointer(to=strides),
                    start_offset=absolute_offset,
                )
                ref src_buffer = source.data_buffer()
                ref dest_buffer = target.data_buffer()
                for index in index_iterator:
                    dest_buffer[index] = src_buffer[src_offset]
                    src_offset += 1

            elif not source.is_contiguous() and strides.is_contiguous(shape):
                ref src_buffer = source.data_buffer()
                ref dest_buffer = target.data_buffer()
                for index in source.index_iterator():
                    dest_buffer[absolute_offset] = src_buffer[index]
                    absolute_offset += 1
            else:  # Neither src tensor nor selected patch contiguous
                ref src_buffer = source.data_buffer()
                ref dest_buffer = target.data_buffer()

                var src_iter = source.index_iterator()
                var dest_iter = IndexIterator(
                    shape=Pointer(to=shape),
                    strides=Pointer(to=strides),
                    start_offset=absolute_offset,
                )
                while src_iter.__has_next__():
                    dest_buffer[dest_iter.__next__()] = src_buffer[
                        src_iter.__next__()
                    ]
        else:
            # Source's shape != target's shape but broadcastable
            var broadcast_shape = ShapeBroadcaster.broadcast_shape[
                validated=True
            ](source_shape, shape)
            if broadcast_shape != shape:
                panic(
                    "Filler → set: broadcast shape",
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
            ref dest_buffer = target.data_buffer()
            for index in index_iterator:
                var coord = coord_iterator.__next__()
                var source_coord = ShapeBroadcaster.translate_index(
                    source_shape, coord, mask, shape
                )
                dest_buffer[index] = source[source_coord]


fn main() raises:
    pass
