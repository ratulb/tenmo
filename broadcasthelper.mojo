from shapes import Shape
from common_utils import panic
from layout.int_tuple import IntArray
from intlist import IntList

from testing import assert_true


fn main() raises:
    pass

@register_passable
struct ShapeBroadcaster:
    """Utility for broadcasting and manipulating shapes for tensor operations.
    """

    @always_inline
    @staticmethod
    fn pad_shapes(shape1: Shape, shape2: Shape) -> Tuple[Shape, Shape]:
        """Pad two shapes with ones to make them the same length."""
        if shape1 == shape2:
            return shape1.copy(), shape2.copy()
        if shape1 == Shape():
            return Shape(1) * len(shape2), shape2.copy()
        if shape2 == Shape():
            return shape1.copy(), Shape(1) * len(shape1)

        var max_len = max(len(shape1), len(shape2))

        # Pad with 1s
        var padded1 = Shape(1) * (max_len - len(shape1)) + shape1
        var padded2 = Shape(1) * (max_len - len(shape2)) + shape2

        return padded1^, padded2^

    @always_inline
    @staticmethod
    fn broadcast_shape[
        validated: Bool = False
    ](this: Shape, that: Shape) -> Shape:
        """Compute the broadcasted shape from two input shapes."""

        @parameter
        if not validated:
            if not ShapeBroadcaster.broadcastable(this, that):
                panic(
                    "ShapeBroadcaster → broadcast_shape - not broadcastable: "
                    + this.__str__()
                    + " <=> "
                    + that.__str__()
                )
        # Explicitly handle true scalars (Shape())
        if this == Shape():
            return that.copy()  # Scalar + Tensor -> Tensor's shape
        if that == Shape():
            return this.copy()  # Tensor + Scalar -> Tensor's shape

        var padded = ShapeBroadcaster.pad_shapes(this, that)
        var shape1 = padded[0].copy()
        var shape2 = padded[1].copy()
        var result_shape = IntList.with_capacity(len(shape1))
        var s1 = shape1.intlist()
        var s2 = shape2.intlist()

        for dims in s1.zip(s2):
            if dims[0] == dims[1]:
                result_shape.append(dims[0])
            elif dims[0] == 1:
                result_shape.append(dims[1])
            elif dims[1] == 1:
                result_shape.append(dims[0])
            else:
                panic(
                    "ShapeBroadcaster → broadcast_shape - cannot broadcast"
                    " shapes: "
                    + this.__str__()
                    + ", "
                    + that.__str__()
                )

        return Shape(result_shape)

    @always_inline
    @staticmethod
    fn translate_index(
        original_shape: Shape,
        indices: IntArray,
        mask: IntArray,
        broadcast_shape: Shape,
    ) -> IntArray:
        """Translate broadcasted indices to original tensor indices.

        Args:
            original_shape: The original shape before broadcasting.
            indices: Position in broadcasted tensor.
            mask: 1 for broadcasted dims, 0 for original.
            broadcast_shape: Shape after broadcasting.

        Returns:
            Indices in original tensor's space.
        """
        # Input Validation
        if original_shape.ndim > broadcast_shape.ndim:
            panic(
                "ShapeBroadcaster → translate_index: original dims greater than"
                " broadcast dims"
            )
        if mask.size() != broadcast_shape.ndim:
            panic(
                "ShapeBroadcaster → translate_index: mask size does not match"
                " broadcast ndim"
            )
        if indices.size() != broadcast_shape.ndim:
            panic(
                "ShapeBroadcaster → translate_index: indices size does not"
                " match broadcast ndim"
            )

        var translated = IntArray(size=original_shape.ndim)
        var offset = broadcast_shape.ndim - original_shape.ndim

        # Perform the translation
        for i in range(original_shape.ndim):
            var broadcast_axis = i + offset

            if mask[broadcast_axis] == 1:
                translated[i] = 0  # Broadcasted dim
            else:
                var original_index = indices[broadcast_axis]
                # CRITICAL: Check if the index is valid for the original shape
                if original_index >= original_shape[i]:
                    panic(
                        "ShapeBroadcaster → translate_index: index out of"
                        " bounds for original tensor"
                    )
                translated[i] = original_index

        return translated^

    @always_inline
    @staticmethod
    fn broadcastable(shape1: Shape, shape2: Shape) -> Bool:
        """Check if two shapes are broadcastable."""
        var dims1 = shape1.intlist()
        var dims2 = shape2.intlist()
        var zip_reversed = dims1.zip_reversed(dims2)
        for dims in zip_reversed:
            if dims[0] != dims[1]:
                if dims[0] != 1 and dims[1] != 1:
                    return False
        return True

    @always_inline
    @staticmethod
    fn broadcast_mask(original_shape: Shape, target_shape: Shape) -> IntArray:
        """Create a broadcast mask indicating which dimensions are broadcasted.
        """
        var mask = IntArray(size=target_shape.ndim)
        var offset = target_shape.ndim - original_shape.ndim
        if offset < 0:
            panic(
                "ShapeBroadcaster → broadcast_mask → target_shape.ndim is"
                " smaller than original_shape.ndim: "
                + String(target_shape.ndim)
                + ", "
                + String(original_shape.ndim)
            )

        for i in range(target_shape.ndim):
            if i < offset:
                mask[i] = 1  # original_shape has no dimension here
            else:
                var base_dim = original_shape[i - offset]
                var target_dim = target_shape[i]
                if base_dim == 1 and target_dim != 1:
                    mask[i] = 1  # original_shape is being expanded
                else:
                    mask[i] = 0  # match or both 1 → not broadcasted

        return mask^

    @staticmethod
    fn broadcasted_indices(
        target_indices: IntArray, target_shape: Shape, source_shape: Shape
    ) -> IntArray:
        # Get coordinates for source tensor given target coordinates
        var source_indices = IntArray(size=len(source_shape))

        for i in range(len(source_shape)):
            target_idx = len(target_shape) - len(source_shape) + i
            if source_shape[i] == 1:
                source_indices[i] = 0  # Broadcasted dimension → use 0
            else:
                source_indices[i] = target_indices[
                    target_idx
                ]  # Normal dimension

        return source_indices^
