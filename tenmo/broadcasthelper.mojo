from .shapes import Shape
from .common_utils import panic
from .intarray import IntArray
from .strides import Strides


@fieldwise_init
struct ShapeBroadcaster(ImplicitlyCopyable, RegisterPassable):
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

        comptime if not validated:
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
        var result_shape = IntArray.with_capacity(len(shape1))
        var s1 = shape1.intarray()
        var s2 = shape2.intarray()

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
        if original_shape.ndim() > broadcast_shape.ndim():
            panic(
                "ShapeBroadcaster → translate_index: original dims greater than"
                " broadcast dims"
            )
        if len(mask) != broadcast_shape.ndim():
            panic(
                "ShapeBroadcaster → translate_index: mask size does not match"
                " broadcast ndim"
            )
        if len(indices) != broadcast_shape.ndim():
            panic(
                "ShapeBroadcaster → translate_index: indices size does not"
                " match broadcast ndim"
            )

        var translated = IntArray.with_capacity(original_shape.ndim())
        var offset = broadcast_shape.ndim() - original_shape.ndim()

        # Perform the translation
        for i in range(original_shape.ndim()):
            var broadcast_axis = i + offset

            if mask[broadcast_axis] == 1:
                translated.append(0)  # Broadcasted dim
            else:
                var original_index = indices[broadcast_axis]
                # Check if the index is valid for the original shape
                if original_index >= original_shape[i]:
                    panic(
                        "ShapeBroadcaster → translate_index: index out of"
                        " bounds for original tensor"
                    )
                translated.append(original_index)

        return translated^

    @always_inline
    @staticmethod
    fn broadcastable(shape1: Shape, shape2: Shape) -> Bool:
        """Check if two shapes are broadcastable."""
        var dims1 = shape1.intarray()
        var dims2 = shape2.intarray()
        var zip_reversed = dims1.zip_reversed(dims2)
        for dims in zip_reversed:
            if dims[0] != dims[1]:
                if dims[0] != 1 and dims[1] != 1:
                    return False
        return True

    @always_inline
    @staticmethod
    fn expandable_to(own: Shape, target: Shape) -> Bool:
        """
        Directed broadcast check: can `own` be expanded to exactly `target`?

        Rules (right-aligned):
            own_dim == target_dim  → ok (exact match)
            own_dim == 1           → ok (will be stretched)
            else                   → False

        Unlike broadcastable(), target dims are fixed — they never stretch.

        Examples:
            (2, 1, 3) → (2, 4, 3)  : True   (dim1: 1 stretches to 4)
            (1,)      → (2, 4, 3)  : True   (all dims stretch)
            (2, 1, 3) → (2, 4, 5)  : False  (dim2: 3 != 5 and 3 != 1)
            (1, 2, 1) → (2, 3)     : False  (dim1: 2 != 3 and 2 != 1)
            (3, 4)    → (2, 3, 4)  : True   (own rank < target, prepend handled)
            (5, 4)    → (2, 3, 4)  : False  (dim1: 5 != 3 and 5 != 1)
        """
        var own_rank = own.rank()
        var target_rank = target.rank()

        if own_rank > target_rank:
            return False

        var offset = target_rank - own_rank  # leading dims own doesn't have

        for i in range(own_rank):
            var own_dim = own[i]
            var target_dim = target[i + offset]
            if own_dim != target_dim and own_dim != 1:
                return False

        return True

    @always_inline
    @staticmethod
    fn broadcast_mask(original_shape: Shape, target_shape: Shape) -> IntArray:
        """Create a broadcast mask indicating which dimensions are broadcasted.
        """
        var mask = IntArray.with_capacity(target_shape.ndim())
        var offset = target_shape.ndim() - original_shape.ndim()
        if offset < 0:
            panic(
                "ShapeBroadcaster → broadcast_mask → target_shape.ndim is"
                " smaller than original_shape.ndim: "
                + String(target_shape.ndim())
                + ", "
                + String(original_shape.ndim())
            )

        for i in range(target_shape.ndim()):
            if i < offset:
                mask.append(1)  # original_shape has no dimension here
            else:
                var base_dim = original_shape[i - offset]
                var target_dim = target_shape[i]
                if base_dim == 1 and target_dim != 1:
                    mask.append(1)  # original_shape is being expanded
                else:
                    mask.append(0)  # match or both 1 → not broadcasted

        return mask^

    @staticmethod
    fn broadcasted_indices(
        target_indices: IntArray, target_shape: Shape, source_shape: Shape
    ) -> IntArray:
        # Get coordinates for source tensor given target coordinates
        var source_indices = IntArray.with_capacity(len(source_shape))

        for i in range(len(source_shape)):
            target_idx = len(target_shape) - len(source_shape) + i
            if source_shape[i] == 1:
                source_indices.append(0)  # Broadcasted dimension → use 0
            else:
                source_indices.append(
                    target_indices[target_idx]
                )  # Normal dimension

        return source_indices^

    @staticmethod
    fn broadcast_strides(
        input_shape: Shape,
        input_strides: Strides,
        broadcast_shape: Shape,
    ) -> Strides:
        var in_rank = input_shape.rank()
        var out_rank = broadcast_shape.rank()

        if in_rank > out_rank:
            panic("Cannot broadcast: input rank greater than target rank")

        var result = Strides.zeros(out_rank)

        # Align dimensions from the right
        var in_dim = in_rank - 1
        var out_dim = out_rank - 1

        while out_dim >= 0:
            var out_size = broadcast_shape[out_dim]

            if in_dim >= 0:
                var in_size = input_shape[in_dim]

                if in_size == out_size:
                    # Same dimension → keep stride
                    result[out_dim] = input_strides[in_dim]

                elif in_size == 1:
                    # Broadcast dimension → stride 0
                    result[out_dim] = 0

                else:
                    panic(
                        "Broadcast error: ",
                        input_shape.__str__(),
                        " → ",
                        broadcast_shape.__str__(),
                    )

                in_dim -= 1

            else:
                # Input had no dimension here (implicit 1)
                result[out_dim] = 0

            out_dim -= 1

        return result


from std.memory import stack_allocation


fn main() raises:
    test_broadcast_strides()
    shape = Shape(10, 1)
    stack = stack_allocation[6144, Int]()
    stack[6143] = shape[0]
    print(stack[6143])


from std.testing import assert_true


fn test_broadcast_strides() raises:
    print("test_broadcast_strides")
    var input_shape = Shape(3)
    var input_strides = Strides(1)
    var target_shape = Shape(2, 3)

    var result = ShapeBroadcaster.broadcast_strides(
        input_shape, input_strides, target_shape
    )
    assert_true(result == Strides(0, 1))

    input_shape = Shape(2, 1, 4)
    input_strides = Strides(4, 4, 1)
    target_shape = Shape(2, 3, 4)

    result = ShapeBroadcaster.broadcast_strides(
        input_shape, input_strides, target_shape
    )
    assert_true(result == Strides(4, 0, 1))
