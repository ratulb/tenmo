from tenmo import Tensor
from shapes import Shape
from testing import assert_true, assert_raises
from broadcasthelper import ShapeBroadcaster


fn test_empty_shape_broadcastable() raises:
    print("test_empty_shape_broadcastable")
    shape = Shape()
    assert_true(
        ShapeBroadcaster.broadcastable(shape, Shape.of(1)),
        "broadcastable assertion 1 failed for empty shape",
    )
    assert_true(
        ShapeBroadcaster.broadcastable(Shape.of(1), shape),
        "broadcastable assertion 2 failed for empty shape",
    )
    broadcast_shape = ShapeBroadcaster.broadcast_shape(shape, Shape.of(1))
    assert_true(
        broadcast_shape == Shape.of(1),
        "Empty shape broadcast to Shape.of(1) assertion failed",
    )

    broadcast_shape = ShapeBroadcaster.broadcast_shape(Shape.of(1), shape)
    assert_true(
        broadcast_shape == Shape.of(1),
        "Shape.of(1) broadcast with empty shape assertion failed",
    )
    broadcast_mask = ShapeBroadcaster.broadcast_mask(shape, Shape.of(1))
    assert_true(
        IntArray(1) == broadcast_mask,
        "Empty shape broadcast mask assertion failed",
    )


fn test_broadcast_shape() raises:
    print("test_broadcast_shape")
    shape1 = Shape.of(32, 16)
    shape2 = Shape.of(
        16,
    )
    result = ShapeBroadcaster.broadcast_shape(shape1, shape2)
    assert_true(
        result == Shape.of(32, 16), "Shape broadcast 1 assertion failed"
    )

    shape1 = Shape.of(4, 16, 32)
    shape2 = Shape.of(
        32,
    )
    result = ShapeBroadcaster.broadcast_shape(shape1, shape2)
    assert_true(
        result == Shape.of(4, 16, 32), "Shape broadcast 2 assertion failed"
    )

    shape1 = Shape.of(4, 16, 32, 64)
    shape2 = Shape.of(
        64,
    )
    result = ShapeBroadcaster.broadcast_shape(shape1, shape2)
    assert_true(
        result == Shape.of(4, 16, 32, 64), "Shape broadcast 3 assertion failed"
    )

    shape1 = Shape.of(1)
    shape2 = Shape.of(
        3,
        4,
    )
    result = ShapeBroadcaster.broadcast_shape(shape1, shape2)
    assert_true(result == Shape.of(3, 4), "Shape broadcast 4 assertion failed")

    result = ShapeBroadcaster.broadcast_shape(Shape.of(2, 1), Shape.of(4, 2, 5))
    assert_true(
        result == Shape.of(4, 2, 5), "Shape broadcast 5 assertion failed"
    )


fn test_broadcastable() raises:
    print("test_broadcastable")
    assert_true(
        ShapeBroadcaster.broadcastable(Shape(1), Shape.of(1)),
        "broadcastable assertion 1 failed",
    )
    assert_true(
        ShapeBroadcaster.broadcastable(Shape(4, 5), Shape.of(1)),
        "broadcastable assertion 2 failed",
    )
    assert_true(
        ShapeBroadcaster.broadcastable(Shape(2, 3, 5), Shape.of(1, 5)),
        "broadcastable assertion 3 failed",
    )
    assert_true(
        ShapeBroadcaster.broadcastable(Shape(2, 3, 5), Shape.of(3, 5)),
        "broadcastable assertion 4 failed",
    )
    tensor1 = Tensor.of(1, 2, 3, 4, 5)
    tensor2 = Tensor.of(6)
    assert_true(
        tensor1.broadcastable(tensor2) and tensor2.broadcastable(tensor1),
        "Tensor shape broadcastable assertion failed",
    )


fn test_pad_shapes() raises:
    print("test_pad_shapes")
    shape1 = Shape.of(3, 4)
    shape2 = Shape.of(
        4,
    )
    padded1, padded2 = ShapeBroadcaster.pad_shapes(shape1, shape2)
    assert_true(
        padded1 == shape1 and padded2 == Shape.of(1, 4),
        "Padding of shapes (3,4) and (4,) assertion failed",
    )
    shape1 = Shape.of(5, 3, 1)
    shape2 = Shape.of(5, 1)
    padded1, padded2 = ShapeBroadcaster.pad_shapes(shape1, shape2)
    assert_true(
        padded1 == shape1 and padded2 == Shape.of(1, 5, 1),
        "Padding of shapes (5,3,1) and (5,1) assertion failed",
    )
    shape1 = Shape.of(
        1,
    )
    shape2 = Shape.of(3, 4)
    padded1, padded2 = ShapeBroadcaster.pad_shapes(shape1, shape2)
    assert_true(
        padded1 == Shape.of(1, 1) and padded2 == shape2,
        "Padding of shapes (1, ) and (3,4) assertion failed",
    )
    shape1 = Shape.of(3, 4, 5, 2)
    shape2 = Shape.of(3, 4, 5, 2)
    padded1, padded2 = ShapeBroadcaster.pad_shapes(shape1, shape2)
    assert_true(
        padded1 == shape1 and padded2 == shape2,
        "Padding of shapes (3,4,5,2 ) and (3,4,5,2) assertion failed",
    )


fn main() raises:
    test_empty_shape_broadcastable()
    test_broadcastable()
    test_pad_shapes()
    test_broadcast_shape()
