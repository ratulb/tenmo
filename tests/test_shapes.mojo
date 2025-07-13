from intlist import IntList
from tensors import Tensor
from shapes import Shape

from testing import assert_true, assert_raises


fn test_reverse() raises:
    shape = Shape.of(1, 2, 3)
    assert_true(
        shape.reverse() == Shape.of(3, 2, 1), "Shape reversal assertion failed"
    )


fn test_equivalence() raises:
    assert_true(Shape(IntList(1, 4)) == Shape.of(1, 4), "Not equivalent")


fn test_empty_shape() raises:
    shape = Shape(IntList.Empty)
    assert_true(shape[0] == -1, "Empty shape __getitem__ assertion failed")
    for each in shape:
        assert_true(
            each == IntList.Empty, "Empty shape iteration assertion failed"
        )
    tensor = Tensor[DType.bool](shape)
    assert_true(
        tensor[IntList.Empty] == False, "Scalar tensor get assertion 1 failed"
    )
    tensor[IntList.Empty] = True
    assert_true(
        tensor[IntList.Empty] == True, "Scalar tensor get assertion 2 failed"
    )
    assert_true(tensor.item() == True, "Scalar tensor item() assertion failed")
    assert_true(
        shape.broadcastable(Shape.of(1)),
        "broadcastable assertion 1 failed for empty shape",
    )
    assert_true(
        Shape.of(1).broadcastable(shape),
        "broadcastable assertion 1 failed for empty shape",
    )

    broadcast_shape = Shape.broadcast_shape(shape, Shape.of(1))
    assert_true(
        broadcast_shape == Shape.of(1),
        "Empty shape broadcast to Shape.of(1) assertion failed",
    )

    broadcast_shape = Shape.broadcast_shape(Shape.of(1), shape)
    assert_true(
        broadcast_shape == Shape.of(1),
        "Shape.of(1) broadcast with empty shape assertion failed",
    )
    broadcast_mask = shape.broadcast_mask(Shape.of(1))
    assert_true(
        broadcast_mask == IntList(1),
        "Empty shape broadcast mask assertion failed",
    )


fn test_replace() raises:
    shape = Shape.of(3, 4, 2)
    shape = shape.replace(2, 5)
    assert_true(shape == Shape.of(3, 4, 5), "replace assertion failed")


fn test_broadcast_shape() raises:
    shape1 = Shape.of(32, 16)
    shape2 = Shape.of(
        16,
    )
    result = Shape.broadcast_shape(shape1, shape2)
    assert_true(
        result == Shape.of(32, 16), "Shape broadcast 1 assertion failed"
    )

    shape1 = Shape.of(4, 16, 32)
    shape2 = Shape.of(
        32,
    )
    result = Shape.broadcast_shape(shape1, shape2)
    assert_true(
        result == Shape.of(4, 16, 32), "Shape broadcast 2 assertion failed"
    )

    shape1 = Shape.of(4, 16, 32, 64)
    shape2 = Shape.of(
        64,
    )
    result = Shape.broadcast_shape(shape1, shape2)
    assert_true(
        result == Shape.of(4, 16, 32, 64), "Shape broadcast 3 assertion failed"
    )

    _ = """shape1 = Shape.of(5, 3, 1)
    shape2 = Shape.of(5, 1)

    with assert_raises():
        _ = Shape.broadcast_shape(shape1, shape2)"""

    shape1 = Shape.of(1)
    shape2 = Shape.of(
        3,
        4,
    )
    result = Shape.broadcast_shape(shape1, shape2)
    assert_true(result == Shape.of(3, 4), "Shape broadcast 4 assertion failed")

    result = Shape.broadcast_shape(Shape.of(2, 1), Shape.of(4, 2, 5))
    assert_true(
        result == Shape.of(4, 2, 5), "Shape broadcast 5 assertion failed"
    )


fn test_index_iter() raises:
    shape = Shape.of(1)
    for each in shape:
        assert_true(
            each == IntList(0),
            "Unit shape(Shape.of(1)) index iteration assertion failed",
        )
    shape = Shape.of(2, 1)
    indices = shape.__iter__()
    assert_true(
        indices.__next__() == IntList(0, 0)
        and indices.__next__() == IntList(1, 0),
        "Shape(2,1) iteration assertion failed",
    )


fn test_broadcastable() raises:
    assert_true(
        Shape.of(1).broadcastable(Shape.of(1)),
        "broadcastable assertion 1 failed",
    )
    assert_true(
        Shape.of(4, 5).broadcastable(Shape.of(1)),
        "broadcastable assertion 2 failed",
    )
    assert_true(
        Shape.of(2, 3, 5).broadcastable(Shape.of(1, 5)),
        "broadcastable assertion 3 failed",
    )
    assert_true(
        Shape.of(2, 3, 5).broadcastable(Shape.of(3, 5)),
        "broadcastable assertion 4 failed",
    )
    tensor1 = Tensor.of(1, 2, 3, 4, 5)
    tensor2 = Tensor.of(6)
    assert_true(
        tensor1.shape.broadcastable(tensor2.shape)
        and tensor2.broadcastable(tensor1),
        "Tensor shape broadcastable assertion failed",
    )


fn test_shape_as_intlist() raises:
    shape = Shape.of(2, 4, 5)
    fa = shape.intlist()
    assert_true(
        fa[0] == 2 and fa[1] == 4 and fa[2] == 5,
        "Shape to IntList assertion failed",
    )


fn test_pad_shapes() raises:
    shape1 = Shape.of(3, 4)
    shape2 = Shape.of(
        4,
    )
    padded1, padded2 = Shape.pad_shapes(shape1, shape2)
    assert_true(
        padded1 == shape1 and padded2 == Shape.of(1, 4),
        "Padding of shapes (3,4) and (4,) assertion failed",
    )
    shape1 = Shape.of(5, 3, 1)
    shape2 = Shape.of(5, 1)
    padded1, padded2 = Shape.pad_shapes(shape1, shape2)
    assert_true(
        padded1 == shape1 and padded2 == Shape.of(1, 5, 1),
        "Padding of shapes (5,3,1) and (5,1) assertion failed",
    )
    shape1 = Shape.of(
        1,
    )
    shape2 = Shape.of(3, 4)
    padded1, padded2 = Shape.pad_shapes(shape1, shape2)
    assert_true(
        padded1 == Shape.of(1, 1) and padded2 == shape2,
        "Padding of shapes (1, ) and (3,4) assertion failed",
    )
    shape1 = Shape.of(3, 4, 5, 2)
    shape2 = Shape.of(3, 4, 5, 2)
    padded1, padded2 = Shape.pad_shapes(shape1, shape2)
    assert_true(
        padded1 == shape1 and padded2 == shape2,
        "Padding of shapes (3,4,5,2 ) and (3,4,5,2) assertion failed",
    )


fn test_zip_reversed() raises:
    shape1 = Shape.of(1, 2, 3, 4, 5)
    shape2 = Shape.of(6)
    rzipped = shape1.intlist().zip_reversed(shape2.intlist())
    for each in rzipped:
        assert_true(
            each[0] == 5 and each[1] == 6, "zip_reversed assertion failed"
        )


fn main() raises:
    test_reverse()
    test_equivalence()
    test_empty_shape()
    test_replace()
    test_broadcastable()
    test_pad_shapes()
    test_broadcast_shape()
    test_shape_as_intlist()
    test_index_iter()
    test_zip_reversed()


