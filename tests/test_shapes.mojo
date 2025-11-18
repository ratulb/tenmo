from intlist import IntList
from tenmo import Tensor
from shapes import Shape

from testing import assert_true, assert_raises
from layout.int_tuple import IntArray

fn test_slice_shape() raises:
    print("test_slice_shape")
    shape = Shape([1, 2, 3, 4])
    assert_true(
        shape[:-1] == Shape.of(1, 2, 3)
        and shape[:-2] == Shape.of(1, 2)
        and shape[:-3] == Shape(1)
        and shape[2::4] == Shape.of(3)
        and shape[-1:] == Shape.of(4)
        and shape[-2:] == Shape.of(3, 4),
        "Shape slice assertion failed",
    )


fn test_negative_indices() raises:
    print("test_negative_indices")
    shape = Shape([1, 2, 3])
    assert_true(
        shape[-1] == 3 and shape[-2] == 2 and shape[-3] == 1,
        "Shape negative indices assertion failed",
    )


fn test_slice_from() raises:
    print("test_slice_from")
    shape = Shape.of(2, 3, 4)
    assert_true(
        shape[0:] == shape,
        "slice_from assertion from beginning failed",
    )
    assert_true(
        shape[1:] == Shape.of(3, 4),
        "slice_from assertion from from index 1 failed",
    )
    assert_true(
        shape[2:] == Shape.of(4),
        "slice_from assertion from from index 2 failed",
    )
    assert_true(
        shape[3:] == Shape(),
        "slice_from assertion from from index 3 failed",
    )


fn test_reverse() raises:
    print("test_reverse")
    shape = Shape.of(1, 2, 3)
    assert_true(
        shape.reverse() == Shape.of(3, 2, 1), "Shape reversal assertion failed"
    )


fn test_equivalence() raises:
    print("test_equivalence")
    assert_true(Shape(IntList(1, 4)) == Shape.of(1, 4), "Not equivalent")


fn test_empty_shape() raises:
    print("test_empty_shape")
    shape = Shape()
    assert_true(shape[0] == -1, "Empty shape __getitem__ assertion failed")
    for each in shape:
        assert_true(IntList() == each, "Empty shape iteration assertion failed")
    tensor = Tensor[DType.bool](shape)
    tensor[IntArray()] = True
    assert_true(
        tensor[IntArray()] == True, "Scalar tensor get assertion 2 failed"
    )

fn test_replace() raises:
    print("test_replace")
    shape = Shape.of(3, 4, 2)
    shape = shape.replace(2, 5)
    assert_true(shape == Shape.of(3, 4, 5), "replace assertion failed")



fn test_index_iter() raises:
    print("test_index_iter")
    shape = Shape.of(1)
    for each in shape:
        assert_true(
            IntList(0) == each,
            "Unit shape(Shape.of(1)) index iteration assertion failed",
        )
    shape = Shape.of(2, 1)
    indices = shape.__iter__()
    assert_true(
        IntList(0, 0) == indices.__next__()
        and IntList(1, 0) == indices.__next__(),
        "Shape(2,1) iteration assertion failed",
    )



fn test_shape_as_intlist() raises:
    print("test_shape_as_intlist")
    shape = Shape.of(2, 4, 5)
    fa = shape.intlist()
    assert_true(
        fa[0] == 2 and fa[1] == 4 and fa[2] == 5,
        "Shape to IntList assertion failed",
    )


fn test_zip_reversed() raises:
    print("test_zip_reversed")
    shape1 = Shape.of(1, 2, 3, 4, 5)
    shape2 = Shape.of(6)
    rzipped = shape1.intlist().zip_reversed(shape2.intlist())
    for each in rzipped:
        assert_true(
            each[0] == 5 and each[1] == 6, "zip_reversed assertion failed"
        )


fn main() raises:
    test_negative_indices()
    test_slice_shape()
    test_slice_from()
    test_reverse()
    test_equivalence()
    test_empty_shape()
    test_replace()
    test_shape_as_intlist()
    test_index_iter()
    test_zip_reversed()
