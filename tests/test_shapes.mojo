from tenmo.tensor import Tensor
from tenmo.shapes import Shape

from std.testing import assert_true, assert_raises, TestSuite
from tenmo.intarray import IntArray

# ============================================
# SHAPE TESTS
# ============================================


def test_shape_default_constructor() raises:
    var s = Shape()
    assert_true(s.rank() == 0, "default constructor should create scalar")
    assert_true(s.num_elements() == 1, "scalar should have 1 element")


def test_shape_void() raises:
    var s = Shape.Void()
    assert_true(s.rank() == 0, "Void() should create scalar")
    assert_true(s.numels() == 1, "scalar should have 1 element")


def test_shape_unit() raises:
    var s = Shape.Unit()
    assert_true(s.rank() == 1, "Unit() should have rank 1")
    assert_true(s[0] == 1, "Unit() dimension should be 1")
    assert_true(s.numels() == 1, "Unit() should have 1 element")


def test_shape_variadic_constructor() raises:
    var s = Shape(2, 3, 4)
    assert_true(
        s.rank() == 3, "variadic constructor should create rank 3 shape"
    )
    assert_true(s[0] == 2, "first dimension should be 2")
    assert_true(s[1] == 3, "second dimension should be 3")
    assert_true(s[2] == 4, "third dimension should be 4")
    assert_true(s.numels() == 24, "shape should have 24 elements")


def test_shape_list_constructor() raises:
    var lst = List[Int]()
    lst.append(5)
    lst.append(6)
    lst.append(7)
    var s = Shape(lst)
    assert_true(s.rank() == 3, "list constructor should create rank 3 shape")
    assert_true(s.numels() == 210, "shape should have 210 elements")


def test_shape_intarray_constructor() raises:
    var arr = IntArray(3, 4, 5)
    var s = Shape(arr)
    assert_true(
        s.rank() == 3, "IntArray constructor should create rank 3 shape"
    )
    assert_true(s[1] == 4, "second dimension should be 4")


def test_shape_getitem() raises:
    var s = Shape(10, 20, 30, 40)
    assert_true(s[0] == 10, "first dimension should be 10")
    assert_true(s[2] == 30, "third dimension should be 30")
    assert_true(s[-1] == 40, "last dimension should be 40")
    assert_true(s[-2] == 30, "second to last should be 30")


def test_shape_slice() raises:
    var s = Shape(10, 20, 30, 40, 50)
    var sliced = s[1:4]
    assert_true(sliced.rank() == 3, "sliced shape should have rank 3")
    assert_true(sliced[0] == 20, "first element should be 20")
    assert_true(sliced[2] == 40, "last element should be 40")


def test_shape_eq() raises:
    var s1 = Shape(2, 3, 4)
    var s2 = Shape(2, 3, 4)
    var s3 = Shape(2, 3, 5)
    assert_true(s1 == s2, "s1 should equal s2")
    assert_true(not (s1 == s3), "s1 should not equal s3")


def test_shape_eq_list() raises:
    var s = Shape(2, 3, 4)
    var lst = List[Int]()
    lst.append(2)
    lst.append(3)
    lst.append(4)
    assert_true(s == lst, "shape should equal list")


def test_shape_ne() raises:
    var s1 = Shape(2, 3)
    var s2 = Shape(2, 4)
    assert_true(s1 != s2, "s1 should not equal s2")


def test_shape_str() raises:
    var s = Shape(2, 3, 4)
    var str_repr = s.__str__()
    assert_true(str_repr == "(2, 3, 4)", "string should be '(2, 3, 4)'")


def test_shape_tolist() raises:
    var s = Shape(5, 10, 15)
    var lst = s.tolist()
    assert_true(len(lst) == 3, "list should have 3 elements")
    assert_true(lst[1] == 10, "second element should be 10")


def test_shape_add_shape() raises:
    var s1 = Shape(2, 3)
    var s2 = Shape(4, 5)
    var result = s1 + s2
    assert_true(result.rank() == 4, "concatenated shape should have rank 4")
    assert_true(result[0] == 2, "first dimension should be 2")
    assert_true(result[3] == 5, "last dimension should be 5")
    assert_true(result.numels() == 120, "shape should have 120 elements")


def test_shape_add_list() raises:
    var s = Shape(2, 3)
    var lst = List[Int]()
    lst.append(4)
    lst.append(5)
    var result = s + lst
    assert_true(result.rank() == 4, "result should have rank 4")
    assert_true(result[2] == 4, "third dimension should be 4")


def test_shape_radd_list() raises:
    var s = Shape(4, 5)
    var lst = List[Int]()
    lst.append(2)
    lst.append(3)
    var result = lst + s
    assert_true(result.rank() == 4, "result should have rank 4")
    assert_true(result[0] == 2, "first dimension should be 2")
    assert_true(result[3] == 5, "last dimension should be 5")


def test_shape_mul_scalar() raises:
    var s = Shape(2, 3)
    var result = s * 3
    assert_true(result.rank() == 6, "repeated shape should have rank 6")
    assert_true(result[0] == 2, "first dimension should be 2")
    assert_true(result[4] == 2, "dimension 4 should be 2")


def test_shape_rmul_scalar() raises:
    var s = Shape(2, 3)
    var result = 2 * s
    assert_true(result.rank() == 4, "repeated shape should have rank 4")


def test_shape_reverse() raises:
    var s = Shape(2, 3, 4, 5)
    var rev = s.reverse()
    assert_true(rev[0] == 5, "first dimension should be 5")
    assert_true(rev[3] == 2, "last dimension should be 2")


def test_shape_replace() raises:
    var s = Shape(2, 3, 4, 5)
    var result = s.replace(1, 10)
    assert_true(result[1] == 10, "second dimension should be 10")
    assert_true(result[0] == 2, "other dimensions unchanged")
    assert_true(result.numels() == 400, "numels should be updated")


def test_shape_permute() raises:
    var s = Shape(10, 20, 30, 40)
    var axes = IntArray(2, 0, 3, 1)
    var result = s.permute(axes)
    assert_true(result[0] == 30, "first dimension should be 30")
    assert_true(result[1] == 10, "second dimension should be 10")
    assert_true(result[2] == 40, "third dimension should be 40")
    assert_true(result[3] == 20, "fourth dimension should be 20")


def test_shape_count_axes_of_size() raises:
    var s = Shape(1, 2, 1, 3, 1)
    assert_true(s.count_axes_of_size(1) == 3, "should have 3 axes of size 1")
    assert_true(s.count_axes_of_size(2) == 1, "should have 1 axis of size 2")


def test_shape_indices_of_axes_with_size() raises:
    var s = Shape(1, 2, 1, 3, 1)
    var indices = s.indices_of_axes_with_size(1)
    assert_true(len(indices) == 3, "should find 3 axes")
    assert_true(indices[0] == 0, "first axis at index 0")
    assert_true(indices[1] == 2, "second axis at index 2")
    assert_true(indices[2] == 4, "third axis at index 4")


def test_shape_first_index() raises:
    var s = Shape(2, 3, 4)
    var idx = s.first_index()
    assert_true(len(idx) == 3, "first index should have 3 elements")
    assert_true(idx[0] == 0, "all elements should be 0")
    assert_true(idx[1] == 0, "all elements should be 0")
    assert_true(idx[2] == 0, "all elements should be 0")


def test_shape_compute_output_shape_reduce_all() raises:
    var s = Shape(2, 3, 4)
    var axes = IntArray()  # Empty = reduce all
    var result_no_keep = s.compute_output_shape(axes, keepdims=False)
    var result_keep = s.compute_output_shape(axes, keepdims=True)
    assert_true(
        result_no_keep.rank() == 0,
        "reduce all without keepdims should be scalar",
    )
    assert_true(
        result_keep.rank() == 3, "reduce all with keepdims should keep rank"
    )
    assert_true(result_keep[0] == 1, "all dims should be 1")


def test_shape_compute_output_shape_single_axis() raises:
    var s = Shape(2, 3, 4)
    var axes = IntArray.with_capacity(1)
    axes.append(1)  # Reduce middle axis
    var result_no_keep = s.compute_output_shape(axes, keepdims=False)
    var result_keep = s.compute_output_shape(axes, keepdims=True)
    assert_true(
        result_no_keep.rank() == 2, "result without keepdims should have rank 2"
    )
    assert_true(result_no_keep[0] == 2, "first dim unchanged")
    assert_true(result_no_keep[1] == 4, "third dim unchanged")
    assert_true(
        result_keep.rank() == 3, "result with keepdims should have rank 3"
    )
    assert_true(result_keep[1] == 1, "reduced dim should be 1")


def test_shape_compute_output_shape_multiple_axes() raises:
    var s = Shape(2, 3, 4, 5)
    var axes = IntArray(1, 3)  # Reduce axes 1 and 3
    var result_no_keep = s.compute_output_shape(axes, keepdims=False)
    var result_keep = s.compute_output_shape(axes, keepdims=True)
    assert_true(
        result_no_keep.rank() == 2, "result without keepdims should have rank 2"
    )
    assert_true(result_no_keep[0] == 2, "first dim unchanged")
    assert_true(result_no_keep[1] == 4, "third dim unchanged")
    assert_true(
        result_keep.rank() == 4, "result with keepdims should have rank 4"
    )
    assert_true(result_keep[1] == 1, "reduced dims should be 1")
    assert_true(result_keep[3] == 1, "reduced dims should be 1")


def test_shape_of() raises:
    var s = Shape.of(2, 3, 4)
    assert_true(s.rank() == 3, "Shape.of should create rank 3 shape")
    assert_true(s[1] == 3, "second dimension should be 3")


def test_shape_product() raises:
    var s = Shape(2, 3, 4)
    assert_true(Shape.product(s) == 24, "product should be 24")
    var scalar = Shape()
    assert_true(Shape.product(scalar) == 1, "scalar product should be 1")


def test_slice_shape() raises:
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


def test_negative_indices() raises:
    shape = Shape([1, 2, 3])
    assert_true(
        shape[-1] == 3 and shape[-2] == 2 and shape[-3] == 1,
        "Shape negative indices assertion failed",
    )


def test_slice_from() raises:
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


def test_reverse() raises:
    shape = Shape.of(1, 2, 3)
    assert_true(
        shape.reverse() == Shape.of(3, 2, 1), "Shape reversal assertion failed"
    )


def test_equivalence() raises:
    assert_true(Shape(IntArray(1, 4)) == Shape.of(1, 4), "Not equivalent")


def test_empty_shape() raises:
    shape = Shape()
    for each in shape:
        assert_true(
            IntArray() == each, "Empty shape iteration assertion failed"
        )
    tensor = Tensor[DType.bool](shape)
    tensor[IntArray()] = True
    assert_true(
        tensor[IntArray()] == True, "Scalar tensor get assertion 2 failed"
    )


def test_replace() raises:
    shape = Shape.of(3, 4, 2)
    shape = shape.replace(2, 5)
    assert_true(shape == Shape.of(3, 4, 5), "replace assertion failed")


def test_index_iter() raises:
    shape = Shape.of(1)
    for each in shape:
        assert_true(
            IntArray(0) == each,
            "Unit shape(Shape.of(1)) index iteration assertion failed",
        )
    shape = Shape.of(2, 1)
    indices = shape.__iter__()
    assert_true(
        IntArray(0, 0) == indices.__next__()
        and IntArray(1, 0) == indices.__next__(),
        "Shape(2,1) iteration assertion failed",
    )


def test_shape_as_intlist() raises:
    shape = Shape.of(2, 4, 5)
    fa = shape.intarray()
    assert_true(
        fa[0] == 2 and fa[1] == 4 and fa[2] == 5,
        "Shape to IntArray assertion failed",
    )


def test_zip_reversed() raises:
    shape1 = Shape.of(1, 2, 3, 4, 5)
    shape2 = Shape.of(6)
    rzipped = shape1.intarray().zip_reversed(shape2.intarray())
    for each in rzipped:
        assert_true(
            each[0] == 5 and each[1] == 6, "zip_reversed assertion failed"
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
