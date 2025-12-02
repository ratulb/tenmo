from tenmo import Tensor
from shapes import Shape

from testing import assert_true, assert_raises
from intarray import IntArray

# ============================================
# SHAPE TESTS
# ============================================

fn test_shape_default_constructor() raises:
    print("test_shape_default_constructor")
    var s = Shape()
    assert_true(s.rank() == 0, "default constructor should create scalar")
    assert_true(s.num_elements() == 1, "scalar should have 1 element")
    print("test_shape_default_constructor passed")

fn test_shape_void() raises:
    print("test_shape_void")
    var s = Shape.Void()
    assert_true(s.rank() == 0, "Void() should create scalar")
    assert_true(s.numels() == 1, "scalar should have 1 element")
    print("test_shape_void passed")

fn test_shape_unit() raises:
    print("test_shape_unit")
    var s = Shape.Unit()
    assert_true(s.rank() == 1, "Unit() should have rank 1")
    assert_true(s[0] == 1, "Unit() dimension should be 1")
    assert_true(s.numels() == 1, "Unit() should have 1 element")
    print("test_shape_unit passed")

fn test_shape_variadic_constructor() raises:
    print("test_shape_variadic_constructor")
    var s = Shape(2, 3, 4)
    assert_true(s.rank() == 3, "variadic constructor should create rank 3 shape")
    assert_true(s[0] == 2, "first dimension should be 2")
    assert_true(s[1] == 3, "second dimension should be 3")
    assert_true(s[2] == 4, "third dimension should be 4")
    assert_true(s.numels() == 24, "shape should have 24 elements")
    print("test_shape_variadic_constructor passed")

fn test_shape_list_constructor() raises:
    print("test_shape_list_constructor")
    var lst = List[Int]()
    lst.append(5)
    lst.append(6)
    lst.append(7)
    var s = Shape(lst)
    assert_true(s.rank() == 3, "list constructor should create rank 3 shape")
    assert_true(s.numels() == 210, "shape should have 210 elements")
    print("test_shape_list_constructor passed")

fn test_shape_intarray_constructor() raises:
    print("test_shape_intarray_constructor")
    var arr = IntArray(3, 4, 5)
    var s = Shape(arr)
    assert_true(s.rank() == 3, "IntArray constructor should create rank 3 shape")
    assert_true(s[1] == 4, "second dimension should be 4")
    print("test_shape_intarray_constructor passed")

fn test_shape_getitem() raises:
    print("test_shape_getitem")
    var s = Shape(10, 20, 30, 40)
    assert_true(s[0] == 10, "first dimension should be 10")
    assert_true(s[2] == 30, "third dimension should be 30")
    assert_true(s[-1] == 40, "last dimension should be 40")
    assert_true(s[-2] == 30, "second to last should be 30")
    print("test_shape_getitem passed")

fn test_shape_slice() raises:
    print("test_shape_slice")
    var s = Shape(10, 20, 30, 40, 50)
    var sliced = s[1:4]
    assert_true(sliced.rank() == 3, "sliced shape should have rank 3")
    assert_true(sliced[0] == 20, "first element should be 20")
    assert_true(sliced[2] == 40, "last element should be 40")
    print("test_shape_slice passed")

fn test_shape_eq() raises:
    print("test_shape_eq")
    var s1 = Shape(2, 3, 4)
    var s2 = Shape(2, 3, 4)
    var s3 = Shape(2, 3, 5)
    assert_true(s1 == s2, "s1 should equal s2")
    assert_true(not (s1 == s3), "s1 should not equal s3")
    print("test_shape_eq passed")

fn test_shape_eq_list() raises:
    print("test_shape_eq_list")
    var s = Shape(2, 3, 4)
    var lst = List[Int]()
    lst.append(2)
    lst.append(3)
    lst.append(4)
    assert_true(s == lst, "shape should equal list")
    print("test_shape_eq_list passed")

fn test_shape_ne() raises:
    print("test_shape_ne")
    var s1 = Shape(2, 3)
    var s2 = Shape(2, 4)
    assert_true(s1 != s2, "s1 should not equal s2")
    print("test_shape_ne passed")

fn test_shape_str() raises:
    print("test_shape_str")
    var s = Shape(2, 3, 4)
    var str_repr = s.__str__()
    assert_true(str_repr == "(2, 3, 4)", "string should be '(2, 3, 4)'")
    print("test_shape_str passed")

fn test_shape_tolist() raises:
    print("test_shape_tolist")
    var s = Shape(5, 10, 15)
    var lst = s.tolist()
    assert_true(len(lst) == 3, "list should have 3 elements")
    assert_true(lst[1] == 10, "second element should be 10")
    print("test_shape_tolist passed")

fn test_shape_add_shape() raises:
    print("test_shape_add_shape")
    var s1 = Shape(2, 3)
    var s2 = Shape(4, 5)
    var result = s1 + s2
    assert_true(result.rank() == 4, "concatenated shape should have rank 4")
    assert_true(result[0] == 2, "first dimension should be 2")
    assert_true(result[3] == 5, "last dimension should be 5")
    assert_true(result.numels() == 120, "shape should have 120 elements")
    print("test_shape_add_shape passed")

fn test_shape_add_list() raises:
    print("test_shape_add_list")
    var s = Shape(2, 3)
    var lst = List[Int]()
    lst.append(4)
    lst.append(5)
    var result = s + lst
    assert_true(result.rank() == 4, "result should have rank 4")
    assert_true(result[2] == 4, "third dimension should be 4")
    print("test_shape_add_list passed")

fn test_shape_radd_list() raises:
    print("test_shape_radd_list")
    var s = Shape(4, 5)
    var lst = List[Int]()
    lst.append(2)
    lst.append(3)
    var result = lst + s
    assert_true(result.rank() == 4, "result should have rank 4")
    assert_true(result[0] == 2, "first dimension should be 2")
    assert_true(result[3] == 5, "last dimension should be 5")
    print("test_shape_radd_list passed")

fn test_shape_mul_scalar() raises:
    print("test_shape_mul_scalar")
    var s = Shape(2, 3)
    var result = s * 3
    assert_true(result.rank() == 6, "repeated shape should have rank 6")
    assert_true(result[0] == 2, "first dimension should be 2")
    assert_true(result[4] == 2, "dimension 4 should be 2")
    print("test_shape_mul_scalar passed")

fn test_shape_rmul_scalar() raises:
    print("test_shape_rmul_scalar")
    var s = Shape(2, 3)
    var result = 2 * s
    assert_true(result.rank() == 4, "repeated shape should have rank 4")
    print("test_shape_rmul_scalar passed")

fn test_shape_reverse() raises:
    print("test_shape_reverse")
    var s = Shape(2, 3, 4, 5)
    var rev = s.reverse()
    assert_true(rev[0] == 5, "first dimension should be 5")
    assert_true(rev[3] == 2, "last dimension should be 2")
    print("test_shape_reverse passed")

fn test_shape_replace() raises:
    print("test_shape_replace")
    var s = Shape(2, 3, 4, 5)
    var result = s.replace(1, 10)
    assert_true(result[1] == 10, "second dimension should be 10")
    assert_true(result[0] == 2, "other dimensions unchanged")
    assert_true(result.numels() == 400, "numels should be updated")
    print("test_shape_replace passed")

fn test_shape_permute() raises:
    print("test_shape_permute")
    var s = Shape(10, 20, 30, 40)
    var axes = IntArray(2, 0, 3, 1)
    var result = s.permute(axes)
    assert_true(result[0] == 30, "first dimension should be 30")
    assert_true(result[1] == 10, "second dimension should be 10")
    assert_true(result[2] == 40, "third dimension should be 40")
    assert_true(result[3] == 20, "fourth dimension should be 20")
    print("test_shape_permute passed")

fn test_shape_count_axes_of_size() raises:
    print("test_shape_count_axes_of_size")
    var s = Shape(1, 2, 1, 3, 1)
    assert_true(s.count_axes_of_size(1) == 3, "should have 3 axes of size 1")
    assert_true(s.count_axes_of_size(2) == 1, "should have 1 axis of size 2")
    print("test_shape_count_axes_of_size passed")

fn test_shape_indices_of_axes_with_size() raises:
    print("test_shape_indices_of_axes_with_size")
    var s = Shape(1, 2, 1, 3, 1)
    var indices = s.indices_of_axes_with_size(1)
    assert_true(len(indices) == 3, "should find 3 axes")
    assert_true(indices[0] == 0, "first axis at index 0")
    assert_true(indices[1] == 2, "second axis at index 2")
    assert_true(indices[2] == 4, "third axis at index 4")
    print("test_shape_indices_of_axes_with_size passed")

fn test_shape_first_index() raises:
    print("test_shape_first_index")
    var s = Shape(2, 3, 4)
    var idx = s.first_index()
    assert_true(len(idx) == 3, "first index should have 3 elements")
    assert_true(idx[0] == 0, "all elements should be 0")
    assert_true(idx[1] == 0, "all elements should be 0")
    assert_true(idx[2] == 0, "all elements should be 0")
    print("test_shape_first_index passed")

fn test_shape_compute_output_shape_reduce_all() raises:
    print("test_shape_compute_output_shape_reduce_all")
    var s = Shape(2, 3, 4)
    var axes = IntArray()  # Empty = reduce all
    var result_no_keep = s.compute_output_shape(axes, keepdims=False)
    var result_keep = s.compute_output_shape(axes, keepdims=True)
    assert_true(result_no_keep.rank() == 0, "reduce all without keepdims should be scalar")
    assert_true(result_keep.rank() == 3, "reduce all with keepdims should keep rank")
    assert_true(result_keep[0] == 1, "all dims should be 1")
    print("test_shape_compute_output_shape_reduce_all passed")


fn test_shape_compute_output_shape_single_axis() raises:
    print("test_shape_compute_output_shape_single_axis")
    var s = Shape(2, 3, 4)
    var axes = IntArray.with_capacity(1)
    axes.append(1)  # Reduce middle axis
    var result_no_keep = s.compute_output_shape(axes, keepdims=False)
    var result_keep = s.compute_output_shape(axes, keepdims=True)
    assert_true(result_no_keep.rank() == 2, "result without keepdims should have rank 2")
    assert_true(result_no_keep[0] == 2, "first dim unchanged")
    assert_true(result_no_keep[1] == 4, "third dim unchanged")
    assert_true(result_keep.rank() == 3, "result with keepdims should have rank 3")
    assert_true(result_keep[1] == 1, "reduced dim should be 1")
    print("test_shape_compute_output_shape_single_axis passed")



fn test_shape_compute_output_shape_single_axis_1() raises:
    print("test_shape_compute_output_shape_single_axis")
    var s = Shape(2, 3, 4)
    var axes = IntArray(1)  # Reduce middle axis
    axes.append(1)  # Reduce middle axis
    var result_no_keep = s.compute_output_shape(axes, keepdims=False)
    var result_keep = s.compute_output_shape(axes, keepdims=True)
    assert_true(result_no_keep.rank() == 2, "result without keepdims should have rank 2")
    assert_true(result_no_keep[0] == 2, "first dim unchanged")
    assert_true(result_no_keep[1] == 4, "third dim unchanged")
    assert_true(result_keep.rank() == 3, "result with keepdims should have rank 3")
    assert_true(result_keep[1] == 1, "reduced dim should be 1")
    print("test_shape_compute_output_shape_single_axis passed")

fn test_shape_compute_output_shape_multiple_axes() raises:
    print("test_shape_compute_output_shape_multiple_axes")
    var s = Shape(2, 3, 4, 5)
    var axes = IntArray(1, 3)  # Reduce axes 1 and 3
    var result_no_keep = s.compute_output_shape(axes, keepdims=False)
    var result_keep = s.compute_output_shape(axes, keepdims=True)
    assert_true(result_no_keep.rank() == 2, "result without keepdims should have rank 2")
    assert_true(result_no_keep[0] == 2, "first dim unchanged")
    assert_true(result_no_keep[1] == 4, "third dim unchanged")
    assert_true(result_keep.rank() == 4, "result with keepdims should have rank 4")
    assert_true(result_keep[1] == 1, "reduced dims should be 1")
    assert_true(result_keep[3] == 1, "reduced dims should be 1")
    print("test_shape_compute_output_shape_multiple_axes passed")

fn test_shape_of() raises:
    print("test_shape_of")
    var s = Shape.of(2, 3, 4)
    assert_true(s.rank() == 3, "Shape.of should create rank 3 shape")
    assert_true(s[1] == 3, "second dimension should be 3")
    print("test_shape_of passed")

fn test_shape_product() raises:
    print("test_shape_product")
    var s = Shape(2, 3, 4)
    assert_true(Shape.product(s) == 24, "product should be 24")
    var scalar = Shape()
    assert_true(Shape.product(scalar) == 1, "scalar product should be 1")
    print("test_shape_product passed")

fn run_all_shape_tests() raises:
    print("\n" + "="*60)
    print("RUNNING SHAPE TESTS")
    print("="*60)

    test_shape_default_constructor()
    test_shape_void()
    test_shape_unit()
    test_shape_variadic_constructor()
    test_shape_list_constructor()
    test_shape_intarray_constructor()
    test_shape_getitem()
    test_shape_slice()
    test_shape_eq()
    test_shape_eq_list()
    test_shape_ne()
    test_shape_str()
    test_shape_tolist()
    test_shape_add_shape()
    test_shape_add_list()
    test_shape_radd_list()
    test_shape_mul_scalar()
    test_shape_rmul_scalar()
    test_shape_reverse()
    test_shape_replace()
    test_shape_permute()
    test_shape_count_axes_of_size()
    test_shape_indices_of_axes_with_size()
    test_shape_first_index()
    test_shape_compute_output_shape_reduce_all()
    test_shape_compute_output_shape_multiple_axes()
    test_shape_of()
    test_shape_product()
    test_shape_compute_output_shape_single_axis()

    print("\n" + "="*60)
    print("ALL SHAPE TESTS PASSED ✓")
    print("="*60)


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
    assert_true(Shape(IntArray(1, 4)) == Shape.of(1, 4), "Not equivalent")


fn test_empty_shape() raises:
    print("test_empty_shape")
    shape = Shape()
    for each in shape:
        assert_true(IntArray() == each, "Empty shape iteration assertion failed")
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



fn test_shape_as_intlist() raises:
    print("test_shape_as_intlist")
    shape = Shape.of(2, 4, 5)
    fa = shape.intarray()
    assert_true(
        fa[0] == 2 and fa[1] == 4 and fa[2] == 5,
        "Shape to IntArray assertion failed",
    )


fn test_zip_reversed() raises:
    print("test_zip_reversed")
    shape1 = Shape.of(1, 2, 3, 4, 5)
    shape2 = Shape.of(6)
    rzipped = shape1.intarray().zip_reversed(shape2.intarray())
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

    run_all_shape_tests()
