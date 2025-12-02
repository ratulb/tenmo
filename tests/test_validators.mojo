from shapes import Shape
from validators import Validator
from testing import assert_true
from intarray import IntArray


# ============================================
# EXTENSIVE TESTS FOR VALIDATOR METHODS
# ============================================

fn test_validate_axes_empty_reduce_all() raises:
    print("test_validate_axes_empty_reduce_all")
    var shape = Shape(2, 3, 4)
    var axes = IntArray()
    var result = Validator.validate_and_normalize_axes(shape, axes)
    assert_true(len(result) == 3, "should return all axes")
    assert_true(result[0] == 0, "first axis should be 0")
    assert_true(result[1] == 1, "second axis should be 1")
    assert_true(result[2] == 2, "third axis should be 2")
    print("test_validate_axes_empty_reduce_all passed")

fn test_validate_axes_single_axis() raises:
    print("test_validate_axes_single_axis")
    var shape = Shape(2, 3, 4, 5)
    var axes = IntArray.with_capacity(1)
    axes.append(2)
    var result = Validator.validate_and_normalize_axes(shape, axes)
    assert_true(len(result) == 1, "should return 1 axis")
    assert_true(result[0] == 2, "axis should be 2")
    print("test_validate_axes_single_axis passed")

fn test_validate_axes_multiple_axes() raises:
    print("test_validate_axes_multiple_axes")
    var shape = Shape(2, 3, 4, 5, 6)
    var axes = IntArray(1, 3, 4)
    var result = Validator.validate_and_normalize_axes(shape, axes)
    assert_true(len(result) == 3, "should return 3 axes")
    assert_true(result[0] == 1, "first axis should be 1")
    assert_true(result[1] == 3, "second axis should be 3")
    assert_true(result[2] == 4, "third axis should be 4")
    print("test_validate_axes_multiple_axes passed")

fn test_validate_axes_negative_indices() raises:
    print("test_validate_axes_negative_indices")
    var shape = Shape(2, 3, 4)
    var axes = IntArray(-1, -2)
    var result = Validator.validate_and_normalize_axes(shape, axes)
    assert_true(len(result) == 2, "should return 2 axes")
    assert_true(result[0] == 1, "normalized -2 should be 1")
    assert_true(result[1] == 2, "normalized -1 should be 2")
    print("test_validate_axes_negative_indices passed")

fn test_validate_axes_unordered() raises:
    print("test_validate_axes_unordered")
    var shape = Shape(2, 3, 4, 5)
    var axes = IntArray(3, 0, 2)
    var result = Validator.validate_and_normalize_axes(shape, axes, ordered=True)
    assert_true(result[0] == 0, "should be sorted: first")
    assert_true(result[1] == 2, "should be sorted: second")
    assert_true(result[2] == 3, "should be sorted: third")
    print("test_validate_axes_unordered passed")

fn test_validate_axes_no_sort() raises:
    print("test_validate_axes_no_sort")
    var shape = Shape(2, 3, 4, 5)
    var axes = IntArray(3, 0, 2)
    var result = Validator.validate_and_normalize_axes(shape, axes, ordered=False)
    assert_true(result[0] == 3, "should maintain order: first")
    assert_true(result[1] == 0, "should maintain order: second")
    assert_true(result[2] == 2, "should maintain order: third")
    print("test_validate_axes_no_sort passed")

fn test_validate_axes_fill_missing() raises:
    print("test_validate_axes_fill_missing")
    var shape = Shape(2, 3, 4, 5)
    var axes = IntArray(1, 3)
    var result = Validator.validate_and_normalize_axes(shape, axes, ordered=True, fill_missing=True)
    assert_true(len(result) == 4, "should have all axes")
    assert_true(result[0] == 0, "specified axis first")
    assert_true(result[1] == 1, "specified axis second")
    assert_true(result[2] == 2, "missing axis 0")
    assert_true(result[3] == 3, "missing axis 2")
    print("test_validate_axes_fill_missing passed")

fn test_validate_axes_scalar_empty() raises:
    print("test_validate_axes_scalar_empty")
    var shape = Shape()
    var axes = IntArray()
    var result = Validator.validate_and_normalize_axes(shape, axes)
    assert_true(len(result) == 0, "scalar with empty axes should return empty")
    print("test_validate_axes_scalar_empty passed")

fn test_validate_axes_scalar_minus_one() raises:
    print("test_validate_axes_scalar_minus_one")
    var shape = Shape()
    var axes = IntArray(-1)
    var result = Validator.validate_and_normalize_axes(shape, axes)
    assert_true(len(result) == 0, "scalar with [-1] should return empty")
    print("test_validate_axes_scalar_minus_one passed")

fn test_validate_axes_all_axes() raises:
    print("test_validate_axes_all_axes")
    var shape = Shape(2, 3, 4)
    var axes = IntArray(0, 1, 2)
    var result = Validator.validate_and_normalize_axes(shape, axes)
    assert_true(len(result) == 3, "should have all 3 axes")
    print("test_validate_axes_all_axes passed")


fn test_reshape_same_shape() raises:
    print("test_reshape_same_shape")
    var current = Shape(2, 3, 4)
    var newdims = IntArray(2, 3, 4)
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result == Shape(2, 3, 4), "should return same shape")
    print("test_reshape_same_shape passed")

fn test_reshape_flatten() raises:
    print("test_reshape_flatten")
    var current = Shape(2, 3, 4)
    var newdims = IntArray(24)
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result == Shape(24), "should flatten to (24,)")
    print("test_reshape_flatten passed")

fn test_reshape_infer_dimension() raises:
    print("test_reshape_infer_dimension")
    var current = Shape(2, 3, 4)
    var newdims = IntArray(2, -1)
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result == Shape(2, 12), "should infer second dimension as 12")
    print("test_reshape_infer_dimension passed")

fn test_reshape_infer_middle() raises:
    print("test_reshape_infer_middle")
    var current = Shape(2, 3, 4)
    var newdims = IntArray(2, -1, 2)
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result == Shape(2, 6, 2), "should infer middle dimension as 6")
    print("test_reshape_infer_middle passed")

fn test_reshape_infer_first() raises:
    print("test_reshape_infer_first")
    var current = Shape(2, 3, 4)
    var newdims = IntArray(-1, 12)
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result == Shape(2, 12), "should infer first dimension as 2")
    print("test_reshape_infer_first passed")

fn test_reshape_scalar_to_scalar() raises:
    print("test_reshape_scalar_to_scalar")
    var current = Shape()
    var newdims = IntArray()
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result.rank() == 0, "scalar to scalar")
    print("test_reshape_scalar_to_scalar passed")

fn test_reshape_scalar_to_one() raises:
    print("test_reshape_scalar_to_one")
    var current = Shape()
    var newdims = IntArray(1)
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result == Shape(1), "scalar to (1,)")
    print("test_reshape_scalar_to_one passed")

fn test_reshape_one_to_scalar() raises:
    print("test_reshape_one_to_scalar")
    var current = Shape(1)
    var newdims = IntArray()
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result.rank() == 0, "(1,) to scalar")
    print("test_reshape_one_to_scalar passed")

fn test_reshape_multidim_to_one() raises:
    print("test_reshape_multidim_to_one")
    var current = Shape(1, 1, 1)
    var newdims = IntArray(1)
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result == Shape(1), "(1,1,1) to (1,)")
    print("test_reshape_multidim_to_one passed")

fn test_reshape_complex() raises:
    print("test_reshape_complex")
    var current = Shape(6, 4, 5)
    var newdims = IntArray(3, 2, -1, 5)
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result == Shape(3, 2, 4, 5), "complex reshape with inference")
    print("test_reshape_complex passed")

fn test_reshape_2d_to_3d() raises:
    print("test_reshape_2d_to_3d")
    var current = Shape(6, 8)
    var newdims = IntArray(2, 3, 8)
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result == Shape(2, 3, 8), "2D to 3D reshape")
    print("test_reshape_2d_to_3d passed")

fn test_reshape_3d_to_2d() raises:
    print("test_reshape_3d_to_2d")
    var current = Shape(2, 3, 4)
    var newdims = IntArray(6, 4)
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result == Shape(6, 4), "3D to 2D reshape")
    print("test_reshape_3d_to_2d passed")

fn test_reshape_infer_last() raises:
    print("test_reshape_infer_last")
    var current = Shape(24)
    var newdims = IntArray(2, 3, -1)
    var result = Validator.validate_and_construct_new_shape(current, newdims)
    assert_true(result == Shape(2, 3, 4), "infer last dimension")
    print("test_reshape_infer_last passed")


# ============================================
# CONSOLIDATED TEST RUNNERS
# ============================================

fn run_validate_axes_tests() raises:
    print("\n" + "="*60)
    print("RUNNING Validator.validate_and_normalize_axes TESTS")
    print("="*60)

    test_validate_axes_empty_reduce_all()
    test_validate_axes_single_axis()
    test_validate_axes_multiple_axes()
    test_validate_axes_negative_indices()
    test_validate_axes_unordered()
    test_validate_axes_no_sort()
    test_validate_axes_fill_missing()
    test_validate_axes_scalar_empty()
    test_validate_axes_scalar_minus_one()
    test_validate_axes_all_axes()

    print("\n" + "="*60)
    print("ALL VALIDATE_AXES TESTS PASSED ✓")
    print("="*60)


fn run_validate_reshape_tests() raises:
    print("\n" + "="*60)
    print("RUNNING VALIDATE_AND_CONSTRUCT_NEW_SHAPE TESTS")
    print("="*60)

    test_reshape_same_shape()
    test_reshape_flatten()
    test_reshape_infer_dimension()
    test_reshape_infer_middle()
    test_reshape_infer_first()
    test_reshape_scalar_to_scalar()
    test_reshape_scalar_to_one()
    test_reshape_one_to_scalar()
    test_reshape_multidim_to_one()
    test_reshape_complex()
    test_reshape_2d_to_3d()
    test_reshape_3d_to_2d()
    test_reshape_infer_last()

    print("\n" + "="*60)
    print("ALL VALIDATE_RESHAPE TESTS PASSED ✓")
    print("="*60)


fn run_all_validator_tests() raises:
    """Run all validator tests."""
    print("\n" + "#"*60)
    print("# VALIDATOR METHODS TEST SUITE")
    print("#"*60)

    run_validate_axes_tests()
    run_validate_reshape_tests()

    print("\n" + "#"*60)
    print("# ALL VALIDATOR TESTS PASSED SUCCESSFULLY ✓✓✓")
    print("#"*60 + "\n")


fn test_validate_and_normalize_axes() raises:
    print("test_validate_and_normalize_axes")
    shape = Shape([2, 3, 4])
    axes = Validator.validate_and_normalize_axes(shape, IntArray())
    print("axes: ", axes)
    assert_true(
        axes == IntArray(0, 1, 2), "Assertion failed for empty axes list"
    )
    axes = Validator.validate_and_normalize_axes(Shape(), IntArray())
    assert_true(
        axes == IntArray(), "Assertion failed for empty shape and empy axes list"
    )
    axes = Validator.validate_and_normalize_axes(
        shape, IntArray(-1, -2), ordered=False, fill_missing=True
    )
    assert_true(
        axes == IntArray(0, 2, 1),
        "Assertion failed for ordered False and fill missing",
    )




fn test_validate_new_shape() raises:
    print("test_validate_new_shape")
    curr_dims = Shape(IntArray([3, 4, 5]))
    new_dims = IntArray([2, -1, 10])
    concrete_shape = Validator.validate_and_construct_new_shape(
        curr_dims, new_dims
    )
    assert_true(
        concrete_shape == Shape.of(2, 3, 10),
        "validate_new_shape assertion 1 failed",
    )
    new_dims = IntArray([-1])
    concrete_shape = Validator.validate_and_construct_new_shape(
        curr_dims, new_dims
    )
    assert_true(
        concrete_shape == Shape.of(60), "validate_new_shape assertion 2 failed"
    )


fn main() raises:
    test_validate_new_shape()
    test_validate_and_normalize_axes()
    run_all_validator_tests()
