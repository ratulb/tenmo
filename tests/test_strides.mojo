from shapes import Shape
from strides import Strides
from testing import assert_true
from intarray import IntArray

# ============================================
# STRIDES TESTS
# ============================================


fn test_strides_default_constructor() raises:
    print("test_strides_default_constructor")
    var s = Strides()
    assert_true(len(s) == 0, "default constructor should create empty strides")
    print("test_strides_default_constructor passed")


fn test_strides_zero() raises:
    print("test_strides_zero")
    var s = Strides.Zero()
    assert_true(len(s) == 0, "Zero() should create empty strides")
    print("test_strides_zero passed")


fn test_strides_variadic_constructor() raises:
    print("test_strides_variadic_constructor")
    var s = Strides(1, 2, 3, 4)
    assert_true(
        len(s) == 4, "variadic constructor should create strides of length 4"
    )
    assert_true(s[0] == 1, "first element should be 1")
    assert_true(s[3] == 4, "last element should be 4")
    print("test_strides_variadic_constructor passed")


fn test_strides_list_constructor() raises:
    print("test_strides_list_constructor")
    var lst = List[Int]()
    lst.append(5)
    lst.append(10)
    lst.append(15)
    var s = Strides(lst)
    assert_true(
        len(s) == 3, "list constructor should create strides of length 3"
    )
    assert_true(s[1] == 10, "second element should be 10")
    print("test_strides_list_constructor passed")


fn test_strides_intarray_constructor() raises:
    print("test_strides_intarray_constructor")
    var arr = IntArray(2, 4, 6)
    var s = Strides(arr)
    assert_true(
        len(s) == 3, "IntArray constructor should create strides of length 3"
    )
    assert_true(s[2] == 6, "third element should be 6")
    print("test_strides_intarray_constructor passed")


fn test_strides_getitem() raises:
    print("test_strides_getitem")
    var s = Strides(10, 20, 30, 40)
    assert_true(s[0] == 10, "first element should be 10")
    assert_true(s[2] == 30, "third element should be 30")
    assert_true(s[-1] == 40, "last element should be 40")
    print("test_strides_getitem passed")


fn test_strides_setitem() raises:
    print("test_strides_setitem")
    var s = Strides(1, 2, 3, 4)
    s[0] = 100
    s[2] = 200
    assert_true(s[0] == 100, "s[0] should be 100")
    assert_true(s[2] == 200, "s[2] should be 200")
    print("test_strides_setitem passed")


fn test_strides_slice() raises:
    print("test_strides_slice")
    var s = Strides(10, 20, 30, 40, 50)
    var sliced = s[1:4]
    assert_true(len(sliced) == 3, "sliced strides should have 3 elements")
    assert_true(sliced[0] == 20, "first element should be 20")
    assert_true(sliced[2] == 40, "last element should be 40")
    print("test_strides_slice passed")


fn test_strides_eq() raises:
    print("test_strides_eq")
    var s1 = Strides(1, 2, 3)
    var s2 = Strides(1, 2, 3)
    var s3 = Strides(1, 2, 4)
    assert_true(s1 == s2, "s1 should equal s2")
    assert_true(not (s1 == s3), "s1 should not equal s3")
    print("test_strides_eq passed")


fn test_strides_str() raises:
    print("test_strides_str")
    var s = Strides(10, 20, 30)
    var str_repr = s.__str__()
    assert_true(str_repr == "(10, 20, 30)", "string should be '(10, 20, 30)'")
    print("test_strides_str passed")


fn test_strides_tolist() raises:
    print("test_strides_tolist")
    var s = Strides(5, 10, 15)
    var lst = s.tolist()
    assert_true(len(lst) == 3, "list should have 3 elements")
    assert_true(lst[1] == 10, "second element should be 10")
    print("test_strides_tolist passed")


fn test_strides_permute() raises:
    print("test_strides_permute")
    var s = Strides(10, 20, 30, 40)
    var axes = IntArray(2, 0, 3, 1)
    var result = s.permute(axes)
    assert_true(result[0] == 30, "first element should be 30")
    assert_true(result[1] == 10, "second element should be 10")
    assert_true(result[2] == 40, "third element should be 40")
    assert_true(result[3] == 20, "fourth element should be 20")
    print("test_strides_permute passed")


fn test_strides_default() raises:
    print("test_strides_default")
    var shape = Shape(2, 3, 4)
    var s = Strides.default(shape)
    assert_true(len(s) == 3, "strides should have 3 elements")
    assert_true(s[0] == 12, "stride[0] should be 12 (3*4)")
    assert_true(s[1] == 4, "stride[1] should be 4")
    assert_true(s[2] == 1, "stride[2] should be 1")
    print("test_strides_default passed")


fn test_strides_default_scalar() raises:
    print("test_strides_default_scalar")
    var shape = Shape()
    var s = Strides.default(shape)
    assert_true(len(s) == 0, "scalar should have empty strides")
    print("test_strides_default_scalar passed")


fn test_strides_is_contiguous() raises:
    print("test_strides_is_contiguous")
    var shape = Shape(2, 3, 4)
    var s1 = Strides(12, 4, 1)
    var s2 = Strides(12, 1, 4)
    assert_true(s1.is_contiguous(shape), "s1 should be contiguous")
    assert_true(not s2.is_contiguous(shape), "s2 should not be contiguous")
    print("test_strides_is_contiguous passed")


fn test_strides_is_contiguous_scalar() raises:
    print("test_strides_is_contiguous_scalar")
    var shape = Shape()
    var s = Strides()
    assert_true(s.is_contiguous(shape), "scalar is trivially contiguous")
    print("test_strides_is_contiguous_scalar passed")


fn test_strides_with_capacity() raises:
    print("test_strides_with_capacity")
    var s = Strides.with_capacity(10)
    assert_true(len(s) == 0, "with_capacity should start with size 0")
    print("test_strides_with_capacity passed")


fn test_compute_default_strides() raises:
    shape = Shape.of(2, 3, 4)
    strides = Strides.default(shape)
    assert_true(
        strides == Strides(12, 4, 1),
        "stride compute assertion 1 failed",
    )


fn run_all_strides_tests() raises:
    print("\n" + "=" * 60)
    print("RUNNING STRIDES TESTS")
    print("=" * 60)

    test_strides_default_constructor()
    test_strides_zero()
    test_strides_variadic_constructor()
    test_strides_list_constructor()
    test_strides_intarray_constructor()
    test_strides_getitem()
    test_strides_setitem()
    test_strides_slice()
    test_strides_eq()
    test_strides_str()
    test_strides_tolist()
    test_strides_permute()
    test_strides_default()
    test_strides_default_scalar()
    test_strides_is_contiguous()
    test_strides_is_contiguous_scalar()
    test_strides_with_capacity()

    print("\n" + "=" * 60)
    print("ALL STRIDES TESTS PASSED ✓")
    print("=" * 60)


fn main() raises:
    print("Running strides tests")
    test_compute_default_strides()
    run_all_strides_tests()
