from tenmo.shapes import Shape
from tenmo.strides import Strides
from std.testing import assert_true, TestSuite
from tenmo.intarray import IntArray

# ============================================
# STRIDES TESTS
# ============================================


def test_strides_default_constructor() raises:
    var s = Strides()
    assert_true(len(s) == 0, "default constructor should create empty strides")


def test_strides_zero() raises:
    var s = Strides.Zero()
    assert_true(len(s) == 0, "Zero() should create empty strides")


def test_strides_variadic_constructor() raises:
    var s = Strides(1, 2, 3, 4)
    assert_true(
        len(s) == 4, "variadic constructor should create strides of length 4"
    )
    assert_true(s[0] == 1, "first element should be 1")
    assert_true(s[3] == 4, "last element should be 4")


def test_strides_list_constructor() raises:
    var lst = List[Int]()
    lst.append(5)
    lst.append(10)
    lst.append(15)
    var s = Strides(lst)
    assert_true(
        len(s) == 3, "list constructor should create strides of length 3"
    )
    assert_true(s[1] == 10, "second element should be 10")


def test_strides_intarray_constructor() raises:
    var arr = IntArray(2, 4, 6)
    var s = Strides(arr)
    assert_true(
        len(s) == 3, "IntArray constructor should create strides of length 3"
    )
    assert_true(s[2] == 6, "third element should be 6")


def test_strides_getitem() raises:
    var s = Strides(10, 20, 30, 40)
    assert_true(s[0] == 10, "first element should be 10")
    assert_true(s[2] == 30, "third element should be 30")
    assert_true(s[-1] == 40, "last element should be 40")


def test_strides_setitem() raises:
    var s = Strides(1, 2, 3, 4)
    s[0] = 100
    s[2] = 200
    assert_true(s[0] == 100, "s[0] should be 100")
    assert_true(s[2] == 200, "s[2] should be 200")


def test_strides_slice() raises:
    var s = Strides(10, 20, 30, 40, 50)
    var sliced = s[1:4]
    assert_true(len(sliced) == 3, "sliced strides should have 3 elements")
    assert_true(sliced[0] == 20, "first element should be 20")
    assert_true(sliced[2] == 40, "last element should be 40")


def test_strides_eq() raises:
    var s1 = Strides(1, 2, 3)
    var s2 = Strides(1, 2, 3)
    var s3 = Strides(1, 2, 4)
    assert_true(s1 == s2, "s1 should equal s2")
    assert_true(not (s1 == s3), "s1 should not equal s3")


def test_strides_str() raises:
    var s = Strides(10, 20, 30)
    var str_repr = s.__str__()
    assert_true(str_repr == "(10, 20, 30)", "string should be '(10, 20, 30)'")


def test_strides_tolist() raises:
    var s = Strides(5, 10, 15)
    var lst = s.tolist()
    assert_true(len(lst) == 3, "list should have 3 elements")
    assert_true(lst[1] == 10, "second element should be 10")


def test_strides_permute() raises:
    var s = Strides(10, 20, 30, 40)
    var axes = IntArray(2, 0, 3, 1)
    var result = s.permute(axes)
    assert_true(result[0] == 30, "first element should be 30")
    assert_true(result[1] == 10, "second element should be 10")
    assert_true(result[2] == 40, "third element should be 40")
    assert_true(result[3] == 20, "fourth element should be 20")


def test_strides_default() raises:
    var shape = Shape(2, 3, 4)
    var s = Strides.default(shape)
    assert_true(len(s) == 3, "strides should have 3 elements")
    assert_true(s[0] == 12, "stride[0] should be 12 (3*4)")
    assert_true(s[1] == 4, "stride[1] should be 4")
    assert_true(s[2] == 1, "stride[2] should be 1")


def test_strides_default_scalar() raises:
    var shape = Shape()
    var s = Strides.default(shape)
    assert_true(len(s) == 0, "scalar should have empty strides")


def test_strides_is_contiguous() raises:
    var shape = Shape(2, 3, 4)
    var s1 = Strides(12, 4, 1)
    var s2 = Strides(12, 1, 4)
    assert_true(s1.is_contiguous(shape), "s1 should be contiguous")
    assert_true(not s2.is_contiguous(shape), "s2 should not be contiguous")


def test_strides_is_contiguous_scalar() raises:
    var shape = Shape()
    var s = Strides()
    assert_true(s.is_contiguous(shape), "scalar is trivially contiguous")


def test_strides_with_capacity() raises:
    var s = Strides.with_capacity(10)
    assert_true(len(s) == 0, "with_capacity should start with size 0")


def test_compute_default_strides() raises:
    shape = Shape.of(2, 3, 4)
    strides = Strides.default(shape)
    assert_true(
        strides == Strides(12, 4, 1),
        "stride compute assertion 1 failed",
    )

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


