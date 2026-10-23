from tenmo.tensor import Tensor
from tenmo.shared.indexhelper import i, s, e, ellipsis, newaxis
from std.testing import assert_true, TestSuite

def test_ellipsis_alone() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var v = t[ellipsis]
    assert_true(v.shape()[0] == 2)
    assert_true(v.shape()[1] == 2)
    assert_true(v[0, 0] == 1.0)
    assert_true(v[1, 1] == 4.0)


def test_ellipsis_trailing_index() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var v = t[e(), i(-1)]
    assert_true(v.shape()[0] == 2)
    assert_true(v[0] == 3.0)
    assert_true(v[1] == 6.0)


def test_ellipsis_leading_index() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var v = t[i(1), e()]
    assert_true(v.shape()[0] == 3)
    assert_true(v[0] == 4.0)
    assert_true(v[2] == 6.0)


def test_ellipsis_with_newaxis() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var v = t[newaxis, e()]
    assert_true(v.shape()[0] == 1)
    assert_true(v.shape()[1] == 3)
    assert_true(v[0, 2] == 3.0)


def test_ellipsis_fill() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].zeros(2, 3)
    t.fill(7.0, e(), i(-1))
    assert_true(t[0, 2] == 7.0)
    assert_true(t[1, 2] == 7.0)
    assert_true(t[0, 0] == 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
