from tenmo.tensor import Tensor
from tenmo.shared.indexhelper import i, s, e
from std.testing import assert_true, TestSuite

def test_setitem_slice_region() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].zeros(4, 4)
    t.__setitem__(s(1, 3), s(), value=Scalar[dtype](5.0))
    assert_true(t[1, 0] == 5.0)
    assert_true(t[2, 3] == 5.0)
    assert_true(t[0, 0] == 0.0)
    assert_true(t[3, 0] == 0.0)


def test_setitem_advanced_region() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].zeros(3, 3)
    t.__setitem__(i(-1), s(), value=Scalar[dtype](9.0))
    assert_true(t[2, 0] == 9.0)
    assert_true(t[2, 2] == 9.0)
    assert_true(t[0, 0] == 0.0)


def test_setitem_tensor_source() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].zeros(2, 2)
    var src = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    t.__setitem__(s(), s(), value=src)
    assert_true(t[0, 0] == 1.0)
    assert_true(t[1, 1] == 4.0)


def test_setitem_ellipsis_region() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].zeros(2, 3)
    t.__setitem__(e(), i(-1), value=Scalar[dtype](7.0))
    assert_true(t[0, 2] == 7.0)
    assert_true(t[1, 2] == 7.0)
    assert_true(t[0, 0] == 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
