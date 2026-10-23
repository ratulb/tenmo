from tenmo.tensor import Tensor
from tenmo.shared.indexhelper import i, s, e, ellipsis, newaxis
from std.testing import assert_true, TestSuite

def test_ellipsis_view_grad() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var y = t[e(), i(-1)]
    var loss = y.sum()
    loss.backward()
    assert_true(t.grad().shape()[0] == 2)
    assert_true(t.grad().shape()[1] == 3)
    assert_true(t.grad()[0, 2] == 1.0)
    assert_true(t.grad()[1, 2] == 1.0)
    assert_true(t.grad()[0, 0] == 0.0)


def test_negative_index_view_grad() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    var y = t[i(-1), s()]
    var loss = y.sum()
    loss.backward()
    assert_true(t.grad().shape()[0] == 3)
    assert_true(t.grad().shape()[1] == 2)
    assert_true(t.grad()[2, 0] == 1.0)
    assert_true(t.grad()[2, 1] == 1.0)
    assert_true(t.grad()[0, 0] == 0.0)


def test_slice_view_grad() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    var y = t[-3:-1]
    var loss = y.sum()
    loss.backward()
    assert_true(t.grad()[3] == 1.0)
    assert_true(t.grad()[4] == 1.0)
    assert_true(t.grad()[0] == 0.0)


def test_newaxis_view_grad() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var y = t[newaxis, s()]
    var loss = y.sum()
    loss.backward()
    assert_true(t.grad()[0] == 1.0)
    assert_true(t.grad()[2] == 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
