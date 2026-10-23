from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite

def test_masked_select_grad() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var mask = Tensor[DType.bool].d1([True, False, True, False])
    var out = t.masked_select(mask)
    var loss = out.sum()
    loss.backward()
    assert_true(t.grad()[0] == 1.0)
    assert_true(t.grad()[1] == 0.0)
    assert_true(t.grad()[2] == 1.0)
    assert_true(t.grad()[3] == 0.0)


def test_masked_select_grad_values() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var mask = Tensor[DType.bool].d1([False, True, False, True])
    var out = t.masked_select(mask)
    var loss = (out * out).sum()
    loss.backward()
    # grad = 2*x at selected positions
    assert_true(t.grad()[1] == 4.0)
    assert_true(t.grad()[3] == 8.0)
    assert_true(t.grad()[0] == 0.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
