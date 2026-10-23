from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite

def test_masked_select_1d() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var mask = Tensor[DType.bool].d1([True, False, True, False])
    var out = t.masked_select(mask)
    assert_true(out.shape()[0] == 2)
    assert_true(out[0] == 1.0)
    assert_true(out[1] == 3.0)


def test_masked_select_all() raises:
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var mask = Tensor[DType.bool].d1([True, True, True])
    var out = t.masked_select(mask)
    assert_true(out.shape()[0] == 3)
    assert_true(out[2] == 3.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
