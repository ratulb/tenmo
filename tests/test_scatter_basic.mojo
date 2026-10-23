from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite

def test_scatter_basic_1d() raises:
    comptime dtype = DType.float32
    var base = Tensor[dtype].zeros(4)
    var idx_list = List[Int]()
    idx_list.append(0)
    idx_list.append(2)
    var src = Tensor[dtype].d1([1.0, 2.0])
    var out = base.scatter(idx_list, src, axis=0)
    assert_true(out[0] == 1.0)
    assert_true(out[1] == 0.0)
    assert_true(out[2] == 2.0)
    assert_true(out[3] == 0.0)
    # base unchanged
    assert_true(base[0] == 0.0)


def test_scatter_basic_overwrite() raises:
    comptime dtype = DType.float32
    var base = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0])
    var idx_list = List[Int]()
    idx_list.append(1)
    var src = Tensor[dtype].d1([99.0])
    var out = base.scatter(idx_list, src, axis=0)
    assert_true(out[0] == 10.0)
    assert_true(out[1] == 99.0)
    assert_true(out[2] == 30.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
