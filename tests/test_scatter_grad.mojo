from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite

def test_scatter_grad_src() raises:
    comptime dtype = DType.float32
    var base = Tensor[dtype].zeros(4)
    var idx_list = List[Int]()
    idx_list.append(0)
    idx_list.append(2)
    var src = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var out = base.scatter(idx_list, src, axis=0)
    var loss = out.sum()
    loss.backward()
    assert_true(src.grad()[0] == 1.0)
    assert_true(src.grad()[1] == 1.0)


def test_scatter_grad_base() raises:
    comptime dtype = DType.float32
    var base = Tensor[dtype].d1([10.0, 20.0, 30.0, 40.0], requires_grad=True)
    var idx_list = List[Int]()
    idx_list.append(1)
    var src = Tensor[dtype].d1([99.0])
    var out = base.scatter(idx_list, src, axis=0)
    var loss = out.sum()
    loss.backward()
    assert_true(base.grad()[0] == 1.0)
    assert_true(base.grad()[1] == 0.0)
    assert_true(base.grad()[2] == 1.0)
    assert_true(base.grad()[3] == 1.0)


def test_scatter_grad_duplicate_indices() raises:
    comptime dtype = DType.float32
    var base = Tensor[dtype].zeros(3)
    var idx_list = List[Int]()
    idx_list.append(0)
    idx_list.append(0)
    var src = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var out = base.scatter(idx_list, src, axis=0)
    var loss = out.sum()
    loss.backward()
    assert_true(src.grad()[0] == 1.0)
    assert_true(src.grad()[1] == 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
