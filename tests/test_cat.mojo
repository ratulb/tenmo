from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite

def test_cat_tuple_axis0() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    var result = Tensor[dtype].cat((A, B), axis=0)

    assert_true(result.shape()[0] == 4)
    assert_true(result.shape()[1] == 3)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
    )
    assert_true(result.all_close[atol=1e-6](expected))


def test_cat_tuple_axis1() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    var result = Tensor[dtype].cat((A, B), axis=1)

    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 6)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 7.0, 8.0, 9.0], [4.0, 5.0, 6.0, 10.0, 11.0, 12.0]]
    )
    assert_true(result.all_close[atol=1e-6](expected))


def test_cat_tuple_three_tensors() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var B = Tensor[dtype].d2([[5.0, 6.0]])
    var C = Tensor[dtype].d2([[7.0, 8.0], [9.0, 10.0]])

    var result = Tensor[dtype].cat((A, B, C), axis=0)

    assert_true(result.shape()[0] == 5)
    assert_true(result.shape()[1] == 2)



def test_cat_grad_axis0_simple() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0]], requires_grad=True)

    var result = Tensor[dtype].cat((A, B), axis=0)
    var loss = result.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].ones(2, 2)
    var expected_grad_B = Tensor[dtype].ones(1, 2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


def test_cat_grad_axis1_simple() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0], [6.0]], requires_grad=True)

    var result = Tensor[dtype].cat((A, B), axis=1)
    var loss = result.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].ones(2, 2)
    var expected_grad_B = Tensor[dtype].ones(2, 1)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


def test_cat_grad_three_tensors() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0]], requires_grad=True)
    var C = Tensor[dtype].d2([[7.0, 8.0], [9.0, 10.0]], requires_grad=True)

    var result = Tensor[dtype].cat((A, B, C), axis=0)
    var loss = result.sum()
    loss.backward()

    assert_true(A.grad().all_close[atol=1e-6](Tensor[dtype].ones(2, 2)))
    assert_true(B.grad().all_close[atol=1e-6](Tensor[dtype].ones(1, 2)))
    assert_true(C.grad().all_close[atol=1e-6](Tensor[dtype].ones(2, 2)))


def test_cat_grad_weighted_loss() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0]], requires_grad=True)

    var result = Tensor[dtype].cat((A, B), axis=0)
    var weights = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var loss = (result * weights).sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var expected_grad_B = Tensor[dtype].d2([[5.0, 6.0]])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
