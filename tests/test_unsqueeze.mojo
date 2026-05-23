from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from std.testing import assert_true, TestSuite


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_unsqueeze_scalar_to_1d() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(7.0, requires_grad=True)  # shape ()
    var u = a.unsqueeze(0)
    assert_true(u.shape() == Shape.of(1))
    assert_true(u.item() == 7.0)
    u.backward()
    assert_true(a.grad().item() == 1.0)


def test_unsqueeze_1d_to_2d_front() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)  # shape (3,)
    var u = a.unsqueeze(0)
    assert_true(u.shape() == Shape.of(1, 3))
    s = u.sum()
    s.backward()
    expected_grad = Tensor[dtype].d1([1.0, 1.0, 1.0])
    assert_true(a.grad().all_close(expected_grad))


def test_unsqueeze_1d_to_2d_back() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)  # shape (3,)
    var u = a.unsqueeze(1)
    assert_true(u.shape() == Shape.of(3, 1))
    s = u.sum()
    s.backward()
    expected_grad = Tensor[dtype].d1([1.0, 1.0, 1.0])
    assert_true(a.grad().all_close(expected_grad))


def test_unsqueeze_2d_insert_middle_dim() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )  # shape (2,2)
    var u = a.unsqueeze(1)  # → shape (2,1,2)
    assert_true(u.shape() == Shape.of(2, 1, 2))
    s = u.sum()
    s.backward()
    expected_grad = Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]])
    assert_true(a.grad().all_close(expected_grad))


def test_unsqueeze_3d_insert_front_and_back() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True
    )  # shape (1,2,2)
    var u1 = a.unsqueeze(0)  # shape (1,1,2,2)
    var u2 = a.unsqueeze(3)  # shape (1,2,2,1)
    assert_true(u1.shape() == Shape.of(1, 1, 2, 2))
    assert_true(u2.shape() == Shape.of(1, 2, 2, 1))
    s = u1.sum() + u2.sum()
    s.backward()
    expected_grad = Tensor[dtype].d3([[[2.0, 2.0], [2.0, 2.0]]])
    assert_true(a.grad().all_close(expected_grad))


def test_unsqueeze_chain_grad_flow() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var u = a.unsqueeze(0)  # shape (1,3)
    var y = u * 2.0
    var z = y.sum()
    z.backward()
    expected_grad = Tensor[dtype].d1([2.0, 2.0, 2.0])
    assert_true(a.grad().all_close(expected_grad))


def test_unsqueeze_multiple_dims() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[5.0, 6.0]], requires_grad=True)  # shape (1,2)
    var uu = a.unsqueeze(0)
    u = uu.unsqueeze(3)  # → (1,1,2,1)
    assert_true(u.shape() == Shape.of(1, 1, 2, 1))
    s = u.sum()
    s.backward()
    expected_grad = Tensor[dtype].d2([[1.0, 1.0]])
    assert_true(a.grad().all_close(expected_grad))


def test_unsqueeze_preserves_buffer_sharing() raises:
    _ = """var a = Tensor[dtype].d1([9.0, 8.0, 7.0], requires_grad=False)
    var u = a.unsqueeze(0)
    # unsqueeze should not copy, only metadata reshape
    assert_true(a.buffer().ptr() == u.buffer().ptr())
    assert_true(u.shape() == Shape.of(1, 3))
    assert_true(u.all_close(Tensor[dtype].d2([[9.0, 8.0, 7.0]])))"""
