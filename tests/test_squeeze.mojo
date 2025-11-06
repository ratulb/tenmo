from tenmo import Tensor
from shapes import Shape



from testing import assert_true

fn main() raises:
    test_squeeze_scalar()
    test_squeeze_1d_no_effect()
    test_squeeze_2d_singleton_row()
    test_squeeze_3d_multiple_singletons()
    test_squeeze_with_specific_dim()
    test_squeeze_with_non_singleton_dim()
    test_squeeze_keep_chain_grad()
    print("passes")



fn test_squeeze_scalar() raises:
    print("test_squeeze_scalar")
    var a = Tensor.scalar(42.0, requires_grad=True)
    var s = a.squeeze()
    assert_true(s.shape() == Shape.of())
    assert_true(s.item() == 42.0)
    s.backward()
    assert_true(a.grad().item() == 1.0)


fn test_squeeze_1d_no_effect() raises:
    print("test_squeeze_1d_no_effect")
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var s = a.squeeze()
    assert_true(s.shape() == Shape.of(3))
    assert_true((s == a))
    s.sum().backward()
    assert_true(a.grad().all_close(Tensor.d1([1.0, 1.0, 1.0])))


fn test_squeeze_2d_singleton_row() raises:
    print("test_squeeze_2d_singleton_row")
    var a = Tensor.d2([[1.0, 2.0, 3.0]], requires_grad=True)  # shape (1,3)
    var s = a.squeeze()
    assert_true(s.shape() == Shape.of(3))
    assert_true(s.all_close(Tensor.d1([1.0, 2.0, 3.0])))
    s.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0, 1.0]])))


fn test_squeeze_3d_multiple_singletons() raises:
    print("test_squeeze_3d_multiple_singletons")
    var a = Tensor.d3([[[10.0, 20.0]]], requires_grad=True)  # shape (1,1,2)
    var s = a.squeeze()
    assert_true(s.shape() == Shape.of(2))
    assert_true(s.all_close(Tensor.d1([10.0, 20.0])))
    s.sum().backward()
    expected_grad = Tensor.d3([[[1.0, 1.0]]])
    assert_true(a.grad().all_close(expected_grad))


fn test_squeeze_with_specific_dim() raises:
    print("test_squeeze_with_specific_dim")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]]],  # shape (1,2,2)
        requires_grad=True
    )
    var s = a.squeeze([0])
    assert_true(s.shape() == Shape.of(2, 2))
    assert_true(s.all_close(Tensor.d2([[1.0, 2.0], [3.0, 4.0]])))
    s.sum().backward()
    expected_grad = Tensor.d3([[[1.0, 1.0], [1.0, 1.0]]])
    assert_true(a.grad().all_close(expected_grad))


fn test_squeeze_with_non_singleton_dim() raises:
    print("test_squeeze_with_non_singleton_dim")
    var a = Tensor.d3(
        [[[1.0, 2.0]], [[3.0, 4.0]]],  # shape (2,1,2)
        requires_grad=True
    )
    var s = a.squeeze([1])  # valid because dim=1 has size 1
    assert_true(s.shape() == Shape.of(2, 2))
    expected = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    assert_true(s.all_close(expected))
    s.sum().backward()
    expected_grad = Tensor.d3(
        [[[1.0, 1.0]], [[1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


fn test_squeeze_keep_chain_grad() raises:
    print("test_squeeze_keep_chain_grad")
    var a = Tensor.d3([[[1.0, 2.0, 3.0]]], requires_grad=True)  # shape (1,1,3)
    var s = a.squeeze()
    var y = s * 2.0
    var z = y.sum()
    z.backward()
    # z = sum(2 * a) → grad(a) = 2
    expected_grad = Tensor.d3([[[2.0, 2.0, 2.0]]])
    assert_true(a.grad().all_close(expected_grad))



