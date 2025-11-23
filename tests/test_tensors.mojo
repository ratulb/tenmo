# %s/Tensor\.walk_backward(\([^)]*\))/\1.backward()/g
# %s/^\(fn test_\(.*\)() raises:\)$/&\r    print("test_\2")/
from testing import assert_true, assert_false, assert_raises
from tenmo import Tensor
from intlist import IntList
from shapes import Shape
from common_utils import *
from utils.numerics import min_finite
from operators import AddTensor
from strides import Strides


fn test_count() raises:
    scalar = Tensor.scalar(10)
    assert_true(scalar.count(10) == 1, "Scalar count assertion 1 failed")
    assert_true(scalar.count(42) == 0, "Scalar count assertion 2 failed")

    full = Tensor.full([2, 3, 4], 42)
    assert_true(full.count(42) == 24, "Tensor count assertion 3 failed")

    v = full[i(0), s(), s()]
    v2 = full[i(1), s(), s()]

    assert_true(v.count(42) == 12, "Tensor view count assertion 4 failed")
    assert_true(v2.count(42) == 12, "Tensor view count assertion 5 failed")


fn test_reshape_slice_sum_backward() raises:
    print("test_reshape_slice_sum_backward")
    var a = Tensor.arange(6, requires_grad=True)
    r = a.reshape([2, 3])
    # Gradient check
    var y = r[0:1, 1:3]
    ss = y.sum()
    ss.backward()
    var expected_grad = Tensor.d1([0, 1, 1, 0, 0, 0])
    assert_true((a.grad() == expected_grad))

    # Full slice
    # var full_slice = r[s(), s()]
    var full_slice = r[:, :]
    assert_true((full_slice == r))
    # Row slice
    var row = r[1, s()]
    assert_true((row == Tensor([3, 4, 5])))

    # Column slice with step
    var col_step = r[s(), s(0, 3, 2)]
    expect = Tensor.d2([[0, 2], [3, 5]])
    assert_true((col_step == expect))


fn test_shared_tensor_twice() raises:
    print("test_shared_tensor_twice")

    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)

    # a is used in two places
    var b = a * 2  # ∂b/∂a = 2
    var c = a * 3  # ∂c/∂a = 3

    var d = b + c  # ∂d/∂a = ∂b/∂a + ∂c/∂a = 2 + 3 = 5

    d.backward()

    # Final grad: ∂d/∂a = [5, 5, 5]
    assert_true(a.grad().all_close(Tensor.d1([5.0, 5.0, 5.0])), "∂d/∂a = 5")


fn test_broadcast_and_reuse() raises:
    print("test_broadcast_and_reuse")

    var a = Tensor.d2([[1.0], [2.0], [3.0]], requires_grad=True)  # shape: (3,1)

    var b = a + 1  # ∂b/∂a = 1
    var c = a * 2  # ∂c/∂a = 2

    var d = b + c  # ∂d/∂a = 1 + 2 = 3

    var loss = d.sum()
    loss.backward()

    # Expected gradient: shape (3,1), all 3's
    assert_true(
        a.grad().all_close(Tensor.d2([[3.0], [3.0], [3.0]])), "∂loss/∂a = 3"
    )


fn test_branching_square_add() raises:
    print("test_branching_square_add")

    var a = Tensor.d1([2.0], requires_grad=True)

    var b = a * a  # b = a²      → ∂b/∂a = 2a = 4
    var c = a + a  # c = 2a      → ∂c/∂a = 2

    var d = b + c  # d = a² + 2a
    d.backward()

    # ∂d/∂a = ∂b/∂a + ∂c/∂a = 4 + 2 = 6
    assert_true(a.grad().all_close(Tensor.d1([6.0])), "∂d/∂a = 6")


fn test_merge_of_dependent_branches() raises:
    print("test_merge_of_dependent_branches")

    var a = Tensor.d1([1.0], requires_grad=True)

    var b = a + 1  # b = a + 1       → ∂b/∂a = 1
    # var c = b * a         # c = (a + 1) * a → ∂c/∂a = b + a = 1 + 1 = 2
    var c = a * b  # c = (a + 1) * a → ∂c/∂a = b + a = 1 + 1 = 2

    c.backward()

    # ∂c/∂a = b + a = 3
    assert_true(a.grad().all_close(Tensor.d1([3.0])), "∂c/∂a = b + a = 3")


fn test_square_and_identity_path() raises:
    print("test_square_and_identity_path")

    var a = Tensor.d1([3.0], requires_grad=True)

    var sq = a * a  # ∂sq/∂a = 2a = 6
    var id = a  # ∂id/∂a = 1

    var out = sq + id  # ∂out/∂a = 6 + 1 = 7
    out.backward()

    assert_true(a.grad().all_close(Tensor.d1([7.0])), "∂out/∂a = 7")


fn test_shared_dependency_multiple_paths() raises:
    print("test_shared_dependency_multiple_paths")
    var a = Tensor.d1([1.0], requires_grad=True)
    var b = a + 1  # ∂b/∂a = 1
    var c = a * 2  # ∂c/∂a = 2
    var d = b + c  # ∂d/∂a = ∂d/∂b * ∂b/∂a + ∂d/∂c * ∂c/∂a = 1 + 1 = 2
    d.backward()

    # Correct gradient: ∂d/∂a = 1 (from b) + 2 (from c) = 3
    assert_true(a.grad().all_close(Tensor.d1([3.0])), "∂d/∂a should be 2")


fn test_diamond_dependency() raises:
    print("test_diamond_dependency")
    var a = Tensor.d1([1.0], requires_grad=True)
    var b = a + 1  # ∂b/∂a = 1
    var c = a * 2  # ∂c/∂a = 2
    var d = b * c  # ∂d/∂a = c * ∂b/∂a + b * ∂c/∂a = 2*1 + 2*2 = 6

    d.backward()

    # Correct gradient: ∂d/∂a = 2 + 4 = 6
    assert_true(a.grad().all_close(Tensor.d1([6.0])), "∂d/∂a should be 6")


fn test_repeated_tensor_use() raises:
    print("test_repeated_tensor_use")
    var a = Tensor.d1([2.0], requires_grad=True)
    var b = a * a  # ∂b/∂a = a + a = 4 (since ∂(a²)/∂a = 2a)

    b.backward()

    # Correct gradient: ∂b/∂a = 2a = 4
    assert_true(a.grad().all_close(Tensor.d1([4.0])), "∂b/∂a should be 4")


fn test_topological_sort_required() raises:
    print("test_topological_sort_required")
    var a = Tensor.d1([1.0], requires_grad=True)
    var b = a * 2  # ∂b/∂a = 2
    var c = a + 1  # ∂c/∂a = 1
    var d = b * c  # ∂d/∂a = c*∂b/∂a + b*∂c/∂a = 2*2 + 2*1 = 6
    d.backward()
    assert_true(
        a.grad().all_close(Tensor.d1([6.0]))
    )  # Might fail without topological sort!


fn test_matmul_scalar_output() raises:
    print("test_matmul_scalar_output")
    var a = Tensor.d2([[1.0, 2.0]])
    var b = Tensor.d2([[3.0], [4.0]])
    var out = a.matmul(b)  # [[1*3 + 2*4]] = [[11]]
    assert_true((out == Tensor.d2([[11.0]])))


fn test_matmul_tensor_tensor() raises:
    print("test_matmul_tensor_tensor")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var b = Tensor.d2([[5.0], [6.0]])
    var out = a.matmul(b)
    assert_true((out == Tensor.d2([[17.0], [39.0]])))


fn test_matmul_tensor_view() raises:
    print("test_matmul_tensor_view")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var b = Tensor.d2([[5.0], [6.0]])
    var view_b = b.view(shape=[2, 1], strides=[1, 1], offset=0)
    print("view_b contiguous? ", view_b.is_contiguous())
    var out = a.matmul(view_b)
    assert_true((out == Tensor.d2([[17.0], [39.0]])))


fn test_matmul_view_tensor() raises:
    print("test_matmul_view_tensor")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var b = Tensor.d2([[5.0], [6.0]])
    var view_a = a.view(shape=[2, 2], strides=[2, 1], offset=0)
    var out = view_a.matmul(b)
    assert_true((out == Tensor.d2([[17.0], [39.0]])))


fn test_matmul_view_view() raises:
    print("test_matmul_view_view")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var b = Tensor.d2([[5.0], [6.0]])
    var view_a = a.view(shape=[2, 2], strides=[2, 1], offset=0)
    var view_b = b.view(shape=[2, 1], strides=[1, 1], offset=0)
    var out = view_a.matmul(view_b)
    assert_true((out == Tensor.d2([[17.0], [39.0]])))


fn test_matmul_transposed_tensor_tensor() raises:
    print("test_matmul_transposed_tensor_tensor")
    var a = Tensor.d2([[1.0, 2.0, 3.0]])  # shape (1,3)
    var b = Tensor.d2([[4.0], [5.0], [6.0]])  # shape (3,1)
    var out = a.matmul(b)
    assert_true((out == Tensor.d2([[32.0]])))


fn test_matmul_transposed_tensor_view() raises:
    print("test_matmul_transposed_tensor_view")
    var a = Tensor.d2([[1.0, 2.0, 3.0]])
    var b = Tensor.d2([[4.0], [5.0], [6.0]])
    var view_b = b.view(shape=[3, 1], strides=[1, 1], offset=0)
    var out = a.matmul(view_b)
    assert_true((out == Tensor.d2([[32.0]])))


fn test_matmul_transposed_view_tensor() raises:
    print("test_matmul_transposed_view_tensor")
    var a = Tensor.d2([[1.0, 2.0, 3.0]])
    var b = Tensor.d2([[4.0], [5.0], [6.0]])
    var view_a = a.view(shape=[1, 3], strides=[3, 1], offset=0)
    var out = view_a.matmul(b)
    assert_true((out == Tensor.d2([[32.0]])))


fn test_matmul_transposed_view_view() raises:
    print("test_matmul_transposed_view_view")
    var a = Tensor.d2([[1.0, 2.0, 3.0]])
    var b = Tensor.d2([[4.0], [5.0], [6.0]])
    var view_a = a.view(shape=[1, 3], strides=[3, 1], offset=0)
    var view_b = b.view(shape=[3, 1], strides=[1, 1], offset=0)
    var out = view_a.matmul(view_b)
    assert_true((out == Tensor.d2([[32.0]])))


fn test_tensor_shared_multiple_paths() raises:
    print("test_tensor_shared_multiple_paths")
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y = x * 3  # 6
    var z = x + 4  # 6
    var out = y + z  # 12

    out.backward()

    # ∂out/∂x = ∂y/∂x + ∂z/∂x = 3 + 1 = 4
    assert_true(out.item() == 12.0, "Value check")
    assert_true(x.grad().item() == 4.0, "Correct accumulated gradient")


fn test_tensor_reuse_broadcasting() raises:
    print("test_tensor_reuse_broadcasting")
    var x = Tensor.d1([1, 2, 3], requires_grad=True)
    var y = x + x  # [2, 4, 6]

    y.sum().backward()

    assert_true(
        x.grad().all_close(Tensor.d1([2, 2, 2])),
        "Gradient doubles due to reuse",
    )


fn test_tensor_reuse_deep_chain() raises:
    print("test_tensor_reuse_deep_chain")
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y = x + 1  # 3
    var z = y * x  # 3 * 2 = 6
    var w = z + x  # 6 + 2 = 8

    w.backward()

    assert_true(w.item() == 8.0, "Value check")
    # ∂w/∂x = ∂z/∂x + 1
    # z = (x + 1) * x → ∂z/∂x = (1)*x + (x+1)*1 = x + x + 1 = 2x + 1 = 5
    assert_true(x.grad().item() == 5 + 1, "∂w/∂x = ∂z/∂x + ∂x/∂x = 6")


fn test_tensor_reuse_in_two_branches() raises:
    print("test_tensor_reuse_in_two_branches")
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y1 = x + 1  # 3
    var y2 = x * 5  # 10
    var z = y1 + y2  # 13

    z.backward()

    assert_true(z.item() == 13.0, "Value check")
    assert_true(x.grad().item() == 1 + 5, "∂z/∂x = 1 (from y1) + 5 (from y2)")


fn test_tensor_reuse_mixed() raises:
    print("test_tensor_reuse_mixed")
    var x = Tensor.scalar(3.0, requires_grad=True)
    var y = x * x + x  # 3 * 3 + 3 = 12

    y.backward()

    assert_true(y.item() == 12.0, "Value check")
    assert_true(x.grad().item() == 2 * 3 + 1, "∂y/∂x = 2x + 1 = 6 + 1 = 7")


fn test_tensor_reuse_add() raises:
    print("test_tensor_reuse_add")
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y = x + x  # 2 + 2 = 4

    y.backward()

    assert_true(y.item() == 4.0, "Value check")
    assert_true(x.grad().item() == 2.0, "∂y/∂x = 1 + 1 = 2 (reuse)")


fn test_simple_chain() raises:
    print("test_simple_chain")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = Tensor.scalar(3.0, requires_grad=True)
    var c = a * b  # 2 * 3 = 6
    var d = c + b  # 6 + 3 = 9

    d.backward()
    assert_true(d.item() == 9.0, "Value check")
    assert_true(a.grad().item() == 3.0, "∂d/∂a = b = 3")


fn test_scalar_mul_scalar() raises:
    print("test_scalar_mul_scalar")
    var a = Tensor.scalar(3.0, requires_grad=True)
    var b = Tensor.scalar(4.0, requires_grad=True)

    var c = a * b
    c.backward()

    assert_true(c.item() == 12.0)
    assert_true(a.grad().item() == 4.0)
    assert_true(b.grad().item() == 3.0)


fn test_1d_mul_1d_same_shape() raises:
    print("test_1d_mul_1d_same_shape")
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor.d1([4.0, 5.0, 6.0], requires_grad=True)

    var c = a * b
    c.sum().backward()

    assert_true((c == Tensor.d1([4.0, 10.0, 18.0])))
    assert_true(a.grad().all_close(Tensor.d1([4.0, 5.0, 6.0])))
    assert_true(b.grad().all_close(Tensor.d1([1.0, 2.0, 3.0])))


fn test_2d_mul_2d_same_shape() raises:
    print("test_2d_mul_2d_same_shape")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var c = a * b
    c.sum().backward()

    assert_true((c == Tensor.d2([[5.0, 12.0], [21.0, 32.0]])))
    assert_true(a.grad().all_close(b))
    assert_true(b.grad().all_close(a))


fn test_broadcast_2d_1d_mul() raises:
    print("test_broadcast_2d_1d_mul")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.d1([5.0, 6.0], requires_grad=True)

    var c = a * b  # broadcasts b along rows
    summ = c.sum()
    summ.backward()

    assert_true((c == Tensor.d2([[5.0, 12.0], [15.0, 24.0]])))
    assert_true(a.grad().all_close(Tensor.d2([[5.0, 6.0], [5.0, 6.0]])))

    # b.grad should sum over rows
    assert_true(b.grad().all_close(Tensor.d1([4.0, 6.0])))


fn test_broadcast_1d_2d_mul() raises:
    print("test_broadcast_1d_2d_mul")
    var a = Tensor.d1([2.0, 3.0], requires_grad=True)
    var b = Tensor.d2([[4.0, 5.0], [6.0, 7.0]], requires_grad=True)

    var c = a * b  # a broadcasts over rows
    c.sum().backward()

    assert_true((c == Tensor.d2([[8.0, 15.0], [12.0, 21.0]])))

    assert_true(a.grad().all_close(Tensor.d1([10.0, 12.0])))
    assert_true(b.grad().all_close(Tensor.d2([[2.0, 3.0], [2.0, 3.0]])))


fn test_3d_broadcast_mul() raises:
    print("test_3d_broadcast_mul")
    var a = Tensor.d3(
        [[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True
    )  # shape (2, 2, 1)
    var b = Tensor.d3([[[5.0, 6.0]]], requires_grad=True)  # shape (1, 1, 2)

    var c = a * b  # result shape (2, 2, 2)
    c.sum().backward()

    assert_true(c.shape() == Shape.of(2, 2, 2))
    assert_true(a.grad().shape() == Shape.of(2, 2, 1))
    assert_true(b.grad().shape() == Shape.of(1, 1, 2))


fn test_mul_one_requires_grad() raises:
    print("test_mul_one_requires_grad")
    var a = Tensor.d1([1.0, 2.0, 3.0])  # no grad
    var b = Tensor.d1([4.0, 5.0, 6.0], requires_grad=True)

    var c = a * b
    c.sum().backward()

    assert_true(b.grad().all_close(a))


fn test_scalar_tensor_mul() raises:
    print("test_scalar_tensor_mul")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = Tensor.d1([1.0, 2.0, 3.0])

    var c = a * b
    c.sum().backward()

    assert_true(a.grad().item() == 6.0)  # sum of b


fn test_unsqueeze() raises:
    print("test_unsqueeze")
    _ = """tensor = Tensor.rand(2,3, requires_grad=True)
    tensor2 = tensor.unsqueeze(0)
    tensor2[IntList(0, 0, 0)] = 100"""
    pass


fn test_tensor_mean() raises:
    print("test_tensor_mean")

    a = Tensor.scalar(5.0, requires_grad=True)
    m = a.mean()
    m.backward()
    assert_true(m.item() == 5.0)
    assert_true(a.grad().item() == 1.0)

    a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    m = a.mean()
    assert_true(m.item() == 2.0)
    m.backward()
    assert_true(a.grad().all_close(Tensor.d1([1 / 3, 1 / 3, 1 / 3])))

    A = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    M = A.mean()
    assert_true(M.item() == 2.5)
    M.backward()

    expected = Tensor.d2([[0.25, 0.25], [0.25, 0.25]])
    assert_true(A.grad().all_close(expected))

    a1 = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    m1 = a1.mean(axes=[0], keepdims=False)
    assert_true(m1.all_close(Tensor.d1([2.0, 3.0])))
    m1.backward()

    assert_true(a1.grad().all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])))

    AA = Tensor.d2([[1.0, 2.0], [3.0, 5.0]], requires_grad=True)
    MM = AA.mean(axes=[1], keepdims=False)
    assert_true(MM.all_close(Tensor.d1([1.5, 4.0])))
    MM.backward()

    assert_true(AA.grad().all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])))

    C = Tensor.d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
    mm = C.mean(axes=[1], keepdims=True)
    assert_true(mm.all_close(Tensor.d2([[2.0], [3.0]])))
    mm.backward()

    assert_true(C.grad().all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])))

    A3 = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    m12 = A3.mean(axes=[1, 2], keepdims=False)  # shape: (2,)
    assert_true(m12.all_close(Tensor.d1([2.5, 6.5])))
    m12.backward()
    expected_grad = Tensor.d3(
        [[[0.25, 0.25], [0.25, 0.25]], [[0.25, 0.25], [0.25, 0.25]]]
    )
    assert_true(A3.grad().all_close(expected_grad))

    A2 = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    m2 = A2.mean(axes=[-1])  # same as axis=1
    assert_true(m2.all_close(Tensor.d1([1.5, 3.5])))

    a2 = Tensor.d2([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    m_ = a2.mean(axes=IntList())
    assert_true(m_.item() == 25.0)


fn test_forward_multivariate_prediction() raises:
    print("test_forward_multivariate_prediction")
    # x.shape() = (3, 2), w.shape() = (2, 1)
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var w = Tensor.d2([[1.0], [2.0]])  # So y = x @ w = [[5], [11], [17]]
    var y_pred = x.matmul(w)

    assert_true((y_pred == Tensor.d2([[5.0], [11.0], [17.0]])))
    var b = Tensor.d1([1.0])
    var y = x.matmul(w) + b
    assert_true((y == Tensor.d2([[6.0], [12.0], [18.0]])))


fn test_weights_bias_gradients() raises:
    print("test_weights_bias_gradients")
    var xx = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var ww = Tensor.d2([[0.1], [0.2]], requires_grad=True)
    var bb = Tensor.d1([0.5], requires_grad=True)
    var target_ = Tensor.d2([[1.0], [2.0]])

    var y_prediction = xx.matmul(ww) + bb
    var _loss = ((y_prediction - target_) ** 2).mean([])
    _loss.backward()

    # Gradient should flow to w and b
    assert_true(ww.grad().shape() == ww.shape())
    assert_true(bb.grad().shape() == bb.shape())


fn test_training_convergence() raises:
    print("test_training_convergence")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    var y = Tensor.d2([[13.0], [23.0], [33.0]]).float()

    var w = Tensor.rand(2, 1, requires_grad=True)
    var b = Tensor.rand(1, requires_grad=True)
    var lr = Scalar[DType.float32](0.01)
    for _ in range(1000):
        var y_pred = x.matmul(w) + b
        var loss = ((y_pred - y) ** 2).mean()
        # loss.print()
        loss.backward()

        # SGD
        w.buffer -= lr * w.grad().buffer
        b.buffer -= lr * b.grad().buffer
        w.zero_grad()
        b.zero_grad()

    # After training
    w.print()
    b.print()
    # assert_true(w.all_close(Tensor.d2([[2.0], [3.0]])))
    # assert_true(b.all_close(Tensor[DType.float32].d1([5.0])))


fn test_transpose_gradients() raises:
    print("test_transpose_gradients")
    # Case 1: Simple 2D transpose
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = a.transpose()  # (2, 2) -> (2, 2)
    b.sum().backward()
    assert_true((a.grad() == Tensor.d2([[1, 1], [1, 1]])))

    # Case 2: Transpose + reshape with non-square
    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
    b = a.transpose()
    r = b.reshape(Shape.of(2, 3))  # (3, 2) → (2, 3)
    s = r.sum()
    s.backward()
    assert_true((a.grad() == Tensor.d2([[1, 1, 1], [1, 1, 1]])))

    # Case 3: Chain transposes (A.T().T())
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    a_t = a.transpose()
    b = a_t.transpose()  # Should equal A
    b.sum().backward()
    assert_true((a.grad() == Tensor.d2([[1, 1], [1, 1]])))


fn test_reshape_grad_flow() raises:
    """Test suite for gradient flow through reshape operations."""
    print("test_reshape_grad_flow")

    # === 1D Tensor Cases ===
    # Case 1: Simple 1D → 1D reshape
    var a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    var b = a.reshape(
        Shape.of(
            4,
        )
    )
    b.sum().backward()
    assert_true((a.grad() == Tensor.d1([1, 1, 1, 1])))

    # Case 2: 1D → 2D reshape
    a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    b = a.reshape(Shape.of(2, 2))
    (b * 2).sum().backward()
    assert_true((a.grad() == Tensor.d1([2, 2, 2, 2])))

    # === 2D Tensor Cases ===
    # Case 3: 2D → 2D reshape (contiguous)
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = a.reshape(Shape.of(4, 1))
    (b**2).sum().backward()
    assert_true((a.grad() == Tensor.d2([[2, 4], [6, 8]])))

    # Case 4: 2D → 1D reshape
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = a.reshape(
        Shape.of(
            4,
        )
    )
    (b + 1).sum().backward()
    assert_true((a.grad() == Tensor.d2([[1, 1], [1, 1]])))

    # === 3D Tensor Cases ===
    # Case 5: 3D → 2D reshape
    _ = """a64 = Tensor[DType.float64].d3([[[1.0, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    b = a.reshape(Shape.of(2, 4))
    b.mean().backward()
    assert_true((a64.grad() == Tensor[DType.float64].full(a64.shape(), 0.125)))

    # === Edge Cases ===
    # Case 6: Empty tensor reshape
    a = Tensor.d1([], requires_grad=True)
    b = a.reshape(Shape.of(0,))
    b.sum().backward()  # Should not crash
    assert_true(a.grad().shape() == Shape.of(0,))"""

    # Case 7: Non-contiguous reshape
    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = a.transpose().reshape(Shape.of(2, 3))  # Tests view tracking
    b.sum().backward()
    assert_true((a.grad() == Tensor.d2([[1, 1, 1], [1, 1, 1]])))

    # === Advanced Cases ===
    # Case 8: Chained reshapes
    a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    b = a.reshape(Shape.of(2, 2)).reshape(
        Shape.of(
            4,
        )
    )
    b.sum().backward()
    assert_true((a.grad() == Tensor.d1([1, 1, 1, 1])))

    # Case 9: Reshape with existing gradients
    a = Tensor.d1([1, 2], requires_grad=True)
    _ = (a * 2).sum().backward()  # a.grad = [2.0, 2.0]
    b = a.reshape(Shape.of(2, 1))
    b.sum().backward()  # Gradient accumulation
    assert_true((a.grad() == Tensor.d1([3, 3])))  # 2 + 1

    # Case 10: Reshape after detach
    _ = """a = Tensor.d1([1, 2], requires_grad=True)
    b = a.detach().reshape(Shape.of(2, 1))  # Should break grad flow
    b.sum().backward()  # Should NOT affect a.grad
    assert_true(a.grad() is None)  # Because of detach()"""


fn test_reshape_gradient() raises:
    print("test_reshape_gradient")
    # 1. Reshape scalar to (1,) and back
    a = Tensor.scalar(42, requires_grad=True)
    b = a.reshape(Shape.of(1))
    c = b.reshape(Shape.of(1))  # back to scalar
    d = c * Tensor.scalar(2)
    d.backward()
    assert_grad(a, Tensor.scalar(2), "scalar reshape chain → a")
    # 2. Reshape 1D → 2D → back to 1D
    a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    b = a.reshape(Shape.of(2, 2))
    c = b.reshape(
        Shape.of(
            4,
        )
    )
    d = c * Tensor.d1([10, 20, 30, 40])
    d.backward()
    assert_grad(a, Tensor.d1([10, 20, 30, 40]), "1D → 2D → 1D grad")

    # 3. Reshape 2D to 1D and multiply
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)  # shape (2,2)
    b = a.reshape(
        Shape.of(
            4,
        )
    )
    c = b * Tensor.d1([10, 20, 30, 40])
    c.backward()
    assert_grad(a, Tensor.d2([[10, 20], [30, 40]]), "2D → 1D grad")

    # 4. Reshape 3D to 1D and back
    a = Tensor.d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )  # shape (2,2,2)
    b = a.reshape(
        Shape.of(
            8,
        )
    )
    c = b.reshape(Shape.of(2, 2, 2))
    d = c * Tensor.d3(
        [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
        ]
    )
    d.backward()
    assert_grad(
        a,
        Tensor.d3(
            [
                [[10, 20], [30, 40]],
                [[50, 60], [70, 80]],
            ]
        ),
        "3D reshape roundtrip grad",
    )
    # 5. Reshape + sum + broadcast backward
    a = Tensor.d1([1, 2, 3, 4], requires_grad=True)  # shape (4,)
    b = a.reshape(Shape.of(2, 2))
    c = b.sum()  # scalar
    c.backward()
    assert_grad(a, Tensor.d1([1, 1, 1, 1]), "reshape → sum → backward")
    # 6. Reshape with degenerate axis: (4,) → (1, 4) → (4,)
    a = Tensor.d1([5, 6, 7, 8], requires_grad=True)
    b = a.reshape(Shape.of(1, 4))
    c = b.reshape(
        Shape.of(
            4,
        )
    )
    d = c * Tensor.d1([1, 2, 3, 4])
    d.backward()
    assert_grad(a, Tensor.d1([1, 2, 3, 4]), "reshape with (1,4) roundtrip")
    assert_true(
        (a.grad() == Tensor.d1([1, 2, 3, 4])),
        "reshape with (1,4) roundtrip",
    )
    # 7. Reshape then broadcast in op
    a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    b = a.reshape(Shape.of(2, 2))
    c = b + Tensor.scalar(10)  # broadcast add
    d = c.sum()
    d.backward()
    assert_grad(a, Tensor.d1([1, 1, 1, 1]), "reshape + broadcast add + sum")
    _ = """# 8. Illegal reshape (shape mismatch) — should raise error
    try:
        a = Tensor.d1([1, 2, 3, 4])
        _ = a.reshape(Shape(3, 2))  # invalid reshape
        assert_true(False, "reshape shape mismatch not caught")
    except ShapeError:
        pass  # expected"""


fn test_broadcast_mul() raises:
    print("test_broadcast_mul")
    # 1. Scalar * Scalar
    a = Tensor.scalar(3, requires_grad=True)
    b = Tensor.scalar(4, requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(a, Tensor.scalar(4), "Scalar * Scalar → a")
    assert_grad(b, Tensor.scalar(3), "Scalar * Scalar → b")
    do_assert(c, Tensor.scalar(12), "Scalar * Scalar")

    # 2. Scalar * 1D
    a = Tensor.scalar(2, requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(a, Tensor.scalar(6), "Scalar * 1D → a")
    assert_grad(b, Tensor.d1([2, 2, 2]), "Scalar * 1D → b")
    do_assert(c, Tensor.d1([2, 4, 6]), "Scalar * 1D")

    # 3. 1D * Scalar
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = Tensor.scalar(2, requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(a, Tensor.d1([2, 2, 2]), "1D * Scalar → a")
    assert_grad(b, Tensor.scalar(6), "1D * Scalar → b")
    do_assert(c, Tensor.d1([2, 4, 6]), "1D * Scalar")

    # 4. 1D * 1D
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = Tensor.d1([4, 5, 6], requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(a, Tensor.d1([4, 5, 6]), "1D * 1D → a")
    assert_grad(b, Tensor.d1([1, 2, 3]), "1D * 1D → b")
    do_assert(c, Tensor.d1([4, 10, 18]), "1D * 1D")

    # 5. 2D * Scalar
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor.scalar(3, requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(a, Tensor.d2([[3, 3], [3, 3]]), "2D * Scalar → a")
    assert_grad(b, Tensor.scalar(10), "2D * Scalar → b")
    do_assert(c, Tensor.d2([[3, 6], [9, 12]]), "2D * Scalar")

    # 6. Scalar * 2D
    a = Tensor.scalar(3, requires_grad=True)
    b = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(a, Tensor.scalar(10), "Scalar * 2D → a")
    assert_grad(b, Tensor.d2([[3, 3], [3, 3]]), "Scalar * 2D → b")
    do_assert(c, Tensor.d2([[3, 6], [9, 12]]), "Scalar * 2D")

    # 7. 2D * 1D (row-wise broadcasting)
    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor.d1([10, 20, 30], requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(a, Tensor.d2([[10, 20, 30], [10, 20, 30]]), "2D * 1D → a")
    assert_grad(b, Tensor.d1([5, 7, 9]), "2D * 1D → b")
    do_assert(c, Tensor.d2([[10, 40, 90], [40, 100, 180]]), "2D * 1D")

    # 8. 1D * 2D (reverse broadcast)
    a = Tensor.d1([10, 20, 30], requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(a, Tensor.d1([5, 7, 9]), "1D * 2D → a")
    assert_grad(b, Tensor.d2([[10, 20, 30], [10, 20, 30]]), "1D * 2D → b")
    do_assert(c, Tensor.d2([[10, 40, 90], [40, 100, 180]]), "1D * 2D")

    # 9. 3D * 1D (broadcast on last dim)
    a = Tensor.d3(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        requires_grad=True,
    )
    b = Tensor.d1([10, 20], requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(
        a,
        Tensor.d3(
            [
                [[10, 20], [10, 20]],
                [[10, 20], [10, 20]],
            ]
        ),
        "3D * 1D → a",
    )
    assert_grad(b, Tensor.d1([16, 20]), "3D * 1D → b")
    do_assert(
        c,
        Tensor.d3(
            [
                [[10, 40], [30, 80]],
                [[50, 120], [70, 160]],
            ]
        ),
        "3D * 1D",
    )

    # 10. 3D * 2D (broadcast over batch dim)
    a = Tensor.d3(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        requires_grad=True,
    )
    b = Tensor.d2([[10, 20], [30, 40]], requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(
        a,
        Tensor.d3(
            [
                [[10, 20], [30, 40]],
                [[10, 20], [30, 40]],
            ]
        ),
        "3D * 2D → a",
    )
    assert_grad(
        b,
        Tensor.d2(
            [
                [6, 8],
                [10, 12],
            ]
        ),
        "3D * 2D → b",
    )
    do_assert(
        c,
        Tensor.d3(
            [
                [[10, 40], [90, 160]],
                [[50, 120], [210, 320]],
            ]
        ),
        "3D * 2D",
    )

    # 11. 3D * Scalar
    a = Tensor.d3(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        requires_grad=True,
    )
    b = Tensor.scalar(10, requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(
        a,
        Tensor.d3(
            [
                [[10, 10], [10, 10]],
                [[10, 10], [10, 10]],
            ]
        ),
        "3D * Scalar → a",
    )
    assert_grad(b, Tensor.scalar(36), "3D * Scalar → b")
    do_assert(
        c,
        Tensor.d3(
            [
                [[10, 20], [30, 40]],
                [[50, 60], [70, 80]],
            ]
        ),
        "3D * Scalar",
    )

    # 12. Degenerate broadcast: (1,) * (3,)
    a = Tensor.d1([5], requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(a, Tensor.d1([6]), "(1,) * (3,) → a")
    assert_grad(b, Tensor.d1([5, 5, 5]), "(1,) * (3,) → b")
    do_assert(c, Tensor.d1([5, 10, 15]), "(1,) * (3,)")

    # 13. Degenerate broadcast: (1,1) * (2,3)
    a = Tensor.d2([[2]], requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a * b
    c.backward()
    assert_grad(a, Tensor.d2([[21]]), "(1,1) * (2,3) → a")
    assert_grad(b, Tensor.d2([[2, 2, 2], [2, 2, 2]]), "(1,1) * (2,3) → b")
    do_assert(c, Tensor.d2([[2, 4, 6], [8, 10, 12]]), "(1,1) * (2,3)")


fn test_broadcast_sub() raises:
    print("test_broadcast_sub")
    # 1. Scalar - Scalar
    X = Tensor.scalar(100, requires_grad=True)
    summ = (X - X).sum()
    summ.backward()
    assert_grad(X, Tensor.scalar(0), "(X - X) scalars → a")
    summ = (X - X - X - X).sum()
    summ.backward()
    assert_true(X.grad().item() == -2)
    assert_grad(X, Tensor.scalar(-2), "(X - X - X - X) scalars → a")
    a = Tensor.scalar(5, requires_grad=True)
    b = Tensor.scalar(3, requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(a, Tensor.scalar(1), "Scalar - Scalar → a")
    assert_grad(b, Tensor.scalar(-1), "Scalar - Scalar → b")
    do_assert(c, Tensor.scalar(2), "Scalar - Scalar")

    # 2. Scalar - 1D
    a = Tensor.scalar(10, requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(a, Tensor.scalar(3), "Scalar - 1D → a")
    assert_grad(b, Tensor.d1([-1, -1, -1]), "Scalar - 1D → b")
    do_assert(c, Tensor.d1([9, 8, 7]), "Scalar - 1D")

    # 3. 1D - Scalar
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = Tensor.scalar(10, requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(a, Tensor.d1([1, 1, 1]), "1D - Scalar → a")
    assert_grad(b, Tensor.scalar(-3), "1D - Scalar → b")
    do_assert(c, Tensor.d1([-9, -8, -7]), "1D - Scalar")

    # 4. 1D - 1D (same shape)
    a = Tensor.d1([5, 6, 7], requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(a, Tensor.d1([1, 1, 1]), "1D - 1D → a")
    assert_grad(b, Tensor.d1([-1, -1, -1]), "1D - 1D → b")
    do_assert(c, Tensor.d1([4, 4, 4]), "1D - 1D (same shape)")

    # 5. 2D - Scalar
    a = Tensor.d2([[10, 20], [30, 40]], requires_grad=True)
    b = Tensor.scalar(5, requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(a, Tensor.d2([[1, 1], [1, 1]]), "2D - Scalar → a")
    assert_grad(b, Tensor.scalar(-4), "2D - Scalar → b")
    do_assert(c, Tensor.d2([[5, 15], [25, 35]]), "2D - Scalar")

    # 6. Scalar - 2D
    a = Tensor.scalar(100, requires_grad=True)
    b = Tensor.d2([[10, 20], [30, 40]], requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(a, Tensor.scalar(4), "Scalar - 2D → a")
    assert_grad(b, Tensor.d2([[-1, -1], [-1, -1]]), "Scalar - 2D → b")
    do_assert(c, Tensor.d2([[90, 80], [70, 60]]), "Scalar - 2D")

    # 7. 2D - 1D (broadcast over rows)
    a = Tensor.d2([[10, 20, 30], [40, 50, 60]], requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(a, Tensor.d2([[1, 1, 1], [1, 1, 1]]), "2D - 1D → a")
    assert_grad(b, Tensor.d1([-2, -2, -2]), "2D - 1D → b")
    do_assert(c, Tensor.d2([[9, 18, 27], [39, 48, 57]]), "2D - 1D")

    # 8. 1D - 2D (reverse broadcast over rows)
    a = Tensor.d1([100, 200, 300], requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(a, Tensor.d1([2, 2, 2]), "1D - 2D → a")
    assert_grad(b, Tensor.d2([[-1, -1, -1], [-1, -1, -1]]), "1D - 2D → b")
    do_assert(c, Tensor.d2([[99, 198, 297], [96, 195, 294]]), "1D - 2D")

    # 9. 3D - 1D (broadcast over last dim)
    a = Tensor.d3(
        [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
        ],
        requires_grad=True,
    )
    b = Tensor.d1([1, 2], requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(
        a,
        Tensor.d3(
            [
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
            ]
        ),
        "3D - 1D → a",
    )
    assert_grad(b, Tensor.d1([-4, -4]), "3D - 1D → b")
    do_assert(
        c,
        Tensor.d3(
            [
                [[9, 18], [29, 38]],
                [[49, 58], [69, 78]],
            ]
        ),
        "3D - 1D",
    )

    # 10. 3D - 2D (broadcast over batch dim)
    a = Tensor.d3(
        [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
        ],
        requires_grad=True,
    )
    b = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(
        a,
        Tensor.d3(
            [
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
            ]
        ),
        "3D - 2D → a",
    )
    assert_grad(
        b, Tensor.d2([[-2.0, -2.0], [-2.0, -2.0]]).float(), "3D - 2D → b"
    )
    do_assert(
        c,
        Tensor.d3(
            [
                [[9, 18], [27, 36]],
                [[49, 58], [67, 76]],
            ]
        ),
        "3D - 2D",
    )

    # 11. 3D - Scalar
    a = Tensor.d3(
        [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
        ],
        requires_grad=True,
    )
    b = Tensor.scalar(5, requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(
        a,
        Tensor.d3(
            [
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
            ]
        ),
        "3D - Scalar → a",
    )
    assert_grad(b, Tensor.scalar(-8), "3D - Scalar → b")
    do_assert(
        c,
        Tensor.d3(
            [
                [[5, 15], [25, 35]],
                [[45, 55], [65, 75]],
            ]
        ),
        "3D - Scalar",
    )

    # 12. Degenerate shape: (1,) - (N,)
    a = Tensor.d1([100], requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(a, Tensor.d1([3]), "(1,) - (3,) → a")
    assert_grad(b, Tensor.d1([-1, -1, -1]), "(1,) - (3,) → b")
    do_assert(c, Tensor.d1([99, 98, 97]), "(1,) - (3,)")

    # 13. Degenerate broadcast: (1, 1) - (2, 3)
    a = Tensor.d2([[100]], requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a - b
    c.backward()
    assert_grad(a, Tensor.d2([[6]]), "(1,1) - (2,3) → a")
    assert_grad(b, Tensor.d2([[-1, -1, -1], [-1, -1, -1]]), "(1,1) - (2,3) → b")
    do_assert(c, Tensor.d2([[99, 98, 97], [96, 95, 94]]), "(1,1) - (2,3)")


fn test_broadcast_add() raises:
    print("test_broadcast_add")
    # 1. Scalar + Scalar
    a = Tensor.scalar(5, requires_grad=True)
    b = Tensor.scalar(3, requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(a, Tensor.scalar(1), "Scalar + Scalar → a")
    assert_grad(b, Tensor.scalar(1), "Scalar + Scalar → b")
    do_assert(c, Tensor.scalar(8), "Scalar + Scalar")

    # 2. Scalar + 1D
    a = Tensor.scalar(2, requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(a, Tensor.scalar(3), "Scalar + 1D → a")
    assert_grad(b, Tensor.d1([1, 1, 1]), "Scalar + 1D → b")
    do_assert(c, Tensor.d1([3, 4, 5]), "Scalar + 1D")

    # 3. 1D + Scalar (reverse)
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = Tensor.scalar(2, requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(a, Tensor.d1([1, 1, 1]), "1D + Scalar → a")
    assert_grad(b, Tensor.scalar(3), "1D + Scalar → b")
    do_assert(c, Tensor.d1([3, 4, 5]), "1D + Scalar")

    # 4. 1D + 1D (same shape)
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = Tensor.d1([4, 5, 6], requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(a, Tensor.d1([1, 1, 1]), "1D + 1D → a")
    assert_grad(b, Tensor.d1([1, 1, 1]), "1D + 1D → b")
    do_assert(c, Tensor.d1([5, 7, 9]), "1D + 1D (same shape)")

    # 5. 2D + Scalar
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor.scalar(10, requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(a, Tensor.d2([[1, 1], [1, 1]]), "2D + Scalar → a")
    assert_grad(b, Tensor.scalar(4), "2D + Scalar → b")
    do_assert(c, Tensor.d2([[11, 12], [13, 14]]), "2D + Scalar")

    # 6. Scalar + 2D (reverse)
    a = Tensor.scalar(10, requires_grad=True)
    b = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(a, Tensor.scalar(4), "Scalar + 2D → a")
    assert_grad(b, Tensor.d2([[1, 1], [1, 1]]), "Scalar + 2D → b")
    do_assert(c, Tensor.d2([[11, 12], [13, 14]]), "Scalar + 2D")

    # 7. 2D + 1D (broadcast over rows)
    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor.d1([10, 20, 30], requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(a, Tensor.d2([[1, 1, 1], [1, 1, 1]]), "2D + 1D → a")
    assert_grad(b, Tensor.d1([2, 2, 2]), "2D + 1D → b")
    do_assert(c, Tensor.d2([[11, 22, 33], [14, 25, 36]]), "2D + 1D row-wise")

    # 8. 1D + 2D (reverse broadcast over rows)
    a = Tensor.d1([10, 20, 30], requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(a, Tensor.d1([2, 2, 2]), "1D + 2D → a")
    assert_grad(b, Tensor.d2([[1, 1, 1], [1, 1, 1]]), "1D + 2D → b")
    do_assert(c, Tensor.d2([[11, 22, 33], [14, 25, 36]]), "1D + 2D row-wise")

    # 9. 3D + 1D (broadcast over last dim)
    a = Tensor.d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    b = Tensor.d1([10, 20], requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(
        a, Tensor.d3([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), "3D + 1D → a"
    )
    assert_grad(b, Tensor.d1([4, 4]), "3D + 1D → b")
    do_assert(
        c,
        Tensor.d3([[[11, 22], [13, 24]], [[15, 26], [17, 28]]]),
        "3D + 1D last-dim broadcast",
    )

    # 10. 3D + 2D (broadcast over batch dim)
    a = Tensor.d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    b = Tensor.d2([[10, 20], [30, 40]], requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(
        a, Tensor.d3([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), "3D + 2D → a"
    )
    assert_grad(b, Tensor.d2([[2, 2], [2, 2]]), "3D + 2D → b")
    do_assert(
        c,
        Tensor.d3([[[11, 22], [33, 44]], [[15, 26], [37, 48]]]),
        "3D + 2D batch-dim broadcast",
    )

    # 11. 3D + Scalar
    a = Tensor.d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    b = Tensor.scalar(100, requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(
        a, Tensor.d3([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), "3D + Scalar → a"
    )
    assert_grad(b, Tensor.scalar(8), "3D + Scalar → b")
    do_assert(
        c,
        Tensor.d3([[[101, 102], [103, 104]], [[105, 106], [107, 108]]]),
        "3D + Scalar",
    )

    # 12. Broadcast shape mismatch (should fail)
    _ = """try:
        a = Tensor.d2([[1, 2], [3, 4]])
        b = Tensor.d1([1, 2, 3])
        _c = a + b
        assert_true(False, "Shape mismatch not caught")
    except BroadcastError:
        pass  # expected"""

    # 13. Degenerate shape: (1,) + (N,)
    a = Tensor.d1([1], requires_grad=True)
    b = Tensor.d1([10, 20, 30], requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(a, Tensor.d1([3]), "(1,) + (3,) → a")
    assert_grad(b, Tensor.d1([1, 1, 1]), "(1,) + (3,) → b")
    do_assert(c, Tensor.d1([11, 21, 31]), "(1,) + (3,)")

    # 14. Degenerate broadcast: (1, 1) + (2, 3)
    a = Tensor.d2([[5]], requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a + b
    c.backward()
    assert_grad(a, Tensor.d2([[6]]), "(1,1) + (2,3) → a")
    assert_grad(b, Tensor.d2([[1, 1, 1], [1, 1, 1]]), "(1,1) + (2,3) → b")
    do_assert(c, Tensor.d2([[6, 7, 8], [9, 10, 11]]), "(1,1) + (2,3)")


fn test_power() raises:
    print("test_power")
    tensor = Tensor.arange(1 * 2 * 3).reshape(1, 2, 3)
    result = tensor**2
    assert_true(result.all_close(Tensor.d3([[[0, 1, 4], [9, 16, 25]]])))


fn test_grad_flow_through_reshape() raises:
    print("test_grad_flow_through_reshape")
    a = Tensor.of(1.0, 2.0, 3.0, requires_grad=True)

    # First operation using 'a'
    b = a + 1.0
    b.sum().backward()
    assert_true((a.grad() == Tensor.of(1.0, 1.0, 1.0)))

    # Reshape should not clone or copy gradients
    reshaped = a.reshape(Shape.of(3))

    # reshaped.grad() should not exist on its own — we assert that it refers to the same grad storage
    # assert_true((reshaped.grad() == Tensor.of(1.0, 1.0, 1.0)))

    # New operation from reshaped
    (reshaped * 2).sum().backward()

    # Original 'a' should now have accumulated gradient
    assert_true((a.grad() == Tensor.of(3.0, 3.0, 3.0)))


fn test_reshape_preserves_grad_accumulation() raises:
    print("test_reshape_preserves_grad_accumulation")
    # Chained reshape should still accumulate gradients
    a = Tensor.of(1.0, 2.0, 3.0, requires_grad=True)
    b = a.reshape(Shape.of(3))
    c = b.reshape(Shape.of(1, 3))

    d = c.sum()
    d.backward()

    assert_true((a.grad() == Tensor.of(1.0, 1, 1)))


fn test_multi_dimensional_reshape() raises:
    print("test_multi_dimensional_reshape")
    # (2, 3) → (3, 2)
    a1 = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b1 = a1.reshape(Shape.of(3, 2))

    assert_true(b1.shape() == Shape.of(3, 2))
    d1 = b1.sum()
    d1.backward()
    assert_true(
        a1.grad().all_close(Tensor.d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    )


fn test_reshape_tensor_to_scalar() raises:
    print("test_reshape_tensor_to_scalar")
    # (1,) → reshape to scalar
    a = Tensor.of(42.0, requires_grad=True)
    b = a.reshape(Shape())

    assert_true(b.is_scalar())
    assert_true(b[IntList()] == Scalar(42.0))

    c = b * 2
    c.backward()


fn test_reshape_scalar_to_tensor() raises:
    print("test_reshape_scalar_to_tensor")
    # Scalar → reshape to (1,)
    a = Tensor.scalar(42.0, requires_grad=True)
    b = a.reshape(Shape.of(1))  # should share data and allow backprop

    assert_true(b[0] == Scalar(42.0))
    c = b * 3
    c.backward()
    assert_true(b.grad().item() == 0 and a.gradbox[].item() == 3)


fn test_miscellaneous() raises:
    print("test_miscellaneous")
    a = Tensor.of(1.0, 2.0, 3.0, requires_grad=True)
    b = Tensor.scalar(5.0)
    c = a + b
    c.sum().backward()
    # should be [1, 1, 1]
    assert_true((a.grad() == Tensor.of(1.0, 1, 1)))
    reshaped = (a + b).mean().reshape()
    Tensor.scalar(42, requires_grad=True).sum().backward()  # This one crashes
    reshaped.backward()  # backward does not return anything


fn test_mean() raises:
    print("test_mean")
    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = a.mean([0])
    assert_true((b == Tensor[DType.float32].d1([2.5, 3.5, 4.5])))
    b.backward()
    assert_true(
        (
            a.grad()
            == Tensor[DType.float32].d2([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        )
    )
    # Mean over all → scalar
    s = a.mean([])
    assert_true((s == Tensor[DType.float32].scalar(3.5)))
    s.backward()
    # a.grad == [[1/6, 1/6, 1/6], [1/6, 1/6, 1/6]] + 0.5 from previous backward call

    a_grad = (
        Tensor.d2(
            [
                [0.1666667, 0.1666667, 0.1666667],
                [0.1666667, 0.1666667, 0.1666667],
            ]
        )
        + 0.5
    ).float()

    assert_true(a.grad().all_close(a_grad))
    a.zero_grad()
    s.zero_grad()
    s.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2(
                [
                    [0.1666666, 0.1666666, 0.1666666],
                    [0.1666666, 0.1666666, 0.1666666],
                ]
            ).float()
        )
    )


fn test_sum() raises:
    print("test_sum")
    # 1. Basic Value Tests
    a = Tensor.of(1, 2, 3)
    b = Tensor.d1([1, 2, 3])
    c = Tensor.of([1, 2, 3])
    assert_true((a.sum([0]) == Tensor.scalar(6)))
    assert_true((b.sum([0]) == Tensor.scalar(6)))
    assert_true((c.sum([0]) == Tensor.scalar(6)))
    assert_true((a.sum([0], keepdims=True) == Tensor.of(6)))
    assert_true((a.sum([0], keepdims=True) == Tensor.of([6])))

    # 2. Multi-Dimensional Tensor Tests
    a = Tensor.d2([[1, 2], [3, 4]])  # Shape (2, 2)
    assert_true((a.sum([0]) == Tensor.of([4, 6])))
    assert_true((a.sum([1]) == Tensor.of([3, 7])))
    assert_true((a.sum([0, 1]) == Tensor.scalar(10)))
    assert_true((a.sum([0, 1], keepdims=True) == Tensor.d2([[10]])))
    # 3. Scalar Input
    a = Tensor.scalar(42)
    assert_true((a.sum([]) == Tensor.scalar(42)))
    # assert_true((a.sum([0]) == Tensor.scalar(42)))
    # 4. Keepdims=True
    a = Tensor.d2([[1, 2], [3, 4]])  # (2,2)
    out = a.sum([1], keepdims=True)  # Should be (2,1)
    assert_true(
        (out == Tensor.d2([[3], [7]])) and out.shape() == Shape.of(2, 1)
    )
    # 5. Gradient Checks
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = a.sum([1])  # b shape (2,)
    b.backward()
    # Now a.grad should be Tensor.of([[1, 1], [1, 1]])
    assert_true(
        (b == Tensor.of(3, 7))
        and b.requires_grad
        and (a.grad() == Tensor.d2([[1, 1], [1, 1]]))
    )

    # 6. Broadcasting Compatibility
    a = Tensor.d2([[1, 2, 3]], requires_grad=True)  # (1,3)
    b = a.sum([0], keepdims=False)  # (3,)
    b.backward()
    # a.grad == Tensor.of([[1, 1, 1]])
    assert_true(
        (b == Tensor.of(1, 2, 3))
        and b.requires_grad
        and (a.grad() == Tensor.d2([[1, 1, 1]]))
    )
    tensor = Tensor.of(1, 2, 3, 4, requires_grad=True)
    result = tensor.sum(axes=[], keepdims=False)
    assert_true((result == Tensor.scalar(10)))
    result.backward()
    assert_true((tensor.grad() == Tensor[DType.float32].of(1.0, 1.0, 1.0, 1.0)))
    tensor = Tensor.arange(24).reshape(2, 3, 4)
    result = tensor.sum(axes=[], keepdims=False)
    assert_true(result.item() == 276.0)
    result = tensor.sum(axes=[], keepdims=True)
    assert_true((result == Tensor.d3([[[276.0]]]).float()))

    ones = Tensor.ones(3, 3)
    summed = ones.sum(axes=[0], keepdims=True)
    assert_true(
        (summed == Tensor.d2([[3, 3, 3]])),
        "keepdim = True sum assertion 1 failed",
    )
    ones = Tensor.ones(3, 3)
    summed = ones.sum(axes=[0])
    expect = Tensor.of(3, 3, 3)
    assert_true((summed == expect), "1D sum assertion failed")

    tensor = Tensor.arange(1, 21).reshape(2, 5, 2)
    summed = tensor.sum(axes=[1])
    _ = """[2D Tensor(2, 2), Type: float32, requires_grad: False]
        [
            [25.0, 30.0, ],
            [75.0, 80.0, ],
    ]"""
    expect = Tensor.of[2](25, 30, 75, 80)
    assert_true((summed == expect), "Sum across axis 1 assertion failed")

    summed = tensor.sum(axes=[0])
    expect = Tensor.of[2](12, 14, 16, 18, 20, 22, 24, 26, 28, 30)
    assert_true((summed == expect), "Sum across axis 0 assertion failed")

    expect = Tensor.scalar(210)
    summed = tensor.sum()
    assert_true((summed == expect), "Sum across axis 2 assertion failed")


fn test_broadcast_add_2_tensors() raises:
    print("test_broadcast_add_2_tensors")
    print("Test broadcast add 2 tensors")

    tensor1 = Tensor.of(1, 2, 3, 4, requires_grad=True)
    tensor2 = Tensor.of(6, requires_grad=True)

    result = tensor1 + tensor2
    assert_true(
        (result == Tensor.of(7, 8, 9, 10)),
        "broadcast add assertion 1 failed",
    )
    result.backward()

    assert_true(
        (
            tensor1.grad()
            == Tensor.d1(
                [
                    1,
                    1,
                    1,
                    1,
                ]
            )
        ),
        "grad check 1 - assertion failed",
    )

    assert_true(
        (tensor2.grad() == Tensor.of([4])),
        "grad check 2 - assertion failed",
    )

    tensor1 = Tensor.of[3](1, 2, 3, 4, 5, 6, requires_grad=True)
    tensor2 = Tensor.of(6, requires_grad=True)

    result = tensor1 + tensor2

    result.backward()

    assert_true(
        (result == Tensor.of[3](7, 8, 9, 10, 11, 12)),
        "broadcast add assertion 2 failed",
    )

    assert_true(
        (
            tensor1.grad()
            == Tensor.d2(
                [
                    [
                        1,
                        1,
                        1,
                    ],
                    [
                        1,
                        1,
                        1,
                    ],
                ]
            )
        ),
        "grad check 3 - assertion failed",
    )

    assert_true(
        (tensor2.grad() == Tensor.of([6])),
        "grad check 4 - assertion failed",
    )

    tensor1 = Tensor.rand(
        2, 1, 4, 1, init_seed=Optional(42), requires_grad=True
    )
    tensor2 = Tensor.rand(2, 1, 5, init_seed=Optional(42), requires_grad=True)

    result = tensor1 + tensor2

    result.backward()

    assert_true(
        (
            tensor1.grad()
            == Tensor.d4(
                [
                    [
                        [
                            [
                                10,
                            ],
                            [
                                10,
                            ],
                            [
                                10,
                            ],
                            [
                                10,
                            ],
                        ],
                    ],
                    [
                        [
                            [
                                10,
                            ],
                            [
                                10,
                            ],
                            [
                                10,
                            ],
                            [
                                10,
                            ],
                        ],
                    ],
                ]
            )
        ),
        "grad check 5 - assertion failed",
    )

    assert_true(
        (
            tensor2.grad()
            == Tensor[DType.float32].d3(
                [
                    [
                        [
                            8.0,
                            8.0,
                            8.0,
                            8.0,
                            8.0,
                        ],
                    ],
                    [
                        [
                            8.0,
                            8.0,
                            8.0,
                            8.0,
                            8.0,
                        ],
                    ],
                ]
            )
        ),
        "grad check 6 - assertion failed",
    )


fn test_add_2_tensors() raises:
    print("test_add_2_tensors")

    tensor_a = Tensor.rand(128, 128, requires_grad=True)
    tensor_b = Tensor.rand(128, 128, requires_grad=True)
    assert_true(
        tensor_a.shape() == tensor_b.shape(),
        "Input tensors shape match assertion failed",
    )
    out_tensor = tensor_a + tensor_b
    assert_true(
        tensor_a.shape() == out_tensor.shape(),
        "Input/output tensors shape match assertion failed",
    )


fn test_arange() raises:
    print("test_arange")
    tensor = Tensor.arange(0, 10)
    expected = Tensor.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    is_true = tensor == expected
    assert_true(is_true, "arange gen check assertion failed")

    tensor1 = Tensor.arange(0, -5, -0.5)
    expected1 = Tensor.of(
        0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5
    )
    is_true = tensor1 == expected1
    assert_true(is_true, "arange negative step assertion failed")


fn test_random() raises:
    print("test_random")
    rand_tensor = Tensor.rand(10)

    fn each(e: Scalar[DType.float32]) -> Bool:
        return e >= 0 and e < 1

    holds_true = rand_tensor.all(each)
    assert_true(holds_true, "rand min and max range assertion failed")

    rand_tensor2 = Tensor.rand(10, 20, min=-2, max=2)

    fn each2(e: Scalar[DType.float32]) -> Bool:
        return e >= -2 and e < 2

    holds_true = rand_tensor2.all(each2)
    assert_true(holds_true, "rand min(-2) and max(2) range assertion failed")


fn test_item() raises:
    print("test_item")
    tensor = Tensor.of(42)
    assert_true(tensor.item() == 42)


fn test_view() raises:
    print("test_view")
    tensor = Tensor.rand(1).reshape()
    view = tensor.view(Shape())
    assert_true(
        tensor.shape() == view.shape(),
        "Tensor and view shape equality asserttion failed",
    )


fn test_tensor_of_list() raises:
    print("test_tensor_of_list")
    tensor = Tensor.d1([1, 3, 4, 5])
    assert_true(
        tensor.numels() == 4 and tensor.dtype == DType.float32,
        "Tensor from list assertion 1 failed",
    )
    tensor_int32 = Tensor[DType.int32].d1([1, 3, 4, 5])
    assert_true(
        tensor_int32.numels() == 4 and tensor_int32.dtype == DType.int32,
        "Tensor from list assertion 2 failed",
    )
    tensor_2d = Tensor.d2(
        List(List(1.0, 2, 3), List(4.0, 5, 6), List(7.0, 8, 9))
    )
    assert_true(
        tensor_2d.shape() == Shape.of(3, 3) and tensor_2d.numels() == 9,
        "Tensor from assertion 3 failed",
    )
    tensor2d = Tensor.d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert_true(
        tensor2d.shape() == Shape.of(3, 3) and tensor2d.numels() == 9,
        "Tensor from assertion 3 failed",
    )

    tensor3d = Tensor.d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert_true(
        tensor3d.shape() == Shape.of(2, 2, 2) and tensor3d.numels() == 8,
        "Tensor from assertion 4 failed",
    )


fn test_scalar_tensor() raises:
    print("test_scalar_tensor")
    tensor = Tensor.scalar(42)
    assert_true(
        (
            tensor.item() == 42.0
            and tensor.shape() == Shape()
            and tensor.numels() == 1
        ),
        "Scalar tensor item and shape assertion failed",
    )


fn test_reshape() raises:
    print("test_reshape")
    tensor = Tensor.rand(3, 3)
    reshaped = tensor.reshape(9)
    assert_true(
        tensor[2, 2] == reshaped[8], "reshape __getitem__ assertion 1 failed"
    )
    assert_true(
        tensor.reshape(1, 9)[0, 8] == tensor[2, 2],
        "reshape __getitem__ assertion 2 failed",
    )
    assert_true(
        tensor.reshape(9, 1)[0, 0] == tensor[0, 0],
        "reshape __getitem__ assertion 3 failed",
    )

    tensor = Tensor.of(42)
    assert_true(
        tensor.shape() == Shape(1), "Unit tensor shape assertion failure"
    )
    reshaped = tensor.reshape(1, 1)
    assert_true(
        reshaped.shape() == Shape.of(1, 1) and reshaped[0, 0] == tensor[0],
        "post reshape shape and get assertion failed",
    )
    tensor = Tensor.scalar(42)
    reshaped = tensor.reshape(1, 1)
    assert_true(
        reshaped.shape() == Shape.of(1, 1) and reshaped[0, 0] == tensor.item(),
        "post reshape shape and get assertion failed for scalar tensor",
    )
    reshaped = tensor.reshape(1)
    assert_true(
        reshaped.shape() == Shape(1) and reshaped[0] == tensor.item(),
        "post reshape 2 - shape and get assertion failed for scalar tensor",
    )
    assert_true(
        reshaped.reshape(1, 1, 1, 1)[0, 0, 0, 0] == tensor.item(),
        "post reshape 3 - item assertion failed for scalar tensor",
    )

    tensor = Tensor.rand(1, 1)
    reshaped = tensor.reshape()
    assert_true(
        reshaped.shape() == Shape() and reshaped.item() == tensor[0, 0],
        "post reshape random tensor - shape and get assertion failed",
    )
    tensor = Tensor.scalar(42, requires_grad=True)
    result = tensor * 3
    result.backward()
    assert_true(tensor.grad().item() == 3.0)
    tensor2 = tensor.reshape(1)
    result = tensor2 * 42

    result.backward()

    tensor3 = tensor2.reshape(1, 1, 1, 1, 1)
    result = tensor3 * 12

    result.backward()


fn test_tensor_multiplications() raises:
    print("test_tensor_multiplications")
    test_scalar_mul_scalar()
    test_1d_mul_1d_same_shape()
    test_2d_mul_2d_same_shape()
    test_broadcast_2d_1d_mul()
    test_broadcast_1d_2d_mul()
    test_3d_broadcast_mul()
    test_scalar_tensor_mul()
    test_mul_one_requires_grad()


fn test_scalar_addition() raises:
    print("test_scalar_addition")
    var a = Tensor.scalar(3.0, requires_grad=True)
    var b = Tensor.scalar(4.0, requires_grad=True)
    var c = a + b
    c.backward()
    assert_true(c.item() == 7.0)
    assert_true(a.grad().item() == 1.0)
    assert_true(b.grad().item() == 1.0)


fn test_broadcast_addition() raises:
    print("test_broadcast_addition")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var b = Tensor.d1([10, 20], requires_grad=True)
    var c = a + b  # shape (2,2)
    s = c.sum()
    s.backward()
    assert_true((c == Tensor.d2([[11, 22], [13, 24]])))
    assert_true(a.grad().all_close(Tensor.d2([[1, 1], [1, 1]])))
    assert_true(
        b.grad().all_close(Tensor.d1([2, 2]))
    )  # Summed over broadcast dim


fn test_sum_all_dims() raises:
    print("test_sum_all_dims")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum()  # scalar
    s.backward()
    assert_true(s.item() == 10.0)
    assert_true(a.grad().all_close(Tensor.d2([[1, 1], [1, 1]])))


fn test_sum_specific_axis() raises:
    print("test_sum_specific_axis")
    var a = Tensor.d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    var s = a.sum(axes=[1], keepdims=True)  # shape (2,1,2)
    s.backward()
    assert_true((s == Tensor.d3([[[4, 6]], [[12, 14]]])))
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_mean_with_keepdims() raises:
    print("test_mean_with_keepdims")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var m = a.mean(axes=[0], keepdims=True)  # shape (1,2)
    s = m.sum()
    s.backward()
    assert_true(m.all_close(Tensor.d2([[2, 3]])))
    assert_true(a.grad().all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]]).float()))


fn test_matmul_shapes() raises:
    print("test_matmul_shapes")
    # Test various matmul shape combinations
    var m1 = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var m2 = Tensor.d2([[5, 6], [7, 8]], requires_grad=True)
    var mm = m1.matmul(m2)
    ss = mm.sum()
    ss.backward()
    assert_true(mm.all_close(Tensor.d2([[19, 22], [43, 50]])))
    assert_true(m1.grad().all_close(Tensor.d2([[11, 15], [11, 15]])))
    assert_true(m2.grad().all_close(Tensor.d2([[4, 4], [6, 6]])))


fn test_matmul_broadcasting() raises:
    print("test_matmul_broadcasting")
    # Batch matmul
    var a = Tensor.d3([[[1, 2]], [[3, 4]]], requires_grad=True)  # shape (2,1,2)
    var b = Tensor.d3([[[5], [6]]], requires_grad=True)  # shape (1,2,1)
    var c = a.matmul(b)  # shape (2,2,1)
    c.sum().backward()
    assert_true(c.all_close(Tensor.d3([[[17]], [[39]]])))


fn test_zero_grad() raises:
    print("test_zero_grad")
    var a = Tensor.scalar(1.0, requires_grad=True)
    var b = a * 2
    b.backward()
    a.zero_grad()
    assert_true(a.grad().item() == 0.0)


fn test_transpose_grad() raises:
    print("test_transpose_grad")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var b = a.transpose()
    var c = b * Tensor.d2([[10, 30], [20, 40]])
    s = c.sum()
    s.backward()
    assert_true(a.grad().all_close(Tensor.d2([[10, 20], [30, 40]])))


fn test_scalar_div_tensor() raises:
    var a = Tensor.d1([2.0, 4.0], requires_grad=True)
    var out = 8.0 / a

    assert_true(
        out.all_close(Tensor.d1([4.0, 2.0])),
        "Forward: scalar / tensor incorrect",
    )

    out.sum().backward()

    # dz/da = -8 / a^2 ⇒ [-2.0, -0.5]
    assert_true(
        a.grad().all_close(Tensor.d1([-2.0, -0.5])),
        "Backward gradient incorrect",
    )


fn test_scalar_div_tensor_multiple() raises:
    var a = Tensor.d1([1.0, 2.0, 4.0], requires_grad=True)
    var out = 8.0 / a

    assert_true(
        out.all_close(Tensor.d1([8.0, 4.0, 2.0])), "Forward scalar / tensor"
    )

    out.sum().backward()

    # ∂/∂a: -8 / a^2 ⇒ [-8.0, -2.0, -0.5]
    assert_true(
        a.grad().all_close(Tensor.d1([-8.0, -2.0, -0.5])),
        "Backward grad mismatch",
    )


fn test_scalar_div_tensor_2d() raises:
    var a = Tensor.d2([[1.0, 2.0], [4.0, 8.0]], requires_grad=True)
    var out = 16.0 / a

    assert_true(
        out.all_close(Tensor.d2([[16.0, 8.0], [4.0, 2.0]])),
        "Forward output incorrect",
    )

    out.sum().backward()

    # Gradient: -16 / a^2
    assert_true(
        a.grad().all_close(Tensor.d2([[-16.0, -4.0], [-1.0, -0.25]])),
        "Backward gradient failed",
    )


fn test_mul_same_shape() raises:
    print("test_mul_same_shape")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    var c = a * b
    assert_true(c.all_close(Tensor.d2([[5.0, 12.0], [21.0, 32.0]])))
    c.sum().backward()
    assert_true(a.grad().all_close(b))
    assert_true(b.grad().all_close(a))


fn test_mul_tensor_scalar() raises:
    print("test_mul_tensor_scalar")
    var a = Tensor.d2([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
    var b = Tensor.scalar(3, requires_grad=True).float64()
    var c = a * b
    assert_true(c.all_close(Tensor.d2([[6.0, 12.0], [18.0, 24.0]])))
    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[3.0, 3.0], [3.0, 3.0]])))
    assert_true(b.grad().item() == 20.0)  # 2+4+6+8 = 20


fn test_mul_scalar_tensor() raises:
    print("test_mul_scalar_tensor")
    var a = Tensor.scalar(5.0, requires_grad=True)
    var b = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var c = a * b
    assert_true(c.all_close(Tensor.d2([[5.0, 10.0], [15.0, 20.0]])))
    c.sum().backward()
    assert_true(a.grad().item() == 10.0)  # 1+2+3+4
    assert_true(b.grad().all_close(Tensor.d2([[5.0, 5.0], [5.0, 5.0]])))


fn test_mul_broadcast_row() raises:
    print("test_mul_broadcast_row")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.d1([10.0, 20.0], requires_grad=True)
    var c = a * b  # row-wise broadcast
    assert_true(c.all_close(Tensor.d2([[10.0, 40.0], [30.0, 80.0]])))
    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[10.0, 20.0], [10.0, 20.0]])))
    assert_true(
        b.grad().all_close(Tensor.d1([4.0, 6.0]))
    )  # col sums: [1+3, 2+4]


fn test_mul_broadcast_col() raises:
    print("test_mul_broadcast_col")
    var a = Tensor.d2([[1.0], [2.0], [3.0]], requires_grad=True)  # shape [3,1]
    var b = Tensor.d2([[4.0, 5.0]], requires_grad=True)  # shape [1,2]
    var c = a * b  # broadcast to [3,2]
    assert_true(c.all_close(Tensor.d2([[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]])))
    c.sum().backward()
    assert_true(
        a.grad().all_close(Tensor.d2([[9.0], [9.0], [9.0]]))
    )  # sum along row for each col
    assert_true(
        b.grad().all_close(Tensor.d2([[6.0, 6.0]]))
    )  # sum along col for each row


fn test_sub_same_shape() raises:
    print("test_sub_same_shape")
    var a = Tensor.d2([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    var b = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var c = a - b
    assert_true(c.all_close(Tensor.d2([[2.0, 2.0], [2.0, 2.0]])))

    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))
    assert_true(b.grad().all_close(Tensor.d2([[-1.0, -1.0], [-1.0, -1.0]])))


fn test_sub_broadcast_row() raises:
    print("test_sub_broadcast_row")
    var a = Tensor.d2([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    var b = Tensor.d1([1.0, 2.0], requires_grad=True)
    var c = a - b
    assert_true(c.all_close(Tensor.d2([[9.0, 18.0], [29.0, 38.0]])))

    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))
    assert_true(b.grad().all_close(Tensor.d1([-2.0, -2.0])))


fn test_sub_scalar_tensor() raises:
    print("test_sub_scalar_tensor")
    var a = Tensor.scalar(10.0, requires_grad=True)
    var b = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var c = a - b
    assert_true(c.all_close(Tensor.d2([[9.0, 8.0], [7.0, 6.0]])))

    c.sum().backward()
    assert_true(a.grad().item() == 4.0)  # 4 elements
    assert_true(b.grad().all_close(Tensor.d2([[-1.0, -1.0], [-1.0, -1.0]])))


fn test_sub_tensor_scalar() raises:
    print("test_sub_tensor_scalar")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.scalar(1.5, requires_grad=True)
    var c = a - b
    assert_true(c.all_close(Tensor.d2([[-0.5, 0.5], [1.5, 2.5]])))

    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))
    assert_true(b.grad().item() == -4.0)


fn test_sub_broadcast_col() raises:
    print("test_sub_broadcast_col")
    var a = Tensor.d2([[10.0], [20.0]], requires_grad=True)  # shape: [2, 1]
    var b = Tensor.d2([[1.0, 2.0]], requires_grad=True)  # shape: [1, 2]
    var c = a - b  # broadcast to [2, 2]
    assert_true(c.all_close(Tensor.d2([[9.0, 8.0], [19.0, 18.0]])))

    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[2.0], [2.0]])))
    assert_true(b.grad().all_close(Tensor.d2([[-2.0, -2.0]])))


fn test_add_scalar_scalar() raises:
    print("test_add_scalar_scalar")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = Tensor.scalar(3.0, requires_grad=True)
    var c = a + b
    assert_true(c.item() == 5.0, "Scalar addition failed")
    c.backward()
    assert_true(a.grad().item() == 1.0)
    assert_true(b.grad().item() == 1.0)


fn test_add_scalar_1d() raises:
    print("test_add_scalar_1d")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var c = a + b
    assert_true(c.all_close(Tensor.d1([3.0, 4.0, 5.0])))
    c.sum().backward()
    assert_true(a.grad().item() == 3.0, "a broadcast to 3 elements")
    assert_true(b.grad().all_close(Tensor.d1([1.0, 1.0, 1.0])))


fn test_add_1d_1d() raises:
    print("test_add_1d_1d")
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor.d1([4.0, 5.0, 6.0], requires_grad=True)
    var c = a + b
    assert_true(c.all_close(Tensor.d1([5.0, 7.0, 9.0])))
    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.d1([1.0, 1.0, 1.0])))
    assert_true(b.grad().all_close(Tensor.d1([1.0, 1.0, 1.0])))


fn test_add_2d_scalar() raises:
    print("test_add_2d_scalar")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.scalar(5.0, requires_grad=True)
    var c = a + b
    assert_true(c.all_close(Tensor.d2([[6.0, 7.0], [8.0, 9.0]])))
    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))
    assert_true(b.grad().item() == 4.0, "b broadcast to 4 elements")


fn test_add_2d_1d() raises:
    print("test_add_2d_1d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.d1([10.0, 20.0], requires_grad=True)
    var c = a + b  # b gets broadcasted to both rows
    assert_true(c.all_close(Tensor.d2([[11.0, 22.0], [13.0, 24.0]])))
    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))
    assert_true(b.grad().all_close(Tensor.d1([2.0, 2.0])))


fn test_add_3d_1d() raises:
    print("test_add_3d_1d")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )

    var b = Tensor.d1([10.0, 20.0], requires_grad=True)

    var c = a + b  # shape (2, 2, 2)
    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.full(a.shape(), 1.0)))
    assert_true(b.grad().all_close(Tensor.d1([4.0, 4.0])))


fn test_add_3d_2d() raises:
    print("test_add_3d_2d")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )

    var b = Tensor.d2([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)

    var c = a + b  # b gets broadcast along dim 0
    assert_true(c.shape() == a.shape())
    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.full(a.shape(), 1.0)))
    assert_true(
        b.grad().all_close(Tensor.full(b.shape(), 2.0))
    )  # repeated twice


fn test_add_broadcast_degenerate() raises:
    print("test_add_broadcast_degenerate")
    var a = Tensor.d3(
        [[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True
    )  # Shape (2, 2, 1)

    var b = Tensor.d1([5.0], requires_grad=True)  # Shape (1,)

    var c = a + b
    assert_true(c.shape() == a.shape())
    c.sum().backward()
    assert_true(b.grad().item() == 4.0, "Broadcasted across 4 elements")


fn test_add_mismatch_shapes() raises:
    print("test_add_mismatch_shapes")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var b = Tensor.d2([[1.0], [2.0], [3.0]])  # Shape mismatch

    with assert_raises():
        _ = a + b


fn test_mean_scalar() raises:
    print("test_mean_scalar")
    var a = Tensor.scalar(4.2, requires_grad=True)
    var m = a.mean()
    assert_true(m.item() == 4.2, "Mean of scalar should be the scalar itself")
    m.backward()
    assert_true(a.grad().item() == 1.0, "Grad of scalar mean should be 1.0")


fn test_mean_1d() raises:
    print("test_mean_1d")
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var m = a.mean()
    assert_true(m.item() == 2.0, "Mean of [1, 2, 3] is 2.0")
    m.backward()
    assert_true(
        a.grad().all_close(Tensor.d1([1 / 3, 1 / 3, 1 / 3])),
        "Equal gradient distribution",
    )


fn test_mean_2d_all_axes() raises:
    print("test_mean_2d_all_axes")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var m = a.mean()
    assert_true(m.item() == 2.5, "Mean of all elements is 2.5")
    m.backward()
    assert_true(
        a.grad().all_close(Tensor.d2([[0.25, 0.25], [0.25, 0.25]])),
        "Each grad is 1/4",
    )


fn test_mean_axis0() raises:
    print("test_mean_axis0")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var m = a.mean(axes=[0])
    assert_true(m.all_close(Tensor.d1([2.0, 3.0])), "Mean along axis 0")
    m.backward()
    assert_true(
        a.grad().all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])),
        "Each input contributes 1/2 to mean(axis=0)",
    )


fn test_mean_axis1_keepdims() raises:
    print("test_mean_axis1_keepdims")
    var a = Tensor.d2([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
    var m = a.mean(axes=[1], keepdims=True)
    assert_true(m.all_close(Tensor.d2([[3.0], [7.0]])), "Mean across rows")
    m.backward()
    assert_true(
        a.grad().all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])),
        "Row-wise mean: each contributes 1/2",
    )


fn test_mean_multiple_axes() raises:
    print("test_mean_multiple_axes")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var m = a.mean(axes=[0, 2])
    assert_true(m.shape() == Shape.of(2), "Shape after reducing [0, 2]")
    m.backward()
    assert_true(
        a.grad().sum().item() == 2.0,
        "Total gradient distributed across all elements",
    )


fn test_mean_no_axes() raises:
    print("test_mean_no_axes")
    var a = Tensor.d2([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
    var m = a.mean(axes=[])
    assert_true(m.item() == 5.0, "Mean of all elements")
    m.backward()
    assert_true(a.grad().all_close(Tensor.d2([[0.25, 0.25], [0.25, 0.25]])))


fn test_mean_no_grad() raises:
    print("test_mean_no_grad")
    var a = Tensor.d2([[10.0, 20.0], [30.0, 40.0]], requires_grad=False)
    var m = a.mean()
    assert_true(m.item() == 25.0, "Correct mean without grad")


fn test_scalar_sum_explicit_axes() raises:
    print("test_scalar_sum_explicit_axes")
    var a = Tensor.scalar(10.0)
    var result = a.sum(axes=[])
    assert_true(
        result.item() == 10.0, "Explicit empty axes should work on scalar"
    )


fn test_scalar_sum_keepdims_true() raises:
    print("test_scalar_sum_keepdims_true")
    var a = Tensor.scalar(7.0)
    var result = a.sum(axes=[], keepdims=True)
    assert_true(
        result.shape().rank() == 0,
        "keepdims=True should still return a scalar shape",
    )
    assert_true(result.item() == 7.0, "Sum with keepdims on scalar")


fn test_scalar_sum_custom_grad() raises:
    print("test_scalar_sum_custom_grad")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var result = a.sum()
    result.backward(Tensor.scalar(5.0))  # Upstream gradient is 5.0
    assert_true(
        a.grad().item() == 5.0,
        "Custom upstream grad should be passed correctly",
    )


# This test needs to be enabled once sum is migrated
fn test_reshape_reused_twice_correct_grad() raises:
    print("test_reshape_reused_twice_correct_grad")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var r = x.reshape(Shape.of(2, 2))
    var y = r + r  # <-- r used twice
    y.backward()

    assert_true(
        x.grad().all_close(Tensor.d1([2.0, 2.0, 2.0, 2.0])),
        "∂y/∂x should be 2s — not duplicated",
    )


# This test need to be enabled
fn test_sum_gradient_accumulation() raises:
    print("test_sum_gradient_accumulation")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s1 = a.sum()
    var s2 = a.sum()
    var s = s1 + s2
    s.backward()

    # ∂s/∂a = ∂s1/∂a + ∂s2/∂a = 1 + 1 = 2
    assert_true(
        a.grad().all_close(Tensor.d2([[2, 2], [2, 2]])),
        "Gradient should accumulate from both paths",
    )


fn test_scalar_sum_backward() raises:
    print("test_scalar_sum_backward")
    var a = Tensor.scalar(3.14, requires_grad=True)
    var result = a.sum()  # Should just return a
    result.backward()
    assert_true(result.item() == 3.14, "Forward sum check")
    assert_true(a.grad().item() == 1.0, "Gradient of scalar sum should be 1.0")


fn test_scalar_sum_forward() raises:
    print("test_scalar_sum_forward")
    var a = Tensor.scalar(42.0)
    var result = a.sum()
    assert_true(
        result.item() == 42.0, "Scalar sum should return the same value"
    )


fn test_sum_all_axes_keepdims() raises:
    print("test_sum_all_axes_keepdims")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum(axes=[0, 1], keepdims=True)
    s.backward(Tensor.d2([[100]]))

    assert_true(s.shape() == Shape.of(1, 1), "keepdims should preserve (1,1)")
    assert_true(a.grad().all_close(Tensor.d2([[100, 100], [100, 100]])))


fn test_sum_multi_axes() raises:
    var a = Tensor.d3(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        requires_grad=True,
    )  # shape: (2,2,2)

    var s = a.sum(axes=[0, 2])
    s.backward(Tensor.d1([10, 20]))  # shape: (2,)

    # Reduced to shape: (2,)
    # Incoming grad should be broadcasted back to shape (2,2,2)
    assert_true(
        a.grad().all_close(
            Tensor.d3(
                [
                    [[10, 10], [20, 20]],
                    [[10, 10], [20, 20]],
                ]
            )
        )
    )


fn test_sum_axis1_nokeepdims() raises:
    print("test_sum_axis1_nokeepdims")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum(axes=[1])
    s.backward(Tensor.d1([5, 6]))  # shape: (2,)
    s.print()
    # Broadcast (2,) to (2,2)
    assert_true(a.grad().all_close(Tensor.d2([[5, 5], [6, 6]])))


fn test_sum_axis1_keepdims() raises:
    print("test_sum_axis1_keepdims")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum(axes=[1], keepdims=True)
    s.backward(Tensor.d2([[10], [20]]))

    # ∂s/∂a = [[10, 10], [20, 20]]
    assert_true(
        a.grad().all_close(Tensor.d2([[10, 10], [20, 20]])),
        "Keepdims should preserve dimension during broadcast",
    )


fn test_sum_axis0() raises:
    print("test_sum_axis0")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum(axes=[0])
    s.backward(Tensor.d1([10, 20]))  # incoming grad shape must match output
    assert_true(s.shape() == Shape.of(2), "Sum axis=0 → shape (2,)")

    # ∂s/∂a = [[10, 20], [10, 20]]
    assert_true(
        a.grad().all_close(Tensor.d2([[10, 20], [10, 20]])),
        "Gradient must be broadcast correctly",
    )


fn test_sum_all_elements() raises:
    print("test_sum_all_elements")
    var a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    var s = a.sum()
    s.backward()
    assert_true(s.item() == 10.0, "Sum of all elements should be 10")
    assert_true(a.grad().all_close(Tensor.d2([[1, 1], [1, 1]])), "∂s/∂a = ones")


# Basic reshape gradient: forward shape is changed, but grads match original shape
fn test_reshape_gradient_2d() raises:
    print("test_reshape_gradient_2d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.reshape(Shape.of(4))  # Flatten
    b.backward()
    assert_true(
        a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])),
        "∂b/∂a should be ones reshaped",
    )


fn test_reshape_gradient_flatten() raises:
    print("test_reshape_gradient_flatten")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var y = x.reshape(Shape.of(2, 2))  # Reshape to 2x2
    var z = y * 2.0
    z.backward()
    assert_true(
        x.grad().all_close(Tensor.d1([2.0, 2.0, 2.0, 2.0])),
        "∂z/∂x should be 2",
    )


fn test_multiple_reshapes() raises:
    print("test_multiple_reshapes")
    var t = Tensor.d1([10.0, 20.0, 30.0, 40.0], requires_grad=True)
    var t2 = t.reshape(Shape.of(2, 2))
    var t3 = t2.reshape(Shape.of(4))
    var y = t3 * 3.0
    y.backward()
    assert_true(
        t.grad().all_close(Tensor.d1([3.0, 3.0, 3.0, 3.0])),
        "Chain of reshapes should yield correct grad",
    )


fn test_reshape_noop() raises:
    print("test_reshape_noop")
    var m = Tensor.d2([[5.0, 6.0]], requires_grad=True)
    var reshaped = m.reshape(Shape.of(1, 2))  # No shape change
    reshaped.backward()
    assert_true(
        m.grad().all_close(Tensor.d2([[1.0, 1.0]])),
        "No-op reshape still propagates grad",
    )


fn test_tensor_div_scalar_2d() raises:
    print("test_tensor_div_scalar_2d")
    var a = Tensor.d2([[2.0, 4.0], [6.0, 8.0]], requires_grad=True)
    var b = a / 2.0

    assert_true(
        b.all_close(Tensor.d2([[1.0, 2.0], [3.0, 4.0]])),
        "Forward division failed",
    )

    b.backward()
    assert_true(
        a.grad().all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])),
        "Gradient should be 1/2",
    )


fn test_tensor_div_scalar_nonuniform() raises:
    print("test_tensor_div_scalar_nonuniform")
    var a = Tensor.d1([10.0, 20.0, 30.0], requires_grad=True)
    var out = a / 10.0

    assert_true(
        out.all_close(Tensor.d1([1.0, 2.0, 3.0])), "Forward output incorrect"
    )

    out.backward()
    # Gradient of each is 1/10
    assert_true(
        a.grad().all_close(Tensor.d1([0.1, 0.1, 0.1])), "Gradient of a wrong"
    )


fn test_tensor_div_scalar() raises:
    print("test_tensor_div_scalar")
    var a = Tensor.d1([4.0, 6.0], requires_grad=True)
    var s = a / 2.0

    assert_true(s.all_close(Tensor.d1([2.0, 3.0])), "Forward result of a / 2")

    s.backward()
    assert_true(
        a.grad().all_close(Tensor.d1([0.5, 0.5])),
        "Grad of a: 1/2 for each element",
    )


fn test_tensor_scalar_subtract() raises:
    print("test_tensor_scalar_subtract")
    # test_scalar_sub
    var a = Tensor.scalar(5.0, requires_grad=True)
    var b = a - 3.0
    b.backward()
    assert_true(a.grad().item() == 1.0, "∂(a - 3)/∂a = 1")

    # test_scalar_rsub
    a = Tensor.scalar(5.0, requires_grad=True)
    b = 10.0 - a
    b.backward()
    assert_true(a.grad().item() == -1.0, "∂(10 - a)/∂a = -1")


fn test_tensor_scalar_add_mul_pow() raises:
    print("test_tensor_scalar_add_mul_pow")
    # ─────── Tensor + scalar ───────
    var a = Tensor.scalar(3.0, requires_grad=True)
    var b = a + 2.0
    b.backward()
    # Expect: b = 5.0, ∂c/∂a = 1 → grad[a] = 1
    assert_true(b.item() == 5.0, "3.0 + 2.0 should be 5.0")
    assert_true(a.grad().item() == 1.0, "∂(a + 2.0)/∂a = 1")

    # ─────── scalar + Tensor ───────
    a = Tensor.scalar(3.0, requires_grad=True)
    b = 2.0 + a  # should dispatch __radd__
    b.backward()
    assert_true(b.item() == 5.0, "2.0 + 3.0 should be 5.0")
    assert_true(a.grad().item() == 1.0, "∂(a + 2.0)/∂a = 1")

    # ─────── Tensor * scalar ───────
    var c = Tensor.scalar(4.0, requires_grad=True)
    var d = c * 3.0
    d.backward()
    assert_true(d.item() == 12.0, "4.0 * 3.0")
    assert_true(c.grad().item() == 3.0, "∂(c * 3)/∂c = 3")

    # ─────── scalar * Tensor ───────
    var e = Tensor.scalar(5.0, requires_grad=True)
    var f = 4.0 * e  # should dispatch __rmul__
    f.backward()
    assert_true(f.item() == 20.0, "4.0 * 5.0")
    assert_true(e.grad().item() == 4.0, "∂(4 * e)/∂e = 4")

    # ─────── Tensor ** scalar ───────
    var g = Tensor.scalar(2.0, requires_grad=True)
    var h = g**3.0  # 2 ** 3 = 8
    h.backward()
    assert_true(h.item() == 8.0, "2.0 ** 3.0 = 8.0")
    assert_true(g.grad().item() == 12.0, "∂(g ** 3)/∂g = 3 * g^2 = 3 * 4 = 12")


fn test_slice_grad() raises:
    print("test_slice_grad")

    var a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    var b = a[1:3]  # [2,3]
    var c = b * Tensor.d1([10, 20])
    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.d1([0, 10, 20, 0])))


fn test_nested_operations() raises:
    print("test_nested_operations")
    _ = """var a = Tensor.d1([1, 2], requires_grad=True)
    var b = Tensor.d1([3, 4], requires_grad=True)
    var c = (a * b).sum() + (a + b).prod()
    c.backward()
    # Verify gradients numerically
    assert_true(abs(a.grad()[0] - 11.0) < 1e-6)  # 3 + (3+4)*1
    assert_true(abs(b.grad()[0] - 8.0) < 1e-6)  # 1 + (1+2)*1"""


fn test_large_tensor_backprop() raises:
    print("test_large_tensor_backprop")
    # Test memory efficiency
    var a = Tensor.rand(Shape([100, 128]), requires_grad=True)
    var b = Tensor.rand(Shape([128, 512]), requires_grad=True)
    var c = a.matmul(b)
    s = c.sum()
    s.backward()
    assert_true(a.grad().shape() == a.shape())
    assert_true(b.grad().shape() == b.shape())


fn test_detach() raises:
    print("test_detach")
    _ = """var a = Tensor.d1([1,2], requires_grad=True)
    var b = a.detach() * 2  # Should not propagate grad
    var c = a * b
    c.sum().backward()
    assert_true(a.grad().all_close(Tensor.d1([2,4])))  # Only from c = a*b"""


fn test_empty_tensor() raises:
    print("test_empty_tensor")
    var a = Tensor.d1([], requires_grad=True)
    var s = a.sum()
    s.backward()
    assert_true(s.item() == min_finite[DType.float32]())
    assert_true(a.grad().shape() == Shape())


fn test_flat_view_chain_backprop() raises:
    print("test_flat_view_chain_backprop")
    var a = Tensor.arange(10, requires_grad=True)
    var v1 = a.view([4, 2], offset=2)
    var v2 = v1.view([2, 4])
    var v3 = v2.view([8])
    v3.backward()
    # assert_eq(a.grad[], Tensor.ones_like(a).slice(2, 10).pad_left(2))
    assert_true((a.grad() == Tensor.of(0, 0, 1, 1, 1, 1, 1, 1, 0, 0)))


fn test_reshape_backward() raises:
    print("test_reshape_backward")
    a = Tensor.d2([[1, 2, 3]], requires_grad=True)
    r = a.reshape(3)
    b = r + 100
    c = r + 200
    d = b + c
    d.backward()

    assert_true(
        (a.gradients()[] == Tensor.d2([[2, 2, 2]])),
        "Tensor view reshape grad assertion failed",
    )


fn test_add_backward() raises:
    print("test_add_backward")
    A1 = Tensor.d2([[1, 2, 3]], requires_grad=True)
    AV = A1.into_view()
    AV.backward(3)
    AV.backward()
    AV.backward()
    assert_true(
        (A1.gradients()[] == Tensor.d2([[5, 5, 5]])),
        "Tensor view backward 4 times grad assertion failed",
    )

    a = Tensor.d2([[1, 2, 3]], requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a + b
    d = b + a
    e = c + d
    e.backward(26)
    assert_true(
        (a.gradients()[] == Tensor.d2([[52, 52, 52]])),
        "2D + 1D grad assertion 1 failed",
    )
    assert_true(
        (b.gradients()[] == Tensor.d1([52, 52, 52])),
        "2D + 1D grad assertion 2 failed",
    )
    ev = e.into_view()
    ev.backward()
    assert_true(
        (a.gradients()[] == Tensor.d2([[158, 158, 158]])),
        "2D + 1D grad assertion 3 failed",
    )
    assert_true(
        (b.gradients()[] == Tensor.d1([158, 158, 158])),
        "2D + 1D grad assertion 4 failed",
    )


fn test_reshape_backward_scalar() raises:
    print("test_reshape_backward_scalar")
    a = Tensor.scalar(100, requires_grad=True)
    r = a.reshape()
    v = r.into_view()
    v.backward(42)
    assert_true(
        a.gradients()[].item() == 42,
        "reshape -> view -> scalar grad assertion failed",
    )


fn test_add_tensor_and_view() raises:
    print("test_add_tensor_and_view")
    a = Tensor.full(Shape.of(3, 3), 2)
    av = a.into_view()
    expected = Tensor.full(Shape.of(3, 3), 4)
    assert_true(
        (a + av == expected),
        "add tensor and view assertion 1 failed",
    )

    expected = Tensor.full(Shape.of(3, 3), 84)

    c = Tensor.full(Shape.of(3, 1), 42)
    cv = c.into_view()
    d = Tensor.full(Shape.of(1, 3), 42)
    dv = d.into_view()
    cvdv = cv + dv
    assert_true((cvdv == expected), "add views assertion 1 failed")
    expected = Tensor.full(Shape.of(3, 3), 44)
    assert_true(
        (a + cv == expected),
        "add tensor and view assertion 2 failed",
    )

    a = Tensor.full(Shape.of(2, 1, 3), 2)
    av = a.into_view()
    b = Tensor.full(Shape.of(3, 1), 2)
    bv = b.into_view()
    expected = Tensor.full(Shape.of(2, 3, 3), 4)
    assert_true((av + bv == expected), "add views assertion 2 failed")
    assert_true((bv + av == expected), "add views assertion 3 failed")
    assert_true(
        (b + av == expected),
        "add tensor and view assertion 3 failed",
    )
    assert_true(
        (bv + a == expected),
        "add tensor and view assertion 4 failed",
    )


fn test_add_tensors() raises:
    print("test_add_tensors")
    a = Tensor.full(Shape.of(3, 3), 2)
    b = Tensor.full(Shape.of(3, 3), 42)
    expected = Tensor.full(Shape.of(3, 3), 44)
    assert_true((a + b == expected), "add tensors assertion 1 failed")

    c = Tensor.full(Shape.of(3, 1), 42)
    assert_true((a + c == expected), "add tensors assertion 2 failed")
    d = Tensor.full(Shape.of(1, 3), 42)
    assert_true((a + d == expected), "add tensors assertion 3 failed")

    expected = Tensor.full(Shape.of(3, 3), 84)
    assert_true((c + d == expected), "add tensors assertion 4 failed")

    expected = Tensor.full(Shape.of(3, 3), 4)
    assert_true((a + a == expected), "add tensors assertion 5 failed")

    a = Tensor.full(Shape.of(2, 1, 3), 2)
    b = Tensor.full(Shape.of(3, 1), 2)
    expected = Tensor.full(Shape.of(2, 3, 3), 4)
    assert_true((a + b == expected), "add tensors assertion 5 failed")
    assert_true((b + a == expected), "add tensors assertion 5 failed")


fn test_add_scalar() raises:
    print("test_add_scalar")
    a = Tensor.full(Shape.of(3, 3), 2)
    b = a + 3
    c = 3 + a
    expected = Tensor.full(Shape.of(3, 3), 5)
    assert_true((b == expected), "add scalar assertion failed")
    assert_true((c == expected), "__radd__ scalar assertion failed")


fn test_subtract_scalar() raises:
    print("test_subtract_scalar")
    a = Tensor.full(Shape.of(3, 3), 5)
    b = a - 3
    c = 7 - a
    expected = Tensor.full(Shape.of(3, 3), 2)
    assert_true((b == expected), "subtract scalar assertion failed")
    assert_true((c == expected), "__rsub__ scalar assertion failed")


fn test_powering() raises:
    print("test_powering")
    a = Tensor.full(Shape.of(3, 3), 2)
    b = a**3
    expected = Tensor.full(Shape.of(3, 3), 8)
    assert_true((b == expected), "pow assertion failed")


fn test_invert() raises:
    print("test_invert")
    a = Tensor[DType.bool].full(Shape.of(3, 3), Scalar[DType.bool](True))
    b = ~a
    expected = Tensor[DType.bool].full(
        Shape.of(3, 3), Scalar[DType.bool](False)
    )
    assert_true((b == expected), "invertion assertion failed")
    assert_true((~b == a), "invertion assertion 2 failed")


fn test_negate_absolute() raises:
    print("test_negate_absolute")
    a = Tensor[DType.float32].full(Shape.of(3, 3), 42)
    negated = -a
    expected = Tensor.full(Shape.of(3, 3), -42)
    assert_true((negated == expected), "negation assertion failed")
    assert_true((negated.__abs__() == a), "__abs__ assertion failed")
    assert_true((abs(negated) == a), "abs assertion failed")


fn test_inplace_update() raises:
    print("test_inplace_update")
    a = Tensor.zeros(3, 3)
    b = Tensor.full(Shape.of(3, 3), 42)
    a += b
    assert_true((a == b), "inplace tensor update assertion failed")


fn test_exponentiation() raises:
    print("test_exponentiation")
    a = Tensor.full(Shape.of(3, 3), 2)
    expected = Tensor.full(Shape.of(3, 3), 7.389056).float()
    b = a.exp()
    assert_true(b.all_close(expected), "exponentiation assertion failed")


fn test_grad_update() raises:
    print("test_grad_update")
    alias dtype = DType.float32
    a = Tensor[dtype].rand(3, 4, requires_grad=True)
    v = a.into_view()
    v.init_gradbox()
    grad = Gradbox[dtype].full(Shape.of(3, 4), 42)
    v.update_grad[AddTensor](grad)
    assert_true(
        (v.gradients()[] == grad),
        "update_grad assertion failed",
    )


fn test_sum_all() raises:
    print("test_sum_all")
    a = Tensor.arange(3 * 4 * 5).reshape(3, 4, 5)
    v = a.view(shape=Shape.of(2, 5, 5), offset=5)
    v2 = a.view(shape=[3, 5, 4], strides=[20, 1, 5], offset=0)
    v3 = a.view(shape=[3, 5, 3], strides=[15, 1, 3], offset=15)
    v4 = v3.view(shape=[5, 3], offset=15)
    v5 = v3.view(shape=[3, 5], strides=[1, 3], offset=15)
    s3 = v3.sum_all()
    s4 = v4.sum_all()
    s5 = v5.sum_all()
    assert_true(
        s3 == 1575.0 and s4 == s5 and s5 == 330.0,
        "view sum_all assertion failed",
    )
    assert_true(
        v.is_contiguous() and v4.is_contiguous(), "contiguity assertion failed"
    )
    assert_true(
        v.sum_all() == 1475,
        "non-owning contiguous tensor sum_all assertion failed",
    )
    assert_true(
        a.sum_all()[0] == 1770, "owning tensor sum_all assertion failed"
    )
    assert_true(
        v2.sum_all()[0] == 1770,
        "owning tensor non-contiguous sum_all assertion failed",
    )


fn test_view_of_view() raises:
    print("test_view_of_view")
    a = Tensor.scalar(10)
    v1 = a.into_view()
    v2 = v1.view(shape=Shape(), strides=Strides(), offset=0)
    v3 = v2.view(shape=Shape(), strides=Strides(), offset=0)
    v4 = v3.view(shape=Shape(), strides=Strides(), offset=0)
    assert_true(v2.item() == 10, "view's view(v2) - item() assertion failed")
    assert_true(v3.item() == 10, "view's view(v3) - item() assertion failed")
    assert_true(v4.item() == 10, "view's view(v4) - item() assertion failed")


fn test_scalar_indexing() raises:
    print("test_scalar_indexing")
    a = Tensor.scalar(10)
    v = a.into_view()
    shape = a.shape()
    v1 = a.view(shape)
    idx = List[Int]()
    assert_true(a.__getitem__(idx) == 10, "scalar indexing get failed")
    assert_true(a[[]] == 10, "scalar indexing get list literal failed")
    a[[]] = 100
    assert_true(
        a[[]] == 100, "scalar indexing get after set list literal failed"
    )
    assert_true(a.item() == 100, "scalar indexing item call failed")
    assert_true(v.item() == 100, "scalar indexing on view - item call failed")
    assert_true(
        v1.item() == 100, "scalar indexing on view 1 - item call failed"
    )


fn test_grads_on_tensor_init() raises:
    print("test_grads_on_tensor_init")
    a = Tensor(6, 3, 4, requires_grad=True)
    b = Tensor(6, 3, 4)
    assert_true(
        a.has_grad() and not b.has_grad(),
        "Initialization grad assertions failed",
    )
    b.fill(42)
    a.seed_grad(b)
    grad = Tensor.full(a.shape(), 42)

    result = a.grad() == grad
    result2 = result

    assert_true(result2, "grad and expected does not match")


fn test_reshape_exp() raises:
    print("test_reshape_exp")
    tensor = Tensor.scalar(42, requires_grad=True)
    result = tensor * 3
    result.backward()
    assert_true(tensor.grad().item() == 3.0)
    tensor2 = tensor.reshape(1)
    result = tensor2 * 42

    result.backward()
    tensor3 = tensor2.reshape(1, 1, 1, 1, 1)
    result = tensor3 * 12
    result.backward()


fn test_validate_matmul_last_2_dims() raises:
    print("test_validate_matmul_last_2_dims")
    a = Tensor.arange(2 * 3 * 5 * 4, requires_grad=True)
    a_reshaped = a.reshape(2, 3, 5, -1)
    b = Tensor.arange(4 * 5, requires_grad=True)
    b_reshaped = b.reshape(4, 5)
    result = a_reshaped.matmul(b_reshaped)
    result.backward()
    expected = Tensor.d2(
        [
            [1740, 1740, 1740, 1740, 1740],
            [1770, 1770, 1770, 1770, 1770],
            [1800, 1800, 1800, 1800, 1800],
            [1830, 1830, 1830, 1830, 1830],
        ]
    )
    assert_true(
        (b.grad().reshape(Shape(4, 5)) == expected),
        "batched tensor multiplication assertion failed",
    )


fn test_tensor_dot() raises:
    print("test_tensor_dot")
    a = Tensor.scalar(5, requires_grad=True)
    b = Tensor.scalar(15, requires_grad=True)
    c = a.matmul(b)
    c.backward()
    assert_true(a.grad().item() == 15)
    assert_true(b.grad().item() == 5)

    d = a.into_view()
    e = d.matmul(b)
    e.backward()
    assert_true(a.grad().item() == 30)
    assert_true(b.grad().item() == 10)
    assert_true(d.grad().item() == 0)

    a = Tensor.arange(10, requires_grad=True)
    b = a[5::2]
    c = Tensor.d1([3, 4, 5])
    d = b.matmul(c)
    d.backward()
    assert_true(a.grad().all_close(Tensor.d1([0, 0, 0, 0, 0, 3, 0, 4, 0, 5])))


fn test_dot_product() raises:
    print("test_dot_product")
    # 1D @ 1D -> scalar (dot product)
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = Tensor.d1([4, 5, 6], requires_grad=True)
    c = a.matmul(b)

    # Verify result: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_true(c.all_close(Tensor.scalar(32)))

    c.backward()
    # da = b, db = a
    assert_true(a.grad().all_close(Tensor.d1([4, 5, 6])))
    assert_true(b.grad().all_close(Tensor.d1([1, 2, 3])))


fn test_vector_matrix_matmul() raises:
    print("test_vector_matrix_matmul")
    # 1D @ 2D -> 1D
    a = Tensor.arange(3, requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True)
    c = a.matmul(b)
    # Verify result: [0*1+1*4+2*7, 0*2+1*5+2*8, 0*3+1*6+2*9] = [18, 21, 24]
    assert_true(c.all_close(Tensor.d1([18, 21, 24])))

    c.backward()
    # db = outer(a, ones) = [[0,0,0], [1,1,1], [2,2,2]]
    assert_true(a.grad().all_close(Tensor.d1([6, 15, 24])))
    assert_true(
        b.grad().all_close(Tensor.d2([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))
    )


fn test_matrix_vector_matmul() raises:
    print("test_matrix_vector_matmul")
    # 2D @ 1D -> 1D
    a = Tensor.d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True)
    b = Tensor.arange(3, requires_grad=True)
    c = a.matmul(b)

    # Verify result: [1*0+2*1+3*2, 4*0+5*1+6*2, 7*0+8*1+9*2] = [8, 17, 26]
    assert_true(c.all_close(Tensor.d1([8, 17, 26])))

    c.backward()
    # da = outer(ones, b) = [[0,1,2], [0,1,2], [0,1,2]]

    assert_true(
        a.grad().all_close(Tensor.d2([[0, 1, 2], [0, 1, 2], [0, 1, 2]]))
    )
    assert_true(b.grad().all_close(Tensor.d1([12, 15, 18])))

    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor.d1([7, 8, 9], requires_grad=True)
    c = a.matmul(b)  # (2,3) @ (3,) -> (2,)
    c.backward()
    # forward: [50, 122]
    assert_true(c.all_close(Tensor.d1([50, 122])))
    assert_true(a.grad().all_close(Tensor.d2([[7, 8, 9], [7, 8, 9]])))
    assert_true(b.grad().all_close(Tensor.d1([5, 7, 9])))


fn test_matrix_matrix_matmul() raises:
    print("test_matrix_matrix_matmul")
    # 2D @ 2D -> 2D
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor.d2([[5, 6], [7, 8]], requires_grad=True)
    c = a.matmul(b)

    # Verify result: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assert_true(c.all_close(Tensor.d2([[19, 22], [43, 50]])))

    c.backward()
    # da = b^T broadcasted: [[5+6, 7+8], [5+6, 7+8]] = [[11, 15], [11, 15]]

    assert_true(a.grad().all_close(Tensor.d2([[11, 15], [11, 15]])))
    assert_true(b.grad().all_close(Tensor.d2([[4, 4], [6, 6]])))
    a = Tensor.d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )  # (2,2,2)
    b = Tensor.d3(
        [[[9, 10], [11, 12]], [[13, 14], [15, 16]]], requires_grad=True
    )  # (2,2,2)
    c = a.matmul(b)  # (2,2,2)
    assert_true(
        c.all_close(Tensor.d3([[[31, 34], [71, 78]], [[155, 166], [211, 226]]]))
    )


fn test_batched_matrix_matmul() raises:
    print("test_batched_matrix_matmul")
    # 3D @ 3D -> 3D (batched matrix multiplication)
    a = Tensor.d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    b = Tensor.d3([[[2, 0], [1, 2]], [[1, 0], [0, 1]]], requires_grad=True)
    c = a.matmul(b)

    # Batch 0: [[1,2],[3,4]] @ [[2,0],[1,2]] = [[4,4],[10,8]]
    # Batch 1: [[5,6],[7,8]] @ [[1,0],[0,1]] = [[5,6],[7,8]]
    assert_true(c.all_close(Tensor.d3([[[4, 4], [10, 8]], [[5, 6], [7, 8]]])))

    c.backward()
    # For batched matmul, gradients are computed per batch
    # da: [[[2+0, 1+2], [2+0, 1+2]], [[1+0, 0+1], [1+0, 0+1]]] = [[[2,3],[2,3]], [[1,1],[1,1]]]

    assert_true(
        a.grad().all_close(Tensor.d3([[[2, 3], [2, 3]], [[1, 1], [1, 1]]]))
    )
    assert_true(
        b.grad().all_close(Tensor.d3([[[4, 4], [6, 6]], [[12, 12], [14, 14]]]))
    )


fn test_broadcasted_matrix_matmul() raises:
    print("test_broadcasted_matrix_matmul")
    # 3D @ 2D -> 3D (broadcasted matmul)
    a = Tensor.d3([[[1, 2]], [[3, 4]]], requires_grad=True)  # shape: (2, 1, 2)
    b = Tensor.d2([[5, 6], [7, 8]], requires_grad=True)  # shape: (2, 2)
    c = a.matmul(b)
    # Batch 0: [[1,2]] @ [[5,6],[7,8]] = [[19,22]]
    # Batch 1: [[3,4]] @ [[5,6],[7,8]] = [[43,50]]
    assert_true(c.all_close(Tensor.d3([[[19, 22]], [[43, 50]]])))

    c.backward()
    # da gradients are computed per batch with broadcasting
    # db gradient accumulates across all batches

    assert_true(
        a.grad().all_close(Tensor.d3([[[11, 15]], [[11, 15]]]))
    )  # [[5+6,7+8]] per batch
    assert_true(b.grad().all_close(Tensor.d2([[4, 4], [6, 6]])))


fn test_high_dim_batched_matmul() raises:
    print("test_high_dim_batched_matmul")
    # 4D @ 4D -> 4D (higher dimensional batched matmul)
    a = Tensor.d4(
        [[[[1, 2], [3, 4]]]], requires_grad=True
    )  # shape: (1, 1, 2, 2)
    b = Tensor.d4(
        [[[[5, 6], [7, 8]]]], requires_grad=True
    )  # shape: (1, 1, 2, 2)
    c = a.matmul(b)
    # Should be same as 2x2 @ 2x2: [[19,22],[43,50]]
    assert_true(c.all_close(Tensor.d4([[[[19, 22], [43, 50]]]])))

    c.backward()
    # Gradients should match the 2D case but with additional batch dimensions

    assert_true(a.grad().all_close(Tensor.d4([[[[11, 15], [11, 15]]]])))
    assert_true(b.grad().all_close(Tensor.d4([[[[4, 4], [6, 6]]]])))


fn test_matmul_no_grad() raises:
    print("test_matmul_no_grad")
    # Test matmul without requiring gradients
    a = Tensor.d1([1, 2, 3])
    b = Tensor.d1([4, 5, 6])
    c = a.matmul(b)

    assert_true(c.all_close(Tensor.scalar(32)))
    # No gradients should be computed
    assert_false(a.requires_grad)
    assert_false(b.requires_grad)


fn test_matmul_mixed_grad() raises:
    print("test_matmul_mixed_grad")
    # Test matmul with only one tensor requiring grad
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = Tensor.d1([4, 5, 6])  # no grad
    c = a.matmul(b)

    assert_true(c.all_close(Tensor.scalar(32)))
    c.backward()

    # Only a should have gradients
    assert_true(a.grad().all_close(Tensor.d1([4, 5, 6])))
    # b should not have gradbox since it doesn't require grad
    assert_false(b.requires_grad)


fn test_matmul_shape_validation() raises:
    print("test_matmul_shape_validation")
    # These should work (valid shapes)
    a1 = Tensor.d1([1, 2, 3])
    b1 = Tensor.d2([[4], [5], [6]])
    c1 = a1.matmul(b1)  # 1D @ 2D -> 1D

    a2 = Tensor.d2([[1, 2]])
    b2 = Tensor.d1([3, 4])
    c2 = a2.matmul(b2)  # 2D @ 1D -> 1D

    assert_true(c1.all_close(Tensor.scalar(32).reshape(1)))
    assert_true(c2.all_close(Tensor.scalar(11).reshape(1)))


fn test_batched_matrix_vector_matmul() raises:
    print("test_batched_matrix_vector_matmul")
    a = Tensor.d3(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], requires_grad=True
    )  # (2,2,3)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2,3)
    b_transposed = b.transpose()
    c = a.matmul(b_transposed)  # (2,2,2)
    c.backward(Tensor.ones_like(c))

    assert_true(
        c.all_close(Tensor.d3([[[14, 32], [32, 77]], [[50, 122], [68, 167]]]))
    )

    assert_true(
        a.grad().all_close(
            Tensor.d3([[[5, 7, 9], [5, 7, 9]], [[5, 7, 9], [5, 7, 9]]])
        )
    )
    assert_true(b.grad().all_close(Tensor.d2([[22, 26, 30], [22, 26, 30]])))


fn test_matrix_vector_mm_simple() raises:
    print("test_matrix_vector_mm_simple")
    var A = Tensor.d2([[1, 2], [3, 4]])  # (2,2)
    var b = Tensor.d1([5, 6])  # (2,)
    var y = A.matmul(b)  # (2,)
    var expected = Tensor.d1([1 * 5 + 2 * 6, 3 * 5 + 4 * 6])  # 17  # 39
    assert_true(y.shape() == [2])
    assert_true(y.all_close(expected))


fn test_matrix_vector_mm_backward_A() raises:
    print("test_matrix_vector_mm_backward_A")
    var A = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)  # (2,2)
    var b = Tensor.d1([5, 6])  # (2,)
    var y = A.matmul(b)  # (2,)
    var s = y.sum()
    s.backward()
    # dA = outer(ones(m), b) => each row == b
    var expected_grad = Tensor.d2([[5, 6], [5, 6]])
    assert_true(A.grad().all_close(expected_grad))


fn test_matrix_vector_mm_backward_b() raises:
    print("test_matrix_vector_mm_backward_b")
    var A = Tensor.d2([[1, 2], [3, 4]])  # (2,2)
    var b = Tensor.d1([5, 6], requires_grad=True)  # (2,)
    var y = A.matmul(b)  # (2,)
    var s = y.sum()
    s.backward()
    # db = A^T @ ones(m) -> column sums of A
    var expected_grad = Tensor.d1([1 + 3, 2 + 4])  # 4  # 6
    assert_true(b.grad().all_close(expected_grad))


fn test_matrix_vector_mm_batched_forward() raises:
    print("test_matrix_vector_mm_batched_forward")
    var A = Tensor.d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  # batch 0  # batch 1
    )  # (2,2,2)
    var b = Tensor.d1([1, 1])  # (2,) broadcast
    var y = A.matmul(b)  # (2,2)
    # batch0: [1,2;3,4]*[1,1]=[3,7]
    # batch1: [5,6;7,8]*[1,1]=[11,15]
    var expected = Tensor.d2([[3, 7], [11, 15]])
    assert_true(y.shape() == [2, 2])
    assert_true(y.all_close(expected))


fn test_matrix_vector_mm_backward_b_batched() raises:
    print("test_matrix_vector_mm_backward_b_batched")
    var A = Tensor.d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  # batch 0  # batch 1
    )  # (2,2,2)
    var b = Tensor.d1([3, 4], requires_grad=True)  # (2,) broadcast
    var y = A.matmul(b)  # (2,2)
    var s = y.sum()
    s.backward()
    # db = sum_over_batch_rows(A)
    # batch0 col sums = [1+3, 2+4] = [4, 6]
    # batch1 col sums = [5+7, 6+8] = [12, 14]
    # total = [16, 20]
    var expected_grad = Tensor.d1([16, 20])
    assert_true(b.grad().all_close(expected_grad))


fn test_matrix_vector_mm_backward_A_batched() raises:
    print("test_matrix_vector_mm_backward_A_batched")
    var A = Tensor.d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],  # batch 0  # batch 1
        requires_grad=True,
    )  # (2,2,2)
    var b = Tensor.d1([2, 3])  # (2,) broadcast
    var y = A.matmul(b)  # (2,2)
    var s = y.sum()
    s.backward()
    # dA per batch: each row == b
    var expected_grad = Tensor.d3([[[2, 3], [2, 3]], [[2, 3], [2, 3]]])
    assert_true(A.grad().all_close(expected_grad))


fn test_matrix_vector_mm_deeper_batch_forward() raises:
    print("test_matrix_vector_mm_deeper_batch_forward")
    var A = Tensor.d3(
        [
            [[1, 2], [3, 4], [5, 6]],  # batch 0: (3,2)
            [[7, 8], [9, 10], [11, 12]],  # batch 1: (3,2)
        ]
    )  # (2,3,2) = (batch, m, n)
    var b = Tensor.d1([1, 2])  # (2,)
    var y = A.matmul(b)  # (2,3)
    # batch0 rows·b = [5, 11, 17]
    # batch1 rows·b = [23, 29, 35]
    var expected = Tensor.d2([[5, 11, 17], [23, 29, 35]])
    assert_true(y.shape() == [2, 3])
    assert_true(y.all_close(expected))


fn test_matrix_vector_mm_backward_b_deeper_batch() raises:
    print("test_matrix_vector_mm_backward_b_deeper_batch")
    var A = Tensor.d3(
        [
            [[1, 2], [3, 4], [5, 6]],  # batch 0
            [[7, 8], [9, 10], [11, 12]],  # batch 1
        ]
    )  # (2,3,2)
    var b = Tensor.d1([1, 1], requires_grad=True)  # (2,)
    var y = A.matmul(b)  # (2,3)
    var s = y.sum()
    s.backward()
    # db = sum over all batch rows:
    # col0 sum = 1+3+5+7+9+11 = 36
    # col1 sum = 2+4+6+8+10+12 = 42
    var expected_grad = Tensor.d1([36, 42])
    assert_true(b.grad().all_close(expected_grad))


fn test_batched_matmul_vector_rhs_broadcast() raises:
    print("test_batched_matmul_vector_rhs_broadcast")
    # A: (2,3,4)  v: (4,)  -> out (2,3)
    A = Tensor.arange(2 * 3 * 4, requires_grad=True)
    r = A.reshape(2, 3, 4)
    v = Tensor.ones(4, requires_grad=True)
    out = r.matmul(v)  # row sums over last axis
    # forward check: sums along last axis
    s00 = 0 + 1 + 2 + 3
    s01 = 4 + 5 + 6 + 7
    s02 = 8 + 9 + 10 + 11
    s10 = 12 + 13 + 14 + 15
    s11 = 16 + 17 + 18 + 19
    s12 = 20 + 21 + 22 + 23
    assert_true(out.all_close(Tensor.d2([[s00, s01, s02], [s10, s11, s12]])))
    out.backward()
    # dv = sum over all A elements per position count (each v_k used 6 times)
    assert_true(v.grad().all_close(Tensor.d1([60, 66, 72, 78])))


fn test_vector_matrix_mm_simple() raises:
    print("test_vector_matrix_mm_simple")
    var a = Tensor.d1([1, 2, 3], requires_grad=True)  # shape (3,)
    # B must have shape (3, 2) so that (3,) @ (3,2) -> (2,)
    var b = Tensor.d2(
        [[1, 4], [2, 5], [3, 6]], requires_grad=True
    )  # shape (3,2)
    var y = a.matmul(b)  # expected shape (2,)

    # expected: [1*1 + 2*2 + 3*3, 1*4 + 2*5 + 3*6] = [14, 32]
    var expected = Tensor.d1([14, 32])
    assert_true(y.shape() == [2])
    assert_true(y.all_close(expected))


fn test_vector_matrix_mm_backward_vector() raises:
    print("test_vector_matrix_mm_backward_vector")
    var a = Tensor.d1([1, 2, 3], requires_grad=True)  # shape (3,)
    # var a = Tensor.d2([[1, 2, 3]], requires_grad=True)           # shape (3,)
    var b = Tensor.d2(
        [[1, 2], [3, 4], [5, 6]], requires_grad=False
    )  # shape (3,2)
    var y = a.matmul(b)  # shape (2,)
    var s = y.sum()  # scalar loss
    s.backward()
    # dy/da = sum over columns of b -> sums of each row
    # row sums: [1+2, 3+4, 5+6] = [3, 7, 11]
    var expected_grad = Tensor.d1([3, 7, 11])
    assert_true(a.grad().all_close(expected_grad))


fn test_vector_matrix_mm_backward_matrix() raises:
    print("test_vector_matrix_mm_backward_matrix")
    var a = Tensor.d1([2, 3], requires_grad=False)  # shape (2,)
    var b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # shape (2,3)
    var y = a.matmul(b)  # shape (3,)
    var s = y.sum()
    s.backward()
    # dy/db = outer(a, ones(cols)) => each column has the vector a
    # expected shape matches b: [[2,2,2], [3,3,3]]
    var expected_grad = Tensor.d2([[2, 2, 2], [3, 3, 3]])
    assert_true(b.grad().all_close(expected_grad))


fn test_vector_matrix_mm_batched_matrix() raises:
    print("test_vector_matrix_mm_batched_matrix")
    var a = Tensor.d1([1, 2], requires_grad=True)  # shape (2,)
    # b: batch of 2 matrices, each (2,2) ; shape (2, 2, 2)
    var b = Tensor.d3(
        [
            [[1, 2], [3, 4]],  # batch 0, shape (2,2)
            [[5, 6], [7, 8]],  # batch 1, shape (2,2)
        ],
        requires_grad=True,
    )
    var y = a.matmul(b)
    y.backward()
    # var y = a.matmul(b)  # expected shape (2,2): (batch, m)
    # For batch 0: [1,2] @ [[1,2],[3,4]] = [7,10]
    # For batch 1: [1,2] @ [[5,6],[7,8]] = [19,22]
    var expected = Tensor.d2([[7, 10], [19, 22]])
    assert_true(y.shape() == [2, 2])  # (batch, m)
    assert_true(y.all_close(expected))


fn test_vector_matrix_mm_backward_batched_matrix_vector_grad() raises:
    print("test_vector_matrix_mm_backward_batched_matrix_vector_grad")
    var a = Tensor.d1([1, 2], requires_grad=True)  # shape (2,)
    var b = Tensor.d3(
        [[[1, 0], [0, 1]], [[2, 3], [4, 5]]],  # batch 0  # batch 1
        requires_grad=False,
    )  # shape (2,2,2)
    var y = a.matmul(b)  # shape (2,2)
    var s = y.sum()
    s.backward()
    # grad wrt vector = sum across batch of column-sums
    # batch0 column sums = [1,1]; batch1 column sums = [6,8]; total = [7,9]
    var expected_grad = Tensor.d1([6, 10])
    assert_true(a.grad().all_close(expected_grad))


fn test_vector_matrix_mm_backward_batched_matrix_matrix_grad() raises:
    print("test_vector_matrix_mm_backward_batched_matrix_matrix_grad")
    var a = Tensor.d1([3, 4], requires_grad=False)  # shape (2,)
    var b = Tensor.d3(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True
    )  # shape (2,2,2)
    var y = a.matmul(b)  # shape (2,2)
    # var y1 = a1.matmul_nd(b1)  # shape (2,2)
    var s = y.sum()
    s.backward()
    # grad wrt b = outer(a, ones(cols)) broadcast across each batch slice
    var expected_grad = Tensor.d3([[[3, 3], [4, 4]], [[3, 3], [4, 4]]])
    assert_true(b.grad().all_close(expected_grad))


fn test_max_min_mixed() raises:
    print("test_max_min_mixed")

    # Test 1: Basic max reduction along axis 1
    var a = Tensor.d2(
        [[42.0, 0.0, -5.0], [0.0, 35.0, 0.0], [51.0, 0.0, 51.0]],
        requires_grad=True,
    )
    var max_result = a.max(IntList(1))
    var expected = Tensor.d1([42.0, 35.0, 51.0])
    assert_true(max_result.all_close(expected))

    max_result.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])
        )
    )

    # Reset gradients for next test
    a.zero_grad()

    # Test 2: Basic min reduction along axis 1
    var min_result = a.min(IntList(1))
    assert_true(min_result.all_close(Tensor.d1([-5.0, 0.0, 0.0])))
    min_result.backward()

    assert_true(
        a.grad().all_close(
            Tensor.d2([[0.0, 0.0, 1.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
        )
    )

    # Test 3: Max reduction along axis 0
    a.zero_grad()
    var max_axis0 = a.max(IntList(0))
    assert_true(max_axis0.all_close(Tensor.d1([51.0, 35.0, 51.0])))
    max_axis0.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        )
    )

    # Test 4: Min reduction along axis 0
    a.zero_grad()
    var min_axis0 = a.min(IntList(0))
    assert_true(min_axis0.all_close(Tensor.d1([0.0, 0.0, -5.0])))
    min_axis0.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2([[0.0, 0.5, 1.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
        )
    )

    # Test 5: Global max (no axis)
    a.zero_grad()
    var global_max = a.max()
    assert_true(global_max.all_close(Tensor.scalar(51.0)))
    global_max.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.5]])
        )
    )

    # Test 6: Global min (no axis)
    a.zero_grad()
    var global_min = a.min()
    assert_true(global_min.all_close(Tensor.scalar(-5.0)))
    global_min.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        )
    )

    # Test 7: Multiple axes reduction
    var b = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )

    var max_axes_01 = b.max(IntList([0, 1]))
    assert_true(max_axes_01.all_close(Tensor.d1([7.0, 8.0])))
    max_axes_01.backward()
    assert_true(
        b.grad().all_close(
            Tensor.d3([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]]])
        )
    )

    # Test 8: Edge case - all same values
    var c = Tensor.d2([[5.0, 5.0], [5.0, 5.0]], requires_grad=True)

    var max_same = c.max(IntList(1))
    assert_true(max_same.all_close(Tensor.d1([5.0, 5.0])))
    max_same.backward()
    assert_true(c.grad().all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])))

    # Test 9: Edge case - negative infinity
    var d = Tensor.d2([[-3.4028235e38, 0.0], [1.0, 2.0]], requires_grad=True)

    var max_with_inf = d.max(IntList(1))
    assert_true(max_with_inf.all_close(Tensor.d1([0.0, 2.0])))
    max_with_inf.backward()
    assert_true(d.grad().all_close(Tensor.d2([[0.0, 1.0], [0.0, 1.0]])))

    # Test 10: Keep dimensions
    var e = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    var max_keepdim = e.max(IntList(1), keepdims=True)
    assert_true(max_keepdim.all_close(Tensor.d2([[2.0], [4.0]])))
    max_keepdim.backward()
    assert_true(e.grad().all_close(Tensor.d2([[0.0, 1.0], [0.0, 1.0]])))


fn test_max_min() raises:
    print("test_max_min")
    a = Tensor.d2(
        [[42.0, 0.0, -5.0], [0.0, 35.0, 0.0], [51.0, 0.0, 51.0]],
        requires_grad=True,
    )

    max_result = a.max(IntList(1))
    expected = Tensor.d1([42.0, 35.0, 51.0])
    assert_true(max_result.all_close(expected))

    max_result.backward()
    assert_true(
        a.grad().all_close(
            Tensor.d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])
        )
    )
    min_result = a.min([1])
    assert_true(min_result.all_close(Tensor.d1([-5.0, 0.0, 0.0])))
    min_result.backward()

    assert_true(
        a.grad().all_close(
            Tensor.d2([[1.0, 0.0, 1.0], [0.5, 1.0, 0.5], [0.5, 1.0, 0.5]])
        )
    )


fn test_mask() raises:
    print("test_mask")
    a = Tensor.arange(2 * 3).reshape(2, 3)
    mask = a != 2
    converted = mask.float64()

    assert_true(
        (converted == Tensor.d2([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])),
        "Mask assertion failed",
    )


fn test_randint() raises:
    print("test_randint")
    low = 10
    high = 30
    a = Tensor.randint([3, 4], low, high)
    count_low = a.count(low)
    count_high = a.count(high)
    assert_true(
        count_low >= 0 and count_high == 0,
        "randint low and high count assertion failed",
    )


# ===================== SINGLE-AXIS SLICES =====================


fn test_slice_single_axis() raises:
    print("test_slice_single_axis")
    x = Tensor.arange(0, 12).reshape([3, 4])

    y = x.slice(1, 3)  # slice along axis 0 (rows 1..2)
    z = x.slice(0, 4, 2, 1)  # slice along axis 1 (cols 0,2)

    assert_true(
        (y == Tensor.d2([[4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]).float())
    )
    assert_true((z == Tensor.d2([[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]]).float()))


fn test_slice_single_axis_positive() raises:
    print("test_slice_single_axis_positive")
    var x = Tensor.arange(0, 10).reshape([10])
    var y = x.slice(axes=[0], starts=[2], ends=[7])
    assert_true((y == Tensor.arange(2, 7)))


fn test_slice_single_axis_negative_indices() raises:
    print("test_slice_single_axis_negative_indices")
    var x = Tensor.arange(0, 10).reshape([10])
    var y = x.slice(axes=[0], starts=[-7], ends=[-2])
    assert_true((y == Tensor.arange(3, 8)))


fn test_slice_single_axis_step_greater_than_1() raises:
    print("test_slice_single_axis_step_greater_than_1")
    var x = Tensor.arange(0, 10).reshape([10])
    var y = x.slice(axes=[0], starts=[1], ends=[9], steps=[2])
    assert_true((y == Tensor([1, 3, 5, 7])))


fn test_slice_single_axis_step_negative() raises:
    print("test_slice_single_axis_step_negative")
    var x = Tensor.arange(0, 10).reshape([10])
    var y = x.slice(axes=[0], starts=[8], ends=[2], steps=[-2])
    assert_true((y == Tensor([8, 6, 4])))


fn test_slice_single_axis_full_axis() raises:
    print("test_slice_single_axis_full_axis")
    var x = Tensor.arange(0, 5).reshape([5])
    var y = x.slice(axes=[0], starts=[0], ends=[5])
    assert_true((y == x))


fn test_slice_single_axis_single_element() raises:
    print("test_slice_single_axis_single_element")
    var x = Tensor.arange(0, 5).reshape([5])
    var y = x.slice(axes=[0], starts=[2], ends=[3])
    assert_true((y == Tensor([2])))


# ===================== MULTI-AXIS SLICES =====================


fn test_slice_multi_axis_basic() raises:
    print("test_slice_multi_axis_basic")
    var x = Tensor.arange(0, 24).reshape([4, 6])
    var y = x.slice(axes=[0, 1], starts=[1, 2], ends=[3, 5])
    assert_true((y == Tensor.d2([[8, 9, 10], [14, 15, 16]])))


fn test_slice_multi_axis_negative_indices() raises:
    print("test_slice_multi_axis_negative_indices")
    var x = Tensor.arange(0, 24).reshape([4, 6])
    var y = x.slice(axes=[0, 1], starts=[-3, -4], ends=[-1, -1])
    assert_true((y == Tensor.d2([[8, 9, 10], [14, 15, 16]])))


fn test_slice_multi_axis_step() raises:
    print("test_slice_multi_axis_step")
    var x = Tensor.arange(0, 24).reshape([4, 6])
    var y = x.slice(axes=[0, 1], starts=[0, 0], ends=[4, 6], steps=[2, 3])
    assert_true((y == Tensor.d2([[0, 3], [12, 15]])))


fn test_slice_multi_axis_mixed() raises:
    print("test_slice_multi_axis_mixed")
    var x = Tensor.arange(0, 24).reshape([4, 6])
    var y = x.slice(axes=[0, 1], starts=[3, 5], ends=[0, 0], steps=[-1, -2])
    var expected = Tensor.d2([[23, 21, 19], [17, 15, 13], [11, 9, 7]])
    assert_true((y == expected))

    assert_true(
        (y.transpose() == Tensor.d2([[23, 17, 11], [21, 15, 9], [19, 13, 7]]))
    )


fn test_repeat_1d_axis0() raises:
    print("test_repeat_1d_axis0")
    var a = Tensor.d1([10, 20, 30])
    var r = a.repeat([2])
    assert_true((r == Tensor.d1([10, 20, 30, 10, 20, 30])))


fn test_repeat_backward_simple() raises:
    print("test_repeat_backward_simple")
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True).float()
    var r = a.repeat([2])  # [1,1,2,2,3,3]
    print("The repeat: ")
    r.print()
    var loss = r.sum()
    loss.backward()
    # each element of a is repeated twice → grad = 2 for each
    assert_true((a.grad().all_close(Tensor.d1([2.0, 2.0, 2.0]).float())))


fn test_repeat_backward_with_view_slice() raises:
    print("test_repeat_backward_with_view_slice")
    var a = Tensor.d1([10.0, 20.0, 30.0, 40.0], requires_grad=True).float()
    var v = a[s(1, 3)]  # view: [20, 30]
    var r = v.repeat([2])  # [20,20,30,30]
    var loss = r.sum()
    loss.backward()
    # a.grad should reflect contributions only to indices 1 and 2
    assert_true((a.grad().all_close(Tensor.d1([0.0, 2.0, 2.0, 0.0]).float())))


fn test_repeat_backward_with_strided_view() raises:
    print("test_repeat_backward_with_strided_view")
    var a = Tensor.d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True
    ).float()
    var v = a[s(0, 6, 2)]  # [1,3,5] → stride view
    var r = v.repeat([3])  # [1,1,1,3,3,3,5,5,5]
    var loss = r.sum()
    loss.backward()
    # v has indices [0,2,4], each repeated 3 times → grads: [3,0,3,0,3,0]
    assert_true(
        (a.grad().all_close(Tensor.d1([3.0, 0.0, 3.0, 0.0, 3.0, 0.0]).float()))
    )


fn test_repeat_backward_multi_axis() raises:
    print("test_repeat_backward_multi_axis")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True).float()
    var r = a.repeat([2, 3])  # shape (4, 6)
    var loss = r.sum()
    loss.backward()
    # each element repeated 2 * 3 = 6 times
    assert_true(
        (a.grad().all_close(Tensor.d2([[6.0, 6.0], [6.0, 6.0]]).float()))
    )


fn test_repeat_backward_chain_view_repeat() raises:
    print("test_repeat_backward_chain_view_repeat")
    var a = Tensor.d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    ).float()
    var v = a[s(0, 2), s(1, 3)]  # [[2,3],[5,6]]
    var r = v.repeat([2, 2])  # shape (4,4)
    var loss = r.sum()
    loss.backward()
    # grads land in positions (0,1),(0,2),(1,1),(1,2), each repeated 4 times
    assert_true(
        (
            a.grad().all_close(
                Tensor.d2([[0.0, 4.0, 4.0], [0.0, 4.0, 4.0]]).float()
            )
        )
    )


fn test_tile_1d_basic() raises:
    print("test_tile_1d_basic")
    var a = Tensor.d1([1.0, 2.0, 3.0]).float()
    var t = a.tile([2])
    assert_true((t == Tensor.d1([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).float()))


fn test_tile_1d_multi() raises:
    print("test_tile_1d_multi")
    var a = Tensor.d1([4.0, 5.0]).float()
    var t = a.tile([3])
    assert_true((t == Tensor.d1([4.0, 5.0, 4.0, 5.0, 4.0, 5.0]).float()))


fn test_tile_2d_row() raises:
    print("test_tile_2d_row")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var t = a.tile([1, 2])
    var expected = Tensor.d2(
        [[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]]
    ).float()
    assert_true((t == expected))


fn test_tile_2d_col() raises:
    print("test_tile_2d_col")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var t = a.tile([2, 1])
    var expected = Tensor.d2(
        [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]]
    ).float()
    assert_true((t == expected))


fn test_tile_2d_both_axes() raises:
    print("test_tile_2d_both_axes")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var t = a.tile([2, 2])
    var expected = Tensor.d2(
        [
            [1.0, 2.0, 1.0, 2.0],
            [3.0, 4.0, 3.0, 4.0],
            [1.0, 2.0, 1.0, 2.0],
            [3.0, 4.0, 3.0, 4.0],
        ]
    ).float()
    assert_true((t == expected))


fn test_tile_backward_1d() raises:
    print("test_tile_backward_1d")
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True).float()
    var t = a.tile([2])
    t.sum().backward()
    assert_true((a.grad().all_close(Tensor.d1([2.0, 2.0, 2.0]).float())))


fn test_tile_backward_2d() raises:
    print("test_tile_backward_2d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True).float()
    var t = a.tile([2, 3])
    t.sum().backward()
    var expected_grad = Tensor.d2([[6.0, 6.0], [6.0, 6.0]]).float()
    assert_true(a.grad().all_close(expected_grad))


fn test_tile_single_axis_repeat_one() raises:
    print("test_tile_single_axis_repeat_one")
    var a = Tensor.d1([5.0, 6.0]).float()
    var t = a.tile([1])
    assert_true((t == a))


fn test_tile_edge_empty_tensor() raises:
    print("test_tile_edge_empty_tensor")
    var a = Tensor.d1([]).float()
    var t = a.tile([3])
    assert_true(t.numels() == 0)


fn test_tile_multi_axis_edge_case() raises:
    print("test_tile_multi_axis_edge_case")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var t = a.tile([0, 3])
    assert_true(t.numels() == 0)


fn test_flatten_forward_contiguous_1d() raises:
    print("test_flatten_forward_contiguous_1d")
    var a = Tensor.d1([1.0, 2.0, 3.0]).float()
    var f = a.flatten()
    assert_true((f == Tensor.d1([1.0, 2.0, 3.0]).float()))


fn test_flatten_forward_contiguous_2d() raises:
    print("test_flatten_forward_contiguous_2d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var f = a.flatten()
    assert_true((f == Tensor.d1([1.0, 2.0, 3.0, 4.0]).float()))


fn test_flatten_backward_contiguous() raises:
    print("test_flatten_backward_contiguous")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True).float()
    var f = a.flatten()
    var loss = f.sum()
    loss.backward()
    # each element contributes once, so gradient should be 1 everywhere
    assert_true(
        (a.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]]).float()))
    )


fn test_flatten_forward_view_slice() raises:
    print("test_flatten_forward_view_slice")
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).float()
    var v = a.slice(0, 1)  # take first row: [[1,2,3]]
    var f = v.flatten()
    assert_true((f == Tensor.d1([1.0, 2.0, 3.0]).float()))


fn test_flatten_backward_view_slice() raises:
    print("test_flatten_backward_view_slice")
    var a = Tensor.d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    ).float()
    var v = a.slice(0, 1)  # first row
    var f = v.flatten()
    var loss = f.sum()
    loss.backward()
    # only first row gets gradient ones, second row untouched
    assert_true(
        (
            a.grad().all_close(
                Tensor.d2([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]).float()
            )
        )
    )


fn test_flatten_backward_non_contiguous_stride() raises:
    print("test_flatten_backward_non_contiguous_stride")
    var a = Tensor.d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    ).float()
    var v = a.slice(1, 2, 1, axis=1)  # take column 1 → [[2],[5]]
    var f = v.flatten()
    var loss = f.sum()
    loss.backward()
    # gradient should land only in column 1
    assert_true(
        (
            a.grad().all_close(
                Tensor.d2([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]).float()
            )
        )
    )


fn test_flatten_forward_contiguous_3d() raises:
    print("test_flatten_forward_contiguous_3d")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    ).float()
    var f = a.flatten()
    assert_true(
        (f == Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).float())
    )


fn test_flatten_backward_contiguous_3d() raises:
    print("test_flatten_backward_contiguous_3d")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    ).float()
    var f = a.flatten()
    var loss = f.sum()
    loss.backward()
    # every element contributes once → gradient all ones
    assert_true(
        (
            a.grad().all_close(
                Tensor.d3(
                    [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
                ).float()
            )
        )
    )


fn test_flatten_forward_3d_slice() raises:
    print("test_flatten_forward_3d_slice")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    ).float()
    var v = a.slice(0, 1)  # take first "batch": [[1,2],[3,4]]
    var f = v.flatten()
    assert_true((f == Tensor.d1([1.0, 2.0, 3.0, 4.0]).float()))


fn test_flatten_backward_3d_slice() raises:
    print("test_flatten_backward_3d_slice")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    ).float()
    var v = a.slice(0, 1)  # first "batch"
    var f = v.flatten()
    var loss = f.sum()
    loss.backward()
    assert_true(
        (
            a.grad().all_close(
                Tensor.d3(
                    [[[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]
                ).float()
            )
        )
    )


fn test_flatten_full_default_forward_2d() raises:
    print("test_flatten_full_default_forward_2d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var f = a.flatten()
    var expected = Tensor.d1([1.0, 2.0, 3.0, 4.0]).float()
    assert_true((f == expected))


fn test_flatten_partial_forward_3d() raises:
    print("test_flatten_partial_forward_3d")
    var a = Tensor.d3(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ]
    ).float()
    # collapse dims 1..2 => new shape (2, 12)
    var f = a.flatten(1, 2)
    var expected = Tensor.d2(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
            ],
        ]
    ).float()
    assert_true((f == expected))


fn test_flatten_start_only_forward_3d() raises:
    print("test_flatten_start_only_forward_3d")
    var a = Tensor.d3(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
    ).float()  # shape (2,3,2)
    # flatten from axis=1 onward (1..end) => (2, 6)
    var f = a.flatten(1)
    var expected = Tensor.d2(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
    ).float()
    assert_true((f == expected))


fn test_flatten_start_eq_end_forward_noop_3d() raises:
    print("test_flatten_start_eq_end_forward_noop_3d")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    ).float()  # shape (2,2,2)
    # flatten a single axis (start == end) => shape should be identical and values unchanged
    var f = a.flatten(1, 1)
    assert_true((f == a))


fn test_flatten_backward_contiguous_2d() raises:
    print("test_flatten_backward_contiguous_2d")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True).float()
    var f = a.flatten()
    var loss = f.sum()
    loss.backward()
    var expected_grad = Tensor.d2([[1.0, 1.0], [1.0, 1.0]]).float()
    assert_true(a.grad().all_close(expected_grad))


fn test_flatten_backward_partial_3d() raises:
    print("test_flatten_backward_partial_3d")
    var a = Tensor.d3(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            [
                [13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0],
            ],
        ],
        requires_grad=True,
    ).float()
    # collapse dims 1..2 => shape (2,12)
    var f = a.flatten(1, 2)
    var loss = f.sum()
    loss.backward()
    # every original element contributed exactly once -> ones everywhere
    var expected_grad = Tensor.d3(
        [
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        ]
    ).float()
    assert_true(a.grad().all_close(expected_grad))


fn test_flatten_backward_view_strided_2d() raises:
    print("test_flatten_backward_view_strided_2d")
    var a = Tensor.d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    ).float()
    # take columns 0 and 2 using step=2 -> view of shape (2,2)
    var v = a.slice(0, 3, 2, axis=1)
    var f = v.flatten()
    var loss = f.sum()
    loss.backward()
    # grads should land only in column 0 and column 2
    var expected_grad = Tensor.d2([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]).float()
    assert_true(a.grad().all_close(expected_grad))


fn test_flatten_backward_3d_strided_view() raises:
    print("test_flatten_backward_3d_strided_view")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    ).float()
    # take the index 1 along axis=2: axis=2 slice [1,2) for axis=2
    var v = a.slice(1, 2, axis=2)  # selects [:,:,1] -> shape (2,2,1)
    var f = v.flatten()
    var loss = f.sum()
    loss.backward()
    var expected_grad = Tensor.d3(
        [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]
    ).float()
    assert_true(a.grad().all_close(expected_grad))


fn test_flatten_gradient_correctness_strided_view() raises:
    print("test_flatten_gradient_correctness_strided_view")

    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    ).float()

    # take the second column across all dims → [:,:,1]
    var v = a.slice(1, 2, axis=2)
    var f = v.flatten()
    var loss = f.sum()
    loss.backward()

    # gradient lands only in column-1 entries
    var expected = Tensor.d3(
        [[[0.0, 1.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 1.0]]]
    ).float()
    assert_true(a.grad().all_close(expected))


fn test_flatten_gradient_correctness() raises:
    print("test_flatten_gradient_correctness")

    var a = Tensor.d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    ).float()

    var f = a.flatten()  # shape (6,)
    var loss = f.sum()  # all elements contribute equally
    loss.backward()

    # each entry in `a` should get gradient 1.0
    var expected = Tensor.d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).float()
    assert_true(a.grad().all_close(expected))


fn test_shuffle() raises:
    print("test_shuffle")
    perm = List(2, 3, 0, 4, 1)
    a = Tensor.arange(5, requires_grad=True)
    shuffled = a.shuffle(perm=perm)
    sliced = shuffled[1:4]
    c = sliced * 42
    c.backward()
    expected = Tensor.d1([42.0, 0.0, 0.0, 42.0, 42.0]).float()
    assert_true(a.grad().all_close(expected))

fn test_fill() raises:
    print("test_fill")
    a = Tensor.zeros(10)
    a.fill(42)
    v = a.view(shape=[3], offset=2)
    v.fill(99)
    assert_true((v == Tensor.d1([99, 99, 99])), "view fill assertion failed")
    assert_true(
        (a == Tensor.d1([42, 42, 99, 99, 99, 42, 42, 42, 42, 42])),
        "view fill propagation1 to parent failed",
    )
    v1 = a.view(shape=[2, 5])
    v2 = v1[il(1), s(2, None, 2)]
    v2.fill(101)

    assert_true(
        (a == Tensor.d1([42, 42, 99, 99, 99, 42, 42, 101, 42, 101])),
        "view fill propagation2 to parent failed",
    )
    assert_true(
        (v.sum_all() == 3 * 99) and (v2.sum_all() == 2 * 101),
        "fill sum_all assertion failed for views",
    )
    b = Tensor.d1([1919, 1919])


    v2.set(b, s())

    assert_true(
        (
            a == Tensor.d1([42, 42, 99, 99, 99, 42, 42, 1919, 42, 1919])
        ),
        "view set propagation to parent failed",
    )


fn test_element_at() raises:
    print("test_element_at")
    a = Tensor.arange(10)
    v = a[s(2, 8, 2)]
    assert_true(
        v.max_index() == 6 and v.element_at(-4) == 2,
        "max_index and element_at assertion failed",
    )


fn test_argmin_max() raises:
    a = Tensor.d1([1, 4, -9, 2, 10, 8])
    v = a.view(shape=[4], offset=2)
    assert_true(
        (a.argmax(0) == Tensor[DType.int32].scalar(4)),
        "argmax assertion 1failed",
    )
    assert_true(
        (a.argmax() == Tensor[DType.int32].scalar(4)),
        "argmax assertion 2 failed",
    )

    assert_true(
        (a.argmin() == Tensor[DType.int32].scalar(2)),
        "tensor argmin assertion 1 failed",
    )
    assert_true(
        (v.argmax() == Tensor[DType.int32].scalar(2)),
        "view argmax assertion 1 failed",
    )

    assert_true(
        (v.argmin() == Tensor[DType.int32].scalar(0)),
        "view argmax assertion 1 failed",
    )


fn test_slice_backward() raises:
    print("test_slice_backward")
    alias dtype = DType.float32
    a = Tensor[dtype].d1([1, 2, 3, 4, 5, 6], requires_grad=True)
    r = a.reshape([2, 3])
    s = r[Slice(1, None, None), Slice(0, 3, 1)]
    s.sum().backward(42)
    assert_true(
        a.grad().as_tensor()[Slice(3, None, None)]
        == Tensor[dtype]([42, 42, 42])
    )


fn test_view_backward() raises:
    print("test_view_backward")
    alias dtype = DType.float32
    a = Tensor[dtype].d1([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], requires_grad=True)
    v = a.view(shape=Shape(2, 4), strides=Strides(4, 1), offset=2)
    assert_true(v == Tensor[dtype].d2([[3, 4, 5, 6], [7, 8, 9, 10]]))

    v2 = v.view(shape=Shape(2, 2), strides=Strides(2, 1), offset=2)

    assert_true(v2 == Tensor[dtype].d2([[3, 4], [5, 6]]))
    loss = v2.mean()
    loss.backward()
    assert_true(
        a.grad() == Tensor[dtype]([0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0])
    )
    assert_true(
        a.grad().as_tensor()[Slice(2, 6, 1)]
        == Tensor[dtype]([0.25, 0.25, 0.25, 0.25])
    )


fn test_complex_mixed_ops_backward() raises:
    print("test_complex_mixed_ops_backward")
    alias dtype = DType.float32

    a = Tensor[dtype].d2(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], requires_grad=True
    )

    v1 = a.view(shape=Shape(2, 4), strides=Strides(4, 1), offset=2)

    v2 = v1.view(shape=Shape(2, 2), strides=Strides(2, 1), offset=2)

    v3 = v2.view(shape=Shape(2, 2), strides=Strides(2, 1), offset=0)

    c = v3.contiguous()

    s = c.mean()

    s.backward(42)

    assert_true(
        a.grad().as_tensor()[Slice(0, 1, None), Slice(2, None, None)]
        == Tensor[dtype].d2([[10.5, 10.5]])
    )


fn test_view_chain_with_hidden_elements() raises:
    print("=== Mojo: View chain with hidden elements ===")

    var a = Tensor.d2(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]],
        requires_grad=True,
    )

    var v1 = a.view(shape=Shape(2, 4), strides=Strides(4, 1), offset=2)
    var v2 = v1.view(shape=Shape(2, 2), strides=Strides(1, 3), offset=2)

    var result = v2.sum()
    result.backward()

    expected = Tensor.d2(
        [[0, 0, 1, 1, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    )

    assert_true(a.grad() == expected)


fn test_gradient_flow_through_views() raises:
    print("=== Testing Gradient Flow Through View Chain ===")
    var a = Tensor.d2([[1, 2, 3, 4], [5, 6, 7, 8]], requires_grad=True)
    var v1 = a.view(shape=Shape(2, 2), strides=Strides(2, 1), offset=1)
    var v2 = v1.view(shape=Shape(1, 2), strides=Strides(2, 1), offset=3)
    var result = v2.sum()
    result.backward()

    var expected = Tensor.d2([[0, 0, 0, 1], [1, 0, 0, 0]])
    assert_true(a.grad() == expected)


fn main() raises:
    print("Starting tensor test cases")
    test_complex_mixed_ops_backward()
    test_view_backward()
    test_slice_backward()
    test_view_chain_with_hidden_elements()
    test_gradient_flow_through_views()

    test_argmin_max()
    test_fill()
    test_element_at()
    test_shuffle()
    test_randint()
    test_slice_single_axis()
    test_slice_single_axis_positive()
    test_slice_single_axis_negative_indices()
    test_slice_single_axis_step_greater_than_1()
    test_slice_single_axis_step_negative()
    test_slice_single_axis_full_axis()
    test_slice_single_axis_single_element()
    test_slice_multi_axis_basic()
    test_slice_multi_axis_negative_indices()
    test_slice_multi_axis_step()
    test_slice_multi_axis_mixed()

    test_mask()
    test_count()
    test_max_min()
    test_max_min_mixed()
    test_vector_matrix_matmul()
    test_tensor_dot()
    test_validate_matmul_last_2_dims()

    test_reshape_gradient_2d()
    test_reshape_noop()
    test_reshape_slice_sum_backward()
    test_simple_chain()
    test_tensor_reuse_add()
    test_tensor_reuse_mixed()
    test_tensor_reuse_in_two_branches()
    test_tensor_reuse_deep_chain()
    test_tensor_reuse_broadcasting()
    test_tensor_shared_multiple_paths()

    test_broadcast_sub()
    test_sum()
    test_tensor_multiplications()
    test_broadcast_add()
    test_tensor_mean()
    test_training_convergence()
    test_grad_flow_through_reshape()
    test_forward_multivariate_prediction()
    test_weights_bias_gradients()
    test_transpose_gradients()
    test_reshape_grad_flow()
    test_broadcast_mul()

    test_reshape_gradient()
    test_reshape()
    test_reshape_preserves_grad_accumulation()
    test_power()

    test_add_2_tensors()
    test_broadcast_add_2_tensors()

    test_tensor_of_list()
    # test_grad_copy_on_reshape()
    test_mean()
    test_arange()
    test_scalar_tensor()
    test_sum()
    test_item()
    test_multi_dimensional_reshape()
    test_reshape_tensor_to_scalar()
    test_reshape_scalar_to_tensor()
    test_miscellaneous()
    test_random()
    test_view()

    test_matmul_broadcasting()
    test_transpose_grad()
    test_zero_grad()
    test_matmul_shapes()
    test_mean_with_keepdims()
    test_scalar_addition()
    test_sum_all_dims()
    test_broadcast_addition()
    test_sum_specific_axis()

    # test_nested_operations()
    # These need to be enabled
    test_reshape_gradient_flatten()
    test_multiple_reshapes()
    test_reshape_reused_twice_correct_grad()

    test_mean_scalar()
    test_mean_1d()
    test_mean_2d_all_axes()
    test_mean_axis0()
    test_mean_axis1_keepdims()
    test_mean_multiple_axes()
    test_mean_no_axes()
    test_mean_no_grad()

    test_tensor_div_scalar_2d()
    test_tensor_div_scalar_nonuniform()
    test_tensor_div_scalar()
    test_tensor_scalar_subtract()
    test_tensor_scalar_add_mul_pow()
    test_sum_all_elements()
    test_sum_axis0()
    test_sum_axis1_keepdims()
    test_sum_multi_axes()
    test_sum_all_axes_keepdims()
    test_sum_gradient_accumulation()
    test_scalar_sum_forward()
    test_scalar_sum_backward()
    test_scalar_sum_custom_grad()
    test_scalar_sum_keepdims_true()
    test_scalar_sum_explicit_axes()

    test_add_scalar_scalar()
    test_add_scalar_1d()
    test_add_1d_1d()
    test_add_2d_scalar()
    test_add_2d_1d()
    test_add_3d_1d()
    test_add_3d_2d()
    test_add_broadcast_degenerate()
    # test_nested_operations()
    # test_detach()
    # View tensor multiplication
    test_matmul_scalar_output()
    test_matmul_tensor_tensor()
    # test_matmul_tensor_view()
    test_matmul_view_tensor()
    test_matmul_view_view()
    test_matmul_transposed_tensor_tensor()
    test_matmul_transposed_tensor_view()
    test_matmul_transposed_view_tensor()
    test_matmul_transposed_view_view()
    # Topological sort verification?

    test_repeated_tensor_use()
    test_diamond_dependency()
    test_shared_dependency_multiple_paths()
    test_shared_tensor_twice()
    test_broadcast_and_reuse()
    test_branching_square_add()
    test_merge_of_dependent_branches()
    test_square_and_identity_path()
    test_topological_sort_required()

    # test_add_mismatch_shapes()
    # __sub__
    test_sub_same_shape()
    test_sub_broadcast_row()
    test_sub_scalar_tensor()
    test_sub_tensor_scalar()
    test_sub_broadcast_col()
    # __mul__
    test_mul_broadcast_col()
    test_mul_broadcast_row()
    test_mul_scalar_tensor()
    test_mul_tensor_scalar()
    test_mul_same_shape()
    # __truediv__/__rtruediv__
    test_scalar_div_tensor()
    test_scalar_div_tensor_multiple()
    test_scalar_div_tensor_2d()
    test_empty_tensor()
    test_flat_view_chain_backprop()
    test_reshape_backward()
    test_reshape_exp()
    test_slice_grad()
    test_add_backward()
    test_reshape_backward_scalar()
    test_add_tensor_and_view()
    test_add_tensors()
    test_subtract_scalar()
    test_add_scalar()
    test_powering()
    test_invert()
    test_negate_absolute()
    test_exponentiation()
    test_inplace_update()
    test_grad_update()
    test_grads_on_tensor_init()
    test_scalar_indexing()
    test_view_of_view()
    test_sum_all()
    test_dot_product()
    test_matrix_vector_matmul()
    test_matrix_matrix_matmul()
    test_batched_matrix_matmul()
    test_broadcasted_matrix_matmul()
    test_high_dim_batched_matmul()
    test_matmul_no_grad()
    test_matmul_mixed_grad()
    test_matmul_shape_validation()
    test_batched_matrix_vector_matmul()
    test_matrix_vector_mm_backward_A_batched()
    test_matrix_vector_mm_batched_forward()
    test_matrix_vector_mm_simple()
    test_matrix_vector_mm_backward_A()
    test_matrix_vector_mm_backward_b()
    test_matrix_vector_mm_backward_b_batched()
    test_matrix_vector_mm_deeper_batch_forward()
    test_matrix_vector_mm_backward_b_deeper_batch()
    test_batched_matmul_vector_rhs_broadcast()

    test_vector_matrix_mm_batched_matrix()
    test_batched_matmul_vector_rhs_broadcast()
    test_vector_matrix_mm_simple()
    test_vector_matrix_mm_backward_vector()

    test_vector_matrix_mm_backward_matrix()

    test_vector_matrix_mm_backward_batched_matrix_vector_grad()
    test_vector_matrix_mm_backward_batched_matrix_matrix_grad()
    test_large_tensor_backprop()

    test_repeat_1d_axis0()
    test_repeat_backward_simple()
    test_repeat_backward_with_view_slice()
    test_repeat_backward_with_strided_view()
    test_repeat_backward_multi_axis()
    test_repeat_backward_chain_view_repeat()
    test_tile_1d_basic()
    test_tile_1d_multi()
    test_tile_2d_row()
    test_tile_2d_col()
    test_tile_2d_both_axes()
    test_tile_backward_1d()
    test_tile_backward_2d()
    test_tile_single_axis_repeat_one()
    # test_tile_edge_empty_tensor()
    # test_tile_multi_axis_edge_case()

    test_flatten_forward_contiguous_1d()
    test_flatten_forward_contiguous_2d()
    test_flatten_backward_contiguous()
    test_flatten_forward_view_slice()
    test_flatten_backward_view_slice()
    test_flatten_backward_non_contiguous_stride()
    test_flatten_forward_contiguous_3d()
    test_flatten_backward_contiguous_3d()
    test_flatten_forward_3d_slice()
    test_flatten_backward_3d_slice()
    test_flatten_backward_3d_strided_view()

    test_flatten_full_default_forward_2d()
    test_flatten_partial_forward_3d()
    test_flatten_start_only_forward_3d()
    test_flatten_start_eq_end_forward_noop_3d()
    test_flatten_backward_contiguous_2d()
    test_flatten_backward_partial_3d()
    test_flatten_backward_view_strided_2d()
    test_flatten_gradient_correctness()
    test_flatten_gradient_correctness_strided_view()

    print("Finished running tensor test cases")
