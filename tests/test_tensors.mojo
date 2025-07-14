# %s/Tensor\.walk_backward(\([^)]*\))/\1.backward()/g

from testing import assert_true
from common_utils import do_assert, assert_grad
from tensors import Tensor
from intlist import IntList
from shapes import Shape
from ancestry import Ancestors
from common_utils import *


fn test_tensor_shared_multiple_paths() raises:
    print("test_tensor_shared_multiple_paths")
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y = x * 3              # 6
    var z = x + 4              # 6
    var out = y + z            # 12

    out.backward()

    # ∂out/∂x = ∂y/∂x + ∂z/∂x = 3 + 1 = 4
    assert_true(out.item() == 12.0, "Value check")
    assert_true(x.grad[].item() == 4.0, "Correct accumulated gradient")


fn test_tensor_reuse_broadcasting() raises:
    print("test_tensor_reuse_broadcasting")
    var x = Tensor.d1([1, 2, 3], requires_grad=True)
    var y = x + x  # [2, 4, 6]

    y.sum().backward()

    assert_true(x.grad[].all_close(Tensor.d1([2, 2, 2])), "Gradient doubles due to reuse")



fn test_tensor_reuse_deep_chain() raises:
    print("test_tensor_reuse_deep_chain")
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y = x + 1           # 3
    var z = y * x           # 3 * 2 = 6
    var w = z + x           # 6 + 2 = 8

    w.backward()

    assert_true(w.item() == 8.0, "Value check")
    # ∂w/∂x = ∂z/∂x + 1
    # z = (x + 1) * x → ∂z/∂x = (1)*x + (x+1)*1 = x + x + 1 = 2x + 1 = 5
    assert_true(x.grad[].item() == 5 + 1, "∂w/∂x = ∂z/∂x + ∂x/∂x = 6")


fn test_tensor_reuse_in_two_branches() raises:
    print("test_tensor_reuse_in_two_branches")
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y1 = x + 1    # 3
    var y2 = x * 5    # 10
    var z = y1 + y2   # 13

    z.backward()

    assert_true(z.item() == 13.0, "Value check")
    assert_true(x.grad[].item() == 1 + 5, "∂z/∂x = 1 (from y1) + 5 (from y2)")


fn test_tensor_reuse_mixed() raises:
    print("test_tensor_reuse_mixed")
    var x = Tensor.scalar(3.0, requires_grad=True)
    var y = x * x + x  # 3 * 3 + 3 = 12

    y.backward()

    assert_true(y.item() == 12.0, "Value check")
    assert_true(x.grad[].item() == 2 * 3 + 1, "∂y/∂x = 2x + 1 = 6 + 1 = 7")


fn test_tensor_reuse_add() raises:
    print("test_tensor_reuse_add")
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y = x + x  # 2 + 2 = 4

    y.backward()

    assert_true(y.item() == 4.0, "Value check")
    assert_true(x.grad[].item() == 2.0, "∂y/∂x = 1 + 1 = 2 (reuse)")


fn test_simple_chain() raises:
    print("test_simple_chain")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = Tensor.scalar(3.0, requires_grad=True)
    var c = a * b  # 2 * 3 = 6
    var d = c + b  # 6 + 3 = 9

    d.backward()
    assert_true(d.item() == 9.0, "Value check")
    assert_true(a.grad[].item() == 3.0, "∂d/∂a = b = 3")

fn test_scalar_mul_scalar() raises:
    print("test_scalar_mul_scalar")
    var a = Tensor.scalar(3.0, requires_grad=True)
    var b = Tensor.scalar(4.0, requires_grad=True)

    var c = a * b
    c.backward()

    assert_true(c.item() == 12.0)
    assert_true(a.grad[].item() == 4.0)
    assert_true(b.grad[].item() == 3.0)


fn test_1d_mul_1d_same_shape() raises:
    print("test_1d_mul_1d_same_shape")
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor.d1([4.0, 5.0, 6.0], requires_grad=True)

    var c = a * b
    c.sum().backward()

    assert_true((c == Tensor.d1([4.0, 10.0, 18.0])).all_true())
    assert_true(a.grad[].all_close(Tensor.d1([4.0, 5.0, 6.0])))
    assert_true(b.grad[].all_close(Tensor.d1([1.0, 2.0, 3.0])))


fn test_2d_mul_2d_same_shape() raises:
    print("test_2d_mul_2d_same_shape")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var c = a * b
    c.sum().backward()

    assert_true((c == Tensor.d2([[5.0, 12.0], [21.0, 32.0]])).all_true())
    assert_true(a.grad[].all_close(b))
    assert_true(b.grad[].all_close(a))


fn test_broadcast_2d_1d_mul() raises:
    print("test_broadcast_2d_1d_mul")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.d1([5.0, 6.0], requires_grad=True).to_dtype[DType.float32]()

    var c = a * b  # broadcasts b along rows
    c.print()
    summ = c.sum()
    summ.print()

    summ.backward()

    assert_true((c == Tensor.d2([[5.0, 12.0], [15.0, 24.0]])).all_true())
    assert_true(a.grad[].all_close(Tensor.d2([[5.0, 6.0], [5.0, 6.0]])))

    # b.grad should sum over rows
    assert_true(
        b.grad[].all_close(Tensor.d1([4.0, 6.0]).to_dtype[DType.float32]())
    )


fn test_broadcast_1d_2d_mul() raises:
    print("test_broadcast_1d_2d_mul")
    var a = Tensor.d1([2.0, 3.0], requires_grad=True).to_dtype[DType.float32]()
    var b = Tensor.d2([[4.0, 5.0], [6.0, 7.0]], requires_grad=True)

    var c = a * b  # a broadcasts over rows
    c.sum().backward()

    assert_true((c == Tensor.d2([[8.0, 15.0], [12.0, 21.0]])).all_true())

    assert_true(
        a.grad[].all_close(Tensor.d1([10.0, 12.0]).to_dtype[DType.float32]())
    )
    assert_true(b.grad[].all_close(Tensor.d2([[2.0, 3.0], [2.0, 3.0]])))


fn test_3d_broadcast_mul() raises:
    print("test_3d_broadcast_mul")
    var a = Tensor.d3(
        [[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True
    )  # shape (2, 2, 1)
    var b = Tensor.d3([[[5.0, 6.0]]], requires_grad=True)  # shape (1, 1, 2)

    var c = a * b  # result shape (2, 2, 2)
    c.sum().backward()

    assert_true(c.shape == Shape.of(2, 2, 2))
    assert_true(a.grad[].shape == Shape.of(2, 2, 1))
    assert_true(b.grad[].shape == Shape.of(1, 1, 2))


fn test_mul_one_requires_grad() raises:
    print("test_mul_one_requires_grad")
    var a = Tensor.d1([1.0, 2.0, 3.0])  # no grad
    var b = Tensor.d1([4.0, 5.0, 6.0], requires_grad=True)

    var c = a * b
    c.sum().backward()

    assert_true(b.grad[].all_close(a))


fn test_scalar_tensor_mul() raises:
    print("test_scalar_tensor_mul")
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = Tensor.d1([1.0, 2.0, 3.0])

    var c = a * b
    c.sum().backward()

    assert_true(a.grad[].item() == 6.0)  # sum of b


fn test_unsqueeze() raises:
    print("test_unsqueeze")
    _ = """tensor = Tensor.rand(2,3, requires_grad=True)
    tensor2 = tensor.unsqueeze(0)
    tensor.print()
    tensor2.target[].print()
    tensor2[IntList(0, 0, 0)] = 100
    tensor.print()
    tensor2.target[].print()"""
    pass


fn test_tensor_mean() raises:
    print("test_tensor_mean")
    a = Tensor.scalar(5.0, requires_grad=True)
    m = a.mean()
    m.backward()
    assert_true(m.item() == 5.0)
    assert_true(a.grad[].item() == 1.0)
    a.free()
    m.free()

    a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    m = a.mean()
    assert_true(m.item() == 2.0)
    m.backward()
    assert_true(a.grad[].all_close(Tensor.d1([1 / 3, 1 / 3, 1 / 3])))
    a.free()
    m.free()

    A = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    M = A.mean()
    assert_true(M.item() == 2.5)
    M.backward()
    expected = Tensor.d2([[0.25, 0.25], [0.25, 0.25]])
    assert_true(A.grad[].all_close(Tensor.d2([[0.25, 0.25], [0.25, 0.25]])))
    A.free()
    M.free()

    a1 = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    m1 = a1.mean(axes=[0], keepdims=False)
    assert_true(m1.all_close(Tensor.d1([2.0, 3.0]).to_dtype[DType.float32]()))
    m1.backward()
    assert_true(
        a1.grad[].all_close(
            Tensor.d2([[0.5, 0.5], [0.5, 0.5]]).to_dtype[DType.float32]()
        )
    )

    AA = Tensor.d2([[1.0, 2.0], [3.0, 5.0]], requires_grad=True)
    MM = AA.mean(axes=[1], keepdims=False)
    assert_true(MM.all_close(Tensor.d1([1.5, 4.0]).to_dtype[DType.float32]()))
    MM.backward()
    assert_true(AA.grad[].all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])))

    C = Tensor.d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
    mm = C.mean(axes=[1], keepdims=True)
    assert_true(mm.all_close(Tensor.d2([[2.0], [3.0]])))
    mm.backward()
    assert_true(C.grad[].all_close(Tensor.d2([[0.5, 0.5], [0.5, 0.5]])))

    A3 = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    m12 = A3.mean(axes=[1, 2], keepdims=False)  # shape: (2,)
    assert_true(m12.all_close(Tensor.d1([2.5, 6.5]).to_dtype[DType.float32]()))
    m12.backward()
    expected_grad = Tensor.d3(
        [[[0.25, 0.25], [0.25, 0.25]], [[0.25, 0.25], [0.25, 0.25]]]
    )
    assert_true(A3.grad[].all_close(expected_grad))

    A2 = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    m2 = A2.mean(axes=[-1])  # same as axis=1
    assert_true(m2.all_close(Tensor[DType.float32].d1([1.5, 3.5])))

    a2 = Tensor.d2([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
    m_ = a2.mean(axes=IntList.Empty)
    assert_true(m_.item() == 25.0)


fn test_forward_multivariate_prediction() raises:
    print("test_forward_multivariate_prediction")
    # x.shape = (3, 2), w.shape = (2, 1)
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var w = Tensor.d2([[1.0], [2.0]])  # So y = x @ w = [[5], [11], [17]]
    var y_pred = x.matmul(w)

    assert_true((y_pred == Tensor.d2([[5.0], [11.0], [17.0]])).all_true())
    var b = Tensor.d1([1.0]).to_dtype[DType.float32]()
    var y = x.matmul(w) + b
    assert_true((y == Tensor.d2([[6.0], [12.0], [18.0]])).all_true())


fn test_weights_bias_gradients() raises:
    print("test_weights_bias_gradients")
    var xx = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var ww = Tensor.d2([[0.1], [0.2]], requires_grad=True)
    var bb = Tensor[DType.float32].d1([0.5], requires_grad=True)
    var target_ = Tensor.d2([[1.0], [2.0]])

    var y_prediction = xx.matmul(ww) + bb
    var _loss = ((y_prediction - target_) ** 2).mean([])
    _loss.backward()

    # Gradient should flow to w and b
    assert_true(ww.grad[].shape == ww.shape)
    assert_true(bb.grad[].shape == bb.shape)


fn test_training_convergence() raises:
    print("test_training_convergence")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var y = Tensor.d2([[13.0], [23.0], [33.0]])

    var w = Tensor.rand(2, 1, requires_grad=True)
    var b = Tensor.rand(1, requires_grad=True)

    for epoch in range(1000):
        var y_pred = x.matmul(w) + b
        var loss = ((y_pred - y) ** 2).mean()
        # loss.print()
        loss.backward()

        # SGD
        w.data[] -= 0.01 * w.grad[].data[]
        b.data[] -= 0.01 * b.grad[].data[]
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
    b = a.T()  # (2, 2) -> (2, 2)
    b.sum().backward()
    assert_true((a.grad[] == Tensor.d2([[1, 1], [1, 1]])).all_true())

    # Case 2: Transpose + reshape with non-square
    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
    b = a.T().reshape(Shape.of(2, 3))  # (3, 2) → (2, 3)
    b.sum().backward()
    assert_true((a.grad[] == Tensor.d2([[1, 1, 1], [1, 1, 1]])).all_true())

    # Case 3: Chain transposes (A.T().T())
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = a.T().T()  # Should equal A
    b.sum().backward()
    assert_true((a.grad[] == Tensor.d2([[1, 1], [1, 1]])).all_true())


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
    assert_true((a.grad[] == Tensor.d1([1, 1, 1, 1])).all_true())

    # Case 2: 1D → 2D reshape
    a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    b = a.reshape(Shape.of(2, 2))
    (b * 2).sum().backward()
    assert_true((a.grad[] == Tensor.d1([2, 2, 2, 2])).all_true())

    # === 2D Tensor Cases ===
    # Case 3: 2D → 2D reshape (contiguous)
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = a.reshape(Shape.of(4, 1))
    (b**2).sum().backward()
    assert_true((a.grad[] == Tensor.d2([[2, 4], [6, 8]])).all_true())

    # Case 4: 2D → 1D reshape
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = a.reshape(
        Shape.of(
            4,
        )
    )
    (b + 1).sum().backward()
    assert_true((a.grad[] == Tensor.d2([[1, 1], [1, 1]])).all_true())

    # === 3D Tensor Cases ===
    # Case 5: 3D → 2D reshape
    _ = """a64 = Tensor[DType.float64].d3([[[1.0, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    b = a.reshape(Shape.of(2, 4))
    b.mean().backward()
    assert_true((a64.grad[] == Tensor[DType.float64].full(a64.shape, 0.125)).all_true())

    # === Edge Cases ===
    # Case 6: Empty tensor reshape
    a = Tensor.d1([], requires_grad=True)
    b = a.reshape(Shape.of(0,))
    b.sum().backward()  # Should not crash
    assert_true(a.grad[].shape == Shape.of(0,))"""

    # Case 7: Non-contiguous reshape
    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = a.T().reshape(Shape.of(2, 3))  # Tests view tracking
    b.sum().backward()
    a.grad[].print()
    assert_true((a.grad[] == Tensor.d2([[1, 1, 1], [1, 1, 1]])).all_true())

    # === Advanced Cases ===
    # Case 8: Chained reshapes
    a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    b = a.reshape(Shape.of(2, 2)).reshape(
        Shape.of(
            4,
        )
    )
    b.sum().backward()
    assert_true((a.grad[] == Tensor.d1([1, 1, 1, 1])).all_true())

    # Case 9: Reshape with existing gradients
    a = Tensor.d1([1, 2], requires_grad=True)
    _ = (a * 2).sum().backward()  # a.grad = [2.0, 2.0]
    b = a.reshape(Shape.of(2, 1))
    b.sum().backward()  # Gradient accumulation
    assert_true((a.grad[] == Tensor.d1([3, 3])).all_true())  # 2 + 1

    # Case 10: Reshape after detach
    _ = """a = Tensor.d1([1, 2], requires_grad=True)
    b = a.detach().reshape(Shape.of(2, 1))  # Should break grad flow
    b.sum().backward()  # Should NOT affect a.grad
    assert_true(a.grad[] is None)  # Because of detach()"""


fn test_reshape_gradient() raises:
    print("test_reshape_gradient")
    # 1. Reshape scalar to (1,) and back
    a = Tensor.scalar(42, requires_grad=True)
    b = a.reshape(Shape.of(1))
    c = b.reshape(Shape.of(1))  # back to scalar
    d = c * Tensor.scalar(2)
    d.backward()
    a.grad[].print()
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
    # d.backward()
    d.backward()
    a.grad[].print()
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
    a.grad[].print()
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
    a.grad[].print()
    a.print()
    assert_grad(a, Tensor.d1([1, 2, 3, 4]), "reshape with (1,4) roundtrip")
    assert_true(
        (a.grad[] == Tensor.d1([1, 2, 3, 4])).all_true(),
        "reshape with (1,4) roundtrip",
    )

    # 7. Reshape then broadcast in op
    a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    b = a.reshape(Shape.of(2, 2))
    c = b + Tensor.scalar(10)  # broadcast add
    d = c.sum()
    d.backward()
    a.grad[].print()
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
    assert_true(X.grad[].item() == -2)
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
    assert_grad(b, Tensor.d2([[-2.0, -2.0], [-2.0, -2.0]]), "3D - 2D → b")
    c.print()
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
    tensor = Tensor.arange(24).reshape(2, 3, 4)
    tensor.print()
    result = tensor**2
    result.print()


fn test_grad_flow_through_reshape() raises:
    print("test_grad_flow_through_reshape")
    a = Tensor.of(1.0, 2.0, 3.0, requires_grad=True)

    # First operation using 'a'
    b = a + 1.0
    b.sum().backward()
    assert_true((a.grad[] == Tensor.of(1.0, 1.0, 1.0)).all_true())

    # Reshape should not clone or copy gradients
    reshaped = a.reshape(Shape.of(3))

    # reshaped.grad[] should not exist on its own — we assert that it refers to the same grad storage
    # assert_true((reshaped.grad[] == Tensor.of(1.0, 1.0, 1.0)).all_true())

    # New operation from reshaped
    (reshaped * 2).sum().backward()

    # Original 'a' should now have accumulated gradient
    assert_true((a.grad[] == Tensor.of(3.0, 3.0, 3.0)).all_true())


fn test_reshape_preserves_grad_accumulation() raises:
    print("test_reshape_preserves_grad_accumulation")
    # Chained reshape should still accumulate gradients
    a = Tensor.of(1.0, 2.0, 3.0, requires_grad=True)
    b = a.reshape(Shape.of(3))
    c = b.reshape(Shape.of(1, 3))

    d = c.sum()
    d.backward()

    a.grad[].print()  # Should be [1.0, 1.0, 1.0]
    assert_true((a.grad[] == Tensor.of(1.0, 1, 1)).all_true())


fn test_multi_dimensional_reshape() raises:
    print("test_multi_dimensional_reshape")
    # (2, 3) → (3, 2)
    a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b = a.reshape(Shape.of(3, 2))

    assert_true(b.shape == Shape.of(3, 2))
    d = b.sum()
    d.backward()

    a.grad[].print()  # Should be [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


fn test_reshape_tensor_to_scalar() raises:
    print("test_reshape_tensor_to_scalar")
    # (1,) → reshape to scalar
    a = Tensor.of(42.0, requires_grad=True)
    b = a.reshape(Shape.Void)

    assert_true(b.is_scalar())
    assert_true(b[IntList()] == Scalar(42.0))

    c = b * 2
    c.backward()

    a.grad[].print()  # Should be [2.0]


fn test_reshape_scalar_to_tensor() raises:
    print("test_reshape_scalar_to_tensor")
    # Scalar → reshape to (1,)
    a = Tensor.scalar(42.0, requires_grad=True)
    b = a.reshape(Shape.of(1))  # should share data and allow backprop

    assert_true(b[0] == Scalar(42.0))
    c = b * 3
    c.backward()
    a.grad[].print()  # Should be [3.0]


fn test_miscellaneous() raises:
    print("test_miscellaneous")
    a = Tensor.of(1.0, 2.0, 3.0, requires_grad=True)
    b = Tensor.scalar(5.0)
    c = a + b
    c.sum().backward()
    # should be [1, 1, 1]
    assert_true((a.grad[] == Tensor.of(1.0, 1, 1)).all_true())
    reshaped = (a + b).mean().reshape()
    Tensor.scalar(42, requires_grad=True).sum().backward()  # This one crashes
    reshaped.backward()  # backward does not return anything


fn test_mean() raises:
    print("test_mean")
    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = a.mean([0])
    assert_true((b == Tensor[DType.float32].d1([2.5, 3.5, 4.5])).all_true())
    b.backward()
    assert_true(
        (
            a.grad[]
            == Tensor[DType.float32].d2([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        ).all_true()
    )
    # Mean over all → scalar
    s = a.mean([])
    assert_true((s == Tensor[DType.float32].scalar(3.5)).all_true())
    s.backward()
    # a.grad == [[1/6, 1/6, 1/6], [1/6, 1/6, 1/6]] + 0.5 from previous backward call
    # a.grad[].print()
    assert_true(
        a.grad[].all_close(
            Tensor.d2(
                [
                    [0.1666666, 0.1666666, 0.1666666],
                    [0.1666666, 0.1666666, 0.1666666],
                ]
            )
            + 0.5
        )
    )
    a.zero_grad()
    s.backward()
    assert_true(
        a.grad[].all_close(
            Tensor.d2(
                [
                    [0.1666666, 0.1666666, 0.1666666],
                    [0.1666666, 0.1666666, 0.1666666],
                ]
            )
        )
    )


fn test_sum() raises:
    print("test_sum")
    # 1. Basic Value Tests
    a = Tensor.of(1, 2, 3)
    b = Tensor.d1([1, 2, 3])
    c = Tensor.of([1, 2, 3])
    assert_true((a.sum([0]) == Tensor.scalar(6)).all_true())
    assert_true((b.sum([0]) == Tensor.scalar(6)).all_true())
    assert_true((c.sum([0]) == Tensor.scalar(6)).all_true())
    assert_true((a.sum([0], keepdims=True) == Tensor.of(6)).all_true())
    assert_true((a.sum([0], keepdims=True) == Tensor.of([6])).all_true())

    # 2. Multi-Dimensional Tensor Tests
    a = Tensor.d2([[1, 2], [3, 4]])  # Shape (2, 2)
    assert_true((a.sum([0]) == Tensor.of([4, 6])).all_true())
    assert_true((a.sum([1]) == Tensor.of([3, 7])).all_true())
    assert_true((a.sum([0, 1]) == Tensor.scalar(10)).all_true())
    assert_true((a.sum([0, 1], keepdims=True) == Tensor.d2([[10]])).all_true())
    # 3. Scalar Input
    a = Tensor.scalar(42)
    assert_true((a.sum([]) == Tensor.scalar(42)).all_true())
    # assert_true((a.sum([0]) == Tensor.scalar(42)).all_true())
    # 4. Keepdims=True
    a = Tensor.d2([[1, 2], [3, 4]])  # (2,2)
    out = a.sum([1], keepdims=True)  # Should be (2,1)
    assert_true(
        (out == Tensor.d2([[3], [7]])).all_true()
        and out.shape == Shape.of(2, 1)
    )
    # 5. Gradient Checks
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = a.sum([1])  # b shape (2,)
    b.backward()
    # Now a.grad should be Tensor.of([[1, 1], [1, 1]])
    assert_true(
        (b == Tensor.of(3, 7)).all_true()
        and b.requires_grad
        and (a.grad[] == Tensor.d2([[1, 1], [1, 1]])).all_true()
    )

    # 6. Broadcasting Compatibility
    a = Tensor.d2([[1, 2, 3]], requires_grad=True)  # (1,3)
    b = a.sum([0], keepdims=False)  # (3,)
    b.backward()
    # a.grad == Tensor.of([[1, 1, 1]])
    assert_true(
        (b == Tensor.of(1, 2, 3)).all_true()
        and b.requires_grad
        and (a.grad[] == Tensor.d2([[1, 1, 1]])).all_true()
    )
    tensor = Tensor.of(1, 2, 3, 4, requires_grad=True)
    result = tensor.sum(axes=[], keepdims=False)
    assert_true((result == Tensor.scalar(10)).all_true())
    result.backward()
    assert_true(
        (
            tensor.grad[] == Tensor[DType.float32].of(1.0, 1.0, 1.0, 1.0)
        ).all_true()
    )
    tensor = Tensor.arange(24).reshape(2, 3, 4)
    result = tensor.sum(axes=[], keepdims=False)
    assert_true(result.item() == 276.0)
    result = tensor.sum(axes=[], keepdims=True)
    assert_true((result == Tensor.d3([[[276.0]]])).all_true())

    ones = Tensor.ones(3, 3)
    summed = ones.sum(axes=[0], keepdims=True)
    assert_true(
        (summed == Tensor.d2([[3, 3, 3]])).all_true(),
        "keepdim = True sum assertion 1 failed",
    )
    ones = Tensor.ones(3, 3)
    summed = ones.sum(axes=[0])
    expect = Tensor.of(3, 3, 3)
    assert_true((summed == expect).all_true(), "1D sum assertion failed")

    tensor = Tensor.arange(1, 21).reshape(2, 5, 2)
    summed = tensor.sum(axes=[1])
    _ = """[2D Tensor(2, 2), Type: float32, requires_grad: False]
        [
            [25.0, 30.0, ],
            [75.0, 80.0, ],
    ]"""
    expect = Tensor.of[2](25, 30, 75, 80)
    assert_true(
        (summed == expect).all_true(), "Sum across axis 1 assertion failed"
    )

    summed = tensor.sum(axes=[0])
    expect = Tensor.of[2](12, 14, 16, 18, 20, 22, 24, 26, 28, 30)
    assert_true(
        (summed == expect).all_true(), "Sum across axis 0 assertion failed"
    )

    expect = Tensor.scalar(210)
    summed = tensor.sum()
    assert_true(
        (summed == expect).all_true(), "Sum across axis 2 assertion failed"
    )


fn test_broadcast_add_2_tensors() raises:
    print("test_broadcast_add_2_tensors")
    print("Test broadcast add 2 tensors")

    tensor1 = Tensor.of(1, 2, 3, 4, requires_grad=True)
    tensor2 = Tensor.of(6, requires_grad=True)

    result = tensor1 + tensor2
    assert_true(
        (result == Tensor.of(7, 8, 9, 10)).all_true(),
        "broadcast add assertion 1 failed",
    )
    result.backward()

    assert_true(
        (
            tensor1.grad[]
            == Tensor.d1(
                [
                    1,
                    1,
                    1,
                    1,
                ]
            )
        ).all_true(),
        "grad check 1 - assertion failed",
    )

    assert_true(
        (tensor2.grad[] == Tensor.of([4])).all_true(),
        "grad check 2 - assertion failed",
    )

    tensor1 = Tensor.of[3](1, 2, 3, 4, 5, 6, requires_grad=True)
    tensor2 = Tensor.of(6, requires_grad=True)

    result = tensor1 + tensor2

    result.backward()

    assert_true(
        (result == Tensor.of[3](7, 8, 9, 10, 11, 12)).all_true(),
        "broadcast add assertion 2 failed",
    )

    assert_true(
        (
            tensor1.grad[]
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
        ).all_true(),
        "grad check 3 - assertion failed",
    )

    assert_true(
        (tensor2.grad[] == Tensor.of([6])).all_true(),
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
            tensor1.grad[]
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
        ).all_true(),
        "grad check 5 - assertion failed",
    )

    assert_true(
        (
            tensor2.grad[]
            == Tensor.d3(
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
        ).all_true(),
        "grad check 6 - assertion failed",
    )


fn test_add_2_tensors() raises:
    print("test_add_2_tensors")
    print("test_add_2_tensors")

    tensor_a = Tensor.rand(256, 256, requires_grad=True)
    tensor_b = Tensor.rand(256, 256, requires_grad=True)
    assert_true(
        tensor_a.shape == tensor_b.shape,
        "Input tensors shape match assertion failed",
    )
    out_tensor = tensor_a + tensor_b
    assert_true(
        tensor_a.shape == out_tensor.shape,
        "Input/output tensors shape match assertion failed",
    )
    parent1 = out_tensor.ancestors.get(0)[].tensor()
    parent2 = out_tensor.ancestors.get(1)[].tensor()
    left_parent_is_tensor1 = (parent1 == tensor_a).all_true()
    right_parent_is_tensor2 = (parent2 == tensor_b).all_true()
    assert_true(
        left_parent_is_tensor1 == True and right_parent_is_tensor2 == True,
        "Output tensor ancestry validation failed",
    )

    out_tensor.invoke_grad_fn()
    Tensor.free_all(tensor_a, tensor_b, out_tensor)


fn test_factor_mul_by() raises:
    print("test_factor_mul_by")
    print("test_factor_mul_by")

    tensor = Tensor.rand(256, 256, requires_grad=True)
    out_tensor = 2 * tensor
    assert_true(
        len(out_tensor.ancestors.get(0)[].tensor()) == 65536,
        "Output tensor ancestors length validation failed",
    )
    out_tensor.invoke_grad_fn()
    print("The following is out tensor gradient")
    Tensor.free_all(tensor, out_tensor)


fn test_mul_by_factor() raises:
    print("test_mul_by_factor")
    print("test_mul_by_factor")
    tensor = Tensor.rand(128, 256, requires_grad=True)
    out_tensor = tensor * 100
    assert_true(
        len(out_tensor.ancestors.get(0)[].tensor()) == 32768,
        "Output tensor ancestors length validation failed",
    )
    out_tensor.invoke_grad_fn()
    print("The following is out tensor gradient")
    Tensor.free_all(tensor, out_tensor)


fn test_add_value() raises:
    print("test_add_value")

    tensor = Tensor.rand(1024, 64, requires_grad=True)
    out_tensor = 2 * tensor
    assert_true(
        len(out_tensor.ancestors.get(0)[].tensor()) == 65536,
        "Output tensor ancestors length validation failed",
    )
    out_tensor.invoke_grad_fn()
    print("The following is out tensor gradient")
    Tensor.free_all(tensor, out_tensor)


fn test_arange() raises:
    print("test_arange")
    tensor = Tensor.arange(0, 10)
    # expected = Tensor.of(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    expected = Tensor.of(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    # print(tensor.dtype, expected.dtype)
    is_true = (tensor == expected).all_true()
    assert_true(is_true, "arange gen check assertion failed")

    Tensor.free_all(tensor, expected)

    tensor1 = Tensor.arange(0, -5, -0.5)
    expected1 = Tensor.of(
        0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5
    )
    is_true = (tensor1 == expected1).all_true()
    assert_true(is_true, "arange negative step assertion failed")
    Tensor.free_all(tensor1, expected1)


fn test_matmul_optim() raises:
    print("test_matmul_optim")
    X = Tensor.rand(3, 4, requires_grad=True)
    W = Tensor.rand(4, 5, requires_grad=True)
    B = Tensor.rand(1, requires_grad=True)
    P = X.matmul_optim(W) + B
    print("P.shape: ", P.shape, P.requires_grad)
    P.backward()
    W.grad[].print()


fn test_transpose_matmul() raises:
    print("test_transpose_matmul")
    A = Tensor.rand(3, 3, requires_grad=True)
    A.grad[].print()
    A_T = A.T()
    A_T.backward()
    A.grad[].print()


fn test_random() raises:
    print("test_random")
    rand_tensor = Tensor.rand(10)
    rand_tensor.print()

    fn each(e: Scalar[DType.float32]) -> Bool:
        return e >= 0 and e < 1

    holds_true = rand_tensor.for_all(each)
    assert_true(holds_true, "rand min and max range assertion failed")

    rand_tensor2 = Tensor.rand(10, 20, min=-2, max=2)

    fn each2(e: Scalar[DType.float32]) -> Bool:
        return e >= -2 and e < 2

    holds_true = rand_tensor2.for_all(each2)
    assert_true(holds_true, "rand min(-2) and max(2) range assertion failed")
    Tensor.free_all(rand_tensor, rand_tensor2)


fn test_item() raises:
    print("test_item")
    tensor = Tensor.of(42)
    assert_true(tensor.item() == 42)


fn test_view() raises:
    print("test_view")
    tensor = Tensor.rand(1).reshape()
    view = tensor.view(Shape.Void)
    view.strides.print()
    print("From view: ", view[IntList()])
    assert_true(
        tensor.shape == view.base_tensor[].shape,
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
        tensor_2d.shape == Shape.of(3, 3) and tensor_2d.numels() == 9,
        "Tensor from assertion 3 failed",
    )
    tensor2d = Tensor.d2([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert_true(
        tensor2d.shape == Shape.of(3, 3) and tensor2d.numels() == 9,
        "Tensor from assertion 3 failed",
    )

    tensor3d = Tensor.d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert_true(
        tensor3d.shape == Shape.of(2, 2, 2) and tensor3d.numels() == 8,
        "Tensor from assertion 4 failed",
    )


fn test_scalar_tensor() raises:
    print("test_scalar_tensor")
    tensor = Tensor.scalar(42)
    assert_true(
        (
            tensor.item() == 42.0
            and tensor.shape == Shape.Void
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
        tensor.shape == Shape.Unit, "Unit tensor shape assertion failure"
    )
    reshaped = tensor.reshape(1, 1)
    assert_true(
        reshaped.shape == Shape.of(1, 1) and reshaped[0, 0] == tensor[0],
        "post reshape shape and get assertion failed",
    )
    tensor = Tensor.scalar(42)
    reshaped = tensor.reshape(1, 1)
    assert_true(
        reshaped.shape == Shape.of(1, 1) and reshaped[0, 0] == tensor.item(),
        "post reshape shape and get assertion failed for scalar tensor",
    )
    reshaped = tensor.reshape(1)
    assert_true(
        reshaped.shape == Shape.Unit and reshaped[0] == tensor.item(),
        "post reshape 2 - shape and get assertion failed for scalar tensor",
    )
    assert_true(
        reshaped.reshape(1, 1, 1, 1)[0, 0, 0, 0] == tensor.item(),
        "post reshape 3 - item assertion failed for scalar tensor",
    )

    tensor = Tensor.rand(1, 1)
    reshaped = tensor.reshape()
    assert_true(
        reshaped.shape == Shape.Void and reshaped.item() == tensor[0, 0],
        "post reshape random tensor - shape and get assertion failed",
    )
    tensor = Tensor.scalar(42, requires_grad=True)
    result = tensor * 3
    # result.backward()
    result.backward()
    assert_true(tensor.grad[].item() == 3.0)
    tensor2 = tensor.reshape(1)
    tensor.print_grad()
    result = tensor2 * 42
    tensor.print_grad()
    tensor2.print_grad()
    # result.backward()
    result.backward()
    tensor.print_grad()
    tensor2.print_grad()
    # tensor3 = tensor2.reshape(1,1,1,1,1)
    tensor3 = tensor2.reshape(1, 1, 1, 1, 1)
    result = tensor3 * 12
    # result.backward()
    result.backward()

    tensor3.print_grad()
    tensor2.print_grad()
    tensor.print_grad()


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


fn main() raises:
    print("Starting tensor test cases")
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
    test_transpose_matmul()
    # test_matmul_optim()

    test_reshape_gradient()
    test_reshape()
    test_reshape_preserves_grad_accumulation()
    test_power()

    test_add_2_tensors()
    test_broadcast_add_2_tensors()

    test_tensor_of_list()
    # test_grad_copy_on_reshape()
    test_mul_by_factor()
    test_mean()
    test_arange()
    test_reshape()
    test_scalar_tensor()
    test_add_value()
    test_sum()
    test_item()
    test_reshape_preserves_grad_accumulation()
    test_multi_dimensional_reshape()
    test_reshape_tensor_to_scalar()
    test_reshape_scalar_to_tensor()
    test_miscellaneous()
    test_random()
    test_transpose_matmul()
    test_factor_mul_by()
    test_view()
    print("Finished running tensor test cases")


