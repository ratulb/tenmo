from common_utils import do_assert, assert_grad


fn test_scalar_mul_scalar() raises:
    var a = Tensor.scalar(3.0, requires_grad=True)
    var b = Tensor.scalar(4.0, requires_grad=True)

    var c = a * b
    Tensor.walk_backward(c)

    assert_true(c.item() == 12.0)
    assert_true(a.grad[].item() == 4.0)
    assert_true(b.grad[].item() == 3.0)


fn test_1d_mul_1d_same_shape() raises:
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor.d1([4.0, 5.0, 6.0], requires_grad=True)

    var c = a * b
    c.sum().backward()

    assert_true((c == Tensor.d1([4.0, 10.0, 18.0])).all_true())
    assert_true(a.grad[].all_close(Tensor.d1([4.0, 5.0, 6.0])))
    assert_true(b.grad[].all_close(Tensor.d1([1.0, 2.0, 3.0])))


fn test_2d_mul_2d_same_shape() raises:
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var c = a * b
    c.sum().backward()

    assert_true((c == Tensor.d2([[5.0, 12.0], [21.0, 32.0]])).all_true())
    assert_true(a.grad[].all_close(b))
    assert_true(b.grad[].all_close(a))


fn test_broadcast_2d_1d_mul() raises:
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
    var a = Tensor.d1([1.0, 2.0, 3.0])  # no grad
    var b = Tensor.d1([4.0, 5.0, 6.0], requires_grad=True)

    var c = a * b
    c.sum().backward()

    assert_true(b.grad[].all_close(a))


fn test_scalar_tensor_mul() raises:
    var a = Tensor.scalar(2.0, requires_grad=True)
    var b = Tensor.d1([1.0, 2.0, 3.0])

    var c = a * b
    c.sum().backward()

    assert_true(a.grad[].item() == 6.0)  # sum of b


fn test_unsqueeze() raises:
    _ = """tensor = Tensor.rand(2,3, requires_grad=True)
    tensor2 = tensor.unsqueeze(0)
    tensor.print()
    tensor2.target[].print()
    tensor2[IntList(0, 0, 0)] = 100
    tensor.print()
    tensor2.target[].print()"""
    pass


fn test_tensor_mean() raises:
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
    # x.shape = (3, 2), w.shape = (2, 1)
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var w = Tensor.d2([[1.0], [2.0]])  # So y = x @ w = [[5], [11], [17]]
    var y_pred = x.matmul(w)

    assert_true((y_pred == Tensor.d2([[5.0], [11.0], [17.0]])).all_true())
    var b = Tensor.d1([1.0]).to_dtype[DType.float32]()
    var y = x.matmul(w) + b
    assert_true((y == Tensor.d2([[6.0], [12.0], [18.0]])).all_true())


fn test_weights_bias_gradients() raises:
    var xx = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var ww = Tensor.d2([[0.1], [0.2]], requires_grad=True)
    var bb = Tensor[DType.float32].d1([0.5], requires_grad=True)
    var target_ = Tensor.d2([[1.0], [2.0]])

    var y_prediction = xx.matmul(ww) + bb
    var _loss = ((y_prediction - target_) ** 2).mean([])
    Tensor.walk_backward(_loss)

    # Gradient should flow to w and b
    assert_true(ww.grad[].shape == ww.shape)
    assert_true(bb.grad[].shape == bb.shape)


fn test_training_convergence() raises:
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var y = Tensor.d2([[13.0], [23.0], [33.0]])

    var w = Tensor.rand(2, 1, requires_grad=True)
    var b = Tensor.rand(1, requires_grad=True)

    for epoch in range(1000):
        var y_pred = x.matmul(w) + b
        var loss = ((y_pred - y) ** 2).mean()
        # loss.print()
        Tensor.walk_backward(loss)

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
    # Case 1: Simple 2D transpose
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = a.T()  # (2, 2) → (2, 2)
    Tensor.walk_backward(b.sum())
    assert_true((a.grad[] == Tensor.d2([[1, 1], [1, 1]])).all_true())

    # Case 2: Transpose + reshape with non-square
    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
    b = a.T().reshape(Shape.of(2, 3))  # (3, 2) → (2, 3)
    Tensor.walk_backward(b.sum())
    assert_true((a.grad[] == Tensor.d2([[1, 1, 1], [1, 1, 1]])).all_true())

    # Case 3: Chain transposes (A.T().T())
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = a.T().T()  # Should equal A
    Tensor.walk_backward(b.sum())
    assert_true((a.grad[] == Tensor.d2([[1, 1], [1, 1]])).all_true())


fn test_reshape_grad_flow() raises:
    """Test suite for gradient flow through reshape operations."""

    # === 1D Tensor Cases ===
    # Case 1: Simple 1D → 1D reshape
    var a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    var b = a.reshape(
        Shape.of(
            4,
        )
    )
    Tensor.walk_backward(b.sum())
    assert_true((a.grad[] == Tensor.d1([1, 1, 1, 1])).all_true())

    # Case 2: 1D → 2D reshape
    a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    b = a.reshape(Shape.of(2, 2))
    Tensor.walk_backward((b * 2).sum())
    assert_true((a.grad[] == Tensor.d1([2, 2, 2, 2])).all_true())

    # === 2D Tensor Cases ===
    # Case 3: 2D → 2D reshape (contiguous)
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = a.reshape(Shape.of(4, 1))
    Tensor.walk_backward((b**2).sum())
    assert_true((a.grad[] == Tensor.d2([[2, 4], [6, 8]])).all_true())

    # Case 4: 2D → 1D reshape
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = a.reshape(
        Shape.of(
            4,
        )
    )
    Tensor.walk_backward((b + 1).sum())
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
    Tensor.walk_backward(b.sum())  # Should not crash
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
    # 1. Reshape scalar to (1,) and back
    a = Tensor.scalar(42, requires_grad=True)
    b = a.reshape(Shape.of(1))
    c = b.reshape(Shape.of(1))  # back to scalar
    d = c * Tensor.scalar(2)
    Tensor.walk_backward(d)
    a.grad[].print()
    assert_grad(a, Tensor.scalar(2), "scalar reshape chain -> a")

    # 2. Reshape 1D → 2D → back to 1D
    a = Tensor.d1([1, 2, 3, 4], requires_grad=True)
    b = a.reshape(Shape.of(2, 2))
    c = b.reshape(
        Shape.of(
            4,
        )
    )
    d = c * Tensor.d1([10, 20, 30, 40])
    # Tensor.walk_backward(d)
    d.backward()
    a.grad[].print()
    assert_grad(a, Tensor.d1([10, 20, 30, 40]), "1D -> 2D -> 1D grad")

    # 3. Reshape 2D to 1D and multiply
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)  # shape (2,2)
    b = a.reshape(
        Shape.of(
            4,
        )
    )
    c = b * Tensor.d1([10, 20, 30, 40])
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d2([[10, 20], [30, 40]]), "2D -> 1D grad")

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
    Tensor.walk_backward(d)
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
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([1, 1, 1, 1]), "reshape -> sum -> backward")

    # 6. Reshape with degenerate axis: (4,) -> (1, 4) -> (4,)
    a = Tensor.d1([5, 6, 7, 8], requires_grad=True)
    b = a.reshape(Shape.of(1, 4))
    c = b.reshape(
        Shape.of(
            4,
        )
    )
    d = c * Tensor.d1([1, 2, 3, 4])
    Tensor.walk_backward(d)
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
    Tensor.walk_backward(d)
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
    # 1. Scalar * Scalar
    a = Tensor.scalar(3, requires_grad=True)
    b = Tensor.scalar(4, requires_grad=True)
    c = a * b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.scalar(4), "Scalar * Scalar -> a")
    assert_grad(b, Tensor.scalar(3), "Scalar * Scalar -> b")
    do_assert(c, Tensor.scalar(12), "Scalar * Scalar")

    # 2. Scalar * 1D
    a = Tensor.scalar(2, requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a * b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.scalar(6), "Scalar * 1D -> a")
    assert_grad(b, Tensor.d1([2, 2, 2]), "Scalar * 1D -> b")
    do_assert(c, Tensor.d1([2, 4, 6]), "Scalar * 1D")

    # 3. 1D * Scalar
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = Tensor.scalar(2, requires_grad=True)
    c = a * b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([2, 2, 2]), "1D * Scalar -> a")
    assert_grad(b, Tensor.scalar(6), "1D * Scalar -> b")
    do_assert(c, Tensor.d1([2, 4, 6]), "1D * Scalar")

    # 4. 1D * 1D
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = Tensor.d1([4, 5, 6], requires_grad=True)
    c = a * b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([4, 5, 6]), "1D * 1D -> a")
    assert_grad(b, Tensor.d1([1, 2, 3]), "1D * 1D -> b")
    do_assert(c, Tensor.d1([4, 10, 18]), "1D * 1D")

    # 5. 2D * Scalar
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor.scalar(3, requires_grad=True)
    c = a * b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d2([[3, 3], [3, 3]]), "2D * Scalar -> a")
    assert_grad(b, Tensor.scalar(10), "2D * Scalar -> b")
    do_assert(c, Tensor.d2([[3, 6], [9, 12]]), "2D * Scalar")

    # 6. Scalar * 2D
    a = Tensor.scalar(3, requires_grad=True)
    b = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    c = a * b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.scalar(10), "Scalar * 2D -> a")
    assert_grad(b, Tensor.d2([[3, 3], [3, 3]]), "Scalar * 2D -> b")
    do_assert(c, Tensor.d2([[3, 6], [9, 12]]), "Scalar * 2D")

    # 7. 2D * 1D (row-wise broadcasting)
    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor.d1([10, 20, 30], requires_grad=True)
    c = a * b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d2([[10, 20, 30], [10, 20, 30]]), "2D * 1D -> a")
    assert_grad(b, Tensor.d1([5, 7, 9]), "2D * 1D -> b")
    do_assert(c, Tensor.d2([[10, 40, 90], [40, 100, 180]]), "2D * 1D")

    # 8. 1D * 2D (reverse broadcast)
    a = Tensor.d1([10, 20, 30], requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a * b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([5, 7, 9]), "1D * 2D -> a")
    assert_grad(b, Tensor.d2([[10, 20, 30], [10, 20, 30]]), "1D * 2D -> b")
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
    Tensor.walk_backward(c)
    assert_grad(
        a,
        Tensor.d3(
            [
                [[10, 20], [10, 20]],
                [[10, 20], [10, 20]],
            ]
        ),
        "3D * 1D -> a",
    )
    assert_grad(b, Tensor.d1([16, 20]), "3D * 1D -> b")
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
    Tensor.walk_backward(c)
    assert_grad(
        a,
        Tensor.d3(
            [
                [[10, 20], [30, 40]],
                [[10, 20], [30, 40]],
            ]
        ),
        "3D * 2D -> a",
    )
    assert_grad(
        b,
        Tensor.d2(
            [
                [6, 8],
                [10, 12],
            ]
        ),
        "3D * 2D -> b",
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
    Tensor.walk_backward(c)
    assert_grad(
        a,
        Tensor.d3(
            [
                [[10, 10], [10, 10]],
                [[10, 10], [10, 10]],
            ]
        ),
        "3D * Scalar -> a",
    )
    assert_grad(b, Tensor.scalar(36), "3D * Scalar -> b")
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
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([6]), "(1,) * (3,) -> a")
    assert_grad(b, Tensor.d1([5, 5, 5]), "(1,) * (3,) -> b")
    do_assert(c, Tensor.d1([5, 10, 15]), "(1,) * (3,)")

    # 13. Degenerate broadcast: (1,1) * (2,3)
    a = Tensor.d2([[2]], requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a * b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d2([[21]]), "(1,1) * (2,3) -> a")
    assert_grad(b, Tensor.d2([[2, 2, 2], [2, 2, 2]]), "(1,1) * (2,3) -> b")
    do_assert(c, Tensor.d2([[2, 4, 6], [8, 10, 12]]), "(1,1) * (2,3)")


fn test_broadcast_sub() raises:
    # 1. Scalar - Scalar
    X = Tensor.scalar(100, requires_grad=True)
    summ = (X - X).sum()
    summ.backward()
    assert_grad(X, Tensor.scalar(0), "(X - X) scalars -> a")
    summ = (X - X - X - X).sum()
    summ.backward()
    assert_true(X.grad[].item() == -2)
    assert_grad(X, Tensor.scalar(-2), "(X - X - X - X) scalars -> a")
    a = Tensor.scalar(5, requires_grad=True)
    b = Tensor.scalar(3, requires_grad=True)
    c = a - b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.scalar(1), "Scalar - Scalar -> a")
    assert_grad(b, Tensor.scalar(-1), "Scalar - Scalar -> b")
    do_assert(c, Tensor.scalar(2), "Scalar - Scalar")

    # 2. Scalar - 1D
    a = Tensor.scalar(10, requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a - b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.scalar(3), "Scalar - 1D -> a")
    assert_grad(b, Tensor.d1([-1, -1, -1]), "Scalar - 1D -> b")
    do_assert(c, Tensor.d1([9, 8, 7]), "Scalar - 1D")

    # 3. 1D - Scalar
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = Tensor.scalar(10, requires_grad=True)
    c = a - b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([1, 1, 1]), "1D - Scalar -> a")
    assert_grad(b, Tensor.scalar(-3), "1D - Scalar -> b")
    do_assert(c, Tensor.d1([-9, -8, -7]), "1D - Scalar")

    # 4. 1D - 1D (same shape)
    a = Tensor.d1([5, 6, 7], requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a - b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([1, 1, 1]), "1D - 1D -> a")
    assert_grad(b, Tensor.d1([-1, -1, -1]), "1D - 1D -> b")
    do_assert(c, Tensor.d1([4, 4, 4]), "1D - 1D (same shape)")

    # 5. 2D - Scalar
    a = Tensor.d2([[10, 20], [30, 40]], requires_grad=True)
    b = Tensor.scalar(5, requires_grad=True)
    c = a - b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d2([[1, 1], [1, 1]]), "2D - Scalar -> a")
    assert_grad(b, Tensor.scalar(-4), "2D - Scalar -> b")
    do_assert(c, Tensor.d2([[5, 15], [25, 35]]), "2D - Scalar")

    # 6. Scalar - 2D
    a = Tensor.scalar(100, requires_grad=True)
    b = Tensor.d2([[10, 20], [30, 40]], requires_grad=True)
    c = a - b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.scalar(4), "Scalar - 2D -> a")
    assert_grad(b, Tensor.d2([[-1, -1], [-1, -1]]), "Scalar - 2D -> b")
    do_assert(c, Tensor.d2([[90, 80], [70, 60]]), "Scalar - 2D")

    # 7. 2D - 1D (broadcast over rows)
    a = Tensor.d2([[10, 20, 30], [40, 50, 60]], requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a - b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d2([[1, 1, 1], [1, 1, 1]]), "2D - 1D -> a")
    assert_grad(b, Tensor.d1([-2, -2, -2]), "2D - 1D -> b")
    do_assert(c, Tensor.d2([[9, 18, 27], [39, 48, 57]]), "2D - 1D")

    # 8. 1D - 2D (reverse broadcast over rows)
    a = Tensor.d1([100, 200, 300], requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a - b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([2, 2, 2]), "1D - 2D -> a")
    assert_grad(b, Tensor.d2([[-1, -1, -1], [-1, -1, -1]]), "1D - 2D -> b")
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
    Tensor.walk_backward(c)
    assert_grad(
        a,
        Tensor.d3(
            [
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
            ]
        ),
        "3D - 1D -> a",
    )
    assert_grad(b, Tensor.d1([-4, -4]), "3D - 1D -> b")
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
    Tensor.walk_backward(c)
    assert_grad(
        a,
        Tensor.d3(
            [
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
            ]
        ),
        "3D - 2D -> a",
    )
    assert_grad(b, Tensor.d2([[-2.0, -2.0], [-2.0, -2.0]]), "3D - 2D -> b")
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
    Tensor.walk_backward(c)
    assert_grad(
        a,
        Tensor.d3(
            [
                [[1, 1], [1, 1]],
                [[1, 1], [1, 1]],
            ]
        ),
        "3D - Scalar -> a",
    )
    assert_grad(b, Tensor.scalar(-8), "3D - Scalar -> b")
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
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([3]), "(1,) - (3,) -> a")
    assert_grad(b, Tensor.d1([-1, -1, -1]), "(1,) - (3,) -> b")
    do_assert(c, Tensor.d1([99, 98, 97]), "(1,) - (3,)")

    # 13. Degenerate broadcast: (1, 1) - (2, 3)
    a = Tensor.d2([[100]], requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a - b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d2([[6]]), "(1,1) - (2,3) -> a")
    assert_grad(
        b, Tensor.d2([[-1, -1, -1], [-1, -1, -1]]), "(1,1) - (2,3) -> b"
    )
    do_assert(c, Tensor.d2([[99, 98, 97], [96, 95, 94]]), "(1,1) - (2,3)")


fn test_broadcast_add() raises:
    # 1. Scalar + Scalar
    a = Tensor.scalar(5, requires_grad=True)
    b = Tensor.scalar(3, requires_grad=True)
    c = a + b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.scalar(1), "Scalar + Scalar -> a")
    assert_grad(b, Tensor.scalar(1), "Scalar + Scalar -> b")
    do_assert(c, Tensor.scalar(8), "Scalar + Scalar")

    # 2. Scalar + 1D
    a = Tensor.scalar(2, requires_grad=True)
    b = Tensor.d1([1, 2, 3], requires_grad=True)
    c = a + b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.scalar(3), "Scalar + 1D -> a")
    assert_grad(b, Tensor.d1([1, 1, 1]), "Scalar + 1D -> b")
    do_assert(c, Tensor.d1([3, 4, 5]), "Scalar + 1D")

    # 3. 1D + Scalar (reverse)
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = Tensor.scalar(2, requires_grad=True)
    c = a + b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([1, 1, 1]), "1D + Scalar -> a")
    assert_grad(b, Tensor.scalar(3), "1D + Scalar -> b")
    do_assert(c, Tensor.d1([3, 4, 5]), "1D + Scalar")

    # 4. 1D + 1D (same shape)
    a = Tensor.d1([1, 2, 3], requires_grad=True)
    b = Tensor.d1([4, 5, 6], requires_grad=True)
    c = a + b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([1, 1, 1]), "1D + 1D -> a")
    assert_grad(b, Tensor.d1([1, 1, 1]), "1D + 1D -> b")
    do_assert(c, Tensor.d1([5, 7, 9]), "1D + 1D (same shape)")

    # 5. 2D + Scalar
    a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor.scalar(10, requires_grad=True)
    c = a + b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d2([[1, 1], [1, 1]]), "2D + Scalar -> a")
    assert_grad(b, Tensor.scalar(4), "2D + Scalar -> b")
    do_assert(c, Tensor.d2([[11, 12], [13, 14]]), "2D + Scalar")

    # 6. Scalar + 2D (reverse)
    a = Tensor.scalar(10, requires_grad=True)
    b = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    c = a + b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.scalar(4), "Scalar + 2D -> a")
    assert_grad(b, Tensor.d2([[1, 1], [1, 1]]), "Scalar + 2D -> b")
    do_assert(c, Tensor.d2([[11, 12], [13, 14]]), "Scalar + 2D")

    # 7. 2D + 1D (broadcast over rows)
    a = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor.d1([10, 20, 30], requires_grad=True)
    c = a + b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d2([[1, 1, 1], [1, 1, 1]]), "2D + 1D -> a")
    assert_grad(b, Tensor.d1([2, 2, 2]), "2D + 1D -> b")
    do_assert(c, Tensor.d2([[11, 22, 33], [14, 25, 36]]), "2D + 1D row-wise")

    # 8. 1D + 2D (reverse broadcast over rows)
    a = Tensor.d1([10, 20, 30], requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a + b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([2, 2, 2]), "1D + 2D -> a")
    assert_grad(b, Tensor.d2([[1, 1, 1], [1, 1, 1]]), "1D + 2D -> b")
    do_assert(c, Tensor.d2([[11, 22, 33], [14, 25, 36]]), "1D + 2D row-wise")

    # 9. 3D + 1D (broadcast over last dim)
    a = Tensor.d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    b = Tensor.d1([10, 20], requires_grad=True)
    c = a + b
    Tensor.walk_backward(c)
    assert_grad(
        a, Tensor.d3([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), "3D + 1D -> a"
    )
    assert_grad(b, Tensor.d1([4, 4]), "3D + 1D -> b")
    do_assert(
        c,
        Tensor.d3([[[11, 22], [13, 24]], [[15, 26], [17, 28]]]),
        "3D + 1D last-dim broadcast",
    )

    # 10. 3D + 2D (broadcast over batch dim)
    a = Tensor.d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    b = Tensor.d2([[10, 20], [30, 40]], requires_grad=True)
    c = a + b
    Tensor.walk_backward(c)
    assert_grad(
        a, Tensor.d3([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), "3D + 2D -> a"
    )
    assert_grad(b, Tensor.d2([[2, 2], [2, 2]]), "3D + 2D -> b")
    do_assert(
        c,
        Tensor.d3([[[11, 22], [33, 44]], [[15, 26], [37, 48]]]),
        "3D + 2D batch-dim broadcast",
    )

    # 11. 3D + Scalar
    a = Tensor.d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    b = Tensor.scalar(100, requires_grad=True)
    c = a + b
    Tensor.walk_backward(c)
    assert_grad(
        a, Tensor.d3([[[1, 1], [1, 1]], [[1, 1], [1, 1]]]), "3D + Scalar -> a"
    )
    assert_grad(b, Tensor.scalar(8), "3D + Scalar -> b")
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
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d1([3]), "(1,) + (3,) -> a")
    assert_grad(b, Tensor.d1([1, 1, 1]), "(1,) + (3,) -> b")
    do_assert(c, Tensor.d1([11, 21, 31]), "(1,) + (3,)")

    # 14. Degenerate broadcast: (1, 1) + (2, 3)
    a = Tensor.d2([[5]], requires_grad=True)
    b = Tensor.d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    c = a + b
    Tensor.walk_backward(c)
    assert_grad(a, Tensor.d2([[6]]), "(1,1) + (2,3) -> a")
    assert_grad(b, Tensor.d2([[1, 1, 1], [1, 1, 1]]), "(1,1) + (2,3) -> b")
    do_assert(c, Tensor.d2([[6, 7, 8], [9, 10, 11]]), "(1,1) + (2,3)")


fn test_power() raises:
    tensor = Tensor.arange(24).reshape(2, 3, 4)
    tensor.print()
    result = tensor**2
    result.print()


fn test_grad_flow_through_reshape() raises:
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
    # Chained reshape should still accumulate gradients
    a = Tensor.of(1.0, 2.0, 3.0, requires_grad=True)
    b = a.reshape(Shape.of(3))
    c = b.reshape(Shape.of(1, 3))

    d = c.sum()
    d.backward()

    a.grad[].print()  # Should be [1.0, 1.0, 1.0]
    assert_true((a.grad[] == Tensor.of(1.0, 1, 1)).all_true())


fn test_multi_dimensional_reshape() raises:
    # (2, 3) → (3, 2)
    a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b = a.reshape(Shape.of(3, 2))

    assert_true(b.shape == Shape.of(3, 2))
    d = b.sum()
    d.backward()

    a.grad[].print()  # Should be [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


fn test_reshape_tensor_to_scalar() raises:
    # (1,) → reshape to scalar
    a = Tensor.of(42.0, requires_grad=True)
    b = a.reshape(Shape.Void)

    assert_true(b.is_scalar())
    assert_true(b[IntList()] == Scalar(42.0))

    c = b * 2
    c.backward()

    a.grad[].print()  # Should be [2.0]


fn test_reshape_scalar_to_tensor() raises:
    # Scalar → reshape to (1,)
    a = Tensor.scalar(42.0, requires_grad=True)
    b = a.reshape(Shape.of(1))  # should share data and allow backprop

    assert_true(b[0] == Scalar(42.0))
    c = b * 3
    c.backward()
    a.grad[].print()  # Should be [3.0]


fn test_miscellaneous() raises:
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
    Tensor.walk_backward(b)
    # Now a.grad should be Tensor.of([[1, 1], [1, 1]])
    assert_true(
        (b == Tensor.of(3, 7)).all_true()
        and b.requires_grad
        and (a.grad[] == Tensor.d2([[1, 1], [1, 1]])).all_true()
    )

    # 6. Broadcasting Compatibility
    a = Tensor.d2([[1, 2, 3]], requires_grad=True)  # (1,3)
    b = a.sum([0], keepdims=False)  # (3,)
    Tensor.walk_backward(b)
    # a.grad == Tensor.of([[1, 1, 1]])
    assert_true(
        (b == Tensor.of(1, 2, 3)).all_true()
        and b.requires_grad
        and (a.grad[] == Tensor.d2([[1, 1, 1]])).all_true()
    )
    tensor = Tensor.of(1, 2, 3, 4, requires_grad=True)
    result = tensor.sum(axes=[], keepdims=False)
    assert_true((result == Tensor.scalar(10)).all_true())
    Tensor.walk_backward(result)
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

    expect = Tensor.of[5](3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
    summed = tensor.sum()
    assert_true(
        (summed == expect).all_true(), "Sum across axis 2 assertion failed"
    )


fn test_broadcast_add_2_tensors() raises:
    print("Test broadcast add 2 tensors")

    tensor1 = Tensor.of(1, 2, 3, 4, requires_grad=True)
    tensor2 = Tensor.of(6, requires_grad=True)

    result = tensor1 + tensor2
    assert_true(
        (result == Tensor.of(7, 8, 9, 10)).all_true(),
        "broadcast add assertion 1 failed",
    )
    Tensor.walk_backward(result)

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

    Tensor.walk_backward(result)

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

    Tensor.walk_backward(result)

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


fn test_tensor_multiplications() raises:
    test_scalar_mul_scalar()
    test_1d_mul_1d_same_shape()
    test_2d_mul_2d_same_shape()
    test_broadcast_2d_1d_mul()
    test_broadcast_1d_2d_mul()
    test_3d_broadcast_mul()
    test_scalar_tensor_mul()
    test_mul_one_requires_grad()


fn main() raises:
    test_tensor_multiplications()
    test_broadcast_add()
    test_broadcast_sub()
    _ = """test_transpose_matmul()
    #test_matmul_optim()
    test_training_convergence()
    test_tensor_mean()
    test_grad_flow_through_reshape()
    test_forward_multivariate_prediction()
    test_weights_bias_gradients()

    test_transpose_gradients()
    test_reshape_grad_flow()
    test_reshape_gradient()
    test_broadcast_mul()
    test_reshape()
    test_reshape_preserves_grad_accumulation()
    test_power()

    test_add_2_tensors()
    test_broadcast_add_2_tensors()

    test_tensor_of_list()
    test_grad_copy_on_reshape()
    test_mul_by_factor()
    test_mean()
    test_sum()
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
    test_view()"""


### Mojo Tensor
### Implement tensor library in mojo from first principles

from math import iota, exp, floor
from random import seed, random_float64
from time import perf_counter_ns
from algorithm import vectorize
from sys import simdwidthof
from utils.numerics import max_finite
from os import abort
from memory import UnsafePointer, memcpy, memset, memset_zero
from shapes import Shape
from intlist import IntList
from ancestry import Ancestors
from common_utils import log_debug, piped
from operators import (
    __tensor_op_tensor__,
    AddTensor,
    SubtractTensor,
    MulTensor,
    __tensor_op_scalar__,
    AddScalar,
    SubtractScalar,
    MulScalar,
    sum_across_rows,
    sum_across_cols,
    Power,
    scalar_ops,
    Add,
    Subtract,
    Multiply,
)
from collections import Set


struct Tensor[dtype: DType = DType.float32](
    Copyable & Movable & Sized & Stringable
):
    alias GradBox = UnsafePointer[Self]
    alias Address = UnsafePointer[Tensor[dtype]]
    var shape: Shape
    var data: UnsafePointer[Scalar[dtype]]
    var requires_grad: Bool
    var grad: Self.GradBox
    var ancestors: Ancestors[dtype]
    var grad_fn: Optional[fn () escaping raises -> None]
    var base: UnsafePointer[Tensor[dtype]]  # Only allocated on need basis

    fn __init__(out self, *axes_spans: Int, requires_grad: Bool = False):
        shape = Shape(axes_spans)
        self = Self(shape, requires_grad)

    fn __init__(
        out self,
        shape: Shape,
        data: UnsafePointer[Scalar[dtype]],
        requires_grad: Bool = False,
    ):
        Shape.validate(shape)
        self.shape = shape
        self.requires_grad = requires_grad
        self.ancestors = Ancestors[dtype].untracked()
        self.grad_fn = None
        self.grad = UnsafePointer[__type_of(self)]()
        self.base = UnsafePointer[Tensor[dtype]]()
        self.data = data
        self.init_gradbox()

    fn __init__(out self, shape: Shape, requires_grad: Bool = False):
        Shape.validate(shape)
        self.shape = shape
        self.requires_grad = requires_grad
        self.ancestors = Ancestors[dtype].untracked()
        self.base = UnsafePointer[Tensor[dtype]]()
        self.grad_fn = None
        self.grad = UnsafePointer[__type_of(self)]()
        if shape.ndim == 0:  # Tensor with Shape ()
            self.data = UnsafePointer[Scalar[self.dtype]].alloc(1)
        else:
            self.data = UnsafePointer[Scalar[self.dtype]].alloc(
                self.shape.num_elements()
            )
        self.init_gradbox()

    @staticmethod
    fn trace_ancestry[
        dtype: DType, //
    ](
        tensor: Tensor[dtype],
        mut visited: Set[Int],
        mut traced: Ancestors[dtype],
    ):
        if tensor.int_addr() not in visited:
            visited.add(Int(tensor.address()))
            for ancestor in tensor.ancestors:
                Self.trace_ancestry(ancestor[], visited, traced)
            traced.append(tensor.address())

    @staticmethod
    fn walk_backward[
        dtype: DType, //
    ](
        tensor: Tensor[dtype],
        start_grad: Scalar[dtype] = 1.0,
        verbose: Bool = False,
    ) raises:
        if tensor.has_grad() == False:
            return
        visited = Set[Int]()
        traced = Ancestors[dtype]()
        Self.trace_ancestry(tensor, visited, traced)
        tensor.grad[].fill(start_grad)
        for each in traced.__reversed__():
            each[].invoke_grad_fn(verbose)

    _ = """fn backward(
        self,
        start_grad: Scalar[dtype] = 1.0,
        start: Bool = True,
        verbose: Bool = False,
    ) raises:
        if not self.requires_grad:
            print(
                "Tensor -> backward: calling backward on a tensor that does not"
                " require grad has no effect"
            )
            return
        if start:
            self.grad[].fill(start_grad)
        self.invoke_grad_fn(verbose)
        for ancestor in self.ancestors:
            ancestor[].backward(start=False)"""

    fn backward(self) raises:
        Tensor.walk_backward(self)

    fn grad_func(self) -> Optional[fn () escaping raises -> None]:
        return self.grad_fn

    @always_inline
    fn address(self) -> UnsafePointer[Self]:
        return UnsafePointer(to=self)

    @always_inline
    fn int_addr(self) -> Int:
        return Int(self.address())

    fn invoke_grad_fn(self, verbose: Bool = False) raises -> None:
        if self.grad_fn:
            if verbose:
                print("\nInvoking  grad_fn\n")
            self.grad_fn.value()()
        else:
            if verbose:
                print("\nNo grad_fn\n")
            pass

    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        if self.shape.ndim == 0 and len(indices) != 0:  # Tensor with Shape ()
            abort("Tensor -> __getitem__: Scalar tensor expects no indices")
        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__getitem__(indices): Invalid indices")
        return self.data.load[volatile=True](index)

    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        if self.shape.ndim == 0:  # Tensor with Shape ()
            abort(
                "Tensor -> __getitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __getitem__(IntList())"
            )

        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__getitem__(*indices): Invalid indices")
        return self.data.load[volatile=True](index)

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]):
        if self.shape.ndim == 0:  # Tensor with Shape ()
            abort(
                "Tensor -> __setitem__(*indices: Int): api not supported for"
                " scalar tensor. Use __setitem__(IntList())"
            )
        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__setitem__(*indices): Invalid indices")
        self.data.store[volatile=True](index, value)

    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        if self.shape.ndim == 0 and len(indices) != 0:  # Tensor with Shape ()
            abort("Tensor -> __setitem__: Scalar tensor expects no indices")
        index = self.shape.flatten_index(indices)
        if index == -1:
            abort("__setitem__(IntList): Invalid indices")
        self.data.store[volatile=True](index, value)

    fn item(self) -> Scalar[self.dtype]:
        if (
            self.shape != Shape.Unit and self.shape.ndim != 0
        ):  # Tensor with Shape ()
            abort(
                "Tensor.item(): Only valid for scalar or singleton tensors, got"
                " shape: "
                + self.shape.__str__()
            )
        return self[0] if self.shape == Shape.Unit else self[IntList.Empty]

    fn __moveinit__(out self, owned other: Self):
        self.shape = other.shape
        self.data = UnsafePointer[Scalar[other.dtype]].alloc(other.numels())
        memcpy(self.data, other.data, other.numels())
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self.base = other.base
        self.ancestors = other.ancestors
        self.grad_fn = other.grad_fn

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.data = UnsafePointer[Scalar[other.dtype]].alloc(other.numels())
        memcpy(self.data, other.data, other.numels())
        self.requires_grad = other.requires_grad
        self.grad = other.grad
        self.base = other.base
        self.ancestors = other.ancestors
        self.grad_fn = other.grad_fn
        self.init_gradbox()

    fn copy(self) -> Self:
        result = Tensor[dtype](self.shape, requires_grad=self.requires_grad)
        memcpy(result.data, self.data, self.numels())
        if result.requires_grad:
            memcpy(result.grad, self.grad, self.numels())
        return result

    fn init_gradbox(mut self):
        if self.requires_grad and self.grad.__as_bool__() == False:
            gradients = Tensor[self.dtype](self.shape)
            self.grad = UnsafePointer[__type_of(self)].alloc(1)
            self.grad.init_pointee_move(gradients^)
            self.zero_grad()

    fn print_grad(self):
        if self.requires_grad == False:
            print("Requires grad? No.")
        elif self.requires_grad and self.has_grad() == False:
            print("Gradbox not initialized")
        else:
            self.grad[].print()

    fn open_gradbox(
        mut self,
    ) raises -> ref [self.grad[]] Tensor[self.dtype]:
        if self.requires_grad == False or self.has_grad() == False:
            err_s = String(
                "Requires grad is: {0}, gradbox: uninitialized? {1}"
            ).format(self.requires_grad, self.has_grad() == False)
            raise Error(err_s)
        return self.grad[]

    # fn __del__(owned self):
    fn free(owned self):
        if self.has_grad():
            for i in range(self.numels()):
                (self.data + i).destroy_pointee()
                (self.grad[].data + i).destroy_pointee()
            self.grad.free()
            log_debug(
                "Tensor__del__ -> freed grad(and pointees) and self data"
                " pointees"
            )
        else:
            for i in range(self.numels()):
                (self.data + i).destroy_pointee()
            log_debug("Tensor__del__ -> freed self data pointees")
        log_debug("Tensor__del__ -> discarded ancestors")
        self.ancestors.free()
        self.shape.free()
        if self.data:
            self.data.free()
        log_debug("Tensor__del__ -> called free on data")
        _ = self^

    fn __len__(self) -> Int:
        return self.numels()

    fn numels(self) -> Int:
        return self.shape.num_elements()

    fn ndim(self) -> Int:
        return self.shape.ndim

    @always_inline
    fn broadcastable(self, to: Tensor[dtype]) -> Bool:
        return self.shape.broadcastable(to.shape)

    fn all_true(self: Tensor[DType.bool]) -> Bool:
        fn all_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == True

        return self.for_all(all_truthy)

    fn any_true(self: Tensor[DType.bool]) -> Bool:
        fn any_truthy(ambivalent: Scalar[DType.bool]) -> Bool:
            return ambivalent == True

        return self.any(any_truthy)

    fn for_all[
        simd_width: Int = simdwidthof[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        num_elems = self.numels()
        simd_blocks = num_elems // simd_width
        remaining = num_elems % simd_width

        for i in range(simd_blocks):
            vector = self.data.load[width=simd_width](i * simd_width)
            for j in range(simd_width):
                if not pred(vector[j]):
                    return False
        for k in range(remaining):
            if not pred(self.data.load[width=1](simd_blocks * simd_width + k)):
                return False
        return True

    fn any[
        simd_width: Int = simdwidthof[dtype]()
    ](self, pred: fn (Scalar[dtype]) -> Bool) -> Bool:
        num_elems = self.numels()
        simd_blocks = num_elems // simd_width
        remaining = num_elems % simd_width

        for i in range(simd_blocks):
            vector = self.data.load[width=simd_width](i * simd_width)
            for j in range(simd_width):
                if pred(vector[j]):
                    return True
        for k in range(remaining):
            if pred(self.data.load[width=1](simd_blocks * simd_width + k)):
                return True
        return False

    fn all_close[
        simd_width: Int = simdwidthof[dtype](),
        rtol: Scalar[dtype] = 1e-5,
        atol: Scalar[dtype] = 1e-8,
    ](self, other: Self) -> Bool:
        constrained[
            dtype.is_floating_point(),
            "all_close is for floating point data types only",
        ]()

        if self.shape != other.shape:
            print(
                "all_close expects same shape 2D tensors: ",
                self.shape,
                ", ",
                other.shape,
            )

        num_elems = self.numels()
        simd_blocks = num_elems // simd_width
        remaining = num_elems % simd_width

        for i in range(simd_blocks):
            vector1 = self.data.load[width=simd_width](i * simd_width)
            vector2 = other.data.load[width=simd_width](i * simd_width)
            diff = abs(vector1 - vector2)
            tolerance = atol + rtol * abs(vector2)
            all_checks_out = (diff < tolerance).reduce_and()
            if all_checks_out == False:
                return False
        for k in range(remaining):
            value1 = self.data.load[width=1](simd_blocks * simd_width + k)
            value2 = other.data.load[width=1](simd_blocks * simd_width + k)
            value_diff = abs(value1 - value2)
            value_tolerance = atol + rtol * abs(value2)
            checks_out = value_diff < value_tolerance
            if checks_out == False:
                return False

        return True

    fn fill(self, value: Scalar[dtype]):
        @parameter
        fn set_value[simd_width: Int](idx: Int):
            self.data.store[width=simd_width](idx, value)

        vectorize[set_value, simdwidthof[dtype]()](self.numels())

    fn __eq__(self, other: Tensor[self.dtype]) -> Tensor[DType.bool]:
        if self.shape != other.shape:
            abort(
                "Tensor __eq__ -> Dimension mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        result = Tensor[DType.bool](self.shape, False)

        @parameter
        fn compare_elems[simd_width: Int](idx: Int):
            result.data.store[width=simd_width, volatile=True](
                idx,
                self.data.load[width=simd_width](idx)
                == other.data.load[width=simd_width](idx),
            )

        vectorize[compare_elems, simdwidthof[DType.bool]()](result.numels())
        return result

    fn __iadd__(self, other: Self) raises:
        if self.shape != other.shape:
            raise Error(
                "iadd -> Dimension mismatch: ", self.shape, ", ", other.shape
            )

        @parameter
        fn add_elems[simd_width: Int](idx: Int):
            self.data.store[width=simd_width](
                idx,
                (
                    self.data.load[width=simd_width](idx)
                    + other.data.load[width=simd_width](idx)
                ),
            )

        vectorize[add_elems, simdwidthof[dtype]()](self.numels())

    fn exp(self) -> Tensor[dtype]:
        requires_grad = self.requires_grad
        result = Tensor[dtype](self.shape, requires_grad)

        @parameter
        fn exp_elems[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, exp(self.data.load[width=simd_width](idx))
            )

        vectorize[exp_elems, simdwidthof[dtype]()](result.numels())
        return result

    fn __neg__(self) -> Tensor[dtype]:
        requires_grad = self.requires_grad
        result = Tensor[dtype](self.shape, requires_grad)

        @parameter
        fn negate_elems[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx).__neg__()
            )

        vectorize[negate_elems, simdwidthof[dtype]()](result.numels())
        return result

    fn __ne__(self, other: Self) raises -> Tensor[DType.bool]:
        if self.shape != other.shape:
            raise Error(
                "__ne__ -> Dimension mismatch: ", self.shape, ", ", other.shape
            )
        result = self == other

        @parameter
        fn invert[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, ~result.data.load[width=simd_width](idx)
            )

        vectorize[invert, simdwidthof[DType.bool]()](result.numels())
        return result

    fn grad_required(self) -> Bool:
        return self.requires_grad

    fn has_grad(self) -> Bool:
        return self.grad.__as_bool__() == True

    fn grad_is_zero(self) -> Bool:
        if not self.requires_grad:
            abort(
                "Tensor -> grad_is_zero: checking grad on a tensor that does"
                " have grad"
            )

        fn all_zero(val: Scalar[dtype]) -> Bool:
            return val == Scalar[dtype](0)

        return self.has_grad() and self.grad[].for_all(all_zero)

    fn zero_grad(self):
        if self.grad_required() and self.has_grad():
            memset_zero(self.grad[].data, self.grad[].numels())

    fn add_ancestry(
        mut self,
        *tensors: Tensor[dtype],
    ):
        self.ancestors.add_ancestry(tensors)

    fn __rmul__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        return self.__mul__(scalar)

    fn __mul__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, MulScalar](
            self,
            scalar,
        )

        if self.requires_grad:

            fn grad_fn() raises -> None:
                out_grad_scaled = __tensor_op_scalar__[dtype, MulScalar](
                    out.address()[].grad[], scalar
                )
                self.address()[].grad[] = __tensor_op_tensor__[
                    dtype, AddTensor
                ](self.address()[].grad[], out_grad_scaled)

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self)

        return out

    fn __pow__(self, scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, Power](
            self,
            scalar,
        )

        if self.requires_grad:

            fn grad_fn() raises -> None:
                self_powed_one_less = __tensor_op_scalar__[dtype, Power](
                    self.address()[], scalar - 1
                )
                self_powed_one_less_scaled = __tensor_op_scalar__[
                    dtype, MulScalar
                ](self_powed_one_less, scalar)

                product = __tensor_op_tensor__[dtype, MulTensor](
                    out.address()[].grad[], self_powed_one_less_scaled
                )

                self.address()[].grad[] = __tensor_op_tensor__[
                    dtype, AddTensor
                ](self.address()[].grad[], product)

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self)

        return out

    fn broadcast_to(self, target_shape: Shape) -> Tensor[dtype]:
        if not self.shape.broadcastable(target_shape):
            abort(
                "Tensor -> broadcast_to: shape "
                + self.shape.__str__()
                + " not broadcastable to "
                + target_shape.__str__()
            )

        mask = self.shape.broadcast_mask(target_shape)
        out = Tensor[dtype](target_shape, requires_grad=self.requires_grad)

        for idx in target_shape:
            src_idx = self.shape.translate_index(idx, mask, target_shape)
            out[idx] = self[src_idx]

        return out

    @staticmethod
    fn grad_unreduced(
        tensor: Tensor[dtype], upstream_grad: Tensor[dtype]
    ) -> Tensor[dtype]:
        upstream_grad_shape = upstream_grad.shape
        tensor_view = tensor.view(upstream_grad_shape)
        result = Tensor[dtype](upstream_grad_shape)
        for indices in upstream_grad_shape:
            result[indices] = upstream_grad[indices] * tensor_view[indices]
        return result

    @always_inline
    fn broadcast_shape(self, other: Self) -> Shape:
        return Shape.broadcast_shape(self.shape, other.shape)

    @always_inline
    fn broadcast_mask(self, broadcast_shape: Shape) -> IntList:
        return self.shape.broadcast_mask(broadcast_shape)

    @always_inline
    fn translate_index(
        self, indices: IntList, mask: IntList, broadcast_shape: Shape
    ) -> IntList:
        return self.shape.translate_index(indices, mask, broadcast_shape)

    fn broadcast_op(
        self,
        other: Self,
        op: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    ) -> Tensor[dtype]:
        if self.shape.rank() == 0 or other.shape.rank() == 0:
            return self.broadcast_scalar_op(other, op)
        else:
            return self.broadcast_tensor_op(other, op)

    fn broadcast_scalar_op(
        self,
        other: Self,
        op: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    ) -> Tensor[dtype]:
        # Decide result shape
        result_shape = other.shape if self.shape.rank() == 0 else self.shape
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor[dtype](result_shape, requires_grad=requires_grad)

        for indices in result_shape:
            self_val = self.item() if self.shape.rank() == 0 else self[indices]
            other_val = (
                other.item() if other.shape.rank() == 0 else other[indices]
            )
            result[indices] = op(self_val, other_val)

        return result

    fn broadcast_tensor_op(
        self,
        other: Self,
        op: fn (Scalar[dtype], Scalar[dtype]) -> Scalar[dtype],
    ) -> Tensor[dtype]:
        result_shape = Shape.broadcast_shape(self.shape, other.shape)
        mask1 = self.broadcast_mask(result_shape)
        mask2 = other.broadcast_mask(result_shape)
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor[dtype](result_shape, requires_grad=requires_grad)

        for indices in result_shape:
            self_indices = self.translate_index(indices, mask1, result_shape)
            other_indices = other.translate_index(indices, mask2, result_shape)
            result[indices] = op(self[self_indices], other[other_indices])

        return result

    fn update_grad[op: Int](self, incoming: Tensor[dtype]):
        self.grad[] = __tensor_op_tensor__[dtype, op](self.grad[], incoming)

    fn broadcast_operation[
        element_wise_op: Int, tensor_op_first: Int, tensor_op_second: Int
    ](self, other: Self) -> Tensor[dtype]:
        result = self.broadcast_op(other, scalar_ops[dtype, element_wise_op])
        if self.requires_grad or other.requires_grad:
            self_ptr = self.address()
            that_ptr = other.address()

            fn grad_fn() raises -> None:
                this = self_ptr[]
                that = that_ptr[]

                if this.requires_grad:
                    upstream_grad = result.address()[].grad[]
                    grad_contrib = this.backward_grad_contrib(
                        that, upstream_grad, False
                    )

                    this.update_grad[tensor_op_first](grad_contrib)

                if that.requires_grad:
                    upstream_grad = result.address()[].grad[]
                    grad_contrib = that.backward_grad_contrib(
                        this, upstream_grad, False
                    )

                    that.update_grad[tensor_op_second](grad_contrib)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self, other)
        return result

    fn backward_grad_contrib(
        self,
        other: Tensor[dtype],
        upstream_grad: Tensor[dtype],
        do_multiply: Bool,
    ) -> Tensor[dtype]:
        var grad_contrib: Tensor[dtype]

        if upstream_grad.shape == Shape.Void:
            grad_contrib = Tensor[dtype].full(
                self.shape, upstream_grad.item(), requires_grad=False
            )
        else:
            grad_contrib = (
                upstream_grad * other if do_multiply else upstream_grad
            )
            if grad_contrib.shape != self.shape:
                axes = self.broadcast_mask(grad_contrib.shape).indices_of(1)
                grad_contrib = grad_contrib.sum(axes=axes, keepdims=True)
            if grad_contrib.shape != self.shape:
                grad_contrib = grad_contrib.reshape(self.shape)
            grad_contrib.requires_grad = False

        return grad_contrib

    fn broadcast_mul(
        self: Self,
        other: Self,
    ) -> Tensor[dtype]:
        result = self.broadcast_op(other, scalar_ops[dtype, Multiply])
        requires_grad = self.requires_grad or other.requires_grad
        if requires_grad:

            fn grad_fn() raises -> None:
                this = self.address()[]
                that = other.address()[]
                output = result.address()[]
                upstream_grad = output.grad[]
                if this.requires_grad:
                    grad_contrib = this.backward_grad_contrib(
                        that, upstream_grad, True
                    )
                    this.update_grad[AddTensor](grad_contrib)
                if that.requires_grad:
                    grad_contrib = that.backward_grad_contrib(
                        this, upstream_grad, True
                    )
                    that.update_grad[AddTensor](grad_contrib)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self, other)

        return result

    fn __add__(self, other: Self) -> Tensor[dtype]:
        if self.address() == other.address():
            return self.__mul__(2)
        if not self.broadcastable(other):
            abort(
                "__add__ -> Dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__()
            )

        if self.shape != other.shape:
            return self.broadcast_operation[Add, AddTensor, AddTensor](
                other,
            )

        var out = __tensor_op_tensor__[dtype, AddTensor](self, other)

        if self.requires_grad or other.requires_grad:

            fn grad_fn() raises -> None:
                out_grad = out.address()[].grad[]
                if self.address()[].requires_grad:
                    self.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](self.address()[].grad[], out_grad)
                if other.address()[].requires_grad:
                    other.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](other.address()[].grad[], out_grad)

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self, other)

        return out

    fn __radd__(self, scalar: Scalar[dtype]) raises -> Tensor[dtype]:
        return self.__add__(scalar)

    fn __add__(self, scalar: Scalar[dtype]) raises -> Tensor[dtype]:
        var out = __tensor_op_scalar__[dtype, AddScalar](
            self,
            scalar,
        )
        if self.requires_grad:

            fn grad_fn() raises -> None:
                self_grad = self.address()[].grad[]
                out_grad = out.address()[].grad[]
                self.address()[].grad[] = __tensor_op_tensor__[
                    dtype, AddTensor
                ](self_grad, out_grad)

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self)

        return out

    fn __iadd__(self, value: Scalar[dtype]):
        @parameter
        fn add_value[simd_width: Int](idx: Int):
            self.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx) + value
            )

        vectorize[add_value, simdwidthof[dtype]()](self.numels())

    fn to_dtype[NewType: DType](self) -> Tensor[NewType]:
        result = Tensor[NewType](self.shape, self.requires_grad)

        @parameter
        fn cast_values[simd_width: Int](idx: Int):
            result.data.store[width=simd_width](
                idx, self.data.load[width=simd_width](idx).cast[NewType]()
            )

        vectorize[cast_values, simdwidthof[NewType]()](result.numels())
        return result

    fn __sub__(self, scalar: Scalar[dtype]) raises -> Self:
        var out = __tensor_op_scalar__[dtype, SubtractScalar](
            self.address()[], scalar
        )

        if self.address()[].requires_grad:

            fn grad_fn() raises -> None:
                self_grad = self.address()[].grad[]
                out_grad = out.address()[].grad[]
                self.address()[].grad[] = __tensor_op_tensor__[
                    dtype, AddTensor
                ](self_grad, out_grad)

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self)
        return out

    # Element wise multiplication of two tensors
    fn __mul__(self, other: Self) -> Tensor[dtype]:
        if not self.broadcastable(other):
            abort(
                "__mul__(self * other) -> Dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__()
            )

        if self.shape != other.shape:
            return self.broadcast_mul(other)

        var out = __tensor_op_tensor__[dtype, MulTensor](
            self,
            other,
        )

        if self.requires_grad or other.requires_grad:

            fn grad_fn() raises -> None:
                out_grad = out.address()[].grad[]

                if self.address()[].requires_grad:
                    requires_grad_original = other.address()[].requires_grad
                    other.address()[].requires_grad = (
                        False  # Prevent requires_grad for grads
                    )
                    product = __tensor_op_tensor__[dtype, MulTensor](
                        out_grad, other.address()[]
                    )
                    other.address()[].requires_grad = requires_grad_original
                    self.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](self.address()[].grad[], product)

                if other.address()[].requires_grad:
                    requires_grad_original = self.address()[].requires_grad
                    self.address()[].requires_grad = False
                    product = __tensor_op_tensor__[dtype, MulTensor](
                        out_grad, self.address()[]
                    )
                    self.address()[].requires_grad = requires_grad_original
                    other.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](other.address()[].grad[], product)

            out.add_ancestry(self, other)
            out.grad_fn = Optional(grad_fn)

        return out

    fn __sub__(self, other: Self) -> Tensor[dtype]:
        if not self.broadcastable(other):
            abort(
                "__sub__ -> Dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__()
            )

        if self.shape != other.shape:
            return self.broadcast_operation[
                Subtract, AddTensor, SubtractTensor
            ](other)
        requires_grad = (
            self.address()[].requires_grad or other.address()[].requires_grad
        )

        out = __tensor_op_tensor__[dtype, SubtractTensor](
            self.address()[], other.address()[]
        )

        if requires_grad:

            fn grad_fn() raises -> None:
                out_grad = out.address()[].grad[]
                if self.address()[].requires_grad:
                    self.address()[].update_grad[AddTensor](out_grad)
                if other.address()[].requires_grad:
                    other.address()[].update_grad[SubtractTensor](out_grad)

            out.grad_fn = Optional(grad_fn)

            out.add_ancestry(self, other)

        return out

    fn __truediv__(self, factor: Scalar[dtype]) -> Tensor[dtype]:
        copy = self

        @parameter
        fn div_by_factor[simd_width: Int](idx: Int):
            copy.data.store[width=simd_width](
                idx, copy.data.load[width=simd_width](idx).__truediv__(factor)
            )

        vectorize[div_by_factor, simdwidthof[dtype]()](copy.numels())
        return copy

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self.data

    fn mse(self, target: Tensor[dtype]) -> Tensor[dtype]:
        return ((self - target) ** 2).mean()

    fn matmul_v1(self, other: Self) -> Tensor[dtype]:
        if self.shape[1] != other.shape[0]:
            abort("Tensor matmul_v1 - Dim mismatch")
        result = Tensor[dtype].zeros(self.shape[0], other.shape[1])
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                for k in range(self.shape[1]):
                    result[i, j] += self[i, k] * other[k, j]
        return result

    fn load[nelts: Int = 1](self, rows: Int, cols: Int) -> SIMD[dtype, nelts]:
        if not self.ndim() == 2:
            abort("Tensor - load is supported only for 2d tensor")
        result = self.data.load[width=nelts](rows * self.shape[1] + cols)
        return result

    fn store[
        nelts: Int = 1
    ](self, rows: Int, cols: Int, val: SIMD[dtype, nelts]):
        if not self.ndim() == 2:
            abort("Tensor - store is supported only for 2d tensor")
        self.data.store(rows * self.shape[1] + cols, val)

    fn matmul_v2(self, other: Self) -> Tensor[dtype]:
        if self.shape[1] != other.shape[0]:
            abort("Tensor matmul_v2 - Dim mismatch")
        requires_grad = self.requires_grad or other.requires_grad

        result = Tensor[dtype](
            self.shape[0], other.shape[1], requires_grad=requires_grad
        )

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(other.shape[1]):
                    result[i, k] += self[i, j] * other[j, k]
        if requires_grad:

            fn grad_fn() raises -> None:
                a = self.address()[]
                b = other.address()[]
                out = result.address()[]
                upstream = out.grad[]

                if a.requires_grad:
                    a_grad = upstream.matmul(b.T())
                    a.grad[] = a.grad[] + a_grad

                if b.requires_grad:
                    b_grad = a.T().matmul(upstream)
                    b.grad[] = b.grad[] + b_grad

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self, other)

        return result

    fn matmul_optim[
        simd_width: Int = simdwidthof[dtype](), nelts: Int = 1
    ](self, other: Self) -> Tensor[dtype]:
        rows, cols = self.shape[0], self.shape[1]
        other_rows, other_cols = other.shape[0], other.shape[1]

        if cols != other_rows:
            abort(
                "Tensor -> matmul_optim - Dim mismatch: "
                + self.shape.__str__()
                + ", "
                + other.shape.__str__()
            )
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor[dtype].zeros(
            rows, other_cols, requires_grad=requires_grad
        )
        for i in range(rows):
            for j in range(cols):

                @parameter
                fn dot[simd_width: Int](idx: Int):
                    result.store[nelts](
                        i,
                        idx,
                        result.load[nelts](i, idx)
                        + self[i, j] * other.load[nelts](j, idx),
                    )

                vectorize[dot, simd_width](other.shape[1])

        if requires_grad:

            fn grad_fn() raises -> None:
                self_ref = self.address()
                other_ref = other.address()
                result_ref = result.address()
                upstream_grad = result_ref[].grad[]

                if self_ref[].requires_grad:
                    transposed = other_ref[].T()
                    grad = upstream_grad.matmul_optim(transposed)
                    self_ref[].update_grad[AddTensor](grad)

                if other_ref[].requires_grad:
                    transposed = self_ref[].T()
                    grad = transposed.matmul_optim(upstream_grad)
                    other.address()[].update_grad[AddTensor](grad)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self, other)

        return result

    fn matmul(self: Tensor[dtype], other: Tensor[dtype]) -> Tensor[dtype]:
        if not self.shape.rank() == 2:
            abort("Only supports 2D matmul for now")
        if not other.shape.rank() == 2:
            abort("Other must be 2D")
        if not self.shape[1] == other.shape[0]:
            abort("Incompatible shapes")

        m, k = self.shape[0], self.shape[1]
        n = other.shape[1]

        requires_grad = self.requires_grad or other.requires_grad
        var result = Tensor[dtype](m, n, requires_grad=requires_grad)

        for i in range(m):
            for j in range(n):
                var summ = Scalar[dtype](0)
                for p in range(k):
                    summ += self[IntList(i, p)] * other[IntList(p, j)]
                result[IntList(i, j)] = summ

        if requires_grad:

            fn grad_fn() raises -> None:
                a = self.address()[]
                b = other.address()[]
                out = result.address()[]
                upstream = out.grad[]

                if a.requires_grad:
                    a_grad = upstream.matmul(b.T())
                    a.update_grad[AddTensor](a_grad)

                if b.requires_grad:
                    b_grad = a.T().matmul(upstream)
                    b.update_grad[AddTensor](b_grad)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self, other)

        return result

    fn T(self, tile_size: Int = 32) raises -> Tensor[dtype]:
        if self.shape.ndim != 2:
            abort("Tensor -> transpose allowed only for 2D tensors")
        rows, cols = (self.shape[0], self.shape[1])
        result = Tensor[dtype](
            self.shape.reverse(), requires_grad=self.requires_grad
        )

        for i in range(0, rows, tile_size):
            for j in range(0, cols, tile_size):
                for ii in range(i, min(i + tile_size, rows)):
                    for jj in range(j, min(j + tile_size, cols)):
                        result[jj, ii] = self[ii, jj]

        if self.requires_grad:

            fn grad_fn() raises:
                upstream_grad = result.address()[].grad[]
                self.address()[].update_grad[AddTensor](upstream_grad.T())

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self)

        return result

    @always_inline
    fn is_scalar(self) -> Bool:
        return self.numels() == 1 and self.shape == Shape.Void

    fn reshape(self) -> Tensor[dtype]:
        if self.numels() != 1:
            abort(
                "Only tensor with single element can be reshaped to scalar"
                " tensor"
            )
        return self.reshape(Shape.Void)

    fn reshape(self, *newdims: Int) -> Tensor[dtype]:
        if len(newdims) == 1 and newdims[0] == 0:
            print("reshape *newdims: ", len(newdims), newdims[0])
            return self.reshape()
        return self.reshape(Shape(newdims))

    fn reshape(self, new_shape: Shape) -> Tensor[dtype]:
        if self.numels() != new_shape.num_elements():
            # if self.shape.product() != new_shape.product():
            abort(
                "Tensor with "
                + String(self.numels())
                + " element(s) can't be converted to a tensor containing "
                + String(new_shape.num_elements())
                + " element(s)"
            )

        requires_grad = self.requires_grad
        result = Tensor[dtype](
            new_shape, self.data, requires_grad=requires_grad
        )

        if requires_grad:
            # Only allocate base if needed
            base = Tensor[dtype].zeros(self.shape)
            result.base = UnsafePointer[Tensor[dtype]].alloc(1)
            result.base.init_pointee_move(base^)

            fn grad_fn() raises -> None:
                upstream_grad = (
                    result.address()[].grad[].reshape(self.address()[].shape)
                )
                # Calculate new contribution (exclude already accumulated)
                new_contrib = __tensor_op_tensor__[dtype, SubtractTensor](
                    upstream_grad, result.address()[].base[]
                )
                # Update parent gradient
                self.address()[].grad[] = __tensor_op_tensor__[
                    dtype, AddTensor
                ](self.address()[].grad[], new_contrib)
                # Update accumulator
                # result.address()[].base[] = upstream_grad
                result.address()[].base.init_pointee_move(upstream_grad^)

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self)

        return result

    fn unsqueeze(self, dim: Int) -> View[__origin_of(self), self.dtype]:
        new_shape = self.shape.intlist().insert(dim, 1)
        # result = Tensor[dtype](Shape(new_shape), requires_grad=self.requires_grad)
        # result.data = self.data  # share same data
        return self.view(Shape(new_shape))
        # return result

    fn mean(
        self, axes: List[Int] = [], keepdims: Bool = False
    ) -> Tensor[dtype]:
        return self.mean(IntList.new(axes), keepdims)

    fn mean(self, axes: IntList, keepdims: Bool = False) -> Tensor[dtype]:
        sorted_axes = Self.validate_and_normalize_axes(self.shape, axes)
        # Compute total count of elements being reduced
        reduce_dims = self.shape.axes_spans.select(sorted_axes)
        var count = 1
        for span in reduce_dims:
            count *= span

        if count == 0:
            abort("Mean reduction over zero elements not allowed.")

        # Perform sum
        summed = self.sum(sorted_axes, keepdims)
        # Divide by count
        var result = summed / Scalar[dtype](count)

        # Gradient logic
        if self.requires_grad:

            fn grad_fn() raises -> None:
                upstream_grad = result.address()[].grad[]
                if upstream_grad.shape == Shape.Void:
                    scalar_grad = (
                        upstream_grad.item()
                        / self.address()[].shape.num_elements()
                    )
                    grad_contrib = Tensor[dtype].full(
                        self.address()[].shape, scalar_grad, requires_grad=False
                    )
                    self.address()[].update_grad[AddTensor](grad_contrib)
                    return

                var expanded = upstream_grad

                if not keepdims:
                    expanded = upstream_grad.reshape(
                        Shape(
                            upstream_grad.shape.intlist().insert(
                                sorted_axes,
                                IntList.with_capacity(len(sorted_axes), 1),
                            )
                        )
                    )

                # Broadcast and divide
                broadcasted = expanded.broadcast_to(self.address()[].shape)
                scaled = broadcasted / Scalar[dtype](count)
                self.address()[].grad[] = __tensor_op_tensor__[
                    dtype, AddTensor
                ](
                    self.address()[].grad[],
                    scaled,
                )

            result.grad_fn = Optional(grad_fn)
            result.add_ancestry(self)

        return result

    fn sum(self, axes: List[Int] = [], keepdims: Bool = False) -> Tensor[dtype]:
        return self.sum(IntList.new(axes), keepdims)

    fn sum(self, axes: IntList, keepdims: Bool = False) -> Tensor[dtype]:
        _axes = Self.validate_and_normalize_axes(self.shape, axes)
        rank = self.shape.rank()

        # Early scalar return - already correct
        if rank == 0:
            scalar_out = Tensor[dtype].zeros(
                Shape.Void, requires_grad=self.requires_grad
            )
            scalar_out[IntList.Empty] = self[IntList.Empty]
            if self.requires_grad:

                fn scalar_grad_fn() raises -> None:
                    self.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](self.address()[].grad[], scalar_out.address()[].grad[])

                scalar_out.grad_fn = Optional(scalar_grad_fn)
                scalar_out.add_ancestry(self)
            return scalar_out

        # FIX 1: Handle full reduction case explicitly
        var out_shape: Shape
        reducing_all = len(_axes) == rank
        if reducing_all and not keepdims:
            # Explicit scalar output for full reduction
            out_shape = Shape.Void
        else:
            spans = IntList.with_capacity(rank)
            for i in range(rank):
                if i in _axes:
                    if keepdims:
                        spans.append(1)
                    else:
                        continue
                else:
                    spans.append(self.shape[i])
            out_shape = Shape(spans)

        out = Tensor[dtype].zeros(out_shape, requires_grad=self.requires_grad)
        reduced_shape = Shape(self.shape.axes_spans.select(_axes))
        # FIX 3: Special handling for full reduction case
        if reducing_all and not keepdims:
            summ = Scalar[dtype](0)
            for idx in self.shape:
                summ += self[idx]
            out[IntList.Empty] = summ
        else:
            for out_idx in out_shape:
                summ = Scalar[dtype](0)
                for red_idx in reduced_shape:
                    if keepdims:
                        full_idx = out_idx.replace(_axes, red_idx)
                    else:
                        full_idx = out_idx.insert(_axes, red_idx)
                    summ += self[full_idx]
                out[out_idx] = summ

        if self.requires_grad:

            fn grad_fn() raises -> None:
                out_grad = out.address()[].grad[]
                self_tensor = self.address()[]
                input_shape = self_tensor.shape

                var expanded: Tensor[dtype]
                if out_grad.shape == Shape.Void or out_grad.shape.rank() == 0:
                    # if not keepdims and out_grad.shape == Shape.Void:
                    gradients = Tensor[dtype].zeros(self.address()[].shape)
                    gradients.fill(out_grad.item())

                    self.address()[].grad[] = __tensor_op_tensor__[
                        dtype, AddTensor
                    ](self.address()[].grad[], gradients)
                    # print("Upstream is scalar, shape:", out_grad.shape)
                    return

                if not keepdims:
                    if out_grad.shape == Shape.Void:
                        # Scalar → reduce over all axes
                        expanded = Tensor[dtype].full(
                            input_shape, out_grad.item(), requires_grad=False
                        )
                    else:
                        # Expand dims on the reduced axes
                        inserted_shape = Shape(
                            out_grad.shape.intlist().insert(
                                _axes,
                                IntList.with_capacity(len(_axes), 1),
                            )
                        )
                        expanded = out_grad.reshape(inserted_shape)
                else:
                    # keepdims=True → shapes align in rank, can broadcast directly
                    expanded = out_grad

                self_tensor.grad[] = __tensor_op_tensor__[dtype, AddTensor](
                    self_tensor.grad[],
                    expanded.broadcast_to(input_shape),
                )

            out.grad_fn = Optional(grad_fn)
            out.add_ancestry(self)

        return out

    @staticmethod
    fn validate_and_normalize_axes(shape: Shape, axes: IntList) -> IntList:
        """Ensure axes are unique, sorted, and within bounds."""
        rank = shape.rank()

        if rank == 0:
            if len(axes) == 1 and axes[0] == -1:
                return (
                    IntList()
                )  # Interpret `[-1]` as "reduce all axes" for scalars
            if len(axes) > 0:
                abort(
                    "Tensor -> validate_and_normalize_axes - cannot reduce over"
                    " axes "
                    + axes.__str__()
                    + " for scalar tensor with shape: "
                    + shape.__str__()
                )
            return IntList()  # Scalar sum over [] is valid

        if len(axes) == 0:
            return IntList.range_list(rank)
        normalized = IntList.with_capacity(len(axes))
        for _axis in axes:
            axis = _axis
            if axis < 0:
                axis += rank
            if axis < 0 or axis >= rank:
                abort(
                    "Tensor -> validate_and_normalize_axes - invalid axis: "
                    + String(_axis)
                    + " for tensor shape: "
                    + shape.__str__()
                )
            normalized.append(axis)
        # Sort and deduplicate
        normalized.sort_and_deduplicate()
        return normalized

    _ = """fn sum(self, axis: Int = -1, keepdim: Bool = False) -> Tensor[dtype]:
        _axis = axis
        if _axis != -1:
            if _axis < 0 or _axis >= self.ndim():
                abort("Invalid axis for tensor sum: " + String(_axis))
        else:
            _axis = self.ndim() - 1

        if self.ndim() == 1:
            result = Tensor[dtype].zeros(1, requires_grad=self.requires_grad)

            @parameter
            fn sum_elems[simd_width: Int](idx: Int):
                result[0] += self.data.load[width=simd_width](idx).reduce_add()

            vectorize[sum_elems, simdwidthof[dtype]()](self.numels())
            return result

        #elif self.ndim() == 2:
            #out = sum_across_rows(self) if _axis == 1 else sum_across_cols(self)
            #return out

        else:
            shape = self.shape
            out_shape = shape.replace(_axis, 1) if keepdim else shape.drop_axis(
                _axis
            )
            out = Tensor[dtype].zeros(
                out_shape, requires_grad=self.requires_grad
            )
            for indices in out_shape:  # all indices of output tensor
                sum_val = Scalar[dtype](0)
                for i in range(self.shape[_axis]):
                    full_idx = indices.replace(
                        _axis, i
                    ) if keepdim else indices.insert(_axis, i)
                    sum_val += self[full_idx]
                out[indices] = sum_val

            return out"""

    fn __str__(self) -> String:
        dims = len(self.shape)
        s = String("[")
        if dims == 1:
            s += "1D Tensor"
        elif dims == 2:
            s += "2D Tensor"
        elif dims == 3:
            s += "3D Tensor"
        elif dims == 4:
            s += "4D Tensor"
        elif dims == 5:
            s += "5D Tensor"
        else:
            s += "Tensor"
        s += self.shape.__str__()
        s += ", Type: " + self.dtype.__str__()
        s += ", requires_grad: " + String(self.requires_grad)
        s += "]"
        return s

    @staticmethod
    fn full(
        shape: Shape, value: Scalar[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        tensor = Tensor[dtype](shape, requires_grad=requires_grad)
        tensor.fill(value)
        return tensor

    @staticmethod
    fn rand(
        *axes_spans: Int,
        min: Scalar[dtype] = 0,
        max: Scalar[dtype] = 1,
        init_seed: Optional[Int] = None,
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        if init_seed:
            seed(init_seed.value())
        else:
            seed()
        shape = Shape(axes_spans)
        tensor = Tensor[dtype](shape, requires_grad)
        # randn(tensor.data, tensor.numels())
        for i in range(tensor.numels()):  # vectorize?
            tensor.data.store[volatile=True](
                i,
                random_float64(
                    min.cast[DType.float64](), max.cast[DType.float64]()
                ).cast[dtype](),
            )
        return tensor

    @staticmethod
    fn arange(
        *args: Scalar[dtype],
        requires_grad: Bool = False,
    ) -> Tensor[dtype]:
        start: Scalar[dtype] = 0
        end: Scalar[dtype] = max_finite[dtype]()
        step: Scalar[dtype] = 1

        n = len(args)
        if n == 1:
            end = args[0]
        elif n == 2:
            start = args[0]
            end = args[1]
        elif n == 3:
            start = args[0]
            end = args[1]
            step = args[2]
        else:
            abort(
                "Tensor.arange expects 1 to 3 arguments:\n"
                + "- arange(end)\n"
                + "- arange(start, end)\n"
                + "- arange(start, end, step)\n"
                + "Got: "
                + String(len(args))
                + " argument(s)"
            )

        if step == 0:
            abort("step can not be zero")
        if (step > 0 and start >= end) or (step < 0 and start <= end):
            abort("Invalid range for the given step")
        delta = end - start
        size = floor(delta / step)
        if size <= 0:
            abort("Error: computed arange size is zero")
        count = size.__int__()
        tensor = Tensor[dtype](count, requires_grad=requires_grad)

        @parameter
        fn fill(i: Int) -> Scalar[dtype]:
            return (i * step + start) % end

        @parameter
        fn mapper[simd_width: Int](idx: Int):
            first_entry = fill(idx).cast[dtype]()
            data = SIMD[dtype, simd_width](first_entry)
            for i in range(1, simd_width):
                data[i] = fill(idx + i).cast[dtype]()
            tensor.data.store[width=simd_width](idx, data)

        vectorize[mapper, simdwidthof[dtype]()](tensor.numels())

        return tensor

    @staticmethod
    fn zeros(*axes_spans: Int, requires_grad: Bool = False) -> Tensor[dtype]:
        shape = Shape(axes_spans)
        return Self.zeros(shape, requires_grad)

    @staticmethod
    fn zeros_like(
        tensor: Tensor[dtype], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        out = Tensor[dtype](tensor.shape, requires_grad)
        memset_zero(out.data, out.numels())
        return out

    @staticmethod
    fn zeros(shape: Shape, requires_grad: Bool = False) -> Tensor[dtype]:
        out = Tensor[dtype](shape, requires_grad)
        memset_zero(out.data, out.numels())
        return out

    alias Row = List[Scalar[dtype]]

    @staticmethod
    fn d1(row: Self.Row, requires_grad: Bool = False) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "d1")
        shape = Shape(IntList(len(row)))
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, row.data, len(row))
        return tensor

    @staticmethod
    fn d2(rows: List[Self.Row], requires_grad: Bool = False) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "d2")
        dims = IntList(len(rows), len(rows[0]))
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for row in rows:
            if len(row) != dims[1]:
                abort("Tensor -> d2 -> not all rows equal in length")
            flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    alias Rows = List[Self.Row]

    @staticmethod
    fn d3(
        blocks: List[Self.Rows], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "d3")
        dims = IntList(len(blocks), len(blocks[0]), len(blocks[0][0]))
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for block in blocks:
            if len(block) != dims[1]:
                abort("Tensor -> d3 -> not all blocks equal in length")
            for row in block:
                if len(row) != dims[2]:
                    abort("Tensor -> d3 -> not all rows equal in length")

                flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    alias Block = List[Self.Rows]

    @staticmethod
    fn d4(
        blockgrid: List[Self.Block], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "d4")
        dims = IntList(
            len(blockgrid),
            len(blockgrid[0]),
            len(blockgrid[0][0]),
            len(blockgrid[0][0][0]),
        )
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for block in blockgrid:
            if len(block) != dims[1]:
                abort(
                    "Tensor -> d4 -> not all blocks are of equal length in the"
                    " blockgrid"
                )
            for matrix in block:
                if len(matrix) != dims[2]:
                    abort(
                        "Tensor -> d4 -> not all matrices are of equal length"
                        " in block"
                    )
                for row in matrix:
                    if len(row) != dims[3]:
                        abort(
                            "Tensor -> d4 not all rows are of equal length in"
                            " matrix"
                        )
                    flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    alias Blocks = List[Self.Block]

    @staticmethod
    fn d5(
        blockhive: List[Self.Blocks], requires_grad: Bool = False
    ) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "d5")
        dims = IntList(
            len(blockhive),
            len(blockhive[0]),
            len(blockhive[0][0]),
            len(blockhive[0][0][0]),
            len(blockhive[0][0][0][0]),
        )
        flattened = List[Scalar[dtype]](capacity=dims.product())
        for blocks in blockhive:
            if len(blocks) != dims[1]:
                abort(
                    "Tensor -> d5 -> not all blocks are of equal length in the"
                    " input"
                )
            for block in blocks:
                if len(block) != dims[2]:
                    abort("Tensor -> d5 -> unequal block length")
                for matrix in block:
                    if len(matrix) != dims[3]:
                        abort(
                            "Tensor -> d5 not all matrices are of equal length"
                            " in block"
                        )
                    for row in matrix:
                        if len(row) != dims[4]:
                            abort(
                                "Tensor -> d5 not all rows are of equal length"
                                " in matrix"
                            )
                        flattened.extend(row)
        shape = Shape(dims)
        tensor = Tensor[dtype](shape, requires_grad)
        memcpy(tensor.data, flattened.data, tensor.numels())
        return tensor

    @staticmethod
    fn of(*elems: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "of(*elems)")
        # shape = Shape.of(len(elems))
        shape = Shape(IntList(len(elems)))
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(len(elems)):
            tensor[i] = elems[i]
        return tensor

    @staticmethod
    fn of(
        # elems: List[Scalar[Self.dtype]], requires_grad: Bool = False
        elems: Self.Row,
        requires_grad: Bool = False,
    ) -> Tensor[Self.dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "of(elems)")
        shape = Shape.of(len(elems))
        tensor = Tensor[Self.dtype](shape, requires_grad)
        for i in range(len(elems)):
            tensor[i] = elems[i]
        return tensor

    @staticmethod
    fn of[
        row_size: Int
    ](*elems: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        Self.validate_dtype_consistency(dtype, requires_grad, "of[row_size]")

        if not (row_size >= 1 and row_size <= len(elems)):
            abort(
                (
                    "Tensor -> of[row_size] -> invalid row size or not enough"
                    " elements"
                ),
            )
        num_rows = len(elems) // row_size
        axes_spans = piped(num_rows, row_size)
        shape = Shape(axes_spans)
        tensor = Tensor[dtype](shape, requires_grad)
        for i in range(num_rows):
            for j in range(row_size):
                tensor[i, j] = elems[i * row_size + j]
        return tensor

    @staticmethod
    fn scalar(val: Scalar[dtype], requires_grad: Bool = False) -> Tensor[dtype]:
        result = Tensor[dtype](Shape.Void, requires_grad=requires_grad)
        result[IntList.Empty] = val
        return result

    @staticmethod
    fn ones(*axes_spans: Int, requires_grad: Bool = False) -> Tensor[dtype]:
        return Self.ones(Shape(axes_spans), requires_grad)

    @staticmethod
    fn ones(shape: Shape, requires_grad: Bool = False) -> Tensor[dtype]:
        tensor = Tensor[dtype](shape, requires_grad=requires_grad)
        var value: SIMD[dtype, 1]

        @parameter
        if dtype.is_floating_point():
            value = SIMD[dtype, 1](1.0)
        else:
            value = SIMD[dtype, 1](1)
        for i in range(tensor.numels()):
            tensor.data.store(i, value)
        return tensor

    @staticmethod
    fn validate_dtype_consistency(
        dtype: DType, requires_grad: Bool, label: String
    ):
        if requires_grad:
            if not (dtype.is_floating_point()):
                abort(
                    "Tensor → "
                    + label
                    + " → requires_grad=True is only supported for floating"
                    " point types. "
                )

    fn print_tensor_recursive(
        self,
        mut indices: IntList,
        level: Int,
        num_first: Int = 10,
        num_last: Int = 10,
    ) raises:
        try:
            if self.ndim() == 0:  # Tensor with Shape ()
                print(self[IntList.Empty])
                return
            current_dim = len(indices)
            indent = " " * (level * 2)
            # Defensive check
            if current_dim >= self.ndim():
                # if current_dim > self.ndim():
                print(
                    "ERROR: current_dim (",
                    current_dim,
                    ") >= ndim (",
                    self.ndim(),
                    ")",
                )
                return

            size = self.shape[current_dim]

            # Size sanity check
            if size < 0 or size > 1_000_000:
                print(
                    "ERROR: suspicious size: ",
                    size,
                    "at dim ",
                    current_dim,
                    self.shape.__str__(),
                )
                return

            # Base case: last dimension (print actual elements)
            if current_dim == self.ndim() - 1:
                print(indent + "[", end="")

                for i in range(size):
                    if i < num_first:
                        indices.append(i)
                        print(
                            self[indices],
                            end=", " if (
                                i != num_first - 1
                                or size > num_first + num_last
                            ) else "",
                        )
                        _ = indices.pop()
                    elif i == num_first:
                        if size > num_first + num_last:
                            print("..., ", end="")
                    elif i >= size - num_last:
                        indices.append(i)
                        print(self[indices], end=", " if i != size - 1 else "")
                        _ = indices.pop()
                    else:
                        # Handles middle region not explicitly caught
                        continue

                print("]", end="")

            else:
                print(indent + "[")
                for i in range(size):
                    if i < num_first:
                        indices.append(i)
                        self.print_tensor_recursive(indices, level + 1)
                        _ = indices.pop()
                        if i != num_first - 1 or size > num_first + num_last:
                            print(",")
                    elif i == num_first:
                        if size > num_first + num_last:
                            print(indent + "  ...,")
                    elif i >= size - num_last:
                        indices.append(i)
                        self.print_tensor_recursive(indices, level + 1)
                        _ = indices.pop()
                        if i != size - 1:
                            print(",")
                    else:
                        # This path was previously missing, which caused silent looping!
                        continue

                print(indent + "]", end="")
                # print("\n")

        except e:
            print("ERROR during tensor printing: ", e)

    fn print(self, num_first: Int = 10, num_last: Int = 10):
        print(self.__str__())
        empty = IntList()
        try:
            self.print_tensor_recursive(
                empty, 1, num_first=num_first, num_last=num_last
            )
        except e:
            print(e)

    @staticmethod
    fn free_all[dtype: DType, //](*tensors: Tensor[dtype]):
        for each in tensors:
            each.free()
            _ = each

    fn view(
        ref self, target_shape: Shape
    ) -> View[__origin_of(self), self.dtype]:
        concrete = True if target_shape == self.shape else False
        mask = self.shape.broadcast_mask(target_shape)
        return View(Pointer(to=self), concrete, mask, target_shape)

    fn view2(self) -> View2[self.dtype]:
        return View2(UnsafePointer(to=self))


@fieldwise_init
struct View2[
    dtype: DType = DType.float32,
]:
    var target: UnsafePointer[Tensor[dtype]]


@fieldwise_init
struct View[
    mutability: Bool, //,
    origin: Origin[mutability],
    dtype: DType = DType.float32,
]:
    var target: Pointer[Tensor[dtype], origin]
    var concrete: Bool
    var mask: IntList
    var target_shape: Shape

    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        if self.concrete:
            return self.target[][indices]
        else:
            target_idx = self.target[].shape.translate_index(
                indices, self.mask, self.target_shape
            )
            return self.target[][target_idx]

    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        if self.concrete:
            self.target[].__setitem__(indices, value)
        else:
            target_idx = self.target[].shape.translate_index(
                indices, self.mask, self.target_shape
            )
            self.target[].__setitem__(target_idx, value)


from testing import assert_true


fn test_add_2_tensors() raises:
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
    parent1 = out_tensor.ancestors.get(0)[]
    parent2 = out_tensor.ancestors.get(1)[]
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

    tensor = Tensor.rand(256, 256, requires_grad=True)
    out_tensor = 2 * tensor
    assert_true(
        len(out_tensor.ancestors.get(0)[]) == 65536,
        "Output tensor ancestors length validation failed",
    )
    out_tensor.invoke_grad_fn()
    print("The following is out tensor gradient")
    out_tensor.open_gradbox().print()
    Tensor.free_all(tensor, out_tensor)


fn test_mul_by_factor() raises:
    print("test_mul_by_factor")
    tensor = Tensor.rand(128, 256, requires_grad=True)
    out_tensor = tensor * 100
    assert_true(
        len(out_tensor.ancestors.get(0)[]) == 32768,
        "Output tensor ancestors length validation failed",
    )
    out_tensor.invoke_grad_fn()
    print("The following is out tensor gradient")
    out_tensor.open_gradbox().print()
    Tensor.free_all(tensor, out_tensor)


fn test_add_value() raises:
    print("test_add_value")

    tensor = Tensor.rand(1024, 64, requires_grad=True)
    out_tensor = 2 * tensor
    assert_true(
        len(out_tensor.ancestors.get(0)[]) == 65536,
        "Output tensor ancestors length validation failed",
    )
    out_tensor.invoke_grad_fn()
    print("The following is out tensor gradient")
    out_tensor.open_gradbox().print()
    Tensor.free_all(tensor, out_tensor)


fn test_arange() raises:
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
    X = Tensor.rand(3, 4, requires_grad=True)
    W = Tensor.rand(4, 5, requires_grad=True)
    B = Tensor.rand(1, requires_grad=True)
    P = X.matmul_optim(W) + B
    print("P.shape: ", P.shape, P.requires_grad)
    Tensor.walk_backward(P)
    W.grad[].print()


fn test_transpose_matmul() raises:
    A = Tensor.rand(3, 3, requires_grad=True)
    A.grad[].print()
    A_T = A.T()
    Tensor.walk_backward(A_T)
    A.grad[].print()


fn test_random() raises:
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
    tensor = Tensor.of(42)
    assert_true(tensor.item() == 42)


fn test_view() raises:
    tensor = Tensor.rand(2, 4, 5)
    view = tensor.view(Shape.Void)
    assert_true(
        tensor.shape == view.target[].shape,
        "Tensor and view shape equality asserttion failed",
    )


fn test_tensor_of_list() raises:
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
    # Tensor.walk_backward(result)
    result.backward()
    assert_true(tensor.grad[].item() == 3.0)
    tensor2 = tensor.reshape(1)
    tensor.print_grad()
    result = tensor2 * 42
    tensor.print_grad()
    tensor2.print_grad()
    # Tensor.walk_backward(result)
    result.backward()
    tensor.print_grad()
    tensor2.print_grad()
    # tensor3 = tensor2.reshape(1,1,1,1,1)
    tensor3 = tensor2.reshape(1, 1, 1, 1, 1)
    result = tensor3 * 12
    # Tensor.walk_backward(result)
    result.backward()

    tensor3.print_grad()
    tensor2.print_grad()
    tensor.print_grad()
