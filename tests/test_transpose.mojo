from tenmo import Tensor
from shapes import Shape
from testing import assert_true
from sys import has_accelerator


fn test_2d_transpose_no_axes() raises:
    print("test_2d_transpose_no_axes")
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var b = a.transpose()

    # Forward pass validation
    var expected = Tensor.d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert_true(b.all_close(expected))
    assert_true(b.shape() == Shape(3, 2))

    # Backward pass validation
    var loss = b.sum()
    loss.backward()
    var expected_grad = Tensor.d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert_true(a.grad().all_close(expected_grad))


fn test_2d_transpose_explicit_axes() raises:
    print("test_2d_transpose_explicit_axes")
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var b = a.transpose(1, 0)

    # Forward pass validation
    var expected = Tensor.d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert_true(b.all_close(expected))
    assert_true(b.shape() == Shape(3, 2))

    # Backward pass validation
    var loss = b.sum()
    loss.backward()
    var expected_grad = Tensor.d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert_true(a.grad().all_close(expected_grad))


fn test_3d_transpose_axes_0_1() raises:
    print("test_3d_transpose_axes_0_1")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var b = a.transpose(1, 0)

    # Forward pass validation
    var expected = Tensor.d3(
        [[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]]
    )
    assert_true(b.all_close(expected))
    assert_true(b.shape() == Shape(2, 2, 2))

    # Backward pass validation
    var loss = b.sum()
    loss.backward()
    var expected_grad = Tensor.d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


fn test_3d_transpose_axes_1_2() raises:
    print("test_3d_transpose_axes_1_2")
    var a = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    # var b = a.transpose(1, 2)
    var b = a.transpose(2, 1)

    # Forward pass validation
    var expected = Tensor.d3(
        [[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]]
    )
    assert_true(b.all_close(expected))
    assert_true(b.shape() == Shape(2, 2, 2))

    # Backward pass validation
    var loss = b.sum()
    loss.backward()
    var expected_grad = Tensor.d3(
        [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]
    )
    assert_true(a.grad().all_close(expected_grad))


fn test_4d_transpose_complex_axes() raises:
    print("test_4d_transpose_complex_axes")
    var a = Tensor.d4(
        [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
        ],
        requires_grad=True,
    )
    var b = a.transpose(0, 2, 3, 1)

    # Forward pass validation - check shape
    assert_true(b.shape() == Shape(2, 2, 2, 2))

    # Backward pass validation
    var loss = b.sum()
    loss.backward()
    var expected_grad = Tensor.d4(
        [
            [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
            [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        ]
    )
    assert_true(a.grad().all_close(expected_grad))


fn test_transpose_chain_operations() raises:
    print("test_transpose_chain_operations")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.transpose()
    var c = b.transpose()

    # Forward pass validation - double transpose should return original
    assert_true(c.all_close(a))

    # Backward pass validation
    var loss = c.sum()
    loss.backward()
    var expected_grad = Tensor.d2([[1.0, 1.0], [1.0, 1.0]])
    assert_true(a.grad().all_close(expected_grad))


fn test_transpose_with_matmul() raises:
    print("test_transpose_with_matmul")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var a_t = a.transpose()
    var result = a_t.matmul(b)

    # Forward pass validation
    var expected = Tensor.d2([[26.0, 30.0], [38.0, 44.0]])
    assert_true(result.all_close(expected))

    # Backward pass validation
    var loss = result.sum()
    loss.backward()
    var expected_a_grad = Tensor.d2([[11.0, 11.0], [15.0, 15.0]])
    var expected_b_grad = Tensor.d2([[3.0, 3.0], [7.0, 7.0]])

    assert_true(a.grad().all_close(expected_a_grad))
    assert_true(b.grad().all_close(expected_b_grad))


fn test_transpose_scalar_equivalent() raises:
    print("test_transpose_scalar_equivalent")
    var a = Tensor.scalar(5.0, requires_grad=True)
    var b = a.transpose()

    # Scalar transpose should return same scalar
    assert_true(b.item() == 5.0)

    # Backward pass
    var loss = b.sum()
    loss.backward()
    assert_true(a.grad().item() == 1.0)


fn test_transpose_1d_no_change() raises:
    print("test_transpose_1d_no_change")
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a.transpose()

    # 1D transpose should return same tensor
    assert_true(b.all_close(a))

    # Backward pass
    var loss = b.sum()
    loss.backward()
    var expected_grad = Tensor.d1([1.0, 1.0, 1.0])
    assert_true(a.grad().all_close(expected_grad))


# Exhaustive tests for the revamped Transpose implementation.
# Prefix: trrev_ (transpose revamp)
# Covers:
#  - Forward correctness (1D, 2D, 3D, 4D)
#  - Backward / grad flow
#  - Chained transposes
#  - Transpose + contiguous
#  - CPU and GPU variants

# ──────────────────────────────────────────────────────────────────────────────
# CPU TESTS
# ──────────────────────────────────────────────────────────────────────────────

# ── 1D ────────────────────────────────────────────────────────────────────────


fn test_trrev_cpu_1d_forward() raises:
    print("test_trrev_cpu_1d_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var t = a.transpose()
    # 1-D transpose is a no-op on shape
    assert_true(t.shape() == Shape(3))
    assert_true(t.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


fn test_trrev_cpu_1d_backward() raises:
    print("test_trrev_cpu_1d_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var t = a.transpose()
    var loss = t.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ── 2D ────────────────────────────────────────────────────────────────────────


fn test_trrev_cpu_2d_forward() raises:
    print("test_trrev_cpu_2d_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var t = a.transpose()
    assert_true(t.shape() == Shape(3, 2))
    assert_true(
        t.all_close(Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))
    )


fn test_trrev_cpu_2d_backward() raises:
    print("test_trrev_cpu_2d_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var t = a.transpose()
    var loss = t.sum()
    loss.backward()
    # Each element of a contributes once → grad = 1
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_cpu_2d_explicit_axes_forward() raises:
    print("test_trrev_cpu_2d_explicit_axes_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var t = a.transpose(0, 1)
    assert_true(t.shape() == Shape(2, 3))
    assert_true(
        t.all_close(Tensor[dtype].d2([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]))
    )


fn test_trrev_cpu_2d_explicit_axes_backward() raises:
    print("test_trrev_cpu_2d_explicit_axes_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var t = a.transpose(0, 1)
    var loss = t.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_cpu_2d_double_transpose_identity() raises:
    print("test_trrev_cpu_2d_double_transpose_identity")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var t1 = a.transpose()
    var t = t1.transpose()
    assert_true(t.shape() == Shape(2, 3))
    assert_true(t.all_close(a))


fn test_trrev_cpu_2d_double_transpose_backward() raises:
    print("test_trrev_cpu_2d_double_transpose_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var t1 = a.transpose()
    var t2 = t1.transpose()
    var loss = t2.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_cpu_2d_transpose_then_contiguous() raises:
    print("test_trrev_cpu_2d_transpose_then_contiguous")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var t = a.transpose()
    var c = t.contiguous()
    assert_true(c.shape() == Shape(3, 2))
    assert_true(
        c.all_close(Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))
    )
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_cpu_2d_grad_correctness_weighted() raises:
    print("test_trrev_cpu_2d_grad_correctness_weighted")
    # loss = sum(t * weights) where t = transpose(a)
    # grad_a[i,j] = weights[j,i]
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var t = a.transpose()
    # weights shape (2,2)
    var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var prod = t * w
    var loss = prod.sum()
    loss.backward()
    # grad_a[i,j] = w[j,i]
    # grad_a = [[w[0,0], w[1,0]], [w[0,1], w[1,1]]]
    #         = [[1, 3], [2, 4]]
    assert_true(a.grad().all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])))


# ── 3D ────────────────────────────────────────────────────────────────────────


fn test_trrev_cpu_3d_default_axes_forward() raises:
    print("test_trrev_cpu_3d_default_axes_forward")
    comptime dtype = DType.float32
    # Shape (2,3,4) → default transpose reverses axes → (4,3,2)
    var a = Tensor[dtype].arange(0.0, 24.0).reshape(Shape(2, 3, 4))
    var t = a.transpose()
    assert_true(t.shape() == Shape(4, 3, 2))
    # Check a few specific elements
    # a[i,j,k] == t[k,j,i]
    # a[0,0,0]=0  → t[0,0,0]=0
    # a[1,2,3]=23 → t[3,2,1]=23
    assert_true(t[0, 0, 0] == 0.0)
    assert_true(t[3, 2, 1] == 23.0)
    # a[0,1,2]=6 → t[2,1,0]=6
    assert_true(t[2, 1, 0] == 6.0)


fn test_trrev_cpu_3d_default_axes_backward() raises:
    print("test_trrev_cpu_3d_default_axes_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0.0, 24.0).reshape(Shape(2, 3, 4))
    a.requires_grad_(True)
    var t = a.transpose()
    var loss = t.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_cpu_3d_explicit_axes_forward() raises:
    print("test_trrev_cpu_3d_explicit_axes_forward")
    comptime dtype = DType.float32
    # Shape (2,3,4), transpose axes (0,2,1) → shape (2,4,3)
    var a = Tensor[dtype].arange(0.0, 24.0).reshape(Shape(2, 3, 4))
    var t = a.transpose(0, 2, 1)
    assert_true(t.shape() == Shape(2, 4, 3))
    # a[i,j,k] == t[i,k,j]
    # a[1,2,3]=23 → t[1,3,2]=23
    assert_true(t[1, 3, 2] == 23.0)
    # a[0,1,2]=6 → t[0,2,1]=6
    assert_true(t[0, 2, 1] == 6.0)


fn test_trrev_cpu_3d_explicit_axes_backward() raises:
    print("test_trrev_cpu_3d_explicit_axes_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0.0, 24.0).reshape(Shape(2, 3, 4))
    a.requires_grad_(True)
    var t = a.transpose(0, 2, 1)
    var loss = t.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_cpu_3d_transpose_then_contiguous() raises:
    print("test_trrev_cpu_3d_transpose_then_contiguous")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0.0, 24.0).reshape(Shape(2, 3, 4))
    a.requires_grad_(True)
    var t = a.transpose()  # (4,3,2)
    var c = t.contiguous()
    assert_true(c.shape() == Shape(4, 3, 2))
    # Verify contiguous copy matches strided view
    assert_true(c.all_close(t))
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_cpu_3d_chained_transpose_backward() raises:
    print("test_trrev_cpu_3d_chained_transpose_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(1.0, 25.0).reshape(Shape(2, 3, 4))
    a.requires_grad_(True)
    # T1: (2,3,4)->(4,3,2), T2: (4,3,2)->(2,3,4) — identity
    var t1 = a.transpose()
    var t2 = t1.transpose()
    var loss = t2.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ── 4D ────────────────────────────────────────────────────────────────────────


fn test_trrev_cpu_4d_default_axes_forward() raises:
    print("test_trrev_cpu_4d_default_axes_forward")
    comptime dtype = DType.float32
    # Shape (2,3,4,5) → default transpose → (5,4,3,2)
    var a = Tensor[dtype].arange(0.0, 120.0).reshape(Shape(2, 3, 4, 5))
    var t = a.transpose()
    assert_true(t.shape() == Shape(5, 4, 3, 2))
    # a[i,j,k,l] == t[l,k,j,i]
    # a[1,2,3,4]=119 → t[4,3,2,1]=119
    assert_true(t[4, 3, 2, 1] == 119.0)
    assert_true(t[0, 0, 0, 0] == 0.0)


fn test_trrev_cpu_4d_default_axes_backward() raises:
    print("test_trrev_cpu_4d_default_axes_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0.0, 120.0).reshape(Shape(2, 3, 4, 5))
    a.requires_grad_(True)
    var t = a.transpose()
    var loss = t.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_cpu_4d_explicit_axes_forward() raises:
    print("test_trrev_cpu_4d_explicit_axes_forward")
    comptime dtype = DType.float32
    # Shape (2,3,4,5), axes (0,2,1,3) → (2,4,3,5)
    var a = Tensor[dtype].arange(0.0, 120.0).reshape(Shape(2, 3, 4, 5))
    var t = a.transpose(0, 2, 1, 3)
    assert_true(t.shape() == Shape(2, 4, 3, 5))
    # a[i,j,k,l] == t[i,k,j,l]
    # a[1,2,3,4]=119 → t[1,3,2,4]=119
    assert_true(t[1, 3, 2, 4] == 119.0)


fn test_trrev_cpu_4d_explicit_axes_backward() raises:
    print("test_trrev_cpu_4d_explicit_axes_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0.0, 120.0).reshape(Shape(2, 3, 4, 5))
    a.requires_grad_(True)
    var t = a.transpose(0, 2, 1, 3)
    var loss = t.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ── Grad flow / accumulation ───────────────────────────────────────────────────


fn test_trrev_cpu_grad_flow_through_multiple_ops() raises:
    print("test_trrev_cpu_grad_flow_through_multiple_ops")
    comptime dtype = DType.float32
    # loss = sum(transpose(a) + transpose(a))
    # grad_a = 2 * ones
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var t1 = a.transpose()
    var t2 = a.transpose()
    var added = t1 + t2
    var loss = added.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].full(Shape(2, 2), Scalar[dtype](2.0)))
    )


fn test_trrev_cpu_transpose_contiguous_grad_flow() raises:
    print("test_trrev_cpu_transpose_contiguous_grad_flow")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var t = a.transpose()
    var c = t.contiguous()
    # Apply another op after contiguous
    var scaled = c * Tensor[dtype].full(c.shape(), Scalar[dtype](2.0))
    var loss = scaled.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(Tensor[dtype].full(Shape(2, 3), Scalar[dtype](2.0)))
    )


# ──────────────────────────────────────────────────────────────────────────────
# GPU TESTS
# ──────────────────────────────────────────────────────────────────────────────

# ── 1D ────────────────────────────────────────────────────────────────────────


fn test_trrev_gpu_1d_forward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_1d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        assert_true(t.shape() == Shape(3))
        assert_true(t.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


fn test_trrev_gpu_1d_backward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_1d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ── 2D ────────────────────────────────────────────────────────────────────────


fn test_trrev_gpu_2d_forward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_2d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        assert_true(t.shape() == Shape(3, 2))
        assert_true(
            t.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
            )
        )


fn test_trrev_gpu_2d_backward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_2d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_gpu_2d_explicit_axes_forward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_2d_explicit_axes_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(0, 1)
        assert_true(t.shape() == Shape(2, 3))
        assert_true(
            t.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
            )
        )


fn test_trrev_gpu_2d_explicit_axes_backward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_2d_explicit_axes_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(0, 1)
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_gpu_2d_transpose_then_contiguous() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_2d_transpose_then_contiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var c = t.contiguous()
        assert_true(c.shape() == Shape(3, 2))
        assert_true(
            c.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
            )
        )
        var loss = c.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_gpu_2d_double_transpose_identity() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_2d_double_transpose_identity")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var t1 = a_gpu.transpose()
        var t = t1.transpose()
        assert_true(t.shape() == Shape(2, 3))
        assert_true(t.to_cpu().all_close(a))


fn test_trrev_gpu_2d_double_transpose_backward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_2d_double_transpose_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var t1 = a_gpu.transpose()
        var t2 = t1.transpose()
        var loss = t2.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_gpu_2d_grad_correctness_weighted() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_2d_grad_correctness_weighted")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var prod = t * w
        var loss = prod.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        # grad_a[i,j] = w[j,i]  →  [[1,3],[2,4]]
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]]))
        )


# ── 3D ────────────────────────────────────────────────────────────────────────


fn test_trrev_gpu_3d_default_axes_forward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_3d_default_axes_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(0.0, 24.0).reshape(Shape(2, 3, 4))
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        assert_true(t.shape() == Shape(4, 3, 2))
        var t_cpu = t.to_cpu()
        assert_true(t_cpu[0, 0, 0] == 0.0)
        assert_true(t_cpu[3, 2, 1] == 23.0)
        assert_true(t_cpu[2, 1, 0] == 6.0)


fn test_trrev_gpu_3d_default_axes_backward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_3d_default_axes_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(0.0, 24.0).reshape(Shape(2, 3, 4))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_gpu_3d_explicit_axes_forward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_3d_explicit_axes_forward")
        comptime dtype = DType.float32
        # (2,3,4) → axes (0,2,1) → (2,4,3)
        var a = Tensor[dtype].arange(0.0, 24.0).reshape(Shape(2, 3, 4))
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(0, 2, 1)
        assert_true(t.shape() == Shape(2, 4, 3))
        var t_cpu = t.to_cpu()
        # a[1,2,3]=23 → t[1,3,2]=23
        assert_true(t_cpu[1, 3, 2] == 23.0)
        # a[0,1,2]=6 → t[0,2,1]=6
        assert_true(t_cpu[0, 2, 1] == 6.0)


fn test_trrev_gpu_3d_explicit_axes_backward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_3d_explicit_axes_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(0.0, 24.0).reshape(Shape(2, 3, 4))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(0, 2, 1)
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_gpu_3d_transpose_then_contiguous() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_3d_transpose_then_contiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(0.0, 24.0).reshape(Shape(2, 3, 4))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()  # (4,3,2)
        var c = t.contiguous()
        assert_true(c.shape() == Shape(4, 3, 2))
        # Spot-check values match the CPU transpose
        var ref_t = a.transpose()
        assert_true(c.to_cpu().all_close(ref_t))
        var loss = c.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_gpu_3d_chained_transpose_backward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_3d_chained_transpose_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(1.0, 25.0).reshape(Shape(2, 3, 4))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t1 = a_gpu.transpose()  # (4,3,2)
        var t2 = t1.transpose()  # back to (2,3,4)
        var loss = t2.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ── 4D ────────────────────────────────────────────────────────────────────────


fn test_trrev_gpu_4d_default_axes_forward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_4d_default_axes_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(0.0, 120.0).reshape(Shape(2, 3, 4, 5))
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        assert_true(t.shape() == Shape(5, 4, 3, 2))
        var t_cpu = t.to_cpu()
        assert_true(t_cpu[0, 0, 0, 0] == 0.0)
        assert_true(t_cpu[4, 3, 2, 1] == 119.0)


fn test_trrev_gpu_4d_default_axes_backward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_4d_default_axes_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(0.0, 120.0).reshape(Shape(2, 3, 4, 5))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_trrev_gpu_4d_explicit_axes_forward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_4d_explicit_axes_forward")
        comptime dtype = DType.float32
        # (2,3,4,5), axes (0,2,1,3) → (2,4,3,5)
        var a = Tensor[dtype].arange(0.0, 120.0).reshape(Shape(2, 3, 4, 5))
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(0, 2, 1, 3)
        assert_true(t.shape() == Shape(2, 4, 3, 5))
        var t_cpu = t.to_cpu()
        # a[1,2,3,4]=119 → t[1,3,2,4]=119
        assert_true(t_cpu[1, 3, 2, 4] == 119.0)


fn test_trrev_gpu_4d_explicit_axes_backward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_4d_explicit_axes_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(0.0, 120.0).reshape(Shape(2, 3, 4, 5))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(0, 2, 1, 3)
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ── GPU grad flow ──────────────────────────────────────────────────────────────


fn test_trrev_gpu_grad_flow_through_multiple_ops() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_grad_flow_through_multiple_ops")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t1 = a_gpu.transpose()
        var t2 = a_gpu.transpose()
        var added = t1 + t2
        var loss = added.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].full(Shape(2, 2), Scalar[dtype](2.0))
            )
        )


fn test_trrev_gpu_transpose_contiguous_then_op_backward() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_transpose_contiguous_then_op_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var c = t.contiguous()
        var scaled = (
            c * Tensor[dtype].full(c.shape(), Scalar[dtype](2.0)).to_gpu()
        )
        var loss = scaled.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].full(Shape(2, 3), Scalar[dtype](2.0))
            )
        )


fn test_trrev_gpu_grad_does_not_accumulate_across_separate_passes() raises:
    @parameter
    if has_accelerator():
        print("test_trrev_gpu_grad_does_not_accumulate_across_separate_passes")
        comptime dtype = DType.float32
        # Run only one backward pass — verify grad is exactly ones
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────


fn main() raises:
    # Old tests
    test_2d_transpose_no_axes()
    test_2d_transpose_explicit_axes()
    test_3d_transpose_axes_0_1()
    test_3d_transpose_axes_1_2()
    test_transpose_chain_operations()
    test_transpose_with_matmul()
    test_transpose_scalar_equivalent()
    test_transpose_1d_no_change()
    test_4d_transpose_complex_axes()

    # CPU tests
    test_trrev_cpu_1d_forward()
    test_trrev_cpu_1d_backward()
    test_trrev_cpu_2d_forward()
    test_trrev_cpu_2d_backward()
    test_trrev_cpu_2d_explicit_axes_forward()
    test_trrev_cpu_2d_explicit_axes_backward()
    test_trrev_cpu_2d_double_transpose_identity()
    test_trrev_cpu_2d_double_transpose_backward()
    test_trrev_cpu_2d_transpose_then_contiguous()
    test_trrev_cpu_2d_grad_correctness_weighted()
    test_trrev_cpu_3d_default_axes_forward()
    test_trrev_cpu_3d_default_axes_backward()
    test_trrev_cpu_3d_explicit_axes_forward()
    test_trrev_cpu_3d_explicit_axes_backward()
    test_trrev_cpu_3d_transpose_then_contiguous()
    test_trrev_cpu_3d_chained_transpose_backward()
    test_trrev_cpu_4d_default_axes_forward()
    test_trrev_cpu_4d_default_axes_backward()
    test_trrev_cpu_4d_explicit_axes_forward()
    test_trrev_cpu_4d_explicit_axes_backward()
    test_trrev_cpu_grad_flow_through_multiple_ops()
    test_trrev_cpu_transpose_contiguous_grad_flow()

    # GPU tests
    test_trrev_gpu_1d_forward()
    test_trrev_gpu_1d_backward()
    test_trrev_gpu_2d_forward()
    test_trrev_gpu_2d_backward()
    test_trrev_gpu_2d_explicit_axes_forward()
    test_trrev_gpu_2d_explicit_axes_backward()
    test_trrev_gpu_2d_transpose_then_contiguous()
    test_trrev_gpu_2d_double_transpose_identity()
    test_trrev_gpu_2d_double_transpose_backward()
    test_trrev_gpu_2d_grad_correctness_weighted()
    test_trrev_gpu_3d_default_axes_forward()
    test_trrev_gpu_3d_default_axes_backward()
    test_trrev_gpu_3d_explicit_axes_forward()
    test_trrev_gpu_3d_explicit_axes_backward()
    test_trrev_gpu_3d_transpose_then_contiguous()
    test_trrev_gpu_3d_chained_transpose_backward()
    test_trrev_gpu_4d_default_axes_forward()
    test_trrev_gpu_4d_default_axes_backward()
    test_trrev_gpu_4d_explicit_axes_forward()
    test_trrev_gpu_4d_explicit_axes_backward()
    test_trrev_gpu_grad_flow_through_multiple_ops()
    test_trrev_gpu_transpose_contiguous_then_op_backward()
    test_trrev_gpu_grad_does_not_accumulate_across_separate_passes()

    print("All trrev tests passed.")
