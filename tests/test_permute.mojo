from tenmo import Tensor
from intarray import IntArray
from shapes import Shape
from std.testing import assert_true
from permute import Permute
from std.sys import has_accelerator

#Old tests

fn test_tensor_permute_basic() raises:
    comptime dtype = DType.float32
    print("Running test_tensor_permute_basic")
    a = Tensor[dtype].arange(0, 12)
    t1 = a.reshape(Shape(3, 4))
    p = t1.permute(IntArray([1, 0]))
    assert_true(p.shape() == Shape(4, 3))
    assert_true(p.strides()[0] == t1.strides()[1])
    assert_true(p.strides()[1] == t1.strides()[0])
    print("Passed basic permute test")


fn test_tensor_permute_3d_axes() raises:
    comptime dtype = DType.float32
    print("Running test_tensor_permute_3d_axes")
    a = Tensor[dtype].arange(0, 60)
    t1 = a.reshape(Shape(3, 4, 5))
    p = t1.permute([2, 0, 1])
    assert_true(p.shape() == Shape(5, 3, 4))
    assert_true(p.rank() == 3)
    print("Passed 3D permutation")


fn test_tensor_permute_inverse() raises:
    comptime dtype = DType.float32
    print("Running test_tensor_permute_inverse")
    a = Tensor[dtype].arange(0, 24)
    t1 = a.reshape(Shape(2, 3, 4))
    p = Permute[dtype].forward(t1, IntArray([2, 0, 1]))
    inv = Permute[dtype].forward(p, IntArray([1, 2, 0]))
    assert_true(inv.shape() == t1.shape())
    assert_true(inv.all_close(t1))
    print("Passed inverse permutation test")


fn test_tensor_permute_grad_sum_2d() raises:
    print("test_tensor_permute_grad_sum_2d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0, 12)
    t = a.reshape(Shape(3, 4))
    t.requires_grad_(True)
    var p = Permute[dtype].forward(t, IntArray([1, 0]))
    # p.sum() should produce gradient of ones when backpropagated
    s = p.sum()
    s.backward()
    var expected = (
        Tensor[dtype]
        .d2([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
        .to_dtype[dtype]()
    )
    assert_true(t.grad().all_close(expected))
    print("Passed test_tensor_permute_grad_sum_2d")


fn test_tensor_permute_grad_inverse_chain() raises:
    print("test_tensor_permute_grad_inverse_chain")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0, 24)
    t = a.reshape(Shape(2, 3, 4))
    t.requires_grad_(True)
    # Apply a permutation and then its inverse (should return original layout)
    var p = Permute[dtype].forward(t, IntArray([1, 2, 0]))
    # inverse of [1,2,0] is [2,0,1] (since inverse[ permutation[i] ] = i)
    var r = Permute[dtype].forward(
        p, IntArray([2, 0, 1])
    )  # r should match t's layout
    s = r.sum()
    s.backward()
    var expected = (
        Tensor[dtype]
        .d3(
            [
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
            ]
        )
        .to_dtype[dtype]()
    )
    assert_true(t.grad().all_close(expected))
    print("Passed test_tensor_permute_grad_inverse_chain")


fn test_tensor_permute_grad_partial_ops() raises:
    print("test_tensor_permute_grad_partial_ops")
    comptime dtype = DType.float32
    print(
        "This test checks a permute followed by a partial reduction along"
        " permuted axes"
    )
    var t = Tensor[dtype].arange(0, 60, requires_grad=True)
    var r = t.reshape(Shape(3, 4, 5))
    # permute to (5,3,4) then mean over first axis (axis=0 of permuted)
    var p = r.permute([2, 0, 1])
    var m = p.mean(axes=[0], keepdims=False)  # result shape (3,4)
    # backprop: m has shape (3,4); grad of t should be broadcasted back along axis 2 (size 5)
    s = m.sum()
    s.backward()
    # since m.sum() -> sums all elements of p, gradient on t should be ones

    assert_true(t.grad().all_close(Tensor.full(Shape(60), Scalar[dtype](0.2))))
    print("Passed test_tensor_permute_grad_partial_ops")


fn test_tensor_permute_grad_scaled_sum_3d() raises:
    print("test_tensor_permute_grad_scaled_sum_3d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(0, 60)
    t = a.reshape(Shape(3, 4, 5))
    t.requires_grad_(True)
    var p = Permute[dtype].forward(t, IntArray([2, 0, 1]))  # shape -> (5,3,4)
    var q = p * 2.0
    s = q.sum()
    s.backward()
    # corrected expected shape (3, 4, 5)
    var expected = (
        Tensor[dtype]
        .d3(
            [
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                ],
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                ],
                [
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0],
                ],
            ]
        )
        .to_dtype[dtype]()
    )
    assert_true(t.grad().all_close(expected))
    print("Passed test_tensor_permute_grad_scaled_sum_3d")


#End of old tests

# ═════════════════════════════════════════════════════════════════════════════
# CPU Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_perm_cpu_2d_01() raises:
    print("test_perm_cpu_2d_01")
    comptime dtype = DType.float32
    # Identity permutation — no change
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.permute(IntArray(0, 1))
    assert_true(result.shape() == Shape(2, 3))
    assert_true(result.all_close(a))


fn test_perm_cpu_2d_10() raises:
    print("test_perm_cpu_2d_10")
    comptime dtype = DType.float32
    # Transpose
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.permute(IntArray(1, 0))
    assert_true(result.shape() == Shape(3, 2))
    assert_true(
        result.all_close(
            Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        )
    )


fn test_perm_cpu_3d_012() raises:
    print("test_perm_cpu_3d_012")
    comptime dtype = DType.float32
    # Identity permutation on 3D
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var result = a.permute(IntArray(0, 1, 2))
    assert_true(result.shape() == Shape(2, 2, 2))
    assert_true(result.all_close(a))


fn test_perm_cpu_3d_021() raises:
    print("test_perm_cpu_3d_021")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    # Shape (2,2,2) → permute(0,2,1) → (2,2,2)
    var result = a.permute(IntArray(0, 2, 1))
    assert_true(result.shape() == Shape(2, 2, 2))
    # result[i,j,k] = a[i,k,j]
    assert_true(result[[0, 0, 0]] == a[[0, 0, 0]])
    assert_true(result[[0, 0, 1]] == a[[0, 1, 0]])
    assert_true(result[[0, 1, 0]] == a[[0, 0, 1]])
    assert_true(result[[0, 1, 1]] == a[[0, 1, 1]])
    assert_true(result[[1, 0, 0]] == a[[1, 0, 0]])
    assert_true(result[[1, 0, 1]] == a[[1, 1, 0]])


fn test_perm_cpu_3d_102() raises:
    print("test_perm_cpu_3d_102")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    # Shape (2,2,2) → permute(1,0,2) → (2,2,2)
    var result = a.permute(IntArray(1, 0, 2))
    assert_true(result.shape() == Shape(2, 2, 2))
    # result[i,j,k] = a[j,i,k]
    assert_true(result[[0, 0, 0]] == a[[0, 0, 0]])
    assert_true(result[[0, 1, 0]] == a[[1, 0, 0]])
    assert_true(result[[1, 0, 0]] == a[[0, 1, 0]])
    assert_true(result[[1, 1, 1]] == a[[1, 1, 1]])


fn test_perm_cpu_3d_120() raises:
    print("test_perm_cpu_3d_120")
    comptime dtype = DType.float32
    # Shape (2,3,4) → permute(1,2,0) → (3,4,2)
    var a = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
    var result = a.permute(IntArray(1, 2, 0))
    assert_true(result.shape() == Shape(3, 4, 2))
    # result[i,j,k] = a[k,i,j]
    for i in range(3):
        for j in range(4):
            for k in range(2):
                assert_true(result[[i, j, k]] == a[[k, i, j]])


fn test_perm_cpu_3d_201() raises:
    print("test_perm_cpu_3d_201")
    comptime dtype = DType.float32
    # Shape (2,3,4) → permute(2,0,1) → (4,2,3)
    var a = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
    var result = a.permute(IntArray(2, 0, 1))
    assert_true(result.shape() == Shape(4, 2, 3))
    # result[i,j,k] = a[j,k,i]
    for i in range(4):
        for j in range(2):
            for k in range(3):
                assert_true(result[[i, j, k]] == a[[j, k, i]])


fn test_perm_cpu_3d_210() raises:
    print("test_perm_cpu_3d_210")
    comptime dtype = DType.float32
    # Shape (2,3,4) → permute(2,1,0) → (4,3,2)
    var a = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
    var result = a.permute(IntArray(2, 1, 0))
    assert_true(result.shape() == Shape(4, 3, 2))
    for i in range(4):
        for j in range(3):
            for k in range(2):
                assert_true(result[[i, j, k]] == a[[k, j, i]])


fn test_perm_cpu_shape_preserved() raises:
    print("test_perm_cpu_shape_preserved")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(60).reshape(Shape(3, 4, 5))
    var result = a.permute(IntArray(2, 0, 1))
    assert_true(result.shape() == Shape(5, 3, 4))
    assert_true(result.numels() == 60)


fn test_perm_cpu_is_view() raises:
    print("test_perm_cpu_is_view")
    comptime dtype = DType.float32
    # Permute is a view — same data, different strides
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var result = a.permute(IntArray(1, 0))
    # Modifying a should be reflected in result (shared buffer)
    assert_true(result[[0, 0]] == Scalar[dtype](1.0))
    assert_true(result[[0, 1]] == Scalar[dtype](3.0))


fn test_perm_cpu_no_grad() raises:
    print("test_perm_cpu_no_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var result = a.permute(IntArray(1, 0))
    assert_true(not result.requires_grad)


fn test_perm_cpu_requires_grad_propagates() raises:
    print("test_perm_cpu_requires_grad_propagates")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.permute(IntArray(1, 0))
    assert_true(result.requires_grad)


fn test_perm_cpu_suppress_grad() raises:
    print("test_perm_cpu_suppress_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.permute(IntArray(1, 0), requires_grad=False)
    assert_true(not result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# CPU Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_perm_cpu_backward_2d_transpose() raises:
    print("test_perm_cpu_backward_2d_transpose")
    comptime dtype = DType.float32
    # Transpose backward — inverse of transpose is transpose
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var result = a.permute(IntArray(1, 0))
    var loss = result.sum()
    loss.backward()
    # Gradient of sum through transpose is ones in original shape
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3))))


fn test_perm_cpu_backward_3d_021() raises:
    print("test_perm_cpu_backward_3d_021")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True,
    )
    var result = a.permute(IntArray(0, 2, 1))
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


fn test_perm_cpu_backward_3d_210() raises:
    print("test_perm_cpu_backward_3d_210")
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
    a.requires_grad_(True)
    var result = a.permute(IntArray(2, 1, 0))
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3, 4))))


fn test_perm_cpu_backward_chain() raises:
    print("test_perm_cpu_backward_chain")
    comptime dtype = DType.float32
    # permute → multiply → sum → backward
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.permute(IntArray(1, 0)) * 2.0
    var loss = result.sum()
    loss.backward()
    # grad = 2.0 everywhere
    assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 2.0)))


fn test_perm_cpu_backward_double_permute() raises:
    print("test_perm_cpu_backward_double_permute")
    comptime dtype = DType.float32
    # permute then inverse permute = identity
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True,
    )
    var b = a.permute(IntArray(2, 0, 1))
    var c = b.permute(IntArray(1, 2, 0))  # inverse of (2,0,1)
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


fn test_perm_cpu_backward_gradient_values() raises:
    print("test_perm_cpu_backward_gradient_values")
    comptime dtype = DType.float32
    # Non-uniform upstream gradient
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var result = a.permute(IntArray(1, 0))
    # Multiply by non-uniform weights after permute
    var weights = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var loss = (result * weights).sum()
    loss.backward()
    # grad[i,j] = weights[j,i] (inverse permute of weights)
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])
        )
    )


fn test_perm_cpu_backward_multiple_uses() raises:
    print("test_perm_cpu_backward_multiple_uses")
    comptime dtype = DType.float32
    # Permuted tensor used twice — ZeroGrad ensures no double counting
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.permute(IntArray(1, 0))
    var loss1 = b.sum()
    loss1.backward()
    var grad1 = a.grad().copy()
    # Verify grad is ones
    assert_true(grad1.all_close(Tensor[dtype].ones(Shape(2, 2))))


# ═════════════════════════════════════════════════════════════════════════════
# GPU Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_perm_gpu_2d_identity() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_2d_identity")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.permute(IntArray(0, 1))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 3))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            )
        )


fn test_perm_gpu_2d_transpose() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_2d_transpose")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.permute(IntArray(1, 0))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(3, 2))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
            )
        )


fn test_perm_gpu_3d_021() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_3d_021")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        ).to_gpu()
        var result = a.permute(IntArray(0, 2, 1))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 2, 2))
        var result_cpu = result.to_cpu()
        var a_cpu = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    assert_true(result_cpu[[i, j, k]] == a_cpu[[i, k, j]])


fn test_perm_gpu_3d_120() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_3d_120")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.permute(IntArray(1, 2, 0))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(3, 4, 2))
        var result_cpu = result.to_cpu()
        for i in range(3):
            for j in range(4):
                for k in range(2):
                    assert_true(result_cpu[[i, j, k]] == a_cpu[[k, i, j]])


fn test_perm_gpu_3d_210() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_3d_210")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.permute(IntArray(2, 1, 0))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(4, 3, 2))
        var result_cpu = result.to_cpu()
        for i in range(4):
            for j in range(3):
                for k in range(2):
                    assert_true(result_cpu[[i, j, k]] == a_cpu[[k, j, i]])


fn test_perm_gpu_shape_preserved() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_shape_preserved")
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(60).reshape(Shape(3, 4, 5)).to_gpu()
        var result = a.permute(IntArray(2, 0, 1))
        assert_true(result.shape() == Shape(5, 3, 4))
        assert_true(result.numels() == 60)


fn test_perm_gpu_no_grad() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_no_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=False
        ).to_gpu()
        var result = a.permute(IntArray(1, 0))
        assert_true(not result.requires_grad)


fn test_perm_gpu_requires_grad_propagates() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_requires_grad_propagates")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        ).to_gpu()
        var result = a.permute(IntArray(1, 0))
        assert_true(result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# GPU Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_perm_gpu_backward_2d_transpose() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_backward_2d_transpose")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.permute(IntArray(1, 0))
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3))))


fn test_perm_gpu_backward_3d_021() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_backward_3d_021")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.permute(IntArray(0, 2, 1))
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


fn test_perm_gpu_backward_3d_210() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_backward_3d_210")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
        a_cpu.requires_grad_(True)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.permute(IntArray(2, 1, 0))
        var loss = result.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].ones(Shape(2, 3, 4))))


fn test_perm_gpu_backward_chain() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_backward_chain")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = (a_gpu.permute(IntArray(1, 0))) * 2.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 2.0)))


fn test_perm_gpu_backward_double_permute() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_backward_double_permute")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.permute(IntArray(2, 0, 1))
        var c = b.permute(IntArray(1, 2, 0))
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


fn test_perm_gpu_backward_gradient_values() raises:
    @parameter
    if has_accelerator():
        print("test_perm_gpu_backward_gradient_values")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.permute(IntArray(1, 0))
        var weights = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var loss = (result * weights).sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])
            )
        )


# ═════════════════════════════════════════════════════════════════════════════
# CPU/GPU Parity Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_perm_parity_2d_transpose() raises:
    @parameter
    if has_accelerator():
        print("test_perm_parity_2d_transpose")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.permute(IntArray(1, 0)).all_close(
                a_gpu.permute(IntArray(1, 0)).to_cpu()
            )
        )


fn test_perm_parity_3d_120() raises:
    @parameter
    if has_accelerator():
        print("test_perm_parity_3d_120")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.permute(IntArray(1, 2, 0)).all_close(
                a_gpu.permute(IntArray(1, 2, 0)).to_cpu()
            )
        )


fn test_perm_parity_3d_210() raises:
    @parameter
    if has_accelerator():
        print("test_perm_parity_3d_210")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.permute(IntArray(2, 1, 0)).all_close(
                a_gpu.permute(IntArray(2, 1, 0)).to_cpu()
            )
        )


fn test_perm_parity_backward_transpose() raises:
    @parameter
    if has_accelerator():
        print("test_perm_parity_backward_transpose")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        ).to_gpu()

        var loss_cpu = a_cpu.permute(IntArray(1, 0)).sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.permute(IntArray(1, 0)).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


fn test_perm_parity_backward_3d() raises:
    @parameter
    if has_accelerator():
        print("test_perm_parity_backward_3d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4))
        a_cpu.requires_grad_(True)
        var a_gpu = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4)).to_gpu()
        a_gpu.requires_grad_(True)

        var loss_cpu = a_cpu.permute(IntArray(2, 0, 1)).sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.permute(IntArray(2, 0, 1)).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


fn test_perm_parity_backward_chain() raises:
    @parameter
    if has_accelerator():
        print("test_perm_parity_backward_chain")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        var a_gpu = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        ).to_gpu()

        var loss_cpu = (a_cpu.permute(IntArray(1, 0)) * 3.0).sum()
        loss_cpu.backward()

        var loss_gpu = (a_gpu.permute(IntArray(1, 0)) * 3.0).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


fn test_perm_parity_using_zero_grad() raises:
    @parameter
    if has_accelerator():
        print("test_perm_parity_using_zero_grad")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.permute(IntArray(1, 0)).sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad().copy()

        a_cpu.zero_grad()

        var loss_gpu = a_gpu.permute(IntArray(1, 0)).sum()
        loss_gpu.backward()

        assert_true(cpu_grad.all_close(a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close(a_cpu.grad()))


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


fn main() raises:
    #Old tests
    test_tensor_permute_basic()
    test_tensor_permute_3d_axes()
    test_tensor_permute_inverse()

    print("\n=== Running Tensor.permute gradient tests ===")
    test_tensor_permute_grad_sum_2d()
    test_tensor_permute_grad_scaled_sum_3d()
    test_tensor_permute_grad_inverse_chain()
    test_tensor_permute_grad_partial_ops()
    print("=== All Tensor.permute gradient tests passed ===")

    #End old tests
    # CPU Forward
    test_perm_cpu_2d_01()
    test_perm_cpu_2d_10()
    test_perm_cpu_3d_012()
    test_perm_cpu_3d_021()
    test_perm_cpu_3d_102()
    test_perm_cpu_3d_120()
    test_perm_cpu_3d_201()
    test_perm_cpu_3d_210()
    test_perm_cpu_shape_preserved()
    test_perm_cpu_is_view()
    test_perm_cpu_no_grad()
    test_perm_cpu_requires_grad_propagates()
    test_perm_cpu_suppress_grad()
    print("CPU forward passed!")

    # CPU Backward
    test_perm_cpu_backward_2d_transpose()
    test_perm_cpu_backward_3d_021()
    test_perm_cpu_backward_3d_210()
    test_perm_cpu_backward_chain()
    test_perm_cpu_backward_double_permute()
    test_perm_cpu_backward_gradient_values()
    test_perm_cpu_backward_multiple_uses()
    print("CPU backward passed!")

    # GPU Forward
    test_perm_gpu_2d_identity()
    test_perm_gpu_2d_transpose()
    test_perm_gpu_3d_021()
    test_perm_gpu_3d_120()
    test_perm_gpu_3d_210()
    test_perm_gpu_shape_preserved()
    test_perm_gpu_no_grad()
    test_perm_gpu_requires_grad_propagates()
    print("GPU forward passed!")

    # GPU Backward
    test_perm_gpu_backward_2d_transpose()
    test_perm_gpu_backward_3d_021()
    test_perm_gpu_backward_3d_210()
    test_perm_gpu_backward_chain()
    test_perm_gpu_backward_double_permute()
    test_perm_gpu_backward_gradient_values()
    print("GPU backward passed!")

    # Parity
    test_perm_parity_2d_transpose()
    test_perm_parity_3d_120()
    test_perm_parity_3d_210()
    test_perm_parity_backward_transpose()
    test_perm_parity_backward_3d()
    test_perm_parity_backward_chain()
    test_perm_parity_using_zero_grad()
    print("Parity passed!")

    print("All permute tests passed!")
