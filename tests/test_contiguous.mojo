from std.testing import assert_true
from std.sys import has_accelerator
from tenmo import Tensor
from shapes import Shape

# ============================================================
# CONTIGUOUS TESTS — CPU
# ============================================================

# ------------------------------------------------------------
# Scalar
# ------------------------------------------------------------


fn test_contig_cpu_scalar_forward() raises:
    print("test_contig_cpu_scalar_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(42.0)
    a.requires_grad_(True)
    var c = a.contiguous()
    assert_true(c.shape() == Shape())
    assert_true(c.all_close(Tensor[dtype].scalar(42.0)))


fn test_contig_cpu_scalar_backward() raises:
    print("test_contig_cpu_scalar_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0)
    a.requires_grad_(True)
    var c = a.contiguous()
    c.backward()
    assert_true(a.grad().all_close(Tensor[dtype].scalar(1.0)))


# ------------------------------------------------------------
# 1D
# ------------------------------------------------------------


fn test_contig_cpu_1d_already_contiguous() raises:
    print("test_contig_cpu_1d_already_contiguous")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var c = a.contiguous()
    assert_true(c.shape() == Shape(4))
    assert_true(c.all_close(a))
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_cpu_1d_slice_view() raises:
    print("test_contig_cpu_1d_slice_view")
    comptime dtype = DType.float32
    # Create a non-contiguous view via transpose of a 2D then flatten — or use
    # a known strided view via unsqueeze + squeeze to produce offset
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    # transpose produces non-contiguous strides
    var t = a.transpose()  # (3,2), non-contiguous
    # var row = t.squeeze([1])                  # squeeze won't help here — use sum to get grad
    var c = t.contiguous()
    assert_true(c.shape() == Shape(3, 2))
    # Values: transpose of [[1,2,3],[4,5,6]] = [[1,4],[2,5],[3,6]]
    assert_true(
        c.all_close(Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))
    )
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_cpu_1d_values_preserved() raises:
    print("test_contig_cpu_1d_values_preserved")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([10.0, 20.0, 30.0], requires_grad=True)
    var c = a.contiguous()
    assert_true(c.all_close(Tensor[dtype].d1([10.0, 20.0, 30.0])))
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# 2D
# ------------------------------------------------------------


fn test_contig_cpu_2d_already_contiguous() raises:
    print("test_contig_cpu_2d_already_contiguous")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var c = a.contiguous()
    assert_true(c.shape() == Shape(2, 2))
    assert_true(c.all_close(a))
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_cpu_2d_transpose_non_contiguous() raises:
    print("test_contig_cpu_2d_transpose_non_contiguous")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var t = a.transpose()  # (3,2), non-contiguous
    var c = t.contiguous()
    assert_true(c.shape() == Shape(3, 2))
    assert_true(
        c.all_close(Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))
    )
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_cpu_2d_non_uniform_values_grad() raises:
    print("test_contig_cpu_2d_non_uniform_values_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var t = a.transpose()  # (2,2), non-contiguous: [[1,3],[2,4]]
    var c = t.contiguous()
    assert_true(c.all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])))
    # Weighted loss to get non-uniform grads
    var weights = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var loss = (c * weights).sum()
    loss.backward()
    # grad flows back through contiguous → transpose → a
    # grad at c: [[1,2],[3,4]]
    # through transpose back to a: a[i,j] ← c[j,i] grad
    # a[0,0]←c[0,0]=1, a[0,1]←c[1,0]=3, a[1,0]←c[0,1]=2, a[1,1]←c[1,1]=4
    assert_true(a.grad().all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])))


fn test_contig_cpu_2d_large_random() raises:
    print("test_contig_cpu_2d_large_random")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(16, 32)
    a.requires_grad_(True)
    var a_copy = a.copy()
    a_copy.requires_grad_(True)
    var c = a.contiguous()
    assert_true(c.all_close(a_copy))
    var loss = c.sum()
    loss.backward()
    var loss_ref = a_copy.sum()
    loss_ref.backward()
    assert_true(a.grad().all_close(a_copy.grad()))


# ------------------------------------------------------------
# 3D
# ------------------------------------------------------------


fn test_contig_cpu_3d_already_contiguous() raises:
    print("test_contig_cpu_3d_already_contiguous")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    var c = a.contiguous()
    assert_true(c.shape() == Shape(2, 2, 2))
    assert_true(c.all_close(a))
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_cpu_3d_transpose_last_axes() raises:
    print("test_contig_cpu_3d_transpose_last_axes")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    # Transpose last two axes: (2,2,2) → (2,2,2) but non-contiguous
    var t = a.transpose(axes=[-1, -2])
    var c = t.contiguous()
    assert_true(c.shape() == Shape(2, 2, 2))
    # [[[1,3],[2,4]],[[5,7],[6,8]]]
    assert_true(
        c.all_close(
            Tensor[dtype].d3(
                [[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]]
            )
        )
    )
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_cpu_3d_unsqueeze_then_contiguous() raises:
    print("test_contig_cpu_3d_unsqueeze_then_contiguous")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var u = a.unsqueeze(1)  # (2,1,3) — view
    var c = u.contiguous()
    assert_true(c.shape() == Shape(2, 1, 3))
    assert_true(
        c.all_close(Tensor[dtype].d3([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]))
    )
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_cpu_3d_squeeze_then_contiguous() raises:
    print("test_contig_cpu_3d_squeeze_then_contiguous")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True  # (2,1,2)
    )
    var s = a.squeeze([1])  # (2,2) — view
    var c = s.contiguous()
    assert_true(c.shape() == Shape(2, 2))
    assert_true(c.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# 4D
# ------------------------------------------------------------


fn test_contig_cpu_4d_already_contiguous() raises:
    print("test_contig_cpu_4d_already_contiguous")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(2, 3, 4, 5)
    a.requires_grad_(True)
    var c = a.contiguous()
    assert_true(c.shape() == Shape(2, 3, 4, 5))
    assert_true(c.all_close(a))
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_cpu_4d_transpose_non_contiguous() raises:
    print("test_contig_cpu_4d_transpose_non_contiguous")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(2, 3, 4, 5)
    a.requires_grad_(True)
    var a_copy = a.copy()
    a_copy.requires_grad_(True)
    var t = a.transpose(axes=[-1, -2])  # (2,3,5,4)
    var c = t.contiguous()
    assert_true(c.shape() == Shape(2, 3, 5, 4))
    # cross-validate: CPU contiguous matches reference
    var t_ref = a_copy.transpose(axes=[-1, -2])
    var c_ref = t_ref.contiguous()
    assert_true(c.all_close(c_ref))
    var loss = c.sum()
    loss.backward()
    var loss_ref = c_ref.sum()
    loss_ref.backward()
    assert_true(a.grad().all_close(a_copy.grad()))


# ------------------------------------------------------------
# track_grad=False
# ------------------------------------------------------------


fn test_contig_cpu_track_grad_false() raises:
    print("test_contig_cpu_track_grad_false")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var c = a.contiguous[track_grad=False]()
    assert_true(c.shape() == Shape(2, 2))
    assert_true(not c.requires_grad)


# ------------------------------------------------------------
# Grad accumulation
# ------------------------------------------------------------


fn test_contig_cpu_grad_accumulation() raises:
    print("test_contig_cpu_grad_accumulation")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var c1 = a.contiguous()
    var c2 = a.contiguous()
    var loss1 = c1.sum()
    loss1.backward()
    var loss2 = c2.sum()
    loss2.backward()
    # Two backward passes each contribute 1.0 → accumulated 2.0
    assert_true(a.grad().all_close(Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])))


# ------------------------------------------------------------
# Chained contiguous
# ------------------------------------------------------------


fn test_contig_cpu_chained_contiguous() raises:
    print("test_contig_cpu_chained_contiguous")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var t = a.transpose()
    var c1 = t.contiguous()
    var c2 = c1.contiguous()  # second contiguous on already-contiguous
    assert_true(c2.all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])))
    var loss = c2.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ============================================================
# CONTIGUOUS TESTS — GPU
# ============================================================

# ------------------------------------------------------------
# Scalar GPU
# ------------------------------------------------------------


fn test_contig_gpu_scalar_forward() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_scalar_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].scalar(42.0)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        assert_true(c.shape() == Shape())
        assert_true(c.to_cpu().all_close(Tensor[dtype].scalar(42.0)))


fn test_contig_gpu_scalar_backward() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_scalar_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].scalar(3.0)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        c.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor[dtype].scalar(1.0)))


# ------------------------------------------------------------
# 1D GPU
# ------------------------------------------------------------


fn test_contig_gpu_1d_already_contiguous() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_1d_already_contiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        assert_true(c.shape() == Shape(4))
        assert_true(c.to_cpu().all_close(a))
        var loss = c.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_gpu_1d_values_preserved() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_1d_values_preserved")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([10.0, 20.0, 30.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        assert_true(c.to_cpu().all_close(Tensor[dtype].d1([10.0, 20.0, 30.0])))
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# 2D GPU
# ------------------------------------------------------------


fn test_contig_gpu_2d_already_contiguous() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_2d_already_contiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        assert_true(c.shape() == Shape(2, 2))
        assert_true(c.to_cpu().all_close(a))
        var loss = c.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_gpu_2d_transpose_non_contiguous() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_2d_transpose_non_contiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var c = t.contiguous()
        assert_true(c.shape() == Shape(3, 2))
        print("c gpu")
        c.print()
        c.to_cpu().print()
        assert_true(
            c.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
            )
        )
        var loss = c.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_gpu_2d_matches_cpu() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_2d_matches_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(8, 16)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t_gpu = a_gpu.transpose()
        var c_gpu = t_gpu.contiguous()
        var t_cpu = a_copy.transpose()
        var c_cpu = t_cpu.contiguous()
        assert_true(c_gpu.to_cpu().all_close(c_cpu))
        var loss_gpu = c_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = c_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


fn test_contig_gpu_2d_non_uniform_grad() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_2d_non_uniform_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var c = t.contiguous()
        var weights = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var loss = (c * weights).sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]]))
        )


# ------------------------------------------------------------
# 3D GPU
# ------------------------------------------------------------


fn test_contig_gpu_3d_already_contiguous() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_3d_already_contiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        assert_true(c.shape() == Shape(2, 2, 2))
        assert_true(c.to_cpu().all_close(a))
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_gpu_3d_transpose_last_axes() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_3d_transpose_last_axes")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(axes=[-1, -2])
        var c = t.contiguous()
        assert_true(c.shape() == Shape(2, 2, 2))
        assert_true(
            c.to_cpu().all_close(
                Tensor[dtype].d3(
                    [[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]]
                )
            )
        )
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_gpu_3d_unsqueeze_then_contiguous() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_3d_unsqueeze_then_contiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(1)
        var c = u.contiguous()
        assert_true(c.shape() == Shape(2, 1, 3))
        assert_true(
            c.to_cpu().all_close(
                Tensor[dtype].d3([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
            )
        )
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_gpu_3d_squeeze_then_contiguous() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_3d_squeeze_then_contiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True  # (2,1,2)
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([1])
        var c = s.contiguous()
        assert_true(c.shape() == Shape(2, 2))
        assert_true(
            c.to_cpu().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]))
        )
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_gpu_3d_matches_cpu() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_3d_matches_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(4, 5, 6)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t_gpu = a_gpu.transpose(axes=[-1, -2])
        var c_gpu = t_gpu.contiguous()
        var t_cpu = a_copy.transpose(axes=[-1, -2])
        var c_cpu = t_cpu.contiguous()
        assert_true(c_gpu.to_cpu().all_close(c_cpu))
        var loss_gpu = c_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = c_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


# ------------------------------------------------------------
# 4D GPU
# ------------------------------------------------------------


fn test_contig_gpu_4d_already_contiguous() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_4d_already_contiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(2, 3, 4, 5)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        assert_true(c.shape() == Shape(2, 3, 4, 5))
        assert_true(c.to_cpu().all_close(a))
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_contig_gpu_4d_transpose_non_contiguous() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_4d_transpose_non_contiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(2, 3, 4, 5)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t_gpu = a_gpu.transpose(axes=[-1, -2])
        var c_gpu = t_gpu.contiguous()
        assert_true(c_gpu.shape() == Shape(2, 3, 5, 4))
        var t_cpu = a_copy.transpose(axes=[-1, -2])
        var c_cpu = t_cpu.contiguous()
        assert_true(c_gpu.to_cpu().all_close(c_cpu))
        var loss_gpu = c_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = c_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


# ------------------------------------------------------------
# track_grad=False GPU
# ------------------------------------------------------------


fn test_contig_gpu_track_grad_false() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_track_grad_false")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous[track_grad=False]()
        assert_true(c.shape() == Shape(2, 2))
        assert_true(not c.requires_grad)


# ------------------------------------------------------------
# Grad accumulation GPU
# ------------------------------------------------------------


fn test_contig_gpu_grad_accumulation() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_grad_accumulation")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c1 = a_gpu.contiguous()
        var c2 = a_gpu.contiguous()
        var loss1 = c1.sum()
        loss1.backward()
        var loss2 = c2.sum()
        a_gpu.zero_grad()
        loss2.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]]))
        )


# ------------------------------------------------------------
# Grad lands on CPU
# ------------------------------------------------------------


fn test_contig_gpu_grad_lands_on_cpu() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_grad_lands_on_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        var loss = c.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# Chained contiguous GPU
# ------------------------------------------------------------


fn test_contig_gpu_chained_contiguous() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_chained_contiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var c1 = t.contiguous()
        var c2 = c1.contiguous()
        assert_true(
            c2.to_cpu().all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]]))
        )
        var loss = c2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# CPU tensor data unchanged after GPU contiguous
# ------------------------------------------------------------


fn test_contig_gpu_cpu_tensor_unchanged() raises:
    @parameter
    if has_accelerator():
        print("test_contig_gpu_cpu_tensor_unchanged")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_snapshot = a.copy()
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        var loss = c.sum()
        loss.backward()
        assert_true(a.all_close(a_snapshot))


# ============================================================
# MAIN
# ============================================================


fn main() raises:
    # CPU scalar
    test_contig_cpu_scalar_forward()
    test_contig_cpu_scalar_backward()

    # CPU 1D
    test_contig_cpu_1d_already_contiguous()
    test_contig_cpu_1d_slice_view()
    test_contig_cpu_1d_values_preserved()

    # CPU 2D
    test_contig_cpu_2d_already_contiguous()
    test_contig_cpu_2d_transpose_non_contiguous()
    test_contig_cpu_2d_non_uniform_values_grad()
    test_contig_cpu_2d_large_random()

    # CPU 3D
    test_contig_cpu_3d_already_contiguous()
    test_contig_cpu_3d_transpose_last_axes()
    test_contig_cpu_3d_unsqueeze_then_contiguous()
    test_contig_cpu_3d_squeeze_then_contiguous()

    # CPU 4D
    test_contig_cpu_4d_already_contiguous()
    test_contig_cpu_4d_transpose_non_contiguous()

    # CPU misc
    test_contig_cpu_track_grad_false()
    test_contig_cpu_grad_accumulation()
    test_contig_cpu_chained_contiguous()

    # GPU scalar
    test_contig_gpu_scalar_forward()
    test_contig_gpu_scalar_backward()

    # GPU 1D
    test_contig_gpu_1d_already_contiguous()
    test_contig_gpu_1d_values_preserved()

    # GPU 2D
    test_contig_gpu_2d_already_contiguous()
    test_contig_gpu_2d_transpose_non_contiguous()
    test_contig_gpu_2d_matches_cpu()
    test_contig_gpu_2d_non_uniform_grad()

    # GPU 3D
    test_contig_gpu_3d_already_contiguous()
    test_contig_gpu_3d_transpose_last_axes()
    test_contig_gpu_3d_unsqueeze_then_contiguous()
    test_contig_gpu_3d_squeeze_then_contiguous()
    test_contig_gpu_3d_matches_cpu()

    # GPU 4D
    test_contig_gpu_4d_already_contiguous()
    test_contig_gpu_4d_transpose_non_contiguous()

    # GPU misc
    test_contig_gpu_track_grad_false()
    test_contig_gpu_grad_accumulation()
    test_contig_gpu_grad_lands_on_cpu()
    test_contig_gpu_chained_contiguous()
    test_contig_gpu_cpu_tensor_unchanged()

    print("All contiguous tests passed.")
