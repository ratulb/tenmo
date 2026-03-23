from tenmo import Tensor
from shapes import Shape
from sys import has_accelerator
from testing import assert_true


fn test_squeeze_scalar() raises:
    print("test_squeeze_scalar")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(42.0, requires_grad=True)
    var s = a.squeeze()
    assert_true(s.shape() == Shape.of())
    assert_true(s.item() == 42.0)
    s.backward()
    assert_true(a.grad().item() == 1.0)


fn test_squeeze_1d_no_effect() raises:
    print("test_squeeze_1d_no_effect")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var s = a.squeeze()
    assert_true(s.shape() == Shape.of(3))
    assert_true((s == a))
    s = s.sum()
    s.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


fn test_squeeze_2d_singleton_row() raises:
    print("test_squeeze_2d_singleton_row")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0]], requires_grad=True
    )  # shape (1,3)
    var s = a.squeeze()
    assert_true(s.shape() == Shape.of(3))
    assert_true(s.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))
    s = s.sum()
    s.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d2([[1.0, 1.0, 1.0]])))


fn test_squeeze_3d_multiple_singletons() raises:
    print("test_squeeze_3d_multiple_singletons")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[10.0, 20.0]]], requires_grad=True
    )  # shape (1,1,2)
    var s = a.squeeze()
    assert_true(s.shape() == Shape.of(2))
    assert_true(s.all_close(Tensor[dtype].d1([10.0, 20.0])))
    s = s.sum()
    s.backward()
    expected_grad = Tensor[dtype].d3([[[1.0, 1.0]]])
    assert_true(a.grad().all_close(expected_grad))


fn test_squeeze_with_specific_dim() raises:
    print("test_squeeze_with_specific_dim")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True  # shape (1,2,2)
    )
    var s = a.squeeze([0])
    assert_true(s.shape() == Shape.of(2, 2))
    assert_true(s.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))
    s = s.sum()
    s.backward()
    expected_grad = Tensor[dtype].d3([[[1.0, 1.0], [1.0, 1.0]]])
    assert_true(a.grad().all_close(expected_grad))


fn test_squeeze_with_non_singleton_dim() raises:
    print("test_squeeze_with_non_singleton_dim")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True  # shape (2,1,2)
    )
    var s = a.squeeze([1])  # valid because dim=1 has size 1
    assert_true(s.shape() == Shape.of(2, 2))
    expected = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    assert_true(s.all_close(expected))
    s = s.sum()
    s.backward()
    expected_grad = Tensor[dtype].d3([[[1.0, 1.0]], [[1.0, 1.0]]])
    assert_true(a.grad().all_close(expected_grad))


fn test_squeeze_keep_chain_grad() raises:
    print("test_squeeze_keep_chain_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0, 3.0]]], requires_grad=True
    )  # shape (1,1,3)
    var s = a.squeeze()
    var y = s * 2.0
    var z = y.sum()
    z.backward()
    # z = sum(2 * a) → grad(a) = 2
    expected_grad = Tensor[dtype].d3([[[2.0, 2.0, 2.0]]])
    assert_true(a.grad().all_close(expected_grad))






# ============================================================
# SQUEEZE TESTS — CPU
# ============================================================

fn test_squz_cpu_single_axis_dim0() raises:
    print("test_squz_cpu_single_axis_dim0")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True  # (1,2,2)
    )
    var s = a.squeeze([0])
    assert_true(s.shape() == Shape.of(2, 2))
    assert_true(s.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_squz_cpu_single_axis_dim1() raises:
    print("test_squz_cpu_single_axis_dim1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True  # (2,1,2)
    )
    var s = a.squeeze([1])
    assert_true(s.shape() == Shape.of(2, 2))
    assert_true(s.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[1.0, 1.0]], [[1.0, 1.0]]])))


fn test_squz_cpu_single_axis_last() raises:
    print("test_squz_cpu_single_axis_last")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True  # (2,2,1)
    )
    var s = a.squeeze([2])
    assert_true(s.shape() == Shape.of(2, 2))
    assert_true(s.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[1.0], [1.0]], [[1.0], [1.0]]])))


fn test_squz_cpu_all_size1_dims() raises:
    print("test_squz_cpu_all_size1_dims")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[5.0]]], requires_grad=True)  # (1,1,1)
    var s = a.squeeze([])   # squeeze all
    assert_true(s.shape() == Shape.of())   # scalar
    assert_true(s.all_close(Tensor[dtype].scalar(5.0)))
    s.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_squz_cpu_multiple_axes() raises:
    print("test_squz_cpu_multiple_axes")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)  # (1,3)
    var s = a.squeeze([0])
    assert_true(s.shape() == Shape.of(3))
    assert_true(s.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_squz_cpu_negative_axis() raises:
    print("test_squz_cpu_negative_axis")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True  # (2,1,2)
    )
    var s = a.squeeze([-2])   # same as axis=1
    assert_true(s.shape() == Shape.of(2, 2))
    assert_true(s.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[1.0, 1.0]], [[1.0, 1.0]]])))


fn test_squz_cpu_no_op_no_size1() raises:
    print("test_squz_cpu_no_op_no_size1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var s = a.squeeze([])
    # No size-1 dims → returns tensor unchanged
    assert_true(s.shape() == Shape.of(2, 2))
    assert_true(s.all_close(a))


fn test_squz_cpu_4d_middle_axes() raises:
    print("test_squz_cpu_4d_middle_axes")
    comptime dtype = DType.float32
    var a_leaf = Tensor[dtype].randn(2, 1, 1, 3)
    a_leaf.requires_grad_(True)
    var s = a_leaf.squeeze([1, 2])   # (2,3)
    assert_true(s.shape() == Shape.of(2, 3))
    var loss = s.sum()
    loss.backward()
    assert_true(a_leaf.grad().all_close(Tensor.ones_like(a_leaf)))


fn test_squz_cpu_grad_accumulation() raises:
    print("test_squz_cpu_grad_accumulation")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)  # (1,1,2)
    var s1 = a.squeeze([0])   # (1,2)
    var s2 = a.squeeze([0])   # (1,2)
    var loss1 = s1.sum()
    loss1.backward()
    var loss2 = s2.sum()
    loss2.backward()
    # Each backward contributes 1.0 → accumulated 2.0
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[2.0, 2.0]]])))


fn test_squz_cpu_track_grad_false() raises:
    print("test_squz_cpu_track_grad_false")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0]], requires_grad=True)
    var s = a.squeeze[track_grad=False]([0])
    assert_true(s.shape() == Shape.of(1))
    assert_true(not s.requires_grad)


fn test_squz_cpu_view_shares_storage() raises:
    print("test_squz_cpu_view_shares_storage")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(1, 4, 1, 6)
    a.requires_grad_(True)
    var s = a.squeeze([0, 2])   # (4,6)
    assert_true(s.shape() == Shape.of(4, 6))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_squz_cpu_chained_squeeze() raises:
    print("test_squz_cpu_chained_squeeze")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[3.0, 6.0]]], requires_grad=True)  # (1,1,2)
    var s1 = a.squeeze([0])    # (1,2)
    var s2 = s1.squeeze([0])   # (2,)
    var loss = s2.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ============================================================
# UNSQUEEZE TESTS — CPU
# ============================================================

fn test_unsquz_cpu_single_axis_front() raises:
    print("test_unsquz_cpu_single_axis_front")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2,2)
    var u = a.unsqueeze(0)   # (1,2,2)
    assert_true(u.shape() == Shape.of(1, 2, 2))
    assert_true(u.all_close(Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]])))
    var loss = u.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_cpu_single_axis_middle() raises:
    print("test_unsquz_cpu_single_axis_middle")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2,2)
    var u = a.unsqueeze(1)   # (2,1,2)
    assert_true(u.shape() == Shape.of(2, 1, 2))
    assert_true(u.all_close(Tensor[dtype].d3([[[1.0, 2.0]], [[3.0, 4.0]]])))
    var loss = u.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_cpu_single_axis_end() raises:
    print("test_unsquz_cpu_single_axis_end")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2,2)
    var u = a.unsqueeze(2)   # (2,2,1)
    assert_true(u.shape() == Shape.of(2, 2, 1))
    var loss = u.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_cpu_negative_axis() raises:
    print("test_unsquz_cpu_negative_axis")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var u = a.unsqueeze(-1)   # same as axis=2 → (2,2,1)
    assert_true(u.shape() == Shape.of(2, 2, 1))
    var loss = u.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_cpu_multiple_axes() raises:
    print("test_unsquz_cpu_multiple_axes")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)  # (3,)
    var u = a.unsqueeze(0, 2)   # (1,3,1)
    assert_true(u.shape() == Shape.of(1, 3, 1))
    var loss = u.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_cpu_1d_to_4d() raises:
    print("test_unsquz_cpu_1d_to_4d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var u = a.unsqueeze(0, 1, 2)   # (1,1,1,2)
    assert_true(u.shape() == Shape.of(1, 1, 1, 2))
    var loss = u.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_cpu_scalar_to_1d() raises:
    print("test_unsquz_cpu_scalar_to_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(7.0)
    a.requires_grad_(True)
    var u = a.unsqueeze(0)   # (1,)
    assert_true(u.shape() == Shape.of(1))
    var loss = u.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].scalar(1.0)))


fn test_unsquz_cpu_grad_accumulation() raises:
    print("test_unsquz_cpu_grad_accumulation")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var u1 = a.unsqueeze(0)
    var u2 = a.unsqueeze(0)
    var loss1 = u1.sum()
    loss1.backward()
    var loss2 = u2.sum()
    loss2.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


fn test_unsquz_cpu_track_grad_false() raises:
    print("test_unsquz_cpu_track_grad_false")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var u = a.unsqueeze[track_grad=False](0)
    assert_true(u.shape() == Shape.of(1, 2))
    assert_true(not u.requires_grad)


fn test_unsquz_cpu_chained_unsqueeze() raises:
    print("test_unsquz_cpu_chained_unsqueeze")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var u1 = a.unsqueeze(0)    # (1,2)
    var u2 = u1.unsqueeze(0)   # (1,1,2)
    var loss = u2.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_cpu_view_shares_storage() raises:
    print("test_unsquz_cpu_view_shares_storage")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(3, 4)
    a.requires_grad_(True)
    var u_orig = a.unsqueeze(0)
    var u = u_orig.unsqueeze(2)   # (1,3,1,4)
    assert_true(u.shape() == Shape.of(1, 3, 1, 4))
    var loss = u.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ============================================================
# SQUEEZE ↔ UNSQUEEZE ROUND-TRIP — CPU
# ============================================================

fn test_squz_unsquz_cpu_round_trip_axis0() raises:
    print("test_squz_unsquz_cpu_round_trip_axis0")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var u = a.unsqueeze(0)     # (1,2,2)
    var s = u.squeeze([0])     # (2,2)
    assert_true(s.shape() == a.shape())
    assert_true(s.all_close(a))
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_squz_unsquz_cpu_round_trip_axis1() raises:
    print("test_squz_unsquz_cpu_round_trip_axis1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(3, 5)
    a.requires_grad_(True)
    var u = a.unsqueeze(1)     # (3,1,5)
    var s = u.squeeze([1])     # (3,5)
    assert_true(s.shape() == a.shape())
    var loss = s.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ============================================================
# GPU SQUEEZE TESTS
# ============================================================

fn test_squz_gpu_single_axis_dim0() raises:
    @parameter
    if has_accelerator():
        print("test_squz_gpu_single_axis_dim0")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True  # (1,2,2)
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([0])
        assert_true(s.shape() == Shape.of(2, 2))
        assert_true(s.to_cpu().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_squz_gpu_single_axis_dim1() raises:
    @parameter
    if has_accelerator():
        print("test_squz_gpu_single_axis_dim1")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True  # (2,1,2)
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([1])
        assert_true(s.shape() == Shape.of(2, 2))
        assert_true(s.to_cpu().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d3([[[1.0, 1.0]], [[1.0, 1.0]]])))


fn test_squz_gpu_single_axis_last() raises:
    @parameter
    if has_accelerator():
        print("test_squz_gpu_single_axis_last")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True  # (2,2,1)
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([2])
        assert_true(s.shape() == Shape.of(2, 2))
        assert_true(s.to_cpu().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d3([[[1.0], [1.0]], [[1.0], [1.0]]])))


fn test_squz_gpu_all_size1_dims() raises:
    @parameter
    if has_accelerator():
        print("test_squz_gpu_all_size1_dims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[5.0]]], requires_grad=True)  # (1,1,1)
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([])
        assert_true(s.shape() == Shape.of())
        assert_true(s.to_cpu().all_close(Tensor[dtype].scalar(5.0)))
        s.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_squz_gpu_negative_axis() raises:
    @parameter
    if has_accelerator():
        print("test_squz_gpu_negative_axis")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True  # (2,1,2)
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([-2])
        assert_true(s.shape() == Shape.of(2, 2))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d3([[[1.0, 1.0]], [[1.0, 1.0]]])))


fn test_squz_gpu_multiple_axes() raises:
    @parameter
    if has_accelerator():
        print("test_squz_gpu_multiple_axes")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(1, 4, 1, 6)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([0, 2])
        assert_true(s.shape() == Shape.of(4, 6))
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_squz_gpu_matches_cpu() raises:
    @parameter
    if has_accelerator():
        print("test_squz_gpu_matches_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(1, 3, 1, 4)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.squeeze([0, 2])
        var s_cpu = a_copy.squeeze([0, 2])
        assert_true(s_gpu.to_cpu().all_close(s_cpu))
        var loss_gpu = s_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = s_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


fn test_squz_gpu_grad_lands_on_cpu() raises:
    @parameter
    if has_accelerator():
        print("test_squz_gpu_grad_lands_on_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0, 2.0, 3.0]]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([0])
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_squz_gpu_chained_squeeze() raises:
    @parameter
    if has_accelerator():
        print("test_squz_gpu_chained_squeeze")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[3.0, 6.0]]], requires_grad=True)  # (1,1,2)
        var a_gpu = a.to_gpu()
        var s1 = a_gpu.squeeze([0])    # (1,2)
        var s2 = s1.squeeze([0])       # (2,)
        var loss = s2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_squz_gpu_grad_accumulation() raises:
    @parameter
    if has_accelerator():
        print("test_squz_gpu_grad_accumulation")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s1 = a_gpu.squeeze([0])
        var s2 = a_gpu.squeeze([0])
        var loss1 = s1.sum()
        loss1.backward()
        a_gpu.zero_grad()
        var loss2 = s2.sum()
        loss2.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d3([[[2.0, 2.0]]])))


# ============================================================
# GPU UNSQUEEZE TESTS
# ============================================================


fn test_unsquz_gpu_single_axis_front() raises:
    @parameter
    if has_accelerator():
        print("test_unsquz_gpu_single_axis_front")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        print("check1")
        var u = a_gpu.unsqueeze(0)
        u.print()
        print("check2")
        assert_true(u.shape() == Shape.of(1, 2, 2))
        print("check3")
        assert_true(u.to_cpu().all_close(Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]])))
        print("check4")
        var loss = u.sum()
        print("check5 ")
        loss.print()
        loss.backward()
        print("check 6 - post backward")
        assert_true(not a.grad().is_on_gpu())
        print("check7")
        assert_true(a.grad().all_close(Tensor.ones_like(a)))
        print("check8")


fn test_unsquz_gpu_single_axis_middle() raises:
    @parameter
    if has_accelerator():
        print("test_unsquz_gpu_single_axis_middle")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(1)
        assert_true(u.shape() == Shape.of(2, 1, 2))
        assert_true(u.to_cpu().all_close(Tensor[dtype].d3([[[1.0, 2.0]], [[3.0, 4.0]]])))
        var loss = u.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_gpu_single_axis_end() raises:
    @parameter
    if has_accelerator():
        print("test_unsquz_gpu_single_axis_end")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(2)
        assert_true(u.shape() == Shape.of(2, 2, 1))
        var loss = u.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_gpu_negative_axis() raises:
    @parameter
    if has_accelerator():
        print("test_unsquz_gpu_negative_axis")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(-1)
        assert_true(u.shape() == Shape.of(2, 2, 1))
        var loss = u.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_gpu_multiple_axes() raises:
    @parameter
    if has_accelerator():
        print("test_unsquz_gpu_multiple_axes")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(0, 2)   # (1,3,1)
        assert_true(u.shape() == Shape.of(1, 3, 1))
        var loss = u.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_gpu_1d_to_4d() raises:
    @parameter
    if has_accelerator():
        print("test_unsquz_gpu_1d_to_4d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(0, 1, 2)   # (1,1,1,2)
        assert_true(u.shape() == Shape.of(1, 1, 1, 2))
        var loss = u.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_gpu_matches_cpu() raises:
    @parameter
    if has_accelerator():
        print("test_unsquz_gpu_matches_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(3, 4)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var u_gpu = a_gpu.unsqueeze(0, 2)
        var u_cpu = a_copy.unsqueeze(0, 2)
        assert_true(u_gpu.to_cpu().all_close(u_cpu))
        var loss_gpu = u_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = u_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


fn test_unsquz_gpu_grad_lands_on_cpu() raises:
    @parameter
    if has_accelerator():
        print("test_unsquz_gpu_grad_lands_on_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(0)
        var loss = u.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_gpu_chained_unsqueeze() raises:
    @parameter
    if has_accelerator():
        print("test_unsquz_gpu_chained_unsqueeze")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u1 = a_gpu.unsqueeze(0)    # (1,2)
        var u2 = u1.unsqueeze(0)       # (1,1,2)
        var loss = u2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_unsquz_gpu_grad_accumulation() raises:
    @parameter
    if has_accelerator():
        print("test_unsquz_gpu_grad_accumulation")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u1 = a_gpu.unsqueeze(0)
        var u2 = a_gpu.unsqueeze(0)
        var loss1 = u1.sum()
        loss1.backward()
        var loss2 = u2.sum()
        loss2.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


# ============================================================
# GPU SQUEEZE ↔ UNSQUEEZE ROUND-TRIP
# ============================================================

fn test_squz_unsquz_gpu_round_trip_axis0() raises:
    @parameter
    if has_accelerator():
        print("test_squz_unsquz_gpu_round_trip_axis0")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(0)     # (1,2,2)
        var s = u.squeeze([0])         # (2,2)
        assert_true(s.shape() == a.shape())
        assert_true(s.to_cpu().all_close(a))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_squz_unsquz_gpu_round_trip_multi() raises:
    @parameter
    if has_accelerator():
        print("test_squz_unsquz_gpu_round_trip_multi")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(3, 5)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(0, 2)   # (1,3,1,5)
        var s = u.squeeze([0, 2])       # (3,5)
        assert_true(s.shape() == a.shape())
        var loss_gpu = s.sum()
        loss_gpu.backward()
        var loss_cpu = a_copy.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


# ============================================================
# MAIN
# ============================================================

fn main() raises:
    #Old tests
    test_squeeze_scalar()
    test_squeeze_1d_no_effect()
    test_squeeze_2d_singleton_row()
    test_squeeze_3d_multiple_singletons()
    test_squeeze_with_specific_dim()
    test_squeeze_with_non_singleton_dim()
    test_squeeze_keep_chain_grad()

    # CPU squeeze
    test_squz_cpu_single_axis_dim0()
    test_squz_cpu_single_axis_dim1()
    test_squz_cpu_single_axis_last()
    test_squz_cpu_all_size1_dims()
    test_squz_cpu_multiple_axes()
    test_squz_cpu_negative_axis()
    test_squz_cpu_no_op_no_size1()
    test_squz_cpu_4d_middle_axes()
    test_squz_cpu_grad_accumulation()
    test_squz_cpu_track_grad_false()
    test_squz_cpu_view_shares_storage()
    test_squz_cpu_chained_squeeze()

    # CPU unsqueeze
    test_unsquz_cpu_single_axis_front()
    test_unsquz_cpu_single_axis_middle()
    test_unsquz_cpu_single_axis_end()
    test_unsquz_cpu_negative_axis()
    test_unsquz_cpu_multiple_axes()
    test_unsquz_cpu_1d_to_4d()
    test_unsquz_cpu_scalar_to_1d()
    test_unsquz_cpu_grad_accumulation()
    test_unsquz_cpu_track_grad_false()
    test_unsquz_cpu_chained_unsqueeze()
    test_unsquz_cpu_view_shares_storage()

    # CPU round-trips
    test_squz_unsquz_cpu_round_trip_axis0()
    test_squz_unsquz_cpu_round_trip_axis1()

    # GPU squeeze
    test_squz_gpu_single_axis_dim0()
    test_squz_gpu_single_axis_dim1()
    test_squz_gpu_single_axis_last()
    test_squz_gpu_all_size1_dims()
    test_squz_gpu_negative_axis()
    test_squz_gpu_multiple_axes()
    test_squz_gpu_matches_cpu()
    test_squz_gpu_grad_lands_on_cpu()
    test_squz_gpu_chained_squeeze()
    test_squz_gpu_grad_accumulation()

    # GPU unsqueeze
    test_unsquz_gpu_single_axis_front()
    test_unsquz_gpu_single_axis_middle()
    test_unsquz_gpu_single_axis_end()
    test_unsquz_gpu_negative_axis()
    test_unsquz_gpu_multiple_axes()
    test_unsquz_gpu_1d_to_4d()
    test_unsquz_gpu_matches_cpu()
    test_unsquz_gpu_grad_lands_on_cpu()
    test_unsquz_gpu_chained_unsqueeze()
    test_unsquz_gpu_grad_accumulation()

    # GPU round-trips
    test_squz_unsquz_gpu_round_trip_axis0()
    test_squz_unsquz_gpu_round_trip_multi()

    print("All squeeze/unsqueeze tests passed.")
