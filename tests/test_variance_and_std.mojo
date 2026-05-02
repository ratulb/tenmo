from std.testing import assert_true
from std.sys import has_accelerator
from tenmo.common_utils import i, s
from tenmo.tensor import Tensor
from tenmo.shapes import Shape



# ===== VARIANCE CPU TESTS =====


fn test_varstd_cpu_variance_scalar_global() raises:
    print("test_varstd_cpu_variance_scalar_global")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1(
        [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], requires_grad=True
    )
    var v = x.variance[track_grad=True](unbiased=False)
    # population variance = 4.0
    assert_true(v.all_close(Tensor[dtype].scalar(4.0)))
    var loss = v.sum()
    loss.backward()
    # grad = 2*(x - mean)/n, mean=5, n=8
    var expected = Tensor[dtype].d1(
        [-0.75, -0.25, -0.25, -0.25, 0.0, 0.0, 0.5, 1.0]
    )
    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_cpu_variance_unbiased_global() raises:
    print("test_varstd_cpu_variance_unbiased_global")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var v = x.variance[track_grad=True](unbiased=True)
    # unbiased variance = 2.5
    assert_true(v.all_close[atol=1e-5](Tensor[dtype].scalar(2.5)))
    var loss = v.sum()
    loss.backward()
    # grad = 2*(x-mean)/(n-1), mean=3, n=5, divisor=4
    var expected = Tensor[dtype].d1([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_cpu_variance_axis0() raises:
    print("test_varstd_cpu_variance_axis0")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var v = x.variance[track_grad=True](axis=0, keepdims=False, unbiased=False)
    # mean along axis 0: [3, 4], var: [8/3, 8/3]
    assert_true(v.shape() == Shape(2))
    assert_true(
        v.all_close[atol=1e-5](Tensor[dtype].d1([2.6666667, 2.6666667]))
    )
    var loss = v.sum()
    loss.backward()
    # grad = 2*(x - mean)/n for each element
    var expected = Tensor[dtype].d2(
        [[-1.3333334, -1.3333334], [0.0, 0.0], [1.3333334, 1.3333334]]
    )
    assert_true(x.grad().all_close[atol=1e-4](expected))


fn test_varstd_cpu_variance_axis1() raises:
    print("test_varstd_cpu_variance_axis1")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 8.0]], requires_grad=True)
    var v = x.variance[track_grad=True](axis=1, keepdims=False, unbiased=False)
    # row 0: mean=2, var=1. row 1: mean=5, var=9
    assert_true(v.shape() == Shape(2))
    assert_true(v.all_close[atol=1e-5](Tensor[dtype].d1([1.0, 9.0])))
    var loss = v.sum()
    loss.backward()
    var expected = Tensor[dtype].d2([[-1.0, 1.0], [-3.0, 3.0]])
    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_cpu_variance_axis1_keepdims() raises:
    print("test_varstd_cpu_variance_axis1_keepdims")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 8.0]], requires_grad=True)
    var v = x.variance[track_grad=True](axis=1, keepdims=True, unbiased=False)
    assert_true(v.shape() == Shape(2, 1))
    assert_true(v.all_close[atol=1e-5](Tensor[dtype].d2([[1.0], [9.0]])))
    var loss = v.sum()
    loss.backward()
    var expected = Tensor[dtype].d2([[-1.0, 1.0], [-3.0, 3.0]])
    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_cpu_variance_3d_axis1() raises:
    print("test_varstd_cpu_variance_3d_axis1")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )
    # shape (2,2,2), var along axis=1, unbiased=False
    var v = x.variance[track_grad=True](axis=1, keepdims=False, unbiased=False)
    assert_true(v.shape() == Shape(2, 2))
    # batch 0: means [2,3], vars [1,1]. batch 1: means [6,7], vars [1,1]
    assert_true(
        v.all_close[atol=1e-5](Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]]))
    )
    var loss = v.sum()
    loss.backward()
    assert_true(
        x.grad().all_close[atol=1e-5](
            Tensor[dtype].d3(
                [[[-1.0, -1.0], [1.0, 1.0]], [[-1.0, -1.0], [1.0, 1.0]]]
            )
        )
    )


fn test_varstd_cpu_variance_no_grad() raises:
    print("test_varstd_cpu_variance_no_grad")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var v = x.variance[track_grad=False](unbiased=False)
    assert_true(v.all_close[atol=1e-5](Tensor[dtype].scalar(0.6666667)))
    assert_true(not v.requires_grad)


# ===== STD CPU TESTS =====


fn test_varstd_cpu_std_global() raises:
    print("test_varstd_cpu_std_global")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1(
        [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], requires_grad=True
    )
    var s = x.std[track_grad=True](unbiased=False)
    # std = 2.0
    assert_true(s.all_close[atol=1e-5](Tensor[dtype].scalar(2.0)))
    var loss = s.sum()
    loss.backward()
    # grad = (x - mean) / (std * n), mean=5, std=2, n=8
    var expected = Tensor[dtype].d1(
        [-0.1875, -0.0625, -0.0625, -0.0625, 0.0, 0.0, 0.125, 0.25]
    )
    assert_true(x.grad().all_close[atol=1e-4](expected))


fn test_varstd_cpu_std_axis0() raises:
    print("test_varstd_cpu_std_axis0")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var s = x.std[track_grad=True](axis=0, keepdims=False, unbiased=False)
    assert_true(s.shape() == Shape(2))
    # col 0: mean=2, std=1. col 1: mean=3, std=1
    assert_true(s.all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0])))
    var loss = s.sum()
    loss.backward()
    var expected = Tensor[dtype].d2([[-0.5, -0.5], [0.5, 0.5]])
    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_cpu_std_axis1() raises:
    print("test_varstd_cpu_std_axis1")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 8.0]], requires_grad=True)
    var s = x.std[track_grad=True](axis=1, keepdims=False, unbiased=False)
    assert_true(s.shape() == Shape(2))
    # row 0: std=1. row 1: std=3
    assert_true(s.all_close[atol=1e-5](Tensor[dtype].d1([1.0, 3.0])))
    var loss = s.sum()
    loss.backward()
    var expected = Tensor[dtype].d2([[-0.5, 0.5], [-0.5, 0.5]])
    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_cpu_std_axis1_keepdims() raises:
    print("test_varstd_cpu_std_axis1_keepdims")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 8.0]], requires_grad=True)
    var s = x.std[track_grad=True](axis=1, keepdims=True, unbiased=False)
    assert_true(s.shape() == Shape(2, 1))
    assert_true(s.all_close[atol=1e-5](Tensor[dtype].d2([[1.0], [3.0]])))
    var loss = s.sum()
    loss.backward()
    var expected = Tensor[dtype].d2([[-0.5, 0.5], [-0.5, 0.5]])
    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_cpu_std_unbiased() raises:
    print("test_varstd_cpu_std_unbiased")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var s = x.std[track_grad=True](unbiased=True)
    # unbiased std = sqrt(2.5) ≈ 1.5811388
    assert_true(s.all_close[atol=1e-5](Tensor[dtype].scalar(1.5811388)))
    var loss = s.sum()
    loss.backward()
    # grad = (x-mean)/(std*(n-1)), mean=3, n=5, divisor=4
    var expected = Tensor[dtype].d1(
        [-0.3162278, -0.1581139, 0.0, 0.1581139, 0.3162278]
    )
    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_cpu_std_3d_axis2() raises:
    print("test_varstd_cpu_std_3d_axis2")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]], requires_grad=True
    )
    var s = x.std[track_grad=True](axis=2, keepdims=False, unbiased=False)
    assert_true(s.shape() == Shape(2, 2))
    # all rows have std=1
    assert_true(
        s.all_close[atol=1e-5](Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]]))
    )
    var loss = s.sum()
    loss.backward()
    assert_true(
        x.grad().all_close[atol=1e-5](
            Tensor[dtype].d3(
                [[[-0.5, 0.5], [-0.5, 0.5]], [[-0.5, 0.5], [-0.5, 0.5]]]
            )
        )
    )


fn test_varstd_cpu_std_no_grad() raises:
    print("test_varstd_cpu_std_no_grad")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var s = x.std[track_grad=False](unbiased=False)
    assert_true(s.all_close[atol=1e-5](Tensor[dtype].scalar(0.8164966)))
    assert_true(not s.requires_grad)


fn test_varstd_cpu_variance_grad_accumulation() raises:
    print("test_varstd_cpu_variance_grad_accumulation")
    comptime dtype = DType.float32
    # x used in two variance ops — grads should accumulate
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var v1 = x.variance[track_grad=True](unbiased=False)
    var v2 = x.variance[track_grad=True](unbiased=False)
    var loss = v1.sum()
    loss.backward()
    var loss2 = v2.sum()
    loss2.backward()
    # Each backward adds grad — should be 2x single backward
    var single_x = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var sv = single_x.variance[track_grad=True](unbiased=False)
    var sloss = sv.sum()
    sloss.backward()
    var expected = single_x.grad() * Tensor[dtype].scalar(2.0)
    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_cpu_std_vs_variance_consistency() raises:
    print("test_varstd_cpu_std_vs_variance_consistency")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var v = x.variance[track_grad=False](axis=1, keepdims=False, unbiased=False)
    var s = x.std[track_grad=False](axis=1, keepdims=False, unbiased=False)
    # std^2 should equal variance
    var s_sq = s * s
    assert_true(s_sq.all_close[atol=1e-5](v))


# ===== GPU TESTS =====


fn test_varstd_gpu_variance_global() raises:
    comptime if has_accelerator():
        print("test_varstd_gpu_variance_global")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1(
            [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], requires_grad=True
        )
        var x_gpu = x.to_gpu()
        var v = x_gpu.variance[track_grad=True](unbiased=False)
        assert_true(v.to_cpu().all_close[atol=1e-5](Tensor[dtype].scalar(4.0)))
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [-0.75, -0.25, -0.25, -0.25, 0.0, 0.0, 0.5, 1.0]
        )
        assert_true(x.grad().all_close[atol=1e-4](expected))


fn test_varstd_gpu_variance_axis0() raises:
    comptime if has_accelerator():
        print("test_varstd_gpu_variance_axis0")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var x_gpu = x.to_gpu()
        var v = x_gpu.variance[track_grad=True](
            axis=0, keepdims=False, unbiased=False
        )
        assert_true(v.shape() == Shape(2))
        assert_true(
            v.to_cpu().all_close[atol=1e-4](
                Tensor[dtype].d1([2.6666667, 2.6666667])
            )
        )
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d2(
           [[-1.3333334, -1.3333334], [0.0, 0.0], [1.3333334, 1.3333334]]
        )
        assert_true(x.grad().all_close[atol=1e-4](expected))


fn test_varstd_gpu_variance_axis1() raises:
    comptime if has_accelerator():
        print("test_varstd_gpu_variance_axis1")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 8.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var v = x_gpu.variance[track_grad=True](
            axis=1, keepdims=False, unbiased=False
        )
        assert_true(
            v.to_cpu().all_close[atol=1e-5](Tensor[dtype].d1([1.0, 9.0]))
        )
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d2([[-1.0, 1.0], [-3.0, 3.0]])
        assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_gpu_variance_keepdims() raises:
    comptime if has_accelerator():
        print("test_varstd_gpu_variance_keepdims")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 8.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var v = x_gpu.variance[track_grad=True](
            axis=1, keepdims=True, unbiased=False
        )
        assert_true(v.shape() == Shape(2, 1))
        assert_true(
            v.to_cpu().all_close[atol=1e-5](Tensor[dtype].d2([[1.0], [9.0]]))
        )
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d2([[-1.0, 1.0], [-3.0, 3.0]])
        assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_gpu_std_global() raises:
    comptime if has_accelerator():
        print("test_varstd_gpu_std_global")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1(
            [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], requires_grad=True
        )
        var x_gpu = x.to_gpu()
        var s = x_gpu.std[track_grad=True](unbiased=False)
        assert_true(s.to_cpu().all_close[atol=1e-5](Tensor[dtype].scalar(2.0)))
        var loss = s.sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [-0.1875, -0.0625, -0.0625, -0.0625, 0.0, 0.0, 0.125, 0.25]
        )
        assert_true(x.grad().all_close[atol=1e-4](expected))


fn test_varstd_gpu_std_axis1() raises:
    comptime if has_accelerator():
        print("test_varstd_gpu_std_axis1")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 8.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var s = x_gpu.std[track_grad=True](
            axis=1, keepdims=False, unbiased=False
        )
        assert_true(
            s.to_cpu().all_close[atol=1e-5](Tensor[dtype].d1([1.0, 3.0]))
        )
        var loss = s.sum()
        loss.backward()
        var expected = Tensor[dtype].d2([[-0.5, 0.5], [-0.5, 0.5]])
        assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_gpu_std_axis0_keepdims() raises:
    comptime if has_accelerator():
        print("test_varstd_gpu_std_axis0_keepdims")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var s = x_gpu.std[track_grad=True](
            axis=0, keepdims=True, unbiased=False
        )
        assert_true(s.shape() == Shape(1, 2))
        assert_true(
            s.to_cpu().all_close[atol=1e-5](Tensor[dtype].d2([[1.0, 1.0]]))
        )
        var loss = s.sum()
        loss.backward()
        var expected = Tensor[dtype].d2([[-0.5, -0.5], [0.5, 0.5]])
        assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_gpu_std_vs_cpu_consistency() raises:
    comptime if has_accelerator():
        print("test_varstd_gpu_std_vs_cpu_consistency")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var x_gpu = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            .to_gpu()
        )
        var s_cpu = x_cpu.std[track_grad=True](
            axis=1, keepdims=False, unbiased=False
        )
        var s_gpu = x_gpu.std[track_grad=True](
            axis=1, keepdims=False, unbiased=False
        )
        assert_true(s_cpu.all_close[atol=1e-5](s_gpu.to_cpu()))
        var loss_cpu = s_cpu.sum()
        loss_cpu.backward()
        var loss_gpu = s_gpu.sum()
        loss_gpu.backward()
        assert_true(x_cpu.grad().all_close[atol=1e-5](x_gpu.grad()))


fn test_varstd_gpu_variance_unbiased() raises:
    comptime if has_accelerator():
        print("test_varstd_gpu_variance_unbiased")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var v = x_gpu.variance[track_grad=True](unbiased=True)
        assert_true(v.to_cpu().all_close[atol=1e-5](Tensor[dtype].scalar(2.5)))
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d1([-1.0, -0.5, 0.0, 0.5, 1.0])
        assert_true(x.grad().all_close[atol=1e-5](expected))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


fn main() raises:
    # ── CPU ──
    test_varstd_cpu_variance_scalar_global()
    test_varstd_cpu_variance_unbiased_global()
    test_varstd_cpu_variance_axis0()
    test_varstd_cpu_variance_axis1()
    test_varstd_cpu_variance_axis1_keepdims()
    test_varstd_cpu_variance_3d_axis1()
    test_varstd_cpu_variance_no_grad()
    test_varstd_cpu_std_global()
    test_varstd_cpu_std_axis0()
    test_varstd_cpu_std_axis1()
    test_varstd_cpu_std_axis1_keepdims()
    test_varstd_cpu_std_unbiased()
    test_varstd_cpu_std_3d_axis2()
    test_varstd_cpu_std_no_grad()
    test_varstd_cpu_variance_grad_accumulation()
    test_varstd_cpu_std_vs_variance_consistency()

    # ── GPU ──
    test_varstd_gpu_variance_global()
    test_varstd_gpu_variance_axis0()
    test_varstd_gpu_variance_axis1()
    test_varstd_gpu_variance_keepdims()
    test_varstd_gpu_std_global()
    test_varstd_gpu_std_axis1()
    test_varstd_gpu_std_axis0_keepdims()
    test_varstd_gpu_std_vs_cpu_consistency()
    test_varstd_gpu_variance_unbiased()

    #Copied

     # ── Variance forward CPU ──
    test_varstd_var_fwd_cpu_global_1d()
    test_varstd_var_fwd_cpu_global_2d()
    test_varstd_var_fwd_cpu_axis0_2d()
    test_varstd_var_fwd_cpu_axis1_2d()
    test_varstd_var_fwd_cpu_axis1_keepdims()
    test_varstd_var_fwd_cpu_constant_is_zero()
    test_varstd_var_fwd_cpu_3d_axis1()

    # ── Variance backward CPU ──
    test_varstd_var_bwd_cpu_global_1d()
    test_varstd_var_bwd_cpu_global_unbiased()
    test_varstd_var_bwd_cpu_axis0_2d()
    test_varstd_var_bwd_cpu_axis1_2d()
    test_varstd_var_bwd_cpu_grad_zero_for_constant()
    test_varstd_var_bwd_cpu_3d_axis1()

    # ── Variance grad flow CPU ──
    test_varstd_var_gradflow_cpu_chained_with_sum()
    test_varstd_var_gradflow_cpu_no_grad_leaf()

    # ── Std forward CPU ──
    test_varstd_std_fwd_cpu_global_1d()
    test_varstd_std_fwd_cpu_global_unbiased()
    test_varstd_std_fwd_cpu_axis0_2d()
    test_varstd_std_fwd_cpu_axis1_2d()
    test_varstd_std_fwd_cpu_constant_near_zero()
    test_varstd_std_fwd_cpu_keepdims()
    test_varstd_std_fwd_cpu_3d()

    # ── Std backward CPU ──
    test_varstd_std_bwd_cpu_global_1d()
    test_varstd_std_bwd_cpu_axis1_2d()
    test_varstd_std_bwd_cpu_grad_zero_for_constant()
    test_varstd_std_bwd_cpu_3d_axis2()

    # ── Std grad flow CPU ──
    test_varstd_std_gradflow_cpu_chained_with_scalar()
    test_varstd_std_gradflow_cpu_no_grad_leaf()

    # ── Variance GPU ──
    test_varstd_var_fwd_gpu_global_1d()
    test_varstd_var_fwd_gpu_axis0_2d()
    test_varstd_var_fwd_gpu_axis1_2d()
    test_varstd_var_fwd_gpu_3d_axis1()
    test_varstd_var_bwd_gpu_global_1d()
    test_varstd_var_bwd_gpu_axis1_2d()
    test_varstd_var_bwd_gpu_3d_axis1()

    # ── Std GPU ──
    test_varstd_std_fwd_gpu_global_1d()
    test_varstd_std_fwd_gpu_axis0_2d()
    test_varstd_std_fwd_gpu_axis1_2d()
    test_varstd_std_fwd_gpu_3d()
    test_varstd_std_bwd_gpu_global_1d()
    test_varstd_std_bwd_gpu_axis1_2d()
    test_varstd_std_bwd_gpu_3d_axis2()

    # ── Parity ──
    test_varstd_parity_var_fwd_global()
    test_varstd_parity_var_fwd_axis1()
    test_varstd_parity_std_fwd_global()
    test_varstd_parity_std_fwd_axis1()
    test_varstd_parity_var_bwd_global()
    test_varstd_parity_std_bwd_axis1()


    print("All variance and std tests passed ✓")

# ═════════════════════════════════════════════════════════════════════════════
# VARIANCE — CPU
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Forward — CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_varstd_var_fwd_cpu_global_1d() raises:
    print("test_varstd_var_fwd_cpu_global_1d")
    comptime dtype = DType.float32
    # [1,2,3,4,5] — population var = 2.0, sample var = 2.5
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var v_pop  = x.variance[track_grad=False](unbiased=False)
    var v_samp = x.variance[track_grad=False](unbiased=True)
    assert_true(v_pop.all_close[atol=1e-5](Tensor[dtype].scalar(2.0)))
    assert_true(v_samp.all_close[atol=1e-5](Tensor[dtype].scalar(2.5)))


fn test_varstd_var_fwd_cpu_global_2d() raises:
    print("test_varstd_var_fwd_cpu_global_2d")
    comptime dtype = DType.float32
    # [[1,2],[3,4]] — global mean=2.5, pop var = (1+0.25+0.25+2.25)/4 = 1.25
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var v = x.variance[track_grad=False](unbiased=False)
    assert_true(v.all_close[atol=1e-5](Tensor[dtype].scalar(1.25)))


fn test_varstd_var_fwd_cpu_axis0_2d() raises:
    print("test_varstd_var_fwd_cpu_axis0_2d")
    comptime dtype = DType.float32
    # [[1,2],[3,4]] — var along axis 0 (over rows):
    # col0: var(1,3)=1.0(pop), col1: var(2,4)=1.0(pop)
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var v = x.variance[track_grad=False](axis=0, unbiased=False)
    assert_true(v.shape() == Shape(2))
    assert_true(v.all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0])))


fn test_varstd_var_fwd_cpu_axis1_2d() raises:
    print("test_varstd_var_fwd_cpu_axis1_2d")
    comptime dtype = DType.float32
    # [[1,3],[2,4]] — var along axis 1 (over cols):
    # row0: var(1,3)=1.0, row1: var(2,4)=1.0
    var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])
    var v = x.variance[track_grad=False](axis=1, unbiased=False)
    assert_true(v.shape() == Shape(2))
    assert_true(v.all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0])))


fn test_varstd_var_fwd_cpu_axis1_keepdims() raises:
    print("test_varstd_var_fwd_cpu_axis1_keepdims")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])
    var v = x.variance[track_grad=False](axis=1, keepdims=True, unbiased=False)
    assert_true(v.shape() == Shape(2, 1))
    assert_true(v.all_close[atol=1e-5](Tensor[dtype].d2([[1.0], [1.0]])))


fn test_varstd_var_fwd_cpu_constant_is_zero() raises:
    print("test_varstd_var_fwd_cpu_constant_is_zero")
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(4), 7.0)
    var v = x.variance[track_grad=False](unbiased=False)
    assert_true(v.all_close[atol=1e-6](Tensor[dtype].scalar(0.0)))


fn test_varstd_var_fwd_cpu_3d_axis1() raises:
    print("test_varstd_var_fwd_cpu_3d_axis1")
    comptime dtype = DType.float32
    # shape (2,3,4) — variance along axis 1
    var x = Tensor[dtype].arange(0.0, 24.0).reshape(2, 3, 4)
    var v = x.variance[track_grad=False](axis=1, unbiased=False)
    assert_true(v.shape() == Shape(2, 4))


# ─────────────────────────────────────────────────────────────────────────────
# Backward — CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_varstd_var_bwd_cpu_global_1d() raises:
    print("test_varstd_var_bwd_cpu_global_1d")
    comptime dtype = DType.float32
    # x = [1,2,3,4,5], mean=3, diff=[-2,-1,0,1,2]
    # pop var grad = 2/n * diff = 2/5 * diff = [-0.8,-0.4,0,0.4,0.8]
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var v = x.variance[track_grad=True](unbiased=False)
    var loss = v.sum()
    loss.backward()
    assert_true(x.grad().all_close[atol=1e-5](
        Tensor[dtype].d1([-0.8, -0.4, 0.0, 0.4, 0.8])
    ))


fn test_varstd_var_bwd_cpu_global_unbiased() raises:
    print("test_varstd_var_bwd_cpu_global_unbiased")
    comptime dtype = DType.float32
    # sample var grad = 2/(n-1) * diff = 2/4 * diff = [-1,-0.5,0,0.5,1]
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var v = x.variance[track_grad=True](unbiased=True)
    var loss = v.sum()
    loss.backward()
    assert_true(x.grad().all_close[atol=1e-5](
        Tensor[dtype].d1([-1.0, -0.5, 0.0, 0.5, 1.0])
    ))


fn test_varstd_var_bwd_cpu_axis0_2d() raises:
    print("test_varstd_var_bwd_cpu_axis0_2d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v = x.variance[track_grad=True](axis=0, unbiased=False)
    var loss = v.sum()
    loss.backward()
    # var(col0)=1, grad = 2/2 * [-1,1] = [-1,1] for col0
    # var(col1)=1, grad = 2/2 * [-1,1] = [-1,1] for col1
    assert_true(x.grad().all_close[atol=1e-5](
        Tensor[dtype].d2([[-1.0, -1.0], [1.0, 1.0]])
    ))


fn test_varstd_var_bwd_cpu_axis1_2d() raises:
    print("test_varstd_var_bwd_cpu_axis1_2d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
    var v = x.variance[track_grad=True](axis=1, unbiased=False)
    var loss = v.sum()
    loss.backward()
    # row0: mean=2, diff=[-1,1], grad=2/2*[-1,1]=[-1,1]
    # row1: mean=3, diff=[-1,1], grad=[-1,1]
    assert_true(x.grad().all_close[atol=1e-5](
        Tensor[dtype].d2([[-1.0, 1.0], [-1.0, 1.0]])
    ))


fn test_varstd_var_bwd_cpu_grad_zero_for_constant() raises:
    print("test_varstd_var_bwd_cpu_grad_zero_for_constant")
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(4), 3.0, requires_grad=True)
    var v = x.variance[track_grad=True](unbiased=False)
    var loss = v.sum()
    loss.backward()
    assert_true(x.grad().all_close[atol=1e-6](Tensor.zeros_like(x)))


fn test_varstd_var_bwd_cpu_3d_axis1() raises:
    print("test_varstd_var_bwd_cpu_3d_axis1")
    comptime dtype = DType.float32
    var x = Tensor[dtype].arange(0.0, 24.0).reshape(2, 3, 4)
    x.requires_grad_(True)
    var v = x.variance[track_grad=True](axis=1, unbiased=False)
    var loss = v.sum()
    loss.backward()
    assert_true(x.grad().shape() == Shape(2, 3, 4))
    # grad sum should be zero — variance grad sums to zero per reduced axis
    assert_true(
        x.gradients()[][i(0), s(), s()].sum().all_close[atol=1e-4](Tensor[dtype].scalar(0.0))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Grad flow — CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_varstd_var_gradflow_cpu_chained_with_sum() raises:
    print("test_varstd_var_gradflow_cpu_chained_with_sum")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var v = x.variance[track_grad=True](unbiased=False)
    var loss = v * Tensor[dtype].scalar(2.0)
    var l = loss.sum()
    l.backward()
    # grad = 2 * (2/3) * diff where diff = [-1,0,1]
    assert_true(x.grad().all_close[atol=1e-5](
        Tensor[dtype].d1([-4.0/3.0, 0.0, 4.0/3.0])
    ))


fn test_varstd_var_gradflow_cpu_no_grad_leaf() raises:
    print("test_varstd_var_gradflow_cpu_no_grad_leaf")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0])   # no grad
    var v = x.variance[track_grad=True](unbiased=False)
    assert_true(not v.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# STD — CPU
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Forward — CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_varstd_std_fwd_cpu_global_1d() raises:
    print("test_varstd_std_fwd_cpu_global_1d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var s = x.std[track_grad=False](unbiased=False)
    # pop std = sqrt(2) ≈ 1.4142135
    assert_true(s.all_close[atol=1e-5](Tensor[dtype].scalar(1.4142135)))


fn test_varstd_std_fwd_cpu_global_unbiased() raises:
    print("test_varstd_std_fwd_cpu_global_unbiased")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var s = x.std[track_grad=False](unbiased=True)
    # sample std = sqrt(2.5) ≈ 1.5811388
    assert_true(s.all_close[atol=1e-5](Tensor[dtype].scalar(1.5811388)))


fn test_varstd_std_fwd_cpu_axis0_2d() raises:
    print("test_varstd_std_fwd_cpu_axis0_2d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var s = x.std[track_grad=False](axis=0, unbiased=False)
    assert_true(s.shape() == Shape(2))
    assert_true(s.all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0])))


fn test_varstd_std_fwd_cpu_axis1_2d() raises:
    print("test_varstd_std_fwd_cpu_axis1_2d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])
    var s = x.std[track_grad=False](axis=1, unbiased=False)
    assert_true(s.shape() == Shape(2))
    assert_true(s.all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0])))


fn test_varstd_std_fwd_cpu_constant_near_zero() raises:
    print("test_varstd_std_fwd_cpu_constant_near_zero")
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(4), 5.0)
    var s = x.std[track_grad=False](unbiased=False)
    # std of constant = 0 (epsilon guards division)
    assert_true(s.all_close[atol=1e-5](Tensor[dtype].scalar(0.0)))


fn test_varstd_std_fwd_cpu_keepdims() raises:
    print("test_varstd_std_fwd_cpu_keepdims")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]])
    var s = x.std[track_grad=False](axis=1, keepdims=True, unbiased=False)
    assert_true(s.shape() == Shape(2, 1))


fn test_varstd_std_fwd_cpu_3d() raises:
    print("test_varstd_std_fwd_cpu_3d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].arange(0.0, 24.0).reshape(2, 3, 4)
    var s = x.std[track_grad=False](axis=2, unbiased=False)
    assert_true(s.shape() == Shape(2, 3))


# ─────────────────────────────────────────────────────────────────────────────
# Backward — CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_varstd_std_bwd_cpu_global_1d() raises:
    print("test_varstd_std_bwd_cpu_global_1d")
    comptime dtype = DType.float32
    # x=[1,2,3,4,5], std=sqrt(2), grad_std=1
    # dL/dx = diff / (std * n) = [-2,-1,0,1,2] / (sqrt(2)*5)
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var s = x.std[track_grad=True](unbiased=False)
    var loss = s.sum()
    loss.backward()
    var std_val = 1.4142135
    var expected = Tensor[dtype].d1([
        Scalar[dtype](-2.0 / (std_val * 5.0)),
        Scalar[dtype](-1.0 / (std_val * 5.0)),
        Scalar[dtype]( 0.0),
        Scalar[dtype]( 1.0 / (std_val * 5.0)),
        Scalar[dtype]( 2.0 / (std_val * 5.0)),
    ])
    assert_true(x.grad().all_close[atol=1e-5](expected))


fn test_varstd_std_bwd_cpu_axis1_2d() raises:
    print("test_varstd_std_bwd_cpu_axis1_2d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
    var s = x.std[track_grad=True](axis=1, unbiased=False)
    var loss = s.sum()
    loss.backward()
    # std=1 for both rows, diff=[-1,1], grad = diff/(std*n) = [-0.5, 0.5]
    assert_true(x.grad().all_close[atol=1e-5](
        Tensor[dtype].d2([[-0.5, 0.5], [-0.5, 0.5]])
    ))


fn test_varstd_std_bwd_cpu_grad_zero_for_constant() raises:
    print("test_varstd_std_bwd_cpu_grad_zero_for_constant")
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(4), 2.0, requires_grad=True)
    var s = x.std[track_grad=True](unbiased=False)
    var loss = s.sum()
    loss.backward()
    # diff = 0 everywhere => grad = 0
    assert_true(x.grad().all_close[atol=1e-5](Tensor.zeros_like(x)))


fn test_varstd_std_bwd_cpu_3d_axis2() raises:
    print("test_varstd_std_bwd_cpu_3d_axis2")
    comptime dtype = DType.float32
    var x = Tensor[dtype].arange(1.0, 25.0).reshape(2, 3, 4)
    x.requires_grad_(True)
    var s = x.std[track_grad=True](axis=2, unbiased=False)
    var loss = s.sum()
    loss.backward()
    assert_true(x.grad().shape() == Shape(2, 3, 4))
    assert_true(x.grad().sum().item() == x.grad().sum().item())  # no NaN


# ─────────────────────────────────────────────────────────────────────────────
# Grad flow — CPU
# ─────────────────────────────────────────────────────────────────────────────

fn test_varstd_std_gradflow_cpu_chained_with_scalar() raises:
    print("test_varstd_std_gradflow_cpu_chained_with_scalar")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var s = x.std[track_grad=True](unbiased=False)
    var scaled = s * Tensor[dtype].scalar(3.0)
    var loss = scaled.sum()
    loss.backward()
    # grad = 3 * diff / (std * n)
    assert_true(x.grad().shape() == Shape(5))
    assert_true(x.grad().sum().item() == x.grad().sum().item())  # no NaN


fn test_varstd_std_gradflow_cpu_no_grad_leaf() raises:
    print("test_varstd_std_gradflow_cpu_no_grad_leaf")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var s = x.std[track_grad=True](unbiased=False)
    assert_true(not s.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# VARIANCE — GPU
# ═════════════════════════════════════════════════════════════════════════════

fn test_varstd_var_fwd_gpu_global_1d() raises:
    comptime if has_accelerator():
        print("test_varstd_var_fwd_gpu_global_1d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0]).to_gpu()
        var v = x.variance[track_grad=False](unbiased=False)
        assert_true(v.to_cpu().all_close[atol=1e-5](Tensor[dtype].scalar(2.0)))


fn test_varstd_var_fwd_gpu_axis0_2d() raises:
    comptime if has_accelerator():
        print("test_varstd_var_fwd_gpu_axis0_2d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var v = x.variance[track_grad=False](axis=0, unbiased=False)
        assert_true(v.shape() == Shape(2))
        assert_true(v.to_cpu().all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0])))


fn test_varstd_var_fwd_gpu_axis1_2d() raises:
    comptime if has_accelerator():
        print("test_varstd_var_fwd_gpu_axis1_2d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]]).to_gpu()
        var v = x.variance[track_grad=False](axis=1, unbiased=False)
        assert_true(v.shape() == Shape(2))
        assert_true(v.to_cpu().all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0])))


fn test_varstd_var_fwd_gpu_3d_axis1() raises:
    comptime if has_accelerator():
        print("test_varstd_var_fwd_gpu_3d_axis1")
        comptime dtype = DType.float32
        var x = Tensor[dtype].arange(0.0, 24.0).reshape(2, 3, 4).to_gpu()
        var v = x.variance[track_grad=False](axis=1, unbiased=False)
        assert_true(v.shape() == Shape(2, 4))


fn test_varstd_var_bwd_gpu_global_1d() raises:
    comptime if has_accelerator():
        print("test_varstd_var_bwd_gpu_global_1d")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var v = x_gpu.variance[track_grad=True](unbiased=False)
        var loss = v.sum()
        loss.backward()
        assert_true(x_cpu.grad().all_close[atol=1e-5](
            Tensor[dtype].d1([-0.8, -0.4, 0.0, 0.4, 0.8])
        ))


fn test_varstd_var_bwd_gpu_axis1_2d() raises:
    comptime if has_accelerator():
        print("test_varstd_var_bwd_gpu_axis1_2d")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var v = x_gpu.variance[track_grad=True](axis=1, unbiased=False)
        var loss = v.sum()
        loss.backward()
        assert_true(x_cpu.grad().all_close[atol=1e-5](
            Tensor[dtype].d2([[-1.0, 1.0], [-1.0, 1.0]])
        ))


fn test_varstd_var_bwd_gpu_3d_axis1() raises:
    comptime if has_accelerator():
        print("test_varstd_var_bwd_gpu_3d_axis1")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].arange(0.0, 24.0).reshape(2, 3, 4)
        x_cpu.requires_grad_(True)
        var x_gpu = x_cpu.to_gpu()
        var v = x_gpu.variance[track_grad=True](axis=1, unbiased=False)
        var loss = v.sum()
        loss.backward()
        assert_true(x_cpu.grad().shape() == Shape(2, 3, 4))
        assert_true(x_cpu.grad().sum().item() == x_cpu.grad().sum().item())


# ═════════════════════════════════════════════════════════════════════════════
# STD — GPU
# ═════════════════════════════════════════════════════════════════════════════

fn test_varstd_std_fwd_gpu_global_1d() raises:
    comptime if has_accelerator():
        print("test_varstd_std_fwd_gpu_global_1d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0]).to_gpu()
        var s = x.std[track_grad=False](unbiased=False)
        assert_true(s.to_cpu().all_close[atol=1e-5](Tensor[dtype].scalar(1.4142135)))


fn test_varstd_std_fwd_gpu_axis0_2d() raises:
    comptime if has_accelerator():
        print("test_varstd_std_fwd_gpu_axis0_2d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var s = x.std[track_grad=False](axis=0, unbiased=False)
        assert_true(s.shape() == Shape(2))
        assert_true(s.to_cpu().all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0])))


fn test_varstd_std_fwd_gpu_axis1_2d() raises:
    comptime if has_accelerator():
        print("test_varstd_std_fwd_gpu_axis1_2d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]]).to_gpu()
        var s = x.std[track_grad=False](axis=1, unbiased=False)
        assert_true(s.shape() == Shape(2))
        assert_true(s.to_cpu().all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0])))


fn test_varstd_std_fwd_gpu_3d() raises:
    comptime if has_accelerator():
        print("test_varstd_std_fwd_gpu_3d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].arange(0.0, 24.0).reshape(2, 3, 4).to_gpu()
        var s = x.std[track_grad=False](axis=2, unbiased=False)
        assert_true(s.shape() == Shape(2, 3))


fn test_varstd_std_bwd_gpu_global_1d() raises:
    comptime if has_accelerator():
        print("test_varstd_std_bwd_gpu_global_1d")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var s = x_gpu.std[track_grad=True](unbiased=False)
        var loss = s.sum()
        loss.backward()
        var std_val = 1.4142135
        var expected = Tensor[dtype].d1([
            Scalar[dtype](-2.0 / (std_val * 5.0)),
            Scalar[dtype](-1.0 / (std_val * 5.0)),
            Scalar[dtype]( 0.0),
            Scalar[dtype]( 1.0 / (std_val * 5.0)),
            Scalar[dtype]( 2.0 / (std_val * 5.0)),
        ])
        assert_true(x_cpu.grad().all_close[atol=1e-5](expected))


fn test_varstd_std_bwd_gpu_axis1_2d() raises:
    comptime if has_accelerator():
        print("test_varstd_std_bwd_gpu_axis1_2d")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
        var x_gpu = x_cpu.to_gpu()
        var s = x_gpu.std[track_grad=True](axis=1, unbiased=False)
        var loss = s.sum()
        loss.backward()
        assert_true(x_cpu.grad().all_close[atol=1e-5](
            Tensor[dtype].d2([[-0.5, 0.5], [-0.5, 0.5]])
        ))


fn test_varstd_std_bwd_gpu_3d_axis2() raises:
    comptime if has_accelerator():
        print("test_varstd_std_bwd_gpu_3d_axis2")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].arange(1.0, 25.0).reshape(2, 3, 4)
        x_cpu.requires_grad_(True)
        var x_gpu = x_cpu.to_gpu()
        var s = x_gpu.std[track_grad=True](axis=2, unbiased=False)
        var loss = s.sum()
        loss.backward()
        assert_true(x_cpu.grad().shape() == Shape(2, 3, 4))
        assert_true(x_cpu.grad().sum().item() == x_cpu.grad().sum().item())


# ═════════════════════════════════════════════════════════════════════════════
# CPU / GPU PARITY
# ═════════════════════════════════════════════════════════════════════════════

fn test_varstd_parity_var_fwd_global() raises:
    comptime if has_accelerator():
        print("test_varstd_parity_var_fwd_global")
        comptime dtype = DType.float32
        var x = Tensor[dtype].arange(1.0, 11.0)
        var cpu_v = x.variance[track_grad=False](unbiased=False)
        var gpu_v = x.to_gpu().variance[track_grad=False](unbiased=False).to_cpu()
        assert_true(cpu_v.all_close[atol=1e-4](gpu_v))


fn test_varstd_parity_var_fwd_axis1() raises:
    comptime if has_accelerator():
        print("test_varstd_parity_var_fwd_axis1")
        comptime dtype = DType.float32
        var x = Tensor[dtype].arange(1.0, 13.0).reshape(3, 4)
        var cpu_v = x.variance[track_grad=False](axis=1, unbiased=False)
        var gpu_v = x.to_gpu().variance[track_grad=False](axis=1, unbiased=False).to_cpu()
        assert_true(cpu_v.all_close[atol=1e-4](gpu_v))


fn test_varstd_parity_std_fwd_global() raises:
    comptime if has_accelerator():
        print("test_varstd_parity_std_fwd_global")
        comptime dtype = DType.float32
        var x = Tensor[dtype].arange(1.0, 11.0)
        var cpu_s = x.std[track_grad=False](unbiased=False)
        var gpu_s = x.to_gpu().std[track_grad=False](unbiased=False).to_cpu()
        assert_true(cpu_s.all_close[atol=1e-4](gpu_s))


fn test_varstd_parity_std_fwd_axis1() raises:
    comptime if has_accelerator():
        print("test_varstd_parity_std_fwd_axis1")
        comptime dtype = DType.float32
        var x = Tensor[dtype].arange(1.0, 13.0).reshape(3, 4)
        var cpu_s = x.std[track_grad=False](axis=1, unbiased=False)
        var gpu_s = x.to_gpu().std[track_grad=False](axis=1, unbiased=False).to_cpu()
        assert_true(cpu_s.all_close[atol=1e-4](gpu_s))


fn test_varstd_parity_var_bwd_global() raises:
    comptime if has_accelerator():
        print("test_varstd_parity_var_bwd_global")
        comptime dtype = DType.float32
        var x_cpu_leaf = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        var loss_cpu = x_cpu_leaf.variance[track_grad=True](unbiased=False).sum()
        loss_cpu.backward()

        var x_gpu_leaf = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        var loss_gpu = x_gpu_leaf.to_gpu().variance[track_grad=True](unbiased=False).sum()
        loss_gpu.backward()

        assert_true(x_cpu_leaf.grad().all_close[atol=1e-5](x_gpu_leaf.grad()))


fn test_varstd_parity_std_bwd_axis1() raises:
    comptime if has_accelerator():
        print("test_varstd_parity_std_bwd_axis1")
        comptime dtype = DType.float32
        var x_cpu_leaf = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
        var loss_cpu = x_cpu_leaf.std[track_grad=True](axis=1, unbiased=False).sum()
        loss_cpu.backward()

        var x_gpu_leaf = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
        var loss_gpu = x_gpu_leaf.to_gpu().std[track_grad=True](axis=1, unbiased=False).sum()
        loss_gpu.backward()

        assert_true(x_cpu_leaf.grad().all_close[atol=1e-5](x_gpu_leaf.grad()))

