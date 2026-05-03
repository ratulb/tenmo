from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.tensor import Tensor
from tenmo.shapes import Shape


# =============================================================================
# CPU FORWARD TESTS
# =============================================================================


fn test_recip_cpu_forward_1d() raises:
    print("test_recip_cpu_forward_1d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 4.0, 8.0])
    var r = x.reciprocal[track_grad=False]()
    assert_true(r.shape() == Shape(4))
    assert_true(r.all_close[atol=1e-6](Tensor[dtype].d1([1.0, 0.5, 0.25, 0.125])))


fn test_recip_cpu_forward_2d() raises:
    print("test_recip_cpu_forward_2d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0], [4.0, 5.0]])
    var r = x.reciprocal[track_grad=False]()
    assert_true(r.shape() == Shape(2, 2))
    assert_true(r.all_close[atol=1e-6](
        Tensor[dtype].d2([[1.0, 0.5], [0.25, 0.2]])
    ))


fn test_recip_cpu_forward_3d() raises:
    print("test_recip_cpu_forward_3d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[1.0, 2.0], [4.0, 8.0]], [[0.5, 0.25], [10.0, 100.0]]]
    )
    var r = x.reciprocal[track_grad=False]()
    assert_true(r.shape() == Shape(2, 2, 2))
    assert_true(r.all_close[atol=1e-5](
        Tensor[dtype].d3(
            [[[1.0, 0.5], [0.25, 0.125]], [[2.0, 4.0], [0.1, 0.01]]]
        )
    ))


fn test_recip_cpu_forward_negative_values() raises:
    print("test_recip_cpu_forward_negative_values")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([-1.0, -2.0, -4.0])
    var r = x.reciprocal[track_grad=False]()
    assert_true(r.all_close[atol=1e-6](Tensor[dtype].d1([-1.0, -0.5, -0.25])))


fn test_recip_cpu_forward_scalar() raises:
    print("test_recip_cpu_forward_scalar")
    comptime dtype = DType.float32
    var x = Tensor[dtype].scalar(4.0)
    var r = x.reciprocal[track_grad=False]()
    assert_true(r.all_close[atol=1e-6](Tensor[dtype].scalar(0.25)))


fn test_recip_cpu_forward_no_grad() raises:
    print("test_recip_cpu_forward_no_grad")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([2.0, 4.0], requires_grad=True)
    var r = x.reciprocal[track_grad=False]()
    assert_true(not r.requires_grad)


# =============================================================================
# CPU BACKWARD TESTS
# =============================================================================


fn test_recip_cpu_backward_1d() raises:
    print("test_recip_cpu_backward_1d")
    comptime dtype = DType.float32
    # x = [1, 2, 4], out = [1, 0.5, 0.25]
    # grad = -1/x² = [-1, -0.25, -0.0625]
    var x = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
    var r = x.reciprocal[track_grad=True]()
    var loss = r.sum()
    loss.backward()
    assert_true(x.grad().shape() == Shape(3))
    assert_true(x.grad().all_close[atol=1e-6](
        Tensor[dtype].d1([-1.0, -0.25, -0.0625])
    ))


fn test_recip_cpu_backward_2d() raises:
    print("test_recip_cpu_backward_2d")
    comptime dtype = DType.float32
    # x = [[1,2],[4,8]], grad = -1/x² = [[-1,-0.25],[-0.0625,-0.015625]]
    var x = Tensor[dtype].d2([[1.0, 2.0], [4.0, 8.0]], requires_grad=True)
    var r = x.reciprocal[track_grad=True]()
    var loss = r.sum()
    loss.backward()
    assert_true(x.grad().shape() == Shape(2, 2))
    assert_true(x.grad().all_close[atol=1e-6](
        Tensor[dtype].d2([[-1.0, -0.25], [-0.0625, -0.015625]])
    ))


fn test_recip_cpu_backward_3d() raises:
    print("test_recip_cpu_backward_3d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[1.0, 2.0], [4.0, 8.0]], [[1.0, 2.0], [4.0, 8.0]]], requires_grad=True
    )
    var r = x.reciprocal[track_grad=True]()
    var loss = r.sum()
    loss.backward()
    assert_true(x.grad().shape() == Shape(2, 2, 2))
    assert_true(x.grad().all_close[atol=1e-6](
        Tensor[dtype].d3(
            [[[-1.0, -0.25], [-0.0625, -0.015625]],
             [[-1.0, -0.25], [-0.0625, -0.015625]]]
        )
    ))


fn test_recip_cpu_backward_seed() raises:
    print("test_recip_cpu_backward_seed")
    comptime dtype = DType.float32
    # backward with non-unit seed — grad = seed * (-1/x²)
    var x = Tensor[dtype].d1([2.0, 4.0], requires_grad=True)
    var r = x.reciprocal[track_grad=True]()
    var loss = r.sum()
    loss.backward(Scalar[dtype](3.0))
    # grad = 3 * (-1/x²) = 3 * [-0.25, -0.0625] = [-0.75, -0.1875]
    assert_true(x.grad().all_close[atol=1e-6](
        Tensor[dtype].d1([-0.75, -0.1875])
    ))


fn test_recip_cpu_backward_negative() raises:
    print("test_recip_cpu_backward_negative")
    comptime dtype = DType.float32
    # x = [-2, -4], grad = -1/x² = [-0.25, -0.0625]
    # sign of x does not affect grad sign — x² is always positive
    var x = Tensor[dtype].d1([-2.0, -4.0], requires_grad=True)
    var r = x.reciprocal[track_grad=True]()
    var loss = r.sum()
    loss.backward()
    assert_true(x.grad().all_close[atol=1e-6](
        Tensor[dtype].d1([-0.25, -0.0625])
    ))


fn test_recip_cpu_backward_grad_accumulation() raises:
    print("test_recip_cpu_backward_grad_accumulation")
    comptime dtype = DType.float32
    # x used twice — grads accumulate
    var x = Tensor[dtype].d1([2.0, 4.0], requires_grad=True)
    var r1 = x.reciprocal[track_grad=True]()
    var r2 = x.reciprocal[track_grad=True]()
    var loss1 = r1.sum()
    loss1.backward()
    var loss2 = r2.sum()
    loss2.backward()
    # each backward adds -1/x² — total = 2 * [-0.25, -0.0625]
    assert_true(x.grad().all_close[atol=1e-6](
        Tensor[dtype].d1([-0.5, -0.125])
    ))


# =============================================================================
# CPU GRAD FLOW TESTS
# =============================================================================


fn test_recip_cpu_grad_flow_chain() raises:
    print("test_recip_cpu_grad_flow_chain")
    comptime dtype = DType.float32
    # loss = sum(1/x) * 2 — chain rule: grad = 2 * (-1/x²)
    var x = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
    var r = x.reciprocal[track_grad=True]()
    var scaled = r * Tensor[dtype].scalar(2.0)
    var loss = scaled.sum()
    loss.backward()
    # grad = 2 * (-1/x²) = [-2, -0.5, -0.125]
    assert_true(x.grad().all_close[atol=1e-6](
        Tensor[dtype].d1([-2.0, -0.5, -0.125])
    ))


fn test_recip_cpu_grad_flow_double_reciprocal() raises:
    print("test_recip_cpu_grad_flow_double_reciprocal")
    comptime dtype = DType.float32
    # loss = sum(1/(1/x)) = sum(x) — grad should be all ones
    var x = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
    var r1 = x.reciprocal[track_grad=True]()
    var r2 = r1.reciprocal[track_grad=True]()
    var loss = r2.sum()
    loss.backward()
    # 1/(1/x) = x, so grad = 1
    assert_true(x.grad().all_close[atol=1e-5](Tensor.ones_like(x)))


fn test_recip_cpu_grad_flow_with_multiply() raises:
    print("test_recip_cpu_grad_flow_with_multiply")
    comptime dtype = DType.float32
    # loss = sum(x * (1/x)) = sum(1s) = n — grad of x:
    # ∂loss/∂x[i] = 1/x[i] + x[i] * (-1/x[i]²) = 1/x - 1/x = 0
    var x = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
    var r = x.reciprocal[track_grad=True]()
    var product = x * r
    var loss = product.sum()
    loss.backward()
    assert_true(x.grad().all_close[atol=1e-5](
        Tensor[dtype].d1([0.0, 0.0, 0.0])
    ))



fn test_recip_cpu_grad_flow_sum_then_recip() raises:
    print("test_recip_cpu_grad_flow_sum_then_recip")
    comptime dtype = DType.float32
    # loss = 1 / sum(x), x = [1,2,3], sum=6
    # ∂loss/∂x[i] = -1/sum(x)² = -1/36
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var s = x.sum[track_grad=True]()
    var r = s.reciprocal[track_grad=True]()
    var loss = r.sum()
    loss.backward()
    # grad = -1/36 for each element
    assert_true(x.grad().all_close[atol=1e-6](
        Tensor[dtype].d1([-0.027778, -0.027778, -0.027778])
    ))


fn test_recip_cpu_consistency_with_divide() raises:
    print("test_recip_cpu_consistency_with_divide")
    comptime dtype = DType.float32
    # 1/x should equal 1.0 / x via scalar divide
    var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var r = x.reciprocal[track_grad=False]()
    var d = Tensor[dtype].scalar(1.0).__truediv__[track_grad=False](x)
    assert_true(r.all_close[atol=1e-6](d))


# =============================================================================
# GPU TESTS
# =============================================================================


fn test_recip_gpu_forward_1d() raises:
    comptime if has_accelerator():
        print("test_recip_gpu_forward_1d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 4.0, 8.0])
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=False]()
        assert_true(r.shape() == Shape(4))
        assert_true(r.to_cpu().all_close[atol=1e-6](
            Tensor[dtype].d1([1.0, 0.5, 0.25, 0.125])
        ))


fn test_recip_gpu_forward_2d() raises:
    comptime if has_accelerator():
        print("test_recip_gpu_forward_2d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0], [4.0, 8.0]])
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=False]()
        assert_true(r.shape() == Shape(2, 2))
        assert_true(r.to_cpu().all_close[atol=1e-6](
            Tensor[dtype].d2([[1.0, 0.5], [0.25, 0.125]])
        ))


fn test_recip_gpu_backward_1d() raises:
    comptime if has_accelerator():
        print("test_recip_gpu_backward_1d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=True]()
        var loss = r.sum()
        loss.backward()
        assert_true(x.grad().shape() == Shape(3))
        assert_true(x.grad().all_close[atol=1e-6](
            Tensor[dtype].d1([-1.0, -0.25, -0.0625])
        ))


fn test_recip_gpu_backward_2d() raises:
    comptime if has_accelerator():
        print("test_recip_gpu_backward_2d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0], [4.0, 8.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=True]()
        var loss = r.sum()
        loss.backward()
        assert_true(x.grad().shape() == Shape(2, 2))
        assert_true(x.grad().all_close[atol=1e-6](
            Tensor[dtype].d2([[-1.0, -0.25], [-0.0625, -0.015625]])
        ))


fn test_recip_gpu_backward_3d() raises:
    comptime if has_accelerator():
        print("test_recip_gpu_backward_3d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d3(
            [[[1.0, 2.0], [4.0, 8.0]], [[1.0, 2.0], [4.0, 8.0]]],
            requires_grad=True,
        )
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=True]()
        var loss = r.sum()
        loss.backward()
        assert_true(x.grad().shape() == Shape(2, 2, 2))
        assert_true(x.grad().all_close[atol=1e-6](
            Tensor[dtype].d3(
                [[[-1.0, -0.25], [-0.0625, -0.015625]],
                 [[-1.0, -0.25], [-0.0625, -0.015625]]]
            )
        ))


fn test_recip_gpu_grad_flow_chain() raises:
    comptime if has_accelerator():
        print("test_recip_gpu_grad_flow_chain")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=True]()
        var scaled = r * Tensor[dtype].scalar(2.0).to_gpu()
        var loss = scaled.sum()
        loss.backward()
        assert_true(x.grad().all_close[atol=1e-6](
            Tensor[dtype].d1([-2.0, -0.5, -0.125])
        ))


fn test_recip_gpu_grad_flow_double_reciprocal() raises:
    comptime if has_accelerator():
        print("test_recip_gpu_grad_flow_double_reciprocal")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var r1 = x_gpu.reciprocal[track_grad=True]()
        var r2 = r1.reciprocal[track_grad=True]()
        var loss = r2.sum()
        loss.backward()
        assert_true(x.grad().all_close[atol=1e-5](Tensor.ones_like(x)))


fn test_recip_gpu_vs_cpu_consistency() raises:
    comptime if has_accelerator():
        print("test_recip_gpu_vs_cpu_consistency")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 4.0], [0.5, 0.25, 8.0]], requires_grad=True
        )
        var x_gpu = Tensor[dtype].d2(
            [[1.0, 2.0, 4.0], [0.5, 0.25, 8.0]], requires_grad=True
        ).to_gpu()
        var r_cpu = x_cpu.reciprocal[track_grad=True]()
        var r_gpu = x_gpu.reciprocal[track_grad=True]()
        assert_true(r_cpu.all_close[atol=1e-6](r_gpu.to_cpu()))
        var loss_cpu = r_cpu.sum()
        loss_cpu.backward()
        var loss_gpu = r_gpu.sum()
        loss_gpu.backward()
        assert_true(x_cpu.grad().all_close[atol=1e-6](x_gpu.grad().to_cpu()))


fn test_recip_gpu_negative_values() raises:
    comptime if has_accelerator():
        print("test_recip_gpu_negative_values")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([-1.0, -2.0, -4.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=True]()
        assert_true(r.to_cpu().all_close[atol=1e-6](
            Tensor[dtype].d1([-1.0, -0.5, -0.25])
        ))
        var loss = r.sum()
        loss.backward()
        assert_true(x.grad().all_close[atol=1e-6](
            Tensor[dtype].d1([-1.0, -0.25, -0.0625])
        ))

fn test_recip_gpu_grad_accumulation() raises:
    comptime if has_accelerator():
        print("test_recip_gpu_grad_accumulation")
        comptime dtype = DType.float32
        var x1 = Tensor[dtype].d1([2.0, 4.0], requires_grad=True)
        var x2 = Tensor[dtype].d1([2.0, 4.0], requires_grad=True)
        var x1_gpu = x1.to_gpu()
        var x2_gpu = x2.to_gpu()
        var r1 = x1_gpu.reciprocal[track_grad=True]()
        var r2 = x2_gpu.reciprocal[track_grad=True]()
        var loss1 = r1.sum()
        loss1.backward()
        var loss2 = r2.sum()
        loss2.backward()
        # x1 and x2 each get one backward — same grad
        assert_true(x1.grad().all_close[atol=1e-6](
            Tensor[dtype].d1([-0.25, -0.0625])
        ))
        assert_true(x2.grad().all_close[atol=1e-6](
            Tensor[dtype].d1([-0.25, -0.0625])
        ))

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
