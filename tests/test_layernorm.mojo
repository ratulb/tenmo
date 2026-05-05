from std.testing import assert_true, TestSuite
from tenmo.tensor import Tensor
from tenmo.layernorm import *
from std.sys import has_accelerator
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from std.math import sqrt



fn test_layernorm_cpu_forward_simple() raises:
    print("test_layernorm_cpu_forward_simple")
    comptime dtype = DType.float32
    # x = [[1,2,3,4]], mean=2.5, var=1.25, std=sqrt(1.25)≈1.118
    # x_hat = (x-2.5)/1.118 ≈ [-1.342,-0.447,0.447,1.342]
    # gamma=ones, beta=zeros → out = x_hat
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]])
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    # mean of out should be ~0, std ~1
    var out_mean = out.mean[track_grad=False]()
    var out_std = out.std[track_grad=False](unbiased=False)
    assert_true(out_mean.all_close[atol=1e-5](Tensor[dtype].scalar(0.0)))
    assert_true(out_std.all_close[atol=1e-4](Tensor[dtype].scalar(1.0)))

fn test_layernorm_cpu_forward_gamma_beta() raises:
    print("test_layernorm_cpu_forward_gamma_beta")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]])
    # gamma=2, beta=1 → out = 2*x_hat + 1
    var gamma = Tensor[dtype].full(Shape(4), Scalar[dtype](2.0))
    var beta = Tensor[dtype].full(Shape(4), Scalar[dtype](1.0))
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    # mean should be 1.0 (beta), std should be 2.0 (gamma)
    var out_mean = out.mean[track_grad=False]()
    var out_std = out.std[track_grad=False](unbiased=False)
    assert_true(out_mean.all_close[atol=1e-4](Tensor[dtype].scalar(1.0)))
    assert_true(out_std.all_close[atol=1e-4](Tensor[dtype].scalar(2.0)))


fn test_layernorm_cpu_backward_dgamma_dbeta() raises:
    print("test_layernorm_cpu_backward_dgamma_dbeta")
    comptime dtype = DType.float32
    # Simple case: batch=1, D=4
    # d_beta = sum(upstream) over batch = upstream (batch=1)
    # d_gamma = sum(upstream * x_hat) over batch
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]])
    var gamma = Tensor[dtype].ones(Shape(4), requires_grad=True)
    var beta = Tensor[dtype].zeros(Shape(4), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # d_beta = upstream = ones(4) since loss=sum
    assert_true(beta.grad().all_close[atol=1e-5](Tensor[dtype].ones(Shape(4))))
    # d_gamma = x_hat (since upstream=ones)
    # x_hat sums to 0, so verify shape and sum
    assert_true(gamma.grad().shape() == Shape(4))
    var gamma_grad_sum = gamma.grad().sum()
    assert_true(gamma_grad_sum.all_close[atol=1e-5](Tensor[dtype].scalar(0.0)))


fn test_layernorm_cpu_backward_dx() raises:
    print("test_layernorm_cpu_backward_dx")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0]], requires_grad=True
    )
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # dx should sum to ~0 (layernorm grad property)
    var dx_sum = x.grad().sum()
    assert_true(dx_sum.all_close[atol=1e-5](Tensor[dtype].scalar(0.0)))
    assert_true(x.grad().shape() == Shape(1, 4))


fn test_layernorm_cpu_layer_wrapper() raises:
    print("test_layernorm_cpu_layer_wrapper")
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](4)  # normalized_shape=4
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]])
    var out = ln(x)
    assert_true(out.shape() == Shape(1, 4))
    # gamma=ones, beta=zeros → same as forward_simple
    var out_mean = out.mean[track_grad=False]()
    assert_true(out_mean.all_close[atol=1e-5](Tensor[dtype].scalar(0.0)))


# =============================================================================
# FORWARD TESTS
# =============================================================================


fn test_layernorm_cpu_fwd_1x4_output() raises:
    print("test_layernorm_cpu_fwd_1x4_output")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]])
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    assert_true(out.shape() == Shape(1, 4))
    assert_true(out.all_close[atol=1e-4](
        Tensor[dtype].d2([[-1.3416, -0.4472, 0.4472, 1.3416]])
    ))


fn test_layernorm_cpu_fwd_2x4_output() raises:
    print("test_layernorm_cpu_fwd_2x4_output")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    assert_true(out.shape() == Shape(2, 4))
    assert_true(out.all_close[atol=1e-4](
        Tensor[dtype].d2([
            [-1.3416, -0.4472, 0.4472, 1.3416],
            [-1.3416, -0.4472, 0.4472, 1.3416],
        ])
    ))

fn test_layernorm_cpu_fwd_3d_output() raises:
    print("test_layernorm_cpu_fwd_3d_output")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3([
        [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0]],
        [[2.0,4.0,6.0,8.0],[1.0,3.0,5.0,7.0],[10.0,20.0,30.0,40.0]],
    ])
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    assert_true(out.shape() == Shape(2, 3, 4))
    # Contiguous expected tensor — all rows normalize to same pattern
    var expected = Tensor[dtype].d3([
        [[-1.3416,-0.4472,0.4472,1.3416],
         [-1.3416,-0.4472,0.4472,1.3416],
         [-1.3416,-0.4472,0.4472,1.3416]],
        [[-1.3416,-0.4472,0.4472,1.3416],
         [-1.3416,-0.4472,0.4472,1.3416],
         [-1.3416,-0.4472,0.4472,1.3416]],
    ])
    assert_true(out.all_close[atol=1e-4](expected))


fn test_layernorm_cpu_fwd_gamma_beta_effect() raises:
    print("test_layernorm_cpu_fwd_gamma_beta_effect")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]])
    var gamma = Tensor[dtype].full(Shape(4), Scalar[dtype](2.0))
    var beta = Tensor[dtype].full(Shape(4), Scalar[dtype](1.0))
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    # out = 2 * x_hat + 1
    assert_true(out.all_close[atol=1e-4](
        Tensor[dtype].d2([[-1.6833, 0.1056, 1.8944, 3.6833]])
    ))


# =============================================================================
# BACKWARD TESTS — dx
# =============================================================================


fn test_layernorm_cpu_bwd_dx_1x4() raises:
    print("test_layernorm_cpu_bwd_dx_1x4")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # PyTorch: dx = [[0,0,0,0]]
    assert_true(x.grad().shape() == Shape(1, 4))
    assert_true(x.grad().all_close[atol=1e-5](
        Tensor[dtype].d2([[0.0, 0.0, 0.0, 0.0]])
    ))


fn test_layernorm_cpu_bwd_dx_2x4() raises:
    print("test_layernorm_cpu_bwd_dx_2x4")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], requires_grad=True
    )
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # PyTorch: dx2 = all zeros
    assert_true(x.grad().shape() == Shape(2, 4))
    assert_true(x.grad().all_close[atol=1e-5](
        Tensor[dtype].d2([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
    ))


fn test_layernorm_cpu_bwd_dx_3d() raises:
    print("test_layernorm_cpu_bwd_dx_3d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3([
        [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0]],
        [[2.0,4.0,6.0,8.0],[1.0,3.0,5.0,7.0],[10.0,20.0,30.0,40.0]],
    ], requires_grad=True)
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # PyTorch: dx3 = all zeros
    assert_true(x.grad().shape() == Shape(2, 3, 4))
    assert_true(x.grad().all_close[atol=1e-5](
        Tensor[dtype].zeros(Shape(2, 3, 4))
    ))


# =============================================================================
# BACKWARD TESTS — d_gamma, d_beta
# =============================================================================


fn test_layernorm_cpu_bwd_dgamma_dbeta_1x4() raises:
    print("test_layernorm_cpu_bwd_dgamma_dbeta_1x4")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]])
    var gamma = Tensor[dtype].ones(Shape(4), requires_grad=True)
    var beta = Tensor[dtype].zeros(Shape(4), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # PyTorch: d_gamma = [-1.3416,-0.4472,0.4472,1.3416]
    assert_true(gamma.grad().shape() == Shape(4))
    assert_true(gamma.grad().all_close[atol=1e-4](
        Tensor[dtype].d1([-1.3416, -0.4472, 0.4472, 1.3416])
    ))
    # PyTorch: d_beta = [1,1,1,1]
    assert_true(beta.grad().shape() == Shape(4))
    assert_true(beta.grad().all_close[atol=1e-5](
        Tensor[dtype].ones(Shape(4))
    ))


fn test_layernorm_cpu_bwd_dgamma_dbeta_2x4() raises:
    print("test_layernorm_cpu_bwd_dgamma_dbeta_2x4")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0,2.0,3.0,4.0],[2.0,4.0,6.0,8.0]])
    var gamma = Tensor[dtype].ones(Shape(4), requires_grad=True)
    var beta = Tensor[dtype].zeros(Shape(4), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # PyTorch: d_gamma2 = [-2.6833,-0.8944,0.8944,2.6833]
    assert_true(gamma.grad().all_close[atol=1e-4](
        Tensor[dtype].d1([-2.6833, -0.8944, 0.8944, 2.6833])
    ))
    # PyTorch: d_beta2 = [2,2,2,2]
    assert_true(beta.grad().all_close[atol=1e-5](
        Tensor[dtype].full(Shape(4), Scalar[dtype](2.0))
    ))

fn test_layernorm_cpu_bwd_dgamma_dbeta_3d() raises:
    print("test_layernorm_cpu_bwd_dgamma_dbeta_3d")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3([
        [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0]],
        [[2.0,4.0,6.0,8.0],[1.0,3.0,5.0,7.0],[10.0,20.0,30.0,40.0]],
    ])
    var gamma = Tensor[dtype].ones(Shape(4), requires_grad=True)
    var beta = Tensor[dtype].zeros(Shape(4), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # PyTorch: d_gamma3 = [-8.0498,-2.6833,2.6833,8.0498]
    assert_true(gamma.grad().all_close[atol=1e-4](
        Tensor[dtype].d1([-8.0498, -2.6833, 2.6833, 8.0498])
    ))
    # PyTorch: d_beta3 = [6,6,6,6]
    assert_true(beta.grad().all_close[atol=1e-5](
        Tensor[dtype].full(Shape(4), Scalar[dtype](6.0))
    ))


# =============================================================================
# GRAD FLOW — x requires_grad, gamma/beta fixed
# =============================================================================


fn test_layernorm_cpu_grad_flow_x_only() raises:
    print("test_layernorm_cpu_grad_flow_x_only")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0,2.0,3.0,4.0]], requires_grad=True)
    var gamma = Tensor[dtype].ones(Shape(4))   # no requires_grad
    var beta = Tensor[dtype].zeros(Shape(4))   # no requires_grad
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    assert_true(x.grad().shape() == Shape(1, 4))
    assert_true(x.grad().all_close[atol=1e-5](
        Tensor[dtype].d2([[0.0, 0.0, 0.0, 0.0]])
    ))


fn test_layernorm_cpu_grad_flow_all_params() raises:
    print("test_layernorm_cpu_grad_flow_all_params")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1.0,2.0,3.0,4.0],[2.0,4.0,6.0,8.0]], requires_grad=True
    )
    var gamma = Tensor[dtype].ones(Shape(4), requires_grad=True)
    var beta = Tensor[dtype].zeros(Shape(4), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # All three should have grads
    assert_true(x.grad().shape() == Shape(2, 4))
    assert_true(gamma.grad().shape() == Shape(4))
    assert_true(beta.grad().shape() == Shape(4))


fn test_layernorm_cpu_layer_params() raises:
    print("test_layernorm_cpu_layer_params")
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](4)
    assert_true(ln.num_parameters() == 8)  # 4 gamma + 4 beta
    var params = ln.parameters()
    assert_true(len(params) == 2)


fn test_layernorm_cpu_eval_no_grad() raises:
    print("test_layernorm_cpu_eval_no_grad")
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](4)
    ln.eval()
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]])
    var out = ln(x)
    assert_true(not out.requires_grad)


fn test_layernorm_cpu_train_has_grad() raises:
    print("test_layernorm_cpu_train_has_grad")
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](4)
    ln.train()
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    var out = ln(x)
    assert_true(out.requires_grad)


fn test_layernorm_gpu_fwd_1x4_output() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_fwd_1x4_output")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]]).to_gpu()
        var gamma = Tensor[dtype].ones(Shape(4)).to_gpu()
        var beta = Tensor[dtype].zeros(Shape(4)).to_gpu()
        var out = LayerNormForward[dtype].forward[track_grad=False](
            x, gamma, beta
        )
        assert_true(out.shape() == Shape(1, 4))
        assert_true(out.to_cpu().all_close[atol=1e-4](
            Tensor[dtype].d2([[-1.3416, -0.4472, 0.4472, 1.3416]])
        ))


fn test_layernorm_gpu_fwd_2x4_output() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_fwd_2x4_output")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2(
            [[1.0,2.0,3.0,4.0],[2.0,4.0,6.0,8.0]]
        ).to_gpu()
        var gamma = Tensor[dtype].ones(Shape(4)).to_gpu()
        var beta = Tensor[dtype].zeros(Shape(4)).to_gpu()
        var out = LayerNormForward[dtype].forward[track_grad=False](
            x, gamma, beta
        )
        assert_true(out.shape() == Shape(2, 4))
        assert_true(out.to_cpu().all_close[atol=1e-4](
            Tensor[dtype].d2([
                [-1.3416, -0.4472, 0.4472, 1.3416],
                [-1.3416, -0.4472, 0.4472, 1.3416],
            ])
        ))


fn test_layernorm_gpu_fwd_3d_output() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_fwd_3d_output")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d3([
            [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0]],
            [[2.0,4.0,6.0,8.0],[1.0,3.0,5.0,7.0],[10.0,20.0,30.0,40.0]],
        ]).to_gpu()
        var gamma = Tensor[dtype].ones(Shape(4)).to_gpu()
        var beta = Tensor[dtype].zeros(Shape(4)).to_gpu()
        var out = LayerNormForward[dtype].forward[track_grad=False](
            x, gamma, beta
        )
        assert_true(out.shape() == Shape(2, 3, 4))
        var expected = Tensor[dtype].d3([
            [[-1.3416,-0.4472,0.4472,1.3416],
             [-1.3416,-0.4472,0.4472,1.3416],
             [-1.3416,-0.4472,0.4472,1.3416]],
            [[-1.3416,-0.4472,0.4472,1.3416],
             [-1.3416,-0.4472,0.4472,1.3416],
             [-1.3416,-0.4472,0.4472,1.3416]],
        ])
        assert_true(out.to_cpu().all_close[atol=1e-4](expected))


fn test_layernorm_gpu_fwd_gamma_beta_effect() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_fwd_gamma_beta_effect")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]]).to_gpu()
        var gamma = Tensor[dtype].full(Shape(4), Scalar[dtype](2.0)).to_gpu()
        var beta = Tensor[dtype].full(Shape(4), Scalar[dtype](1.0)).to_gpu()
        var out = LayerNormForward[dtype].forward[track_grad=False](
            x, gamma, beta
        )
        assert_true(out.to_cpu().all_close[atol=1e-4](
            Tensor[dtype].d2([[-1.6833, 0.1056, 1.8944, 3.6833]])
        ))


# =============================================================================
# GPU BACKWARD TESTS — dx
# =============================================================================


fn test_layernorm_gpu_bwd_dx_1x4() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_bwd_dx_1x4")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var gamma = Tensor[dtype].ones(Shape(4)).to_gpu()
        var beta = Tensor[dtype].zeros(Shape(4)).to_gpu()
        var out = LayerNormForward[dtype].forward[track_grad=True](
            x_gpu, gamma, beta
        )
        var loss = out.sum()
        loss.backward()
        assert_true(x.grad().shape() == Shape(1, 4))
        assert_true(x.grad().all_close[atol=1e-5](
            Tensor[dtype].d2([[0.0, 0.0, 0.0, 0.0]])
        ))


fn test_layernorm_gpu_bwd_dx_2x4() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_bwd_dx_2x4")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2(
            [[1.0,2.0,3.0,4.0],[2.0,4.0,6.0,8.0]], requires_grad=True
        )
        var x_gpu = x.to_gpu()
        var gamma = Tensor[dtype].ones(Shape(4)).to_gpu()
        var beta = Tensor[dtype].zeros(Shape(4)).to_gpu()
        var out = LayerNormForward[dtype].forward[track_grad=True](
            x_gpu, gamma, beta
        )
        var loss = out.sum()
        loss.backward()
        assert_true(x.grad().shape() == Shape(2, 4))
        assert_true(x.grad().all_close[atol=1e-5](
            Tensor[dtype].d2([[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
        ))


fn test_layernorm_gpu_bwd_dx_3d() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_bwd_dx_3d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d3([
            [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0]],
            [[2.0,4.0,6.0,8.0],[1.0,3.0,5.0,7.0],[10.0,20.0,30.0,40.0]],
        ], requires_grad=True)
        var x_gpu = x.to_gpu()
        var gamma = Tensor[dtype].ones(Shape(4)).to_gpu()
        var beta = Tensor[dtype].zeros(Shape(4)).to_gpu()
        var out = LayerNormForward[dtype].forward[track_grad=True](
            x_gpu, gamma, beta
        )
        var loss = out.sum()
        loss.backward()
        assert_true(x.grad().shape() == Shape(2, 3, 4))
        assert_true(x.grad().all_close[atol=1e-5](
            Tensor[dtype].zeros(Shape(2, 3, 4))
        ))

# =============================================================================
# GPU BACKWARD TESTS — d_gamma, d_beta
# =============================================================================


fn test_layernorm_gpu_bwd_dgamma_dbeta_1x4() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_bwd_dgamma_dbeta_1x4")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]]).to_gpu()
        var gamma = Tensor[dtype].ones(Shape(4), requires_grad=True)
        var gamma_gpu = gamma.to_gpu()
        var beta = Tensor[dtype].zeros(Shape(4), requires_grad=True)
        var beta_gpu = beta.to_gpu()
        var out = LayerNormForward[dtype].forward[track_grad=True](
            x, gamma_gpu, beta_gpu
        )
        var loss = out.sum()
        loss.backward()
        assert_true(gamma.grad().shape() == Shape(4))
        assert_true(gamma.grad().all_close[atol=1e-4](
            Tensor[dtype].d1([-1.3416, -0.4472, 0.4472, 1.3416])
        ))
        assert_true(beta.grad().shape() == Shape(4))
        assert_true(beta.grad().all_close[atol=1e-5](
            Tensor[dtype].ones(Shape(4))
        ))


fn test_layernorm_gpu_bwd_dgamma_dbeta_2x4() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_bwd_dgamma_dbeta_2x4")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2(
            [[1.0,2.0,3.0,4.0],[2.0,4.0,6.0,8.0]]
        ).to_gpu()
        var gamma = Tensor[dtype].ones(Shape(4), requires_grad=True)
        var gamma_gpu = gamma.to_gpu()
        var beta = Tensor[dtype].zeros(Shape(4), requires_grad=True)
        var beta_gpu = beta.to_gpu()
        var out = LayerNormForward[dtype].forward[track_grad=True](
            x, gamma_gpu, beta_gpu
        )
        var loss = out.sum()
        loss.backward()
        assert_true(gamma.grad().all_close[atol=1e-4](
            Tensor[dtype].d1([-2.6833, -0.8944, 0.8944, 2.6833])
        ))
        assert_true(beta.grad().all_close[atol=1e-5](
            Tensor[dtype].full(Shape(4), Scalar[dtype](2.0))
        ))


fn test_layernorm_gpu_bwd_dgamma_dbeta_3d() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_bwd_dgamma_dbeta_3d")
        comptime dtype = DType.float32
        var x = Tensor[dtype].d3([
            [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0]],
            [[2.0,4.0,6.0,8.0],[1.0,3.0,5.0,7.0],[10.0,20.0,30.0,40.0]],
        ]).to_gpu()
        var gamma = Tensor[dtype].ones(Shape(4), requires_grad=True)
        var gamma_gpu = gamma.to_gpu()
        var beta = Tensor[dtype].zeros(Shape(4), requires_grad=True)
        var beta_gpu = beta.to_gpu()
        var out = LayerNormForward[dtype].forward[track_grad=True](
            x, gamma_gpu, beta_gpu
        )
        var loss = out.sum()
        loss.backward()
        assert_true(gamma.grad().all_close[atol=1e-4](
            Tensor[dtype].d1([-8.0498, -2.6833, 2.6833, 8.0498])
        ))
        assert_true(beta.grad().all_close[atol=1e-5](
            Tensor[dtype].full(Shape(4), Scalar[dtype](6.0))
        ))


# =============================================================================
# GPU vs CPU CONSISTENCY
# =============================================================================


fn test_layernorm_gpu_vs_cpu_fwd_consistency() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_vs_cpu_fwd_consistency")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2([[1.0,2.0,3.0,4.0],[2.0,4.0,6.0,8.0]])
        var x_gpu = x_cpu.to_gpu()
        var gamma_cpu = Tensor[dtype].ones(Shape(4))
        var gamma_gpu = gamma_cpu.to_gpu()
        var beta_cpu = Tensor[dtype].zeros(Shape(4))
        var beta_gpu = beta_cpu.to_gpu()
        var out_cpu = LayerNormForward[dtype].forward[track_grad=False](
            x_cpu, gamma_cpu, beta_cpu
        )
        var out_gpu = LayerNormForward[dtype].forward[track_grad=False](
            x_gpu, gamma_gpu, beta_gpu
        )
        assert_true(out_cpu.all_close[atol=1e-5](out_gpu.to_cpu()))

fn test_layernorm_gpu_vs_cpu_bwd_consistency() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_vs_cpu_bwd_consistency")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2(
            [[1.0,2.0,3.0,4.0],[2.0,4.0,6.0,8.0]], requires_grad=True
        )
        var x_gpu = Tensor[dtype].d2(
            [[1.0,2.0,3.0,4.0],[2.0,4.0,6.0,8.0]], requires_grad=True
        ).to_gpu()
        var gamma_cpu = Tensor[dtype].ones(Shape(4), requires_grad=True)
        var gamma_gpu = Tensor[dtype].ones(Shape(4), requires_grad=True)
        var gamma_gpu_t = gamma_gpu.to_gpu()
        var beta_cpu = Tensor[dtype].zeros(Shape(4), requires_grad=True)
        var beta_gpu = Tensor[dtype].zeros(Shape(4), requires_grad=True)
        var beta_gpu_t = beta_gpu.to_gpu()
        var out_cpu = LayerNormForward[dtype].forward[track_grad=True](
            x_cpu, gamma_cpu, beta_cpu
        )
        var out_gpu = LayerNormForward[dtype].forward[track_grad=True](
            x_gpu, gamma_gpu_t, beta_gpu_t
        )
        var loss_cpu = out_cpu.sum()
        loss_cpu.backward()
        var loss_gpu = out_gpu.sum()
        loss_gpu.backward()
        assert_true(x_cpu.grad().all_close[atol=1e-5](x_gpu.grad().to_cpu()))
        assert_true(gamma_cpu.grad().all_close[atol=1e-4](gamma_gpu.grad().to_cpu()))
        assert_true(beta_cpu.grad().all_close[atol=1e-5](beta_gpu.grad().to_cpu()))


# =============================================================================
# GPU LAYER WRAPPER
# =============================================================================


fn test_layernorm_gpu_layer_wrapper_fwd() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_layer_wrapper_fwd")
        comptime dtype = DType.float32
        var ln = LayerNorm[dtype](4)
        var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]]).to_gpu()
        var ln_gpu = ln.to_gpu()
        var out = ln_gpu(x)
        assert_true(out.shape() == Shape(1, 4))
        assert_true(out.to_cpu().all_close[atol=1e-4](
            Tensor[dtype].d2([[-1.3416, -0.4472, 0.4472, 1.3416]])
        ))


fn test_layernorm_gpu_layer_wrapper_bwd() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_layer_wrapper_bwd")
        comptime dtype = DType.float32
        var ln_gpu = LayerNorm[dtype](4).to_gpu()
        var x = Tensor[dtype].d2(
            [[1.0,2.0,3.0,4.0]], requires_grad=True
        ).to_gpu()
        var out = ln_gpu(x)
        var loss = out.sum()
        loss.backward()
        assert_true(ln_gpu.gamma.grad().shape() == Shape(4))
        assert_true(ln_gpu.beta.grad().shape() == Shape(4))
        # d_beta = ones(4), d_gamma = x_hat
        assert_true(ln_gpu.beta.grad().all_close[atol=1e-5](
            Tensor[dtype].ones(Shape(4)).to_gpu()
        ))
        assert_true(ln_gpu.gamma.grad().all_close[atol=1e-4](
            Tensor[dtype].d1([-1.3416, -0.4472, 0.4472, 1.3416]).to_gpu()
        ))

fn test_layernorm_gpu_eval_no_grad() raises:
    comptime if has_accelerator():
        print("test_layernorm_gpu_eval_no_grad")
        comptime dtype = DType.float32
        var ln = LayerNorm[dtype](4)
        ln.eval()
        var ln_gpu = ln.to_gpu()
        var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]]).to_gpu()
        var out = ln_gpu(x)
        assert_true(not out.requires_grad)



# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

fn layernorm_ref[dtype: DType](
    x: Tensor[dtype],
    gamma: Tensor[dtype],
    beta: Tensor[dtype],
    eps: Scalar[dtype] = Scalar[dtype](1e-5),
) -> Tensor[dtype]:
    """Naive reference LayerNorm over last dim for small tensors."""
    var out = Tensor[dtype].zeros_like(x)
    var D = x.shape()[-1]
    var n_outer = x.numels() // D
    for i in range(n_outer):
        # compute mean over last dim for this slice
        var mean = Scalar[dtype](0)
        for d in range(D):
            mean += x.get(i * D + d)
        mean /= Scalar[dtype](D)
        # compute var
        var var_ = Scalar[dtype](0)
        for d in range(D):
            var diff = x.get(i * D + d) - mean
            var_ += diff * diff
        var_ /= Scalar[dtype](D)
        var rstd = Scalar[dtype](1) / sqrt(var_ + eps)
        for d in range(D):
            var x_hat = (x.get(i * D + d) - mean) * rstd
            out.set(i * D + d, gamma.get(d) * x_hat + beta.get(d))
    return out^


# ═════════════════════════════════════════════════════════════════════════════
# FORWARD — CPU
# ═════════════════════════════════════════════════════════════════════════════

fn test_layernorm_fwd_cpu_1d_identity_gamma_beta() raises:
    print("test_layernorm_fwd_cpu_1d_identity_gamma_beta")
    comptime dtype = DType.float32
    # gamma=ones, beta=zeros => output is just x_hat
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var ln = LayerNorm[dtype](5)
    var out = ln(x)
    # mean=3, var=2, std=sqrt(2), x_hat=[-1.414,-0.707,0,0.707,1.414]
    assert_true(out.shape() == Shape(5))
    assert_true(out.mean[track_grad=False]().all_close[atol=1e-5](
        Tensor[dtype].scalar(0.0)
    ))


fn test_layernorm_fwd_cpu_1d_matches_ref() raises:
    print("test_layernorm_fwd_cpu_1d_matches_ref")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var gamma = Tensor[dtype].d1([2.0, 1.0, 0.5, 1.0, 2.0])
    var beta  = Tensor[dtype].d1([0.1, 0.2, 0.3, 0.4, 0.5])
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    var ref_ = layernorm_ref(x, gamma, beta)
    assert_true(out.all_close[atol=1e-5](ref_))


fn test_layernorm_fwd_cpu_2d_shape() raises:
    print("test_layernorm_fwd_cpu_2d_shape")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var ln = LayerNorm[dtype](3)
    var out = ln(x)
    assert_true(out.shape() == Shape(2, 3))


fn test_layernorm_fwd_cpu_2d_matches_ref() raises:
    print("test_layernorm_fwd_cpu_2d_matches_ref")
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var gamma = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var beta  = Tensor[dtype].d1([0.1, 0.1, 0.1])
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    var ref_ = layernorm_ref(x, gamma, beta)
    assert_true(out.all_close[atol=1e-5](ref_))


fn test_layernorm_fwd_cpu_3d_shape() raises:
    print("test_layernorm_fwd_cpu_3d_shape")
    comptime dtype = DType.float32
    # Transformer-like: (B=2, T=4, D=8)
    var x = Tensor[dtype].randn(Shape(2, 4, 8), mean=0.0, std=1.0)
    var ln = LayerNorm[dtype](8)
    var out = ln(x)
    assert_true(out.shape() == Shape(2, 4, 8))


fn test_layernorm_fwd_cpu_3d_matches_ref() raises:
    print("test_layernorm_fwd_cpu_3d_matches_ref")
    comptime dtype = DType.float32
    var x = Tensor[dtype].arange(1.0, 25.0).reshape(2, 3, 4)
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta  = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    var ref_ = layernorm_ref(x, gamma, beta)
    assert_true(out.all_close[atol=1e-4](ref_))


fn test_layernorm_fwd_cpu_output_mean_near_zero() raises:
    # With gamma=ones, beta=zeros each row should have mean~0
    print("test_layernorm_fwd_cpu_output_mean_near_zero")
    comptime dtype = DType.float32
    var x = Tensor[dtype].randn(Shape(4, 8), mean=5.0, std=3.0)
    var ln = LayerNorm[dtype](8)
    var out = ln(x)
    # mean over last dim for each row should be ~0
    var row_means = out.mean[track_grad=False](axes=[-1])
    assert_true(row_means.all_close[atol=1e-5](Tensor[dtype].zeros_like(row_means)))


fn test_layernorm_fwd_cpu_output_std_near_one() raises:
    # With gamma=ones, beta=zeros each row should have std~1
    print("test_layernorm_fwd_cpu_output_std_near_one")
    comptime dtype = DType.float32
    var x = Tensor[dtype].randn(Shape(4, 8), mean=5.0, std=3.0)
    var ln = LayerNorm[dtype](8)
    var out = ln(x)
    var row_vars = out.variance[track_grad=False](axis=-1, unbiased=False)
    assert_true(row_vars.all_close[atol=1e-4](Tensor[dtype].ones_like(row_vars)))


fn test_layernorm_fwd_cpu_constant_input() raises:
    # Constant input — var=0, eps saves from division by zero
    # output should be beta (since x_hat=0)
    print("test_layernorm_fwd_cpu_constant_input")
    comptime dtype = DType.float32
    var x     = Tensor[dtype].full(Shape(3, 4), 7.0)
    var gamma = Tensor[dtype].full(Shape(4), 2.0)
    var beta  = Tensor[dtype].full(Shape(4), 0.5)
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    # x_hat = 0 everywhere => out = gamma*0 + beta = beta
    assert_true(out.all_close[atol=1e-5](
        Tensor[dtype].full(Shape(3, 4), 0.5)
    ))


fn test_layernorm_fwd_cpu_eval_mode_no_grad() raises:
    print("test_layernorm_fwd_cpu_eval_mode_no_grad")
    comptime dtype = DType.float32
    var x = Tensor[dtype].randn(Shape(2, 4), mean=0.0, std=1.0)
    var ln = LayerNorm[dtype](4)
    ln.eval()
    var out = ln(x)
    assert_true(not out.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# BACKWARD — CPU
# ═════════════════════════════════════════════════════════════════════════════

fn test_layernorm_bwd_cpu_gamma_grad_shape() raises:
    print("test_layernorm_bwd_cpu_gamma_grad_shape")
    comptime dtype = DType.float32
    var x     = Tensor[dtype].randn(Shape(2, 4, 8), mean=0.0, std=1.0, requires_grad=True)
    var gamma = Tensor[dtype].ones(Shape(8),  requires_grad=True)
    var beta  = Tensor[dtype].zeros(Shape(8), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    assert_true(gamma.grad().shape() == Shape(8))
    assert_true(beta.grad().shape()  == Shape(8))
    assert_true(x.grad().shape()     == Shape(2, 4, 8))


fn test_layernorm_bwd_cpu_beta_grad_equals_sum_upstream() raises:
    # d_beta = sum(upstream) over all non-D dims
    # upstream = ones (from sum loss) => d_beta = B*T for each element
    print("test_layernorm_bwd_cpu_beta_grad_equals_sum_upstream")
    comptime dtype = DType.float32
    var B = 2; var T = 3; var D = 4
    var x     = Tensor[dtype].randn(Shape(B, T, D), mean=0.0, std=1.0, requires_grad=True)
    var gamma = Tensor[dtype].ones(Shape(D),  requires_grad=True)
    var beta  = Tensor[dtype].zeros(Shape(D), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # upstream is all ones => d_beta[d] = sum over B,T of 1 = B*T
    assert_true(beta.grad().all_close[atol=1e-4](
        Tensor[dtype].full(Shape(D), Float32(B * T))
    ))

fn test_layernorm_bwd_cpu_gamma_grad_value() raises:
    print("test_layernorm_bwd_cpu_gamma_grad_value")
    comptime dtype = DType.float32
    var B = 2; var T = 3; var D = 4
    var x     = Tensor[dtype].arange(1.0, 25.0).reshape(B, T, D)
    x.requires_grad_(True)
    var gamma = Tensor[dtype].ones(Shape(D),  requires_grad=True)
    var beta  = Tensor[dtype].zeros(Shape(D), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # d_gamma = sum(upstream * x_hat) over B,T
    # upstream = ones => d_gamma[d] = sum over B,T of x_hat[b,t,d]
    # Verify shape and no NaN — exact value depends on x_hat which is correct
    # if forward is correct
    assert_true(gamma.grad().shape() == Shape(D))
    assert_true(gamma.grad().sum().item() == gamma.grad().sum().item())  # no NaN

    # Cross-check: compute expected d_gamma manually
    # x_hat = (x - mean) * rstd per row — recompute reference
    var mean_t = x.mean[track_grad=False](axes=IntArray(-1), keepdims=True)
    var var_t  = x.variance[track_grad=False](axis=-1, keepdims=True, unbiased=False)
    var rstd_t = (var_t + Scalar[dtype](1e-5)).sqrt[track_grad=False]().reciprocal[track_grad=False]()
    var x_hat  = (x - mean_t) * rstd_t   # (B, T, D)
    # d_gamma = sum(x_hat) over B,T dims
    var expected_d_gamma = x_hat
    for _ax in range(x_hat.rank() - 1):
        expected_d_gamma = expected_d_gamma.sum[track_grad=False](axes=IntArray(0), keepdims=False)
    assert_true(gamma.grad().all_close[atol=1e-4](expected_d_gamma))

fn test_layernorm_bwd_cpu_dx_grad_sums_to_zero() raises:
    # The three-term formula guarantees that dx sums to zero per token
    # (gradient is orthogonal to constant and to x_hat)
    print("test_layernorm_bwd_cpu_dx_grad_sums_to_zero")
    comptime dtype = DType.float32
    var B = 2; var T = 3; var D = 8
    var x     = Tensor[dtype].randn(Shape(B, T, D), mean=0.0, std=1.0, requires_grad=True)
    var gamma = Tensor[dtype].ones(Shape(D),  requires_grad=True)
    var beta  = Tensor[dtype].zeros(Shape(D), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # sum of dx over last dim for each token should be ~0
    var dx_row_sums = x.grad().sum(axes=IntArray(-1))
    assert_true(dx_row_sums.all_close[atol=1e-4](Tensor[dtype].zeros(dx_row_sums.shape())))


fn test_layernorm_bwd_cpu_no_grad_input() raises:
    # If x has no requires_grad, only gamma and beta get grads
    print("test_layernorm_bwd_cpu_no_grad_input")
    comptime dtype = DType.float32
    var x     = Tensor[dtype].randn(Shape(2, 4), mean=0.0, std=1.0)  # no grad
    var gamma = Tensor[dtype].ones(Shape(4),  requires_grad=True)
    var beta  = Tensor[dtype].zeros(Shape(4), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    assert_true(gamma.grad().shape() == Shape(4))
    assert_true(beta.grad().shape()  == Shape(4))


fn test_layernorm_bwd_cpu_2d_dx_no_nan() raises:
    print("test_layernorm_bwd_cpu_2d_dx_no_nan")
    comptime dtype = DType.float32
    var x     = Tensor[dtype].randn(Shape(4, 8), mean=0.0, std=2.0, requires_grad=True)
    var gamma = Tensor[dtype].ones(Shape(8),  requires_grad=True)
    var beta  = Tensor[dtype].zeros(Shape(8), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    # NaN check: x == x is False only for NaN
    assert_true(x.grad().sum().item() == x.grad().sum().item())
    assert_true(gamma.grad().sum().item() == gamma.grad().sum().item())
    assert_true(beta.grad().sum().item() == beta.grad().sum().item())


fn test_layernorm_bwd_cpu_3d_dx_no_nan() raises:
    print("test_layernorm_bwd_cpu_3d_dx_no_nan")
    comptime dtype = DType.float32
    var x     = Tensor[dtype].randn(Shape(2, 4, 8), mean=0.0, std=1.0, requires_grad=True)
    var gamma = Tensor[dtype].ones(Shape(8),  requires_grad=True)
    var beta  = Tensor[dtype].zeros(Shape(8), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    assert_true(x.grad().sum().item() == x.grad().sum().item())
    assert_true(gamma.grad().sum().item() == gamma.grad().sum().item())
    assert_true(beta.grad().sum().item() == beta.grad().sum().item())


# ═════════════════════════════════════════════════════════════════════════════
# GRAD FLOW — CPU
# ═════════════════════════════════════════════════════════════════════════════

fn test_layernorm_gradflow_cpu_chained_with_linear() raises:
    # LayerNorm -> sum -> backward — grad flows through LN to x
    print("test_layernorm_gradflow_cpu_chained_with_linear")
    comptime dtype = DType.float32
    var x     = Tensor[dtype].randn(Shape(2, 4), mean=0.0, std=1.0, requires_grad=True)
    var gamma = Tensor[dtype].ones(Shape(4),  requires_grad=True)
    var beta  = Tensor[dtype].zeros(Shape(4), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var scaled = out * Tensor[dtype].full_like(out, 2.0)
    var loss = scaled.sum()
    loss.backward()
    # grad scaled by 2 — dx should be nonzero where x varies
    assert_true(x.grad().shape() == Shape(2, 4))
    assert_true(x.grad().sum().item() == x.grad().sum().item())


fn test_layernorm_gradflow_cpu_gamma_ones_beta_zeros_dx_sum_zero() raises:
    # Classic property: sum(dx) over last dim == 0 per token
    print("test_layernorm_gradflow_cpu_gamma_ones_beta_zeros_dx_sum_zero")
    comptime dtype = DType.float32
    var x     = Tensor[dtype].arange(1.0, 13.0).reshape(3, 4)
    x.requires_grad_(True)
    var gamma = Tensor[dtype].ones(Shape(4),  requires_grad=True)
    var beta  = Tensor[dtype].zeros(Shape(4), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    var dx_sums = x.grad().sum(axes=IntArray(-1))
    assert_true(dx_sums.all_close[atol=1e-4](Tensor[dtype].zeros(dx_sums.shape())))


fn test_layernorm_gradflow_cpu_no_grad_no_ancestry() raises:
    print("test_layernorm_gradflow_cpu_no_grad_no_ancestry")
    comptime dtype = DType.float32
    var x     = Tensor[dtype].randn(Shape(2, 4), mean=0.0, std=1.0)
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta  = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    assert_true(not out.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# FORWARD — GPU
# ═════════════════════════════════════════════════════════════════════════════

fn test_layernorm_fwd_gpu_1d_shape() raises:
    comptime if has_accelerator():
        print("test_layernorm_fwd_gpu_1d_shape")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
        var ln = LayerNorm[dtype](5)
        ln = ln.to_gpu()
        var out = ln(x_cpu.to_gpu())
        assert_true(out.shape() == Shape(5))


fn test_layernorm_fwd_gpu_2d_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_layernorm_fwd_gpu_2d_matches_cpu")
        comptime dtype = DType.float32
        var x     = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var gamma = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var beta  = Tensor[dtype].d1([0.1, 0.1, 0.1])
        var cpu_out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
        var gpu_out = LayerNormForward[dtype].forward[track_grad=False](
            x.to_gpu(),
            gamma.to_gpu(),
            beta.to_gpu(),
        ).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


fn test_layernorm_fwd_gpu_3d_shape() raises:
    comptime if has_accelerator():
        print("test_layernorm_fwd_gpu_3d_shape")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].randn(Shape(2, 4, 8), mean=0.0, std=1.0)
        var ln = LayerNorm[dtype](8)
        ln = ln.to_gpu()
        var out = ln(x_cpu.to_gpu())
        assert_true(out.shape() == Shape(2, 4, 8))


fn test_layernorm_fwd_gpu_output_mean_near_zero() raises:
    comptime if has_accelerator():
        print("test_layernorm_fwd_gpu_output_mean_near_zero")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].randn(Shape(4, 8), mean=5.0, std=3.0)
        var ln = LayerNorm[dtype](8)
        ln = ln.to_gpu()
        var out = ln(x_cpu.to_gpu()).to_cpu()
        var row_means = out.mean[track_grad=False](axes=[-1])
        assert_true(row_means.all_close[atol=1e-4](Tensor[dtype].zeros_like(row_means)))


fn test_layernorm_fwd_gpu_output_std_near_one() raises:
    comptime if has_accelerator():
        print("test_layernorm_fwd_gpu_output_std_near_one")
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].randn(Shape(4, 8), mean=5.0, std=3.0)
        var ln = LayerNorm[dtype](8)
        ln = ln.to_gpu()
        var out = ln(x_cpu.to_gpu()).to_cpu()
        var row_vars = out.variance[track_grad=False](axis=-1, unbiased=False)
        assert_true(row_vars.all_close[atol=1e-4](Tensor[dtype].ones_like(row_vars)))


fn test_layernorm_fwd_gpu_constant_input() raises:
    comptime if has_accelerator():
        print("test_layernorm_fwd_gpu_constant_input")
        comptime dtype = DType.float32
        var x_cpu   = Tensor[dtype].full(Shape(3, 4), 7.0)
        var gamma   = Tensor[dtype].full(Shape(4), 2.0)
        var beta    = Tensor[dtype].full(Shape(4), 0.5)
        var out = LayerNormForward[dtype].forward[track_grad=False](
            x_cpu.to_gpu(), gamma.to_gpu(), beta.to_gpu()
        ).to_cpu()
        assert_true(out.all_close[atol=1e-5](Tensor[dtype].full(Shape(3, 4), 0.5)))


# ═════════════════════════════════════════════════════════════════════════════
# BACKWARD — GPU
# ═════════════════════════════════════════════════════════════════════════════

fn test_layernorm_bwd_gpu_grad_shapes() raises:
    comptime if has_accelerator():
        print("test_layernorm_bwd_gpu_grad_shapes")
        comptime dtype = DType.float32
        var x_cpu     = Tensor[dtype].randn(Shape(2, 4, 8), mean=0.0, std=1.0, requires_grad=True)
        var gamma_cpu = Tensor[dtype].ones(Shape(8),  requires_grad=True)
        var beta_cpu  = Tensor[dtype].zeros(Shape(8), requires_grad=True)
        var out = LayerNormForward[dtype].forward[track_grad=True](
            x_cpu.to_gpu(), gamma_cpu.to_gpu(), beta_cpu.to_gpu()
        )
        var loss = out.sum()
        loss.backward()
        assert_true(x_cpu.grad().shape()     == Shape(2, 4, 8))
        assert_true(gamma_cpu.grad().shape() == Shape(8))
        assert_true(beta_cpu.grad().shape()  == Shape(8))


fn test_layernorm_bwd_gpu_beta_grad_value() raises:
    comptime if has_accelerator():
        print("test_layernorm_bwd_gpu_beta_grad_value")
        comptime dtype = DType.float32
        var B = 2; var T = 3; var D = 4
        var x_cpu     = Tensor[dtype].randn(Shape(B, T, D), mean=0.0, std=1.0, requires_grad=True)
        var gamma_cpu = Tensor[dtype].ones(Shape(D),  requires_grad=True)
        var beta_cpu  = Tensor[dtype].zeros(Shape(D), requires_grad=True)
        var out = LayerNormForward[dtype].forward[track_grad=True](
            x_cpu.to_gpu(), gamma_cpu.to_gpu(), beta_cpu.to_gpu()
        )
        var loss = out.sum()
        loss.backward()
        assert_true(beta_cpu.grad().all_close[atol=1e-4](
            Tensor[dtype].full(Shape(D), Float32(B * T))
        ))


fn test_layernorm_bwd_gpu_dx_sum_zero() raises:
    comptime if has_accelerator():
        print("test_layernorm_bwd_gpu_dx_sum_zero")
        comptime dtype = DType.float32
        var x_cpu     = Tensor[dtype].randn(Shape(2, 3, 8), mean=0.0, std=1.0, requires_grad=True)
        var gamma_cpu = Tensor[dtype].ones(Shape(8),  requires_grad=True)
        var beta_cpu  = Tensor[dtype].zeros(Shape(8), requires_grad=True)
        var out = LayerNormForward[dtype].forward[track_grad=True](
            x_cpu.to_gpu(), gamma_cpu.to_gpu(), beta_cpu.to_gpu()
        )
        var loss = out.sum()
        loss.backward()
        var dx_sums = x_cpu.grad().sum(axes=IntArray(-1))
        assert_true(dx_sums.all_close[atol=1e-4](Tensor[dtype].zeros(dx_sums.shape())))


fn test_layernorm_bwd_gpu_no_nan() raises:
    comptime if has_accelerator():
        print("test_layernorm_bwd_gpu_no_nan")
        comptime dtype = DType.float32
        var x_cpu     = Tensor[dtype].randn(Shape(2, 4, 8), mean=0.0, std=1.0, requires_grad=True)
        var gamma_cpu = Tensor[dtype].ones(Shape(8),  requires_grad=True)
        var beta_cpu  = Tensor[dtype].zeros(Shape(8), requires_grad=True)
        var out = LayerNormForward[dtype].forward[track_grad=True](
            x_cpu.to_gpu(), gamma_cpu.to_gpu(), beta_cpu.to_gpu()
        )
        var loss = out.sum()
        loss.backward()
        assert_true(x_cpu.grad().sum().item()     == x_cpu.grad().sum().item())
        assert_true(gamma_cpu.grad().sum().item() == gamma_cpu.grad().sum().item())
        assert_true(beta_cpu.grad().sum().item()  == beta_cpu.grad().sum().item())


# ═════════════════════════════════════════════════════════════════════════════
# CPU / GPU PARITY
# ═════════════════════════════════════════════════════════════════════════════

fn test_layernorm_parity_fwd_2d() raises:
    comptime if has_accelerator():
        print("test_layernorm_parity_fwd_2d")
        comptime dtype = DType.float32
        var x     = Tensor[dtype].arange(1.0, 13.0).reshape(3, 4)
        var gamma = Tensor[dtype].ones(Shape(4))
        var beta  = Tensor[dtype].zeros(Shape(4))
        var cpu_out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
        var gpu_out = LayerNormForward[dtype].forward[track_grad=False](
            x.to_gpu(), gamma.to_gpu(), beta.to_gpu()
        ).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


fn test_layernorm_parity_fwd_3d() raises:
    comptime if has_accelerator():
        print("test_layernorm_parity_fwd_3d")
        comptime dtype = DType.float32
        var x     = Tensor[dtype].randn(Shape(2, 4, 8), mean=0.0, std=1.0)
        var gamma = Tensor[dtype].ones(Shape(8))
        var beta  = Tensor[dtype].zeros(Shape(8))
        var cpu_out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
        var gpu_out = LayerNormForward[dtype].forward[track_grad=False](
            x.to_gpu(), gamma.to_gpu(), beta.to_gpu()
        ).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


fn test_layernorm_parity_bwd_beta_grad() raises:
    comptime if has_accelerator():
        print("test_layernorm_parity_bwd_beta_grad")
        comptime dtype = DType.float32
        var x = Tensor[dtype].randn(Shape(2, 3, 4), mean=0.0, std=1.0)

        var gamma_cpu = Tensor[dtype].ones(Shape(4),  requires_grad=True)
        var beta_cpu  = Tensor[dtype].zeros(Shape(4), requires_grad=True)
        var x_cpu     = x.copy()
        x_cpu.requires_grad_(True)
        var loss_cpu = LayerNormForward[dtype].forward[track_grad=True](
            x_cpu, gamma_cpu, beta_cpu
        ).sum()
        loss_cpu.backward()

        var gamma_gpu = Tensor[dtype].ones(Shape(4),  requires_grad=True)
        var beta_gpu  = Tensor[dtype].zeros(Shape(4), requires_grad=True)
        var x_gpu_leaf = x.copy()
        x_gpu_leaf.requires_grad_(True)
        var loss_gpu = LayerNormForward[dtype].forward[track_grad=True](
            x_gpu_leaf.to_gpu(), gamma_gpu.to_gpu(), beta_gpu.to_gpu()
        ).sum()
        loss_gpu.backward()

        assert_true(beta_cpu.grad().all_close[atol=1e-4](beta_gpu.grad()))
        assert_true(gamma_cpu.grad().all_close[atol=1e-4](gamma_gpu.grad()))


fn test_layernorm_parity_bwd_dx() raises:
    comptime if has_accelerator():
        print("test_layernorm_parity_bwd_dx")
        comptime dtype = DType.float32
        var x_data = Tensor[dtype].randn(Shape(2, 3, 4), mean=0.0, std=1.0)

        var x_cpu     = x_data.copy()
        x_cpu.requires_grad_(True)
        var gamma_cpu = Tensor[dtype].ones(Shape(4),  requires_grad=True)
        var beta_cpu  = Tensor[dtype].zeros(Shape(4), requires_grad=True)
        var loss_cpu  = LayerNormForward[dtype].forward[track_grad=True](
            x_cpu, gamma_cpu, beta_cpu
        ).sum()
        loss_cpu.backward()

        var x_gpu_leaf = x_data.copy()
        x_gpu_leaf.requires_grad_(True)
        var gamma_gpu = Tensor[dtype].ones(Shape(4),  requires_grad=True)
        var beta_gpu  = Tensor[dtype].zeros(Shape(4), requires_grad=True)
        var loss_gpu  = LayerNormForward[dtype].forward[track_grad=True](
            x_gpu_leaf.to_gpu(), gamma_gpu.to_gpu(), beta_gpu.to_gpu()
        ).sum()
        loss_gpu.backward()

        assert_true(x_cpu.grad().all_close[atol=1e-4](x_gpu_leaf.grad()))


# ═════════════════════════════════════════════════════════════════════════════
# LAYER WRAPPER TESTS
# ═════════════════════════════════════════════════════════════════════════════

fn test_layernorm_layer_parameters() raises:
    print("test_layernorm_layer_parameters")
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](8)
    assert_true(ln.num_parameters() == 16)   # gamma(8) + beta(8)
    assert_true(len(ln.parameters()) == 2)


fn test_layernorm_layer_train_eval_toggle() raises:
    print("test_layernorm_layer_train_eval_toggle")
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](4)
    var x = Tensor[dtype].randn(Shape(2, 4), mean=0.0, std=1.0, requires_grad=True)
    ln.train()
    var out_train = ln(x)
    assert_true(out_train.requires_grad)
    ln.eval()
    var out_eval = ln(x)
    assert_true(not out_eval.requires_grad)


fn test_layernorm_layer_gamma_ones_beta_zeros_init() raises:
    print("test_layernorm_layer_gamma_ones_beta_zeros_init")
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](4)
    assert_true(ln.gamma.all_close(Tensor[dtype].ones(Shape(4))))
    assert_true(ln.beta.all_close(Tensor[dtype].zeros(Shape(4))))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll layernorm tests passed!")

