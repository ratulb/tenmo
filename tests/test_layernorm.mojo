from std.testing import assert_true, TestSuite
from tenmo.tensor import Tensor
from tenmo.layernorm import *
from std.sys import has_accelerator
from tenmo.shapes import Shape


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

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll layernorm tests passed!")
