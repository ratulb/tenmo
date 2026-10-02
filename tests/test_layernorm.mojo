from std.testing import assert_true, TestSuite
from tenmo.tensor import Tensor
from tenmo.layernorm import *
from std.sys import has_accelerator
from tenmo.shared.shapes import Shape
from tenmo.shared.intarray import IntArray
from std.math import sqrt



def test_layernorm_cpu_forward_simple() raises:
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

def test_layernorm_cpu_forward_simple_big_parallel_path() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].arange(1, 5000).reshape(1, 4999).expand(5, 4999)
    var gamma = Tensor[dtype].ones(Shape(4999))
    var beta = Tensor[dtype].zeros(Shape(4999))
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    # mean of out should be ~0, std ~1
    var out_mean = out.mean[track_grad=False]()
    var out_std = out.std[track_grad=False](unbiased=False)
    assert_true(out_mean.all_close[atol=1e-5](Tensor[dtype].scalar(0.0)))
    assert_true(out_std.all_close[atol=1e-4](Tensor[dtype].scalar(1.0)))


def test_layernorm_cpu_forward_gamma_beta() raises:
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


def test_layernorm_cpu_backward_dgamma_dbeta() raises:
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


def test_layernorm_cpu_backward_dx() raises:
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


def test_layernorm_cpu_forward_backward_strided_input() raises:
    comptime dtype = DType.float32
    # Strided input as a transposed view: logical [[1,4,7],[2,5,8]] (2,3).
    # D=3 with non-progression rows is required: D=2 rows always normalize
    # to +-1, progression rows are affine-identical, and property asserts
    # (row mean~0/std~1) hold for wrong outputs too — only exact values pin.
    # Each row has mean 4/5, var 6, r = 1/sqrt(6+eps) -> x_hat = [-3r,0,3r].
    var base = Tensor[dtype].d2(
        [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], requires_grad=True
    )
    var x = base.transpose()  # (2,3), strided storage
    var gamma = Tensor[dtype].ones(Shape(3), requires_grad=True)
    var beta = Tensor[dtype].zeros(Shape(3), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var expected = Tensor[dtype].d2(
        [[-1.22474487, 0.0, 1.22474487], [-1.22474487, 0.0, 1.22474487]]
    )
    assert_true(out.all_close[atol=1e-5](expected))
    # Non-uniform upstream (uniform upstream gives dx=0 for ANY x_hat since
    # mean(x_hat)==0 by construction — blind to this bug by design):
    # loss = (out * w).sum(), w = [[1,0,0],[0,0,0]].
    # d_x_hat row0 = [1,0,0]: m1 = 1/3, m2 = -r -> dx row0 = [r/6,-r/3,r/6].
    # Asserted on base (the leaf): x is a view conduit — ViewBackward routes
    # grad to the parent and always clears the view's own gradbox, so
    # x.grad() reads back zeros by design; base.grad() holds dx transposed.
    var w = Tensor[dtype].d2([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    var loss = (out * w).sum()
    loss.backward()
    var expected_dx_base = Tensor[dtype].d2(
        [[0.0680414, 0.0], [-0.1360828, 0.0], [0.0680414, 0.0]]
    )
    assert_true(base.grad().all_close[atol=1e-4](expected_dx_base))
    # dgamma = sum(upstream*x_hat) over rows = [-3r, 0, 0];
    # dbeta = [1, 0, 0] (wiring sanity, x_hat-independent).
    var expected_dgamma = Tensor[dtype].d1([-1.22474487, 0.0, 0.0])
    assert_true(gamma.grad().all_close[atol=1e-4](expected_dgamma))
    var expected_dbeta = Tensor[dtype].d1([1.0, 0.0, 0.0])
    assert_true(beta.grad().all_close[atol=1e-5](expected_dbeta))
    _ = base


def test_layernorm_cpu_layer_wrapper() raises:
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


def test_layernorm_cpu_fwd_1x4_output() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]])
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    assert_true(out.shape() == Shape(1, 4))
    assert_true(out.all_close[atol=1e-4](
        Tensor[dtype].d2([[-1.3416, -0.4472, 0.4472, 1.3416]])
    ))


def test_layernorm_cpu_fwd_2x4_output() raises:
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

def test_layernorm_cpu_fwd_3d_output() raises:
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


def test_layernorm_cpu_fwd_gamma_beta_effect() raises:
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


def test_layernorm_cpu_bwd_dx_1x4() raises:
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


def test_layernorm_cpu_bwd_dx_2x4() raises:
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


def test_layernorm_cpu_bwd_dx_3d() raises:
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


def test_layernorm_cpu_bwd_dgamma_dbeta_1x4() raises:
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


def test_layernorm_cpu_bwd_dgamma_dbeta_2x4() raises:
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

def test_layernorm_cpu_bwd_dgamma_dbeta_3d() raises:
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


def test_layernorm_cpu_grad_flow_x_only() raises:
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


def test_layernorm_cpu_grad_flow_all_params() raises:
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


def test_layernorm_cpu_layer_params() raises:
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](4)
    assert_true(ln.num_parameters() == 8)  # 4 gamma + 4 beta
    var params = ln.parameters()
    assert_true(len(params) == 2)


def test_layernorm_cpu_eval_no_grad() raises:
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](4)
    ln.eval()
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]])
    var out = ln(x)
    assert_true(not out.requires_grad)


def test_layernorm_cpu_train_has_grad() raises:
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](4)
    ln.train()
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    var out = ln(x)
    assert_true(out.requires_grad)


def test_layernorm_gpu_fwd_1x4_output() raises:
    comptime if has_accelerator():
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


def test_layernorm_gpu_fwd_2x4_output() raises:
    comptime if has_accelerator():
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


def test_layernorm_gpu_fwd_3d_output() raises:
    comptime if has_accelerator():
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


def test_layernorm_gpu_fwd_gamma_beta_effect() raises:
    comptime if has_accelerator():
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


def test_layernorm_gpu_bwd_dx_1x4() raises:
    comptime if has_accelerator():
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


def test_layernorm_gpu_bwd_dx_2x4() raises:
    comptime if has_accelerator():
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


def test_layernorm_gpu_bwd_dx_3d() raises:
    comptime if has_accelerator():
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


def test_layernorm_gpu_bwd_dgamma_dbeta_1x4() raises:
    comptime if has_accelerator():
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


def test_layernorm_gpu_bwd_dgamma_dbeta_2x4() raises:
    comptime if has_accelerator():
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


def test_layernorm_gpu_bwd_dgamma_dbeta_3d() raises:
    comptime if has_accelerator():
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


def test_layernorm_gpu_vs_cpu_fwd_consistency() raises:
    comptime if has_accelerator():
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

def test_layernorm_gpu_vs_cpu_bwd_consistency() raises:
    comptime if has_accelerator():
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


def test_layernorm_gpu_layer_wrapper_fwd() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var ln = LayerNorm[dtype](4)
        var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]]).to_gpu()
        var ln_gpu = ln.to_gpu()
        var out = ln_gpu(x)
        assert_true(out.shape() == Shape(1, 4))
        assert_true(out.to_cpu().all_close[atol=1e-4](
            Tensor[dtype].d2([[-1.3416, -0.4472, 0.4472, 1.3416]])
        ))


def test_layernorm_gpu_layer_wrapper_bwd() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var ln_gpu = LayerNorm[dtype](4)
        ln_gpu = ln_gpu.to_gpu()
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

def test_layernorm_gpu_eval_no_grad() raises:
    comptime if has_accelerator():
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

def layernorm_ref[dtype: DType](
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

def test_layernorm_fwd_cpu_1d_identity_gamma_beta() raises:
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


def test_layernorm_fwd_cpu_1d_matches_ref() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var gamma = Tensor[dtype].d1([2.0, 1.0, 0.5, 1.0, 2.0])
    var beta  = Tensor[dtype].d1([0.1, 0.2, 0.3, 0.4, 0.5])
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    var ref_ = layernorm_ref(x, gamma, beta)
    assert_true(out.all_close[atol=1e-5](ref_))


def test_layernorm_fwd_cpu_2d_shape() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var ln = LayerNorm[dtype](3)
    var out = ln(x)
    assert_true(out.shape() == Shape(2, 3))


def test_layernorm_fwd_cpu_2d_matches_ref() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var gamma = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var beta  = Tensor[dtype].d1([0.1, 0.1, 0.1])
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    var ref_ = layernorm_ref(x, gamma, beta)
    assert_true(out.all_close[atol=1e-5](ref_))


def test_layernorm_fwd_cpu_3d_shape() raises:
    comptime dtype = DType.float32
    # Transformer-like: (B=2, T=4, D=8)
    var x = Tensor[dtype].randn(Shape(2, 4, 8), mean=0.0, std=1.0)
    var ln = LayerNorm[dtype](8)
    var out = ln(x)
    assert_true(out.shape() == Shape(2, 4, 8))


def test_layernorm_fwd_cpu_3d_matches_ref() raises:
    comptime dtype = DType.float32
    var _tmp0 = Tensor[dtype].arange(1.0, 25.0)
    var x = _tmp0.reshape(2, 3, 4)
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta  = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    var ref_ = layernorm_ref(x, gamma, beta)
    assert_true(out.all_close[atol=1e-4](ref_))


def test_layernorm_fwd_cpu_output_mean_near_zero() raises:
    # With gamma=ones, beta=zeros each row should have mean~0
    comptime dtype = DType.float32
    var x = Tensor[dtype].randn(Shape(4, 8), mean=5.0, std=3.0)
    var ln = LayerNorm[dtype](8)
    var out = ln(x)
    # mean over last dim for each row should be ~0
    var row_means = out.mean[track_grad=False](axes=[-1])
    assert_true(row_means.all_close[atol=1e-5](Tensor[dtype].zeros_like(row_means)))


def test_layernorm_fwd_cpu_output_std_near_one() raises:
    # With gamma=ones, beta=zeros each row should have std~1
    comptime dtype = DType.float32
    var x = Tensor[dtype].randn(Shape(4, 8), mean=5.0, std=3.0)
    var ln = LayerNorm[dtype](8)
    var out = ln(x)
    var row_vars = out.variance[track_grad=False](axis=-1, unbiased=False)
    assert_true(row_vars.all_close[atol=1e-4](Tensor[dtype].ones_like(row_vars)))


def test_layernorm_fwd_cpu_constant_input() raises:
    # Constant input — var=0, eps saves from division by zero
    # output should be beta (since x_hat=0)
    comptime dtype = DType.float32
    var x     = Tensor[dtype].full(Shape(3, 4), 7.0)
    var gamma = Tensor[dtype].full(Shape(4), 2.0)
    var beta  = Tensor[dtype].full(Shape(4), 0.5)
    var out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
    # x_hat = 0 everywhere => out = gamma*0 + beta = beta
    assert_true(out.all_close[atol=1e-5](
        Tensor[dtype].full(Shape(3, 4), 0.5)
    ))


def test_layernorm_fwd_cpu_eval_mode_no_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].randn(Shape(2, 4), mean=0.0, std=1.0)
    var ln = LayerNorm[dtype](4)
    ln.eval()
    var out = ln(x)
    assert_true(not out.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# BACKWARD — CPU
# ═════════════════════════════════════════════════════════════════════════════

def test_layernorm_bwd_cpu_gamma_grad_shape() raises:
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


def test_layernorm_bwd_cpu_beta_grad_equals_sum_upstream() raises:
    # d_beta = sum(upstream) over all non-D dims
    # upstream = ones (from sum loss) => d_beta = B*T for each element
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

def test_layernorm_bwd_cpu_gamma_grad_value() raises:
    comptime dtype = DType.float32
    var B = 2; var T = 3; var D = 4
    var _tmp0 = Tensor[dtype].arange(1.0, 25.0)
    var x     = _tmp0.reshape(B, T, D)
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

def test_layernorm_bwd_cpu_dx_grad_sums_to_zero() raises:
    # The three-term formula guarantees that dx sums to zero per token
    # (gradient is orthogonal to constant and to x_hat)
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


def test_layernorm_bwd_cpu_no_grad_input() raises:
    # If x has no requires_grad, only gamma and beta get grads
    comptime dtype = DType.float32
    var x     = Tensor[dtype].randn(Shape(2, 4), mean=0.0, std=1.0)  # no grad
    var gamma = Tensor[dtype].ones(Shape(4),  requires_grad=True)
    var beta  = Tensor[dtype].zeros(Shape(4), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    assert_true(gamma.grad().shape() == Shape(4))
    assert_true(beta.grad().shape()  == Shape(4))


def test_layernorm_bwd_cpu_2d_dx_no_nan() raises:
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


def test_layernorm_bwd_cpu_3d_dx_no_nan() raises:
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

def test_layernorm_gradflow_cpu_chained_with_linear() raises:
    # LayerNorm -> sum -> backward — grad flows through LN to x
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


def test_layernorm_gradflow_cpu_gamma_ones_beta_zeros_dx_sum_zero() raises:
    # Classic property: sum(dx) over last dim == 0 per token
    comptime dtype = DType.float32
    var _tmp0 = Tensor[dtype].arange(1.0, 13.0)
    var x     = _tmp0.reshape(3, 4)
    x.requires_grad_(True)
    var gamma = Tensor[dtype].ones(Shape(4),  requires_grad=True)
    var beta  = Tensor[dtype].zeros(Shape(4), requires_grad=True)
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = out.sum()
    loss.backward()
    var dx_sums = x.grad().sum(axes=IntArray(-1))
    assert_true(dx_sums.all_close[atol=1e-4](Tensor[dtype].zeros(dx_sums.shape())))


def test_layernorm_gradflow_cpu_no_grad_no_ancestry() raises:
    comptime dtype = DType.float32
    var x     = Tensor[dtype].randn(Shape(2, 4), mean=0.0, std=1.0)
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta  = Tensor[dtype].zeros(Shape(4))
    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    assert_true(not out.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# FORWARD — GPU
# ═════════════════════════════════════════════════════════════════════════════

def test_layernorm_fwd_gpu_1d_shape() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
        var ln = LayerNorm[dtype](5)
        ln = ln.to_gpu()
        var out = ln(x_cpu.to_gpu())
        assert_true(out.shape() == Shape(5))


def test_layernorm_fwd_gpu_2d_matches_cpu() raises:
    comptime if has_accelerator():
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


def test_layernorm_fwd_gpu_3d_shape() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].randn(Shape(2, 4, 8), mean=0.0, std=1.0)
        var ln = LayerNorm[dtype](8)
        ln = ln.to_gpu()
        var out = ln(x_cpu.to_gpu())
        assert_true(out.shape() == Shape(2, 4, 8))


def test_layernorm_fwd_gpu_output_mean_near_zero() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].randn(Shape(4, 8), mean=5.0, std=3.0)
        var ln = LayerNorm[dtype](8)
        ln = ln.to_gpu()
        var out = ln(x_cpu.to_gpu()).to_cpu()
        var row_means = out.mean[track_grad=False](axes=[-1])
        assert_true(row_means.all_close[atol=1e-4](Tensor[dtype].zeros_like(row_means)))


def test_layernorm_fwd_gpu_output_std_near_one() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].randn(Shape(4, 8), mean=5.0, std=3.0)
        var ln = LayerNorm[dtype](8)
        ln = ln.to_gpu()
        var out = ln(x_cpu.to_gpu()).to_cpu()
        var row_vars = out.variance[track_grad=False](axis=-1, unbiased=False)
        assert_true(row_vars.all_close[atol=1e-4](Tensor[dtype].ones_like(row_vars)))


def test_layernorm_fwd_gpu_constant_input() raises:
    comptime if has_accelerator():
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

def test_layernorm_bwd_gpu_grad_shapes() raises:
    comptime if has_accelerator():
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


def test_layernorm_bwd_gpu_beta_grad_value() raises:
    comptime if has_accelerator():
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


def test_layernorm_bwd_gpu_dx_sum_zero() raises:
    comptime if has_accelerator():
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


def test_layernorm_bwd_gpu_no_nan() raises:
    comptime if has_accelerator():
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

def test_layernorm_parity_fwd_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(1.0, 13.0)
        var x     = _tmp0.reshape(3, 4)
        var gamma = Tensor[dtype].ones(Shape(4))
        var beta  = Tensor[dtype].zeros(Shape(4))
        var cpu_out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
        var gpu_out = LayerNormForward[dtype].forward[track_grad=False](
            x.to_gpu(), gamma.to_gpu(), beta.to_gpu()
        ).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


def test_layernorm_parity_fwd_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x     = Tensor[dtype].randn(Shape(2, 4, 8), mean=0.0, std=1.0)
        var gamma = Tensor[dtype].ones(Shape(8))
        var beta  = Tensor[dtype].zeros(Shape(8))
        var cpu_out = LayerNormForward[dtype].forward[track_grad=False](x, gamma, beta)
        var gpu_out = LayerNormForward[dtype].forward[track_grad=False](
            x.to_gpu(), gamma.to_gpu(), beta.to_gpu()
        ).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


def test_layernorm_parity_bwd_beta_grad() raises:
    comptime if has_accelerator():
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


def test_layernorm_parity_bwd_dx() raises:
    comptime if has_accelerator():
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

def test_layernorm_layer_parameters() raises:
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](8)
    assert_true(ln.num_parameters() == 16)   # gamma(8) + beta(8)
    assert_true(len(ln.parameters()) == 2)


def test_layernorm_layer_train_eval_toggle() raises:
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](4)
    var x = Tensor[dtype].randn(Shape(2, 4), mean=0.0, std=1.0, requires_grad=True)
    ln.train()
    var out_train = ln(x)
    assert_true(out_train.requires_grad)
    ln.eval()
    var out_eval = ln(x)
    assert_true(not out_eval.requires_grad)


def test_layernorm_layer_gamma_ones_beta_zeros_init() raises:
    comptime dtype = DType.float32
    var ln = LayerNorm[dtype](4)
    assert_true(ln.gamma.all_close(Tensor[dtype].ones(Shape(4))))
    assert_true(ln.beta.all_close(Tensor[dtype].zeros(Shape(4))))


def test_layernorm_module_direct_default_parity() raises:
    comptime dtype = DType.float32
    # Audit item 15: the module default eps (1e-5) and the direct-forward
    # default eps must agree — no silent numeric divergence between call
    # styles. Same kernel, same eps → bit-identical output.
    var x = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, -1.0, 5.0]], requires_grad=True
    )
    var ln = LayerNorm[dtype](4)
    var out_module = ln(x)
    var gamma = Tensor[dtype].ones(Shape(4))
    var beta = Tensor[dtype].zeros(Shape(4))
    var out_direct = LayerNormForward[dtype].forward[track_grad=False](
        x, gamma, beta
    )
    assert_true(out_module == out_direct)


def layernorm_fd_loss(
    x: Tensor[DType.float32],
    gamma: Tensor[DType.float32],
    beta: Tensor[DType.float32],
    w: Tensor[DType.float32],
) -> Scalar[DType.float32]:
    """Weighted-loss forward for finite differences (no graph)."""
    var out = LayerNormForward[DType.float32].forward[track_grad=False](
        x, gamma, beta
    )
    var loss = (out * w).sum()
    return loss.item()


def test_layernorm_bwd_finite_diff_batched() raises:
    comptime dtype = DType.float32
    # Batched (B=2, T=3, D=4) central differences vs analytic grads.
    # The weighted loss is load-bearing: loss = out.sum() (uniform upstream)
    # gives dx == 0 for ANY x_hat, so it cannot catch a miswired three-term
    # formula. The batched shape exercises the sequential axis-0 reduction
    # loops for d_gamma/d_beta. Follows the test_cnn.mojo clone()+perturb
    # pattern (perturb a clone, never the live tensor).
    var eps = Scalar[dtype](1e-3)
    var tol = Scalar[dtype](1e-2)
    var _tmp = Tensor[dtype].arange(1.0, 25.0)
    var x = _tmp.reshape(2, 3, 4).contiguous()
    x.requires_grad_(True)
    var gamma = Tensor[dtype].d1([2.0, 0.5, 1.0, 1.5], requires_grad=True)
    var beta = Tensor[dtype].d1([0.1, -0.2, 0.3, 0.0], requires_grad=True)
    var w = Tensor[dtype].arange(1.0, 25.0).reshape(2, 3, 4) * 0.05

    var out = LayerNormForward[dtype].forward[track_grad=True](x, gamma, beta)
    var loss = (out * w).sum()
    loss.backward()

    for idx in range(x.numels()):
        var xp = x.clone()
        xp.buffer.data_buffer()[idx] += eps
        var lp = layernorm_fd_loss(xp, gamma, beta, w)
        var xm = x.clone()
        xm.buffer.data_buffer()[idx] -= eps
        var lm = layernorm_fd_loss(xm, gamma, beta, w)
        var num = (lp - lm) / (Scalar[dtype](2.0) * eps)
        var an = x.gradbox[].buffer().buffer[idx]
        assert_true(
            abs(an - num) < tol, "layernorm fd dx idx=" + String(idx)
        )

    for idx in range(gamma.numels()):
        var gp = gamma.clone()
        gp.buffer.data_buffer()[idx] += eps
        var lp = layernorm_fd_loss(x, gp, beta, w)
        var gm = gamma.clone()
        gm.buffer.data_buffer()[idx] -= eps
        var lm = layernorm_fd_loss(x, gm, beta, w)
        var num = (lp - lm) / (Scalar[dtype](2.0) * eps)
        var an = gamma.gradbox[].buffer().buffer[idx]
        assert_true(
            abs(an - num) < tol, "layernorm fd dgamma idx=" + String(idx)
        )

    for idx in range(beta.numels()):
        var bp = beta.clone()
        bp.buffer.data_buffer()[idx] += eps
        var lp = layernorm_fd_loss(x, gamma, bp, w)
        var bm = beta.clone()
        bm.buffer.data_buffer()[idx] -= eps
        var lm = layernorm_fd_loss(x, gamma, bm, w)
        var num = (lp - lm) / (Scalar[dtype](2.0) * eps)
        var an = beta.gradbox[].buffer().buffer[idx]
        assert_true(
            abs(an - num) < tol, "layernorm fd dbeta idx=" + String(idx)
        )


def test_layernorm_cpu_offset_strided_gamma_beta_view() raises:
    comptime dtype = DType.float32
    # Sliced gamma/beta (offset != 0 / strided): pass 2 reads them flat from
    # index 0, so forward must materialize a contiguous copy. Without the
    # guard this silently computes with the wrong elements.
    var x = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]])
    var gamma = Tensor[dtype].d1([2.0, 0.5, 1.0, 1.5])
    var beta = Tensor[dtype].d1([0.1, -0.2, 0.3, 0.0])
    var ref_ = LayerNormForward[dtype].forward[track_grad=False](
        x, gamma, beta
    )
    # Offset view: same values, storage offset 1.
    var big_g = Tensor[dtype].d1([99.0, 2.0, 0.5, 1.0, 1.5, 88.0])
    var big_b = Tensor[dtype].d1([9.0, 0.1, -0.2, 0.3, 0.0, 8.0])
    var out_off = LayerNormForward[dtype].forward[track_grad=False](
        x, big_g.slice(1, 5), big_b.slice(1, 5)
    )
    assert_true(out_off.all_close[atol=1e-5](ref_))
    # Strided view: same values at stride 2.
    var big_gs = Tensor[dtype].d1([2.0, 99.0, 0.5, 99.0, 1.0, 99.0, 1.5, 99.0])
    var big_bs = Tensor[dtype].d1([0.1, 9.0, -0.2, 9.0, 0.3, 9.0, 0.0, 9.0])
    var out_str = LayerNormForward[dtype].forward[track_grad=False](
        x, big_gs.slice(0, 8, 2), big_bs.slice(0, 8, 2)
    )
    assert_true(out_str.all_close[atol=1e-5](ref_))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll layernorm tests passed!")

