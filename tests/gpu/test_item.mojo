from tenmo.tensor import Tensor
from std.sys import has_accelerator
from tenmo.layernorm import *
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from std.math import sqrt
from std.testing import assert_true, assert_false, TestSuite
from tenmo.common_utils import isnan, isinf, Epsilon
from std.math import log




def test_item_gpu_scalar_tensor() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].scalar(42.0)
        var a_gpu = a.to_gpu()
        assert_true(a_gpu.item() == 42.0)


def test_item_gpu_1d_tensor() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([7.0])
        var a_gpu = a.to_gpu()
        assert_true(a_gpu.item() == 7.0)


def test_item_gpu_gradbox_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].scalar(3.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = Tensor[dtype].scalar(4.0, requires_grad=True)
        var b_gpu = b.to_gpu()
        var c_gpu = a_gpu + b_gpu
        # Manually seed gradbox
        var seed_cpu = Tensor[dtype].full(c_gpu.shape(), Scalar[dtype](1.0))
        var seed = seed_cpu.to_gpu()
        c_gpu.seed_grad(seed)
        # item() on GPU gradbox
        assert_true(c_gpu.gradients().item() == 1.0)


def test_item_gpu_gradbox_after_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].scalar(3.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = Tensor[dtype].scalar(4.0)
        var b_gpu = b.to_gpu()
        var c_gpu = a_gpu + b_gpu
        c_gpu.backward()
        # a.gradbox should be 1.0 after backward
        assert_true(a.grad().item() == 1.0)


def test_item_gpu_sum_result() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var s = a_gpu.sum()  # Shape() scalar
        assert_true(s.item() == 10.0)


def test_item_gpu_mean_result() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var m = a_gpu.mean()  # Shape() scalar
        assert_true(m.item() == 2.5)



# =============================================================================
# FORWARD TESTS
# =============================================================================


# =============================================================================
# BACKWARD TESTS — dx
# =============================================================================


# =============================================================================
# BACKWARD TESTS — d_gamma, d_beta
# =============================================================================


# =============================================================================
# GRAD FLOW — x requires_grad, gamma/beta fixed
# =============================================================================


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

# ═════════════════════════════════════════════════════════════════════════════
# BACKWARD — CPU
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# GRAD FLOW — CPU
# ═════════════════════════════════════════════════════════════════════════════

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



# ============================================================================
# TESTS - Forward Pass
# ============================================================================


# ============================================================================
# TESTS - Backward Pass
# ============================================================================


# ============================================================================
# NUMERICAL STABILITY TESTS
# ============================================================================


# ============================================================================
# EDGE CASES
# ============================================================================


# ── CPU Forward Tests ─────────────────────────────────────────────────────────


# ── CPU Backward Tests ────────────────────────────────────────────────────────


# ── GPU Forward Tests ─────────────────────────────────────────────────────────


def test_log_gpu_1d_basic_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.log()
        assert_true(result.is_on_gpu())
        var expect = Tensor[dtype].d1(
            [log(Float32(1.0)), log(Float32(2.0)), log(Float32(3.0))]
        )
        assert_true(result.to_cpu().all_close(expect))


def test_log_gpu_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var result = a.log()
        assert_true(result.is_on_gpu())
        var expect = Tensor[dtype].d2(
            [
                [log(Float32(1.0)), log(Float32(2.0))],
                [log(Float32(3.0)), log(Float32(4.0))],
            ]
        )
        assert_true(result.to_cpu().all_close(expect))


def test_log_gpu_3d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var result = a.log()
        assert_true(result.is_on_gpu())
        var expect = Tensor[dtype].d3(
            [
                [
                    [log(Float32(1.0)), log(Float32(2.0))],
                    [log(Float32(3.0)), log(Float32(4.0))],
                ],
                [
                    [log(Float32(5.0)), log(Float32(6.0))],
                    [log(Float32(7.0)), log(Float32(8.0))],
                ],
            ]
        )
        assert_true(result.to_cpu().all_close(expect))


def test_log_gpu_ones_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].ones(Shape(4)).to_gpu()
        var result = a.log()
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close(Tensor[dtype].zeros(Shape(4))))


def test_log_gpu_epsilon_clamping() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([0.0, -1.0, 1.0]).to_gpu()
        var result = a.log()
        var result_cpu = result.to_cpu()
        var expected_clamped = log(Float32(1e-7))
        assert_true(result_cpu[[0]] == expected_clamped)
        assert_true(result_cpu[[1]] == expected_clamped)
        assert_true(result_cpu[[2]] == Float32(0.0))


def test_log_gpu_large_values() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([100.0, 1000.0, 10000.0]).to_gpu()
        var result = a.log()
        var expect = Tensor[dtype].d1(
            [log(Float32(100.0)), log(Float32(1000.0)), log(Float32(10000.0))]
        )
        assert_true(result.to_cpu().all_close(expect))


# ── GPU Backward Tests ────────────────────────────────────────────────────────
def test_log_gpu_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 0.5, 0.25])))


def test_log_gpu_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [4.0, 8.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log()
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 0.5], [0.25, 0.125]]))
        )


def test_log_gpu_3d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [4.0, 8.0]], [[1.0, 4.0], [2.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.log()
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [
                        [[1.0, 0.5], [0.25, 0.125]],
                        [[1.0, 0.25], [0.5, 0.125]],
                    ]
                )
            )
        )


def test_log_gpu_backward_chain() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log() * 2.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 1.0, 0.5])))


def test_log_gpu_backward_epsilon_clamping() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([0.0, 1.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad()[[0]] == Float32(1.0) / Float32(1e-7))
        assert_true(a.grad()[[1]] == Float32(1.0))
        assert_true(a.grad()[[2]] == Float32(0.5))


def test_log_gpu_backward_chained_with_exp() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # log(exp(x)) = x, grad should be 1.0 everywhere
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.exp().log()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(3))))


def test_log_gpu_backward_custom_epsilon() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([0.0, 1.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log[epsilon=Scalar[dtype](1e-6)]()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad()[[0]] == Float32(1.0) / Float32(1e-6))
        assert_true(a.grad()[[1]] == Float32(1.0))
        assert_true(a.grad()[[2]] == Float32(0.25))


# ── CPU/GPU Parity Tests ──────────────────────────────────────────────────────


def test_log_parity_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
        var a_gpu = a_cpu.to_gpu()
        var result_cpu = a_cpu.log()
        var result_gpu = a_gpu.log()
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


def test_log_parity_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a_cpu.to_gpu()
        var result_cpu = a_cpu.log()
        var result_gpu = a_gpu.log()
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


def test_log_parity_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 4.0, 8.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.log().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.log().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


def test_log_parity_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [4.0, 8.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.log().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.log().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


def test_log_parity_epsilon_clamping() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([0.0, -1.0, 1.0, 2.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.log().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.log().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


def test_log_parity_chain_exp() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.exp().log().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.exp().log().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))




# ── CPU Tests: Max ────────────────────────────────────────────────────────────


# ── CPU Tests: Min ────────────────────────────────────────────────────────────
# ── CPU Tests: Max + Min combined in chain ────────────────────────────────────


# ── GPU Tests: Max ────────────────────────────────────────────────────────────


def test_maxmin_gpu_max_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        assert_true(
            b.to_cpu().all_close(Tensor[dtype].d1([4.0, 5.0, 4.0, 7.0, 4.0]))
        )


def test_maxmin_gpu_max_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 0.0, 1.0, 0.0]))
        )


def test_maxmin_gpu_max_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 6.0], [3.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        assert_true(
            b.to_cpu().all_close(Tensor[dtype].d2([[4.0, 6.0], [4.0, 4.0]]))
        )


def test_maxmin_gpu_max_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 6.0], [3.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[0.0, 1.0], [0.0, 0.0]]))
        )


def test_maxmin_gpu_max_3d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 5.0], [3.0, 8.0]], [[6.0, 2.0], [4.0, 9.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[0.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
                )
            )
        )


def test_maxmin_gpu_max_all_below_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(10.0)
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


def test_maxmin_gpu_max_all_above_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([5.0, 6.0, 7.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


def test_maxmin_gpu_max_chained() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 8.0], [5.0, 3.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(4.0)
        var c = b * 2.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[0.0, 2.0], [2.0, 0.0]]))
        )


# ── GPU Tests: Min ────────────────────────────────────────────────────────────


def test_maxmin_gpu_min_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        assert_true(
            b.to_cpu().all_close(Tensor[dtype].d1([1.0, 4.0, 3.0, 4.0, 2.0]))
        )


def test_maxmin_gpu_min_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([1.0, 0.0, 1.0, 0.0, 1.0]))
        )


def test_maxmin_gpu_min_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 6.0], [3.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        assert_true(
            b.to_cpu().all_close(Tensor[dtype].d2([[1.0, 4.0], [3.0, 2.0]]))
        )


def test_maxmin_gpu_min_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 6.0], [3.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 0.0], [1.0, 1.0]]))
        )


def test_maxmin_gpu_min_3d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 5.0], [3.0, 8.0]], [[6.0, 2.0], [4.0, 9.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]
                )
            )
        )


def test_maxmin_gpu_min_all_above_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([5.0, 6.0, 7.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


def test_maxmin_gpu_min_all_below_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


def test_maxmin_gpu_min_chained() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 8.0], [5.0, 3.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.min(4.0)
        var c = b * 2.0
        var loss = c.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[2.0, 0.0], [0.0, 2.0]]))
        )


# ── GPU Tests: Max + Min combined ─────────────────────────────────────────────


def test_maxmin_gpu_max_then_min_chained() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # clamp(x, 2.0, 6.0) = min(max(x, 2.0), 6.0)
        var a = Tensor[dtype].d1([1.0, 3.0, 5.0, 7.0, 9.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(2.0)
        var c = b.min(6.0)
        var loss = c.sum()
        loss.backward()
        # Grad = 1.0 only where 2.0 < a < 6.0
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 1.0, 0.0, 0.0]))
        )


def test_maxmin_gpu_negated_grad_flow() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.max(2.0)
        var c = -b
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, -1.0, -1.0])))


# ── CPU/GPU parity check ──────────────────────────────────────────────────────


def test_maxmin_gpu_parity_max() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 3.0, 7.0], [8.0, 2.0, 5.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var b_cpu = a_cpu.max(4.0)
        var b_gpu = a_gpu.max(4.0)
        assert_true(b_cpu.all_close(b_gpu.to_cpu()))

        var loss_cpu = b_cpu.sum()
        loss_cpu.backward()

        var loss_gpu = b_gpu.sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


def test_maxmin_gpu_parity_min() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 3.0, 7.0], [8.0, 2.0, 5.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var b_cpu = a_cpu.min(4.0)
        var b_gpu = a_gpu.min(4.0)
        assert_true(b_cpu.all_close(b_gpu.to_cpu()))

        var loss_cpu = b_cpu.sum()
        loss_cpu.backward()

        var loss_gpu = b_gpu.sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


# ── Main ──────────────────────────────────────────────────────────────────────



# Exhaustive tests for the revamped MinMax implementation.
# Prefix: mmrev_ (minmax revamp)
# Covers:
#  - Forward correctness (0D, 1D, 2D, 3D, 4D)
#  - keepdims=True and keepdims=False
#  - Backward / grad flow (mask correctness, ties, unique max/min)
#  - Multi-axis reductions
#  - Global reduction (no axes = all axes)
#  - CPU and GPU variants
#
# ──────────────────────────────────────────────────────────────────────────────
# CPU TESTS — MAX
# ──────────────────────────────────────────────────────────────────────────────

# ── 1D ────────────────────────────────────────────────────────────────────────


# ── 2D ────────────────────────────────────────────────────────────────────────


# ── 3D ────────────────────────────────────────────────────────────────────────


# ── 4D ────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# CPU TESTS — MIN
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# CPU — grad flow
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# GPU TESTS — MAX
# ──────────────────────────────────────────────────────────────────────────────


def test_mmrev_gpu_max_1d_forward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_1d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
        var a_gpu = a.to_gpu()
        var m = a_gpu.max()
        assert_true(m.shape() == Shape())
        assert_true(m.to_cpu().item() == 9.0)


def test_mmrev_gpu_max_1d_backward_unique() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_1d_backward_unique")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([3.0, 1.0, 9.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.max()
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 1.0, 0.0])))


def test_mmrev_gpu_max_1d_backward_tied() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_1d_backward_tied")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 5.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.max()
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 0.5, 0.0, 0.5])))


def test_mmrev_gpu_max_2d_axis0_forward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_2d_axis0_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([0])
        assert_true(m.shape() == Shape(3))
        assert_true(m.to_cpu().all_close(Tensor[dtype].d1([4.0, 5.0, 6.0])))


def test_mmrev_gpu_max_2d_axis0_keepdims_forward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_2d_axis0_keepdims_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([0], keepdims=True)
        assert_true(m.shape() == Shape(1, 3))
        assert_true(m.to_cpu().all_close(Tensor[dtype].d2([[4.0, 5.0, 6.0]])))


def test_mmrev_gpu_max_2d_axis1_forward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_2d_axis1_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([1])
        assert_true(m.shape() == Shape(2))
        assert_true(m.to_cpu().all_close(Tensor[dtype].d1([5.0, 6.0])))


def test_mmrev_gpu_max_2d_axis1_keepdims_forward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_2d_axis1_keepdims_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([1], keepdims=True)
        assert_true(m.shape() == Shape(2, 1))
        assert_true(m.to_cpu().all_close(Tensor[dtype].d2([[5.0], [6.0]])))


def test_mmrev_gpu_max_2d_axis0_backward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_2d_axis0_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([0])
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
            )
        )


def test_mmrev_gpu_max_2d_axis1_backward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_2d_axis1_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([1])
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            )
        )


def test_mmrev_gpu_max_2d_axis1_keepdims_backward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_2d_axis1_keepdims_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([1], keepdims=True)
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            )
        )


def test_mmrev_gpu_max_2d_global_backward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_2d_global_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 7.0], [3.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.max()
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[0.0, 1.0], [0.0, 0.0]]))
        )


def test_mmrev_gpu_max_2d_tied_backward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_2d_tied_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[3.0, 5.0, 5.0], [1.0, 2.0, 4.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([1])
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
            )
        )


def test_mmrev_gpu_max_3d_axis0_forward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_3d_axis0_forward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var a = _tmp0.reshape(Shape(2, 3, 4))
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([0])
        assert_true(m.shape() == Shape(3, 4))
        var _tmp1 = Tensor[dtype].arange(12.0, 24.0)
        var _tmp2 = _tmp1.reshape(Shape(3, 4))
        assert_true(m.to_cpu().all_close(_tmp2))


def test_mmrev_gpu_max_3d_axis0_backward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_3d_axis0_backward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var a = _tmp0.reshape(Shape(2, 3, 4))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([0])
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        var expected = Tensor[dtype].zeros(Shape(2, 3, 4))
        for j in range(3):
            for k in range(4):
                expected[1, j, k] = 1.0
        assert_true(a.grad().all_close(expected))


def test_mmrev_gpu_max_3d_multi_axis_forward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_3d_multi_axis_forward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var a = _tmp0.reshape(Shape(2, 3, 4))
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([0, 2])
        assert_true(m.shape() == Shape(3))
        assert_true(m.to_cpu().all_close(Tensor[dtype].d1([15.0, 19.0, 23.0])))


def test_mmrev_gpu_max_3d_multi_axis_backward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_3d_multi_axis_backward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var a = _tmp0.reshape(Shape(2, 3, 4))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([0, 2])
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        var expected = Tensor[dtype].zeros(Shape(2, 3, 4))
        expected[1, 0, 3] = 1.0
        expected[1, 1, 3] = 1.0
        expected[1, 2, 3] = 1.0
        assert_true(a.grad().all_close(expected))


def test_mmrev_gpu_max_4d_axis1_forward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_4d_axis1_forward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 120.0)
        var a = _tmp0.reshape(Shape(2, 3, 4, 5))
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([1])
        assert_true(m.shape() == Shape(2, 4, 5))


def test_mmrev_gpu_max_4d_axis1_backward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_4d_axis1_backward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 120.0)
        var a = _tmp0.reshape(Shape(2, 3, 4, 5))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([1])
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        var total_grad = a.grad().sum().item()
        assert_true(total_grad == 40.0)


# ──────────────────────────────────────────────────────────────────────────────
# GPU TESTS — MIN
# ──────────────────────────────────────────────────────────────────────────────


def test_mmrev_gpu_min_1d_forward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_min_1d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([3.0, 1.0, 4.0, 1.0, 5.0])
        var a_gpu = a.to_gpu()
        var m = a_gpu.min()
        assert_true(m.shape() == Shape())
        assert_true(m.to_cpu().item() == 1.0)


def test_mmrev_gpu_min_1d_backward_unique() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_min_1d_backward_unique")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([3.0, 1.0, 4.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.min()
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 0.0, 0.0])))


def test_mmrev_gpu_min_1d_backward_tied() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_min_1d_backward_tied")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 5.0, 1.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.min()
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor[dtype].d1([0.5, 0.0, 0.5, 0.0])))


def test_mmrev_gpu_min_2d_axis0_forward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_min_2d_axis0_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        var a_gpu = a.to_gpu()
        var m = a_gpu.min([0])
        assert_true(m.shape() == Shape(3))
        assert_true(m.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


def test_mmrev_gpu_min_2d_axis1_forward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_min_2d_axis1_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        var a_gpu = a.to_gpu()
        var m = a_gpu.min([1])
        assert_true(m.shape() == Shape(2))
        assert_true(m.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0])))


def test_mmrev_gpu_min_2d_axis0_backward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_min_2d_axis0_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m = a_gpu.min([0])
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
            )
        )


def test_mmrev_gpu_min_2d_axis1_backward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_min_2d_axis1_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m = a_gpu.min([1])
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )
        )


def test_mmrev_gpu_min_2d_keepdims_backward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_min_2d_keepdims_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m = a_gpu.min([1], keepdims=True)
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )
        )


def test_mmrev_gpu_min_3d_axis2_forward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_min_3d_axis2_forward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var a = _tmp0.reshape(Shape(2, 3, 4))
        var a_gpu = a.to_gpu()
        var m = a_gpu.min([2])
        assert_true(m.shape() == Shape(2, 3))
        assert_true(
            m.to_cpu().all_close(
                Tensor[dtype].d2([[0.0, 4.0, 8.0], [12.0, 16.0, 20.0]])
            )
        )


def test_mmrev_gpu_min_3d_axis2_backward() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_min_3d_axis2_backward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var a = _tmp0.reshape(Shape(2, 3, 4))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.min([2])
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        var expected = Tensor[dtype].zeros(Shape(2, 3, 4))
        for i in range(2):
            for j in range(3):
                expected[i, j, 0] = 1.0
        assert_true(a.grad().all_close(expected))


# ──────────────────────────────────────────────────────────────────────────────
# GPU — grad flow
# ──────────────────────────────────────────────────────────────────────────────


def test_mmrev_gpu_max_grad_flow_chained() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_grad_flow_chained")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 8.0, 3.0], [4.0, 2.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m1 = a_gpu.max([1])
        var m2 = m1.max([0])
        var loss = m2.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
            )
        )


def test_mmrev_gpu_min_grad_flow_chained() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_min_grad_flow_chained")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[4.0, 8.0, 3.0], [4.0, 2.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var m1 = a_gpu.min([1])
        var m2 = m1.min([0])
        var loss = m2.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d2([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            )
        )


def test_mmrev_gpu_max_then_op_grad_flow() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_max_then_op_grad_flow")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 3.0], [4.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([1], keepdims=True)  # [[3],[4]]
        var ones = Tensor[dtype].ones(m.shape()).to_gpu()
        var prod = m * ones
        var loss = prod.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[0.0, 1.0], [1.0, 0.0]]))
        )


def test_mmrev_gpu_grad_lands_on_cpu() raises:
    comptime if has_accelerator():
        print("test_mmrev_gpu_grad_lands_on_cpu")
        comptime dtype = DType.float32
        # Verify grads always land on CPU regardless of op chain
        var _tmp0 = Tensor[dtype].arange(0.0, 12.0)
        var a = _tmp0.reshape(Shape(3, 4))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var m = a_gpu.max([1])
        var loss = m.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())


# ============================================================================
# EXHAUSTIVE MIN/MAX TEST SUITE
# Tests all branches: vectorization, parallelization, edge cases
# ============================================================================

# ============================================================================
# TEST GROUP 1: SCALAR INPUT (rank == 0 branch)
# ============================================================================


# ============================================================================
# TEST GROUP 2: FULL REDUCTION TO SCALAR (vectorized path)
# ============================================================================


# ============================================================================
# TEST GROUP 3: PARTIAL REDUCTION (parallelized path)
# ============================================================================


# ============================================================================
# TEST GROUP 4: KEEPDIMS FUNCTIONALITY
# ============================================================================


# ============================================================================
# TEST GROUP 5: EDGE CASES
# ============================================================================


# ============================================================================
# TEST GROUP 6: GRADIENT ACCUMULATION
# ============================================================================


# ============================================================================
# TEST GROUP 7: NO GRADIENT TRACKING
# ============================================================================


# ============================================================================
# TEST GROUP 8: MIXED MIN/MAX OPERATIONS
# ============================================================================


# ============================================================================
# TEST GROUP 9: NEGATIVE AXIS INDEXING
# ============================================================================


# ============================================================================
# TEST GROUP 10: HIGH-DIMENSIONAL TENSORS
# ============================================================================


# ============================================================================
# TEST GROUP 11: ZERO-SIZED DIMENSIONS (Edge case)
# ============================================================================


# ============================================================================
# TEST GROUP 12: NUMERICAL STABILITY
# ============================================================================


# ============================================================================
# TEST GROUP 13: STRESS TEST - MANY TIES
# ============================================================================


# ============================================================================
# TEST GROUP 14: INTERACTION WITH OTHER OPS
# ============================================================================


# ============================================================================
# TEST GROUP 15: BACKWARDS COMPATIBILITY
# ============================================================================


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll minmax tests passed!")
