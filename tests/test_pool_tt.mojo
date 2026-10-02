"""Parity suite for GPU MaxPoolTT — CPU MaxPool2d untouched.

Every test runs MaxPool2d on CPU against MaxPoolTT on GPU: forward parity
(dense, padded, all-padded windows, strided input) plus backward gradflow
(ties collapse, overlapping windows accumulate). GPU bodies start with the
`comptime if has_accelerator()` guard so the file compiles everywhere and
the cpu_all generator skips them.
"""
from tenmo.tensor import Tensor
from tenmo.pooling import MaxPool2d
from tenmo.pool_tt import MaxPoolTT

from std.testing import assert_almost_equal, assert_true, TestSuite
from std.utils.numerics import isinf
from std.random import seed
from std.sys import has_accelerator


def test_pool_tt_forward_parity() raises:
    """GPU MaxPoolTT.forward matches CPU MaxPool2d (k=2, dense)."""
    print("test_pool_tt_forward_parity")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(7)
        var x = Tensor[dtype].randn(2, 3, 8, 8)
        var pool_c = MaxPool2d[dtype](kernel_size=2)
        var expected = pool_c(x)
        var pool_g = MaxPoolTT[dtype](kernel_size=2)
        var result = pool_g(x.to_gpu())
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))


def test_pool_tt_forward_stride_pad() raises:
    """Parity with stride 3x3, stride 2, padding 1 (bounds guards)."""
    print("test_pool_tt_forward_stride_pad")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(11)
        var x = Tensor[dtype].randn(1, 2, 7, 6)
        var pool_c = MaxPool2d[dtype](kernel_size=3, stride=2, padding=1)
        var expected = pool_c(x)
        var pool_g = MaxPoolTT[dtype](kernel_size=3, stride=2, padding=1)
        var result = pool_g(x.to_gpu())
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))


def test_pool_tt_forward_all_padded() raises:
    """All-padded windows stay neg_inf on GPU, matching CPU (mask -1)."""
    print("test_pool_tt_forward_all_padded")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].zeros(1, 1, 2, 2)
        x[0, 0, 0, 0] = 1.0
        x[0, 0, 0, 1] = 2.0
        x[0, 0, 1, 0] = 3.0
        x[0, 0, 1, 1] = 4.0
        # k=3 s=1 pad=2 on 2x2 -> 4x4 out; corners see no valid cells.
        var pool_c = MaxPool2d[dtype](kernel_size=3, stride=1, padding=2)
        var expected = pool_c(x)
        var pool_g = MaxPoolTT[dtype](kernel_size=3, stride=1, padding=2)
        var got = pool_g(x.to_gpu()).to_cpu()
        for oy in range(4):
            for ox in range(4):
                var e = expected[0, 0, oy, ox]
                var g = got[0, 0, oy, ox]
                if isinf(e):
                    assert_true(isinf(g))
                else:
                    assert_almost_equal(g, e, atol=1e-5)


def test_pool_tt_forward_strided_input() raises:
    """Strided GPU input (transpose view) pools correctly, no materialize."""
    print("test_pool_tt_forward_strided_input")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var base = Tensor[dtype].zeros(1, 1, 4, 4)
        for i in range(4):
            for j in range(4):
                base[0, 0, i, j] = Float32(i * 4 + j)
        # CPU reference: same values, materialized dense.
        var view_c = base.transpose(0, 1, 3, 2).contiguous()
        var pool_c = MaxPool2d[dtype](kernel_size=2)
        var expected = pool_c(view_c)
        # GPU: transpose AFTER upload, so the device view is strided.
        var view_g = base.to_gpu().transpose(0, 1, 3, 2)
        assert_true(not view_g.is_contiguous())
        var pool_g = MaxPoolTT[dtype](kernel_size=2)
        var result = pool_g(view_g)
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close[atol=1e-5](expected))


def test_pool_tt_backward_gradflow() raises:
    """GPU backward grads match CPU (ties collapse, overlaps accumulate)."""
    print("test_pool_tt_backward_gradflow")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(13)
        # Overlapping windows (k=3, s=1): shared maxima accumulate.
        var x_c = Tensor[dtype].randn(1, 2, 6, 6)
        x_c.requires_grad_(True)
        var x_g = x_c.clone().to_gpu()
        x_g.requires_grad_(True)
        var pool_c = MaxPool2d[dtype](kernel_size=3, stride=1)
        var pool_g = MaxPoolTT[dtype](kernel_size=3, stride=1)
        var out_c = pool_c(x_c)
        var out_g = pool_g(x_g)
        var loss_c = out_c.sum()
        var loss_g = out_g.sum()
        loss_c.backward()
        loss_g.backward()
        assert_true(x_g.grad().is_on_gpu())
        assert_true(
            x_g.grad().to_cpu().all_close[atol=1e-4](x_c.grad().as_tensor())
        )
        # Pure ties (constant input): first-max wins, single target.
        var t_c = Tensor[dtype].ones(1, 1, 4, 4)
        t_c.requires_grad_(True)
        var t_g = t_c.clone().to_gpu()
        t_g.requires_grad_(True)
        var tie_c = MaxPool2d[dtype](kernel_size=2)
        var tie_g = MaxPoolTT[dtype](kernel_size=2)
        var tie_loss_c = tie_c(t_c).sum()
        var tie_loss_g = tie_g(t_g).sum()
        tie_loss_c.backward()
        tie_loss_g.backward()
        assert_true(t_g.grad().is_on_gpu())
        assert_true(
            t_g.grad().to_cpu().all_close[atol=1e-4](t_c.grad().as_tensor())
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
