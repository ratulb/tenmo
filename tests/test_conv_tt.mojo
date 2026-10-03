"""Parity suite for GPU ConvTT forward — CPU Conv untouched.

Every test runs Conv on CPU against ConvTT on GPU: dense parity (k=3, k=5,
k=1, dilation, channel chunking, partial edge tiles, asymmetric padding)
plus one fallback-shape test (k=9 → global-direct kernel) and one strided
input view. Backward gradflow runs sum-loss backward on CPU and GPU
and compares image, kernel, and bias grads — all GPU-resident. Also adds
layer parity (ConvTT2D vs CPU Conv), MixedSequential integration
(ConvTT2D+MaxPool2d+Linear, GPU shapes+gradflow), and a teacher-student
convergence test proving end-to-end SGD on GPU. GPU bodies start with
the `comptime if has_accelerator()` guard so the file compiles everywhere
and the cpu_all generator skips them.
"""
from tenmo.tensor import Tensor
from tenmo.conv import Conv
from tenmo.conv_tt import ConvTT, ConvTT2D
from tenmo.pooling import MaxPool2d
from tenmo.net import MixedSequential, Linear, Flatten, ReLU
from tenmo.mse import MSELoss
from tenmo.optim import SGD

from std.testing import assert_true, TestSuite
from std.random import seed
from std.sys import has_accelerator


def _bias(c_out: Int) -> Tensor[DType.float32]:
    var b = Tensor[DType.float32].zeros(c_out)
    for i in range(c_out):
        b[i] = Float32(i) * 0.5 - 1.0
    return b^


def test_conv_tt_forward_parity() raises:
    """GPU ConvTT.forward matches CPU Conv (k=3, s=1, pad=1, dense)."""
    print("test_conv_tt_forward_parity")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(7)
        var x = Tensor[dtype].randn(2, 3, 8, 8)
        var k = Tensor[dtype].randn(4, 3, 3, 3)
        var b = _bias(4)
        var expected = Conv[dtype].forward(x, k, b, 1, 1, 1, 1, 1, 1)
        var result = ConvTT[dtype].forward(
            x.to_gpu(), k.to_gpu(), b.to_gpu(), 1, 1, 1, 1, 1, 1
        )
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))


def test_conv_tt_stride_asym_pad() raises:
    """Parity with s=2, asymmetric padding, partial tiles (C_out=5)."""
    print("test_conv_tt_stride_asym_pad")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(11)
        var x = Tensor[dtype].randn(1, 2, 7, 6)
        var k = Tensor[dtype].randn(5, 2, 3, 3)
        var b = _bias(5)
        var expected = Conv[dtype].forward(x, k, b, 2, 1, 1, 0, 1, 0)
        var result = ConvTT[dtype].forward(
            x.to_gpu(), k.to_gpu(), b.to_gpu(), 2, 1, 1, 0, 1, 0
        )
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))


def test_conv_tt_dilation_chunking() raises:
    """Parity with dilation=2, C_in=10/C_out=10 (chunk + partial tiles)."""
    print("test_conv_tt_dilation_chunking")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(13)
        var x = Tensor[dtype].randn(1, 10, 9, 9)
        var k = Tensor[dtype].randn(10, 10, 3, 3)
        var b = _bias(10)
        var expected = Conv[dtype].forward(x, k, b, 1, 2, 2, 2, 2, 2)
        var result = ConvTT[dtype].forward(
            x.to_gpu(), k.to_gpu(), b.to_gpu(), 1, 2, 2, 2, 2, 2
        )
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))


def test_conv_tt_k1() raises:
    """Parity for 1x1 conv (no halo) with C_in=4."""
    print("test_conv_tt_k1")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(17)
        var x = Tensor[dtype].randn(2, 4, 5, 5)
        var k = Tensor[dtype].randn(3, 4, 1, 1)
        var b = _bias(3)
        var expected = Conv[dtype].forward(x, k, b)
        var result = ConvTT[dtype].forward(x.to_gpu(), k.to_gpu(), b.to_gpu())
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))


def test_conv_tt_fallback() raises:
    """K=9 exceeds staged caps → global-direct kernel, still parity."""
    print("test_conv_tt_fallback")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(19)
        var x = Tensor[dtype].randn(1, 2, 12, 12)
        var k = Tensor[dtype].randn(2, 2, 9, 9)
        var b = _bias(2)
        var expected = Conv[dtype].forward(x, k, b, 1, 1, 4, 4, 4, 4)
        var result = ConvTT[dtype].forward(
            x.to_gpu(), k.to_gpu(), b.to_gpu(), 1, 1, 4, 4, 4, 4
        )
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))


def test_conv_tt_strided_input() raises:
    """Strided GPU input (transpose view) convolves correctly."""
    print("test_conv_tt_strided_input")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(23)
        var base = Tensor[dtype].randn(1, 1, 6, 6)
        var k = Tensor[dtype].randn(2, 1, 3, 3)
        var b = _bias(2)
        # CPU reference: same values, materialized dense.
        var view_c = base.transpose(0, 1, 3, 2).contiguous()
        var expected = Conv[dtype].forward(view_c, k, b, 1, 1, 1, 1, 1, 1)
        # GPU: transpose AFTER upload, so the device view is strided.
        var view_g = base.to_gpu().transpose(0, 1, 3, 2)
        assert_true(not view_g.is_contiguous())
        var result = ConvTT[dtype].forward(
            view_g, k.to_gpu(), b.to_gpu(), 1, 1, 1, 1, 1, 1
        )
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close[atol=1e-4](expected))


def test_conv_tt_backward_gradflow() raises:
    """GPU backward grads match CPU (image, kernel, bias — all on GPU)."""
    print("test_conv_tt_backward_gradflow")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(29)
        var x_c = Tensor[dtype].randn(1, 2, 6, 6)
        var k_c = Tensor[dtype].randn(3, 2, 3, 3)
        var b_c = _bias(3)
        x_c.requires_grad_(True)
        k_c.requires_grad_(True)
        b_c.requires_grad_(True)
        var x_g = x_c.clone().to_gpu()
        var k_g = k_c.clone().to_gpu()
        var b_g = b_c.clone().to_gpu()
        x_g.requires_grad_(True)
        k_g.requires_grad_(True)
        b_g.requires_grad_(True)
        var out_c = Conv[dtype].forward(x_c, k_c, b_c, 1, 1, 1, 1, 1, 1)
        var out_g = ConvTT[dtype].forward(
            x_g, k_g, b_g, 1, 1, 1, 1, 1, 1
        )
        var loss_c = out_c.sum()
        var loss_g = out_g.sum()
        loss_c.backward()
        loss_g.backward()
        assert_true(x_g.grad().is_on_gpu())
        assert_true(k_g.grad().is_on_gpu())
        assert_true(b_g.grad().is_on_gpu())
        assert_true(
            x_g.grad().to_cpu().all_close[atol=1e-4](x_c.grad().as_tensor())
        )
        assert_true(
            k_g.grad().to_cpu().all_close[atol=1e-4](k_c.grad().as_tensor())
        )
        assert_true(
            b_g.grad().to_cpu().all_close[atol=1e-4](b_c.grad().as_tensor())
        )


def test_conv_tt_backward_stride_dil() raises:
    """Gradflow with s=2, dilation=2, asymmetric pads, partial tiles.

    NOTE: C_out=3 (not 2) is load-bearing. The CPU reference's db path
    miscomputes when its packed dY has < 16 elements in `parallelize`
    mode (pre-existing CPU-side landmine, reproduced CPU-only with FD
    ground truth; CPU Conv is untouched per mandate). With M=6, C_out=3
    the packed P has 18 elements and the CPU db is exact and stable.
    """
    print("test_conv_tt_backward_stride_dil")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(31)
        var x_c = Tensor[dtype].randn(1, 3, 8, 7)
        var k_c = Tensor[dtype].randn(3, 3, 3, 3)
        var b_c = _bias(3)
        x_c.requires_grad_(True)
        k_c.requires_grad_(True)
        b_c.requires_grad_(True)
        var x_g = x_c.clone().to_gpu()
        var k_g = k_c.clone().to_gpu()
        var b_g = b_c.clone().to_gpu()
        x_g.requires_grad_(True)
        k_g.requires_grad_(True)
        b_g.requires_grad_(True)
        var out_c = Conv[dtype].forward(x_c, k_c, b_c, 2, 2, 1, 1, 0, 1)
        var out_g = ConvTT[dtype].forward(
            x_g, k_g, b_g, 2, 2, 1, 1, 0, 1
        )
        var loss_c = out_c.sum()
        var loss_g = out_g.sum()
        loss_c.backward()
        loss_g.backward()
        assert_true(x_g.grad().is_on_gpu())
        assert_true(k_g.grad().is_on_gpu())
        assert_true(b_g.grad().is_on_gpu())
        assert_true(
            x_g.grad().to_cpu().all_close[atol=1e-4](x_c.grad().as_tensor())
        )
        assert_true(
            k_g.grad().to_cpu().all_close[atol=1e-4](k_c.grad().as_tensor())
        )
        assert_true(
            b_g.grad().to_cpu().all_close[atol=1e-4](b_c.grad().as_tensor())
        )


def test_conv_tt_layer_parity() raises:
    """ConvTT2D layer matches CPU Conv (train and eval modes)."""
    print("test_conv_tt_layer_parity")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(41)
        var layer = ConvTT2D[dtype](
            in_channels=2,
            out_channels=3,
            kernel_size=3,
            padding=1,
            init_seed=7,
        )
        var x = Tensor[dtype].randn(1, 2, 6, 6)
        var expected = Conv[dtype].forward(
            x, layer.weight, layer.bias.value(), 1, 1, 1, 1, 1, 1
        )
        # The layer's params live on CPU at construction; move the layer
        # (CPU input against GPU params — or vice versa — panics by design).
        var layer_g = layer.to_gpu()
        # Train mode (default): grad-tracked, same values.
        var got_train = layer_g(x.to_gpu()).to_cpu()
        assert_true(got_train.all_close[atol=1e-4](expected))
        # Eval mode: no graph, same values.
        layer_g.eval()
        var got_eval = layer_g(x.to_gpu()).to_cpu()
        assert_true(got_eval.all_close[atol=1e-4](expected))
        assert_true(layer_g.num_parameters() == 3 * 2 * 9 + 3)


def test_conv_tt_mixed_sequential() raises:
    """ConvTT2D+MaxPool2d ride MixedSequential to GPU (shapes+gradflow)."""
    print("test_conv_tt_mixed_sequential")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(43)
        var model = MixedSequential()
        model.append(ConvTT2D[dtype](2, 3, 3, padding=1, init_seed=7))
        model.append(ReLU[dtype]())
        model.append(MaxPool2d[dtype](kernel_size=2))
        model.append(Flatten[dtype]())
        model.append(Linear[dtype](27, 5, init_method="he", bias_zero=True))
        assert_true(model.num_parameters() == (3 * 2 * 9 + 3) + (27 * 5 + 5))
        model.to_gpu()
        model.train()
        var x = Tensor[dtype].randn(1, 2, 6, 6)
        var out = model.forward[dtype, dtype](x.to_gpu())
        assert_true(out.is_on_gpu())
        assert_true(out.shape()[0] == 1 and out.shape()[1] == 5)
        var loss = out.sum()
        loss.backward()
        var params = model.parameters_of[dtype]()
        assert_true(len(params) == 4)  # conv w+b, linear w+b
        assert_true(params[0][].grad().is_on_gpu())


def test_conv_tt_teacher_student() raises:
    """GPU student overfits a CPU teacher's mapping (SGD end-to-end)."""
    print("test_conv_tt_teacher_student")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        seed(101)
        var x_c = Tensor[dtype].randn(32, 2, 8, 8)
        var k_t = Tensor[dtype].randn(2, 2, 3, 3)
        var b_t = _bias(2)
        # Fixed CPU teacher mapping (no grad needed on targets).
        var y_c = Conv[dtype].forward(x_c, k_t, b_t, 1, 1, 1, 1, 1, 1)
        var x_g = x_c.clone().to_gpu()
        var y_g = y_c.clone().to_gpu()
        var student = ConvTT2D[dtype](
            in_channels=2,
            out_channels=2,
            kernel_size=3,
            padding=1,
            init_seed=5,
        )
        student = student.to_gpu()
        var criterion = MSELoss[dtype]()
        var optimizer = SGD[dtype](student.parameters(), lr=0.02)
        var init_loss = Float32(0.0)
        var final_loss = Float32(0.0)
        for step in range(600):
            var pred = student(x_g)
            var loss = criterion(pred, y_g)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            var lv = loss.item()
            if step == 0:
                init_loss = lv
            final_loss = lv
        print("  teacher-student init/final:", init_loss, final_loss)
        assert_true(final_loss < 0.1 * init_loss)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
