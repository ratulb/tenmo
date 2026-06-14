from std.testing import assert_true, TestSuite
from tenmo import Tensor, Shape, Reduction, IntArray
from std.sys import has_accelerator
import std.math


# ═══════════════════════════════════════════════════════════════
#  HELPERS — hand-computed reference values
#
#  BCE:              loss_i = -[t*log(p+ε) + (1-t)*log(1-p+ε)]
#  BCE grad (mean):  dp_i   = [-(t/(p+ε)) + (1-t)/(1-p+ε)] / N
#  BCE grad (sum):   dp_i   = -(t/(p+ε)) + (1-t)/(1-p+ε)
#  BCE grad (none):  dp_i   = -(t/(p+ε)) + (1-t)/(1-p+ε)   (no scaling)
#
#  BCEWithLogits:    σ = sigmoid(logit),  loss_i = BCE(σ, t)
#  BWL grad (mean):  d/dlogit = (σ - t) / N
#  BWL grad (sum):   d/dlogit = (σ - t)
#  BWL grad (none):  d/dlogit = (σ - t)
# ═══════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────
#  CPU — BCE  forward  (mean / sum / none)
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  CPU — BCE  backward  (mean / sum / none)
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  CPU — BCE  all-ones / all-zeros targets
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  CPU — BCE  custom epsilon
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  CPU — BCE  float64
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  CPU — BCE  non-contiguous views  (scalar fallback)
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  CPU — BCEWithLogits  non-contiguous views
# ─────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════
#  GPU — BCE  (default grad flow: GPU → CPU)
# ═══════════════════════════════════════════════════════════════


def test_bce_gpu_1d_mean_forward_grad_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d1([2.0, -1.0, 0.5])
        var pred = logits.sigmoid()
        var pred2 = pred.detach()
        pred2.requires_grad_(True)
        var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
        var pred_gpu = pred2.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy(pred_gpu, target_gpu)
        assert_true(
            loss.to_cpu().all_close[atol=1e-4](
                Tensor[dtype].full(loss.to_cpu().shape(), 0.3048)
            )
        )
        loss.backward()
        # grad flows back to pred2 on CPU
        assert_true(
            pred2.grad().all_close[atol=1e-4](
                Tensor[dtype].d1([-0.3784, 0.4560, -0.5355])
            )
        )


def test_bce_gpu_1d_sum_forward_grad_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d1([2.0, -1.0, 0.5])
        var pred = logits.sigmoid()
        var pred2 = pred.detach()
        pred2.requires_grad_(True)
        var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
        var pred_gpu = pred2.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy(
            pred_gpu, target_gpu, reduction=Reduction("sum")
        )
        loss.backward()
        assert_true(
            pred2.grad().all_close[atol=1e-4](
                Tensor[dtype].d1([-1.1353, 1.3679, -1.6065])
            )
        )


def test_bce_gpu_1d_none_forward_grad_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d1([2.0, -1.0, 0.5])
        var pred = logits.sigmoid()
        var pred2 = pred.detach()
        pred2.requires_grad_(True)
        var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
        var pred_gpu = pred2.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy(
            pred_gpu, target_gpu, reduction=Reduction("none")
        )
        loss.backward()
        assert_true(
            pred2.grad().all_close[atol=1e-4](
                Tensor[dtype].d1([-1.1353, 1.3679, -1.6065])
            )
        )


def test_bce_gpu_2d_mean_forward_grad_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d2([[1.0, -1.0], [2.0, -2.0]])
        var pred = logits.sigmoid()
        var pred2 = pred.detach()
        pred2.requires_grad_(True)
        var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
        var pred_gpu = pred2.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy(pred_gpu, target_gpu)
        loss.backward()
        assert_true(
            pred2.grad().all_close[atol=1e-4](
                Tensor[dtype].d2([[-0.3420, 0.3420], [-0.2838, 0.2838]])
            )
        )


def test_bce_gpu_3d_mean_forward_grad_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d3(
            [[[1.0, -1.0], [2.0, -2.0]], [[0.5, -0.5], [1.5, -1.5]]]
        )
        var pred = logits.sigmoid()
        var pred2 = pred.detach()
        pred2.requires_grad_(True)
        var target = Tensor[dtype].d3(
            [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]
        )
        var pred_gpu = pred2.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy(pred_gpu, target_gpu)
        loss.backward()
        assert_true(
            pred2.grad().all_close[atol=1e-4](
                Tensor[dtype].d3(
                    [
                        [[-0.1710, 0.1710], [-0.1419, 0.1419]],
                        [[-0.2008, 0.2008], [-0.1529, 0.1529]],
                    ]
                )
            )
        )


# ─────────────────────────────────────────────
#  GPU — BCE  stop_grad=True  (grad stops at GPU tensor)
# ─────────────────────────────────────────────


def test_bce_gpu_1d_mean_stop_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d1([2.0, -1.0, 0.5])
        var pred = logits.sigmoid()
        var pred2 = pred.detach()
        pred2.requires_grad_(True)
        var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
        # stop_grad=True: grad does NOT flow back to pred2
        var pred_gpu = pred2.to_gpu(stop_grad=True)
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy(pred_gpu, target_gpu)
        loss.backward()
        # pred2 on CPU must NOT have received any grad (grad values stay zero)
        assert_true(
            pred2.grad().all_close(Tensor[dtype].zeros(pred2.shape()))
        )


def test_bce_gpu_2d_mean_stop_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d2([[1.0, -1.0], [2.0, -2.0]])
        var pred = logits.sigmoid()
        var pred2 = pred.detach()
        pred2.requires_grad_(True)
        var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
        var pred_gpu = pred2.to_gpu(stop_grad=True)
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy(pred_gpu, target_gpu)
        loss.backward()
        assert_true(
            pred2.grad().all_close(Tensor[dtype].zeros(pred2.shape()))
        )


# ═══════════════════════════════════════════════════════════════
#  GPU — BCEWithLogits  (default grad flow: GPU → CPU)
# ═══════════════════════════════════════════════════════════════


def test_bwl_gpu_1d_mean_forward_grad_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d1([2.0, -1.0, 0.5], requires_grad=True)
        var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
        var logits_gpu = logits.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu
        )
        assert_true(
            loss.to_cpu().all_close[atol=1e-4](
                Tensor[dtype].full(loss.to_cpu().shape(), 0.3048)
            )
        )
        loss.backward()
        assert_true(
            logits.grad().all_close[atol=1e-4](
                Tensor[dtype].d1([-0.0397, 0.0896, -0.1258])
            )
        )


def test_bwl_gpu_1d_sum_forward_grad_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d1([2.0, -1.0, 0.5], requires_grad=True)
        var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
        var logits_gpu = logits.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu, reduction=Reduction("sum")
        )
        loss.backward()
        assert_true(
            logits.grad().all_close[atol=1e-4](
                Tensor[dtype].d1([-0.1192, 0.2689, -0.3775])
            )
        )


def test_bwl_gpu_1d_none_forward_grad_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d1([2.0, -1.0, 0.5], requires_grad=True)
        var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
        var logits_gpu = logits.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu, reduction=Reduction("none")
        )
        loss.backward()
        assert_true(
            logits.grad().all_close[atol=1e-4](
                Tensor[dtype].d1([-0.1192, 0.2689, -0.3775])
            )
        )


def test_bwl_gpu_2d_mean_forward_grad_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d2(
            [[1.0, -1.0], [2.0, -2.0]], requires_grad=True
        )
        var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
        var logits_gpu = logits.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu
        )
        loss.backward()
        assert_true(
            logits.grad().all_close[atol=1e-4](
                Tensor[dtype].d2([[-0.0672, 0.0672], [-0.0298, 0.0298]])
            )
        )


def test_bwl_gpu_2d_sum_forward_grad_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d2(
            [[1.0, -1.0], [2.0, -2.0]], requires_grad=True
        )
        var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
        var logits_gpu = logits.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu, reduction=Reduction("sum")
        )
        loss.backward()
        assert_true(
            logits.grad().all_close[atol=1e-4](
                Tensor[dtype].d2([[-0.2689, 0.2689], [-0.1192, 0.1192]])
            )
        )


def test_bwl_gpu_2d_none_forward_grad_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d2(
            [[1.0, -1.0], [2.0, -2.0]], requires_grad=True
        )
        var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
        var logits_gpu = logits.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu, reduction=Reduction("none")
        )
        loss.backward()
        assert_true(
            logits.grad().all_close[atol=1e-4](
                Tensor[dtype].d2([[-0.2689, 0.2689], [-0.1192, 0.1192]])
            )
        )


def test_bwl_gpu_3d_mean_forward_grad_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d3(
            [[[1.0, -1.0], [2.0, -2.0]], [[0.5, -0.5], [1.5, -1.5]]],
            requires_grad=True,
        )
        var target = Tensor[dtype].d3(
            [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]
        )
        var logits_gpu = logits.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu
        )
        loss.backward()
        assert_true(
            logits.grad().all_close[atol=1e-4](
                Tensor[dtype].d3(
                    [
                        [[-0.0336, 0.0336], [-0.0149, 0.0149]],
                        [[-0.0472, 0.0472], [-0.0228, 0.0228]],
                    ]
                )
            )
        )


# ─────────────────────────────────────────────
#  GPU — BCEWithLogits  stop_grad=True
# ─────────────────────────────────────────────


def test_bwl_gpu_1d_mean_stop_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d1([2.0, -1.0, 0.5], requires_grad=True)
        var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
        var logits_gpu = logits.to_gpu(stop_grad=True)
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu
        )
        loss.backward()
        assert_true(
            logits.grad().all_close(Tensor[dtype].zeros(logits.shape()))
        )


def test_bwl_gpu_1d_sum_stop_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d1([2.0, -1.0, 0.5], requires_grad=True)
        var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
        var logits_gpu = logits.to_gpu(stop_grad=True)
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu, reduction=Reduction("sum")
        )
        loss.backward()
        assert_true(
            logits.grad().all_close(Tensor[dtype].zeros(logits.shape()))
        )


def test_bwl_gpu_1d_none_stop_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d1([2.0, -1.0, 0.5], requires_grad=True)
        var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
        var logits_gpu = logits.to_gpu(stop_grad=True)
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu, reduction=Reduction("none")
        )
        loss.backward()
        assert_true(
            logits.grad().all_close(Tensor[dtype].zeros(logits.shape()))
        )


def test_bwl_gpu_2d_mean_stop_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d2(
            [[1.0, -1.0], [2.0, -2.0]], requires_grad=True
        )
        var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
        var logits_gpu = logits.to_gpu(stop_grad=True)
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu
        )
        loss.backward()
        assert_true(
            logits.grad().all_close(Tensor[dtype].zeros(logits.shape()))
        )


def test_bwl_gpu_3d_mean_stop_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d3(
            [[[1.0, -1.0], [2.0, -2.0]], [[0.5, -0.5], [1.5, -1.5]]],
            requires_grad=True,
        )
        var target = Tensor[dtype].d3(
            [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]
        )
        var logits_gpu = logits.to_gpu(stop_grad=True)
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu
        )
        loss.backward()
        assert_true(
            logits.grad().all_close(Tensor[dtype].zeros(logits.shape()))
        )


# ─────────────────────────────────────────────
#  GPU — numerical stability on GPU matches CPU
# ─────────────────────────────────────────────


def test_bwl_gpu_large_logits_stable() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d1([50.0, 100.0], requires_grad=True)
        var target = Tensor[dtype].d1([1.0, 1.0])
        var logits_gpu = logits.to_gpu()
        var target_gpu = target.to_gpu()
        var loss = Tensor[dtype].binary_cross_entropy_with_logits(
            logits_gpu, target_gpu
        )
        assert_true(
            loss.to_cpu().all_close[atol=1e-4](
                Tensor[dtype].full(loss.to_cpu().shape(), 0.0)
            )
        )


def test_bwl_gpu_bce_matches_bwl() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits_raw = Tensor[dtype].d1([2.0, -1.0, 0.5])
        var sigmoided = logits_raw.sigmoid()
        var pred = sigmoided.detach()
        pred.requires_grad_(True)
        var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
        var pred_gpu = pred.to_gpu()
        var target_gpu = target.to_gpu()
        var loss_bce = Tensor[dtype].binary_cross_entropy(pred_gpu, target_gpu)

        var logits2 = Tensor[dtype].d1([2.0, -1.0, 0.5], requires_grad=True)
        var logits2_gpu = logits2.to_gpu()
        var loss_bwl = Tensor[dtype].binary_cross_entropy_with_logits(
            logits2_gpu, target_gpu
        )
        assert_true(loss_bce.to_cpu().all_close[atol=1e-4](loss_bwl.to_cpu()))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
