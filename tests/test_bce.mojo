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


def test_bce_cpu_1d_mean_forward() raises:
    comptime dtype = DType.float32
    # p = sigmoid([2.0, -1.0, 0.5]) ≈ [0.8808, 0.2689, 0.6225]
    var logits = Tensor[dtype].d1([2.0, -1.0, 0.5])
    var pred = logits.sigmoid()
    var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
    var loss = Tensor[dtype].binary_cross_entropy(pred, target)
    # loss_i: -log(0.8808), -log(1-0.2689), -log(0.6225)
    # ≈ 0.1269, 0.3133, 0.4741  → mean ≈ 0.3048
    assert_true(
        loss.all_close[atol=1e-4](Tensor[dtype].full(loss.shape(), 0.3048))
    )


def test_bce_cpu_1d_sum_forward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d1([2.0, -1.0, 0.5])
    var pred = logits.sigmoid()
    var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
    var loss = Tensor[dtype].binary_cross_entropy(
        pred, target, reduction=Reduction("sum")
    )
    # sum ≈ 0.9143
    assert_true(
        loss.all_close[atol=1e-4](Tensor[dtype].full(loss.shape(), 0.9143))
    )


def test_bce_cpu_1d_none_forward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d1([2.0, -1.0, 0.5])
    var pred = logits.sigmoid()
    var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
    var loss = Tensor[dtype].binary_cross_entropy(
        pred, target, reduction=Reduction("none")
    )
    assert_true(
        loss.all_close[atol=1e-4](Tensor[dtype].d1([0.1269, 0.3133, 0.4741]))
    )


def test_bce_cpu_2d_mean_forward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[1.0, -1.0], [2.0, -2.0]])
    var pred = logits.sigmoid()
    var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
    var loss = Tensor[dtype].binary_cross_entropy(pred, target)
    # σ: [[0.7311,0.2689],[0.8808,0.1192]]
    # loss_i: [0.3133,0.3133,0.1269,0.1269] → mean=0.2201
    assert_true(
        loss.all_close[atol=1e-4](Tensor[dtype].full(loss.shape(), 0.2201))
    )


def test_bce_cpu_2d_sum_forward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[1.0, -1.0], [2.0, -2.0]])
    var pred = logits.sigmoid()
    var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
    var loss = Tensor[dtype].binary_cross_entropy(
        pred, target, reduction=Reduction("sum")
    )
    # sum ≈ 0.8804
    assert_true(
        loss.all_close[atol=1e-4](Tensor[dtype].full(loss.shape(), 0.8804))
    )


def test_bce_cpu_2d_none_forward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[1.0, -1.0], [2.0, -2.0]])
    var pred = logits.sigmoid()
    var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
    var loss = Tensor[dtype].binary_cross_entropy(
        pred, target, reduction=Reduction("none")
    )
    assert_true(
        loss.all_close[atol=1e-4](
            Tensor[dtype].d2([[0.3133, 0.3133], [0.1269, 0.1269]])
        )
    )


def test_bce_cpu_3d_mean_forward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d3(
        [[[1.0, -1.0], [2.0, -2.0]], [[0.5, -0.5], [1.5, -1.5]]]
    )
    var pred = logits.sigmoid()
    var target = Tensor[dtype].d3(
        [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]
    )
    var loss = Tensor[dtype].binary_cross_entropy(pred, target)
    # All (logit,target) pairs are symmetric: loss_i = softplus(-logit) for t=1
    # mean of [0.3133,0.3133,0.1269,0.1269,0.4741,0.4741,0.2014,0.2014] ≈ 0.2789
    assert_true(
        loss.all_close[atol=1e-4](Tensor[dtype].full(loss.shape(), 0.2789))
    )


# ─────────────────────────────────────────────
#  CPU — BCE  backward  (mean / sum / none)
# ─────────────────────────────────────────────


def test_bce_cpu_1d_mean_backward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d1([2.0, -1.0, 0.5])
    var pred = logits.sigmoid()
    var pred2 = pred.detach()
    pred2.requires_grad_(True)
    var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
    var loss = Tensor[dtype].binary_cross_entropy(pred2, target)
    loss.backward()
    assert_true(
        pred2.grad().all_close[atol=1e-4](
            Tensor[dtype].d1([-0.3784, 0.4560, -0.5355])
        )
    )


def test_bce_cpu_1d_sum_backward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d1([2.0, -1.0, 0.5])
    var pred = logits.sigmoid()
    var pred2 = pred.detach()
    pred2.requires_grad_(True)
    var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
    var loss = Tensor[dtype].binary_cross_entropy(
        pred2, target, reduction=Reduction("sum")
    )
    loss.backward()
    # no /N scaling — raw BCE grad
    assert_true(
        pred2.grad().all_close[atol=1e-4](
            Tensor[dtype].d1([-1.1353, 1.3679, -1.6065])
        )
    )


def test_bce_cpu_1d_none_backward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d1([2.0, -1.0, 0.5])
    var pred = logits.sigmoid()
    var pred2 = pred.detach()
    pred2.requires_grad_(True)
    var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
    var loss = Tensor[dtype].binary_cross_entropy(
        pred2, target, reduction=Reduction("none")
    )
    loss.backward()
    # same as sum — no mean, no reduction, seed=1
    assert_true(
        pred2.grad().all_close[atol=1e-4](
            Tensor[dtype].d1([-1.1353, 1.3679, -1.6065])
        )
    )


def test_bce_cpu_2d_mean_backward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[1.0, -1.0], [2.0, -2.0]])
    var pred = logits.sigmoid()
    var pred2 = pred.detach()
    pred2.requires_grad_(True)
    var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
    var loss = Tensor[dtype].binary_cross_entropy(pred2, target)
    loss.backward()
    # σ: [[0.7311,0.2689],[0.8808,0.1192]]
    # raw: [[-1.3679,1.3679],[-1.1353,1.1353]] / 4
    assert_true(
        pred2.grad().all_close[atol=1e-4](
            Tensor[dtype].d2([[-0.3420, 0.3420], [-0.2838, 0.2838]])
        )
    )


def test_bce_cpu_2d_sum_backward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[1.0, -1.0], [2.0, -2.0]])
    var pred = logits.sigmoid()
    var pred2 = pred.detach()
    pred2.requires_grad_(True)
    var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
    var loss = Tensor[dtype].binary_cross_entropy(
        pred2, target, reduction=Reduction("sum")
    )
    loss.backward()
    assert_true(
        pred2.grad().all_close[atol=1e-4](
            Tensor[dtype].d2([[-1.3679, 1.3679], [-1.1353, 1.1353]])
        )
    )


def test_bce_cpu_2d_none_backward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[1.0, -1.0], [2.0, -2.0]])
    var pred = logits.sigmoid()
    var pred2 = pred.detach()
    pred2.requires_grad_(True)
    var target = Tensor[dtype].d2([[1.0, 0.0], [1.0, 0.0]])
    var loss = Tensor[dtype].binary_cross_entropy(
        pred2, target, reduction=Reduction("none")
    )
    loss.backward()
    assert_true(
        pred2.grad().all_close[atol=1e-4](
            Tensor[dtype].d2([[-1.3679, 1.3679], [-1.1353, 1.1353]])
        )
    )


def test_bce_cpu_3d_mean_backward() raises:
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
    var loss = Tensor[dtype].binary_cross_entropy(pred2, target)
    loss.backward()
    # raw grads / 8
    # σ vals: [0.7311,0.2689,0.8808,0.1192,0.6225,0.3775,0.8176,0.1824]
    # raw: [-1.3679,1.3679,-1.1353,1.1353,-1.6065,1.6065,-1.2232,1.2232]
    # /8:  [-0.1710,0.1710,-0.1419,0.1419,-0.2008,0.2008,-0.1529,0.1529]
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
#  CPU — BCE  all-ones / all-zeros targets
# ─────────────────────────────────────────────


def test_bce_cpu_all_ones_target_forward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var pred = logits.sigmoid()
    var target = Tensor[dtype].ones_like(pred)
    var loss = Tensor[dtype].binary_cross_entropy(pred, target)
    # loss_i = -log(σ(x)) = softplus(-x)
    # ≈ [0.3133, 0.1269, 0.0486] → mean ≈ 0.1629
    assert_true(
        loss.all_close[atol=1e-4](Tensor[dtype].full(loss.shape(), 0.1629))
    )


def test_bce_cpu_all_zeros_target_forward() raises:
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var pred = logits.sigmoid()
    var target = Tensor[dtype].zeros_like(pred)
    var loss = Tensor[dtype].binary_cross_entropy(pred, target)
    # loss_i = -log(1-σ(x)) = softplus(x)
    # ≈ [1.3133, 2.1269, 3.0486] → mean ≈ 2.1629
    assert_true(
        loss.all_close[atol=1e-4](Tensor[dtype].full(loss.shape(), 2.1629))
    )


# ─────────────────────────────────────────────
#  CPU — BCE  custom epsilon
# ─────────────────────────────────────────────


def test_bce_cpu_custom_epsilon_forward() raises:
    comptime dtype = DType.float32
    # use extreme logits so sigmoid values near 0/1 are affected by epsilon
    var logits = Tensor[dtype].d1([10.0, -10.0])
    var pred = logits.sigmoid()
    var target = Tensor[dtype].d1([1.0, 0.0])
    var loss_default = Tensor[dtype].binary_cross_entropy(pred, target)
    var loss_eps = Tensor[dtype].binary_cross_entropy(
        pred, target, epsilon=Scalar[dtype](0.1)
    )
    assert_true(not loss_default.all_close[atol=1e-4](loss_eps))


# ─────────────────────────────────────────────
#  CPU — BCE  float64
# ─────────────────────────────────────────────


def test_bce_cpu_float64_mean_forward_backward() raises:
    comptime dtype = DType.float64
    var logits = Tensor[dtype].d1([2.0, -1.0, 0.5])
    var pred = logits.sigmoid()
    var pred2 = pred.detach()
    pred2.requires_grad_(True)
    var target = Tensor[dtype].d1([1.0, 0.0, 1.0])
    var loss = Tensor[dtype].binary_cross_entropy(pred2, target)
    assert_true(
        loss.all_close[atol=1e-6](Tensor[dtype].full(loss.shape(), 0.304756))
    )
    loss.backward()
    assert_true(
        pred2.grad().all_close[atol=1e-6](
            Tensor[dtype].d1([-0.378445, 0.455960, -0.535510])
        )
    )


# ─────────────────────────────────────────────
#  CPU — BCE  non-contiguous views  (scalar fallback)
# ─────────────────────────────────────────────


def test_bce_cpu_view_forward_matches_contiguous() raises:
    """BCELoss forward on non-contiguous (transposed) input matches contiguous.
    Exercises scalar fallback in bce_forward_reduce_cpu."""
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, -1.0, 0.5], [1.0, -2.0, -0.5]])
    var pred = logits.sigmoid()
    var target = Tensor[dtype].d2([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    var loss_contig = Tensor[dtype].binary_cross_entropy(pred, target)
    # transpose both — non-contiguous view exercises scalar fallback
    var loss_view = Tensor[dtype].binary_cross_entropy(
        pred.transpose(), target.transpose()
    )
    assert_true(loss_contig.all_close[atol=1e-4](loss_view))


def test_bce_cpu_view_backward_runs() raises:
    """BCELoss backward on non-contiguous (transposed) prediction runs and produces grad.
    """
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, -1.0, 0.5], [1.0, -2.0, -0.5]])
    var pred = logits.sigmoid()
    var target = Tensor[dtype].d2([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    var pred_v = pred.detach()
    var pred_v_t = pred_v.transpose()
    pred_v_t.requires_grad_(True)
    var loss = Tensor[dtype].binary_cross_entropy(pred_v_t, target.transpose())
    loss.backward()
    assert_true(loss.requires_grad)
    assert_true(pred_v_t.requires_grad)
    assert_true(pred_v_t.has_grad())


# ─────────────────────────────────────────────
#  CPU — BCEWithLogits  non-contiguous views
# ─────────────────────────────────────────────


def test_bwl_cpu_view_backward_runs() raises:
    """BCEWithLogits backward on non-contiguous (transposed) logits runs and produces grad.
    """
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, -1.0, 0.5], [1.0, -2.0, -0.5]])
    var target = Tensor[dtype].d2([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    var logits_v = logits.detach()
    var logits_v_t = logits_v.transpose()
    logits_v_t.requires_grad_(True)
    var loss = Tensor[dtype].binary_cross_entropy_with_logits(
        logits_v_t, target.transpose()
    )
    loss.backward()
    assert_true(logits_v_t.has_grad())


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
