from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite

# ===----------------------------------------------------------------------=== #
# fake_quant (STE) tests — prefix: fq_
#
#     out = scale * clamp(round(x / scale), qmin, qmax)
#
# The crux is the GRADIENT, not the forward. Because k = clamp(round(x/s)) is
# piecewise constant, the TRUE derivative w.r.t. x is 0 a.e. The STE overrides
# it to 1 while leaving d/d(scale) = k (the true value).
#
# Every expectation below was hand-verified in Python before any Mojo was
# written, so a failure is the kernel's fault, not bad arithmetic.
# Half-to-even rounding throughout (Mojo's round is banker's rounding):
# round(63.5) = 64, round(-63.5) = -64.
#
# Scale is either a SCALAR (per-tensor, gradient reduces to a scalar) or the
# SAME SHAPE as x (per-element). Per-channel (1,C) is not implemented yet and
# panics -- see tenmo/fakequant.mojo.
# ===----------------------------------------------------------------------=== #

comptime F32 = DType.float32
comptime F64 = DType.float64

comptime QMIN = Scalar[F32](-128.0)
comptime QMAX = Scalar[F32](127.0)
comptime QMIN64 = Scalar[F64](-128.0)
comptime QMAX64 = Scalar[F64](127.0)

comptime S127 = Float32(1.0 / 127.0)


# ===----------------------------------------------------------------------=== #
# Forward
# ===----------------------------------------------------------------------=== #


def test_fq_cpu_forward_basic() raises:
    # s = 1/127: z = x/s = 46.99, -46.99, 63.5, 0
    #   -> k = 47, -47, 64 (half-to-even), 0  -> out = s*k
    var s = Tensor[F32].scalar(S127)
    var x = Tensor[F32].d1([0.37, -0.37, 0.5, 0.0])
    var y = x.fake_quant[track_grad=False](s, QMIN, QMAX)
    var expected = Tensor[F32].d1([47.0, -47.0, 64.0, 0.0]) * S127
    assert_true(y.all_close[atol=1e-6](expected))


def test_fq_cpu_forward_per_element_scale() raises:
    # Same result, but the scale has x's shape, so no reduction on backward.
    var x = Tensor[F32].d1([0.37, -0.37, 0.5, 0.0])
    var s = Tensor[F32].d1([S127, S127, S127, S127])
    var y = x.fake_quant[track_grad=False](s, QMIN, QMAX)
    var expected = Tensor[F32].d1([47.0, -47.0, 64.0, 0.0]) * S127
    assert_true(y.all_close[atol=1e-6](expected))


def test_fq_cpu_forward_saturates_high() raises:
    var s = Tensor[F32].scalar(0.01)
    var x = Tensor[F32].scalar(5.0)
    var y = x.fake_quant[track_grad=False](s, QMIN, QMAX)
    # z = 500 -> clamps to k = 127 -> out = 1.27
    assert_true(y.all_close[atol=1e-6](Tensor[F32].scalar(1.27)))


def test_fq_cpu_forward_saturates_low() raises:
    var s = Tensor[F32].scalar(0.01)
    var x = Tensor[F32].scalar(-5.0)
    var y = x.fake_quant[track_grad=False](s, QMIN, QMAX)
    # z = -500 -> clamps to k = -128 -> out = -1.28
    assert_true(y.all_close[atol=1e-6](Tensor[F32].scalar(-1.28)))


def test_fq_cpu_forward_no_track_grad_is_leaf() raises:
    var s = Tensor[F32].scalar(S127)
    var x = Tensor[F32].d1([0.37, 0.5, 1.0, -1.0], requires_grad=True)
    var y = x.fake_quant[track_grad=False](s, QMIN, QMAX)
    assert_true(not y.has_ancestry())


def test_fq_f64_forward() raises:
    var s = Tensor[F64].scalar(Float64(1.0 / 127.0))
    var x = Tensor[F64].d1([0.37, -0.37])
    var y = x.fake_quant[track_grad=False](s, QMIN64, QMAX64)
    var expected = Tensor[F64].d1([47.0, -47.0]) * Float64(1.0 / 127.0)
    assert_true(y.all_close(expected))


def test_fq_asymmetric_qmin_qmax() raises:
    # uint8-style: levels 0..255
    var s = Tensor[F32].scalar(0.01)
    var x = Tensor[F32].d1([2.5, -2.5])
    var y = x.fake_quant[track_grad=False](s, Scalar[F32](0.0), Scalar[F32](255.0))
    # z = 250, -250 -> k = 250, 0 (low end clamps to 0) -> out = 2.5, 0.0
    var expected = Tensor[F32].d1([2.5, 0.0])
    assert_true(y.all_close[atol=1e-6](expected))


# ===----------------------------------------------------------------------=== #
# The STE gradient — d/dx = upstream EXACTLY
# ===----------------------------------------------------------------------=== #


def test_fq_ste_grad_x_is_exactly_upstream() raises:
    # out = fq(x) * w, summed. d/dx_i = w_i exactly, because the STE sets it
    # to 1. If the op were honest (true derivative) this would be all zeros.
    var s = Tensor[F32].scalar(S127)
    var x = Tensor[F32].d1([0.37, -0.37, 0.5, 2.5], requires_grad=True)
    var w = Tensor[F32].d1([2.0, 3.0, -1.0, 0.5], requires_grad=True)
    var y = (x.fake_quant(s, QMIN, QMAX) * w).sum()
    y.backward()
    assert_true(x.grad().all_close(w))


def test_fq_ste_grad_x_nonzero_where_true_derivative_is_zero() raises:
    # Every element sits strictly inside a quantization cell (z = 7.4, -7.4,
    # 14.8, -14.8 with s=0.05), so the TRUE d/dx is 0 for every one. A non-zero
    # gradient can only come from the STE.
    var s = Tensor[F32].scalar(0.05)
    var x = Tensor[F32].d1([0.37, -0.37, 0.74, -0.74], requires_grad=True)
    var y = x.fake_quant(s, QMIN, QMAX).sum()
    y.backward()
    # d/dx of a sum is 1 per element.
    assert_true(x.grad().all_close(Tensor[F32].d1([1.0, 1.0, 1.0, 1.0])))


def test_fq_grad_x_survives_saturation() raises:
    # Saturated cells are still inside the op, so the STE applies there too.
    var s = Tensor[F32].scalar(0.01)
    var x = Tensor[F32].d1([5.0, -5.0, 0.0], requires_grad=True)
    var y = x.fake_quant(s, QMIN, QMAX).sum()
    y.backward()
    assert_true(x.grad().all_close(Tensor[F32].d1([1.0, 1.0, 1.0])))


# ===----------------------------------------------------------------------=== #
# d/dscale = k  (the TRUE derivative — not overridden)
# ===----------------------------------------------------------------------=== #


def test_fq_grad_scale_reduces_to_k_per_tensor() raises:
    # sum(fq(x, s)) with s scalar -> d L/ds = sum_i k_i.
    # s = 1/127: k = 47, -47, 64, 0  =>  sum k = 64
    var s = Tensor[F32].scalar(S127, requires_grad=True)
    var x = Tensor[F32].d1([0.37, -0.37, 0.5, 0.0])
    var y = x.fake_quant(s, QMIN, QMAX).sum()
    y.backward()
    assert_true(s.grad().all_close[atol=1e-3](Tensor[F32].scalar(64.0)))


def test_fq_grad_scale_includes_saturation() raises:
    # k is the CLAMPED level, so saturated cells contribute qmax/qmin rather
    # than the unclamped round() value. z = 500, -500 -> k = 127, -128, sum -1.
    var s = Tensor[F32].scalar(0.01, requires_grad=True)
    var x = Tensor[F32].d1([5.0, -5.0])
    var y = x.fake_quant(s, QMIN, QMAX).sum()
    y.backward()
    assert_true(s.grad().all_close[atol=1e-3](Tensor[F32].scalar(-1.0)))


def test_fq_grad_scale_per_element_needs_no_reduction() raises:
    # Same math with a per-element scale: d L/ds_i = upstream_i * k_i.
    # s = 1/127, weights all 1 -> k = 47, -47, 64, 0 elementwise.
    var s = Tensor[F32].d1([S127, S127, S127, S127], requires_grad=True)
    var x = Tensor[F32].d1([0.37, -0.37, 0.5, 0.0])
    var y = x.fake_quant(s, QMIN, QMAX).sum()
    y.backward()
    var expected = Tensor[F32].d1([47.0, -47.0, 64.0, 0.0])
    assert_true(s.grad().all_close[atol=1e-3](expected))


def test_fq_true_dx_is_zero_inside_a_cell() raises:
    # The TRUE derivative the STE overrides, demonstrated directly: inside one
    # cell the dequantized output is exactly s*k and does not depend on x, so
    # perturbing x by +-1e-4 changes nothing. FD on this function is only valid
    # strictly inside a cell -- round() is discontinuous, and straddling a step
    # returns garbage (an early draft of the design math came back with the
    # wrong sign for d/ds for exactly that reason).
    var s = Tensor[F32].scalar(0.05)
    var x = Tensor[F32].scalar(0.37)  # z = 7.4, interior to cell k=7
    var h = Float32(1e-4)
    var plus = (x + h).fake_quant[track_grad=False](s, QMIN, QMAX)
    var minus = (x - h).fake_quant[track_grad=False](s, QMIN, QMAX)
    assert_true(plus.all_close[atol=1e-6](minus))


def test_fq_grad_scale_fd_value_interior_cell() raises:
    # d out/ds = k = 7 for x=0.37, s=0.05 (z=7.4, interior to cell k=7).
    # Verified by finite difference in Python: FD d/ds = +7.000000.
    var s = Tensor[F32].scalar(0.05, requires_grad=True)
    var x = Tensor[F32].scalar(0.37)
    var y = x.fake_quant(s, QMIN, QMAX).sum()
    y.backward()
    assert_true(s.grad().all_close[atol=1e-3](Tensor[F32].scalar(7.0)))


# ===----------------------------------------------------------------------=== #
# Two parents, two rules — both receive gradient in one backward
# ===----------------------------------------------------------------------=== #


def test_fq_both_parents_receive_gradient() raises:
    # s=0.05: z = 7.4, -7.4, 14.8, 22.2 -> k = 7, -7, 15, 22
    # y = (fq(x) * w).sum() with w = 1,2,3,4
    #   dL/dx_i   = w_i                      (STE)
    #   dL/ds     = sum_i w_i * k_i = 7 - 14 + 45 + 88 = 126
    var s = Tensor[F32].scalar(0.05, requires_grad=True)
    var x = Tensor[F32].d1([0.37, -0.37, 0.74, 1.11], requires_grad=True)
    var w = Tensor[F32].d1([1.0, 2.0, 3.0, 4.0])
    var y = (x.fake_quant(s, QMIN, QMAX) * w).sum()
    y.backward()
    assert_true(x.grad().all_close(w))
    assert_true(s.grad().all_close[atol=1e-2](Tensor[F32].scalar(126.0)))


def test_fq_repeat_backward_reruns() raises:
    # backward() never frees ancestry, so a second call must reproduce the same
    # gradients (documented Tensor.backward behaviour).
    var s = Tensor[F32].scalar(0.05, requires_grad=True)
    var x = Tensor[F32].d1([0.37, 0.74], requires_grad=True)
    var y = x.fake_quant(s, QMIN, QMAX).sum()
    y.backward()
    var gx1 = x.grad().clone()
    var gs1 = s.grad().clone()
    # backward() never frees ancestry and LEAF GRADS ACCUMULATE across calls,
    # so the second run doubles them rather than reproducing them.
    y.backward()
    assert_true(x.grad().all_close[atol=1e-3](gx1 + gx1))
    assert_true(s.grad().all_close[atol=1e-3](gs1 + gs1))


def test_fq_eval_erases_graph() raises:
    # track_grad=False must produce a leaf even from a grad-requiring input.
    var s = Tensor[F32].scalar(0.05, requires_grad=True)
    var x = Tensor[F32].d1([0.37, 0.74], requires_grad=True)
    var y = x.fake_quant[track_grad=False](s, QMIN, QMAX)
    assert_true(not y.has_ancestry())


# ===----------------------------------------------------------------------=== #
# Chaining — the op must compose with the rest of the graph
# ===----------------------------------------------------------------------=== #


def test_fq_chains_into_matmul() raises:
    # s=0.05: k = [[7,-7],[15,22]] -> fq(x) = s*k = [[0.35,-0.35],[0.75,1.10]]
    # loss = sum(fq(x) @ w) => dL/dw = fq(x)^T @ ones = fq(x)^T, shape (2,2).
    var s = Tensor[F32].scalar(0.05)
    var x = Tensor[F32].d2([[0.37, -0.37], [0.74, 1.11]], requires_grad=True)
    var w = Tensor[F32].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.fake_quant(s, QMIN, QMAX).matmul(w).sum()
    y.backward()
    # loss = sum(y), so dL/dy[a,j] = 1 for all a,j. With F = fq(x):
    #   dL/dF[a,i] = sum_j w[i,j] = row sum of w  -> 3, 7  (varies with i)
    #   dL/dw[i,j] = sum_a F[a,j] = col sum of F  -> 1.1, 0.75 (varies with j)
    # The STE then passes dL/dF straight through to x.
    # F = [[0.35, -0.35], [0.75, 1.10]], so col sums are 1.1 and 0.75.
    var expected_w = Tensor[F32].d2([[1.1, 1.1], [0.75, 0.75]])
    assert_true(w.grad().all_close[atol=1e-3](expected_w))
    var expected_x = Tensor[F32].d2([[3.0, 7.0], [3.0, 7.0]])
    assert_true(x.grad().all_close(expected_x))


def test_fq_chains_through_relu() raises:
    # ReLU is downstream of the quantizer: gradient must pass through both.
    var s = Tensor[F32].scalar(0.05, requires_grad=True)
    var x = Tensor[F32].d1([0.37, -0.37, 0.74, 1.11], requires_grad=True)
    var y = x.fake_quant(s, QMIN, QMAX).relu().sum()
    y.backward()
    # relu gate: k = 7, -7, 15, 22 -> out = 0.35, -0.35, 0.75, 1.10
    #          gate = 1, 0, 1, 1
    assert_true(x.grad().all_close(Tensor[F32].d1([1.0, 0.0, 1.0, 1.0])))
    # dL/ds = 1*7 + 0*(-7) + 1*15 + 1*22 = 44
    assert_true(s.grad().all_close[atol=1e-2](Tensor[F32].scalar(44.0)))


def test_fq_two_quantizers_in_sequence() raises:
    # Quantizing twice is idempotent-ish: the second pass sees s*k, so z is
    # integral and round is a no-op. Gradient still flows through both nodes.
    var s = Tensor[F32].scalar(0.05)
    var x = Tensor[F32].d1([0.37, 0.74], requires_grad=True)
    var y = x.fake_quant(s, QMIN, QMAX).fake_quant(s, QMIN, QMAX).sum()
    y.backward()
    assert_true(x.grad().all_close(Tensor[F32].d1([1.0, 1.0])))


# ===----------------------------------------------------------------------=== #
# Shapes
# ===----------------------------------------------------------------------=== #


def test_fq_2d_input_scalar_scale() raises:
    var s = Tensor[F32].scalar(0.05)
    var x = Tensor[F32].d2([[0.37, -0.37], [0.74, 1.11]], requires_grad=True)
    var y = x.fake_quant(s, QMIN, QMAX).sum()
    y.backward()
    assert_true(x.grad().all_close(Tensor[F32].d2([[1.0, 1.0], [1.0, 1.0]])))


def test_fq_2d_input_2d_scale() raises:
    var s = Tensor[F32].d2([[0.05, 0.05], [0.05, 0.05]], requires_grad=True)
    var x = Tensor[F32].d2([[0.37, -0.37], [0.74, 1.11]], requires_grad=True)
    var y = x.fake_quant(s, QMIN, QMAX).sum()
    y.backward()
    # k = [[7,-7],[15,22]] -> per-element scale grad = k (upstream all 1)
    var expected_s = Tensor[F32].d2([[7.0, -7.0], [15.0, 22.0]])
    assert_true(s.grad().all_close[atol=1e-2](expected_s))
    assert_true(x.grad().all_close(Tensor[F32].d2([[1.0, 1.0], [1.0, 1.0]])))


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
