# =============================================================================
# test_product_reduction.mojo
#
# Exhaustive tests for product(axes, keepdims) forward + backward.
# All test names carry prefix: prd_
#
# Coverage:
#   Forward:    all-positive, all-negative, mixed signs, zeros (one/two+),
#               keepdims, multi-axis, scalar, dtype variants
#   Backward:   mask correctness, zero-count cases, shape preservation
#   Grad flow:  chained ops, no-grad tensors, store vs recompute excl_product
#   Dimensions: 0-d (scalar), 1-d, 2-d, 3-d, 4-d, non-contiguous
#   Devices:    CPU (all tests), GPU (guarded with has_accelerator())
#   Parity:     CPU vs GPU numerical agreement
#
# Notes:
#   - Each test creates fresh tensors — no shared state across tests.
#   - GPU parity tests run TWO separate backward passes (cpu then gpu).
#     Each pass accumulates into its own leaf tensor grad (no sharing).
#   - all_close uses atol=1e-4 for float32 log-space results.
# =============================================================================

from std.sys import has_accelerator
from std.testing import assert_true, TestSuite, assert_equal, assert_false
from tenmo.tensor import Tensor
from std.math import abs
from tenmo.intarray import IntArray
from tenmo.shapes import Shape
from tenmo.permute import Permute
from tenmo.device import CPU, GPU
from tenmo.numpy_interop import to_ndarray, from_ndarray

# =============================================================================
# ── SECTION 1: CPU FORWARD ───────────────────────────────────────────────────
# =============================================================================


# =============================================================================
# ── SECTION 2: CPU BACKWARD ──────────────────────────────────────────────────
# =============================================================================


# =============================================================================
# ── SECTION 3: CPU GRAD FLOW ─────────────────────────────────────────────────
# =============================================================================


# =============================================================================
# ── SECTION 4: CPU NON-CONTIGUOUS ────────────────────────────────────────────
# =============================================================================


# =============================================================================
# ── SECTION 5: GPU FORWARD ───────────────────────────────────────────────────
# =============================================================================


def test_prd_gpu_fwd_all_positive_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].scalar(24.0))
        )


def test_prd_gpu_fwd_all_negative_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([-2.0, -3.0, -4.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].scalar(-24.0))
        )


def test_prd_gpu_fwd_mixed_signs_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([-2.0, 3.0, -4.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].scalar(24.0))
        )


def test_prd_gpu_fwd_single_zero() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 0.0, 4.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].scalar(0.0))
        )


def test_prd_gpu_fwd_two_zeros() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 0.0, 0.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].scalar(0.0))
        )


def test_prd_gpu_fwd_2d_axis0() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].d1([3.0, 8.0]))
        )


def test_prd_gpu_fwd_2d_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var out = a.product(IntArray(1))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].d1([2.0, 12.0]))
        )


def test_prd_gpu_fwd_2d_keepdims() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var out = a.product(IntArray(1), keepdims=True)
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].d2([[2.0], [12.0]]))
        )


def test_prd_gpu_fwd_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].full([2, 2, 2], 2.0).to_gpu()
        var out = a.product(IntArray(2))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].full([2, 2], 4.0))
        )


def test_prd_gpu_fwd_4d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].full([2, 2, 2, 2], 2.0).to_gpu()
        var out = a.product(IntArray(3))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](
                Tensor[dtype].full([2, 2, 2], 4.0)
            )
        )


def test_prd_gpu_fwd_large() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # All ones — product=1, exercises multi-block dispatch
        var a = Tensor[dtype].full([65536], 1.0).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-3](Tensor[dtype].scalar(1.0))
        )


def test_prd_gpu_fwd_dtype_float64() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-8](Tensor[dtype].scalar(24.0))
        )


def test_prd_gpu_fwd_negative_sign_tracking() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([-1.0, 2.0, 3.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].scalar(-6.0))
        )


# =============================================================================
# ── SECTION 6: GPU BACKWARD ──────────────────────────────────────────────────
# =============================================================================


def test_prd_gpu_bwd_all_positive_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.product(IntArray(0))
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-4](
                Tensor[dtype].d1([12.0, 8.0, 6.0])
            )
        )


def test_prd_gpu_bwd_single_zero() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 0.0, 4.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.product(IntArray(0))
        var loss = out.sum()
        loss.backward()
        # Only the zero element gets non-zero grad
        assert_true(
            a_cpu.grad().all_close[atol=1e-4](Tensor[dtype].d1([0.0, 8.0, 0.0]))
        )


def test_prd_gpu_bwd_two_zeros() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 0.0, 0.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.product(IntArray(0))
        var loss = out.sum()
        loss.backward()
        # All grads zero — two zeros in slice
        assert_true(
            a_cpu.grad().all_close[atol=1e-4](Tensor[dtype].d1([0.0, 0.0, 0.0]))
        )


def test_prd_gpu_bwd_negative_elements() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-2.0, 3.0, -4.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.product(IntArray(0))
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-4](
                Tensor[dtype].d1([-12.0, 8.0, -6.0])
            )
        )


def test_prd_gpu_bwd_2d_axis0() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        var a = a_cpu.to_gpu()
        var out = a.product(IntArray(0))
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-4](
                Tensor[dtype].d2([[3.0, 4.0], [1.0, 2.0]])
            )
        )


def test_prd_gpu_bwd_2d_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        var a = a_cpu.to_gpu()
        var out = a.product(IntArray(1))
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-4](
                Tensor[dtype].d2([[2.0, 1.0], [4.0, 3.0]])
            )
        )


def test_prd_gpu_bwd_keepdims() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        var a = a_cpu.to_gpu()
        var out = a.product(IntArray(1), keepdims=True)
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-4](
                Tensor[dtype].d2([[2.0, 1.0], [4.0, 3.0]])
            )
        )


def test_prd_gpu_bwd_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full([2, 2, 2], 2.0, requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.product(IntArray(2))
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-4](
                Tensor[dtype].full([2, 2, 2], 2.0)
            )
        )


def test_prd_gpu_bwd_grad_shape_preserved() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a = a_cpu.to_gpu()
        var out = a.product(IntArray(1))
        var loss = out.sum()
        loss.backward()
        assert_equal(a_cpu.grad().shape(), a_cpu.shape())


def test_prd_gpu_bwd_recompute_path() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.product[store_excl_product=False](IntArray(0))
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-4](
                Tensor[dtype].d1([12.0, 8.0, 6.0])
            )
        )


def test_prd_gpu_bwd_large() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # All ones — excl_product = 1 everywhere — exercises multi-block backward
        var a_cpu = Tensor[dtype].full([1024], 1.0, requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.product(IntArray(0))
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close[atol=1e-3](Tensor[dtype].ones(1024)))


# =============================================================================
# ── SECTION 7: GPU GRAD FLOW ─────────────────────────────────────────────────
# =============================================================================


def test_prd_gpu_grad_chain_product_mul() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var p = a.product(IntArray(0))
        var out = p * Tensor[dtype].scalar(2.0).to_gpu()
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-4](
                Tensor[dtype].d1([24.0, 16.0, 12.0])
            )
        )


def test_prd_gpu_grad_chain_sum_of_product() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        var a = a_cpu.to_gpu()
        var p = a.product(IntArray(1))
        var loss = p.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-4](
                Tensor[dtype].d2([[2.0, 1.0], [4.0, 3.0]])
            )
        )


def test_prd_gpu_grad_two_inputs() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var b_cpu = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var b = b_cpu.to_gpu()
        var c = a * b
        var out = c.product(IntArray(0))
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-4](Tensor[dtype].d1([24.0, 12.0]))
        )
        assert_true(
            b_cpu.grad().all_close[atol=1e-4](Tensor[dtype].d1([8.0, 6.0]))
        )


# =============================================================================
# ── SECTION 8: GPU NON-CONTIGUOUS ────────────────────────────────────────────
# =============================================================================


def test_prd_gpu_noncontig_transposed_fwd() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var t = a.transpose()
        var out = t.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].d1([2.0, 12.0]))
        )


def test_prd_gpu_noncontig_transposed_bwd() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        var a = a_cpu.to_gpu()
        var t = a.transpose()
        var out = t.product(IntArray(0))
        var loss = out.sum()
        loss.backward()
        assert_equal(a_cpu.grad().shape(), a_cpu.shape())


# =============================================================================
# ── SECTION 9: CPU / GPU PARITY ──────────────────────────────────────────────
# Each parity test uses SEPARATE leaf tensors for CPU and GPU backward
# passes to avoid retained grad accumulation.
# =============================================================================


def test_prd_parity_fwd_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var data = Tensor[dtype].d1([2.0, -3.0, 4.0])
        var cpu_out = data.product(IntArray(0))
        var gpu_out = data.to_gpu().product(IntArray(0)).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


def test_prd_parity_fwd_2d_axis0() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var data = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var cpu_out = data.product(IntArray(0))
        var gpu_out = data.to_gpu().product(IntArray(0)).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


def test_prd_parity_fwd_single_zero() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var data = Tensor[dtype].d1([2.0, 0.0, 4.0])
        var cpu_out = data.product(IntArray(0))
        var gpu_out = data.to_gpu().product(IntArray(0)).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


def test_prd_parity_fwd_two_zeros() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var data = Tensor[dtype].d1([2.0, 0.0, 0.0])
        var cpu_out = data.product(IntArray(0))
        var gpu_out = data.to_gpu().product(IntArray(0)).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


def test_prd_parity_bwd_all_positive() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # Separate leaf tensors — no retained grad cross-contamination
        var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var loss_cpu = a_cpu.product(IntArray(0)).sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad()

        var a_cpu2 = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var loss_gpu = a_cpu2.to_gpu().product(IntArray(0)).sum()
        loss_gpu.backward()
        var gpu_grad = a_cpu2.grad()

        assert_true(cpu_grad.all_close[atol=1e-4](gpu_grad))


def test_prd_parity_bwd_single_zero() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 0.0, 4.0], requires_grad=True)
        var loss_cpu = a_cpu.product(IntArray(0)).sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad()

        var a_cpu2 = Tensor[dtype].d1([2.0, 0.0, 4.0], requires_grad=True)
        var loss_gpu = a_cpu2.to_gpu().product(IntArray(0)).sum()
        loss_gpu.backward()
        var gpu_grad = a_cpu2.grad()

        assert_true(cpu_grad.all_close[atol=1e-4](gpu_grad))


def test_prd_parity_bwd_two_zeros() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 0.0, 0.0], requires_grad=True)
        var loss_cpu = a_cpu.product(IntArray(0)).sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad()

        var a_cpu2 = Tensor[dtype].d1([2.0, 0.0, 0.0], requires_grad=True)
        var loss_gpu = a_cpu2.to_gpu().product(IntArray(0)).sum()
        loss_gpu.backward()
        var gpu_grad = a_cpu2.grad()

        assert_true(cpu_grad.all_close[atol=1e-4](gpu_grad))


def test_prd_parity_bwd_negative_elements() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-2.0, 3.0, -4.0], requires_grad=True)
        var loss_cpu = a_cpu.product(IntArray(0)).sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad()

        var a_cpu2 = Tensor[dtype].d1([-2.0, 3.0, -4.0], requires_grad=True)
        var loss_gpu = a_cpu2.to_gpu().product(IntArray(0)).sum()
        loss_gpu.backward()
        var gpu_grad = a_cpu2.grad()

        assert_true(cpu_grad.all_close[atol=1e-4](gpu_grad))


def test_prd_parity_store_vs_recompute() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # store path
        var a1 = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var loss1 = a1.product(IntArray(1)).sum()
        loss1.backward()
        var grad_store = a1.grad()

        # recompute path — fresh tensor, no retained grad
        var a2 = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var loss2 = a2.product[store_excl_product=False](IntArray(1)).sum()
        loss2.backward()
        var grad_recompute = a2.grad()

        assert_true(grad_store.all_close[atol=1e-4](grad_recompute))


# =============================================================================
# ── MAIN ─────────────────────────────────────────────────────────────────────
# =============================================================================



# =============================================================================
# CPU FORWARD TESTS
# =============================================================================


# =============================================================================
# CPU BACKWARD TESTS
# =============================================================================


# =============================================================================
# CPU GRAD FLOW TESTS
# =============================================================================


# =============================================================================
# GPU TESTS
# =============================================================================


def test_recip_gpu_forward_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 4.0, 8.0])
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=False]()
        assert_true(r.shape() == Shape(4))
        assert_true(
            r.to_cpu().all_close[atol=1e-6](
                Tensor[dtype].d1([1.0, 0.5, 0.25, 0.125])
            )
        )


def test_recip_gpu_forward_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0], [4.0, 8.0]])
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=False]()
        assert_true(r.shape() == Shape(2, 2))
        assert_true(
            r.to_cpu().all_close[atol=1e-6](
                Tensor[dtype].d2([[1.0, 0.5], [0.25, 0.125]])
            )
        )


def test_recip_gpu_backward_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=True]()
        var loss = r.sum()
        loss.backward()
        assert_true(x.grad().shape() == Shape(3))
        assert_true(
            x.grad().all_close[atol=1e-6](
                Tensor[dtype].d1([-1.0, -0.25, -0.0625])
            )
        )


def test_recip_gpu_backward_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0], [4.0, 8.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=True]()
        var loss = r.sum()
        loss.backward()
        assert_true(x.grad().shape() == Shape(2, 2))
        assert_true(
            x.grad().all_close[atol=1e-6](
                Tensor[dtype].d2([[-1.0, -0.25], [-0.0625, -0.015625]])
            )
        )


def test_recip_gpu_backward_3d() raises:
    comptime if has_accelerator():
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
        assert_true(
            x.grad().all_close[atol=1e-6](
                Tensor[dtype].d3(
                    [
                        [[-1.0, -0.25], [-0.0625, -0.015625]],
                        [[-1.0, -0.25], [-0.0625, -0.015625]],
                    ]
                )
            )
        )


def test_recip_gpu_grad_flow_chain() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=True]()
        var scaled = r * Tensor[dtype].scalar(2.0).to_gpu()
        var loss = scaled.sum()
        loss.backward()
        assert_true(
            x.grad().all_close[atol=1e-6](
                Tensor[dtype].d1([-2.0, -0.5, -0.125])
            )
        )


def test_recip_gpu_grad_flow_double_reciprocal() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var r1 = x_gpu.reciprocal[track_grad=True]()
        var r2 = r1.reciprocal[track_grad=True]()
        var loss = r2.sum()
        loss.backward()
        assert_true(x.grad().all_close[atol=1e-5](Tensor.ones_like(x)))


def test_recip_gpu_vs_cpu_consistency() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 4.0], [0.5, 0.25, 8.0]], requires_grad=True
        )
        var x_gpu = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 4.0], [0.5, 0.25, 8.0]], requires_grad=True)
            .to_gpu()
        )
        var r_cpu = x_cpu.reciprocal[track_grad=True]()
        var r_gpu = x_gpu.reciprocal[track_grad=True]()
        assert_true(r_cpu.all_close[atol=1e-6](r_gpu.to_cpu()))
        var loss_cpu = r_cpu.sum()
        loss_cpu.backward()
        var loss_gpu = r_gpu.sum()
        loss_gpu.backward()
        assert_true(x_cpu.grad().all_close[atol=1e-6](x_gpu.grad().to_cpu()))


def test_recip_gpu_negative_values() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([-1.0, -2.0, -4.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var r = x_gpu.reciprocal[track_grad=True]()
        assert_true(
            r.to_cpu().all_close[atol=1e-6](
                Tensor[dtype].d1([-1.0, -0.5, -0.25])
            )
        )
        var loss = r.sum()
        loss.backward()
        assert_true(
            x.grad().all_close[atol=1e-6](
                Tensor[dtype].d1([-1.0, -0.25, -0.0625])
            )
        )


def test_recip_gpu_grad_accumulation() raises:
    comptime if has_accelerator():
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
        assert_true(
            x1.grad().all_close[atol=1e-6](Tensor[dtype].d1([-0.25, -0.0625]))
        )
        assert_true(
            x2.grad().all_close[atol=1e-6](Tensor[dtype].d1([-0.25, -0.0625]))
        )




# Old tests


# End of old tests

# ═════════════════════════════════════════════════════════════════════════════
# CPU Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# CPU Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GPU Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_perm_gpu_2d_identity() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.permute(IntArray(0, 1))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 3))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            )
        )


def test_perm_gpu_2d_transpose() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.permute(IntArray(1, 0))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(3, 2))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
            )
        )


def test_perm_gpu_3d_021() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var result = a.permute(IntArray(0, 2, 1))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 2, 2))
        var result_cpu = result.to_cpu()
        var a_cpu = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    assert_true(result_cpu[[i, j, k]] == a_cpu[[i, k, j]])


def test_perm_gpu_3d_120() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(24)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4))
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.permute(IntArray(1, 2, 0))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(3, 4, 2))
        var result_cpu = result.to_cpu()
        for i in range(3):
            for j in range(4):
                for k in range(2):
                    assert_true(result_cpu[[i, j, k]] == a_cpu[[k, i, j]])


def test_perm_gpu_3d_210() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(24)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4))
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.permute(IntArray(2, 1, 0))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(4, 3, 2))
        var result_cpu = result.to_cpu()
        for i in range(4):
            for j in range(3):
                for k in range(2):
                    assert_true(result_cpu[[i, j, k]] == a_cpu[[k, j, i]])


def test_perm_gpu_shape_preserved() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(60)
        var _tmp1 = _tmp0.reshape(Shape(3, 4, 5))
        var a = _tmp1.to_gpu()
        var result = a.permute(IntArray(2, 0, 1))
        assert_true(result.shape() == Shape(5, 3, 4))
        assert_true(result.numels() == 60)


def test_perm_gpu_no_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
            .to_gpu()
        )
        var result = a.permute(IntArray(1, 0))
        assert_true(not result.requires_grad)


def test_perm_gpu_requires_grad_propagates() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            .to_gpu()
        )
        var result = a.permute(IntArray(1, 0))
        assert_true(result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# GPU Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_perm_gpu_backward_2d_transpose() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.permute(IntArray(1, 0))
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3))))


def test_perm_gpu_backward_3d_021() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.permute(IntArray(0, 2, 1))
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


def test_perm_gpu_backward_3d_210() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(24)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4))
        a_cpu.requires_grad_(True)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.permute(IntArray(2, 1, 0))
        var loss = result.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].ones(Shape(2, 3, 4))))


def test_perm_gpu_backward_chain() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = (a_gpu.permute(IntArray(1, 0))) * 2.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 2.0)))


def test_perm_gpu_backward_double_permute() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var b = a_gpu.permute(IntArray(2, 0, 1))
        var c = b.permute(IntArray(1, 2, 0))
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


def test_perm_gpu_backward_gradient_values() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.permute(IntArray(1, 0))
        var weights = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var loss = (result * weights).sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]]))
        )


# ═════════════════════════════════════════════════════════════════════════════
# CPU/GPU Parity Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_perm_parity_2d_transpose() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.permute(IntArray(1, 0)).all_close(
                a_gpu.permute(IntArray(1, 0)).to_cpu()
            )
        )


def test_perm_parity_3d_120() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(24)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4))
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.permute(IntArray(1, 2, 0)).all_close(
                a_gpu.permute(IntArray(1, 2, 0)).to_cpu()
            )
        )


def test_perm_parity_3d_210() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(24)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4))
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.permute(IntArray(2, 1, 0)).all_close(
                a_gpu.permute(IntArray(2, 1, 0)).to_cpu()
            )
        )


def test_perm_parity_backward_transpose() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            .to_gpu()
        )

        var loss_cpu = a_cpu.permute(IntArray(1, 0)).sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.permute(IntArray(1, 0)).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


def test_perm_parity_backward_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(24)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4))
        a_cpu.requires_grad_(True)
        var _tmp1 = Tensor[dtype].arange(24)
        _tmp1 = _tmp1.reshape(Shape(2, 3, 4))
        var a_gpu = _tmp1.to_gpu()
        a_gpu.requires_grad_(True)

        var loss_cpu = a_cpu.permute(IntArray(2, 0, 1)).sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.permute(IntArray(2, 0, 1)).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


def test_perm_parity_backward_chain() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        var a_gpu = (
            Tensor[dtype]
            .d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            .to_gpu()
        )

        var loss_cpu = (a_cpu.permute(IntArray(1, 0)) * 3.0).sum()
        loss_cpu.backward()

        var loss_gpu = (a_gpu.permute(IntArray(1, 0)) * 3.0).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


def test_perm_parity_using_zero_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.permute(IntArray(1, 0)).sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad().copy()

        a_cpu.zero_grad()

        var loss_gpu = a_gpu.permute(IntArray(1, 0)).sum()
        loss_gpu.backward()

        assert_true(cpu_grad.all_close(a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close(a_cpu.grad()))


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════




# =============================================================================
# Exhaustive tests for Tensor.outer()
# Prefix: test_outer_ on all test names to avoid collision with existing tests
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CPU · Forward correctness · 1-D × 1-D
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CPU · Forward · 2-D and higher inputs (auto-flatten)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CPU · Backward · grad flows through both inputs
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CPU · track_grad=False explicit
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GPU · Forward correctness
# ─────────────────────────────────────────────────────────────────────────────


def test_outer_gpu_fwd_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var b = Tensor[dtype].d1([4.0, 5.0]).to_gpu()
        var result = a.outer(b)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [4.0, 5.0],
                        [8.0, 10.0],
                        [12.0, 15.0],
                    ]
                )
            )
        )


def test_outer_gpu_fwd_vector_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        var b = Tensor[dtype].scalar(2.0).to_gpu()
        var result = a.outer(b)
        assert_true(result.shape() == Shape(4, 1))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[2.0], [4.0], [6.0], [8.0]])
            )
        )


def test_outer_gpu_fwd_negative_values() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, -2.0]).to_gpu()
        var b = Tensor[dtype].d1([-3.0, 4.0]).to_gpu()
        var result = a.outer(b)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [-3.0, 4.0],
                        [6.0, -8.0],
                    ]
                )
            )
        )


def test_outer_gpu_fwd_result_shape() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0]).to_gpu()
        var b = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.outer(b)
        assert_true(result.shape() == Shape(5, 3))


def test_outer_gpu_fwd_2d_flattened() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var b = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.outer(b)
        assert_true(result.shape() == Shape(4, 3))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [1.0, 2.0, 3.0],
                        [2.0, 4.0, 6.0],
                        [3.0, 6.0, 9.0],
                        [4.0, 8.0, 12.0],
                    ]
                )
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GPU · Backward
# ─────────────────────────────────────────────────────────────────────────────


def test_outer_gpu_bwd_both_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].d1([4.0, 5.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var result = a_gpu.outer(b_gpu)
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([9.0, 9.0, 9.0])))
        assert_true(b.grad().all_close(Tensor[dtype].d1([6.0, 6.0])))


def test_outer_gpu_bwd_vector_scalar() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].scalar(2.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var result = a_gpu.outer(b_gpu)
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))
        assert_true(b.grad().all_close(Tensor[dtype].scalar(6.0)))


def test_outer_gpu_bwd_lhs_only() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].d1([4.0, 5.0])
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var result = a_gpu.outer(b_gpu)
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([9.0, 9.0, 9.0])))


def test_outer_gpu_bwd_chained_multiply() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var b = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var outer = a_gpu.outer(b_gpu)
        var scale = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]]).to_gpu()
        var c = outer * scale
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([14.0, 14.0])))
        assert_true(b.grad().all_close(Tensor[dtype].d1([6.0, 6.0])))


def test_outer_gpu_no_track_grad_explicit() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].d1([4.0, 5.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var result = a_gpu.outer[track_grad=False](b_gpu)
        assert_false(result.requires_grad)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [4.0, 5.0],
                        [8.0, 10.0],
                        [12.0, 15.0],
                    ]
                )
            )
        )


# =============================================================================
# Main
# =============================================================================



# ── CPU Tests ─────────────────────────────────────────────────────────────────


# ── GPU Tests ─────────────────────────────────────────────────────────────────


def test_onehot_gpu_1d_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d1([0.0, 1.0, 2.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 3)
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                )
            )
        )


def test_onehot_gpu_1d_mixed() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d1([2.0, 0.0, 1.0, 3.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 4)
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            )
        )


def test_onehot_gpu_2d_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d2([[0.0, 2.0], [1.0, 3.0]]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 4)
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d3(
                    [
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                        ],
                        [
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    ]
                )
            )
        )


def test_onehot_gpu_first_class() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d1([0.0, 0.0, 0.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 4)
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                )
            )
        )


def test_onehot_gpu_last_class() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d1([3.0, 3.0, 3.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 4)
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            )
        )


def test_onehot_gpu_shape() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices = (
            Tensor[dtype].d2([[0.0, 1.0], [2.0, 0.0], [1.0, 2.0]]).to_gpu()
        )
        var result = Tensor[dtype].onehot(indices, 5)
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(3, 2, 5))


def test_onehot_gpu_all_zeros_except_one() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 4)
        var result_cpu = result.to_cpu()
        # Each row must sum to exactly 1.0
        for i in range(4):
            var row_sum = Scalar[dtype](0)
            for j in range(4):
                row_sum += result_cpu[[i, j]]
            assert_true(row_sum == Scalar[dtype](1.0))


def test_onehot_gpu_large_num_classes() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices = Tensor[dtype].d1([0.0, 4.0, 9.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 10)
        var result_cpu = result.to_cpu()
        assert_true(result_cpu[[0, 0]] == Scalar[dtype](1.0))
        assert_true(result_cpu[[0, 1]] == Scalar[dtype](0.0))
        assert_true(result_cpu[[1, 4]] == Scalar[dtype](1.0))
        assert_true(result_cpu[[1, 3]] == Scalar[dtype](0.0))
        assert_true(result_cpu[[2, 9]] == Scalar[dtype](1.0))
        assert_true(result_cpu[[2, 8]] == Scalar[dtype](0.0))


def test_onehot_gpu_explicit_device() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # CPU indices but explicit GPU device
        var indices = Tensor[dtype].d1([0.0, 1.0, 2.0])
        var result = Tensor[dtype].onehot(indices, 3, GPU().into())
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                )
            )
        )


def test_onehot_gpu_override_to_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # GPU indices but force CPU result
        var indices = Tensor[dtype].d1([0.0, 1.0, 2.0]).to_gpu()
        var result = Tensor[dtype].onehot(indices, 3, CPU().into())
        assert_true(result.is_on_cpu())
        assert_true(
            result.all_close(
                Tensor[dtype].d2(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                )
            )
        )


# ── CPU/GPU Parity ────────────────────────────────────────────────────────────


def test_onehot_gpu_parity_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices_cpu = Tensor[dtype].d1([0.0, 2.0, 1.0, 3.0])
        var indices_gpu = indices_cpu.to_gpu()
        var result_cpu = Tensor[dtype].onehot(indices_cpu, 4)
        var result_gpu = Tensor[dtype].onehot(indices_gpu, 4)
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


def test_onehot_gpu_parity_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var indices_cpu = Tensor[dtype].d2([[0.0, 2.0], [1.0, 3.0]])
        var indices_gpu = indices_cpu.to_gpu()
        var result_cpu = Tensor[dtype].onehot(indices_cpu, 4)
        var result_gpu = Tensor[dtype].onehot(indices_gpu, 4)
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


# ── Main ──────────────────────────────────────────────────────────────────────







def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_gpu_tensor_to_numpy_contiguous() raises:
    """GPU contiguous tensor → to_ndarray must not read empty CPU buffer."""
    print("test_gpu_tensor_to_numpy_contiguous")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var cpu_t = Tensor[dtype].arange(120)
        var gpu_t = cpu_t.to_gpu()
        var nd = to_ndarray(gpu_t)
        var back = from_ndarray[dtype](nd)
        assert_true(back.all_close(cpu_t))


def test_gpu_tensor_to_numpy_view() raises:
    """GPU non-contiguous view (offset/strides) → to_ndarray."""
    print("test_gpu_tensor_to_numpy_view")
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var cpu_t = Tensor[dtype].arange(32)
        var cpu_t_2d = cpu_t.reshape(4, 8)
        var cpu_v = cpu_t_2d.view([2, 3], offset=5)
        var gpu_v = cpu_v.to_gpu()
        var nd = to_ndarray(gpu_v)
        var back = from_ndarray[dtype](nd)
        assert_true(back.all_close(cpu_v))


def test_gpu_bool_tensor_to_numpy() raises:
    """GPU bool tensor (uint8 internal) → to_ndarray."""
    print("test_gpu_bool_tensor_to_numpy")
    comptime if has_accelerator():
        var cpu_t = Tensor[DType.bool].full([3, 4], True)
        var gpu_t = cpu_t.to_gpu()
        var nd = to_ndarray(gpu_t)
        var back = from_ndarray[DType.bool](nd)
        assert_true(back.all_close(cpu_t))


def test_gpu_tensor_print() raises:
    """GPU tensor .print() must not crash."""
    print("test_gpu_tensor_print")
    comptime dtype = DType.float32
    var cpu_t = Tensor[dtype].arange(16)
    var cpu_t_2d = cpu_t.reshape(4, 4)
    cpu_t_2d.print()
    comptime if has_accelerator():
        var gpu_t = cpu_t.to_gpu()
        gpu_t.print()
