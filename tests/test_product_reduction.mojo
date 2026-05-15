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
from std.testing import assert_true, TestSuite, assert_equal
from tenmo.tensor import Tensor
from std.math import abs
from tenmo.intarray import IntArray


# =============================================================================
# ── SECTION 1: CPU FORWARD ───────────────────────────────────────────────────
# =============================================================================


fn test_prd_cpu_fwd_all_positive_1d() raises:
    print("test_prd_cpu_fwd_all_positive_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=False)
    var out = a.product(IntArray(0))
    # 2 * 3 * 4 = 24
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].scalar(24.0)))


fn test_prd_cpu_fwd_all_negative_1d() raises:
    print("test_prd_cpu_fwd_all_negative_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-2.0, -3.0, -4.0], requires_grad=False)
    var out = a.product(IntArray(0))
    # (-2)*(-3)*(-4) = -24
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].scalar(-24.0)))


fn test_prd_cpu_fwd_mixed_signs_1d() raises:
    print("test_prd_cpu_fwd_mixed_signs_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-2.0, 3.0, -4.0], requires_grad=False)
    var out = a.product(IntArray(0))
    # (-2)*3*(-4) = 24
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].scalar(24.0)))


fn test_prd_cpu_fwd_single_zero_1d() raises:
    print("test_prd_cpu_fwd_single_zero_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 0.0, 4.0], requires_grad=False)
    var out = a.product(IntArray(0))
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].scalar(0.0)))


fn test_prd_cpu_fwd_two_zeros_1d() raises:
    print("test_prd_cpu_fwd_two_zeros_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 0.0, 0.0], requires_grad=False)
    var out = a.product(IntArray(0))
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].scalar(0.0)))


fn test_prd_cpu_fwd_single_element() raises:
    print("test_prd_cpu_fwd_single_element")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([5.0], requires_grad=False)
    var out = a.product(IntArray(0))
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].scalar(5.0)))


fn test_prd_cpu_fwd_2d_axis0() raises:
    print("test_prd_cpu_fwd_2d_axis0")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var out = a.product(IntArray(0))
    # col0: 1*3=3, col1: 2*4=8
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].d1([3.0, 8.0])))


fn test_prd_cpu_fwd_2d_axis1() raises:
    print("test_prd_cpu_fwd_2d_axis1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var out = a.product(IntArray(1))
    # row0: 1*2=2, row1: 3*4=12
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].d1([2.0, 12.0])))


fn test_prd_cpu_fwd_2d_keepdims() raises:
    print("test_prd_cpu_fwd_2d_keepdims")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var out = a.product(IntArray(1), keepdims=True)
    # shape [2,1]: [[2],[12]]
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].d2([[2.0], [12.0]])))


fn test_prd_cpu_fwd_2d_all_axes() raises:
    print("test_prd_cpu_fwd_2d_all_axes")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var axes = IntArray(0, 1)
    var out = a.product(axes)
    # 1*2*3*4 = 24
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].scalar(24.0)))


fn test_prd_cpu_fwd_3d_axis1() raises:
    print("test_prd_cpu_fwd_3d_axis1")
    comptime dtype = DType.float32
    # shape [2,2,2]
    var a = Tensor[dtype].full([2, 2, 2], 2.0, requires_grad=False)
    var out = a.product(IntArray(1))
    # each slice along axis1: 2*2 = 4
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].full([2, 2], 4.0)))


fn test_prd_cpu_fwd_4d() raises:
    print("test_prd_cpu_fwd_4d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full([2, 2, 2, 2], 2.0, requires_grad=False)
    var out = a.product(IntArray(3))
    # each slice along last axis: 2*2 = 4
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].full([2, 2, 2], 4.0)))


fn test_prd_cpu_fwd_scalar_input() raises:
    print("test_prd_cpu_fwd_scalar_input")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(7.0, requires_grad=False)
    var out = a.product(IntArray())
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].scalar(7.0)))


fn test_prd_cpu_fwd_dtype_float64() raises:
    print("test_prd_cpu_fwd_dtype_float64")
    comptime dtype = DType.float64
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=False)
    var out = a.product(IntArray(0))
    assert_true(out.all_close[atol=1e-8](Tensor[dtype].scalar(24.0)))


fn test_prd_cpu_fwd_dtype_int32() raises:
    print("test_prd_cpu_fwd_dtype_int32")
    comptime dtype = DType.int32
    var a = Tensor[dtype].d1([2, 3, 4], requires_grad=False)
    var out = a.product(IntArray(0))
    assert_equal(out.item(), Scalar[dtype](24))


fn test_prd_cpu_fwd_negative_sign_tracking() raises:
    print("test_prd_cpu_fwd_negative_sign_tracking")
    comptime dtype = DType.float32
    # odd negatives → negative result
    var a = Tensor[dtype].d1([-1.0, 2.0, 3.0], requires_grad=False)
    var out = a.product(IntArray(0))
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].scalar(-6.0)))
    # even negatives → positive result
    var b = Tensor[dtype].d1([-1.0, -2.0, 3.0], requires_grad=False)
    var out2 = b.product(IntArray(0))
    assert_true(out2.all_close[atol=1e-4](Tensor[dtype].scalar(6.0)))


# =============================================================================
# ── SECTION 2: CPU BACKWARD ──────────────────────────────────────────────────
# =============================================================================


fn test_prd_cpu_bwd_all_positive_1d() raises:
    print("test_prd_cpu_bwd_all_positive_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var out = a.product(IntArray(0))
    var loss = out.sum()
    loss.backward()
    # grad_x[i] = product / x[i] = 24 / x[i]
    assert_true(
        a.grad().all_close[atol=1e-4](Tensor[dtype].d1([12.0, 8.0, 6.0]))
    )


fn test_prd_cpu_bwd_single_zero() raises:
    print("test_prd_cpu_bwd_single_zero")
    comptime dtype = DType.float32
    # x = [2, 0, 4] — one zero
    # grad_x[1] = 2*4 = 8 (the zero element gets the non-zero grad)
    # grad_x[0] = 0*4 = 0
    # grad_x[2] = 2*0 = 0
    var a = Tensor[dtype].d1([2.0, 0.0, 4.0], requires_grad=True)
    var out = a.product(IntArray(0))
    var loss = out.sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-4](Tensor[dtype].d1([0.0, 8.0, 0.0]))
    )


fn test_prd_cpu_bwd_two_zeros() raises:
    print("test_prd_cpu_bwd_two_zeros")
    comptime dtype = DType.float32
    # x = [2, 0, 0] — two zeros → all grads zero
    var a = Tensor[dtype].d1([2.0, 0.0, 0.0], requires_grad=True)
    var out = a.product(IntArray(0))
    var loss = out.sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-4](Tensor[dtype].d1([0.0, 0.0, 0.0]))
    )


fn test_prd_cpu_bwd_negative_elements() raises:
    print("test_prd_cpu_bwd_negative_elements")
    comptime dtype = DType.float32
    # x = [-2, 3, -4] → product = 24
    # grad_x[0] = 3*(-4) = -12
    # grad_x[1] = (-2)*(-4) = 8
    # grad_x[2] = (-2)*3 = -6
    var a = Tensor[dtype].d1([-2.0, 3.0, -4.0], requires_grad=True)
    var out = a.product(IntArray(0))
    var loss = out.sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-4](Tensor[dtype].d1([-12.0, 8.0, -6.0]))
    )


fn test_prd_cpu_bwd_2d_axis0() raises:
    print("test_prd_cpu_bwd_2d_axis0")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var out = a.product(IntArray(0))
    var loss = out.sum()
    loss.backward()
    # grad_a[0,0]=3, grad_a[1,0]=1, grad_a[0,1]=4, grad_a[1,1]=2
    assert_true(
        a.grad().all_close[atol=1e-4](
            Tensor[dtype].d2([[3.0, 4.0], [1.0, 2.0]])
        )
    )


fn test_prd_cpu_bwd_2d_axis1() raises:
    print("test_prd_cpu_bwd_2d_axis1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var out = a.product(IntArray(1))
    var loss = out.sum()
    loss.backward()
    # grad_a[0,0]=2, grad_a[0,1]=1, grad_a[1,0]=4, grad_a[1,1]=3
    assert_true(
        a.grad().all_close[atol=1e-4](
            Tensor[dtype].d2([[2.0, 1.0], [4.0, 3.0]])
        )
    )


fn test_prd_cpu_bwd_keepdims() raises:
    print("test_prd_cpu_bwd_keepdims")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var out = a.product(IntArray(1), keepdims=True)
    var loss = out.sum()
    loss.backward()
    # Same grad as axis1 without keepdims — keepdims only affects shape
    assert_true(
        a.grad().all_close[atol=1e-4](
            Tensor[dtype].d2([[2.0, 1.0], [4.0, 3.0]])
        )
    )


fn test_prd_cpu_bwd_3d() raises:
    print("test_prd_cpu_bwd_3d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full([2, 2, 2], 2.0, requires_grad=True)
    var out = a.product(IntArray(2))
    var loss = out.sum()
    loss.backward()
    # Each slice: [2,2] → product=4, grad each = 2
    assert_true(
        a.grad().all_close[atol=1e-4](Tensor[dtype].full([2, 2, 2], 2.0))
    )


fn test_prd_cpu_bwd_grad_shape_preserved() raises:
    print("test_prd_cpu_bwd_grad_shape_preserved")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var out = a.product(IntArray(1))
    var loss = out.sum()
    loss.backward()
    assert_equal(a.grad().shape(), a.shape())


fn test_prd_cpu_bwd_recompute_path() raises:
    print("test_prd_cpu_bwd_recompute_path")
    comptime dtype = DType.float32
    # store_excl_product=False → recompute path in backward
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var out = a.product[store_excl_product=False](IntArray(0))
    var loss = out.sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-4](Tensor[dtype].d1([12.0, 8.0, 6.0]))
    )


# =============================================================================
# ── SECTION 3: CPU GRAD FLOW ─────────────────────────────────────────────────
# =============================================================================


fn test_prd_cpu_grad_chain_product_mul() raises:
    print("test_prd_cpu_grad_chain_product_mul")
    comptime dtype = DType.float32
    # y = product(a) * 2  →  grad_a[i] = 2 * excl_product[i]
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var p = a.product(IntArray(0))
    var out = p * Tensor[dtype].scalar(2.0)
    var loss = out.sum()
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-4](Tensor[dtype].d1([24.0, 16.0, 12.0]))
    )


fn test_prd_cpu_grad_chain_sum_of_product() raises:
    print("test_prd_cpu_grad_chain_sum_of_product")
    comptime dtype = DType.float32
    # Reduce along axis1, then sum
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var p = a.product(IntArray(1))  # [2, 12]
    var loss = p.sum()  # 14
    loss.backward()
    assert_true(
        a.grad().all_close[atol=1e-4](
            Tensor[dtype].d2([[2.0, 1.0], [4.0, 3.0]])
        )
    )


fn test_prd_cpu_grad_no_grad_tensor() raises:
    print("test_prd_cpu_grad_no_grad_tensor")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=False)
    var out = a.product(IntArray(0))
    var loss = out.sum()
    loss.backward()
    assert_true(not a.requires_grad)


fn test_prd_cpu_grad_two_inputs() raises:
    print("test_prd_cpu_grad_two_inputs")
    comptime dtype = DType.float32
    # z = product(a * b, axis=0)
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var b = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
    var c = a * b  # [3, 8]
    var out = c.product(IntArray(0))  # 24
    var loss = out.sum()
    loss.backward()
    # grad_c = [8, 3], grad_a = grad_c * b = [24, 12], grad_b = grad_c * a = [8, 6]
    assert_true(a.grad().all_close[atol=1e-4](Tensor[dtype].d1([24.0, 12.0])))
    assert_true(b.grad().all_close[atol=1e-4](Tensor[dtype].d1([8.0, 6.0])))


fn test_prd_cpu_grad_large_tensor() raises:
    print("test_prd_cpu_grad_large_tensor")
    comptime dtype = DType.float32
    # All ones — product=1, excl_product=1 everywhere
    var a = Tensor[dtype].full([1024], 1.0, requires_grad=True)
    var out = a.product(IntArray(0))
    var loss = out.sum()
    loss.backward()
    assert_true(a.grad().all_close[atol=1e-3](Tensor[dtype].ones(1024)))


# =============================================================================
# ── SECTION 4: CPU NON-CONTIGUOUS ────────────────────────────────────────────
# =============================================================================


fn test_prd_cpu_noncontig_transposed_fwd() raises:
    print("test_prd_cpu_noncontig_transposed_fwd")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
    var t = a.transpose()
    var out = t.product(IntArray(0))
    # transposed: [[1,3],[2,4]], axis0 product: [1*2=2, 3*4=12]
    assert_true(out.all_close[atol=1e-4](Tensor[dtype].d1([2.0, 12.0])))


fn test_prd_cpu_noncontig_transposed_bwd() raises:
    print("test_prd_cpu_noncontig_transposed_bwd")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var t = a.transpose()
    var out = t.product(IntArray(0))
    var loss = out.sum()
    loss.backward()
    assert_equal(a.grad().shape(), a.shape())


# =============================================================================
# ── SECTION 5: GPU FORWARD ───────────────────────────────────────────────────
# =============================================================================


fn test_prd_gpu_fwd_all_positive_1d() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_all_positive_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].scalar(24.0))
        )


fn test_prd_gpu_fwd_all_negative_1d() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_all_negative_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([-2.0, -3.0, -4.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].scalar(-24.0))
        )


fn test_prd_gpu_fwd_mixed_signs_1d() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_mixed_signs_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([-2.0, 3.0, -4.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].scalar(24.0))
        )


fn test_prd_gpu_fwd_single_zero() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_single_zero")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 0.0, 4.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].scalar(0.0))
        )


fn test_prd_gpu_fwd_two_zeros() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_two_zeros")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 0.0, 0.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].scalar(0.0))
        )


fn test_prd_gpu_fwd_2d_axis0() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_2d_axis0")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].d1([3.0, 8.0]))
        )


fn test_prd_gpu_fwd_2d_axis1() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_2d_axis1")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var out = a.product(IntArray(1))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].d1([2.0, 12.0]))
        )


fn test_prd_gpu_fwd_2d_keepdims() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_2d_keepdims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var out = a.product(IntArray(1), keepdims=True)
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].d2([[2.0], [12.0]]))
        )


fn test_prd_gpu_fwd_3d() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_3d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].full([2, 2, 2], 2.0).to_gpu()
        var out = a.product(IntArray(2))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].full([2, 2], 4.0))
        )


fn test_prd_gpu_fwd_4d() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_4d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].full([2, 2, 2, 2], 2.0).to_gpu()
        var out = a.product(IntArray(3))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](
                Tensor[dtype].full([2, 2, 2], 4.0)
            )
        )


fn test_prd_gpu_fwd_large() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_large")
        comptime dtype = DType.float32
        # All ones — product=1, exercises multi-block dispatch
        var a = Tensor[dtype].full([65536], 1.0).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-3](Tensor[dtype].scalar(1.0))
        )


fn test_prd_gpu_fwd_dtype_float64() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_dtype_float64")
        comptime dtype = DType.float64
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-8](Tensor[dtype].scalar(24.0))
        )


fn test_prd_gpu_fwd_negative_sign_tracking() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_fwd_negative_sign_tracking")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([-1.0, 2.0, 3.0]).to_gpu()
        var out = a.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].scalar(-6.0))
        )


# =============================================================================
# ── SECTION 6: GPU BACKWARD ──────────────────────────────────────────────────
# =============================================================================


fn test_prd_gpu_bwd_all_positive_1d() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_bwd_all_positive_1d")
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


fn test_prd_gpu_bwd_single_zero() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_bwd_single_zero")
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


fn test_prd_gpu_bwd_two_zeros() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_bwd_two_zeros")
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


fn test_prd_gpu_bwd_negative_elements() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_bwd_negative_elements")
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


fn test_prd_gpu_bwd_2d_axis0() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_bwd_2d_axis0")
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


fn test_prd_gpu_bwd_2d_axis1() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_bwd_2d_axis1")
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


fn test_prd_gpu_bwd_keepdims() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_bwd_keepdims")
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


fn test_prd_gpu_bwd_3d() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_bwd_3d")
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


fn test_prd_gpu_bwd_grad_shape_preserved() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_bwd_grad_shape_preserved")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a = a_cpu.to_gpu()
        var out = a.product(IntArray(1))
        var loss = out.sum()
        loss.backward()
        assert_equal(a_cpu.grad().shape(), a_cpu.shape())


fn test_prd_gpu_bwd_recompute_path() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_bwd_recompute_path")
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


fn test_prd_gpu_bwd_large() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_bwd_large")
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


fn test_prd_gpu_grad_chain_product_mul() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_grad_chain_product_mul")
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


fn test_prd_gpu_grad_chain_sum_of_product() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_grad_chain_sum_of_product")
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


fn test_prd_gpu_grad_two_inputs() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_grad_two_inputs")
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


fn test_prd_gpu_noncontig_transposed_fwd() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_noncontig_transposed_fwd")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var t = a.transpose()
        var out = t.product(IntArray(0))
        assert_true(
            out.to_cpu().all_close[atol=1e-4](Tensor[dtype].d1([2.0, 12.0]))
        )


fn test_prd_gpu_noncontig_transposed_bwd() raises:
    comptime if has_accelerator():
        print("test_prd_gpu_noncontig_transposed_bwd")
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


fn test_prd_parity_fwd_1d() raises:
    comptime if has_accelerator():
        print("test_prd_parity_fwd_1d")
        comptime dtype = DType.float32
        var data = Tensor[dtype].d1([2.0, -3.0, 4.0])
        var cpu_out = data.product(IntArray(0))
        var gpu_out = data.to_gpu().product(IntArray(0)).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


fn test_prd_parity_fwd_2d_axis0() raises:
    comptime if has_accelerator():
        print("test_prd_parity_fwd_2d_axis0")
        comptime dtype = DType.float32
        var data = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var cpu_out = data.product(IntArray(0))
        var gpu_out = data.to_gpu().product(IntArray(0)).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


fn test_prd_parity_fwd_single_zero() raises:
    comptime if has_accelerator():
        print("test_prd_parity_fwd_single_zero")
        comptime dtype = DType.float32
        var data = Tensor[dtype].d1([2.0, 0.0, 4.0])
        var cpu_out = data.product(IntArray(0))
        var gpu_out = data.to_gpu().product(IntArray(0)).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


fn test_prd_parity_fwd_two_zeros() raises:
    comptime if has_accelerator():
        print("test_prd_parity_fwd_two_zeros")
        comptime dtype = DType.float32
        var data = Tensor[dtype].d1([2.0, 0.0, 0.0])
        var cpu_out = data.product(IntArray(0))
        var gpu_out = data.to_gpu().product(IntArray(0)).to_cpu()
        assert_true(cpu_out.all_close[atol=1e-4](gpu_out))


fn test_prd_parity_bwd_all_positive() raises:
    comptime if has_accelerator():
        print("test_prd_parity_bwd_all_positive")
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


fn test_prd_parity_bwd_single_zero() raises:
    comptime if has_accelerator():
        print("test_prd_parity_bwd_single_zero")
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


fn test_prd_parity_bwd_two_zeros() raises:
    comptime if has_accelerator():
        print("test_prd_parity_bwd_two_zeros")
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


fn test_prd_parity_bwd_negative_elements() raises:
    comptime if has_accelerator():
        print("test_prd_parity_bwd_negative_elements")
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


fn test_prd_parity_store_vs_recompute() raises:
    comptime if has_accelerator():
        print("test_prd_parity_store_vs_recompute")
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


fn main() raises:
    _="""# ── CPU Forward ──────────────────────────────────────────────────────────
    test_prd_cpu_fwd_all_positive_1d()
    test_prd_cpu_fwd_all_negative_1d()
    test_prd_cpu_fwd_mixed_signs_1d()
    test_prd_cpu_fwd_single_zero_1d()
    test_prd_cpu_fwd_two_zeros_1d()
    test_prd_cpu_fwd_single_element()
    test_prd_cpu_fwd_2d_axis0()
    test_prd_cpu_fwd_2d_axis1()
    test_prd_cpu_fwd_2d_keepdims()
    test_prd_cpu_fwd_2d_all_axes()
    test_prd_cpu_fwd_3d_axis1()
    test_prd_cpu_fwd_4d()
    test_prd_cpu_fwd_scalar_input()
    test_prd_cpu_fwd_dtype_float64()
    test_prd_cpu_fwd_dtype_int32()
    test_prd_cpu_fwd_negative_sign_tracking()

    # ── CPU Backward ─────────────────────────────────────────────────────────
    test_prd_cpu_bwd_all_positive_1d()
    test_prd_cpu_bwd_single_zero()
    test_prd_cpu_bwd_two_zeros()
    test_prd_cpu_bwd_negative_elements()
    test_prd_cpu_bwd_2d_axis0()
    test_prd_cpu_bwd_2d_axis1()
    test_prd_cpu_bwd_keepdims()
    test_prd_cpu_bwd_3d()
    test_prd_cpu_bwd_grad_shape_preserved()
    test_prd_cpu_bwd_recompute_path()

    # ── CPU Grad Flow ─────────────────────────────────────────────────────────
    test_prd_cpu_grad_chain_product_mul()
    test_prd_cpu_grad_chain_sum_of_product()
    test_prd_cpu_grad_no_grad_tensor()
    test_prd_cpu_grad_two_inputs()
    test_prd_cpu_grad_large_tensor()

    # ── CPU Non-contiguous ───────────────────────────────────────────────────
    test_prd_cpu_noncontig_transposed_fwd()
    test_prd_cpu_noncontig_transposed_bwd()

    # ── GPU Forward ──────────────────────────────────────────────────────────
    test_prd_gpu_fwd_all_positive_1d()
    test_prd_gpu_fwd_all_negative_1d()
    test_prd_gpu_fwd_mixed_signs_1d()
    test_prd_gpu_fwd_single_zero()
    test_prd_gpu_fwd_two_zeros()
    test_prd_gpu_fwd_2d_axis0()
    test_prd_gpu_fwd_2d_axis1()
    test_prd_gpu_fwd_2d_keepdims()
    test_prd_gpu_fwd_3d()
    test_prd_gpu_fwd_4d()
    test_prd_gpu_fwd_large()
    test_prd_gpu_fwd_dtype_float64()
    test_prd_gpu_fwd_negative_sign_tracking()

    # ── GPU Backward ─────────────────────────────────────────────────────────
    test_prd_gpu_bwd_all_positive_1d()
    test_prd_gpu_bwd_single_zero()
    test_prd_gpu_bwd_two_zeros()
    test_prd_gpu_bwd_negative_elements()
    test_prd_gpu_bwd_2d_axis0()
    test_prd_gpu_bwd_2d_axis1()
    test_prd_gpu_bwd_keepdims()
    test_prd_gpu_bwd_3d()
    test_prd_gpu_bwd_grad_shape_preserved()
    test_prd_gpu_bwd_recompute_path()
    test_prd_gpu_bwd_large()

    # ── GPU Grad Flow ─────────────────────────────────────────────────────────
    test_prd_gpu_grad_chain_product_mul()
    test_prd_gpu_grad_chain_sum_of_product()
    test_prd_gpu_grad_two_inputs()

    # ── GPU Non-contiguous ───────────────────────────────────────────────────
    test_prd_gpu_noncontig_transposed_fwd()
    test_prd_gpu_noncontig_transposed_bwd()

    # ── CPU / GPU Parity ─────────────────────────────────────────────────────
    test_prd_parity_fwd_1d()
    test_prd_parity_fwd_2d_axis0()
    test_prd_parity_fwd_single_zero()
    test_prd_parity_fwd_two_zeros()
    test_prd_parity_bwd_all_positive()
    test_prd_parity_bwd_single_zero()
    test_prd_parity_bwd_two_zeros()
    test_prd_parity_bwd_negative_elements()
    test_prd_parity_store_vs_recompute()

    print("All prd_ tests passed.")"""
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll product reduction tests passed!")
