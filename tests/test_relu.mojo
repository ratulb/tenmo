from tenmo import Tensor
from relu import ReLU
from std.sys import has_accelerator
from std.testing import assert_true, assert_equal
from shapes import Shape
from std.math import abs


fn test_relu_basic() raises:
    print("test_relu_basic")
    comptime dtype = DType.float32

    var t = Tensor[dtype].d1([-1.0, 0.0, 1.0, 2.0])
    t.requires_grad_(True)
    var out = ReLU[dtype].forward[True](t)
    s = out.sum()
    s.backward()

    assert_true(out == Tensor[dtype].d1([0.0, 0.0, 1.0, 2.0]))
    assert_true(t.grad() == Tensor[dtype].d1([0.0, 0.0, 1.0, 1.0]))
    print("✓ Passed ReLU forward and backward")

fn test_relu_multidim() raises:
    print("test_relu_multidim")
    comptime dtype = DType.float32

    # 2×3 input tensor
    var t = Tensor[dtype].d2([
        [-1.0, 2.0, 0.0],
        [3.0, -4.0, 5.0],
    ])
    t.requires_grad_(True)

    # Apply ReLU
    var out = t.relu()
    assert_true(out == Tensor[dtype].d2([
        [0.0, 2.0, 0.0],
        [3.0, 0.0, 5.0],
    ]))

    # Backward on sum of outputs
    s = out.sum()
    s.backward()

    # Gradient should be 1 where input > 0, else 0
    var expected_grad = Tensor[dtype].d2([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ])
    assert_true(t.grad() == expected_grad)

    print("✓ Passed ReLU multidimensional forward/backward test")



# =============================================================================
# test_relu_revamped.mojo
#
# Exhaustive tests for revamped ReLU (new ArgumentType / NDBuffer mask path).
# All test names carry prefix:  rlv_  (relu_revamped)
#
# Coverage:
#   Forward:      values, zeros, negatives, mixed, dtype variants
#   Backward:     mask correctness, grad shape, all-negative (zero grad)
#   Grad flow:    chained ops, scalar chain, no-grad tensors
#   Dimensions:   0-d (scalar), 1-d, 2-d, 3-d, 4-d, non-contiguous (slice/transpose)
#   Devices:      CPU (all tests), GPU (guarded with has_accelerator())
# =============================================================================


# =============================================================================
# ── SECTION 1: CPU FORWARD ───────────────────────────────────────────────────
# =============================================================================

fn test_rlv_cpu_fwd_all_positive() raises:
    print("test_rlv_cpu_fwd_all_positive")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=False)
    var out = a.relu()
    assert_true(out.all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


fn test_rlv_cpu_fwd_all_negative() raises:
    print("test_rlv_cpu_fwd_all_negative")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-1.0, -2.0, -3.0], requires_grad=False)
    var out = a.relu()
    assert_true(out.all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


fn test_rlv_cpu_fwd_mixed() raises:
    print("test_rlv_cpu_fwd_mixed")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-3.0, 0.0, 2.0, -1.0, 5.0], requires_grad=False)
    var out = a.relu()
    assert_true(out.all_close(Tensor[dtype].d1([0.0, 0.0, 2.0, 0.0, 5.0])))


fn test_rlv_cpu_fwd_zeros() raises:
    print("test_rlv_cpu_fwd_zeros")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([0.0, 0.0, 0.0], requires_grad=False)
    var out = a.relu()
    assert_true(out.all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


fn test_rlv_cpu_fwd_2d() raises:
    print("test_rlv_cpu_fwd_2d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]], requires_grad=False)
    var out = a.relu()
    assert_true(out.all_close(Tensor[dtype].d2([[0.0, 2.0], [3.0, 0.0]])))


fn test_rlv_cpu_fwd_3d() raises:
    print("test_rlv_cpu_fwd_3d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full([2, 2, 2], -1.0)
    var out = a.relu()
    assert_true(out.all_close(Tensor[dtype].zeros([2, 2, 2])))


fn test_rlv_cpu_fwd_4d() raises:
    print("test_rlv_cpu_fwd_4d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full([2, 2, 2, 2], 3.0)
    var out = a.relu()
    assert_true(out.all_close(Tensor[dtype].full([2, 2, 2, 2], 3.0)))


fn test_rlv_cpu_fwd_scalar() raises:
    print("test_rlv_cpu_fwd_scalar")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(-5.0)
    var out = a.relu()
    assert_true(out.all_close(Tensor[dtype].scalar(0.0)))


fn test_rlv_cpu_fwd_dtype_float64() raises:
    print("test_rlv_cpu_fwd_dtype_float64")
    comptime dtype = DType.float64
    var a = Tensor[dtype].d1([-1.0, 0.0, 1.0], requires_grad=False)
    var out = a.relu()
    assert_true(out.all_close(Tensor[dtype].d1([0.0, 0.0, 1.0])))


# =============================================================================
# ── SECTION 2: CPU BACKWARD (mask correctness) ───────────────────────────────
# =============================================================================

fn test_rlv_cpu_bwd_all_positive() raises:
    print("test_rlv_cpu_bwd_all_positive")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var out = a.relu()
    var loss = out.sum()
    loss.backward()
    # All inputs > 0 → mask = 1 everywhere → grad = 1
    assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


fn test_rlv_cpu_bwd_all_negative() raises:
    print("test_rlv_cpu_bwd_all_negative")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-1.0, -2.0, -3.0], requires_grad=True)
    var out = a.relu()
    var loss = out.sum()
    loss.backward()
    # All inputs < 0 → mask = 0 everywhere → grad = 0
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


fn test_rlv_cpu_bwd_mixed() raises:
    print("test_rlv_cpu_bwd_mixed")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
    var out = a.relu()
    var loss = out.sum()
    loss.backward()
    # mask follows sign of input
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 0.0, 1.0])))


fn test_rlv_cpu_bwd_zero_boundary() raises:
    print("test_rlv_cpu_bwd_zero_boundary")
    comptime dtype = DType.float32
    # At exactly 0: mask = 0 (x > 0 is false)
    var a = Tensor[dtype].d1([0.0, 1.0, -1.0], requires_grad=True)
    var out = a.relu()
    var loss = out.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 0.0])))


fn test_rlv_cpu_bwd_2d() raises:
    print("test_rlv_cpu_bwd_2d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    var out = a.relu()
    var loss = out.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d2([[0.0, 1.0], [1.0, 0.0]])))


fn test_rlv_cpu_bwd_3d() raises:
    print("test_rlv_cpu_bwd_3d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full([2, 2, 2], 1.0, requires_grad=True)
    var out = a.relu()
    var loss = out.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


fn test_rlv_cpu_bwd_4d() raises:
    print("test_rlv_cpu_bwd_4d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].full([2, 3, 4, 5], -1.0, requires_grad=True)
    var out = a.relu()
    var loss = out.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].zeros([2, 3, 4, 5])))


fn test_rlv_cpu_bwd_grad_shape_preserved() raises:
    print("test_rlv_cpu_bwd_grad_shape_preserved")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, -1.0, 2.0], [-2.0, 3.0, -3.0]], requires_grad=True)
    var out = a.relu()
    var loss = out.sum()
    loss.backward()
    assert_equal(a.grad().shape(), a.shape())


# =============================================================================
# ── SECTION 3: CPU GRAD FLOW ─────────────────────────────────────────────────
# =============================================================================

fn test_rlv_cpu_grad_chain_relu_mul() raises:
    print("test_rlv_cpu_grad_chain_relu_mul")
    comptime dtype = DType.float32
    # y = relu(a) * 2  →  dy/da = 2 * mask
    var a = Tensor[dtype].d1([-1.0, 2.0, 3.0], requires_grad=True)
    var r = a.relu()
    var out = r * Tensor[dtype].full([3], 2.0)
    var loss = out.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 2.0, 2.0])))


fn test_rlv_cpu_grad_chain_relu_relu() raises:
    print("test_rlv_cpu_grad_chain_relu_relu")
    comptime dtype = DType.float32
    # Double relu: second relu is identity for positive outputs of first
    var a = Tensor[dtype].d1([-1.0, 2.0, 3.0], requires_grad=True)
    var r1 = a.relu()
    var r2 = r1.relu()
    var loss = r2.sum()
    loss.backward()
    # grad = mask1 * mask2 = mask1 (since r1 >= 0 always)
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 1.0])))


fn test_rlv_cpu_grad_chain_add_relu() raises:
    print("test_rlv_cpu_grad_chain_add_relu")
    comptime dtype = DType.float32
    # z = relu(a + b)
    var a = Tensor[dtype].d1([-1.0, 1.0], requires_grad=True)
    var b = Tensor[dtype].d1([2.0, -3.0], requires_grad=True)
    var out = (a + b).relu()
    var loss = out.sum()
    loss.backward()
    # a+b = [1.0, -2.0] → mask = [1, 0]
    assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 0.0])))
    assert_true(b.grad().all_close(Tensor[dtype].d1([1.0, 0.0])))


fn test_rlv_cpu_grad_no_grad_tensor() raises:
    print("test_rlv_cpu_grad_no_grad_tensor")
    comptime dtype = DType.float32
    # requires_grad=False → grad should not be populated
    var a = Tensor[dtype].d1([1.0, -1.0, 2.0], requires_grad=False)
    var out = a.relu()
    var loss = out.sum()
    loss.backward()
    assert_true(not a.requires_grad)


fn test_rlv_cpu_grad_scalar_chain() raises:
    print("test_rlv_cpu_grad_scalar_chain")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var out = a.relu()
    var loss = out.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].scalar(1.0)))


fn test_rlv_cpu_grad_large_tensor() raises:
    print("test_rlv_cpu_grad_large_tensor")
    comptime dtype = DType.float32
    # Large 1-d tensor — exercises SIMD chunking in Buffer.unary_ops_with_mask
    var a = Tensor[dtype].full([65536], 1.0, requires_grad=True)
    var out = a.relu()
    var loss = out.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].ones((65536))))


# =============================================================================
# ── SECTION 4: CPU NON-CONTIGUOUS ────────────────────────────────────────────
# =============================================================================

fn test_rlv_cpu_noncontig_transposed() raises:
    print("test_rlv_cpu_noncontig_transposed")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    var t = a.transpose()           # non-contiguous view
    var out = t.relu()
    var loss = out.sum()
    loss.backward()
    # transpose relu forward: max(0, [[−1,3],[2,−4]]) = [[0,3],[2,0]]
    assert_true(out.contiguous().all_close(
        Tensor[dtype].d2([[0.0, 3.0], [2.0, 0.0]])
    ))


fn test_rlv_cpu_noncontig_slice_bwd() raises:
    print("test_rlv_cpu_noncontig_slice_bwd")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-2.0, 1.0, -1.0, 3.0, -5.0], requires_grad=True)
    # Slice [1:4] → [1.0, -1.0, 3.0] (non-contiguous if strides differ, else view)
    var s = a.slice(1, 4)
    var out = s.relu()
    var loss = out.sum()
    loss.backward()
    # Only positions 1,2,3 in a receive grad; mask = [1,0,1]
    assert_true(out.all_close(Tensor[dtype].d1([1.0, 0.0, 3.0])))


# =============================================================================
# ── SECTION 5: GPU FORWARD ───────────────────────────────────────────────────
# =============================================================================

fn test_rlv_gpu_fwd_all_positive() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_fwd_all_positive")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


fn test_rlv_gpu_fwd_all_negative() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_fwd_all_negative")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([-1.0, -2.0, -3.0]).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


fn test_rlv_gpu_fwd_mixed() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_fwd_mixed")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([-3.0, 0.0, 2.0, -1.0, 5.0]).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].d1([0.0, 0.0, 2.0, 0.0, 5.0])))


fn test_rlv_gpu_fwd_2d() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_fwd_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]]).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].d2([[0.0, 2.0], [3.0, 0.0]])))


fn test_rlv_gpu_fwd_3d() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_fwd_3d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].full([2, 3, 4], -1.0).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].zeros([2, 3, 4])))


fn test_rlv_gpu_fwd_4d() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_fwd_4d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].full([2, 2, 2, 2], 3.0).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].full([2, 2, 2, 2], 3.0)))


fn test_rlv_gpu_fwd_large() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_fwd_large")
        comptime dtype = DType.float32
        # Exercises multi-block dispatch in the kernel
        var a = Tensor[dtype].full([131072], 2.0).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].full([131072], 2.0)))


fn test_rlv_gpu_fwd_dtype_float64() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_fwd_dtype_float64")
        comptime dtype = DType.float64
        var a = Tensor[dtype].d1([-1.0, 0.0, 1.0]).to_gpu()
        var out = a.relu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].d1([0.0, 0.0, 1.0])))


# =============================================================================
# ── SECTION 6: GPU BACKWARD (mask stays on device) ───────────────────────────
# =============================================================================

fn test_rlv_gpu_bwd_all_positive() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_bwd_all_positive")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


fn test_rlv_gpu_bwd_all_negative() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_bwd_all_negative")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, -2.0, -3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


fn test_rlv_gpu_bwd_mixed() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_bwd_mixed")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 0.0, 1.0])))


fn test_rlv_gpu_bwd_zero_boundary() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_bwd_zero_boundary")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([0.0, 1.0, -1.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 0.0])))


fn test_rlv_gpu_bwd_2d() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_bwd_2d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d2([[0.0, 1.0], [1.0, 0.0]])))


fn test_rlv_gpu_bwd_3d() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_bwd_3d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full([2, 2, 2], 1.0, requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


fn test_rlv_gpu_bwd_large() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_bwd_large")
        comptime dtype = DType.float32
        # mask stays on GPU through backward — verifies NDBuffer ArgumentType path
        var a_cpu = Tensor[dtype].full([65536], 1.0, requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].ones((65536))))


fn test_rlv_gpu_bwd_grad_shape_preserved() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_bwd_grad_shape_preserved")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, -1.0, 2.0], [-2.0, 3.0, -3.0]], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_equal(a_cpu.grad().shape(), a_cpu.shape())


# =============================================================================
# ── SECTION 7: GPU GRAD FLOW ─────────────────────────────────────────────────
# =============================================================================

fn test_rlv_gpu_grad_chain_relu_mul() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_grad_chain_relu_mul")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, 2.0, 3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var r = a.relu()
        var out = r * Tensor[dtype].full([3], 2.0).to_gpu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([0.0, 2.0, 2.0])))


fn test_rlv_gpu_grad_chain_relu_relu() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_grad_chain_relu_relu")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, 2.0, 3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var r1 = a.relu()
        var r2 = r1.relu()
        var loss = r2.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([0.0, 1.0, 1.0])))


fn test_rlv_gpu_grad_chain_add_relu() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_grad_chain_add_relu")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, 1.0], requires_grad=True)
        var b_cpu = Tensor[dtype].d1([2.0, -3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var b = b_cpu.to_gpu()
        var out = (a + b).relu()
        var loss = out.sum()
        loss.backward()
        # a+b = [1.0, -2.0] → mask = [1, 0]
        assert_true(a_cpu.grad().all_close(Tensor[dtype].d1([1.0, 0.0])))
        assert_true(b_cpu.grad().all_close(Tensor[dtype].d1([1.0, 0.0])))


fn test_rlv_gpu_grad_scalar_chain() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_grad_scalar_chain")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].scalar(3.0, requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        assert_true(a_cpu.grad().all_close(Tensor[dtype].scalar(1.0)))


# =============================================================================
# ── SECTION 8: GPU NON-CONTIGUOUS ────────────────────────────────────────────
# (exercises contiguous_device_state() single-sweep path)
# =============================================================================

fn test_rlv_gpu_noncontig_transposed_fwd() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_noncontig_transposed_fwd")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]]).to_gpu()
        var t = a.transpose()       # non-contiguous on GPU
        var out = t.relu()
        assert_true(out.to_cpu().all_close(
            Tensor[dtype].d2([[0.0, 3.0], [2.0, 0.0]])
        ))


fn test_rlv_gpu_noncontig_transposed_bwd() raises:
    comptime if has_accelerator():
        print("test_rlv_gpu_noncontig_transposed_bwd")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
        var a = a_cpu.to_gpu()
        var t = a.transpose()
        var out = t.relu()
        var loss = out.sum()
        loss.backward()
        # Gradient flows back through transpose: mask on transposed layout
        assert_equal(a_cpu.grad().shape(), a_cpu.shape())


fn test_rlv_gpu_noncontig_single_map_to_host() raises:
    comptime if has_accelerator():
        # Specifically verifies that non-contiguous GPU input does NOT
        # cause per-element map_to_host calls — exercises the
        # contiguous_device_state() single-sweep path in launch_with_mask.
        print("test_rlv_gpu_noncontig_single_map_to_host")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full([64, 64], 1.0, requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var a = a_gpu.transpose()  # non-contiguous 64x64
        var out = a.relu()
        var loss = out.sum()
        loss.backward()
        # All positive → grad = 1 everywhere, shape [64, 64] transposed back
        assert_equal(a_cpu.grad().shape(), a_cpu.shape())


# =============================================================================
# ── SECTION 9: CPU / GPU PARITY ──────────────────────────────────────────────
# =============================================================================

fn test_rlv_parity_fwd_1d() raises:
    comptime if has_accelerator():
        print("test_rlv_parity_fwd_1d")
        comptime dtype = DType.float32
        var data = Tensor[dtype].d1([-3.0, -1.0, 0.0, 1.0, 3.0])
        var cpu_out = data.relu()
        var gpu_out = data.to_gpu().relu().to_cpu()
        assert_true(cpu_out.all_close(gpu_out))


fn test_rlv_parity_fwd_2d() raises:
    comptime if has_accelerator():
        print("test_rlv_parity_fwd_2d")
        comptime dtype = DType.float32
        var data = Tensor[dtype].d2([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
        var cpu_out = data.relu()
        var gpu_out = data.to_gpu().relu().to_cpu()
        assert_true(cpu_out.all_close(gpu_out))


fn test_rlv_parity_bwd_2d() raises:
    comptime if has_accelerator():
        print("test_rlv_parity_bwd_2d")
        comptime dtype = DType.float32

        var a_cpu = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
        var out_cpu = a_cpu.relu()
        var loss_cpu = out_cpu.sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad()

        var a_cpu2 = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
        var a_gpu = a_cpu2.to_gpu()
        var out_gpu = a_gpu.relu()
        var loss_gpu = out_gpu.sum()
        loss_gpu.backward()
        var gpu_grad = a_cpu2.grad()

        assert_true(cpu_grad.all_close(gpu_grad))


# =============================================================================
# ── MAIN ─────────────────────────────────────────────────────────────────────
# =============================================================================

fn main() raises:
    # ── CPU Forward ──────────────────────────────────────────────────────────
    test_rlv_cpu_fwd_all_positive()
    test_rlv_cpu_fwd_all_negative()
    test_rlv_cpu_fwd_mixed()
    test_rlv_cpu_fwd_zeros()
    test_rlv_cpu_fwd_2d()
    test_rlv_cpu_fwd_3d()
    test_rlv_cpu_fwd_4d()
    test_rlv_cpu_fwd_scalar()
    test_rlv_cpu_fwd_dtype_float64()

    # ── CPU Backward ─────────────────────────────────────────────────────────
    test_rlv_cpu_bwd_all_positive()
    test_rlv_cpu_bwd_all_negative()
    test_rlv_cpu_bwd_mixed()
    test_rlv_cpu_bwd_zero_boundary()
    test_rlv_cpu_bwd_2d()
    test_rlv_cpu_bwd_3d()
    test_rlv_cpu_bwd_4d()
    test_rlv_cpu_bwd_grad_shape_preserved()

    # ── CPU Grad Flow ─────────────────────────────────────────────────────────
    test_rlv_cpu_grad_chain_relu_mul()
    test_rlv_cpu_grad_chain_relu_relu()
    test_rlv_cpu_grad_chain_add_relu()
    test_rlv_cpu_grad_no_grad_tensor()
    test_rlv_cpu_grad_scalar_chain()
    test_rlv_cpu_grad_large_tensor()

    # ── CPU Non-contiguous ───────────────────────────────────────────────────
    test_rlv_cpu_noncontig_transposed()
    test_rlv_cpu_noncontig_slice_bwd()

    # ── GPU Forward ──────────────────────────────────────────────────────────
    test_rlv_gpu_fwd_all_positive()
    test_rlv_gpu_fwd_all_negative()
    test_rlv_gpu_fwd_mixed()
    test_rlv_gpu_fwd_2d()
    test_rlv_gpu_fwd_3d()
    test_rlv_gpu_fwd_4d()
    test_rlv_gpu_fwd_large()
    test_rlv_gpu_fwd_dtype_float64()

    # ── GPU Backward ─────────────────────────────────────────────────────────
    test_rlv_gpu_bwd_all_positive()
    test_rlv_gpu_bwd_all_negative()
    test_rlv_gpu_bwd_mixed()
    test_rlv_gpu_bwd_zero_boundary()
    test_rlv_gpu_bwd_2d()
    test_rlv_gpu_bwd_3d()
    test_rlv_gpu_bwd_large()
    test_rlv_gpu_bwd_grad_shape_preserved()

    # ── GPU Grad Flow ─────────────────────────────────────────────────────────
    test_rlv_gpu_grad_chain_relu_mul()
    test_rlv_gpu_grad_chain_relu_relu()
    test_rlv_gpu_grad_chain_add_relu()
    test_rlv_gpu_grad_scalar_chain()

    # ── GPU Non-contiguous ───────────────────────────────────────────────────
    test_rlv_gpu_noncontig_transposed_fwd()
    test_rlv_gpu_noncontig_transposed_bwd()
    test_rlv_gpu_noncontig_single_map_to_host()

    # ── CPU / GPU Parity ─────────────────────────────────────────────────────
    test_rlv_parity_fwd_1d()
    test_rlv_parity_fwd_2d()
    test_rlv_parity_bwd_2d()

    #Old
    test_relu_basic()
    test_relu_multidim()


    print("All rlv_ tests passed.")
