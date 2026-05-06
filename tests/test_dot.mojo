from std.testing import assert_true, assert_false, TestSuite
from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from std.sys import has_accelerator


# =============================================================================
# Exhaustive tests for Tensor.dot()
# Prefix: test_dot_ on all test names to avoid collision with existing tests
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CPU · Forward correctness · vector · vector
# ─────────────────────────────────────────────────────────────────────────────

fn test_dot_cpu_fwd_vector_vector_basic() raises:
    print("test_dot_cpu_fwd_vector_vector_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = Tensor[dtype].d1([4.0, 5.0, 6.0])
    var result = a.dot(b)
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_true(result.all_close(Tensor[dtype].scalar(32.0)))


fn test_dot_cpu_fwd_vector_vector_ones() raises:
    print("test_dot_cpu_fwd_vector_vector_ones")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0])
    var b = Tensor[dtype].d1([2.0, 3.0, 4.0, 5.0])
    var result = a.dot(b)
    # 1*2 + 1*3 + 1*4 + 1*5 = 14
    assert_true(result.all_close(Tensor[dtype].scalar(14.0)))


fn test_dot_cpu_fwd_vector_vector_negative() raises:
    print("test_dot_cpu_fwd_vector_vector_negative")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, -2.0, 3.0])
    var b = Tensor[dtype].d1([-1.0, 2.0, -3.0])
    var result = a.dot(b)
    # -1 - 4 - 9 = -14
    assert_true(result.all_close(Tensor[dtype].scalar(-14.0)))


fn test_dot_cpu_fwd_vector_vector_single_element() raises:
    print("test_dot_cpu_fwd_vector_vector_single_element")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([3.0])
    var b = Tensor[dtype].d1([7.0])
    var result = a.dot(b)
    assert_true(result.all_close(Tensor[dtype].scalar(21.0)))


fn test_dot_cpu_fwd_vector_vector_orthogonal() raises:
    print("test_dot_cpu_fwd_vector_vector_orthogonal")
    comptime dtype = DType.float32
    # Orthogonal vectors → dot = 0
    var a = Tensor[dtype].d1([1.0, 0.0])
    var b = Tensor[dtype].d1([0.0, 1.0])
    var result = a.dot(b)
    assert_true(result.all_close(Tensor[dtype].scalar(0.0)))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CPU · Forward correctness · scalar · vector and vector · scalar
# ─────────────────────────────────────────────────────────────────────────────

fn test_dot_cpu_fwd_scalar_tensor_dot_vector() raises:
    print("test_dot_cpu_fwd_scalar_tensor_dot_vector")
    comptime dtype = DType.float32
    # scalar tensor (numels=1) · vector → scalar broadcast then dot
    var a = Tensor[dtype].scalar(2.0)
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.dot(b)
    # 2*1 + 2*2 + 2*3 = 2 + 4 + 6 = 12
    assert_true(result.all_close(Tensor[dtype].scalar(12.0)))


fn test_dot_cpu_fwd_vector_dot_scalar_tensor() raises:
    print("test_dot_cpu_fwd_vector_dot_scalar_tensor")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = Tensor[dtype].scalar(3.0)
    var result = a.dot(b)
    # 1*3 + 2*3 + 3*3 = 3 + 6 + 9 = 18
    assert_true(result.all_close(Tensor[dtype].scalar(18.0)))


fn test_dot_cpu_fwd_scalar_literal_dot_vector() raises:
    print("test_dot_cpu_fwd_scalar_literal_dot_vector")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([4.0, 5.0, 6.0])
    var result = a.dot(2.0)
    # 4*2 + 5*2 + 6*2 = 8 + 10 + 12 = 30
    assert_true(result.all_close(Tensor[dtype].scalar(30.0)))


fn test_dot_cpu_fwd_d1_single_dot_scalar_tensor() raises:
    print("test_dot_cpu_fwd_d1_single_dot_scalar_tensor")
    comptime dtype = DType.float32
    # d1([x]) vs scalar — both numels=1
    var a = Tensor[dtype].d1([5.0])
    var b = Tensor[dtype].scalar(4.0)
    var result = a.dot(b)
    assert_true(result.all_close(Tensor[dtype].scalar(20.0)))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CPU · Backward · vector · vector
# ─────────────────────────────────────────────────────────────────────────────

fn test_dot_cpu_bwd_vector_vector_both_grad() raises:
    print("test_dot_cpu_bwd_vector_vector_both_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)
    var loss = a.dot(b)
    loss.backward()
    # d(dot)/da = b, d(dot)/db = a
    assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 5.0, 6.0])))
    assert_true(b.grad().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


fn test_dot_cpu_bwd_vector_vector_lhs_only() raises:
    print("test_dot_cpu_bwd_vector_vector_lhs_only")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].d1([4.0, 5.0, 6.0])
    var loss = a.dot(b)
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 5.0, 6.0])))


fn test_dot_cpu_bwd_vector_vector_rhs_only() raises:
    print("test_dot_cpu_bwd_vector_vector_rhs_only")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)
    var loss = a.dot(b)
    loss.backward()
    assert_true(b.grad().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


fn test_dot_cpu_bwd_vector_vector_ones() raises:
    print("test_dot_cpu_bwd_vector_vector_ones")
    comptime dtype = DType.float32
    # dot with all-ones vector → grad is all-ones
    var a = Tensor[dtype].d1([3.0, 7.0, 2.0], requires_grad=True)
    var b = Tensor[dtype].d1([1.0, 1.0, 1.0], requires_grad=True)
    var loss = a.dot(b)
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))
    assert_true(b.grad().all_close(Tensor[dtype].d1([3.0, 7.0, 2.0])))


fn test_dot_cpu_bwd_vector_vector_chained() raises:
    print("test_dot_cpu_bwd_vector_vector_chained")
    comptime dtype = DType.float32
    # dot then scale then backward
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)
    var d = a.dot(b)
    # d = 32; scale by 2
    var loss = d + d
    loss.backward()
    # grad of (2 * dot) w.r.t a = 2 * b, w.r.t b = 2 * a
    assert_true(a.grad().all_close(Tensor[dtype].d1([8.0, 10.0, 12.0])))
    assert_true(b.grad().all_close(Tensor[dtype].d1([2.0, 4.0, 6.0])))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CPU · Backward · scalar · vector (broadcast path)
# ─────────────────────────────────────────────────────────────────────────────

fn test_dot_cpu_bwd_scalar_tensor_dot_vector() raises:
    print("test_dot_cpu_bwd_scalar_tensor_dot_vector")
    comptime dtype = DType.float32
    # scalar (numels=1) broadcast to match vector
    var a = Tensor[dtype].scalar(2.0, requires_grad=True)
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var loss = a.dot(b)
    loss.backward()
    # a is broadcast to [2,2,2]; grad of a = sum(b) = 6
    assert_true(a.grad().all_close(Tensor[dtype].scalar(6.0)))
    # grad of b = broadcast(a) = [2,2,2]
    assert_true(b.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


fn test_dot_cpu_bwd_vector_dot_scalar_tensor() raises:
    print("test_dot_cpu_bwd_vector_dot_scalar_tensor")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].scalar(3.0, requires_grad=True)
    var loss = a.dot(b)
    loss.backward()
    # grad of a = broadcast(b) = [3,3,3]
    assert_true(a.grad().all_close(Tensor[dtype].d1([3.0, 3.0, 3.0])))
    # grad of b = sum(a) = 6
    assert_true(b.grad().all_close(Tensor[dtype].scalar(6.0)))


fn test_dot_cpu_bwd_scalar_literal_dot_vector() raises:
    print("test_dot_cpu_bwd_scalar_literal_dot_vector")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)
    var loss = a.dot(2.0)
    loss.backward()
    # scalar literal has no grad; grad of a = [2,2,2]
    assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


fn test_dot_cpu_bwd_scalar_tensor_lhs_only() raises:
    print("test_dot_cpu_bwd_scalar_tensor_lhs_only")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(5.0, requires_grad=True)
    var b = Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0])
    var loss = a.dot(b)
    loss.backward()
    # grad of a = sum(b) = 4
    assert_true(a.grad().all_close(Tensor[dtype].scalar(4.0)))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GPU · Forward correctness
# ─────────────────────────────────────────────────────────────────────────────

fn test_dot_gpu_fwd_vector_vector_basic() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_fwd_vector_vector_basic")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var b = Tensor[dtype].d1([4.0, 5.0, 6.0]).to_gpu()
        var result = a.dot(b)
        assert_true(result.to_cpu().all_close(Tensor[dtype].scalar(32.0)))


fn test_dot_gpu_fwd_vector_vector_negative() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_fwd_vector_vector_negative")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, -2.0, 3.0]).to_gpu()
        var b = Tensor[dtype].d1([-1.0, 2.0, -3.0]).to_gpu()
        var result = a.dot(b)
        assert_true(result.to_cpu().all_close(Tensor[dtype].scalar(-14.0)))


fn test_dot_gpu_fwd_scalar_tensor_dot_vector() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_fwd_scalar_tensor_dot_vector")
        comptime dtype = DType.float32
        var a = Tensor[dtype].scalar(2.0).to_gpu()
        var b = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.dot(b)
        assert_true(result.to_cpu().all_close(Tensor[dtype].scalar(12.0)))


fn test_dot_gpu_fwd_vector_dot_scalar_tensor() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_fwd_vector_dot_scalar_tensor")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var b = Tensor[dtype].scalar(3.0).to_gpu()
        var result = a.dot(b)
        assert_true(result.to_cpu().all_close(Tensor[dtype].scalar(18.0)))


fn test_dot_gpu_fwd_scalar_literal_dot_vector() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_fwd_scalar_literal_dot_vector")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([4.0, 5.0, 6.0]).to_gpu()
        var result = a.dot(2.0)
        assert_true(result.to_cpu().all_close(Tensor[dtype].scalar(30.0)))


fn test_dot_gpu_fwd_orthogonal() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_fwd_orthogonal")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 0.0]).to_gpu()
        var b = Tensor[dtype].d1([0.0, 1.0]).to_gpu()
        var result = a.dot(b)
        assert_true(result.to_cpu().all_close(Tensor[dtype].scalar(0.0)))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GPU · Backward · vector · vector
# ─────────────────────────────────────────────────────────────────────────────

fn test_dot_gpu_bwd_vector_vector_both_grad() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_bwd_vector_vector_both_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var loss = a_gpu.dot(b_gpu)
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 5.0, 6.0])))
        assert_true(b.grad().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


fn test_dot_gpu_bwd_vector_vector_lhs_only() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_bwd_vector_vector_lhs_only")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].d1([4.0, 5.0, 6.0])
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var loss = a_gpu.dot(b_gpu)
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 5.0, 6.0])))


fn test_dot_gpu_bwd_vector_vector_rhs_only() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_bwd_vector_vector_rhs_only")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var b = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var loss = a_gpu.dot(b_gpu)
        loss.backward()
        assert_true(b.grad().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — GPU · Backward · scalar · vector (broadcast path)
# ─────────────────────────────────────────────────────────────────────────────

fn test_dot_gpu_bwd_scalar_tensor_dot_vector() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_bwd_scalar_tensor_dot_vector")
        comptime dtype = DType.float32
        var a = Tensor[dtype].scalar(2.0, requires_grad=True)
        var b = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var loss = a_gpu.dot(b_gpu)
        loss.backward()
        # grad of a = sum(b) = 6
        assert_true(a.grad().all_close(Tensor[dtype].scalar(6.0)))
        # grad of b = broadcast(a) = [2,2,2]
        assert_true(b.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


fn test_dot_gpu_bwd_vector_dot_scalar_tensor() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_bwd_vector_dot_scalar_tensor")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].scalar(3.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var loss = a_gpu.dot(b_gpu)
        loss.backward()
        # grad of a = [3,3,3]
        assert_true(a.grad().all_close(Tensor[dtype].d1([3.0, 3.0, 3.0])))
        # grad of b = sum(a) = 6
        assert_true(b.grad().all_close(Tensor[dtype].scalar(6.0)))


fn test_dot_gpu_bwd_scalar_literal_dot_vector() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_bwd_scalar_literal_dot_vector")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var loss = a_gpu.dot(2.0)
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


fn test_dot_gpu_bwd_chained() raises:
    comptime if has_accelerator():
        print("test_dot_gpu_bwd_chained")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var d = a_gpu.dot(b_gpu)
        var loss = d + d
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([8.0, 10.0, 12.0])))
        assert_true(b.grad().all_close(Tensor[dtype].d1([2.0, 4.0, 6.0])))


# =============================================================================
# Main
# =============================================================================

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll [dot] tests passed!")
