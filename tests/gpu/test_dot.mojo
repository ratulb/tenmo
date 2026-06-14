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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CPU · Forward correctness · scalar · vector and vector · scalar
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CPU · Backward · vector · vector
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CPU · Backward · scalar · vector (broadcast path)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GPU · Forward correctness
# ─────────────────────────────────────────────────────────────────────────────


def test_dot_gpu_fwd_vector_vector_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var b = Tensor[dtype].d1([4.0, 5.0, 6.0]).to_gpu()
        var result = a.dot(b)
        assert_true(result.to_cpu().all_close(Tensor[dtype].scalar(32.0)))


def test_dot_gpu_fwd_vector_vector_negative() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, -2.0, 3.0]).to_gpu()
        var b = Tensor[dtype].d1([-1.0, 2.0, -3.0]).to_gpu()
        var result = a.dot(b)
        assert_true(result.to_cpu().all_close(Tensor[dtype].scalar(-14.0)))


def test_dot_gpu_fwd_scalar_tensor_dot_vector() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].scalar(2.0).to_gpu()
        var b = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.dot(b)
        assert_true(result.to_cpu().all_close(Tensor[dtype].scalar(12.0)))


def test_dot_gpu_fwd_vector_dot_scalar_tensor() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var b = Tensor[dtype].scalar(3.0).to_gpu()
        var result = a.dot(b)
        assert_true(result.to_cpu().all_close(Tensor[dtype].scalar(18.0)))


def test_dot_gpu_fwd_scalar_literal_dot_vector() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([4.0, 5.0, 6.0]).to_gpu()
        var result = a.dot(2.0)
        assert_true(result.to_cpu().all_close(Tensor[dtype].scalar(30.0)))


def test_dot_gpu_fwd_orthogonal() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 0.0]).to_gpu()
        var b = Tensor[dtype].d1([0.0, 1.0]).to_gpu()
        var result = a.dot(b)
        assert_true(result.to_cpu().all_close(Tensor[dtype].scalar(0.0)))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GPU · Backward · vector · vector
# ─────────────────────────────────────────────────────────────────────────────


def test_dot_gpu_bwd_vector_vector_both_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var loss = a_gpu.dot(b_gpu)
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 5.0, 6.0])))
        assert_true(b.grad().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


def test_dot_gpu_bwd_vector_vector_lhs_only() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].d1([4.0, 5.0, 6.0])
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var loss = a_gpu.dot(b_gpu)
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 5.0, 6.0])))


def test_dot_gpu_bwd_vector_vector_rhs_only() raises:
    comptime if has_accelerator():
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


def test_dot_gpu_bwd_scalar_tensor_dot_vector() raises:
    comptime if has_accelerator():
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


def test_dot_gpu_bwd_vector_dot_scalar_tensor() raises:
    comptime if has_accelerator():
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


def test_dot_gpu_bwd_scalar_literal_dot_vector() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var loss = a_gpu.dot(2.0)
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


def test_dot_gpu_bwd_chained() raises:
    comptime if has_accelerator():
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
