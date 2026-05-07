from std.testing import assert_true, assert_false, TestSuite
from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from std.sys import has_accelerator


# =============================================================================
# Exhaustive tests for Tensor.outer()
# Prefix: test_outer_ on all test names to avoid collision with existing tests
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CPU · Forward correctness · 1-D × 1-D
# ─────────────────────────────────────────────────────────────────────────────

fn test_outer_cpu_fwd_basic() raises:
    print("test_outer_cpu_fwd_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = Tensor[dtype].d1([4.0, 5.0])
    var result = a.outer(b)
    # result[i,j] = a[i] * b[j]
    assert_true(result.all_close(Tensor[dtype].d2([
        [4.0,  5.0],
        [8.0,  10.0],
        [12.0, 15.0],
    ])))


fn test_outer_cpu_fwd_square() raises:
    print("test_outer_cpu_fwd_square")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.outer(b)
    assert_true(result.all_close(Tensor[dtype].d2([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
    ])))


fn test_outer_cpu_fwd_vector_scalar() raises:
    print("test_outer_cpu_fwd_vector_scalar")
    comptime dtype = DType.float32
    # (hidden_size,) × scalar → (hidden_size, 1) — the IMDB gradient case
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var b = Tensor[dtype].scalar(2.0)
    var result = a.outer(b)
    assert_true(result.shape() == Shape(4, 1))
    assert_true(result.all_close(Tensor[dtype].d2([
        [2.0], [4.0], [6.0], [8.0]
    ])))


fn test_outer_cpu_fwd_scalar_vector() raises:
    print("test_outer_cpu_fwd_scalar_vector")
    comptime dtype = DType.float32
    # scalar × vector → (1, n)
    var a = Tensor[dtype].scalar(3.0)
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.outer(b)
    assert_true(result.shape() == Shape(1, 3))
    assert_true(result.all_close(Tensor[dtype].d2([[3.0, 6.0, 9.0]])))


fn test_outer_cpu_fwd_negative_values() raises:
    print("test_outer_cpu_fwd_negative_values")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, -2.0])
    var b = Tensor[dtype].d1([-3.0, 4.0])
    var result = a.outer(b)
    assert_true(result.all_close(Tensor[dtype].d2([
        [-3.0,  4.0],
        [ 6.0, -8.0],
    ])))


fn test_outer_cpu_fwd_ones() raises:
    print("test_outer_cpu_fwd_ones")
    comptime dtype = DType.float32
    # outer(ones, ones) = all-ones matrix
    var a = Tensor[dtype].d1([1.0, 1.0, 1.0])
    var b = Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0])
    var result = a.outer(b)
    assert_true(result.all_close(Tensor[dtype].ones(Shape(3, 4))))


fn test_outer_cpu_fwd_single_elements() raises:
    print("test_outer_cpu_fwd_single_elements")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([5.0])
    var b = Tensor[dtype].d1([7.0])
    var result = a.outer(b)
    assert_true(result.shape() == Shape(1, 1))
    assert_true(result.all_close(Tensor[dtype].d2([[35.0]])))


fn test_outer_cpu_fwd_result_shape() raises:
    print("test_outer_cpu_fwd_result_shape")
    comptime dtype = DType.float32
    # Shape must always be (m, n) regardless of input shapes
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.outer(b)
    assert_true(result.shape() == Shape(5, 3))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CPU · Forward · 2-D and higher inputs (auto-flatten)
# ─────────────────────────────────────────────────────────────────────────────

fn test_outer_cpu_fwd_2d_inputs_flattened() raises:
    print("test_outer_cpu_fwd_2d_inputs_flattened")
    comptime dtype = DType.float32
    # (2,2) × (2,2) → flatten to (4,) × (4,) → (4,4)
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var b = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]])
    var result = a.outer(b)
    assert_true(result.shape() == Shape(4, 4))
    # a flattened = [1,2,3,4], b flattened = [1,0,0,1]
    assert_true(result.all_close(Tensor[dtype].d2([
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 2.0],
        [3.0, 0.0, 0.0, 3.0],
        [4.0, 0.0, 0.0, 4.0],
    ])))


fn test_outer_cpu_fwd_2d_vector_flattened() raises:
    print("test_outer_cpu_fwd_2d_vector_flattened")
    comptime dtype = DType.float32
    # (2,2) flattened × (3,) → (4,3)
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.outer(b)
    assert_true(result.shape() == Shape(4, 3))
    assert_true(result.all_close(Tensor[dtype].d2([
        [1.0, 2.0,  3.0],
        [2.0, 4.0,  6.0],
        [3.0, 6.0,  9.0],
        [4.0, 8.0, 12.0],
    ])))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CPU · Backward · grad flows through both inputs
# ─────────────────────────────────────────────────────────────────────────────

fn test_outer_cpu_bwd_both_grad() raises:
    print("test_outer_cpu_bwd_both_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].d1([4.0, 5.0], requires_grad=True)
    var result = a.outer(b)
    var loss = result.sum()
    loss.backward()
    # d(sum(outer(a,b)))/da[i] = sum_j(b[j]) = 4+5 = 9 for all i
    assert_true(a.grad().all_close(Tensor[dtype].d1([9.0, 9.0, 9.0])))
    # d(sum(outer(a,b)))/db[j] = sum_i(a[i]) = 1+2+3 = 6 for all j
    assert_true(b.grad().all_close(Tensor[dtype].d1([6.0, 6.0])))


fn test_outer_cpu_bwd_lhs_only() raises:
    print("test_outer_cpu_bwd_lhs_only")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].d1([4.0, 5.0])
    var result = a.outer(b)
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([9.0, 9.0, 9.0])))


fn test_outer_cpu_bwd_rhs_only() raises:
    print("test_outer_cpu_bwd_rhs_only")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = Tensor[dtype].d1([4.0, 5.0], requires_grad=True)
    var result = a.outer(b)
    var loss = result.sum()
    loss.backward()
    assert_true(b.grad().all_close(Tensor[dtype].d1([6.0, 6.0])))


fn test_outer_cpu_bwd_vector_scalar() raises:
    print("test_outer_cpu_bwd_vector_scalar")
    comptime dtype = DType.float32
    # The IMDB gradient case: hidden (n,) × output_delta scalar → (n,1)
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].scalar(2.0, requires_grad=True)
    var result = a.outer(b)
    var loss = result.sum()
    loss.backward()
    # d/da[i] = sum_j(b[j]) = 2.0 for all i
    assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))
    # d/db = sum_i(a[i]) = 1+2+3 = 6
    assert_true(b.grad().all_close(Tensor[dtype].scalar(6.0)))


fn test_outer_cpu_bwd_ones() raises:
    print("test_outer_cpu_bwd_ones")
    comptime dtype = DType.float32
    # outer(ones_m, ones_n) → all-ones (m,n); grad of sum = ones
    # d/da[i] = sum_j(1) = n = 4 for all i
    # d/db[j] = sum_i(1) = m = 3 for all j
    var a = Tensor[dtype].d1([1.0, 1.0, 1.0], requires_grad=True)
    var b = Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0], requires_grad=True)
    var result = a.outer(b)
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 4.0, 4.0])))
    assert_true(b.grad().all_close(Tensor[dtype].d1([3.0, 3.0, 3.0, 3.0])))


fn test_outer_cpu_bwd_chained_multiply() raises:
    print("test_outer_cpu_bwd_chained_multiply")
    comptime dtype = DType.float32
    # outer then scale then sum — verify chain rule
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var b = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
    var outer = a.outer(b)                            # [[3,4],[6,8]]
    var scale = Tensor[dtype].d2([[2.0, 2.0],[2.0, 2.0]])
    var c = outer * scale                             # [[6,8],[12,16]]
    var loss = c.sum()                                # 42
    loss.backward()
    # upstream grad = scale = [[2,2],[2,2]]
    # d/da[i] = sum_j(upstream[i,j] * b[j]) = 2*3 + 2*4 = 14 for all i
    assert_true(a.grad().all_close(Tensor[dtype].d1([14.0, 14.0])))
    # d/db[j] = sum_i(upstream[i,j] * a[i]) = 2*1 + 2*2 = 6 for all j
    assert_true(b.grad().all_close(Tensor[dtype].d1([6.0, 6.0])))


fn test_outer_cpu_bwd_chained_add() raises:
    print("test_outer_cpu_bwd_chained_add")
    comptime dtype = DType.float32
    # outer + another tensor then sum — verifies grad accumulation
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].d1([1.0, 1.0], requires_grad=True)
    var outer = a.outer(b)                            # [[1,1],[2,2],[3,3]]
    var c = Tensor[dtype].d2([[1.0,1.0],[1.0,1.0],[1.0,1.0]])
    var d = outer + c
    var loss = d.sum()
    loss.backward()
    # upstream grad to outer = all-ones (3,2)
    # d/da[i] = sum_j(b[j]) = 1+1 = 2 for all i
    assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))
    # d/db[j] = sum_i(a[i]) = 1+2+3 = 6 for all j
    assert_true(b.grad().all_close(Tensor[dtype].d1([6.0, 6.0])))


fn test_outer_cpu_bwd_no_grad_when_neither_requires() raises:
    print("test_outer_cpu_bwd_no_grad_when_neither_requires")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var b = Tensor[dtype].d1([4.0, 5.0])
    var result = a.outer(b)
    assert_false(result.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CPU · track_grad=False explicit
# ─────────────────────────────────────────────────────────────────────────────

fn test_outer_cpu_no_track_grad_explicit() raises:
    print("test_outer_cpu_no_track_grad_explicit")
    comptime dtype = DType.float32
    # Even if inputs require grad, track_grad=False suppresses it
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].d1([4.0, 5.0], requires_grad=True)
    var result = a.outer[track_grad=False](b)
    assert_false(result.requires_grad)
    # Values still correct
    assert_true(result.all_close(Tensor[dtype].d2([
        [4.0,  5.0],
        [8.0,  10.0],
        [12.0, 15.0],
    ])))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GPU · Forward correctness
# ─────────────────────────────────────────────────────────────────────────────

fn test_outer_gpu_fwd_basic() raises:
    comptime if has_accelerator():
        print("test_outer_gpu_fwd_basic")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var b = Tensor[dtype].d1([4.0, 5.0]).to_gpu()
        var result = a.outer(b)
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([
            [4.0,  5.0],
            [8.0,  10.0],
            [12.0, 15.0],
        ])))


fn test_outer_gpu_fwd_vector_scalar() raises:
    comptime if has_accelerator():
        print("test_outer_gpu_fwd_vector_scalar")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        var b = Tensor[dtype].scalar(2.0).to_gpu()
        var result = a.outer(b)
        assert_true(result.shape() == Shape(4, 1))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([
            [2.0], [4.0], [6.0], [8.0]
        ])))


fn test_outer_gpu_fwd_negative_values() raises:
    comptime if has_accelerator():
        print("test_outer_gpu_fwd_negative_values")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, -2.0]).to_gpu()
        var b = Tensor[dtype].d1([-3.0, 4.0]).to_gpu()
        var result = a.outer(b)
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([
            [-3.0,  4.0],
            [ 6.0, -8.0],
        ])))


fn test_outer_gpu_fwd_result_shape() raises:
    comptime if has_accelerator():
        print("test_outer_gpu_fwd_result_shape")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0]).to_gpu()
        var b = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.outer(b)
        assert_true(result.shape() == Shape(5, 3))


fn test_outer_gpu_fwd_2d_flattened() raises:
    comptime if has_accelerator():
        print("test_outer_gpu_fwd_2d_flattened")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0],[3.0, 4.0]]).to_gpu()
        var b = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.outer(b)
        assert_true(result.shape() == Shape(4, 3))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([
            [1.0, 2.0,  3.0],
            [2.0, 4.0,  6.0],
            [3.0, 6.0,  9.0],
            [4.0, 8.0, 12.0],
        ])))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GPU · Backward
# ─────────────────────────────────────────────────────────────────────────────

fn test_outer_gpu_bwd_both_grad() raises:
    comptime if has_accelerator():
        print("test_outer_gpu_bwd_both_grad")
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


fn test_outer_gpu_bwd_vector_scalar() raises:
    comptime if has_accelerator():
        print("test_outer_gpu_bwd_vector_scalar")
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


fn test_outer_gpu_bwd_lhs_only() raises:
    comptime if has_accelerator():
        print("test_outer_gpu_bwd_lhs_only")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].d1([4.0, 5.0])
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var result = a_gpu.outer(b_gpu)
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([9.0, 9.0, 9.0])))


fn test_outer_gpu_bwd_chained_multiply() raises:
    comptime if has_accelerator():
        print("test_outer_gpu_bwd_chained_multiply")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var b = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var outer = a_gpu.outer(b_gpu)
        var scale = Tensor[dtype].d2([[2.0, 2.0],[2.0, 2.0]]).to_gpu()
        var c = outer * scale
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([14.0, 14.0])))
        assert_true(b.grad().all_close(Tensor[dtype].d1([6.0, 6.0])))


fn test_outer_gpu_no_track_grad_explicit() raises:
    comptime if has_accelerator():
        print("test_outer_gpu_no_track_grad_explicit")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var b = Tensor[dtype].d1([4.0, 5.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var result = a_gpu.outer[track_grad=False](b_gpu)
        assert_false(result.requires_grad)
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([
            [4.0,  5.0],
            [8.0,  10.0],
            [12.0, 15.0],
        ])))


# =============================================================================
# Main
# =============================================================================

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll [outer] tests passed!")
