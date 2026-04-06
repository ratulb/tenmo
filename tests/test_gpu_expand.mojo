from std.testing import assert_true
from tenmo import Tensor
from shapes import Shape
from std.sys import has_accelerator
# ============================================================
# GPU EXPAND TESTS — forward shape, values, and backward grad flow
# Grad always flows back to the original CPU tensor.
# ============================================================


# ------------------------------------------------------------
# Basic 1D → 2D expansion
# ------------------------------------------------------------

fn test_gpu_expand_1d_to_2d_new_batch_dim() raises:
    print("test_gpu_expand_1d_to_2d_new_batch_dim")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(4, 3)
    assert_true(e.shape() == Shape.of(4, 3))
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    )
    assert_true(e.to_cpu().all_close(expected))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 4.0, 4.0])))


fn test_gpu_expand_1d_to_3d() raises:
    print("test_gpu_expand_1d_to_3d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(3, 4, 2)
    assert_true(e.shape() == Shape.of(3, 4, 2))
    var s = e.sum(axes=[0, 1]).to_cpu()
    assert_true(s.all_close(Tensor[dtype].d1([12.0, 24.0])))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([12.0, 12.0])))


# ------------------------------------------------------------
# 2D → 2D expansions
# ------------------------------------------------------------

fn test_gpu_expand_2d_row_vector_to_matrix() raises:
    print("test_gpu_expand_2d_row_vector_to_matrix")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(4, 3)
    assert_true(e.shape() == Shape.of(4, 3))
    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    )
    assert_true(e.to_cpu().all_close(expected))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0, 4.0, 4.0]])))


fn test_gpu_expand_2d_col_vector_to_matrix() raises:
    print("test_gpu_expand_2d_col_vector_to_matrix")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(3, 4)
    assert_true(e.shape() == Shape.of(3, 4))
    var expected = Tensor[dtype].d2(
        [[1.0, 1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0, 2.0],
         [3.0, 3.0, 3.0, 3.0]]
    )
    assert_true(e.to_cpu().all_close(expected))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0], [4.0], [4.0]])))


fn test_gpu_expand_2d_both_dims_size1() raises:
    print("test_gpu_expand_2d_both_dims_size1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[5.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(3, 4)
    assert_true(e.shape() == Shape.of(3, 4))
    assert_true(e.to_cpu().all_close(Tensor[dtype].full(Shape.of(3, 4), 5.0)))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d2([[12.0]])))


fn test_gpu_expand_2d_no_op_same_shape() raises:
    print("test_gpu_expand_2d_no_op_same_shape")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(2, 2)
    assert_true(e.shape() == Shape.of(2, 2))
    assert_true(e.to_cpu().all_close(a))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# 3D expansions
# ------------------------------------------------------------

fn test_gpu_expand_3d_first_dim() raises:
    print("test_gpu_expand_3d_first_dim")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(5, 2, 2)
    assert_true(e.shape() == Shape.of(5, 2, 2))
    var s = e.sum(axes=[0]).to_cpu()
    assert_true(s.all_close(Tensor[dtype].d2([[5.0, 10.0], [15.0, 20.0]])))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[5.0, 5.0], [5.0, 5.0]]])))


fn test_gpu_expand_3d_last_dim() raises:
    print("test_gpu_expand_3d_last_dim")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(2, 2, 6)
    assert_true(e.shape() == Shape.of(2, 2, 6))
    var s = e.sum(axes=[-1]).to_cpu()
    assert_true(s.all_close(Tensor[dtype].d2([[6.0, 12.0], [18.0, 24.0]])))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[6.0], [6.0]], [[6.0], [6.0]]])))


fn test_gpu_expand_3d_middle_dim() raises:
    print("test_gpu_expand_3d_middle_dim")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(2, 5, 2)
    assert_true(e.shape() == Shape.of(2, 5, 2))
    var s = e.sum(axes=[1]).to_cpu()
    assert_true(s.all_close(Tensor[dtype].d2([[5.0, 10.0], [15.0, 20.0]])))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[5.0, 5.0]], [[5.0, 5.0]]])))


fn test_gpu_expand_3d_two_dims_broadcast() raises:
    print("test_gpu_expand_3d_two_dims_broadcast")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(3, 4, 2)
    assert_true(e.shape() == Shape.of(3, 4, 2))
    var s = e.sum(axes=[0, 1]).to_cpu()
    assert_true(s.all_close(Tensor[dtype].d1([12.0, 24.0])))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[12.0, 12.0]]])))


fn test_gpu_expand_3d_all_dims_size1() raises:
    print("test_gpu_expand_3d_all_dims_size1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[7.0]]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(2, 3, 4)
    assert_true(e.shape() == Shape.of(2, 3, 4))
    assert_true(e.to_cpu().all_close(Tensor[dtype].full(Shape.of(2, 3, 4), 7.0)))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[24.0]]])))


# ------------------------------------------------------------
# Shape API overload
# ------------------------------------------------------------

fn test_gpu_expand_shape_api_overload() raises:
    print("test_gpu_expand_shape_api_overload")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(Shape.of(3, 2))
    assert_true(e.shape() == Shape.of(3, 2))
    var expected = Tensor[dtype].d2([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    assert_true(e.to_cpu().all_close(expected))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d2([[3.0, 3.0]])))


# ------------------------------------------------------------
# Grad correctness: non-uniform values
# ------------------------------------------------------------

fn test_gpu_expand_grad_non_uniform_values() raises:
    print("test_gpu_expand_grad_non_uniform_values")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var a_unsqueezed = a.unsqueeze(0)          # (1,2,2) — no grad
    var a_gpu = a_unsqueezed.to_gpu()
    var e = a_gpu.expand(3, 2, 2)
    assert_true(e.shape() == Shape.of(3, 2, 2))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d2([[3.0, 3.0], [3.0, 3.0]])))


fn test_gpu_expand_grad_weighted_loss() raises:
    print("test_gpu_expand_grad_weighted_loss")
    comptime dtype = DType.float32
    var bias = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    var bias_gpu = bias.to_gpu()
    var e = bias_gpu.expand(3, 4, requires_grad=True)
    var weights = Tensor[dtype].d2(
        [[1.0, 2.0, 1.0, 2.0],
         [2.0, 1.0, 2.0, 1.0],
         [1.0, 1.0, 1.0, 1.0]]
    )
    var weights_gpu = weights.to_gpu()
    var loss = (e * weights_gpu).sum()
    loss.backward()
    # col0: 1+2+1=4, col1: 2+1+1=4, col2: 1+2+1=4, col3: 2+1+1=4
    assert_true(bias.grad().all_close(Tensor[dtype].d2([[4.0, 4.0, 4.0, 4.0]])))
    print("passed test_gpu_expand_grad_weighted_loss")

# ------------------------------------------------------------
# Expand then reduce — round-trip grad check
# ------------------------------------------------------------

fn test_gpu_expand_then_sum_axis_round_trip() raises:
    print("test_gpu_expand_then_sum_axis_round_trip")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(5, 3)
    var s = e.sum(axes=[0], keepdims=True)
    assert_true(s.to_cpu().all_close(Tensor[dtype].d2([[5.0, 10.0, 15.0]])))
    es = s.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d2([[5.0, 5.0, 5.0]])))


fn test_gpu_expand_then_mean_grad() raises:
    print("test_gpu_expand_then_mean_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[2.0, 4.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(4, 2)
    var m = e.mean(axes=[0])
    assert_true(m.to_cpu().all_close(Tensor[dtype].d1([2.0, 4.0])))
    ms = m.sum()
    ms.backward()
    # grad through mean (÷4) then broadcast over 4 rows = 4*(1/4) = 1.0
    assert_true(a.grad().all_close(Tensor[dtype].d2([[1.0, 1.0]])))


# ------------------------------------------------------------
# Common ML patterns
# ------------------------------------------------------------

fn test_gpu_expand_bias_broadcast_pattern() raises:
    print("test_gpu_expand_bias_broadcast_pattern")
    comptime dtype = DType.float32
    var bias = Tensor[dtype].d2([[0.5, 1.0, 1.5]], requires_grad=True)
    var bias_gpu = bias.to_gpu()
    var e = bias_gpu.expand(6, 3)
    assert_true(e.shape() == Shape.of(6, 3))
    var row_sum = e.sum(axes=[0]).to_cpu()
    assert_true(row_sum.all_close(Tensor[dtype].d1([3.0, 6.0, 9.0])))
    es = e.sum()
    es.backward()
    assert_true(bias.grad().all_close(Tensor[dtype].d2([[6.0, 6.0, 6.0]])))


fn test_gpu_expand_grad_accumulation_two_expands() raises:
    print("test_gpu_expand_grad_accumulation_two_expands")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e1 = a_gpu.expand(3, 2)
    var e2 = a_gpu.expand(5, 2)
    es1 = e1.sum()
    es1.backward()
    a_gpu.zero_grad()
    es2 = e2.sum()
    es2.backward()
    # e1 contributes 3.0, e2 contributes 5.0 → accumulated 8.0
    assert_true(a.grad().all_close(Tensor[dtype].d2([[8.0, 8.0]])))


# ------------------------------------------------------------
# track_grad=False on GPU
# ------------------------------------------------------------

fn test_gpu_expand_no_grad_tracking() raises:
    print("test_gpu_expand_no_grad_tracking")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand[track_grad=False](4, 2)
    assert_true(e.shape() == Shape.of(4, 2))
    assert_true(not e.requires_grad)


# ------------------------------------------------------------
# 4D expansions
# ------------------------------------------------------------

fn test_gpu_expand_4d_first_two_dims() raises:
    print("test_gpu_expand_4d_first_two_dims")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(1, 1, 3, 4)
    a.requires_grad_(True)
    var a_ref = a.copy()
    a_ref.requires_grad_(True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(2, 5, 3, 4)
    assert_true(e.shape() == Shape.of(2, 5, 3, 4))
    # Cross-validate forward against CPU
    var e_cpu_ref = a_ref.expand(2, 5, 3, 4)
    assert_true(e.to_cpu().all_close(e_cpu_ref.to_cpu()))
    es = e.sum()
    es.backward()
    e_cpu_ref_s = e_cpu_ref.sum()
    e_cpu_ref_s.backward()
    assert_true(a.grad().all_close(a_ref.grad()))


fn test_gpu_expand_4d_last_dim_only() raises:
    print("test_gpu_expand_4d_last_dim_only")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(2, 3, 4, 1)
    a.requires_grad_(True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(2, 3, 4, 7)
    assert_true(e.shape() == Shape.of(2, 3, 4, 7))
    es = e.sum()
    es.backward()
    assert_true(a.grad().all_close(Tensor[dtype].full(Shape.of(2, 3, 4, 1), 7.0)))


# ------------------------------------------------------------
# GPU forward matches CPU forward (cross-validate)
# ------------------------------------------------------------

fn test_gpu_expand_matches_cpu_forward() raises:
    print("test_gpu_expand_matches_cpu_forward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(1, 8)
    a.requires_grad_(True)
    var a_copy = a.copy()
    a_copy.requires_grad_(True)
    var a_gpu = a.to_gpu()
    var e_gpu = a_gpu.expand(16, 8)
    var e_cpu = a_copy.expand(16, 8)
    assert_true(e_gpu.to_cpu().all_close(e_cpu))
    e_gpu_s = e_gpu.sum()
    e_gpu_s.backward()
    e_cpu_s = e_cpu.sum()
    e_cpu_s.backward()
    assert_true(a.grad().all_close(a_copy.grad()))


fn test_gpu_expand_matches_cpu_forward_3d() raises:
    print("test_gpu_expand_matches_cpu_forward_3d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].randn(1, 4, 1)
    a.requires_grad_(True)
    var a_copy = a.copy()
    a_copy.requires_grad_(True)
    var a_gpu = a.to_gpu()
    var e_gpu = a_gpu.expand(5, 4, 6)
    var e_cpu = a_copy.expand(5, 4, 6)
    assert_true(e_gpu.to_cpu().all_close(e_cpu))
    e_gpu_s = e_gpu.sum()
    e_gpu_s.backward()
    e_cpu_s = e_cpu.sum()
    e_cpu_s.backward()

    assert_true(a.grad().all_close(a_copy.grad()))


# ------------------------------------------------------------
# Grad lands on CPU, not GPU
# ------------------------------------------------------------

fn test_gpu_expand_grad_lands_on_cpu() raises:
    print("test_gpu_expand_grad_lands_on_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(10, 2)
    es = e.sum()
    es.backward()
    assert_true(not a.grad().is_on_gpu())
    assert_true(a.grad().all_close(Tensor[dtype].d2([[10.0, 10.0]])))


fn test_gpu_expand_cpu_tensor_data_unchanged() raises:
    print("test_gpu_expand_cpu_tensor_data_unchanged")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[3.0, 6.0]], requires_grad=True)
    var a_snapshot = a.copy()
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(5, 2)
    es = e.sum()
    es.backward()
    # Original CPU values must be untouched
    assert_true(a.all_close(a_snapshot))


# ------------------------------------------------------------
# Chained expand ops on GPU
# ------------------------------------------------------------

fn test_gpu_expand_chained_two_expands() raises:
    print("test_gpu_expand_chained_two_expands")
    comptime dtype = DType.float32
    # (1,1,2) → expand to (2,3,2) → sum axis0 → (3,2) → expand to (4,3,2)
    var a = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e1 = a_gpu.expand(2, 3, 2)        # (2,3,2)
    var s = e1.sum(axes=[0], keepdims=True) # (1,3,2)
    var e2 = s.expand(4, 3, 2)             # (4,3,2)
    e2_s = e2.sum()
    e2_s.backward()
    # e1 broadcast factor: 2*3=6 via e1 path, then *4 via e2 = 24 per element
    # But sum over axis0 reduces e1's first dim → count = 2*4*3 = 24
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[24.0, 24.0]]])))


fn test_gpu_expand_then_sum_then_expand() raises:
    print("test_gpu_expand_then_sum_then_expand")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e1 = a_gpu.expand(4, 3)                    # (4,3)
    var s = e1.sum(axes=[0], keepdims=True)         # (1,3)
    var e2 = s.expand(2, 3)                         # (2,3)
    e2_s = e2.sum()
    e2_s.backward()
    # grad: expand(4) * sum(1) * expand(2) = 4*2 = 8 per element
    assert_true(a.grad().all_close(Tensor[dtype].d2([[8.0, 8.0, 8.0]])))


# ------------------------------------------------------------
# Zero-stride view property on GPU
# ------------------------------------------------------------

fn test_gpu_expand_is_zero_stride_view() raises:
    print("test_gpu_expand_is_zero_stride_view")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[10.0, 20.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var e = a_gpu.expand(100, 2)
    var row_sum = e.sum(axes=[0]).to_cpu()
    assert_true(row_sum.all_close(Tensor[dtype].d1([1000.0, 2000.0])))
    es = e.sum()
    es.backward()
    # grad = 1.0 per element * 100 rows = 100.0 per column (values irrelevant)
    assert_true(a.grad().all_close(Tensor[dtype].d2([[100.0, 100.0]])))


# ============================================================
# MAIN
# ============================================================

fn main() raises:
    @parameter
    if has_accelerator():
        # 1D → nD
        test_gpu_expand_1d_to_2d_new_batch_dim()
        test_gpu_expand_1d_to_3d()

        # 2D expansions
        test_gpu_expand_2d_row_vector_to_matrix()
        test_gpu_expand_2d_col_vector_to_matrix()
        test_gpu_expand_2d_both_dims_size1()
        test_gpu_expand_2d_no_op_same_shape()

        # 3D expansions
        test_gpu_expand_3d_first_dim()
        test_gpu_expand_3d_last_dim()
        test_gpu_expand_3d_middle_dim()
        test_gpu_expand_3d_two_dims_broadcast()
        test_gpu_expand_3d_all_dims_size1()

        # Shape API overload
        test_gpu_expand_shape_api_overload()

        # Grad correctness
        test_gpu_expand_grad_non_uniform_values()
        test_gpu_expand_grad_weighted_loss()

        # Round-trip reduce
        test_gpu_expand_then_sum_axis_round_trip()
        test_gpu_expand_then_mean_grad()

        # Common patterns
        test_gpu_expand_bias_broadcast_pattern()
        test_gpu_expand_grad_accumulation_two_expands()

        # track_grad=False
        test_gpu_expand_no_grad_tracking()

        # 4D
        test_gpu_expand_4d_first_two_dims()
        test_gpu_expand_4d_last_dim_only()

        # CPU/GPU cross-validation
        test_gpu_expand_matches_cpu_forward()
        test_gpu_expand_matches_cpu_forward_3d()

        # Device transfer backward
        test_gpu_expand_grad_lands_on_cpu()
        test_gpu_expand_cpu_tensor_data_unchanged()

        # Chained ops
        test_gpu_expand_chained_two_expands()
        test_gpu_expand_then_sum_then_expand()

        # Zero-stride view
        test_gpu_expand_is_zero_stride_view()

        print("All GPU expand tests passed.")
    else:
        print("System does not have any acclerator")
