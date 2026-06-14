from std.testing import assert_true, TestSuite
from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from std.sys import has_accelerator
from tenmo.ndbuffer import NDBuffer
from tenmo.sum_mean_reduction import SumMeanReduction
from tenmo.intarray import IntArray
from tenmo.shapes import Shape
from tenmo.mnemonics import vm, mv
from tenmo.gradbox import Gradbox
from tenmo.kernels.matmul_kernel import MatmulNdGpu
from tenmo.mnemonics import MEAN, SUM

comptime dtype = DType.float32


# ============================================================
# GPU EXPAND TESTS — forward shape, values, and backward grad flow
# Grad always flows back to the original CPU tensor.
# ============================================================


# ------------------------------------------------------------
# Basic 1D → 2D expansion
# ------------------------------------------------------------


def test_gpu_expand_1d_to_2d_new_batch_dim() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_1d_to_2d_new_batch_dim")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(4, 3)
        assert_true(e.shape() == Shape(4, 3))
        var expected = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        )
        assert_true(e.to_cpu().all_close(expected))
        es = e.sum()
        es.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 4.0, 4.0])))


def test_gpu_expand_1d_to_3d() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_1d_to_3d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(3, 4, 2)
        assert_true(e.shape() == Shape(3, 4, 2))
        var s = e.sum(axes=[0, 1]).to_cpu()
        assert_true(s.all_close(Tensor[dtype].d1([12.0, 24.0])))
        es = e.sum()
        es.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([12.0, 12.0])))


# ------------------------------------------------------------
# 2D → 2D expansions
# ------------------------------------------------------------


def test_gpu_expand_2d_row_vector_to_matrix() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_2d_row_vector_to_matrix")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(4, 3)
        assert_true(e.shape() == Shape(4, 3))
        var expected = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
        )
        assert_true(e.to_cpu().all_close(expected))
        es = e.sum()
        es.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0, 4.0, 4.0]])))


def test_gpu_expand_2d_col_vector_to_matrix() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_2d_col_vector_to_matrix")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(3, 4)
        assert_true(e.shape() == Shape(3, 4))
        var expected = Tensor[dtype].d2(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]
        )
        assert_true(e.to_cpu().all_close(expected))
        es = e.sum()
        es.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0], [4.0], [4.0]])))


def test_gpu_expand_2d_both_dims_size1() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_2d_both_dims_size1")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[5.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(3, 4)
        assert_true(e.shape() == Shape(3, 4))
        assert_true(e.to_cpu().all_close(Tensor[dtype].full(Shape(3, 4), 5.0)))
        es = e.sum()
        es.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d2([[12.0]])))


def test_gpu_expand_2d_no_op_same_shape() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_2d_no_op_same_shape")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(2, 2)
        assert_true(e.shape() == Shape(2, 2))
        assert_true(e.to_cpu().all_close(a))
        es = e.sum()
        es.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# 3D expansions
# ------------------------------------------------------------


def test_gpu_expand_3d_first_dim() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_3d_first_dim")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(5, 2, 2)
        assert_true(e.shape() == Shape(5, 2, 2))
        var s = e.sum(axes=[0]).to_cpu()
        assert_true(s.all_close(Tensor[dtype].d2([[5.0, 10.0], [15.0, 20.0]])))
        es = e.sum()
        es.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d3([[[5.0, 5.0], [5.0, 5.0]]]))
        )


def test_gpu_expand_3d_last_dim() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_3d_last_dim")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(2, 2, 6)
        assert_true(e.shape() == Shape(2, 2, 6))
        var s = e.sum(axes=[-1]).to_cpu()
        assert_true(s.all_close(Tensor[dtype].d2([[6.0, 12.0], [18.0, 24.0]])))
        es = e.sum()
        es.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3([[[6.0], [6.0]], [[6.0], [6.0]]])
            )
        )


def test_gpu_expand_3d_middle_dim() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_3d_middle_dim")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(2, 5, 2)
        assert_true(e.shape() == Shape(2, 5, 2))
        var s = e.sum(axes=[1]).to_cpu()
        assert_true(s.all_close(Tensor[dtype].d2([[5.0, 10.0], [15.0, 20.0]])))
        es = e.sum()
        es.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d3([[[5.0, 5.0]], [[5.0, 5.0]]]))
        )


def test_gpu_expand_3d_two_dims_broadcast() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_3d_two_dims_broadcast")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(3, 4, 2)
        assert_true(e.shape() == Shape(3, 4, 2))
        var s = e.sum(axes=[0, 1]).to_cpu()
        assert_true(s.all_close(Tensor[dtype].d1([12.0, 24.0])))
        es = e.sum()
        es.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d3([[[12.0, 12.0]]])))


def test_gpu_expand_3d_all_dims_size1() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_3d_all_dims_size1")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[7.0]]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(2, 3, 4)
        assert_true(e.shape() == Shape(2, 3, 4))
        assert_true(
            e.to_cpu().all_close(Tensor[dtype].full(Shape(2, 3, 4), 7.0))
        )
        es = e.sum()
        es.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d3([[[24.0]]])))


# ------------------------------------------------------------
# Shape API overload
# ------------------------------------------------------------


def test_gpu_expand_shape_api_overload() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_shape_api_overload")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(Shape(3, 2))
        assert_true(e.shape() == Shape(3, 2))
        var expected = Tensor[dtype].d2([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        assert_true(e.to_cpu().all_close(expected))
        es = e.sum()
        es.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d2([[3.0, 3.0]])))


# ------------------------------------------------------------
# Grad correctness: non-uniform values
# ------------------------------------------------------------


def test_gpu_expand_grad_non_uniform_values() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_grad_non_uniform_values")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_unsqueezed = a.unsqueeze(0)  # (1,2,2) — no grad
        var a_gpu = a_unsqueezed.to_gpu()
        var e = a_gpu.expand(3, 2, 2)
        assert_true(e.shape() == Shape(3, 2, 2))
        es = e.sum()
        es.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[3.0, 3.0], [3.0, 3.0]]))
        )


def test_gpu_expand_grad_weighted_loss() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_grad_weighted_loss")
        comptime dtype = DType.float32
        var bias = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        var bias_gpu = bias.to_gpu()
        var e = bias_gpu.expand(3, 4, requires_grad=True)
        var weights = Tensor[dtype].d2(
            [[1.0, 2.0, 1.0, 2.0], [2.0, 1.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        )
        var weights_gpu = weights.to_gpu()
        var loss = (e * weights_gpu).sum()
        loss.backward()
        # col0: 1+2+1=4, col1: 2+1+1=4, col2: 1+2+1=4, col3: 2+1+1=4
        assert_true(
            bias.grad().all_close(Tensor[dtype].d2([[4.0, 4.0, 4.0, 4.0]]))
        )
        print("passed test_gpu_expand_grad_weighted_loss")


# ------------------------------------------------------------
# Expand then reduce — round-trip grad check
# ------------------------------------------------------------


def test_gpu_expand_then_sum_axis_round_trip() raises:
    comptime if has_accelerator():
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


def test_gpu_expand_then_mean_grad() raises:
    comptime if has_accelerator():
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


def test_gpu_expand_bias_broadcast_pattern() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_bias_broadcast_pattern")
        comptime dtype = DType.float32
        var bias = Tensor[dtype].d2([[0.5, 1.0, 1.5]], requires_grad=True)
        var bias_gpu = bias.to_gpu()
        var e = bias_gpu.expand(6, 3)
        assert_true(e.shape() == Shape(6, 3))
        var row_sum = e.sum(axes=[0]).to_cpu()
        assert_true(row_sum.all_close(Tensor[dtype].d1([3.0, 6.0, 9.0])))
        es = e.sum()
        es.backward()
        assert_true(bias.grad().all_close(Tensor[dtype].d2([[6.0, 6.0, 6.0]])))


def test_gpu_expand_grad_accumulation_two_expands() raises:
    comptime if has_accelerator():
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


def test_gpu_expand_no_grad_tracking() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_no_grad_tracking")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand[track_grad=False](4, 2)
        assert_true(e.shape() == Shape(4, 2))
        assert_true(not e.requires_grad)


# ------------------------------------------------------------
# 4D expansions
# ------------------------------------------------------------


def test_gpu_expand_4d_first_two_dims() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_4d_first_two_dims")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(1, 1, 3, 4)
        a.requires_grad_(True)
        var a_ref = a.copy()
        a_ref.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(2, 5, 3, 4)
        assert_true(e.shape() == Shape(2, 5, 3, 4))
        # Cross-validate forward against CPU
        var e_cpu_ref = a_ref.expand(2, 5, 3, 4)
        assert_true(e.to_cpu().all_close(e_cpu_ref.to_cpu()))
        es = e.sum()
        es.backward()
        e_cpu_ref_s = e_cpu_ref.sum()
        e_cpu_ref_s.backward()
        assert_true(a.grad().all_close(a_ref.grad()))


def test_gpu_expand_4d_last_dim_only() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_4d_last_dim_only")
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(2, 3, 4, 1)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(2, 3, 4, 7)
        assert_true(e.shape() == Shape(2, 3, 4, 7))
        es = e.sum()
        es.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].full(Shape(2, 3, 4, 1), 7.0))
        )


# ------------------------------------------------------------
# GPU forward matches CPU forward (cross-validate)
# ------------------------------------------------------------


def test_gpu_expand_matches_cpu_forward() raises:
    comptime if has_accelerator():
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


def test_gpu_expand_matches_cpu_forward_3d() raises:
    comptime if has_accelerator():
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


def test_gpu_expand_grad_lands_on_cpu() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_grad_lands_on_cpu")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e = a_gpu.expand(10, 2)
        es = e.sum()
        es.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor[dtype].d2([[10.0, 10.0]])))


def test_gpu_expand_cpu_tensor_data_unchanged() raises:
    comptime if has_accelerator():
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


def test_gpu_expand_chained_two_expands() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_chained_two_expands")
        comptime dtype = DType.float32
        # (1,1,2) → expand to (2,3,2) → sum axis0 → (3,2) → expand to (4,3,2)
        var a = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e1 = a_gpu.expand(2, 3, 2)  # (2,3,2)
        var s = e1.sum(axes=[0], keepdims=True)  # (1,3,2)
        var e2 = s.expand(4, 3, 2)  # (4,3,2)
        e2_s = e2.sum()
        e2_s.backward()
        # e1 broadcast factor: 2*3=6 via e1 path, then *4 via e2 = 24 per element
        # But sum over axis0 reduces e1's first dim → count = 2*4*3 = 24
        assert_true(a.grad().all_close(Tensor[dtype].d3([[[24.0, 24.0]]])))


def test_gpu_expand_then_sum_then_expand() raises:
    comptime if has_accelerator():
        print("test_gpu_expand_then_sum_then_expand")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var e1 = a_gpu.expand(4, 3)  # (4,3)
        var s = e1.sum(axes=[0], keepdims=True)  # (1,3)
        var e2 = s.expand(2, 3)  # (2,3)
        e2_s = e2.sum()
        e2_s.backward()
        # grad: expand(4) * sum(1) * expand(2) = 4*2 = 8 per element
        assert_true(a.grad().all_close(Tensor[dtype].d2([[8.0, 8.0, 8.0]])))


# ------------------------------------------------------------
# Zero-stride view property on GPU
# ------------------------------------------------------------


def test_gpu_expand_is_zero_stride_view() raises:
    comptime if has_accelerator():
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



def close_enough[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype]) raises -> Bool:
    if (a.is_on_cpu() and b.is_on_cpu()) or (a.is_on_gpu() and b.is_on_gpu()):
        return a.all_close(b)
    elif a.is_on_cpu() and b.is_on_gpu():
        return a.all_close(b.to_cpu())
    else:
        return a.to_cpu().all_close(b)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ===============================================================================


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor.sum — CPU
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor.mean — CPU
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# NDBuffer.reduce (CPU) — sum and mean via reduce[mean=False/True]
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# GPU tests — Tensor.sum and Tensor.mean
# ═══════════════════════════════════════════════════════════════════════════════


def test_v2_gpu_tensor_sum_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1, 2, 3, 4, 5])
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum()
        var gpu_result = a_gpu.sum()
        assert_true(cpu_result.to_gpu().all_close(gpu_result))


def test_v2_gpu_tensor_sum_2d_axis0() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[0])
        var gpu_result = a_gpu.sum(axes=[0])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))


def test_v2_gpu_tensor_sum_2d_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[1])
        var gpu_result = a_gpu.sum(axes=[1])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))


def test_v2_gpu_tensor_sum_3d_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(24)
        a = a.reshape(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[1])
        var gpu_result = a_gpu.sum(axes=[1])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))


def test_v2_gpu_tensor_sum_3d_axes_0_2() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(24)
        a = a.reshape(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[0, 2])
        var gpu_result = a_gpu.sum(axes=[0, 2])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))


def test_v2_gpu_tensor_sum_keepdims() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(24)
        a = a.reshape(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum(axes=[1], keepdims=True)
        var gpu_result = a_gpu.sum(axes=[1], keepdims=True)
        assert_true(cpu_result.to_gpu().all_close(gpu_result))


def test_v2_gpu_tensor_sum_all_axes() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].ones(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.sum()
        var gpu_result = a_gpu.sum()
        assert_true(cpu_result.to_gpu().all_close(gpu_result))


def test_v2_gpu_tensor_mean_2d_axis0() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [3, 4, 5]])
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean(axes=[0])
        var gpu_result = a_gpu.mean(axes=[0])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))


def test_v2_gpu_tensor_mean_2d_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean(axes=[1])
        var gpu_result = a_gpu.mean(axes=[1])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))


def test_v2_gpu_tensor_mean_3d_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(24)
        a = a.reshape(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean(axes=[1])
        var gpu_result = a_gpu.mean(axes=[1])
        assert_true(cpu_result.to_gpu().all_close(gpu_result))


def test_v2_gpu_tensor_mean_keepdims() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(24)
        a = a.reshape(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean(axes=[1], keepdims=True)
        var gpu_result = a_gpu.mean(axes=[1], keepdims=True)
        assert_true(cpu_result.to_gpu().all_close(gpu_result))


def test_v2_gpu_tensor_mean_all_axes() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].arange(24)
        a = a.reshape(2, 3, 4)
        var a_gpu = a.to_gpu()
        var cpu_result = a.mean()
        var gpu_result = a_gpu.mean()
        assert_true(cpu_result.to_gpu().all_close(gpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# GPU tests — NDBuffer.reduce
# ═══════════════════════════════════════════════════════════════════════════════


def test_v2_gpu_ndbuffer_sum_2d_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var a_gpu = a.to_gpu()
        var gpu_ndb = a_gpu.buffer
        var gpu_result = SumMeanReduction[dtype].reduce[op_code=SUM](
            gpu_ndb, IntArray(1), keepdims=False
        )
        var tensor = Tensor[dtype].d1([6, 15])
        var g_tensor = tensor.to_gpu()
        assert_true(g_tensor.buffer.all_close(gpu_result))


def test_v2_gpu_ndbuffer_mean_2d_axis0() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1, 2, 3], [3, 4, 5]])
        var a_gpu = a.to_gpu()
        var gpu_ndb = a_gpu.buffer
        var gpu_result = SumMeanReduction[dtype].reduce[op_code=MEAN](
            gpu_ndb, IntArray(0), keepdims=False
        )
        var expected = Tensor[dtype].d1([2, 3, 4])
        expected = expected.to_gpu()
        assert_true(expected.buffer.all_close(gpu_result))


def test_gpu_grad_flow() raises:
    comptime if has_accelerator():
        var A = Tensor[dtype].arange(9 * 30, requires_grad=True)
        var B = Tensor[dtype].arange(30 * 5)

        var A_reshaped = A.reshape(Shape(1, 9, 30))
        var B_reshaped = B.reshape(Shape(30, 5))

        var A_gpu = A_reshaped.to_gpu()
        var A_r = A_gpu.reshape(3, 3, 1, 30)
        var A_rr = A_r.reshape(3, 2, 3, 1, 15)
        var B_gpu = B_reshaped.to_gpu()
        var A_gpu_reshaped = A_rr.reshape(Shape(9, 1, 30))
        var C_gpu = A_gpu_reshaped.matmul(B_gpu)

        C_gpu.backward()
        var A_grad = A.grad().copy()

        var grad_out = Gradbox[dtype].full(C_gpu.shape(), 1)
        # ===== GRADIENT FOR A: dL/dA = grad_out × B^T =====
        var A_grad_expected = grad_out.matmul(B_reshaped.transpose(-1, -2))
        A_grad_expected = A_grad_expected.reshape(Shape(9 * 30))

        assert_true(A_grad.all_close(A_grad_expected))


def test_vmnd_1d_v_2d_M() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # v = [1, 2, 3],  M = [[1,0],[0,1],[1,1]]
        # out = [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
        var v = Tensor[dtype].d1([1, 2, 3])
        var M = Tensor[dtype].d2([[1, 0], [0, 1], [1, 1]])
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_vmnd_identity_matrix() raises:
    """V @ I = v."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].d1([3, 1, 4, 1, 5])
        var I = Tensor[dtype].eye(5)
        var cpu_result = v.matmul[mode=vm](I)
        var v_gpu = v.to_gpu()
        var I_gpu = I.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](I_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_vmnd_zero_vector() raises:
    """Zero vector gives zero output."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].zeros(4)
        var M = Tensor[dtype].arange(12)
        M = M.reshape(4, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_vmnd_ones_vector() raises:
    """Ones vector sums columns of M."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(3)
        var M = Tensor[dtype].d2([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        # out = [1+5+9, 2+6+10, 3+7+11, 4+8+12] = [15, 18, 21, 24]
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_vmnd_single_output_element() raises:
    """N=1: output is a scalar-like vector."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].d1([2, 3, 4])
        var M = Tensor[dtype].d2([[1], [2], [3]])
        # out = [2*1 + 3*2 + 4*3] = [20]
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_vmnd_large_k() raises:
    """Large k to stress the dot product loop."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var k = 512
        var n = 64
        var v = Tensor[dtype].ones(k)
        var M = Tensor[dtype].ones(k, n)
        # out[j] = k for all j
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_vmnd_large_n() raises:
    """N > block_size to exercise multi-block coverage."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var k = 8
        var n = 1024  # larger than default block_size=256
        var v = Tensor[dtype].ones(k)
        var M = Tensor[dtype].ones(k, n)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# Batched — v and M same batch shape, no broadcast
# ═══════════════════════════════════════════════════════════════════════════════


def test_vmnd_batched_2d_v_3d_M() raises:
    """V[b, k] @ M[b, k, n] → out[b, n]."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].arange(6)
        v = v.reshape(2, 3)  # (2, 3)
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)  # (2, 3, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        # result_cpu = gpu_result.to_cpu()
        assert_true(close_enough(cpu_result, gpu_result))


def test_vmnd_batched_3d_v_4d_M() raises:
    """V[a, b, k] @ M[a, b, k, n] → out[a, b, n]."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(2, 3, 4)  # (2, 3, 4)
        var M = Tensor[dtype].ones(2, 3, 4, 5)  # (2, 3, 4, 5)
        # out[a,b,j] = 4.0 for all (a,b,j)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_vmnd_batched_arange_values() raises:
    """Batched with non-trivial values to catch index mapping errors."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # v: (3, 4),  M: (3, 4, 2)
        var v = Tensor[dtype].arange(12)
        v = v.reshape(3, 4)
        var M = Tensor[dtype].arange(24)
        M = M.reshape(3, 4, 2)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# Broadcast — v and M have different batch ranks
# ═══════════════════════════════════════════════════════════════════════════════


def test_vmnd_broadcast_v1d_M3d() raises:
    """V[k] broadcast against M[b, k, n] → out[b, n]."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].d1([1, 0, 1])  # (3,)
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)  # (2, 3, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_vmnd_broadcast_v2d_M3d() raises:
    """V[1, k] broadcast against M[b, k, n] → out[b, n]."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(1, 4)  # (1, 4)
        var M = Tensor[dtype].arange(48)
        M = M.reshape(3, 4, 4)  # (3, 4, 4)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_vmnd_broadcast_v3d_M2d() raises:
    """V[a, b, k] broadcast against M[k, n] → out[a, b, n]."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].arange(24)
        v = v.reshape(2, 3, 4)  # (2, 3, 4)
        var M = Tensor[dtype].arange(12)
        M = M.reshape(4, 3)  # (4, 3)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_vmnd_broadcast_both_size1() raises:
    """Both v and M have a size-1 batch dim that broadcasts."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var v = Tensor[dtype].ones(1, 4)  # (1, 4) → broadcasts to (3, 4)
        var M = Tensor[dtype].ones(3, 4, 5)  # (3, 4, 5)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_vmnd_broadcast_large_batch() raises:
    """Large broadcast batch to stress multi-block output coverage."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # v: (32, k), M: (k, n) — M broadcast across 32 batch elements
        var k = 64
        var n = 128
        var v = Tensor[dtype].ones(32, k)
        var M = Tensor[dtype].ones(k, n)
        var cpu_result = v.matmul[mode=vm](M)
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# Numerical correctness — spot-check known values
# ═══════════════════════════════════════════════════════════════════════════════


def test_vmnd_known_values_no_batch() raises:
    """Hand-computed result verified against GPU."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # v = [1, 2],  M = [[3, 4, 5], [6, 7, 8]]
        # out = [1*3+2*6, 1*4+2*7, 1*5+2*8] = [15, 18, 21]
        var v = Tensor[dtype].d1([1, 2])
        var M = Tensor[dtype].d2([[3, 4, 5], [6, 7, 8]])
        var expected = Tensor[dtype].d1([15, 18, 21])
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(expected.to_gpu().all_close(gpu_result))


def test_vmnd_known_values_batched() raises:
    """Hand-computed batched result verified against GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # batch 0: v=[1,0] @ M=[[1,2],[3,4]] = [1, 2]
        # batch 1: v=[0,1] @ M=[[5,6],[7,8]] = [7, 8]
        var v = Tensor[dtype].d2([[1, 0], [0, 1]])
        var M = Tensor[dtype].d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        var expected = Tensor[dtype].d2([[1, 2], [7, 8]])
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(expected.to_gpu().all_close(gpu_result))


def test_vmnd_known_values_broadcast() raises:
    """Hand-computed broadcast result verified against GPU."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # v = [1, 1]  (no batch)
        # M[0] = [[1,2],[3,4]] → out[0] = [4, 6]
        # M[1] = [[5,6],[7,8]] → out[1] = [12, 14]
        var v = Tensor[dtype].d1([1, 1])
        var M = Tensor[dtype].d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        var expected = Tensor[dtype].d2([[4, 6], [12, 14]])
        var v_gpu = v.to_gpu()
        var M_gpu = M.to_gpu()
        var gpu_result = v_gpu.matmul[mode=vm](M_gpu)
        assert_true(expected.to_gpu().all_close(gpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# Basic correctness — no batch dims
# ═══════════════════════════════════════════════════════════════════════════════


def test_mvnd_2d_M_1d_v() raises:
    """M[m, k] @ v[k] → out[m]. Simplest case, no batch dims."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # M = [[1,2,3],[4,5,6]], v = [1,1,1]
        # out = [1+2+3, 4+5+6] = [6, 15]
        var M = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
        var v = Tensor[dtype].d1([1, 1, 1])
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mvnd_known_values() raises:
    """Hand-computed result verified directly against GPU output."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # M = [[1,0],[0,1],[1,1]], v = [3, 4]
        # out = [1*3+0*4, 0*3+1*4, 1*3+1*4] = [3, 4, 7]
        var M = Tensor[dtype].d2([[1, 0], [0, 1], [1, 1]])
        var v = Tensor[dtype].d1([3, 4])
        var expected = Tensor[dtype].d1([3, 4, 7])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))


def test_mvnd_identity_matrix() raises:
    """I @ v = v."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].eye(4)
        var v = Tensor[dtype].d1([2, 5, 1, 8])
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mvnd_zero_vector() raises:
    """M @ zero_vector = zero output."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(12)
        M = M.reshape(3, 4)
        var v = Tensor[dtype].zeros(4)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mvnd_ones_vector() raises:
    """M @ ones = row sums of M."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # row sums: [1+2+3+4, 5+6+7+8, 9+10+11+12] = [10, 26, 42]
        var M = Tensor[dtype].d2([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        var v = Tensor[dtype].ones(4)
        var expected = Tensor[dtype].d1([10, 26, 42])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))


def test_mvnd_single_row_matrix() raises:
    """M[1, k] @ v[k] → out[1]. Single row edge case."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].d2([[2, 3, 4]])  # (1, 3)
        var v = Tensor[dtype].d1([1, 2, 3])
        # out = [2*1 + 3*2 + 4*3] = [20]
        var expected = Tensor[dtype].d1([20])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))


def test_mvnd_single_col_matrix() raises:
    """M[m, 1] @ v[1] → out[m]. k=1 edge case."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].d2([[2], [3], [5]])  # (3, 1)
        var v = Tensor[dtype].d1([4])
        # out = [8, 12, 20]
        var expected = Tensor[dtype].d1([8, 12, 20])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))


def test_mvnd_large_k() raises:
    """Large k to stress the dot product loop."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var k = 512
        var m = 32
        var M = Tensor[dtype].ones(m, k)
        var v = Tensor[dtype].ones(k)
        # each output element = k = 512
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mvnd_large_m() raises:
    """M > block_size to exercise multi-block coverage."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var m = 1024  # larger than default block_size=256
        var k = 8
        var M = Tensor[dtype].ones(m, k)
        var v = Tensor[dtype].ones(k)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mvnd_negative_values() raises:
    """Negative values in both M and v."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].d2([[-1, 2], [3, -4]])
        var v = Tensor[dtype].d1([-1, 2])
        # out = [(-1*-1 + 2*2), (3*-1 + -4*2)] = [5, -11]
        var expected = Tensor[dtype].d1([5, -11])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# Batched — M and v same batch shape, no broadcast
# ═══════════════════════════════════════════════════════════════════════════════


def test_mvnd_batched_3d_M_2d_v() raises:
    """M[b, m, k] @ v[b, k] → out[b, m]."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)  # (2, 3, 3)
        var v = Tensor[dtype].arange(6)
        v = v.reshape(2, 3)  # (2, 3)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mvnd_batched_4d_M_3d_v() raises:
    """M[a, b, m, k] @ v[a, b, k] → out[a, b, m]."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].ones(2, 3, 4, 5)  # (2, 3, 4, 5)
        var v = Tensor[dtype].ones(2, 3, 5)  # (2, 3, 5)
        # each output element = k = 5
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mvnd_batched_arange_values() raises:
    """Batched with non-trivial arange values to catch index mapping errors."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # M: (3, 4, 2),  v: (3, 2)
        var M = Tensor[dtype].arange(24)
        M = M.reshape(3, 4, 2)
        var v = Tensor[dtype].arange(6)
        v = v.reshape(3, 2)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mvnd_known_values_batched() raises:
    """Hand-computed batched result verified directly against GPU."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # batch 0: [[1,0],[0,1]] @ [1,2] = [1, 2]
        # batch 1: [[2,0],[0,2]] @ [3,4] = [6, 8]
        var M = Tensor[dtype].d3([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
        var v = Tensor[dtype].d2([[1, 2], [3, 4]])
        var expected = Tensor[dtype].d2([[1, 2], [6, 8]])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# Broadcast — M and v have different batch ranks
# ═══════════════════════════════════════════════════════════════════════════════


def test_mvnd_broadcast_3d_M_1d_v() raises:
    """M[b, m, k] broadcast against v[k] → out[b, m]."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(18)
        M = M.reshape(2, 3, 3)  # (2, 3, 3)
        var v = Tensor[dtype].d1([1, 0, 1])  # (3,) broadcasts over batch
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mvnd_broadcast_2d_M_3d_v() raises:
    """M[m, k] broadcast against v[b, k] → out[b, m]."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].arange(12)
        M = M.reshape(4, 3)  # (4, 3) — no batch
        var v = Tensor[dtype].arange(9)
        v = v.reshape(3, 3)  # (3, 3) — batch of 3
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mvnd_broadcast_4d_M_2d_v() raises:
    """M[a, b, m, k] broadcast against v[b, k] → out[a, b, m]."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].ones(2, 3, 4, 5)  # (2, 3, 4, 5)
        var v = Tensor[dtype].ones(3, 5)  # (3, 5) — broadcasts over dim 0
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mvnd_broadcast_size1_batch() raises:
    """V with size-1 batch dim that broadcasts across M's batch."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var M = Tensor[dtype].ones(4, 3, 5)  # (4, 3, 5)
        var v = Tensor[dtype].ones(1, 5)  # (1, 5) → broadcasts to (4, 5)
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mvnd_broadcast_known_values() raises:
    """Hand-computed broadcast result verified directly against GPU."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # M[0] = [[1,2],[3,4]], M[1] = [[5,6],[7,8]]
        # v = [1, 1]  (no batch — broadcasts across both)
        # out[0] = [3, 7],  out[1] = [11, 15]
        var M = Tensor[dtype].d3([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        var v = Tensor[dtype].d1([1, 1])
        var expected = Tensor[dtype].d2([[3, 7], [11, 15]])
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(expected, gpu_result))


def test_mvnd_broadcast_large_batch() raises:
    """Large broadcast batch to stress multi-block output coverage."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var m = 64
        var k = 32
        var M = Tensor[dtype].ones(128, m, k)  # large batch on M side
        var v = Tensor[dtype].ones(k)  # no batch — broadcasts
        var cpu_result = M.matmul[mode=mv](v)
        var M_gpu = M.to_gpu()
        var v_gpu = v.to_gpu()
        var gpu_result = M_gpu.matmul[mode=mv](v_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# Basic correctness — no batch dims
# ═══════════════════════════════════════════════════════════════════════════════


def test_mmnd_2d_known_values() raises:
    """Hand-computed 2D matmul verified directly against GPU."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        # C[0,0]=1*5+2*7=19, C[0,1]=1*6+2*8=22
        # C[1,0]=3*5+4*7=43, C[1,1]=3*6+4*8=50
        var A = Tensor[dtype].d2([[1, 2], [3, 4]])
        var B = Tensor[dtype].d2([[5, 6], [7, 8]])
        var expected = Tensor[dtype].d2([[19, 22], [43, 50]])
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(expected, gpu_result))


def test_mmnd_2d_identity() raises:
    """A @ I = A."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(9)
        A = A.reshape(3, 3)
        var I = Tensor[dtype].eye(3)
        var cpu_result = A.matmul(I)
        var a_gpu = A.to_gpu()
        var i_gpu = I.to_gpu()
        var gpu_result = a_gpu.matmul(i_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_2d_zero_matrix() raises:
    """A @ zeros = zeros."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(12)
        A = A.reshape(3, 4)
        var B = Tensor[dtype].zeros(4, 5)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_2d_ones() raises:
    """Ones @ ones = matrix of k."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var k = 8
        var A = Tensor[dtype].ones(4, k)
        var B = Tensor[dtype].ones(k, 5)
        # every output element = k
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_2d_rectangular() raises:
    """Non-square matrices: (m, k) @ (k, n) where m != k != n."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(15)
        A = A.reshape(3, 5)  # (3, 5)
        var B = Tensor[dtype].arange(20)
        B = B.reshape(5, 4)  # (5, 4)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_2d_negative_values() raises:
    """Negative values in both A and B."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d2([[-1, 2], [3, -4]])
        var B = Tensor[dtype].d2([[1, -2], [-3, 4]])
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_2d_single_element() raises:
    """(1, k) @ (k, 1) → (1, 1): inner product as matmul."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d2([[1, 2, 3, 4]])  # (1, 4)
        var B = Tensor[dtype].d2([[1], [2], [3], [4]])  # (4, 1)
        # result = [[1+4+9+16]] = [[30]]
        var expected = Tensor[dtype].d2([[30]])
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(expected, gpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# Tile boundary stress — sizes not multiples of TILE_SIZE
# ═══════════════════════════════════════════════════════════════════════════════


def test_mmnd_2d_non_tile_multiple_m() raises:
    """M(m) not a multiple of TILE_SIZE."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(17, 16)  # m=17, not multiple of 16
        var B = Tensor[dtype].ones(16, 16)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_2d_non_tile_multiple_n() raises:
    """N(n) not a multiple of TILE_SIZE."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(16, 16)
        var B = Tensor[dtype].ones(16, 19)  # n=19, not multiple of 16
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_2d_non_tile_multiple_k() raises:
    """K(k) not a multiple of TILE_SIZE."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(16, 13)  # k=13, not multiple of 16
        var B = Tensor[dtype].ones(13, 16)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_2d_all_non_tile_multiples() raises:
    """M(m), k, n all non-multiples of TILE_SIZE simultaneously."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(17 * 13)
        A = A.reshape(17, 13)
        var B = Tensor[dtype].arange(13 * 19)
        B = B.reshape(13, 19)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_2d_smaller_than_tile() raises:
    """M(m), k, n all smaller than TILE_SIZE."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(12)
        A = A.reshape(3, 4)  # both < 16
        var B = Tensor[dtype].arange(8)
        B = B.reshape(4, 2)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# Large matrices — stress multi-block coverage
# ═══════════════════════════════════════════════════════════════════════════════


def test_mmnd_2d_large_square() raises:
    """Large square matrices well beyond tile size."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(128, 128)
        var B = Tensor[dtype].ones(128, 128)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_2d_large_rectangular() raises:
    """Large rectangular matrices with non-tile-multiple dimensions."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(65, 100)
        var B = Tensor[dtype].ones(100, 70)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# Batched — same batch shape, no broadcast
# ═══════════════════════════════════════════════════════════════════════════════


def test_mmnd_batched_3d_known_values() raises:
    """Hand-computed batched matmul verified directly against GPU."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # batch 0: [[1,0],[0,1]] @ [[2,3],[4,5]] = [[2,3],[4,5]]
        # batch 1: [[1,1],[1,1]] @ [[1,0],[0,1]] = [[1,1],[1,1]]
        var A = Tensor[dtype].d3([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])
        var B = Tensor[dtype].d3([[[2, 3], [4, 5]], [[1, 0], [0, 1]]])
        var expected = Tensor[dtype].d3([[[2, 3], [4, 5]], [[1, 1], [1, 1]]])
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(expected, gpu_result))


def test_mmnd_batched_3d_arange() raises:
    """A[b, m, k] @ B[b, k, n] with arange values."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(24)
        A = A.reshape(2, 3, 4)  # (2, 3, 4)
        var B = Tensor[dtype].arange(24)
        B = B.reshape(2, 4, 3)  # (2, 4, 3)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_batched_4d() raises:
    """A[a, b, m, k] @ B[a, b, k, n] — 4D batch."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(2, 3, 4, 5)  # (2, 3, 4, 5)
        var B = Tensor[dtype].ones(2, 3, 5, 4)  # (2, 3, 5, 4)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_batched_large_batch() raises:
    """Many batch elements to stress grid.z coverage."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(32, 16, 8)  # 32 batch elements
        var B = Tensor[dtype].ones(32, 8, 16)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_batched_non_tile_multiples() raises:
    """Batched with m, k, n not multiples of TILE_SIZE."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(2 * 11 * 7)
        A = A.reshape(2, 11, 7)
        var B = Tensor[dtype].arange(2 * 7 * 13)
        B = B.reshape(2, 7, 13)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


# ═══════════════════════════════════════════════════════════════════════════════
# Broadcast — different batch ranks
# ═══════════════════════════════════════════════════════════════════════════════


def test_mmnd_broadcast_3d_A_2d_B() raises:
    """A[b, m, k] @ B[k, n] — B broadcasts across batch."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(24)
        A = A.reshape(2, 3, 4)  # (2, 3, 4)
        var B = Tensor[dtype].arange(12)
        B = B.reshape(4, 3)  # (4, 3) — no batch, broadcasts
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_broadcast_2d_A_3d_B() raises:
    """A[m, k] @ B[b, k, n] — A broadcasts across batch."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].arange(12)
        A = A.reshape(3, 4)  # (3, 4) — no batch, broadcasts
        var B = Tensor[dtype].arange(24)
        B = B.reshape(2, 4, 3)  # (2, 4, 3)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_broadcast_4d_A_3d_B() raises:
    """A[a, b, m, k] @ B[b, k, n] — B missing leading batch dim."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(2, 3, 4, 5)  # (2, 3, 4, 5)
        var B = Tensor[dtype].ones(3, 5, 4)  # (3, 5, 4) — broadcasts over dim 0
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_broadcast_size1_batch_dim() raises:
    """Size-1 batch dim in A broadcasts across B's batch."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(
            1, 4, 5
        )  # (1, 4, 5) → broadcasts to (3, 4, 5)
        var B = Tensor[dtype].ones(3, 5, 4)  # (3, 5, 4)
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_mmnd_broadcast_known_values() raises:
    """Hand-computed broadcast result verified directly against GPU."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        # A = [[1,0],[0,1]] (identity, no batch)
        # B[0] = [[2,3],[4,5]], B[1] = [[6,7],[8,9]]
        # out[0] = I @ B[0] = [[2,3],[4,5]]
        # out[1] = I @ B[1] = [[6,7],[8,9]]
        var A = Tensor[dtype].d2([[1, 0], [0, 1]])
        var B = Tensor[dtype].d3([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
        var expected = Tensor[dtype].d3([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(expected, gpu_result))


def test_mmnd_broadcast_large() raises:
    """Large broadcast batch to stress multi-block and multi-z coverage."""

    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].ones(16, 32, 32)  # (16, 32, 32)
        var B = Tensor[dtype].ones(32, 32)  # (32, 32) — broadcasts over 16
        var cpu_result = A.matmul(B)
        var a_gpu = A.to_gpu()
        var b_gpu = B.to_gpu()
        var gpu_result = a_gpu.matmul(b_gpu)
        assert_true(close_enough(cpu_result, gpu_result))


def test_gpu_transfer_fidelity() raises:
    comptime if has_accelerator():
        var B = Tensor[dtype].rand(80, 20)
        var B_gpu = B.to_gpu()
        var B_back = B_gpu.to_cpu()
        assert_true(B.all_close(B_back))


def test_forward_matmul_fidelity() raises:
    comptime if has_accelerator():
        var A = Tensor[dtype].rand(9, 80, requires_grad=True)
        var B = Tensor[dtype].rand(80, 20)
        var A_gpu = A.to_gpu()
        var B_gpu = B.to_gpu()
        var C_cpu = A.matmul(B)
        var C_gpu = A_gpu.matmul(B_gpu)
        assert_true(C_cpu.all_close(C_gpu.to_cpu()))


def test_backward_grad_A_fidelity() raises:
    comptime if has_accelerator():
        var AA = Tensor[dtype].arange(9 * 30, requires_grad=True)
        var A = AA.reshape(9, 30)
        var _tmp0 = Tensor[dtype].arange(30 * 5)
        var B = _tmp0.reshape(30, 5)

        var A_gpu = A.to_gpu()
        var B_gpu = B.to_gpu()
        var C_cpu = A.matmul(B)
        var C_gpu = A_gpu.matmul(B_gpu)

        C_cpu.backward()

        assert_true(
            A_gpu.grad().to_cpu().all_close(Tensor[dtype].zeros(Shape(9, 30)))
        )

        C_gpu.backward()

        var _tmp1 = AA.grad().to_gpu()
        var _tmp2 = _tmp1.reshape(Shape(9, 30))
        assert_true(_tmp2.all_close(A_gpu.grad() * 2))


def test_transposed_matmul_fidelity() raises:
    comptime if has_accelerator():
        var B = Tensor[dtype].rand(80, 20)
        var B_gpu = B.to_gpu()
        var BT_cpu = B.transpose(axes=IntArray(-1, -2))
        var BT_gpu = B_gpu.buffer.transpose(axes=IntArray(-1, -2))

        var grad_out = Tensor[dtype].ones(9, 20)
        var grad_out_gpu = grad_out.to_gpu()

        var grad_A_cpu = grad_out.matmul(BT_cpu)

        var grad_A_ndb = MatmulNdGpu[dtype].launch[tile_size=32](
            grad_out_gpu.buffer, BT_gpu
        )
        var grad_A_GPU = Tensor[dtype](grad_A_ndb^)
        var grad_A_gpu = grad_A_GPU.to_cpu()

        assert_true(grad_A_cpu.all_close(grad_A_gpu))



