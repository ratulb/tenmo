# =============================================================================
# Dedicated Welford tests — tests/test_welford.mojo
#
# Welford computes mean + variance in a single online pass. This suite targets
# the CPU paths directly:
#   * ReductionWalk/ReductionOdometer fallback (multi-dim odometer `advance`,
#     non-suffix axes, non-contiguous inputs) — the path that must stay correct
#     whenever those shared cursors are refactored.
#   * Classic Welford (M2 / divisor) for mean and (biased/unbiased) variance.
#   * Global scalar reductions (serial + segmented).
#   * public Tensor.variance()/std() packaging + variance backward.
# =============================================================================

from tenmo.tensor import Tensor
from tenmo.welford import Welford
from tenmo.shared.shapes import Shape
from tenmo.shared.intarray import IntArray
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator

# ── Direct Welford.forward — multi-axis (num_red>1, non-suffix) fallback ─────


def test_welford_multi_axis_mean_var_keepdims() raises:
    comptime dtype = DType.float32
    # 3D [2,2,2]; reduce axes {0,1} → out [1,1,2] (keepdims).
    # col0 group [1,3,5,7]: mean 4, biased var 5
    # col1 group [2,4,6,8]: mean 5, biased var 5
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var axes = IntArray(0, 1)
    var (mean_ndb, var_ndb) = Welford[dtype].forward(
        a.buffer, axes, unbiased=False, keepdims=True
    )
    var mean_t = Tensor[dtype](mean_ndb^, requires_grad=False)
    var var_t = Tensor[dtype](var_ndb^, requires_grad=False)
    assert_true(mean_t.shape() == Shape(1, 1, 2))
    assert_true(var_t.shape() == Shape(1, 1, 2))
    assert_true(mean_t.all_close[atol=1e-4](Tensor[dtype].d3([[[4.0, 5.0]]])))
    assert_true(var_t.all_close[atol=1e-4](Tensor[dtype].d3([[[5.0, 5.0]]])))


def test_welford_multi_axis_unbiased() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    var axes = IntArray(0, 1)
    var (mean_ndb, var_ndb) = Welford[dtype].forward(
        a.buffer, axes, unbiased=True, keepdims=True
    )
    var mean_t = Tensor[dtype](mean_ndb^, requires_grad=False)
    var var_t = Tensor[dtype](var_ndb^, requires_grad=False)
    # M2=20 per col, unbiased → 20/3
    assert_true(mean_t.all_close[atol=1e-4](Tensor[dtype].d3([[[4.0, 5.0]]])))
    assert_true(
        var_t.all_close[atol=1e-4](
            Tensor[dtype].d3([[[6.6666667, 6.6666667]]])
        )
    )


# ── Direct Welford.forward — single non-suffix axis (force walk path) ────────


def test_welford_axis0_non_suffix_mean_var() raises:
    comptime dtype = DType.float32
    # 2D [3,2]; reduce axis0 (non-suffix) → out [1,2] (keepdims).
    # col0 [1,3,5]: mean 3, biased var 8/3
    # col1 [2,4,6]: mean 4, biased var 8/3
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var axes = IntArray(0)
    var (mean_ndb, var_ndb) = Welford[dtype].forward(
        a.buffer, axes, unbiased=False, keepdims=True
    )
    var mean_t = Tensor[dtype](mean_ndb^, requires_grad=False)
    var var_t = Tensor[dtype](var_ndb^, requires_grad=False)
    assert_true(mean_t.all_close[atol=1e-4](Tensor[dtype].d2([[3.0, 4.0]])))
    assert_true(
        var_t.all_close[atol=1e-4](Tensor[dtype].d2([[2.6666667, 2.6666667]]))
    )


def test_welford_global_non_contiguous() raises:
    comptime dtype = DType.float32
    # Non-contiguous (transposed) input forces the walk/serial path, not the
    # contiguous fast path. Global reduction (out == Shape()).
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var at = a.transpose()  # shape [2,2], strides (1,2)
    var (mean_ndb, var_ndb) = Welford[dtype].forward(
        at.buffer, IntArray(0, 1), unbiased=False, keepdims=False
    )
    var mean_t = Tensor[dtype](mean_ndb^, requires_grad=False)
    var var_t = Tensor[dtype](var_ndb^, requires_grad=False)
    # elements 1,2,3,4 → mean 2.5, biased var = ((1-2.5)²+(2-2.5)²+(3-2.5)²+(4-2.5)²)/4 = 1.25
    assert_true(mean_t.all_close[atol=1e-4](Tensor[dtype].scalar(2.5)))
    assert_true(var_t.all_close[atol=1e-4](Tensor[dtype].scalar(1.25)))


# ── Public Tensor.variance()/std() — welford packaging ───────────────────────


def test_welford_var_axis1_3d_non_suffix() raises:
    comptime dtype = DType.float32
    # 3D [2,2,1]; reduce axis1 (non-suffix, contiguous) → walk path.
    var a = Tensor[dtype].d3([[[1.0], [2.0]], [[3.0], [4.0]]])
    var v = a.variance[track_grad=False](axis=1, unbiased=False)
    assert_true(v.shape() == Shape(2, 1))
    # (i=0,k=0) group [1,2] → var 0.25 ; (i=1,k=0) group [3,4] → var 0.25
    assert_true(v.all_close[atol=1e-4](Tensor[dtype].d2([[0.25], [0.25]])))


def test_welford_std_axis0_2d() raises:
    comptime dtype = DType.float32
    # axis0 of 2D contiguous is non-suffix → walk path; std = sqrt(var).
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var s = a.std[track_grad=False](axis=0, unbiased=False)
    # cols [1,3,5] var 8/3 and [2,4,6] var 8/3 → std sqrt(8/3) ≈ 1.6329932
    assert_true(
        s.all_close[atol=1e-4](
            Tensor[dtype].d1([1.6329932, 1.6329932])
        )
    )


def test_welford_global_biased_vs_unbiased() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var vb = a.variance[track_grad=False](unbiased=False)
    var vu = a.variance[track_grad=False](unbiased=True)
    # biased var = 1.25 ; unbiased var = 5/3 ≈ 1.6666667
    assert_true(vb.item() > 1.25 - 1e-4 and vb.item() < 1.25 + 1e-4)
    assert_true(vu.item() > 1.6666667 - 1e-4 and vu.item() < 1.6666667 + 1e-4)


def test_welford_constant_is_zero() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[5.0, 5.0], [5.0, 5.0]])
    var v = a.variance[track_grad=False](unbiased=False)
    assert_true(v.item() == 0.0)


def test_welford_single_element_stable() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([7.0])
    # unbiased with n=1 must use divisor 1, not 0
    var v = a.variance[track_grad=False](unbiased=True)
    assert_true(v.item() == 0.0)


# ── variance backward (gradient through Welford-saved mean) ──────────────────


def test_welford_var_bwd_axis1() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var v = a.variance(axis=1, unbiased=False)
    var loss = v.sum()
    loss.backward()
    # grad_ij = 2*(x_ij - mean_row)/n, n=3 ; rows mean 2 and 5
    var expected = Tensor[dtype].d2(
        [[2.0 * (1.0 - 2.0) / 3.0, 2.0 * (2.0 - 2.0) / 3.0, 2.0 * (3.0 - 2.0) / 3.0],
         [2.0 * (4.0 - 5.0) / 3.0, 2.0 * (5.0 - 5.0) / 3.0, 2.0 * (6.0 - 5.0) / 3.0]]
    )
    assert_true(a.grad().all_close[atol=1e-4](expected))


# ── GPU-guarded (run only when an accelerator is present) ────────────────────


def test_welford_gpu_multi_axis() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )
        var ag = a.to_gpu()
        var (mean_ndb, var_ndb) = Welford[dtype].forward(
            ag.buffer, IntArray(0, 1), unbiased=False, keepdims=True
        )
        var mean_cpu = mean_ndb.to_cpu()
        var var_cpu = var_ndb.to_cpu()
        var mean_t = Tensor[dtype](mean_cpu, requires_grad=False)
        var var_t = Tensor[dtype](var_cpu, requires_grad=False)
        assert_true(mean_t.all_close[atol=1e-3](Tensor[dtype].d3([[[4.0, 5.0]]])))
        assert_true(var_t.all_close[atol=1e-3](Tensor[dtype].d3([[[5.0, 5.0]]])))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll welford tests passed!")
