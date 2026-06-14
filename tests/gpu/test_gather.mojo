from std.testing import assert_true, assert_false, assert_equal, TestSuite
from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from tenmo.common_utils import i, s
from std.sys import has_accelerator
from tenmo.shared import Reduction

# =============================================================================
# Exhaustive tests for Tensor.gather()
# Prefix: gather_  on all test names to avoid collision with existing tests
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CPU · 1-D tensor
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CPU · 2-D tensor · axis=0 (row selection)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CPU · 2-D tensor · axis=1 (column selection)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CPU · 3-D tensor · all three axes
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — CPU · Copy path · no grad (irregular indices)
# Irregular gather copies data → grad does NOT flow back through it.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — GPU · Copy path (irregular) · no grad
# ─────────────────────────────────────────────────────────────────────────────


def test_gather_gpu_2d_axis0_irregular_copy() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],
            ]
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(1)
        idx.append(5)  # irregular → copy
        var result = a_gpu.gather(idx, axis=0)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [16.0, 17.0, 18.0]]
                )
            )
        )
        assert_false(result.requires_grad)


def test_gather_gpu_3d_axis0_irregular_copy() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
                [[13.0, 14.0], [15.0, 16.0]],
            ]
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(3)  # step=3 → actually regular! test 0,2,3 for copy
        idx = IntArray()
        idx.append(0)
        idx.append(2)
        idx.append(3)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d3(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[9.0, 10.0], [11.0, 12.0]],
                        [[13.0, 14.0], [15.0, 16.0]],
                    ]
                )
            )
        )


def test_gather_gpu_2d_axis1_irregular_copy() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        idx.append(3)  # irregular → copy
        var result = a_gpu.gather(idx, axis=1)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 3.0, 4.0], [5.0, 7.0, 8.0]])
            )
        )


# =============================================================================
# SECTION 9 — MEMCPY FAST PATH: rank==2, axis=0, unit column stride
# Prefix: mcpy_ to avoid name collision with existing gather_ tests
# =============================================================================

# ── CPU / Forward ─────────────────────────────────────────────────────────────


# ── CPU / Grad flow ───────────────────────────────────────────────────────────


# ── CPU / fuse_sum ────────────────────────────────────────────────────────────


# ── CPU / MEAN forward ──────────────────────────────────────────────────────────


# ── GPU / Forward ─────────────────────────────────────────────────────────────


def test_mcpy_gpu_2d_single_row() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(1)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d2([[4.0, 5.0, 6.0]]))
        )


def test_mcpy_gpu_2d_multi_row_reversed() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(2)
        idx.append(1)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[5.0, 6.0], [3.0, 4.0], [1.0, 2.0]])
            )
        )


def test_mcpy_gpu_2d_duplicate_indices() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            ]
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(1)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    ]
                )
            )
        )


def test_mcpy_gpu_2d_single_col() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0], [2.0], [3.0], [4.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(3)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([[4.0], [1.0]])))


def test_mcpy_gpu_2d_all_rows_identity() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(1)
        idx.append(2)
        var result = a_gpu.gather(idx, axis=0)
        assert_true(result.to_cpu().all_close(a))


# ── GPU / Grad flow ───────────────────────────────────────────────────────────


def test_mcpy_gpu_2d_no_grad_flows_to_source() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a_gpu.gather(idx, axis=0)
        var loss = result.sum()
        loss.backward()
        var grad = a.grad()
        assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
        assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))
        assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))


def test_mcpy_gpu_2d_result_requires_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0)
        assert_equal(result.requires_grad, True)


def test_mcpy_gpu_2d_grad_through_downstream_op() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var b = Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
        var b_gpu = b.to_gpu()
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(1)
        var gathered = a_gpu.gather(idx, axis=0)
        var out = gathered + b_gpu
        var loss = out.sum()
        loss.backward()
        assert_true(b.grad().all_close(Tensor[dtype].ones_like(b)))
        var grad = a.grad()
        assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
        assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
        assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))


# ── GPU / fuse_sum ────────────────────────────────────────────────────────────


def test_mcpy_gpu_2d_fuse_sum_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a_gpu.gather(idx, axis=0, reduction=Reduction(1))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([8.0, 10.0, 12.0]))
        )


def test_mcpy_gpu_2d_fuse_sum_duplicate_indices() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(0)
        idx.append(0)
        var result = a_gpu.gather(idx, axis=0, reduction=Reduction(1))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d1([3.0, 6.0])))


def test_mcpy_gpu_2d_fuse_sum_no_grad_flows() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a_gpu.gather(idx, axis=0, reduction=Reduction(1))
        var loss = result.sum()
        loss.backward()
        var grad = a.grad()
        assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))
        assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))
        assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([1.0, 1.0])))


# ── GPU / MEAN forward ──────────────────────────────────────────────────────────


def test_mcpy_gpu_2d_fuse_mean_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            )
            .to_gpu()
        )
        var idx = IntArray()
        idx.append(0)
        idx.append(1)
        var result = a.gather(idx, axis=0, reduction=Reduction(0))
        assert_true(result.shape() == Shape(3))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([2.5, 3.5, 4.5]))
        )


def test_mcpy_gpu_2d_fuse_mean_duplicate_indices() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2(
                [
                    [10.0, 20.0],
                    [30.0, 40.0],
                ]
            )
            .to_gpu()
        )
        var idx = IntArray()
        idx.append(0)
        idx.append(0)
        idx.append(1)
        var result = a.gather(idx, axis=0, reduction=Reduction(0))
        assert_true(result.shape() == Shape(2))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d1([50.0 / 3.0, 80.0 / 3.0])
            )
        )


def test_mcpy_gpu_2d_fuse_mean_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a_gpu.gather(idx, axis=0, reduction=Reduction(0))
        var loss = result.sum()
        loss.backward()
        var grad = a.grad()
        assert_true(grad[i(0), s()].all_close(Tensor[dtype].d1([0.5, 0.5])))
        assert_true(grad[i(1), s()].all_close(Tensor[dtype].d1([0.0, 0.0])))
        assert_true(grad[i(2), s()].all_close(Tensor[dtype].d1([0.5, 0.5])))


# ── CPU / MEAN / CPU–GPU parity ───────────────────────────────────────────────


def test_mcpy_cpu_gpu_parity_2d_fuse_mean() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=0, reduction=Reduction(0))
        var gpu_result = (
            a.to_gpu().gather(idx, axis=0, reduction=Reduction(0)).to_cpu()
        )
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


# ── CPU / Higher dimensions ───────────────────────────────────────────────────


# ── CPU / CPU↔GPU value parity ────────────────────────────────────────────────


def test_mcpy_cpu_gpu_parity_2d_axis0() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        )
        var idx = IntArray()
        idx.append(3)
        idx.append(1)
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=0)
        var gpu_result = a.to_gpu().gather(idx, axis=0).to_cpu()
        assert_true(cpu_result.all_close(gpu_result))


def test_mcpy_cpu_gpu_parity_2d_fuse_sum() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        var idx = IntArray()
        idx.append(0)
        idx.append(1)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=0, reduction=Reduction(1))
        var gpu_result = (
            a.to_gpu().gather(idx, axis=0, reduction=Reduction(1)).to_cpu()
        )
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


# ─── General-case reduction / 3D axis0 SUM forward ────────────


# ─── General-case reduction / 3D axis0 MEAN forward ───────────


# ─── General-case reduction / 2D axis1 SUM forward ────────────


# ─── General-case reduction / 2D axis1 MEAN forward ───────────


# ─── General-case reduction / 3D axis0 SUM backward ───────────


# ─── General-case reduction / 3D axis0 MEAN backward ──────────


# ─── General-case reduction / 2D axis1 SUM backward ───────────


# ─── General-case reduction / 2D axis1 MEAN backward ──────────


# ─── GPU: General-case reduction SUM forward ──────────────────


def test_mcpy_gpu_3d_axis0_sum_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )
        var _tmp1 = _tmp0.reshape(3, 2, 2)
        var a = _tmp1.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=0, reduction=Reduction(1))
        var _tmp2 = Tensor[dtype].d1([10, 12, 14, 16])
        var expected = _tmp2.reshape(2, 2)
        assert_true(result.to_cpu().all_close(expected))


def test_mcpy_gpu_2d_axis1_sum_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=1, reduction=Reduction(1))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d1([4.0, 10.0])))


# ─── GPU: General-case reduction MEAN forward ─────────────────


def test_mcpy_gpu_3d_axis0_mean_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )
        var _tmp1 = _tmp0.reshape(3, 2, 2)
        var a = _tmp1.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=0, reduction=Reduction(0))
        var _tmp2 = Tensor[dtype].d1([5, 6, 7, 8])
        var expected = _tmp2.reshape(2, 2)
        assert_true(result.to_cpu().all_close(expected))


def test_mcpy_gpu_2d_axis1_mean_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=1, reduction=Reduction(0))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d1([2.0, 5.0])))


# ─── GPU: General-case reduction SUM backward ─────────────────


def test_mcpy_gpu_3d_axis0_sum_general_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            requires_grad=True,
        )
        var _tmp1 = _tmp0.reshape(3, 2, 2)
        var a = _tmp1.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=0, reduction=Reduction(1))
        var loss = result.sum()
        loss.backward()
        var grad = a.grad()
        var _tmp2 = Tensor[dtype].d1(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        )
        var expected_grad = _tmp2.reshape(3, 2, 2)
        assert_true(grad.all_close[atol=1e-5](expected_grad))


def test_mcpy_gpu_2d_axis1_sum_general_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            .to_gpu()
        )
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=1, reduction=Reduction(1))
        var loss = result.sum()
        loss.backward()
        var grad = a.grad()
        assert_true(
            grad[i(0), s()].all_close(Tensor[dtype].d1([1.0, 0.0, 1.0]))
        )
        assert_true(
            grad[i(1), s()].all_close(Tensor[dtype].d1([1.0, 0.0, 1.0]))
        )


# ─── GPU: General-case reduction MEAN backward ────────────────


def test_mcpy_gpu_3d_axis0_mean_general_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            requires_grad=True,
        )
        var _tmp1 = _tmp0.reshape(3, 2, 2)
        var a = _tmp1.to_gpu()
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=0, reduction=Reduction(0))
        var loss = result.sum()
        loss.backward()
        var grad = a.grad()
        var _tmp2 = Tensor[dtype].d1(
            [0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]
        )
        var expected_grad = _tmp2.reshape(3, 2, 2)
        assert_true(grad.all_close[atol=1e-5](expected_grad))


def test_mcpy_gpu_2d_axis1_mean_general_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            .to_gpu()
        )
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var result = a.gather(idx, axis=1, reduction=Reduction(0))
        var loss = result.sum()
        loss.backward()
        var grad = a.grad()
        assert_true(
            grad[i(0), s()].all_close(Tensor[dtype].d1([0.5, 0.0, 0.5]))
        )
        assert_true(
            grad[i(1), s()].all_close(Tensor[dtype].d1([0.5, 0.0, 0.5]))
        )


# ─── CPU–GPU parity: general-case reduction ────────────────────


def test_mcpy_cpu_gpu_parity_3d_axis0_sum_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )
        var a = _tmp0.reshape(3, 2, 2)
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=0, reduction=Reduction(1))
        var gpu_result = (
            a.to_gpu().gather(idx, axis=0, reduction=Reduction(1)).to_cpu()
        )
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


def test_mcpy_cpu_gpu_parity_2d_axis1_sum_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=1, reduction=Reduction(1))
        var gpu_result = (
            a.to_gpu().gather(idx, axis=1, reduction=Reduction(1)).to_cpu()
        )
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


def test_mcpy_cpu_gpu_parity_3d_axis0_mean_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )
        var a = _tmp0.reshape(3, 2, 2)
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=0, reduction=Reduction(0))
        var gpu_result = (
            a.to_gpu().gather(idx, axis=0, reduction=Reduction(0)).to_cpu()
        )
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


def test_mcpy_cpu_gpu_parity_2d_axis1_mean_general() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var idx = IntArray()
        idx.append(0)
        idx.append(2)
        var cpu_result = a.gather(idx, axis=1, reduction=Reduction(0))
        var gpu_result = (
            a.to_gpu().gather(idx, axis=1, reduction=Reduction(0)).to_cpu()
        )
        assert_true(cpu_result.all_close[atol=1e-5](gpu_result))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll gather tests passed!")
