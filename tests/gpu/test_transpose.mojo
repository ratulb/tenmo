from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator

from tenmo.gradbox import Gradbox
from tenmo.ndbuffer import NDBuffer
from tenmo.common_utils import i, s
from tenmo.device import GPU
from tenmo.common_utils import now
from tenmo.intarray import IntArray

comptime dtype = DType.float32




# Exhaustive tests for the revamped Transpose implementation.
# Prefix: trrev_ (transpose revamp)
# Covers:
#  - Forward correctness (1D, 2D, 3D, 4D)
#  - Backward / grad flow
#  - Chained transposes
#  - Transpose + contiguous
#  - CPU and GPU variants

# ──────────────────────────────────────────────────────────────────────────────
# CPU TESTS
# ──────────────────────────────────────────────────────────────────────────────

# ── 1D ────────────────────────────────────────────────────────────────────────


# ── 2D ────────────────────────────────────────────────────────────────────────


# ── 3D ────────────────────────────────────────────────────────────────────────


# ── 4D ────────────────────────────────────────────────────────────────────────


# ── Grad flow / accumulation ───────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# GPU TESTS
# ──────────────────────────────────────────────────────────────────────────────

# ── 1D ────────────────────────────────────────────────────────────────────────


def test_trrev_gpu_1d_forward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_1d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        assert_true(t.shape() == Shape(3))
        assert_true(t.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


def test_trrev_gpu_1d_backward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_1d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ── 2D ────────────────────────────────────────────────────────────────────────


def test_trrev_gpu_2d_forward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_2d_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        assert_true(t.shape() == Shape(3, 2))
        assert_true(
            t.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
            )
        )


def test_trrev_gpu_2d_backward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_2d_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_trrev_gpu_2d_explicit_axes_forward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_2d_explicit_axes_forward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(1, 0)
        assert_true(t.shape() == Shape(2, 3))
        assert_true(
            t.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
            )
        )


def test_trrev_gpu_2d_explicit_axes_backward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_2d_explicit_axes_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(1, 0)
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_trrev_gpu_2d_transpose_then_contiguous() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_2d_transpose_then_contiguous")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var c = t.contiguous()
        assert_true(c.shape() == Shape(3, 2))
        assert_true(
            c.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
            )
        )
        var loss = c.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_trrev_gpu_2d_double_transpose_identity() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_2d_double_transpose_identity")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var t1 = a_gpu.transpose()
        var t = t1.transpose()
        assert_true(t.shape() == Shape(2, 3))
        assert_true(t.to_cpu().all_close(a))


def test_trrev_gpu_2d_double_transpose_backward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_2d_double_transpose_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var t1 = a_gpu.transpose()
        var t2 = t1.transpose()
        var loss = t2.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_trrev_gpu_2d_grad_correctness_weighted() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_2d_grad_correctness_weighted")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var prod = t * w
        var loss = prod.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        # grad_a[i,j] = w[j,i]  →  [[1,3],[2,4]]
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]]))
        )


# ── 3D ────────────────────────────────────────────────────────────────────────


def test_trrev_gpu_3d_default_axes_forward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_3d_default_axes_forward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var a = _tmp0.reshape(Shape(2, 3, 4))
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        assert_true(t.shape() == Shape(4, 3, 2))
        var t_cpu = t.to_cpu()
        assert_true(t_cpu[0, 0, 0] == 0.0)
        assert_true(t_cpu[3, 2, 1] == 23.0)
        assert_true(t_cpu[2, 1, 0] == 6.0)


def test_trrev_gpu_3d_default_axes_backward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_3d_default_axes_backward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var a = _tmp0.reshape(Shape(2, 3, 4))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_trrev_gpu_3d_explicit_axes_forward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_3d_explicit_axes_forward")
        comptime dtype = DType.float32
        # (2,3,4) → axes (0,2,1) → (2,4,3)
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var a = _tmp0.reshape(Shape(2, 3, 4))
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(0, 2, 1)
        assert_true(t.shape() == Shape(2, 4, 3))
        var t_cpu = t.to_cpu()
        # a[1,2,3]=23 → t[1,3,2]=23
        assert_true(t_cpu[1, 3, 2] == 23.0)
        # a[0,1,2]=6 → t[0,2,1]=6
        assert_true(t_cpu[0, 2, 1] == 6.0)


def test_trrev_gpu_3d_explicit_axes_backward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_3d_explicit_axes_backward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var a = _tmp0.reshape(Shape(2, 3, 4))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(0, 2, 1)
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_trrev_gpu_3d_transpose_then_contiguous() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_3d_transpose_then_contiguous")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var a = _tmp0.reshape(Shape(2, 3, 4))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()  # (4,3,2)
        var c = t.contiguous()
        assert_true(c.shape() == Shape(4, 3, 2))
        # Spot-check values match the CPU transpose
        var ref_t = a.transpose()
        assert_true(c.to_cpu().all_close(ref_t))
        var loss = c.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_trrev_gpu_3d_chained_transpose_backward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_3d_chained_transpose_backward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(1.0, 25.0)
        var a = _tmp0.reshape(Shape(2, 3, 4))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t1 = a_gpu.transpose()  # (4,3,2)
        var t2 = t1.transpose()  # back to (2,3,4)
        var loss = t2.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ── 4D ────────────────────────────────────────────────────────────────────────


def test_trrev_gpu_4d_default_axes_forward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_4d_default_axes_forward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 120.0)
        var a = _tmp0.reshape(Shape(2, 3, 4, 5))
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        assert_true(t.shape() == Shape(5, 4, 3, 2))
        var t_cpu = t.to_cpu()
        assert_true(t_cpu[0, 0, 0, 0] == 0.0)
        assert_true(t_cpu[4, 3, 2, 1] == 119.0)


def test_trrev_gpu_4d_default_axes_backward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_4d_default_axes_backward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 120.0)
        var a = _tmp0.reshape(Shape(2, 3, 4, 5))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_trrev_gpu_4d_explicit_axes_forward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_4d_explicit_axes_forward")
        comptime dtype = DType.float32
        # (2,3,4,5), axes (0,2,1,3) → (2,4,3,5)
        var _tmp0 = Tensor[dtype].arange(0.0, 120.0)
        var a = _tmp0.reshape(Shape(2, 3, 4, 5))
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(0, 2, 1, 3)
        assert_true(t.shape() == Shape(2, 4, 3, 5))
        var t_cpu = t.to_cpu()
        # a[1,2,3,4]=119 → t[1,3,2,4]=119
        assert_true(t_cpu[1, 3, 2, 4] == 119.0)


def test_trrev_gpu_4d_explicit_axes_backward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_4d_explicit_axes_backward")
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 120.0)
        var a = _tmp0.reshape(Shape(2, 3, 4, 5))
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(0, 2, 1, 3)
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ── GPU grad flow ──────────────────────────────────────────────────────────────


def test_trrev_gpu_grad_flow_through_multiple_ops() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_grad_flow_through_multiple_ops")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t1 = a_gpu.transpose()
        var t2 = a_gpu.transpose()
        var added = t1 + t2
        var loss = added.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].full(Shape(2, 2), Scalar[dtype](2.0))
            )
        )


def test_trrev_gpu_transpose_contiguous_then_op_backward() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_transpose_contiguous_then_op_backward")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var c = t.contiguous()
        var scaled = (
            c * Tensor[dtype].full(c.shape(), Scalar[dtype](2.0)).to_gpu()
        )
        var loss = scaled.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(
            a.grad().all_close(
                Tensor[dtype].full(Shape(2, 3), Scalar[dtype](2.0))
            )
        )


def test_trrev_gpu_grad_does_not_accumulate_across_separate_passes() raises:
    comptime if has_accelerator():
        print("test_trrev_gpu_grad_does_not_accumulate_across_separate_passes")
        comptime dtype = DType.float32
        # Run only one backward pass — verify grad is exactly ones
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var loss = t.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────



# Start of old tests
# End of old tests


# ═════════════════════════════════════════════════════════════════════════════
# CPU tile Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# CPU tile Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GPU tile Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_tile_gpu_1d_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.tile([3])
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(9))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d1([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
            )
        )


def test_tile_gpu_1d_once() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.tile([1])
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(3))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0]))
        )


def test_tile_gpu_2d_both_dims() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var result = a.tile([2, 3])
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(4, 6))
        var result_cpu = result.to_cpu()
        assert_true(result_cpu[[0, 0]] == Scalar[dtype](1.0))
        assert_true(result_cpu[[0, 2]] == Scalar[dtype](1.0))
        assert_true(result_cpu[[0, 4]] == Scalar[dtype](1.0))
        assert_true(result_cpu[[2, 0]] == Scalar[dtype](1.0))


def test_tile_gpu_2d_rows_only() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.tile([3])
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 9))


def test_tile_gpu_2d_extra_repeat_dims() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.tile([2, 3, 4])
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 6, 12))


def test_tile_gpu_3d_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var result = a.tile([1, 2, 1])
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 4, 2))


def test_tile_gpu_values_preserved() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([10.0, 20.0]).to_gpu()
        var result = a.tile([4])
        var result_cpu = result.to_cpu()
        for i in range(4):
            assert_true(result_cpu[[i * 2]] == Scalar[dtype](10.0))
            assert_true(result_cpu[[i * 2 + 1]] == Scalar[dtype](20.0))


def test_tile_gpu_no_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=False).to_gpu()
        var result = a.tile([3])
        assert_true(not result.requires_grad)


def test_tile_gpu_requires_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True).to_gpu()
        var result = a.tile([3])
        assert_true(result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# GPU tile Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_tile_gpu_backward_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.tile([3])
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(3), 3.0)))


def test_tile_gpu_backward_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.tile([2, 3])
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 6.0)))


def test_tile_gpu_backward_chain() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.tile([4]) * 2.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2), 8.0)))


def test_tile_gpu_backward_nonuniform_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.tile([2])
        var weights = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        var loss = (result * weights).sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 6.0])))


def test_tile_gpu_backward_extra_dims() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.tile([2, 1, 3])
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 6.0)))


# ═════════════════════════════════════════════════════════════════════════════
# CPU/GPU Parity Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_tile_parity_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.tile([3]).all_close(a_gpu.tile([3]).to_cpu()))


def test_tile_parity_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.tile([2, 3]).all_close(a_gpu.tile([2, 3]).to_cpu()))


def test_tile_parity_extra_dims() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.tile([2, 3, 4]).all_close(a_gpu.tile([2, 3, 4]).to_cpu())
        )


def test_tile_parity_backward_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = (
            Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True).to_gpu()
        )

        var loss_cpu = a_cpu.tile([3]).sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.tile([3]).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


def test_tile_parity_backward_2d() raises:
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

        var loss_cpu = a_cpu.tile([2, 3]).sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.tile([2, 3]).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


def test_tile_parity_using_zero_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.tile([3]).sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad().copy()

        a_cpu.zero_grad()

        var loss_gpu = a_gpu.tile([3]).sum()
        loss_gpu.backward()

        assert_true(cpu_grad.all_close(a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close(a_cpu.grad()))


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def tensors_close[
    dtype: DType
](a: Tensor[dtype], b: Tensor[dtype],) raises -> Bool:
    return a.all_close[atol=Scalar[dtype](1e-5)](b)


def gradboxes_close[
    dtype: DType
](a: Gradbox[dtype], b: Gradbox[dtype],) raises -> Bool:
    return a.all_close[atol=Scalar[dtype](1e-5)](b)


# ═══════════════════════════════════════════════════════════════════════════════
# as_gradbox — CPU
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# as_tensor — CPU
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# roundtrip — CPU
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# as_gradbox — GPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_as_gradbox_gpu_1d_contiguous() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].arange(6).to_gpu()
        var g = t.as_gradbox()
        assert_true(g.is_on_gpu())
        assert_true(g.is_contiguous())
        assert_true(g.shape() == Shape(6))
        # verify values via CPU
        var g_cpu = g.to_cpu()
        var expected = Gradbox[dtype](
            NDBuffer[dtype](0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
        )
        assert_true(gradboxes_close(g_cpu, expected))


def test_as_gradbox_gpu_2d_contiguous() raises:
    comptime if has_accelerator():
        var _tmp0 = Tensor[dtype].arange(6)
        var _tmp1 = _tmp0.reshape(Shape(2, 3))
        var t = _tmp1.to_gpu()
        var g = t.as_gradbox()
        assert_true(g.is_on_gpu())
        assert_true(g.is_contiguous())
        assert_true(g.shape() == Shape(2, 3))
        var g_cpu = g.to_cpu()
        var expected = Gradbox[dtype](
            NDBuffer[dtype](0.0, 1.0, 2.0, 3.0, 4.0, 5.0).reshape(Shape(2, 3))
        )
        assert_true(gradboxes_close(g_cpu, expected))


def test_as_gradbox_gpu_3d_contiguous() raises:
    comptime if has_accelerator():
        var _tmp0 = Tensor[dtype].arange(24)
        var _tmp1 = _tmp0.reshape(Shape(2, 3, 4))
        var t = _tmp1.to_gpu()
        var g = t.as_gradbox()
        assert_true(g.is_on_gpu())
        assert_true(g.is_contiguous())
        assert_true(g.shape() == Shape(2, 3, 4))


def test_as_gradbox_gpu_4d() raises:
    comptime if has_accelerator():
        var _tmp0 = Tensor[dtype].arange(120)
        var _tmp1 = _tmp0.reshape(Shape(2, 3, 4, 5))
        var t = _tmp1.to_gpu()
        var g = t.as_gradbox()
        assert_true(g.is_on_gpu())
        assert_true(g.is_contiguous())
        assert_true(g.shape() == Shape(2, 3, 4, 5))


def test_as_gradbox_gpu_contiguous_false() raises:
    comptime if has_accelerator():
        var _tmp0 = Tensor[dtype].arange(6)
        var _tmp1 = _tmp0.reshape(Shape(2, 3))
        var t = _tmp1.to_gpu()
        var g = t.as_gradbox(contiguous=False)
        assert_true(g.is_on_gpu())
        assert_true(g.shape() == Shape(2, 3))


def test_as_gradbox_gpu_values_preserved() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(4, 5)
        var t_gpu = t_cpu.to_gpu()
        var g_gpu = t_gpu.as_gradbox()
        var g_cpu = g_gpu.to_cpu()
        assert_true(gradboxes_close(g_cpu, t_cpu.as_gradbox()))


def test_as_gradbox_gpu_large() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(64, 128)
        var t_gpu = t_cpu.to_gpu()
        var g_gpu = t_gpu.as_gradbox()
        assert_true(g_gpu.is_on_gpu())
        assert_true(g_gpu.shape() == Shape(64, 128))
        var g_cpu = g_gpu.to_cpu()
        assert_true(gradboxes_close(g_cpu, t_cpu.as_gradbox()))


# ═══════════════════════════════════════════════════════════════════════════════
# as_tensor — GPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_as_tensor_gpu_1d_contiguous() raises:
    comptime if has_accelerator():
        var g_cpu = Gradbox[dtype](NDBuffer[dtype](1.0, 2.0, 3.0, 4.0))
        var t_gpu = g_cpu.as_tensor().to_gpu()
        var g_gpu = t_gpu.as_gradbox()
        var t2 = g_gpu.as_tensor()
        assert_true(t2.is_on_gpu())
        assert_true(t2.shape() == Shape(4))
        var t2_cpu = t2.to_cpu()
        assert_true(
            tensors_close(t2_cpu, Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]))
        )


def test_as_tensor_gpu_2d_contiguous() raises:
    comptime if has_accelerator():
        var g_cpu = Gradbox[dtype].zeros(Shape(3, 4))
        var t_gpu = g_cpu.as_tensor().to_gpu()
        var g_gpu = t_gpu.as_gradbox()
        var t2 = g_gpu.as_tensor()
        assert_true(t2.is_on_gpu())
        assert_true(t2.shape() == Shape(3, 4))
        assert_true(
            tensors_close(t2.to_cpu(), Tensor[dtype].zeros(Shape(3, 4)))
        )
        var t_cpu = Tensor[dtype].rand(2, 3, 4)
        t_gpu = t_cpu.to_gpu()
        g_gpu = t_gpu.as_gradbox()
        t2 = g_gpu.as_tensor()
        assert_true(t2.is_on_gpu())
        assert_true(t2.shape() == Shape(2, 3, 4))
        assert_true(tensors_close(t2.to_cpu(), t_cpu))


def test_as_tensor_gpu_requires_grad() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(3, 4)
        var t_gpu = t_cpu.to_gpu()
        var g_gpu = t_gpu.as_gradbox()
        var t2 = g_gpu.as_tensor(requires_grad=True)
        assert_true(t2.requires_grad)
        assert_true(t2.is_on_gpu())


def test_as_tensor_gpu_large() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(64, 128)
        var t_gpu = t_cpu.to_gpu()
        var g_gpu = t_gpu.as_gradbox()
        var t2 = g_gpu.as_tensor()
        assert_true(t2.is_on_gpu())
        assert_true(t2.shape() == Shape(64, 128))
        assert_true(tensors_close(t2.to_cpu(), t_cpu))


def test_as_tensor_gpu_4d() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(2, 3, 4, 5)
        var t_gpu = t_cpu.to_gpu()
        var g_gpu = t_gpu.as_gradbox()
        var t2 = g_gpu.as_tensor()
        assert_true(t2.is_on_gpu())
        assert_true(t2.shape() == Shape(2, 3, 4, 5))
        assert_true(tensors_close(t2.to_cpu(), t_cpu))


# ═══════════════════════════════════════════════════════════════════════════════
# roundtrip — GPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_roundtrip_tensor_gradbox_tensor_gpu() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(3, 4)
        var t_gpu = t_cpu.to_gpu()
        var g_gpu = t_gpu.as_gradbox()
        var t2 = g_gpu.as_tensor()
        assert_true(t2.is_on_gpu())
        assert_true(tensors_close(t2.to_cpu(), t_cpu))


def test_roundtrip_gradbox_tensor_gradbox_gpu() raises:
    comptime if has_accelerator():
        var g_cpu = Gradbox[dtype].rand(Shape(4, 5))
        var t_gpu = g_cpu.as_tensor().to_gpu()
        var g_gpu = t_gpu.as_gradbox()
        var t2 = g_gpu.as_tensor()
        var g2 = t2.as_gradbox()
        assert_true(g2.is_on_gpu())
        assert_true(gradboxes_close(g2.to_cpu(), g_cpu))


def test_roundtrip_cpu_to_gpu_to_cpu() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(5, 6)
        var g_cpu = t_cpu.as_gradbox()
        var t_gpu = g_cpu.as_tensor().to_gpu()
        var g_gpu = t_gpu.as_gradbox()
        var t_back = g_gpu.as_tensor().to_cpu()
        assert_true(tensors_close(t_back, t_cpu))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# SQRT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_uop_sqrt_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 4.0], [9.0, 16.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.sqrt()
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
            )
        )
        var loss = result.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[0.5, 0.25], [0.16667, 0.125]])
            )
        )


def test_uop_sqrt_gpu_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full(Shape(2, 3, 4), 4.0, requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.sqrt()
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].full(Shape(2, 3, 4), 2.0))
        )
        var loss = result.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].full(Shape(2, 3, 4), 0.25)
            )
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NEGATE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_uop_negate_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, -2.0], [-3.0, 4.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()
        var result = -a_gpu
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]])
            )
        )
        var loss = result.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(Tensor[dtype].full(Shape(2, 2), -1.0))
        )


def test_uop_negate_gpu_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full(Shape(2, 3, 4), 5.0, requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var result = -a_gpu
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].full(Shape(2, 3, 4), -5.0))
        )
        var loss = result.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(Tensor[dtype].full(Shape(2, 3, 4), -1.0))
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ABS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_uop_abs_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[-1.0, 2.0], [-3.0, 4.0]])
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.__abs__()
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
            )
        )


def test_uop_abs_gpu_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full(Shape(2, 3, 4), -5.0)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.__abs__()
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].full(Shape(2, 3, 4), 5.0))
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RELU TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_uop_relu_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[-1.0, 2.0], [-3.0, 4.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.relu()
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[0.0, 2.0], [0.0, 4.0]])
            )
        )
        var loss = result.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(Tensor[dtype].d2([[0.0, 1.0], [0.0, 1.0]]))
        )


def test_uop_relu_gpu_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].full(Shape(2, 3, 4), -1.0, requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.relu()
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].full(Shape(2, 3, 4), 0.0))
        )
        var loss = result.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(Tensor[dtype].full(Shape(2, 3, 4), 0.0))
        )


# ═══════════════════════════════════════════════════════════════════════════════
# INVERT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_uop_invert_bool_gpu() raises:
    comptime if has_accelerator():
        var a_cpu = Tensor[DType.bool].d2([[True, False], [False, True]])
        var a_gpu = a_cpu.to_gpu()
        var result = ~a_gpu
        assert_true(
            result.to_cpu()
            == Tensor[DType.bool].d2([[False, True], [True, False]])
        )


def test_uop_invert_bool_gpu_3d() raises:
    comptime if has_accelerator():
        var a_cpu = Tensor[DType.bool].full(
            Shape(2, 3, 4), Scalar[DType.bool](True)
        )
        var a_gpu = a_cpu.to_gpu()
        var result = ~a_gpu
        assert_true(
            result.to_cpu()
            == Tensor[DType.bool].full(
                Shape(2, 3, 4), Scalar[DType.bool](False)
            )
        )


def test_uop_invert_bool_gpu_double() raises:
    comptime if has_accelerator():
        # ~~a == a on GPU
        var a_cpu = Tensor[DType.bool].d1([True, False, True, False])
        var a_gpu = a_cpu.to_gpu()
        assert_true((~~a_gpu).to_cpu() == a_cpu)


def test_uop_invert_int_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.int32
        var a_cpu = Tensor[dtype].d2([[1, 2], [3, 4]])
        var a_gpu = a_cpu.to_gpu()
        var result = ~a_gpu
        assert_true(result.to_cpu() == Tensor[dtype].d2([[-2, -3], [-4, -5]]))


def test_uop_invert_int_gpu_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.int32
        var a_cpu = Tensor[dtype].full(Shape(2, 3, 4), 1)
        var a_gpu = a_cpu.to_gpu()
        var result = ~a_gpu
        assert_true(result.to_cpu() == Tensor[dtype].full(Shape(2, 3, 4), -2))


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS OP GRAD FLOW — ensure INVERT did not break other ops
# ═══════════════════════════════════════════════════════════════════════════════


def test_uop_cross_sqrt_negate_grad_flow_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([4.0, 9.0, 16.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var loss = (-a_gpu.sqrt()).sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([-0.25, -0.16667, -0.125])
            )
        )


def test_uop_cross_relu_negate_grad_flow_gpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var loss = (-a_gpu).relu().sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(Tensor[dtype].d1([-1.0, 0.0, -1.0, 0.0]))
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════



# ====== Complex and edge cases =======


def test_varstd_gpu_variance_welford_numerical_stability() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1(
            [1_000_001.0, 1_000_002.0, 1_000_003.0], requires_grad=True
        )
        var x_gpu = x.to_gpu()
        var v = x_gpu.variance[track_grad=True](unbiased=False)
        assert_true(
            v.to_cpu().all_close[atol=1e-1](Tensor[dtype].scalar(0.6666667))
        )
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d1([-0.6666667, 0.0, 0.6666667])
        assert_true(x.grad().all_close[atol=1e-3](expected))


def _varstd_cpu_variance_welford_numerical_stability_ubiased() raises:
    comptime dtype = DType.float32
    # Large offset — two-pass formula loses precision, Welford stays stable
    # var([1e6+1, 1e6+2, 1e6+3]) = var([1,2,3]) = 1.0 (population)
    var x = Tensor[dtype].d1(
        [1_000_001.0, 1_000_002.0, 1_000_003.0], requires_grad=True
    )
    var v = x.variance[track_grad=True](unbiased=True)
    assert_true(v.all_close[atol=1e-1](Tensor[dtype].scalar(1.0)))
    var loss = v.sum()
    loss.backward()
    # grad = 2*(x-mean)/n, mean=1e6+2, n=3
    var expected = Tensor[dtype].d1([-0.6666667, 0.0, 0.6666667])
    assert_true(x.grad().all_close[atol=1e-3](expected))


# ===== VARIANCE CPU TESTS =====


# ===== STD CPU TESTS =====


# ===== GPU TESTS =====


def test_varstd_gpu_variance_global() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1(
            [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], requires_grad=True
        )
        var x_gpu = x.to_gpu()
        var v = x_gpu.variance[track_grad=True](unbiased=False)
        assert_true(v.to_cpu().all_close[atol=1e-5](Tensor[dtype].scalar(4.0)))
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [-0.75, -0.25, -0.25, -0.25, 0.0, 0.0, 0.5, 1.0]
        )
        assert_true(x.grad().all_close[atol=1e-4](expected))


def test_varstd_gpu_variance_axis0() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        var x_gpu = x.to_gpu()
        var v = x_gpu.variance[track_grad=True](
            axis=0, keepdims=False, unbiased=False
        )
        assert_true(v.shape() == Shape(2))
        assert_true(
            v.to_cpu().all_close[atol=1e-4](
                Tensor[dtype].d1([2.6666667, 2.6666667])
            )
        )
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d2(
            [[-1.3333334, -1.3333334], [0.0, 0.0], [1.3333334, 1.3333334]]
        )
        assert_true(x.grad().all_close[atol=1e-4](expected))


def test_varstd_gpu_variance_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 8.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var v = x_gpu.variance[track_grad=True](
            axis=1, keepdims=False, unbiased=False
        )
        assert_true(
            v.to_cpu().all_close[atol=1e-5](Tensor[dtype].d1([1.0, 9.0]))
        )
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d2([[-1.0, 1.0], [-3.0, 3.0]])
        assert_true(x.grad().all_close[atol=1e-5](expected))


def test_varstd_gpu_variance_keepdims() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 8.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var v = x_gpu.variance[track_grad=True](
            axis=1, keepdims=True, unbiased=False
        )
        assert_true(v.shape() == Shape(2, 1))
        assert_true(
            v.to_cpu().all_close[atol=1e-5](Tensor[dtype].d2([[1.0], [9.0]]))
        )
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d2([[-1.0, 1.0], [-3.0, 3.0]])
        assert_true(x.grad().all_close[atol=1e-5](expected))


def test_varstd_gpu_std_global() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1(
            [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], requires_grad=True
        )
        var x_gpu = x.to_gpu()
        var s = x_gpu.std[track_grad=True](unbiased=False)
        assert_true(s.to_cpu().all_close[atol=1e-5](Tensor[dtype].scalar(2.0)))
        var loss = s.sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [-0.1875, -0.0625, -0.0625, -0.0625, 0.0, 0.0, 0.125, 0.25]
        )
        assert_true(x.grad().all_close[atol=1e-4](expected))


def test_varstd_gpu_std_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 8.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var s = x_gpu.std[track_grad=True](
            axis=1, keepdims=False, unbiased=False
        )
        assert_true(
            s.to_cpu().all_close[atol=1e-5](Tensor[dtype].d1([1.0, 3.0]))
        )
        var loss = s.sum()
        loss.backward()
        var expected = Tensor[dtype].d2([[-0.5, 0.5], [-0.5, 0.5]])
        assert_true(x.grad().all_close[atol=1e-5](expected))


def test_varstd_gpu_std_axis0_keepdims() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var s = x_gpu.std[track_grad=True](
            axis=0, keepdims=True, unbiased=False
        )
        assert_true(s.shape() == Shape(1, 2))
        assert_true(
            s.to_cpu().all_close[atol=1e-5](Tensor[dtype].d2([[1.0, 1.0]]))
        )
        var loss = s.sum()
        loss.backward()
        var expected = Tensor[dtype].d2([[-0.5, -0.5], [0.5, 0.5]])
        assert_true(x.grad().all_close[atol=1e-5](expected))


def test_varstd_gpu_std_vs_cpu_consistency() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var x_gpu = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            .to_gpu()
        )
        var s_cpu = x_cpu.std[track_grad=True](
            axis=1, keepdims=False, unbiased=False
        )
        var s_gpu = x_gpu.std[track_grad=True](
            axis=1, keepdims=False, unbiased=False
        )
        assert_true(s_cpu.all_close[atol=1e-5](s_gpu.to_cpu()))
        var loss_cpu = s_cpu.sum()
        loss_cpu.backward()
        var loss_gpu = s_gpu.sum()
        loss_gpu.backward()
        assert_true(x_cpu.grad().all_close[atol=1e-5](x_gpu.grad().to_cpu()))


def test_varstd_gpu_variance_unbiased() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var v = x_gpu.variance[track_grad=True](unbiased=True)
        assert_true(v.to_cpu().all_close[atol=1e-5](Tensor[dtype].scalar(2.5)))
        var loss = v.sum()
        loss.backward()
        var expected = Tensor[dtype].d1([-1.0, -0.5, 0.0, 0.5, 1.0])
        assert_true(x.grad().all_close[atol=1e-5](expected))



def test_varstd_var_fwd_gpu_global_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0]).to_gpu()
        var v = x.variance[track_grad=False](unbiased=False)
        assert_true(v.to_cpu().all_close[atol=1e-5](Tensor[dtype].scalar(2.0)))


def test_varstd_var_fwd_gpu_axis0_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var v = x.variance[track_grad=False](axis=0, unbiased=False)
        assert_true(v.shape() == Shape(2))
        assert_true(
            v.to_cpu().all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0]))
        )


def test_varstd_var_fwd_gpu_axis1_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]]).to_gpu()
        var v = x.variance[track_grad=False](axis=1, unbiased=False)
        assert_true(v.shape() == Shape(2))
        assert_true(
            v.to_cpu().all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0]))
        )


def test_varstd_var_fwd_gpu_3d_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var _tmp1 = _tmp0.reshape(2, 3, 4)
        var x = _tmp1.to_gpu()
        var v = x.variance[track_grad=False](axis=1, unbiased=False)
        assert_true(v.shape() == Shape(2, 4))


def test_varstd_var_bwd_gpu_global_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True
        )
        var x_gpu = x_cpu.to_gpu()
        var v = x_gpu.variance[track_grad=True](unbiased=False)
        var loss = v.sum()
        loss.backward()
        assert_true(
            x_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([-0.8, -0.4, 0.0, 0.4, 0.8])
            )
        )


def test_varstd_var_bwd_gpu_axis1_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2(
            [[1.0, 3.0], [2.0, 4.0]], requires_grad=True
        )
        var x_gpu = x_cpu.to_gpu()
        var v = x_gpu.variance[track_grad=True](axis=1, unbiased=False)
        var loss = v.sum()
        loss.backward()
        assert_true(
            x_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[-1.0, 1.0], [-1.0, 1.0]])
            )
        )


def test_varstd_var_bwd_gpu_3d_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var x_cpu = _tmp0.reshape(2, 3, 4)
        x_cpu.requires_grad_(True)
        var x_gpu = x_cpu.to_gpu()
        var v = x_gpu.variance[track_grad=True](axis=1, unbiased=False)
        var loss = v.sum()
        loss.backward()
        assert_true(x_cpu.grad().shape() == Shape(2, 3, 4))
        assert_true(x_cpu.grad().sum().item() == x_cpu.grad().sum().item())


# ═════════════════════════════════════════════════════════════════════════════
# STD — GPU
# ═════════════════════════════════════════════════════════════════════════════


def test_varstd_std_fwd_gpu_global_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0]).to_gpu()
        var s = x.std[track_grad=False](unbiased=False)
        assert_true(
            s.to_cpu().all_close[atol=1e-5](Tensor[dtype].scalar(1.4142135))
        )


def test_varstd_std_fwd_gpu_axis0_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var s = x.std[track_grad=False](axis=0, unbiased=False)
        assert_true(s.shape() == Shape(2))
        assert_true(
            s.to_cpu().all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0]))
        )


def test_varstd_std_fwd_gpu_axis1_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]]).to_gpu()
        var s = x.std[track_grad=False](axis=1, unbiased=False)
        assert_true(s.shape() == Shape(2))
        assert_true(
            s.to_cpu().all_close[atol=1e-5](Tensor[dtype].d1([1.0, 1.0]))
        )


def test_varstd_std_fwd_gpu_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(0.0, 24.0)
        var _tmp1 = _tmp0.reshape(2, 3, 4)
        var x = _tmp1.to_gpu()
        var s = x.std[track_grad=False](axis=2, unbiased=False)
        assert_true(s.shape() == Shape(2, 3))


def test_varstd_std_bwd_gpu_global_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True
        )
        var x_gpu = x_cpu.to_gpu()
        var s = x_gpu.std[track_grad=True](unbiased=False)
        var loss = s.sum()
        loss.backward()
        var std_val = 1.4142135
        var expected = Tensor[dtype].d1(
            [
                Scalar[dtype](-2.0 / (std_val * 5.0)),
                Scalar[dtype](-1.0 / (std_val * 5.0)),
                Scalar[dtype](0.0),
                Scalar[dtype](1.0 / (std_val * 5.0)),
                Scalar[dtype](2.0 / (std_val * 5.0)),
            ]
        )
        assert_true(x_cpu.grad().all_close[atol=1e-5](expected))


def test_varstd_std_bwd_gpu_axis1_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2(
            [[1.0, 3.0], [2.0, 4.0]], requires_grad=True
        )
        var x_gpu = x_cpu.to_gpu()
        var s = x_gpu.std[track_grad=True](axis=1, unbiased=False)
        var loss = s.sum()
        loss.backward()
        assert_true(
            x_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[-0.5, 0.5], [-0.5, 0.5]])
            )
        )


def test_varstd_std_bwd_gpu_3d_axis2() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(1.0, 25.0)
        var x_cpu = _tmp0.reshape(2, 3, 4)
        x_cpu.requires_grad_(True)
        var x_gpu = x_cpu.to_gpu()
        var s = x_gpu.std[track_grad=True](axis=2, unbiased=False)
        var loss = s.sum()
        loss.backward()
        assert_true(x_cpu.grad().shape() == Shape(2, 3, 4))
        assert_true(x_cpu.grad().sum().item() == x_cpu.grad().sum().item())


# ═════════════════════════════════════════════════════════════════════════════
# CPU / GPU PARITY
# ═════════════════════════════════════════════════════════════════════════════


def test_varstd_parity_var_fwd_global() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].arange(1.0, 11.0)
        var cpu_v = x.variance[track_grad=False](unbiased=False)
        var gpu_v = (
            x.to_gpu().variance[track_grad=False](unbiased=False).to_cpu()
        )
        assert_true(cpu_v.all_close[atol=1e-4](gpu_v))


def test_varstd_parity_var_fwd_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(1.0, 13.0)
        var x = _tmp0.reshape(3, 4)
        var cpu_v = x.variance[track_grad=False](axis=1, unbiased=False)
        var gpu_v = (
            x.to_gpu()
            .variance[track_grad=False](axis=1, unbiased=False)
            .to_cpu()
        )
        assert_true(cpu_v.all_close[atol=1e-4](gpu_v))


def test_varstd_parity_std_fwd_global() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].arange(1.0, 11.0)
        var cpu_s = x.std[track_grad=False](unbiased=False)
        var gpu_s = x.to_gpu().std[track_grad=False](unbiased=False).to_cpu()
        assert_true(cpu_s.all_close[atol=1e-4](gpu_s))


def test_varstd_parity_std_fwd_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(1.0, 13.0)
        var x = _tmp0.reshape(3, 4)
        var cpu_s = x.std[track_grad=False](axis=1, unbiased=False)
        var gpu_s = (
            x.to_gpu().std[track_grad=False](axis=1, unbiased=False).to_cpu()
        )
        assert_true(cpu_s.all_close[atol=1e-4](gpu_s))


def test_varstd_parity_var_bwd_global() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu_leaf = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True
        )
        var loss_cpu = x_cpu_leaf.variance[track_grad=True](
            unbiased=False
        ).sum()
        loss_cpu.backward()

        var x_gpu_leaf = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True
        )
        var loss_gpu = (
            x_gpu_leaf.to_gpu().variance[track_grad=True](unbiased=False).sum()
        )
        loss_gpu.backward()

        assert_true(x_cpu_leaf.grad().all_close[atol=1e-5](x_gpu_leaf.grad()))


def test_varstd_parity_std_bwd_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu_leaf = Tensor[dtype].d2(
            [[1.0, 3.0], [2.0, 4.0]], requires_grad=True
        )
        var loss_cpu = x_cpu_leaf.std[track_grad=True](
            axis=1, unbiased=False
        ).sum()
        loss_cpu.backward()

        var x_gpu_leaf = Tensor[dtype].d2(
            [[1.0, 3.0], [2.0, 4.0]], requires_grad=True
        )
        var loss_gpu = (
            x_gpu_leaf.to_gpu()
            .std[track_grad=True](axis=1, unbiased=False)
            .sum()
        )
        loss_gpu.backward()

        assert_true(x_cpu_leaf.grad().all_close[atol=1e-5](x_gpu_leaf.grad()))



# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASIC TESTS - 1D
# ═══════════════════════════════════════════════════════════════════════════════


def test_sum_1d_forward_gpu() raises:
    """Test 1D sum forward on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum()
        assert_true(result.to_cpu().item() == 15.0)


def test_sum_1d_backward_gpu() raises:
    """Test 1D sum backward on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var loss = a_gpu.sum()
        loss.backward()
        var expected = Tensor[dtype].ones_like(a)
        assert_true(a.grad().all_close(expected))


def test_mean_1d_forward_gpu() raises:
    """Test 1D mean forward on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
        var a_gpu = a.to_gpu()
        var result = a_gpu.mean()
        assert_true(result.to_cpu().item() == 3.0)


def test_mean_1d_backward_gpu() raises:
    """Test 1D mean backward on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var loss = a_gpu.mean()
        loss.backward()
        var expected = Tensor[dtype].d1([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        assert_true(a.grad().all_close(expected))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 2D TESTS - Axes and Keepdims
# ═══════════════════════════════════════════════════════════════════════════════


def test_sum_2d_axis0_gpu() raises:
    """Test sum along axis 0 (rows) on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum(axes=[0])
        var expected = Tensor[dtype].d1([5.0, 7.0, 9.0])
        assert_true(result.to_cpu().all_close(expected))


def test_sum_2d_axis1_gpu() raises:
    """Test sum along axis 1 (columns) on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum(axes=[1])
        var expected = Tensor[dtype].d1([6.0, 15.0])
        assert_true(result.to_cpu().all_close(expected))


def test_sum_2d_keepdims_gpu() raises:
    """Test sum with keepdims=True on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum(axes=[1], keepdims=True)
        var expected = Tensor[dtype].d2([[6.0], [15.0]])
        assert_true(result.to_cpu().all_close(expected))


def test_sum_2d_backward_axis0_gpu() raises:
    """Test backward of sum along axis 0 on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var loss = a_gpu.sum(axes=[0])
        loss.backward()
        var expected = Tensor[dtype].d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        assert_true(a.grad().all_close(expected))


def test_mean_2d_axis0_gpu() raises:
    """Test mean along axis 0 on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.mean(axes=[0])
        var expected = Tensor[dtype].d1([2.5, 3.5, 4.5])
        assert_true(result.to_cpu().all_close(expected))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 3D TESTS - Higher Dimensions
# ═══════════════════════════════════════════════════════════════════════════════


def test_sum_3d_all_gpu() raises:
    """Test sum of all elements in 3D tensor on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum()
        assert_true(result.to_cpu().item() == 36.0)


def test_sum_3d_axis12_gpu() raises:
    """Test sum over last two axes on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum(axes=[1, 2])
        var expected = Tensor[dtype].d1([10.0, 26.0])
        assert_true(result.to_cpu().all_close(expected))


def test_mean_3d_axis0_gpu() raises:
    """Test mean along first axis on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.mean(axes=[0])
        var expected = Tensor[dtype].d2([[3.0, 4.0], [5.0, 6.0]])
        assert_true(result.to_cpu().all_close(expected))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GRADIENT FLOW VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


def test_sum_grad_flow_chain_gpu() raises:
    """Test gradient flow through chain of operations on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu * 2.0
        var loss = b.sum()
        loss.backward()
        var expected = Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]])
        assert_true(a.grad().all_close(expected))


def test_mean_grad_flow_scaling_gpu() raises:
    """Test mean gradient scaling on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var loss = a_gpu.mean()
        loss.backward()
        var expected = Tensor[dtype].d1([0.25, 0.25, 0.25, 0.25])
        assert_true(a.grad().all_close(expected))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MULTIPLE BACKWARD PASSES (Grad Accumulation)
# ═══════════════════════════════════════════════════════════════════════════════


def test_sum_multiple_backward_gpu() raises:
    """Test multiple backward passes on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()

        var loss1 = a_gpu.sum()
        loss1.backward()
        var grad_after_first = a.grad().copy()

        var a_gpu2 = a.to_gpu()  # Re-get with accumulated grad
        var loss2 = (a_gpu2 * 2.0).sum()
        loss2.backward()

        var expected = grad_after_first + Tensor[dtype].d1([2.0, 2.0, 2.0])
        assert_true(a.grad().all_close(expected))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════


def test_sum_single_element_gpu() raises:
    """Test sum of single element tensor on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].scalar(42.0)
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum()
        assert_true(result.to_cpu().item() == 42.0)


def test_sum_negative_axis_gpu() raises:
    """Test sum with negative axes on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum(axes=[-1])
        var expected = Tensor[dtype].d1([3.0, 7.0])
        assert_true(result.to_cpu().all_close(expected))


def test_sum_empty_axes_gpu() raises:
    """Test sum with empty axes on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum(axes=[])
        var expected = Tensor[dtype].scalar(10.0)
        assert_true(result.to_cpu().all_close(expected))


def test_mean_empty_axes_gpu() raises:
    """Test mean with empty axes on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.mean(axes=[])
        var expected = Tensor[dtype].scalar(2.5)
        assert_true(result.to_cpu().all_close(expected))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. LARGE TENSOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_sum_large_1d_gpu() raises:
    """Test sum of large 1D tensor on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var size = 10000
        var data = List[Scalar[dtype]]()
        for i in range(size):
            data.append(Scalar[dtype](i + 1))
        var a = Tensor[dtype].d1(data)
        var a_gpu = a.to_gpu()
        var result = a_gpu.sum()
        var expected = (size * (size + 1)) // 2
        assert_true(result.to_cpu().item() == Float32(expected))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. CPU-GPU CONSISTENCY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_sum_cpu_gpu_consistency_2d() raises:
    """Test CPU and GPU results match for 2D sum."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        )

        var cpu_result = a.sum()
        var gpu_result = a.to_gpu().sum()

        assert_true(cpu_result.all_close(gpu_result.to_cpu()))


def test_mean_cpu_gpu_consistency_3d() raises:
    """Test CPU and GPU results match for 3D mean."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )

        var cpu_result = a.mean(axes=[1, 2])
        var gpu_result = a.to_gpu().mean(axes=[1, 2])

        assert_true(cpu_result.all_close(gpu_result.to_cpu()))


# ═══════════════════════════════════════════════════════════════════════════════
# 9. BENCHMARK: SIMD FAST PATH TIMING ($ ./execute.sh summean ... --only test_*benchmark*)
# ═══════════════════════════════════════════════════════════════════════════════

# These are timing benchmarks, not correctness tests. They verify that the
# SIMD fast path for suffix-axis reductions is actually ~100x+ faster than
# the coord-by-coord fallback. Run them individually to get timing output.


def assert_fast_path_timing(
    shape: Shape, axes: IntArray, num_trials: Int
) raises:
    """Measure a suffix-axis reduction and assert it completes in < 10ms."""
    comptime dtype = DType.float32
    var t = Tensor[dtype].rand(shape)
    for _ in range(5):
        var _ = t.sum(axes=axes)
    var start = now()
    for _ in range(num_trials):
        var _ = t.sum(axes=axes)
    var end = now()
    var avg_ms = (end - start) * 1000.0 / Float64(num_trials)
    var label = (
        String(shape)
        + " axes="
        + String(axes)
        + " avg="
        + String(avg_ms)
        + "ms"
    )
    print(label)
    # Fast path should easily clear this bar (typically < 0.5ms for these sizes)
    assert_true(avg_ms < 10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════════


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
