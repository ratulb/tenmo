from tenmo.tensor import Tensor
from tenmo.net import Tanh, Linear
from tenmo.common_utils import isnan, isinf
from tenmo.shapes import Shape
from tenmo.intarray import IntArray
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from std.math import tanh as scalar_tanh, abs as scalar_abs

comptime dtype = DType.float32
comptime tol = Float32(1e-4)
from tenmo.optim import SGD
from tenmo.filler import Filler
from std.math import log, exp


from tenmo.mnemonics import AddTensor


# ============================================================================
# Tanh Activation Tests
# ============================================================================


# ============================================================================
# Edge Cases
# ============================================================================


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def tanh_close(a: Tensor[dtype], b: Tensor[dtype]) raises -> Bool:
    return a.all_close[atol=tol](b)


def tanh_expected_1d() -> Tensor[dtype]:
    """T[t]anh([0, 0.5, -0.5, 1, -1])."""
    return Tensor[dtype].d1(
        [
            scalar_tanh(Float32(0.0)),
            scalar_tanh(Float32(0.5)),
            scalar_tanh(Float32(-0.5)),
            scalar_tanh(Float32(1.0)),
            scalar_tanh(Float32(-1.0)),
        ]
    )


def tanh_grad_expected_1d() -> Tensor[dtype]:
    """1 - tanh^2([0, 0.5, -0.5, 1, -1])."""
    return Tensor[dtype].d1(
        [
            Float32(1) - scalar_tanh(Float32(0.0)) ** 2,
            Float32(1) - scalar_tanh(Float32(0.5)) ** 2,
            Float32(1) - scalar_tanh(Float32(-0.5)) ** 2,
            Float32(1) - scalar_tanh(Float32(1.0)) ** 2,
            Float32(1) - scalar_tanh(Float32(-1.0)) ** 2,
        ]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD — CPU
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD — CPU
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# FORWARD — GPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_tanh_gpu_fwd_scalar_zero() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].scalar(0.0).to_gpu()
        var out = t.tanh()
        assert_true(out.is_on_gpu())
        assert_true(tanh_close(out.to_cpu(), Tensor[dtype].scalar(0.0)))


def test_tanh_gpu_fwd_1d_zeros() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(8)).to_gpu()
        var out = t.tanh()
        assert_true(out.is_on_gpu())
        assert_true(tanh_close(out.to_cpu(), Tensor[dtype].zeros(Shape(8))))


def test_tanh_gpu_fwd_1d_known() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].d1([0.0, 0.5, -0.5, 1.0, -1.0])
        var out_gpu = t_cpu.to_gpu().tanh()
        assert_true(out_gpu.is_on_gpu())
        assert_true(tanh_close(out_gpu.to_cpu(), tanh_expected_1d()))


def test_tanh_gpu_fwd_2d_known() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].d2([[0.0, 1.0], [-1.0, 0.5]])
        var out_cpu = t_cpu.tanh()
        var out_gpu = t_cpu.to_gpu().tanh()
        assert_true(out_gpu.is_on_gpu())
        assert_true(tanh_close(out_gpu.to_cpu(), out_cpu))


def test_tanh_gpu_fwd_3d() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(Shape(2, 3, 4))
        var out_cpu = t_cpu.tanh()
        var out_gpu = t_cpu.to_gpu().tanh()
        assert_true(out_gpu.is_on_gpu())
        assert_true(tanh_close(out_gpu.to_cpu(), out_cpu))


def test_tanh_gpu_fwd_4d() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(Shape(2, 3, 4, 5))
        var out_cpu = t_cpu.tanh()
        var out_gpu = t_cpu.to_gpu().tanh()
        assert_true(out_gpu.is_on_gpu())
        assert_true(tanh_close(out_gpu.to_cpu(), out_cpu))


def test_tanh_gpu_fwd_large() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(Shape(64, 128))
        var out_cpu = t_cpu.tanh()
        var out_gpu = t_cpu.to_gpu().tanh()
        assert_true(out_gpu.is_on_gpu())
        assert_true(tanh_close(out_gpu.to_cpu(), out_cpu))


def test_tanh_gpu_fwd_matches_cpu() raises:
    comptime if has_accelerator():
        var t_cpu = Tensor[dtype].rand(Shape(9, 20))
        assert_true(tanh_close(t_cpu.to_gpu().tanh().to_cpu(), t_cpu.tanh()))


def test_tanh_gpu_fwd_no_requires_grad() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].d1([1.0, 2.0]).to_gpu()
        var out = t.tanh[track_grad=False]()
        assert_true(not out.requires_grad)
        assert_true(not out.has_ancestry())


def test_tanh_gpu_fwd_requires_grad_propagates() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].d1([1.0, 2.0], requires_grad=True).to_gpu()
        var out = t.tanh()
        assert_true(out.is_on_gpu())
        assert_true(out.requires_grad)
        assert_true(out.has_ancestry())


def test_tanh_gpu_fwd_range_clamping() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].d1([-10.0, -1.0, 0.0, 1.0, 10.0]).to_gpu()
        var out = t.tanh().to_cpu()
        var data = out.data_ptr()
        for i in range(5):
            assert_true(data[i] >= Float32(-1.0) and data[i] <= Float32(1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD — GPU
# ═══════════════════════════════════════════════════════════════════════════════


def test_tanh_gpu_bwd_zeros_1d() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(4), requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(
            tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(4)))
        )


def test_tanh_gpu_bwd_scalar_zero() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].scalar(0.0, requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(tanh_close(t.grad().as_tensor(), Tensor[dtype].scalar(1.0)))


def test_tanh_gpu_bwd_1d_known() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].d1(
            [0.0, 0.5, -0.5, 1.0, -1.0], requires_grad=True
        )
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(tanh_close(t.grad().as_tensor(), tanh_grad_expected_1d()))


def test_tanh_gpu_bwd_matches_cpu() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].rand(Shape(4, 5), requires_grad=True)

        # CPU backward
        var out_cpu = t.tanh()
        var loss_cpu = out_cpu.sum()
        loss_cpu.backward()
        var grad_cpu = t.grad().as_tensor().copy()
        t.zero_grad()

        # GPU backward
        var t_gpu = t.to_gpu()
        var out_gpu = t_gpu.tanh()
        var loss_gpu = out_gpu.sum()
        loss_gpu.backward()

        assert_true(tanh_close(t.grad().as_tensor(), grad_cpu))


def test_tanh_gpu_bwd_2d() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(3, 4), requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(
            tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(3, 4)))
        )


def test_tanh_gpu_bwd_3d() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(2, 3, 4), requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(
            tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(2, 3, 4)))
        )


def test_tanh_gpu_bwd_4d() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(2, 3, 4, 5), requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(
            tanh_close(
                t.grad().as_tensor(), Tensor[dtype].ones(Shape(2, 3, 4, 5))
            )
        )


def test_tanh_gpu_bwd_chain_mul() raises:
    comptime if has_accelerator():
        # y = tanh(2*x), dy/dx = 2*(1-tanh(2x)^2)
        var t = Tensor[dtype].d1([0.0, 1.0], requires_grad=True)
        var t_gpu = t.to_gpu()
        var t2 = t_gpu * Scalar[dtype](2)
        var out = t2.tanh()
        var loss = out.sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [
                Float32(2) * (Float32(1) - scalar_tanh(Float32(0.0)) ** 2),
                Float32(2) * (Float32(1) - scalar_tanh(Float32(2.0)) ** 2),
            ]
        )
        assert_true(tanh_close(t.grad().as_tensor(), expected))


def test_tanh_gpu_bwd_double_tanh() raises:
    comptime if has_accelerator():
        # y = tanh(tanh(x)), x=0 → grad=1
        var t = Tensor[dtype].scalar(0.0, requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh().tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(tanh_close(t.grad().as_tensor(), Tensor[dtype].scalar(1.0)))


def test_tanh_gpu_bwd_large() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].zeros(Shape(32, 32), requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        assert_true(
            tanh_close(t.grad().as_tensor(), Tensor[dtype].ones(Shape(32, 32)))
        )


def test_tanh_gpu_bwd_chain_add() raises:
    comptime if has_accelerator():
        # y = tanh(x + 1), dy/dx = 1 - tanh(x+1)^2
        var t = Tensor[dtype].d1([0.0, 1.0], requires_grad=True)
        var t_gpu = t.to_gpu()
        var t2 = t_gpu + Scalar[dtype](1)
        var out = t2.tanh()
        var loss = out.sum()
        loss.backward()
        var expected = Tensor[dtype].d1(
            [
                Float32(1) - scalar_tanh(Float32(1.0)) ** 2,
                Float32(1) - scalar_tanh(Float32(2.0)) ** 2,
            ]
        )
        assert_true(tanh_close(t.grad().as_tensor(), expected))


def test_tanh_gpu_bwd_grad_flow_two_paths() raises:
    comptime if has_accelerator():
        # loss = tanh(x).sum() + tanh(x).sum() → grad = 2*(1-tanh(x)^2)
        var t = Tensor[dtype].zeros(Shape(3), requires_grad=True)
        var t_gpu = t.to_gpu()
        var a = t_gpu.tanh()
        var b = t_gpu.tanh()
        var loss_a = a.sum()
        var loss_b = b.sum()
        var loss = loss_a + loss_b
        loss.backward()
        assert_true(
            tanh_close(
                t.grad().as_tensor(), Tensor[dtype].full(Shape(3), Float32(2.0))
            )
        )


def test_tanh_gpu_bwd_scalar_one() raises:
    comptime if has_accelerator():
        var t = Tensor[dtype].scalar(1.0, requires_grad=True)
        var t_gpu = t.to_gpu()
        var out = t_gpu.tanh()
        var loss = out.sum()
        loss.backward()
        var expected = Tensor[dtype].scalar(
            Float32(1) - scalar_tanh(Float32(1.0)) ** 2
        )
        assert_true(tanh_close(t.grad().as_tensor(), expected))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


# ============================================================
# SQUEEZE TESTS — CPU
# ============================================================


# ============================================================
# UNSQUEEZE TESTS — CPU
# ============================================================


# ============================================================
# SQUEEZE ↔ UNSQUEEZE ROUND-TRIP — CPU
# ============================================================


# ============================================================
# GPU SQUEEZE TESTS
# ============================================================


def test_squz_gpu_single_axis_dim0() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True  # (1,2,2)
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([0])
        assert_true(s.shape() == Shape(2, 2))
        assert_true(
            s.to_cpu().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]))
        )
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_squz_gpu_single_axis_dim1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True  # (2,1,2)
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([1])
        assert_true(s.shape() == Shape(2, 2))
        assert_true(
            s.to_cpu().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]))
        )
        var loss = s.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d3([[[1.0, 1.0]], [[1.0, 1.0]]]))
        )


def test_squz_gpu_single_axis_last() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True  # (2,2,1)
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([2])
        assert_true(s.shape() == Shape(2, 2))
        assert_true(
            s.to_cpu().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]))
        )
        var loss = s.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3([[[1.0], [1.0]], [[1.0], [1.0]]])
            )
        )


def test_squz_gpu_all_size1_dims() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[5.0]]], requires_grad=True)  # (1,1,1)
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([])
        assert_true(s.shape() == Shape())
        assert_true(s.to_cpu().all_close(Tensor[dtype].scalar(5.0)))
        s.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_squz_gpu_negative_axis() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True  # (2,1,2)
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([-2])
        assert_true(s.shape() == Shape(2, 2))
        var loss = s.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d3([[[1.0, 1.0]], [[1.0, 1.0]]]))
        )


def test_squz_gpu_multiple_axes() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(1, 4, 1, 6)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([0, 2])
        assert_true(s.shape() == Shape(4, 6))
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_squz_gpu_matches_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(1, 3, 1, 4)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var s_gpu = a_gpu.squeeze([0, 2])
        var s_cpu = a_copy.squeeze([0, 2])
        assert_true(s_gpu.to_cpu().all_close(s_cpu))
        var loss_gpu = s_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = s_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


def test_squz_gpu_grad_lands_on_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0, 2.0, 3.0]]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([0])
        var loss = s.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_squz_gpu_chained_squeeze() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[3.0, 6.0]]], requires_grad=True)  # (1,1,2)
        var a_gpu = a.to_gpu()
        var s1 = a_gpu.squeeze([0])  # (1,2)
        var s2 = s1.squeeze([0])  # (2,)
        var loss = s2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_squz_gpu_grad_accumulation() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0, 2.0]]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var s1 = a_gpu.squeeze([0])
        var s2 = a_gpu.squeeze([0])
        var loss1 = s1.sum()
        loss1.backward()
        a_gpu.zero_grad()
        var loss2 = s2.sum()
        loss2.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d3([[[2.0, 2.0]]])))


# ============================================================
# GPU UNSQUEEZE TESTS
# ============================================================


def test_unsquz_gpu_single_axis_front() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(0)
        assert_true(u.shape() == Shape(1, 2, 2))
        assert_true(
            u.to_cpu().all_close(Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]]))
        )
        var loss = u.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_unsquz_gpu_single_axis_middle() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(1)
        assert_true(u.shape() == Shape(2, 1, 2))
        assert_true(
            u.to_cpu().all_close(Tensor[dtype].d3([[[1.0, 2.0]], [[3.0, 4.0]]]))
        )
        var loss = u.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_unsquz_gpu_single_axis_end() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(2)
        assert_true(u.shape() == Shape(2, 2, 1))
        var loss = u.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_unsquz_gpu_negative_axis() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(-1)
        assert_true(u.shape() == Shape(2, 2, 1))
        var loss = u.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_unsquz_gpu_multiple_axes() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(0, 2)  # (1,3,1)
        assert_true(u.shape() == Shape(1, 3, 1))
        var loss = u.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_unsquz_gpu_1d_to_4d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(0, 1, 2)  # (1,1,1,2)
        assert_true(u.shape() == Shape(1, 1, 1, 2))
        var loss = u.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_unsquz_gpu_matches_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(3, 4)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var u_gpu = a_gpu.unsqueeze(0, 2)
        var u_cpu = a_copy.unsqueeze(0, 2)
        assert_true(u_gpu.to_cpu().all_close(u_cpu))
        var loss_gpu = u_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = u_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


def test_unsquz_gpu_grad_lands_on_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(0)
        var loss = u.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_unsquz_gpu_chained_unsqueeze() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u1 = a_gpu.unsqueeze(0)  # (1,2)
        var u2 = u1.unsqueeze(0)  # (1,1,2)
        var loss = u2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_unsquz_gpu_grad_accumulation() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u1 = a_gpu.unsqueeze(0)
        var u2 = a_gpu.unsqueeze(0)
        var loss1 = u1.sum()
        loss1.backward()
        var loss2 = u2.sum()
        a_gpu.zero_grad()
        loss2.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


# ============================================================
# GPU SQUEEZE ↔ UNSQUEEZE ROUND-TRIP
# ============================================================


def test_squz_unsquz_gpu_round_trip_axis0() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(0)  # (1,2,2)
        var s = u.squeeze([0])  # (2,2)
        assert_true(s.shape() == a.shape())
        assert_true(s.to_cpu().all_close(a))
        var loss = s.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_squz_unsquz_gpu_round_trip_multi() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(3, 5)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(0, 2)  # (1,3,1,5)
        var s = u.squeeze([0, 2])  # (3,5)
        assert_true(s.shape() == a.shape())
        var loss_gpu = s.sum()
        loss_gpu.backward()
        var loss_cpu = a_copy.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


# ============================================================
# MAIN
# ============================================================


# ─────────────────────────────────────────────────────────────────────────────
# 1. FORWARD PASS – CPU
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 2. BACKWARD PASS – CPU   (grad of sqrt(x) = 1 / (2*sqrt(x)))
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRADIENT FLOW VERIFICATION – CPU
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 4. FORWARD PASS – GPU
# ─────────────────────────────────────────────────────────────────────────────


def test_sqrt_fwd_gpu_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([0.0, 1.0, 4.0, 9.0, 16.0])
        var a_gpu = a_cpu.to_gpu()
        var out = a_gpu.sqrt()
        assert_true(
            out.to_cpu().all_close(Tensor[dtype].d1([0.0, 1.0, 2.0, 3.0, 4.0]))
        )


def test_sqrt_fwd_gpu_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 4.0], [9.0, 16.0]])
        var a_gpu = a_cpu.to_gpu()
        var out = a_gpu.sqrt()
        assert_true(
            out.to_cpu().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]))
        )


def test_sqrt_fwd_gpu_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(1.0, 9.0)
        var a_cpu = _tmp0.reshape(2, 2, 2)
        var a_gpu = a_cpu.to_gpu()
        var out = a_gpu.sqrt()
        var _tmp1 = Tensor[dtype].d1(
            [
                1.0,
                1.4142135,
                1.7320508,
                2.0,
                2.2360680,
                2.4494897,
                2.6457513,
                2.8284271,
            ]
        )
        var expected = _tmp1.reshape(2, 2, 2)
        assert_true(out.to_cpu().all_close[atol=1e-5](expected))


def test_sqrt_fwd_gpu_zero() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_gpu = Tensor[dtype].d1([0.0]).to_gpu()
        var out = a_gpu.sqrt()
        assert_true(out.to_cpu().all_close(Tensor[dtype].d1([0.0])))


def test_sqrt_fwd_gpu_fractional() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_gpu = Tensor[dtype].d1([0.25, 0.5, 2.0]).to_gpu()
        var out = a_gpu.sqrt()
        assert_true(
            out.to_cpu().all_close[atol=1e-5](
                Tensor[dtype].d1([0.5, 0.7071068, 1.4142135])
            )
        )


def test_sqrt_fwd_gpu_large_tensor() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # 1024 elements: values 1..1024, expected sqrt(k)
        var a_cpu = Tensor[dtype].arange(1.0, 1025.0)
        var expected = a_cpu.sqrt()  # CPU reference
        var out = a_cpu.to_gpu().sqrt().to_cpu()
        assert_true(out.all_close[atol=1e-4](expected))


# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKWARD PASS – GPU
# ─────────────────────────────────────────────────────────────────────────────


def test_sqrt_bwd_gpu_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 4.0, 9.0, 16.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var loss = a_gpu.sqrt().sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([0.5, 0.25, 0.16666667, 0.125])
            )
        )


def test_sqrt_bwd_gpu_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 4.0], [9.0, 16.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()
        var loss = a_gpu.sqrt().sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[0.5, 0.25], [0.16666667, 0.125]])
            )
        )


def test_sqrt_bwd_gpu_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(1.0, 9.0)
        var a_cpu = _tmp0.reshape(2, 2, 2)
        a_cpu.requires_grad_(True)
        var a_gpu = a_cpu.to_gpu()
        var loss = a_gpu.sqrt().sum()
        loss.backward()
        var _tmp1 = Tensor[dtype].d1(
            [
                0.5,
                0.35355339,
                0.28867513,
                0.25,
                0.22360680,
                0.20412415,
                0.18898224,
                0.17677670,
            ]
        )
        var expected = _tmp1.reshape(2, 2, 2)
        assert_true(a_cpu.grad().all_close[atol=1e-5](expected))


def test_sqrt_bwd_gpu_chain_mul() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 4.0, 9.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var s = a_gpu.sqrt()
        var scaled = s * Tensor[dtype].full_like(s, 2.0).to_gpu()
        var loss = scaled.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([1.0, 0.5, 0.33333334])
            )
        )


def test_sqrt_bwd_gpu_chain_add() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([4.0, 9.0], requires_grad=True)
        var b_cpu = Tensor[dtype].d1([1.0, 16.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var b_gpu = b_cpu.to_gpu()
        var loss = (a_gpu.sqrt() + b_gpu.sqrt()).sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([0.25, 0.16666667])
            )
        )
        assert_true(
            b_cpu.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.5, 0.125]))
        )


def test_sqrt_bwd_gpu_sqrt_of_sqrt() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 16.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var loss = a_gpu.sqrt().sqrt().sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.25, 0.03125]))
        )


def test_sqrt_bwd_gpu_large_tensor() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].arange(1.0, 1025.0)
        a_cpu.requires_grad_(True)
        var a_gpu = a_cpu.to_gpu()
        var loss = a_gpu.sqrt().sum()
        loss.backward()
        # expected: 0.5 / sqrt(k) for k in 1..1024
        var expected = (Tensor[dtype].full_like(a_cpu, 0.5)) / a_cpu.sqrt()
        assert_true(a_cpu.grad().all_close[atol=1e-4](expected))


# ─────────────────────────────────────────────────────────────────────────────
# 6. GRADIENT FLOW – GPU
# ─────────────────────────────────────────────────────────────────────────────


def test_sqrt_gradflow_gpu_no_grad_leaf() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([4.0, 9.0])  # no grad
        var b_cpu = Tensor[dtype].d1([1.0, 4.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var b_gpu = b_cpu.to_gpu()
        var loss = (a_gpu.sqrt() + b_gpu.sqrt()).sum()
        loss.backward()
        assert_true(
            b_cpu.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.5, 0.25]))
        )


def test_sqrt_gradflow_gpu_multi_use_leaf() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([4.0, 9.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var s1 = a_gpu.sqrt()
        var s2 = a_gpu.sqrt()
        var loss = (s1 + s2).sum()
        loss.backward()
        # two sqrt paths accumulate: 1/sqrt(a)
        assert_true(
            a_cpu.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([0.5, 0.33333334])
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. CPU / GPU PARITY
# ─────────────────────────────────────────────────────────────────────────────


def test_sqrt_parity_fwd_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
        var cpu_out = a_cpu.sqrt()
        var gpu_out = a_cpu.to_gpu().sqrt().to_cpu()
        assert_true(cpu_out.all_close[atol=1e-5](gpu_out))


def test_sqrt_parity_bwd_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu_leaf = Tensor[dtype].d1(
            [1.0, 2.0, 3.0, 4.0], requires_grad=True
        )
        var a_gpu = a_gpu_leaf.to_gpu()

        var loss_cpu = a_cpu.sqrt().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.sqrt().sum()
        loss_gpu.backward()

        # CPU grad has been accumulated once; GPU grad is fresh — should match
        assert_true(a_cpu.grad().all_close[atol=1e-5](a_gpu_leaf.grad()))


def test_sqrt_parity_bwd_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 4.0], [9.0, 16.0]], requires_grad=True
        )
        var a_gpu_leaf = Tensor[dtype].d2(
            [[1.0, 4.0], [9.0, 16.0]], requires_grad=True
        )
        var a_gpu = a_gpu_leaf.to_gpu()

        var loss_cpu = a_cpu.sqrt().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.sqrt().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close[atol=1e-5](a_gpu_leaf.grad()))


def test_sqrt_parity_fwd_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(1.0, 9.0)
        var a_cpu = _tmp0.reshape(2, 2, 2)
        var cpu_out = a_cpu.sqrt()
        var gpu_out = a_cpu.to_gpu().sqrt().to_cpu()
        assert_true(cpu_out.all_close[atol=1e-5](gpu_out))


def test_sqrt_parity_bwd_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(1.0, 9.0)
        var a_cpu = _tmp0.reshape(2, 2, 2)
        a_cpu.requires_grad_(True)
        var _tmp1 = Tensor[dtype].arange(1.0, 9.0)
        var a_gpu_leaf = _tmp1.reshape(2, 2, 2)
        a_gpu_leaf.requires_grad_(True)
        var a_gpu = a_gpu_leaf.to_gpu()

        var loss_cpu = a_cpu.sqrt().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.sqrt().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close[atol=1e-5](a_gpu_leaf.grad()))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN – run all tests
# ─────────────────────────────────────────────────────────────────────────────


def test_gpu_sparse_sgd_step_only_updates_specified_rows() raises:
    """Sparse step on GPU: only rows at specified indices change."""
    comptime if not has_accelerator():
        print(
            "No GPU available — skipping"
            " test_gpu_sparse_sgd_step_only_updates_specified_rows"
        )
        return
    var w_cpu = Tensor[dtype].d2(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        requires_grad=True,
    )
    var w = w_cpu.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD[dtype](params, lr=0.1)
    w.seed_grad(1.0)
    sgd.step(IntArray([0, 2]))
    var result = w.to_cpu()
    var expected = Tensor[dtype].d2(
        [[0.9, 0.9], [2.0, 2.0], [2.9, 2.9], [4.0, 4.0]],
    )
    assert_true(result.all_close(expected))


def test_gpu_sparse_sgd_with_momentum() raises:
    """Sparse momentum step on GPU: only specified rows updated."""
    comptime if not has_accelerator():
        print("No GPU available — skipping test_gpu_sparse_sgd_with_momentum")
        return
    var w_cpu = Tensor[dtype].d2(
        [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
        requires_grad=True,
    )
    var w = w_cpu.to_gpu()
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD[dtype](params, lr=0.1, momentum=0.9)
    var idx = IntArray([1])
    w.seed_grad(1.0)
    sgd.step(idx)
    var result = w.to_cpu()
    # Row 1: v = 0.9*0 + 1 = 1, w = 20 - 0.1*1 = 19.9
    # Rows 0 and 2 unchanged
    var expected = Tensor[dtype].d2(
        [[10.0, 10.0], [19.9, 19.9], [30.0, 30.0]],
    )
    assert_true(result.all_close(expected))


# Old tests
# End of old tests


# ── Helper: verify softmax properties ────────────────────────────────────────


def verify_softmax_properties_1d(result: Tensor[DType.float32]) raises:
    """Verify 1D softmax: all in [0,1] and sum == 1."""
    comptime dtype = DType.float32
    var total = Scalar[dtype](0)
    for i in range(result.shape()[0]):
        var val = result[[i]]
        assert_true(val >= Scalar[dtype](0) and val <= Scalar[dtype](1))
        total += val
    assert_true(abs(total - Scalar[dtype](1.0)) < Scalar[dtype](1e-5))


def verify_softmax_properties_2d_axis1(
    result: Tensor[DType.float32],
) raises:
    """Verify 2D softmax along axis=1: each row sums to 1."""
    comptime dtype = DType.float32
    for i in range(result.shape()[0]):
        var row_sum = Scalar[dtype](0)
        for j in range(result.shape()[1]):
            var val = result[[i, j]]
            assert_true(val >= Scalar[dtype](0) and val <= Scalar[dtype](1))
            row_sum += val
        assert_true(abs(row_sum - Scalar[dtype](1.0)) < Scalar[dtype](1e-5))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP A: CPU Softmax Forward (11 tests)
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GROUP B: CPU Softmax Backward (5 tests)
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GROUP C: CPU LogSoftmax Forward (4 tests)
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GROUP D: CPU LogSoftmax Backward (3 tests)
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GROUP E: GPU Softmax Forward (5 tests)
# ═════════════════════════════════════════════════════════════════════════════


def test_softmax_gpu_1d_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.softmax()
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close[atol=1e-5](
                Tensor[dtype].d1([0.09003057, 0.24472848, 0.66524094]),
            )
        )


def test_softmax_gpu_1d_uniform() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0]).to_gpu()
        var result = a.softmax()
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close[atol=1e-5](
                Tensor[dtype].full(Shape(4), 0.25)
            )
        )


def test_softmax_gpu_1d_numerical_stability() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1000.0, 1001.0, 1002.0]).to_gpu()
        var result = a.softmax()
        assert_true(result.is_on_gpu())
        var result_cpu = result.to_cpu()
        var total = result_cpu[[0]] + result_cpu[[1]] + result_cpu[[2]]
        assert_true(abs(total - Scalar[dtype](1.0)) < Scalar[dtype](1e-5))


def test_softmax_gpu_2d_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.softmax(axes=[1])
        assert_true(result.is_on_gpu())
        verify_softmax_properties_2d_axis1(result.to_cpu())


def test_softmax_gpu_3d_axis2() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]],
                ]
            )
            .to_gpu()
        )
        var result = a.softmax(axes=[2])
        assert_true(result.is_on_gpu())
        var result_cpu = result.to_cpu()
        for i in range(2):
            for j in range(2):
                var slice_sum = Scalar[dtype](0)
                for k in range(3):
                    slice_sum += result_cpu[[i, j, k]]
                assert_true(
                    abs(slice_sum - Scalar[dtype](1.0)) < Scalar[dtype](1e-5)
                )


# ═════════════════════════════════════════════════════════════════════════════
# GROUP F: GPU Softmax Backward (3 tests)
# ═════════════════════════════════════════════════════════════════════════════


def test_softmax_gpu_1d_backward_sum_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.softmax()
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close[atol=1e-5](Tensor[dtype].zeros(Shape(3)))
        )


def test_softmax_gpu_2d_backward_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.softmax(axes=[1])
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close[atol=1e-5](Tensor[dtype].zeros(Shape(2, 3)))
        )


def test_softmax_gpu_backward_chain() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var loss = (a_gpu.softmax() * 2.0).sum()
        loss.backward()
        assert_true(
            a.grad().all_close[atol=1e-5](Tensor[dtype].zeros(Shape(3)))
        )


# ═════════════════════════════════════════════════════════════════════════════
# GROUP G: GPU LogSoftmax Forward (3 tests)
# ═════════════════════════════════════════════════════════════════════════════


def test_log_softmax_gpu_1d_basic() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.softmax[log=True]()
        assert_true(result.is_on_gpu())
        var cpu_result = Tensor[dtype].d1([1.0, 2.0, 3.0]).softmax[log=True]()
        assert_true(result.to_cpu().all_close[atol=1e-5](cpu_result))


def test_log_softmax_gpu_1d_numerical_stability() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1000.0, 1001.0, 1002.0]).to_gpu()
        var result = a.softmax[log=True]()
        assert_true(result.is_on_gpu())
        var result_cpu = result.to_cpu()
        for i in range(3):
            assert_true(result_cpu[[i]] < Scalar[dtype](0))


def test_log_softmax_gpu_2d_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.softmax[log=True](axes=[1])
        assert_true(result.is_on_gpu())
        var result_cpu = result.to_cpu()
        for i in range(2):
            var row_sum = Scalar[dtype](0)
            for j in range(3):
                row_sum += exp(result_cpu[[i, j]])
            assert_true(abs(row_sum - Scalar[dtype](1.0)) < Scalar[dtype](1e-5))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP H: GPU LogSoftmax Backward (2 tests)
# ═════════════════════════════════════════════════════════════════════════════


def test_log_softmax_gpu_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_gpu = (
            Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True).to_gpu()
        )
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var loss_gpu = a_gpu.softmax[log=True]().sum()
        loss_gpu.backward()
        var loss_cpu = a_cpu.softmax[log=True]().sum()
        loss_cpu.backward()
        assert_true(a_gpu.grad().to_cpu().all_close[atol=1e-5](a_cpu.grad()))


def test_log_softmax_gpu_2d_backward_axis1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_gpu = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            .to_gpu()
        )
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var loss_gpu = a_gpu.softmax[log=True](axes=[1]).sum()
        loss_gpu.backward()
        var loss_cpu = a_cpu.softmax[log=True](axes=[1]).sum()
        loss_cpu.backward()
        assert_true(a_gpu.grad().to_cpu().all_close[atol=1e-5](a_cpu.grad()))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP I: Parity Tests (8 tests)
# ═════════════════════════════════════════════════════════════════════════════


def test_softmax_parity_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.softmax().all_close[atol=1e-5](a_gpu.softmax().to_cpu())
        )


def test_softmax_parity_2d_axis1_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.softmax(axes=[1]).all_close[atol=1e-5](
                a_gpu.softmax(axes=[1]).to_cpu()
            )
        )


def test_softmax_parity_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = (
            Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True).to_gpu()
        )
        var loss_cpu = (
            a_cpu.softmax() * Tensor[dtype].d1([1.0, 0.0, 0.0])
        ).sum()
        loss_cpu.backward()
        var loss_gpu = (
            a_gpu.softmax() * Tensor[dtype].d1([1.0, 0.0, 0.0]).to_gpu()
        ).sum()
        loss_gpu.backward()
        assert_true(a_cpu.grad().all_close[atol=1e-5](a_gpu.grad().to_cpu()))


def test_softmax_parity_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            .to_gpu()
        )
        var loss_cpu = a_cpu.softmax(axes=[1]).sum()
        loss_cpu.backward()
        var loss_gpu = a_gpu.softmax(axes=[1]).sum()
        loss_gpu.backward()
        assert_true(a_cpu.grad().all_close[atol=1e-5](a_gpu.grad().to_cpu()))


def test_log_softmax_parity_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.softmax[log=True]().all_close[atol=1e-5](
                a_gpu.softmax[log=True]().to_cpu()
            )
        )


def test_log_softmax_parity_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = (
            Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True).to_gpu()
        )
        var loss_cpu = a_cpu.softmax[log=True]().sum()
        loss_cpu.backward()
        var loss_gpu = a_gpu.softmax[log=True]().sum()
        loss_gpu.backward()
        assert_true(a_cpu.grad().all_close[atol=1e-5](a_gpu.grad().to_cpu()))


def test_softmax_parity_using_zero_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var loss_cpu = a_cpu.softmax().sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad().copy()
        a_cpu.zero_grad()
        var loss_gpu = a_gpu.softmax().sum()
        loss_gpu.backward()
        assert_true(cpu_grad.all_close[atol=1e-5](a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close[atol=1e-5](a_cpu.grad()))


def test_log_softmax_parity_using_zero_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var loss_cpu = a_cpu.softmax[log=True]().sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad().copy()
        a_cpu.zero_grad()
        var loss_gpu = a_gpu.softmax[log=True]().sum()
        loss_gpu.backward()
        assert_true(cpu_grad.all_close[atol=1e-5](a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close[atol=1e-5](a_cpu.grad()))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — uncomment groups one at a time to binary search LLVM lowering issue
# Start with Group A only, then add B, then C, D, E, F, G, H, I
# ═════════════════════════════════════════════════════════════════════════════


# ===----------------------------------------------------------------------=== #
# Sigmoid exhaustive tests — prefix: sig_
# Covers: forward, backward, grad flow, 0-D through 4-D, CPU & GPU
# ===----------------------------------------------------------------------=== #


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------
comptime F32 = DType.float32
comptime F64 = DType.float64


# ===----------------------------------------------------------------------=== #
# CPU – Forward pass
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------=== #
# CPU – Backward pass
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------=== #
# CPU – Gradient-flow verifications
# ===----------------------------------------------------------------------=== #


# ===----------------------------------------------------------------------=== #
# GPU – Forward pass
# ===----------------------------------------------------------------------=== #


def test_sig_gpu_scalar_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].scalar(0.0).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        assert_true(y.to_cpu().all_close[atol=1e-6](Tensor[dtype].scalar(0.5)))


def test_sig_gpu_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([0.0, 1.0, -1.0]).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        var expected = Tensor[dtype].d1([0.5, 0.7310586, 0.2689414])
        assert_true(y.to_cpu().all_close[atol=1e-5](expected))


def test_sig_gpu_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[0.0, 2.0], [-2.0, 0.0]]).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        var expected = Tensor[dtype].d2([[0.5, 0.8807970], [0.1192030, 0.5]])
        assert_true(y.to_cpu().all_close[atol=1e-5](expected))


def test_sig_gpu_3d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].zeros([2, 3, 4]).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        var expected = Tensor[dtype].full([2, 3, 4], 0.5)
        assert_true(y.to_cpu().all_close[atol=1e-6](expected))


def test_sig_gpu_4d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].zeros([2, 2, 3, 4]).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        var expected = Tensor[dtype].full([2, 2, 3, 4], 0.5)
        assert_true(y.to_cpu().all_close[atol=1e-6](expected))


def test_sig_gpu_f64_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var x = Tensor[dtype].d1([0.0, 1.0, -1.0]).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        var expected = Tensor[dtype].d1(
            [0.5, 0.7310585975646973, 0.2689414024353027]
        )
        assert_true(y.to_cpu().all_close[atol=1e-10](expected))


# ===----------------------------------------------------------------------=== #
# GPU – Backward pass
# ===----------------------------------------------------------------------=== #


def test_sig_gpu_scalar_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].scalar(0.0, requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var loss = y.sum()
        loss.backward()
        # grad accumulates on the CPU leaf
        assert_true(
            x_cpu.grad().all_close[atol=1e-6](Tensor[dtype].scalar(0.25))
        )


def test_sig_gpu_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1([0.0, 1.0, -1.0], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var loss = y.sum()
        loss.backward()
        var s = Tensor[dtype].d1([0.5, 0.7310586, 0.2689414])
        var expected = s * (Tensor[dtype].ones_like(s) - s)
        assert_true(x_cpu.grad().all_close[atol=1e-5](expected))


def test_sig_gpu_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2(
            [[0.0, 2.0], [-2.0, 0.0]], requires_grad=True
        )
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var loss = y.sum()
        loss.backward()
        var s = Tensor[dtype].d2([[0.5, 0.8807970], [0.1192030, 0.5]])
        var expected = s * (Tensor[dtype].ones_like(s) - s)
        assert_true(x_cpu.grad().all_close[atol=1e-5](expected))


def test_sig_gpu_3d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].zeros([2, 3, 4], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var loss = y.sum()
        loss.backward()
        var expected = Tensor[dtype].full([2, 3, 4], 0.25)
        assert_true(x_cpu.grad().all_close[atol=1e-6](expected))


def test_sig_gpu_4d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].zeros([2, 2, 3, 4], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var loss = y.sum()
        loss.backward()
        var expected = Tensor[dtype].full([2, 2, 3, 4], 0.25)
        assert_true(x_cpu.grad().all_close[atol=1e-6](expected))


def test_sig_gpu_f64_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var x_cpu = Tensor[dtype].d1([0.0], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var loss = y.sum()
        loss.backward()
        assert_true(
            x_cpu.grad().all_close[atol=1e-12](Tensor[dtype].d1([0.25]))
        )


# ===----------------------------------------------------------------------=== #
# GPU – Gradient-flow verifications
# ===----------------------------------------------------------------------=== #


def test_sig_gpu_grad_chained_with_add() raises:
    """Grad flows correctly through sigmoid followed by addition on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1([0.0, 1.0], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var z = y + y  # 2 * sigmoid(x)
        var loss = z.sum()
        loss.backward()
        var s = Tensor[dtype].d1([0.5, 0.7310586])
        var expected = (
            (Tensor[dtype].ones_like(s) - s) * s * Tensor[dtype].d1([2.0, 2.0])
        )
        assert_true(x_cpu.grad().all_close[atol=1e-5](expected))


def test_sig_gpu_grad_chained_with_mul() raises:
    """Grad flows correctly through sigmoid followed by multiplication on GPU.
    """
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1([0.0, -1.0], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var w = Tensor[dtype].d1([3.0, 3.0]).to_gpu()
        var z = y * w
        var loss = z.sum()
        loss.backward()
        var s = Tensor[dtype].d1([0.5, 0.2689414])
        var expected = (
            Tensor[dtype].d1([3.0, 3.0]) * s * (Tensor[dtype].ones_like(s) - s)
        )
        assert_true(x_cpu.grad().all_close[atol=1e-5](expected))


def test_sig_gpu_grad_double_sigmoid() raises:
    """Grad flows through two stacked sigmoids on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d1([0.0], requires_grad=True)
        var x = x_cpu.to_gpu()
        var y = x.sigmoid()
        var z = y.sigmoid()
        var loss = z.sum()
        loss.backward()
        var s1: Float32 = 0.5
        var s2: Float32 = 0.6224593
        var expected_val: Float32 = s2 * (1.0 - s2) * s1 * (1.0 - s1)
        assert_true(
            x_cpu.grad().all_close[atol=1e-5](Tensor[dtype].d1([expected_val]))
        )


def test_sig_gpu_grad_track_grad_false() raises:
    """T[t]rack_grad=False on GPU: output must not carry requires_grad."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([1.0, 2.0], requires_grad=True).to_gpu()
        var y = x.sigmoid[track_grad=False]()
        assert_true(not y.requires_grad)


def test_sig_gpu_cpu_forward_parity() raises:
    """CPU and GPU sigmoid produce identical results for the same input."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu = Tensor[dtype].d2([[0.5, -0.5], [1.5, -1.5]])
        var x_gpu = x_cpu.to_gpu()
        var y_cpu = x_cpu.sigmoid[track_grad=False]()
        var y_gpu = x_gpu.sigmoid[track_grad=False]()
        assert_true(y_cpu.all_close[atol=1e-6](y_gpu.to_cpu()))


def test_sig_gpu_cpu_backward_parity() raises:
    """CPU and GPU sigmoid produce identical gradients for the same input."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x_cpu_ref = Tensor[dtype].d2(
            [[0.5, -0.5], [1.5, -1.5]], requires_grad=True
        )
        var y_cpu = x_cpu_ref.sigmoid()
        var loss_cpu = y_cpu.sum()
        loss_cpu.backward()

        var x_cpu_leaf = Tensor[dtype].d2(
            [[0.5, -0.5], [1.5, -1.5]], requires_grad=True
        )
        var x_gpu = x_cpu_leaf.to_gpu()
        var y_gpu = x_gpu.sigmoid()
        var loss_gpu = y_gpu.sum()
        loss_gpu.backward()

        assert_true(x_cpu_ref.grad().all_close[atol=1e-6](x_cpu_leaf.grad()))


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


def test_gpu_scalar_add_backward_only() raises:
    comptime if has_accelerator():
        var a = Tensor[dtype].scalar(3.0, requires_grad=True)
        var b = Tensor[dtype].scalar(4.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var gpu_result = a_gpu + b_gpu  # Shape() on GPU
        assert_true(gpu_result.gradbox.value().is_on_gpu())
        assert_true(gpu_result.gradbox.value().buffer().numels() == 1)
        gpu_result.backward()


def test_gpu_scalar_add_forward_only() raises:
    comptime if has_accelerator():
        var a = Tensor[dtype].scalar(3.0, requires_grad=True)
        var b = Tensor[dtype].scalar(4.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var gpu_result = a_gpu + b_gpu  # Shape() on GPU
        var gpu_result_cpu = gpu_result.to_cpu()
        assert_true(gpu_result_cpu.all_close(Tensor[dtype].scalar(7.0)))


def test_gpu_scalar_add_backward_seed_only() raises:
    comptime if has_accelerator():
        var a = Tensor[dtype].scalar(3.0, requires_grad=True)
        var b = Tensor[dtype].scalar(4.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var gpu_result = a_gpu + b_gpu
        # Manually replicate what backward() does before graph traversal
        var shape = gpu_result.shape()
        var seed_tensor = Tensor[dtype].full(shape, Scalar[dtype](1.0))
        var seed_gpu = seed_tensor.to_gpu()
        assert_true(seed_gpu.buffer.numels() == 1)
        gpu_result.seed_grad(seed_gpu)


def test_gpu_scalar_add_backward_manual_handler() raises:
    comptime if has_accelerator():
        var a = Tensor[dtype].scalar(3.0, requires_grad=True)
        var b = Tensor[dtype].scalar(4.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var gpu_result = a_gpu + b_gpu

        # Manually seed
        var seed_tensor = Tensor[dtype].full(
            gpu_result.shape(), Scalar[dtype](1.0)
        )
        var seed_gpu = seed_tensor.to_gpu()
        gpu_result.seed_grad(seed_gpu)


def test_gpu_scalar_add_backward_update_grad() raises:
    comptime if has_accelerator():
        var a = Tensor[dtype].scalar(3.0, requires_grad=True)
        var b = Tensor[dtype].scalar(4.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var gpu_result = a_gpu + b_gpu

        var seed_tensor = Tensor[dtype].full(
            gpu_result.shape(), Scalar[dtype](1.0)
        )
        var seed_gpu = seed_tensor.to_gpu()
        gpu_result.seed_grad(seed_gpu)


def test_gpu_scalar_add_full_backward() raises:
    comptime if has_accelerator():
        var a = Tensor[dtype].scalar(3.0, requires_grad=True)
        var b = Tensor[dtype].scalar(4.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var cpu_result = a + b
        cpu_result.backward()
        var a_cpu_grad = a.grad().copy()
        var b_cpu_grad = b.grad().copy()
        a.zero_grad()
        b.zero_grad()
        var gpu_result = a_gpu + b_gpu
        gpu_result.backward()
        assert_true(a.grad().all_close(a_cpu_grad))
        assert_true(b.grad().all_close(b_cpu_grad))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
