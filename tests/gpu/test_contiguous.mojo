from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.tensor import Tensor
from tenmo.shapes import Shape

# ============================================================
# CONTIGUOUS TESTS — CPU
# ============================================================

# ------------------------------------------------------------
# Scalar
# ------------------------------------------------------------


# ------------------------------------------------------------
# 1D
# ------------------------------------------------------------


# ------------------------------------------------------------
# 2D
# ------------------------------------------------------------


# ------------------------------------------------------------
# 3D
# ------------------------------------------------------------


# ------------------------------------------------------------
# 4D
# ------------------------------------------------------------


# ------------------------------------------------------------
# track_grad=False
# ------------------------------------------------------------


# ------------------------------------------------------------
# Grad accumulation
# ------------------------------------------------------------


# ------------------------------------------------------------
# Chained contiguous
# ------------------------------------------------------------


# ============================================================
# CONTIGUOUS TESTS — GPU
# ============================================================

# ------------------------------------------------------------
# Scalar GPU
# ------------------------------------------------------------


def test_contig_gpu_scalar_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].scalar(42.0)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        assert_true(c.shape() == Shape())
        assert_true(c.to_cpu().all_close(Tensor[dtype].scalar(42.0)))


def test_contig_gpu_scalar_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].scalar(3.0)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        c.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor[dtype].scalar(1.0)))


# ------------------------------------------------------------
# 1D GPU
# ------------------------------------------------------------


def test_contig_gpu_1d_already_contiguous() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        assert_true(c.shape() == Shape(4))
        assert_true(c.to_cpu().all_close(a))
        var loss = c.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_contig_gpu_1d_values_preserved() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([10.0, 20.0, 30.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        assert_true(c.to_cpu().all_close(Tensor[dtype].d1([10.0, 20.0, 30.0])))
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# 2D GPU
# ------------------------------------------------------------


def test_contig_gpu_2d_already_contiguous() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        assert_true(c.shape() == Shape(2, 2))
        assert_true(c.to_cpu().all_close(a))
        var loss = c.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_contig_gpu_2d_transpose_non_contiguous() raises:
    comptime if has_accelerator():
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


def test_contig_gpu_2d_matches_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(8, 16)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t_gpu = a_gpu.transpose()
        var c_gpu = t_gpu.contiguous()
        var t_cpu = a_copy.transpose()
        var c_cpu = t_cpu.contiguous()
        assert_true(c_gpu.to_cpu().all_close(c_cpu))
        var loss_gpu = c_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = c_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


def test_contig_gpu_2d_non_uniform_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var c = t.contiguous()
        var weights = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var loss = (c * weights).sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]]))
        )


# ------------------------------------------------------------
# 3D GPU
# ------------------------------------------------------------


def test_contig_gpu_3d_already_contiguous() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        assert_true(c.shape() == Shape(2, 2, 2))
        assert_true(c.to_cpu().all_close(a))
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_contig_gpu_3d_transpose_last_axes() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose(axes=[-1, -2])
        var c = t.contiguous()
        assert_true(c.shape() == Shape(2, 2, 2))
        assert_true(
            c.to_cpu().all_close(
                Tensor[dtype].d3(
                    [[[1.0, 3.0], [2.0, 4.0]], [[5.0, 7.0], [6.0, 8.0]]]
                )
            )
        )
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_contig_gpu_3d_unsqueeze_then_contiguous() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var u = a_gpu.unsqueeze(1)
        var c = u.contiguous()
        assert_true(c.shape() == Shape(2, 1, 3))
        assert_true(
            c.to_cpu().all_close(
                Tensor[dtype].d3([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
            )
        )
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_contig_gpu_3d_squeeze_then_contiguous() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True  # (2,1,2)
        )
        var a_gpu = a.to_gpu()
        var s = a_gpu.squeeze([1])
        var c = s.contiguous()
        assert_true(c.shape() == Shape(2, 2))
        assert_true(
            c.to_cpu().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]))
        )
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_contig_gpu_3d_matches_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(4, 5, 6)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t_gpu = a_gpu.transpose(axes=[-1, -2])
        var c_gpu = t_gpu.contiguous()
        var t_cpu = a_copy.transpose(axes=[-1, -2])
        var c_cpu = t_cpu.contiguous()
        assert_true(c_gpu.to_cpu().all_close(c_cpu))
        var loss_gpu = c_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = c_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


# ------------------------------------------------------------
# 4D GPU
# ------------------------------------------------------------


def test_contig_gpu_4d_already_contiguous() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(2, 3, 4, 5)
        a.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        assert_true(c.shape() == Shape(2, 3, 4, 5))
        assert_true(c.to_cpu().all_close(a))
        var loss = c.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_contig_gpu_4d_transpose_non_contiguous() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].randn(2, 3, 4, 5)
        a.requires_grad_(True)
        var a_copy = a.copy()
        a_copy.requires_grad_(True)
        var a_gpu = a.to_gpu()
        var t_gpu = a_gpu.transpose(axes=[-1, -2])
        var c_gpu = t_gpu.contiguous()
        assert_true(c_gpu.shape() == Shape(2, 3, 5, 4))
        var t_cpu = a_copy.transpose(axes=[-1, -2])
        var c_cpu = t_cpu.contiguous()
        assert_true(c_gpu.to_cpu().all_close(c_cpu))
        var loss_gpu = c_gpu.sum()
        loss_gpu.backward()
        var loss_cpu = c_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(a_copy.grad()))


# ------------------------------------------------------------
# track_grad=False GPU
# ------------------------------------------------------------


def test_contig_gpu_track_grad_false() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous[track_grad=False]()
        assert_true(c.shape() == Shape(2, 2))
        assert_true(not c.requires_grad)


# ------------------------------------------------------------
# Grad accumulation GPU
# ------------------------------------------------------------


def test_contig_gpu_grad_accumulation() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c1 = a_gpu.contiguous()
        var c2 = a_gpu.contiguous()
        var loss1 = c1.sum()
        loss1.backward()
        var loss2 = c2.sum()
        a_gpu.zero_grad()
        loss2.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[2.0, 2.0], [2.0, 2.0]]))
        )


# ------------------------------------------------------------
# Grad lands on CPU
# ------------------------------------------------------------


def test_contig_gpu_grad_lands_on_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        var loss = c.sum()
        loss.backward()
        assert_true(not a.grad().is_on_gpu())
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# Chained contiguous GPU
# ------------------------------------------------------------


def test_contig_gpu_chained_contiguous() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var c1 = t.contiguous()
        var c2 = c1.contiguous()
        assert_true(
            c2.to_cpu().all_close(Tensor[dtype].d2([[1.0, 3.0], [2.0, 4.0]]))
        )
        var loss = c2.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ------------------------------------------------------------
# CPU tensor data unchanged after GPU contiguous
# ------------------------------------------------------------


def test_contig_gpu_cpu_tensor_unchanged() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_snapshot = a.copy()
        var a_gpu = a.to_gpu()
        var c = a_gpu.contiguous()
        var loss = c.sum()
        loss.backward()
        assert_true(a.all_close(a_snapshot))


# ============================================================
# MAIN
# ============================================================


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
