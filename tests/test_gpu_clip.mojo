from std.testing import assert_true, TestSuite
from tenmo.tensor import Tensor
from std.sys import has_accelerator

comptime F32 = DType.float32


# ============================================================
# GPU Clip — forward + backward
# ============================================================


def test_gpu_clip_forward_basic() raises:
    comptime if has_accelerator():
        print("test_gpu_clip_forward_basic")
        var x = Tensor[F32].d1([-2.0, 0.5, 3.0, 7.0])
        var y = x.to_gpu().clip(0.0, 5.0)
        assert_true(
            y.to_cpu().all_close(Tensor[F32].d1([0.0, 0.5, 3.0, 5.0]))
        )


def test_gpu_clip_forward_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_gpu_clip_forward_matches_cpu")
        var x = Tensor[F32].randn(16, 32)
        var ref_out = x.clip(Scalar[F32](-0.5), Scalar[F32](0.5))
        var got = x.to_gpu().clip(Scalar[F32](-0.5), Scalar[F32](0.5))
        assert_true(got.to_cpu().all_close(ref_out))


def test_gpu_clip_grad_contiguous() raises:
    comptime if has_accelerator():
        print("test_gpu_clip_grad_contiguous")
        var x = Tensor[F32].d1([-2.0, 2.0, 7.0], requires_grad=True)
        var y = x.to_gpu().clip(0.0, 5.0).sum()
        y.backward()
        assert_true(x.grad().all_close(Tensor[F32].d1([0.0, 1.0, 0.0])))


def test_gpu_clip_grad_transposed_view() raises:
    comptime if has_accelerator():
        print("test_gpu_clip_grad_transposed_view")
        var x = Tensor[F32].d2([[1.0, 5.0], [2.0, 6.0]], requires_grad=True)
        var t = x.to_gpu().transpose(1, 0)
        var y = t.clip(2.0, 5.0)
        assert_true(
            y.to_cpu().all_close(Tensor[F32].d2([[2.0, 2.0], [5.0, 5.0]]))
        )
        var loss = y.sum()
        loss.backward()
        assert_true(
            x.grad().all_close(Tensor[F32].d2([[0.0, 1.0], [1.0, 0.0]]))
        )


def test_gpu_clip_grad_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_gpu_clip_grad_matches_cpu")
        var a = Tensor[F32].randn(8, 16)
        a.requires_grad_(True)
        var b = a.copy()
        b.requires_grad_(True)
        var y_gpu = a.to_gpu().clip(Scalar[F32](-0.5), Scalar[F32](0.5))
        var loss_gpu = y_gpu.sum()
        loss_gpu.backward()
        var y_cpu = b.clip(Scalar[F32](-0.5), Scalar[F32](0.5))
        var loss_cpu = y_cpu.sum()
        loss_cpu.backward()
        assert_true(a.grad().all_close(b.grad()))


def test_gpu_clip_weighted_upstream() raises:
    comptime if has_accelerator():
        print("test_gpu_clip_weighted_upstream")
        var x = Tensor[F32].d2([[1.0, 5.0], [2.0, 6.0]], requires_grad=True)
        var t = x.to_gpu().transpose(1, 0)
        var w = Tensor[F32].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var loss = (t.clip(2.0, 5.0) * w).sum()
        loss.backward()
        assert_true(
            x.grad().all_close(Tensor[F32].d2([[0.0, 3.0], [2.0, 0.0]]))
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
