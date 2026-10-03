from std.testing import assert_true, TestSuite
from tenmo.tensor import Tensor
from std.sys import has_accelerator

comptime F32 = DType.float32
comptime F64 = DType.float64


# ============================================================
# GPU round / floor — forward only, non-differentiable, no ancestry
# ============================================================


def test_gpu_round_half_to_even_ties() raises:
    comptime if has_accelerator():
        print("test_gpu_round_half_to_even_ties")
        var x = Tensor[F32].d1([2.5, 3.5, -2.5, -3.5, -0.5, 0.5])
        var y = x.to_gpu().round()
        assert_true(
            y.to_cpu().all_close(Tensor[F32].d1([2.0, 4.0, -2.0, -4.0, -0.0, 0.0]))
        )


def test_gpu_floor_basic() raises:
    comptime if has_accelerator():
        print("test_gpu_floor_basic")
        var x = Tensor[F32].d1([2.5, -2.5, 1.9, -1.9, 3.0])
        var y = x.to_gpu().floor()
        assert_true(
            y.to_cpu().all_close(Tensor[F32].d1([2.0, -3.0, 1.0, -2.0, 3.0]))
        )


def test_gpu_round_floor_f64() raises:
    comptime if has_accelerator():
        print("test_gpu_round_floor_f64")
        var x = Tensor[F64].d1([2.5, -2.5, 1.9, -1.9])
        assert_true(
            x.to_gpu().floor().to_cpu().all_close(
                Tensor[F64].d1([2.0, -3.0, 1.0, -2.0])
            )
        )
        assert_true(
            x.to_gpu().round().to_cpu().all_close(
                Tensor[F64].d1([2.0, -2.0, 2.0, -2.0])
            )
        )


def test_gpu_round_floor_matches_cpu() raises:
    comptime if has_accelerator():
        print("test_gpu_round_floor_matches_cpu")
        var x = Tensor[F32].randn(32, 32)
        var r_gpu = x.to_gpu().round().to_cpu()
        var f_gpu = x.to_gpu().floor().to_cpu()
        assert_true(r_gpu.all_close(x.round()))
        assert_true(f_gpu.all_close(x.floor()))


def test_gpu_round_floor_transposed_view() raises:
    comptime if has_accelerator():
        print("test_gpu_round_floor_transposed_view")
        var x = Tensor[F32].d2([[2.5, 3.5], [-2.5, -3.5]])
        var t = x.to_gpu().transpose(1, 0)
        assert_true(
            t.round().to_cpu().all_close(Tensor[F32].d2([[2.0, -2.0], [4.0, -4.0]]))
        )
        assert_true(
            t.floor().to_cpu().all_close(Tensor[F32].d2([[2.0, -3.0], [3.0, -4.0]]))
        )


def test_gpu_round_floor_no_ancestry() raises:
    comptime if has_accelerator():
        print("test_gpu_round_floor_no_ancestry")
        var x = Tensor[F32].d1([1.5, 2.5])
        var r = x.to_gpu().round()
        var f = x.to_gpu().floor()
        assert_true(not r.has_ancestry())
        assert_true(not f.has_ancestry())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
