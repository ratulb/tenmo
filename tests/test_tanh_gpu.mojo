from tenmo.tensor import Tensor
from std.utils.numerics import isinf, isnan
from tenmo.shared.shapes import Shape
from tenmo.shared.intarray import IntArray
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from std.math import tanh as scalar_tanh, abs as scalar_abs

comptime dtype = DType.float32
comptime tol = Float32(1e-4)


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
            assert_true(
                data[unsafe_offset=i] >= Float32(-1.0)
                and data[unsafe_offset=i] <= Float32(1.0)
            )


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
        var grad_cpu = t.grad().as_tensor().clone()
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
