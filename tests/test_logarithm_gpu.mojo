from tenmo.tensor import Tensor
from std.testing import assert_true, assert_false, TestSuite
from tenmo.shared.constants import Epsilon
from std.utils.numerics import isinf, isnan
from std.math import log
from std.sys import has_accelerator
from std.math import log
from tenmo.shared.shapes import Shape


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


# ── GPU Forward Tests ─────────────────────────────────────────────────────────


def test_log_gpu_1d_basic_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.log()
        assert_true(result.is_on_gpu())
        var expect = Tensor[dtype].d1(
            [log(Float32(1.0)), log(Float32(2.0)), log(Float32(3.0))]
        )
        assert_true(result.to_cpu().all_close(expect))


def test_log_gpu_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var result = a.log()
        assert_true(result.is_on_gpu())
        var expect = Tensor[dtype].d2(
            [
                [log(Float32(1.0)), log(Float32(2.0))],
                [log(Float32(3.0)), log(Float32(4.0))],
            ]
        )
        assert_true(result.to_cpu().all_close(expect))


def test_log_gpu_3d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var result = a.log()
        assert_true(result.is_on_gpu())
        var expect = Tensor[dtype].d3(
            [
                [
                    [log(Float32(1.0)), log(Float32(2.0))],
                    [log(Float32(3.0)), log(Float32(4.0))],
                ],
                [
                    [log(Float32(5.0)), log(Float32(6.0))],
                    [log(Float32(7.0)), log(Float32(8.0))],
                ],
            ]
        )
        assert_true(result.to_cpu().all_close(expect))


def test_log_gpu_ones_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].ones(Shape(4)).to_gpu()
        var result = a.log()
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close(Tensor[dtype].zeros(Shape(4))))


def test_log_gpu_epsilon_clamping() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([0.0, -1.0, 1.0]).to_gpu()
        var result = a.log()
        var result_cpu = result.to_cpu()
        var expected_clamped = log(Float32(1e-7))
        assert_true(result_cpu[[0]] == expected_clamped)
        assert_true(result_cpu[[1]] == expected_clamped)
        assert_true(result_cpu[[2]] == Float32(0.0))


def test_log_gpu_large_values() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([100.0, 1000.0, 10000.0]).to_gpu()
        var result = a.log()
        var expect = Tensor[dtype].d1(
            [log(Float32(100.0)), log(Float32(1000.0)), log(Float32(10000.0))]
        )
        assert_true(result.to_cpu().all_close(expect))


# ── GPU Backward Tests ────────────────────────────────────────────────────────
def test_log_gpu_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 0.5, 0.25])))


def test_log_gpu_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [4.0, 8.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log()
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 0.5], [0.25, 0.125]]))
        )


def test_log_gpu_3d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [4.0, 8.0]], [[1.0, 4.0], [2.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.log()
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [
                        [[1.0, 0.5], [0.25, 0.125]],
                        [[1.0, 0.25], [0.5, 0.125]],
                    ]
                )
            )
        )


def test_log_gpu_backward_chain() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log() * 2.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 1.0, 0.5])))


def test_log_gpu_backward_epsilon_clamping() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([0.0, 1.0, 2.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad()[[0]] == Float32(1.0) / Float32(1e-7))
        assert_true(a.grad()[[1]] == Float32(1.0))
        assert_true(a.grad()[[2]] == Float32(0.5))


def test_log_gpu_backward_chained_with_exp() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # log(exp(x)) = x, grad should be 1.0 everywhere
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.exp().log()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(3))))


def test_log_gpu_backward_custom_epsilon() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([0.0, 1.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.log[epsilon=Scalar[dtype](1e-6)]()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad()[[0]] == Float32(1.0) / Float32(1e-6))
        assert_true(a.grad()[[1]] == Float32(1.0))
        assert_true(a.grad()[[2]] == Float32(0.25))


# ── CPU/GPU Parity Tests ──────────────────────────────────────────────────────


def test_log_parity_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])
        var a_gpu = a_cpu.to_gpu()
        var result_cpu = a_cpu.log()
        var result_gpu = a_gpu.log()
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


def test_log_parity_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a_cpu.to_gpu()
        var result_cpu = a_cpu.log()
        var result_gpu = a_gpu.log()
        assert_true(result_cpu.all_close(result_gpu.to_cpu()))


def test_log_parity_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 4.0, 8.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.log().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.log().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


def test_log_parity_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0], [4.0, 8.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.log().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.log().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


def test_log_parity_epsilon_clamping() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([0.0, -1.0, 1.0, 2.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.log().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.log().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))


def test_log_parity_chain_exp() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.exp().log().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.exp().log().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(2 * a_gpu.grad().to_cpu()))
