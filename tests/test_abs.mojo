from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator

# ===----------------------------------------------------------------------=== #
# Abs tests — prefix: abs_
# Covers: forward, backward, grad flow, CPU & GPU, int & float
# ===----------------------------------------------------------------------=== #

comptime F32 = DType.float32
comptime F64 = DType.float64

# ===----------------------------------------------------------------------=== #
# CPU – Forward pass
# ===----------------------------------------------------------------------=== #


def test_abs_cpu_scalar_forward() raises:
    var x = Tensor[F32].scalar(-3.0)
    var y = x.abs[track_grad=False]()
    assert_true(y.all_close(Tensor[F32].scalar(3.0)))


def test_abs_cpu_1d_forward_known_values() raises:
    var x = Tensor[F32].d1([-2.0, -1.0, 0.0, 1.0, 2.0])
    var y = x.abs[track_grad=False]()
    var expected = Tensor[F32].d1([2.0, 1.0, 0.0, 1.0, 2.0])
    assert_true(y.all_close(expected))


def test_abs_cpu_2d_forward() raises:
    var x = Tensor[F32].d2([[-1.0, 2.0], [-3.0, 0.0]])
    var y = x.abs[track_grad=False]()
    var expected = Tensor[F32].d2([[1.0, 2.0], [3.0, 0.0]])
    assert_true(y.all_close(expected))


def test_abs_cpu_3d_forward() raises:
    var x = Tensor[F32].full([2, 3, 4], -2.0)
    var y = x.abs[track_grad=False]()
    var expected = Tensor[F32].full([2, 3, 4], 2.0)
    assert_true(y.all_close(expected))


def test_abs_cpu_f64_forward() raises:
    var x = Tensor[F64].d1([-1.5, 0.0, 2.5])
    var y = x.abs[track_grad=False]()
    var expected = Tensor[F64].d1([1.5, 0.0, 2.5])
    assert_true(y.all_close(expected))


def test_abs_cpu_int_forward() raises:
    var x = Tensor[DType.int32].d1([-3, -1, 0, 1, 5])
    var y = x.abs[track_grad=False]()
    var expected: List[Int32] = [3, 1, 0, 1, 5]
    for i in range(5):
        assert_true(y[i] == expected[i])


# ===----------------------------------------------------------------------=== #
# CPU – Backward pass
# ===----------------------------------------------------------------------=== #


def test_abs_cpu_scalar_backward() raises:
    var x = Tensor[F32].scalar(-3.0, requires_grad=True)
    var y = x.abs()
    var loss = y.sum()
    loss.backward()
    # d(abs(-3))/dx = -1
    assert_true(x.grad().all_close[atol=1e-6](Tensor[F32].scalar(-1.0)))


def test_abs_cpu_1d_backward() raises:
    var x = Tensor[F32].d1([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    var y = x.abs()
    var loss = y.sum()
    loss.backward()
    var expected = Tensor[F32].d1([-1.0, -1.0, 0.0, 1.0, 1.0])
    assert_true(x.grad().all_close[atol=1e-6](expected))


def test_abs_cpu_2d_backward() raises:
    var x = Tensor[F32].d2([[-1.0, 2.0], [-3.0, 0.0]], requires_grad=True)
    var y = x.abs()
    var loss = y.sum()
    loss.backward()
    var expected = Tensor[F32].d2([[-1.0, 1.0], [-1.0, 0.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected))


def test_abs_cpu_f64_backward() raises:
    var x = Tensor[F64].d1([-2.0, 0.0, 3.0], requires_grad=True)
    var y = x.abs()
    var loss = y.sum()
    loss.backward()
    var expected = Tensor[F64].d1([-1.0, 0.0, 1.0])
    assert_true(x.grad().all_close[atol=1e-12](expected))


# ===----------------------------------------------------------------------=== #
# CPU – Gradient-flow verifications
# ===----------------------------------------------------------------------=== #


def test_abs_cpu_no_grad_leaf() raises:
    var x = Tensor[F32].d1([-1.0, 2.0], requires_grad=False)
    var y = x.abs()
    var loss = y.sum()
    loss.backward()
    assert_true(not x.requires_grad)


def test_abs_cpu_chained_with_add() raises:
    var x = Tensor[F32].d1([-1.0, 2.0], requires_grad=True)
    var y = x.abs()
    var z = y + Tensor[F32].d1([10.0, 10.0])
    var loss = z.sum()
    loss.backward()
    # d(abs(-1))/dx = -1, d(abs(2))/dx = 1
    var expected = Tensor[F32].d1([-1.0, 1.0])
    assert_true(x.grad().all_close[atol=1e-6](expected))


def test_abs_cpu_requires_grad_false() raises:
    var x = Tensor[F32].d1([-1.0, 2.0], requires_grad=True)
    var y = x.abs[track_grad=False]()
    assert_true(not y.requires_grad)


# ===----------------------------------------------------------------------=== #
# GPU – Forward and backward parity
# ===----------------------------------------------------------------------=== #


def test_abs_gpu_forward() raises:
    comptime if has_accelerator():
        var x_cpu = Tensor[F32].d1([-2.0, -1.0, 0.0, 1.0, 2.0])
        var y_cpu = x_cpu.abs[track_grad=False]()
        var x_gpu = x_cpu.to_gpu()
        var y_gpu = x_gpu.abs[track_grad=False]().to_cpu()
        assert_true(y_cpu.all_close[atol=1e-6](y_gpu))


def test_abs_gpu_backward() raises:
    comptime if has_accelerator():
        var x_cpu = Tensor[F32].d1([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        var y_cpu = x_cpu.abs()
        var loss_cpu = y_cpu.sum()
        loss_cpu.backward()

        var x_gpu = Tensor[F32].d1([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True).to_gpu()
        var y_gpu = x_gpu.abs()
        var loss_gpu = y_gpu.sum()
        loss_gpu.backward()

        assert_true(x_cpu.grad().all_close[atol=1e-6](x_gpu.grad()))


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
