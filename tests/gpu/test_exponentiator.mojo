from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator
from tenmo.shapes import Shape


# ── CPU Forward Tests ─────────────────────────────────────────────────────────


# ── CPU Backward Tests ────────────────────────────────────────────────────────


# ── GPU Forward Tests ─────────────────────────────────────────────────────────


def test_exp_gpu_1d_square() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0]).to_gpu()
        var result = a**2.0
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([4.0, 9.0, 16.0]))
        )


def test_exp_gpu_1d_cube() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0]).to_gpu()
        var result = a**3.0
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([8.0, 27.0, 64.0]))
        )


def test_exp_gpu_1d_fractional() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([4.0, 9.0, 16.0]).to_gpu()
        var result = a**0.5
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([2.0, 3.0, 4.0]))
        )


def test_exp_gpu_1d_zero_exponent() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0]).to_gpu()
        var result = a**0.0
        assert_true(result.is_on_gpu())
        assert_true(result.to_cpu().all_close(Tensor[dtype].ones(Shape(3))))


def test_exp_gpu_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]]).to_gpu()
        var result = a**2.0
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[4.0, 9.0], [16.0, 25.0]])
            )
        )


def test_exp_gpu_3d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[2.0, 3.0], [4.0, 5.0]], [[1.0, 2.0], [3.0, 4.0]]])
            .to_gpu()
        )
        var result = a**2.0
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d3(
                    [
                        [[4.0, 9.0], [16.0, 25.0]],
                        [[1.0, 4.0], [9.0, 16.0]],
                    ]
                )
            )
        )


def test_exp_gpu_ones_any_exponent() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu**2.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([4.0, 6.0, 8.0])))


def test_exp_gpu_1d_backward_cube() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu**3.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([12.0, 27.0, 48.0])))


def test_exp_gpu_1d_backward_fractional() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([4.0, 9.0, 16.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu**0.5
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d1([0.25, 0.16666667, 0.125]))
        )


def test_exp_gpu_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu**2.0
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[4.0, 6.0], [8.0, 10.0]]))
        )


def test_exp_gpu_3d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[2.0, 3.0], [4.0, 5.0]], [[1.0, 2.0], [3.0, 4.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu**2.0
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close(
                Tensor[dtype].d3(
                    [
                        [[4.0, 6.0], [8.0, 10.0]],
                        [[2.0, 4.0], [6.0, 8.0]],
                    ]
                )
            )
        )


def test_exp_gpu_backward_chain() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = (a_gpu**2.0) * 3.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([12.0, 18.0, 24.0])))


def test_exp_gpu_backward_chained_pow() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = (a_gpu**2.0) ** 2.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([32.0, 108.0])))


def test_exp_gpu_backward_one_exponent() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu**1.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(3))))


# ── CPU/GPU Parity Tests ──────────────────────────────────────────────────────


def test_exp_parity_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0, 5.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true((a_cpu**2.0).all_close((a_gpu**2.0).to_cpu()))


def test_exp_parity_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true((a_cpu**3.0).all_close((a_gpu**3.0).to_cpu()))


def test_exp_parity_fractional_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([4.0, 9.0, 16.0, 25.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true((a_cpu**0.5).all_close((a_gpu**0.5).to_cpu()))


def test_exp_parity_1d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = (
            Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True).to_gpu()
        )

        var loss_cpu = (a_cpu**2.0).sum()
        loss_cpu.backward()

        var loss_gpu = (a_gpu**2.0).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


def test_exp_parity_2d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[2.0, 3.0], [4.0, 5.0]], requires_grad=True
        )
        var a_gpu = (
            Tensor[dtype]
            .d2([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
            .to_gpu()
        )

        var loss_cpu = (a_cpu**2.0).sum()
        loss_cpu.backward()

        var loss_gpu = (a_gpu**2.0).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


def test_exp_parity_chain_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = (
            Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True).to_gpu()
        )

        var loss_cpu = ((a_cpu**2.0) * 3.0).sum()
        loss_cpu.backward()

        var loss_gpu = ((a_gpu**2.0) * 3.0).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


def test_exp_parity_using_zero_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # Use same tensor for both passes — zero_grad between them
        var a_cpu = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = (a_cpu**2.0).sum()
        loss_cpu.backward()
        # Save CPU grad
        var cpu_grad = a_cpu.grad().copy()

        # Clear retained grad before second pass
        a_cpu.zero_grad()

        var loss_gpu = (a_gpu**2.0).sum()
        loss_gpu.backward()

        # Now a_cpu.grad() reflects only GPU backward contribution
        assert_true(cpu_grad.all_close(a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close(a_cpu.grad()))


# ── Main ──────────────────────────────────────────────────────────────────────


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
