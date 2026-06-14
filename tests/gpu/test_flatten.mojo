from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from tenmo.shapes import Shape
from tenmo.strides import Strides
from std.sys import has_accelerator


# Old tests
# here


# --- View + Expand + Contiguous chain tests ---


# ═════════════════════════════════════════════════════════════════════════════
# CPU Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# CPU Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GPU Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_flat_gpu_1d_noop() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        var result = a.flatten()
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(4))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]))
        )


def test_flat_gpu_2d_full() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.flatten()
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(6))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            )
        )


def test_flat_gpu_3d_full() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var result = a.flatten()
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(8))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            )
        )


def test_flat_gpu_3d_start0_end1() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var result = a.flatten(0, 1)
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(4, 2))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
                )
            )
        )


def test_flat_gpu_3d_start1_end2() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
            .to_gpu()
        )
        var result = a.flatten(1, 2)
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 4))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
            )
        )


def test_flat_gpu_4d_middle() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(120)
        var _tmp1 = _tmp0.reshape(Shape(2, 3, 4, 5))
        var a = _tmp1.to_gpu()
        var result = a.flatten(1, 2)
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 12, 5))
        assert_true(result.numels() == 120)


def test_flat_gpu_values_preserved() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(6)
        var a_cpu = _tmp0.reshape(Shape(2, 3))
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.flatten()
        var result_cpu = result.to_cpu()
        for i in range(6):
            assert_true(result_cpu[[i]] == Scalar[dtype](i))


def test_flat_gpu_no_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=False)
            .to_gpu()
        )
        var result = a.flatten()
        assert_true(not result.requires_grad)


def test_flat_gpu_requires_grad_propagates() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            .to_gpu()
        )
        var result = a.flatten()
        assert_true(result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# GPU Backward Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_flat_gpu_backward_2d_full() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.flatten()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 3))))


def test_flat_gpu_backward_3d_full() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.flatten()
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


def test_flat_gpu_backward_3d_partial() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.flatten(1, 2)
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].ones(Shape(2, 2, 2))))


def test_flat_gpu_backward_chain() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.flatten() * 3.0
        var loss = result.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].full(Shape(2, 2), 3.0)))


def test_flat_gpu_backward_grad_shape() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(24)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4))
        a_cpu.requires_grad_(True)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.flatten()
        var loss = result.sum()
        loss.backward()
        assert_true(a_cpu.grad().shape() == Shape(2, 3, 4))
        assert_true(a_cpu.grad().all_close(Tensor[dtype].ones(Shape(2, 3, 4))))


def test_flat_gpu_backward_nonuniform_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.flatten()
        var weights = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0]).to_gpu()
        var loss = (result * weights).sum()
        loss.backward()
        assert_true(
            a.grad().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]))
        )


def test_flat_gpu_backward_4d_partial() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(120)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4, 5))
        a_cpu.requires_grad_(True)
        var a_gpu = a_cpu.to_gpu()
        var result = a_gpu.flatten(1, 2)
        var loss = result.sum()
        loss.backward()
        assert_true(a_cpu.grad().shape() == Shape(2, 3, 4, 5))
        assert_true(
            a_cpu.grad().all_close(Tensor[dtype].ones(Shape(2, 3, 4, 5)))
        )


# ═════════════════════════════════════════════════════════════════════════════
# CPU/GPU Parity Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_flat_parity_2d_full_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.flatten().all_close(a_gpu.flatten().to_cpu()))


def test_flat_parity_3d_partial_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(24)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4))
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.flatten(1, 2).all_close(a_gpu.flatten(1, 2).to_cpu()))


def test_flat_parity_4d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(120)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4, 5))
        var a_gpu = a_cpu.to_gpu()
        assert_true(a_cpu.flatten(0, 2).all_close(a_gpu.flatten(0, 2).to_cpu()))


def test_flat_parity_2d_backward() raises:
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

        var loss_cpu = a_cpu.flatten().sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.flatten().sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


def test_flat_parity_3d_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var _tmp0 = Tensor[dtype].arange(24)
        var a_cpu = _tmp0.reshape(Shape(2, 3, 4))
        a_cpu.requires_grad_(True)
        var _tmp1 = Tensor[dtype].arange(24)
        var _tmp2 = _tmp1.reshape(Shape(2, 3, 4))
        var a_gpu = _tmp2.to_gpu()
        a_gpu.requires_grad_(True)

        var loss_cpu = a_cpu.flatten(1, 2).sum()
        loss_cpu.backward()

        var loss_gpu = a_gpu.flatten(1, 2).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


def test_flat_parity_chain_backward() raises:
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

        var loss_cpu = (a_cpu.flatten() * 2.0).sum()
        loss_cpu.backward()

        var loss_gpu = (a_gpu.flatten() * 2.0).sum()
        loss_gpu.backward()

        assert_true(a_cpu.grad().all_close(a_gpu.grad().to_cpu()))


def test_flat_parity_using_zero_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var loss_cpu = a_cpu.flatten().sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad().copy()

        a_cpu.zero_grad()

        var loss_gpu = a_gpu.flatten().sum()
        loss_gpu.backward()

        assert_true(cpu_grad.all_close(a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close(a_cpu.grad()))


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
