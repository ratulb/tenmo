from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from tenmo.shapes import Shape
from tenmo.mnemonics import AddTensor
from std.sys import has_accelerator

comptime dtype = DType.float32


def test_cpu_add_scalar_tensor_result() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var b = Tensor[dtype].scalar(4.0, requires_grad=True)
    var result = a + b  # Shape()
    result.backward()
    assert_true(
        a.grad() == Tensor[dtype].scalar(1)
        and b.grad() == Tensor[dtype].scalar(1)
    )


def test_gpu_scalar_add_backward_only() raises:
    comptime if has_accelerator():
        var a = Tensor[dtype].scalar(3.0, requires_grad=True)
        var b = Tensor[dtype].scalar(4.0, requires_grad=True)
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var gpu_result = a_gpu + b_gpu  # Shape() on GPU
        assert_true(gpu_result.gradbox.unsafe_value()[].is_on_gpu())
        assert_true(gpu_result.gradbox.unsafe_value()[].buffer.numels() == 1)
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
