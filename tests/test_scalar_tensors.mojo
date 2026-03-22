from tenmo import Tensor
from testing import assert_true
from sys import has_accelerator
from shapes import Shape

comptime dtype = DType.float32

fn test_cpu_add_scalar_tensor_result() raises:
    print("test_cpu_add_scalar_tensor_result")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var b = Tensor[dtype].scalar(4.0, requires_grad=True)
    var result = a + b  # Shape()
    result.backward()
    a.grad().print()
    b.grad().print()
    print("passed")

fn test_gpu_scalar_add_backward_only() raises:
    print("test_gpu_scalar_add_backward_only")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var b = Tensor[dtype].scalar(4.0, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var gpu_result = a_gpu + b_gpu  # Shape() on GPU
    print("gpu_result requires_grad:", gpu_result.requires_grad)
    print("gpu_result has_backward_fn:", gpu_result.has_backward_fn())
    print("gpu_result gradbox is_on_gpu:", gpu_result.gradbox[].is_on_gpu())
    print("gpu_result gradbox numels:", gpu_result.gradbox[].buffer.numels())
    gpu_result.backward()
    print("a.grad():")
    a.grad().print()
    print("b.grad():")
    b.grad().print()
    print("passed")

fn test_gpu_scalar_add_forward_only() raises:
    print("test_gpu_scalar_add_forward_only")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var b = Tensor[dtype].scalar(4.0, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var gpu_result = a_gpu + b_gpu  # Shape() on GPU
    print("gpu_result shape:", gpu_result.shape().__str__())
    print("gpu_result is_on_gpu:", gpu_result.is_on_gpu())
    var gpu_result_cpu = gpu_result.to_cpu()
    print("gpu_result value:")
    gpu_result_cpu.print()
    assert_true(gpu_result_cpu.all_close(Tensor[dtype].scalar(7.0)))
    print("passed")

fn main() raises:
    #test_cpu_add_scalar_tensor_result()
    test_gpu_scalar_add_forward_only()
    test_gpu_scalar_add_backward_only()
