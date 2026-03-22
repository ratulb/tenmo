from tenmo import Tensor
from testing import assert_true
from sys import has_accelerator
from shapes import Shape
from mnemonics import AddTensor

comptime dtype = DType.float32

fn test_cpu_add_scalar_tensor_result() raises:
    print("test_cpu_add_scalar_tensor_result")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var b = Tensor[dtype].scalar(4.0, requires_grad=True)
    var result = a + b  # Shape()
    result.backward()
    #a.grad().print()
    #b.grad().print()
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

fn test_gpu_scalar_add_backward_seed_only() raises:
    print("test_gpu_scalar_add_backward_seed_only")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var b = Tensor[dtype].scalar(4.0, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var gpu_result = a_gpu + b_gpu
    # Manually replicate what backward() does before graph traversal
    var shape = gpu_result.shape()
    print("seed shape:", shape.__str__())
    var seed_tensor = Tensor[dtype].full(shape, Scalar[dtype](1.0))
    print("seed_tensor created, shape:", seed_tensor.shape().__str__())
    print("seed_tensor value:", seed_tensor.item())
    var seed_gpu = seed_tensor.to_gpu()
    print("seed_gpu created, is_on_gpu:", seed_gpu.is_on_gpu())
    print("seed_gpu numels:", seed_gpu.buffer.numels())
    gpu_result.seed_grad(seed_gpu)
    print("seed_grad done")
    print("gpu_result gradbox after seed:")
    gpu_result.grad().print()
    print("passed")

fn test_gpu_scalar_add_backward_manual_handler() raises:
    print("test_gpu_scalar_add_backward_manual_handler")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var b = Tensor[dtype].scalar(4.0, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var gpu_result = a_gpu + b_gpu

    # Manually seed
    var seed_tensor = Tensor[dtype].full(gpu_result.shape(), Scalar[dtype](1.0))
    var seed_gpu = seed_tensor.to_gpu()
    gpu_result.seed_grad(seed_gpu)
    print("gradbox seeded")

    # Manually fire backward handler
    print("firing backward fn")
    var results = gpu_result.backward_fn()(gpu_result)
    print("backward fn returned, num results:", len(results))
    for i in range(len(results)):
        print("result", i, "grad:")
        results[i][1].print()
    print("passed")


fn test_gpu_scalar_add_backward_update_grad() raises:
    print("test_gpu_scalar_add_backward_update_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var b = Tensor[dtype].scalar(4.0, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var gpu_result = a_gpu + b_gpu

    var seed_tensor = Tensor[dtype].full(gpu_result.shape(), Scalar[dtype](1.0))
    var seed_gpu = seed_tensor.to_gpu()
    gpu_result.seed_grad(seed_gpu)

    var results = gpu_result.backward_fn()(gpu_result)
    print("backward fn returned")

    # result[0] targets a_gpu, result[1] targets b_gpu
    var target_0 = results[0][0]
    var grad_0 = results[0][1]
    var target_1 = results[1][0]
    var grad_1 = results[1][1]

    print("target_0 is_on_gpu:", target_0.is_on_gpu())
    print("target_0 gradbox is_on_gpu:", target_0.gradbox[].is_on_gpu())
    print("grad_0 is_on_gpu:", grad_0.is_on_gpu())
    print("calling update_grad on target_0")
    target_0.update_grad[AddTensor](grad_0)
    print("update_grad target_0 done")
    print("target_0 gradbox after update:")
    target_0.grad().print()

    print("target_1 is_on_gpu:", target_1.is_on_gpu())
    print("target_1 gradbox is_on_gpu:", target_1.gradbox[].is_on_gpu())
    print("grad_1 is_on_gpu:", grad_1.is_on_gpu())
    print("calling update_grad on target_1")
    target_1.update_grad[AddTensor](grad_1)
    print("update_grad target_1 done")
    print("target_1 gradbox after update:")
    target_1.grad().print()

    print("passed")

fn test_gpu_scalar_add_full_backward() raises:
    print("test_gpu_scalar_add_full_backward")
    comptime dtype = DType.float32
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
    print("a.grad():")
    a.grad().print()
    print("b.grad():")
    b.grad().print()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))
    print("passed")

fn main() raises:
    test_cpu_add_scalar_tensor_result()
    @parameter
    if has_accelerator():
        test_gpu_scalar_add_forward_only()
        test_gpu_scalar_add_backward_only()
        test_gpu_scalar_add_backward_seed_only()
        test_gpu_scalar_add_backward_manual_handler()
        test_gpu_scalar_add_backward_update_grad()
        test_gpu_scalar_add_full_backward()
