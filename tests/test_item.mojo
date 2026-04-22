from tenmo.tensor import Tensor
from std.testing import assert_true
from std.sys import has_accelerator

fn test_item_cpu_scalar_tensor() raises:
    print("test_item_cpu_scalar_tensor")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(42.0)
    assert_true(a.item() == 42.0)
    print("passed")


fn test_item_cpu_1d_tensor() raises:
    print("test_item_cpu_1d_tensor")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([7.0])
    assert_true(a.item() == 7.0)
    print("passed")


fn test_item_cpu_gradbox_scalar() raises:
    print("test_item_cpu_gradbox_scalar")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var b = Tensor[dtype].scalar(4.0, requires_grad=True)
    var c = a + b
    c.backward()
    # c.gradbox is seeded with 1.0 — shape ()
    assert_true(c.gradients()[].item() == 1.0)
    print("passed")


fn test_item_gpu_scalar_tensor() raises:
    print("test_item_gpu_scalar_tensor")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(42.0)
    var a_gpu = a.to_gpu()
    assert_true(a_gpu.item() == 42.0)
    print("passed")


fn test_item_gpu_1d_tensor() raises:
    print("test_item_gpu_1d_tensor")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([7.0])
    var a_gpu = a.to_gpu()
    assert_true(a_gpu.item() == 7.0)
    print("passed")


fn test_item_gpu_gradbox_scalar() raises:
    print("test_item_gpu_gradbox_scalar")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b = Tensor[dtype].scalar(4.0, requires_grad=True)
    var b_gpu = b.to_gpu()
    var c_gpu = a_gpu + b_gpu
    # Manually seed gradbox
    var seed_cpu = Tensor[dtype].full(c_gpu.shape(), Scalar[dtype](1.0))
    var seed =seed_cpu.to_gpu()
    c_gpu.seed_grad(seed)
    # item() on GPU gradbox
    assert_true(c_gpu.gradients()[].item() == 1.0)
    print("passed")


fn test_item_gpu_gradbox_after_backward() raises:
    print("test_item_gpu_gradbox_after_backward")
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b = Tensor[dtype].scalar(4.0)
    var b_gpu = b.to_gpu()
    var c_gpu = a_gpu + b_gpu
    c_gpu.backward()
    # a.gradbox should be 1.0 after backward
    assert_true(a.grad().item() == 1.0)
    print("passed")


fn test_item_gpu_sum_result() raises:
    print("test_item_gpu_sum_result")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var a_gpu = a.to_gpu()
    var s = a_gpu.sum()  # Shape() scalar
    assert_true(s.item() == 10.0)
    print("passed")


fn test_item_gpu_mean_result() raises:
    print("test_item_gpu_mean_result")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var a_gpu = a.to_gpu()
    var m = a_gpu.mean()  # Shape() scalar
    assert_true(m.item() == 2.5)
    print("passed")


fn main() raises:
    # CPU first
    test_item_cpu_scalar_tensor()
    test_item_cpu_1d_tensor()
    test_item_cpu_gradbox_scalar()

    comptime if not has_accelerator():
        print("No GPU — skipping GPU item tests")
        return

    test_item_gpu_scalar_tensor()
    test_item_gpu_1d_tensor()
    test_item_gpu_gradbox_scalar()
    test_item_gpu_gradbox_after_backward()
    test_item_gpu_sum_result()
    test_item_gpu_mean_result()

    print("\n=== ALL ITEM TESTS PASSED ===")
