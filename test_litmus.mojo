from tenmo import Tensor
from testing import assert_true
from sys import has_accelerator
from shapes import Shape
from gradbox import Gradbox

comptime dtype = DType.float32

fn main() raises:
    test_cpu_grad_flow()
    @parameter
    if has_accelerator():
        test_gpu_grad_flow()

fn test_cpu_grad_flow() raises:
    print("=== Test : Backward grad flow CPU ===")
    var A = Tensor[dtype].arange(9 * 30, requires_grad=True)
    var B = Tensor[dtype].arange(30 * 5)

    var A_reshaped = A.reshape(Shape(1, 9, 30))
    var B_reshaped = B.reshape(Shape(30, 5))

    var C = A_reshaped.matmul(B_reshaped)

    C.backward()
    var A_grad = A.grad().copy()

    var grad_out = Gradbox[dtype].full(C.shape(), 1)
    # ===== GRADIENT FOR A: dL/dA = grad_out × B^T =====
    var A_grad_expected = grad_out.matmul(B_reshaped.transpose(-1, -2))
    A_grad_expected = A_grad_expected.reshape(Shape(1 * 9 * 30))

    assert_true(A_grad.all_close(A_grad_expected))

    print("PASSED: CPU grad flow")


fn test_gpu_grad_flow() raises:
    print("=== Test : Backward grad flow GPU ===")
    var A = Tensor[dtype].arange(9 * 30, requires_grad=True)
    var B = Tensor[dtype].arange(30 * 5)

    var A_reshaped = A.reshape(Shape(1, 9, 30))
    var B_reshaped = B.reshape(Shape(30, 5))

    var A_gpu = A_reshaped.to_gpu()
    var B_gpu = B_reshaped.to_gpu()
    var A_gpu_reshaped = A_gpu.reshape(Shape(9, 1, 30))
    var C_gpu = A_gpu_reshaped.matmul(B_gpu)

    C_gpu.backward()
    var A_grad = A.grad().copy()

    var grad_out = Gradbox[dtype].full(C_gpu.shape(), 1)
    # ===== GRADIENT FOR A: dL/dA = grad_out × B^T =====
    var A_grad_expected = grad_out.matmul(B_reshaped.transpose(-1, -2))
    A_grad_expected = A_grad_expected.reshape(Shape(9 * 30))

    assert_true(A_grad.all_close(A_grad_expected))

    print("PASSED: GPU backward flow")
