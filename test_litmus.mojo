from tenmo import Tensor
from testing import assert_true
from sys import has_accelerator
from shapes import Shape

comptime dtype = DType.float32

fn main() raises:
    @parameter
    if not has_accelerator():
        print("No GPU available — skipping tests")
        return
    else:

        print("=== Test 4: Backward grad_A fidelity ===")
        var AA = Tensor[dtype].arange(9 * 30, requires_grad=True)
        var BB = Tensor[dtype].arange(30 * 5)

        var A = AA.reshape(Shape(9, 30))
        var B = BB.reshape(Shape(30, 5))

        var A_gpu = A.to_gpu()
        var B_gpu = B.to_gpu()
        var C_cpu = A.matmul(B)
        var C_gpu = A_gpu.matmul(B_gpu)

        C_cpu.backward()
        var AA_cpu_grad = AA.grad().copy().reshape(Shape(9, 30))
        AA.zero_grad()

        assert_true(A_gpu.grad().all_close(Tensor[dtype].zeros(Shape(9, 30))))

        C_gpu.backward()

        assert_true(AA.grad().all_close(AA_cpu_grad), "Grad propagation failed")


        print("PASSED: GPU backward grad_A == CPU backward grad_A")


