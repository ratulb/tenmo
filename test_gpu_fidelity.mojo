from tenmo import Tensor
from matmul_kernel import MatmulNdGpu
from testing import assert_true
from sys import has_accelerator
from intarray import IntArray
from shapes import Shape

comptime dtype = DType.float32


fn test_gpu_transfer_fidelity() raises:
    print("=== Test 1: GPU transfer fidelity ===")
    var B = Tensor[dtype].rand(80, 20)
    var B_gpu = B.to_gpu()
    var B_back = B_gpu.to_cpu()
    assert_true(B.all_close(B_back))
    print("PASSED: B == B_gpu.to_cpu()")


fn test_ancestry_storage_fidelity() raises:
    print("=== Test 2: Ancestry storage fidelity ===")
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var A_gpu = A.to_gpu()
    var B = Tensor[dtype].rand(80, 20)
    var B_gpu = B.to_gpu()
    var C_gpu = A_gpu.matmul(B_gpu)
    var B_from_ancestry = C_gpu.ancestry().get(1)
    var B_ancestry_back = B_from_ancestry.to_cpu()
    assert_true(B.all_close(B_ancestry_back))
    print("PASSED: B from ancestry == original B")


fn test_forward_matmul_fidelity() raises:
    print("=== Test 3: Forward matmul fidelity ===")
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var B = Tensor[dtype].rand(80, 20)
    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()
    var C_cpu = A.matmul(B)
    var C_gpu = A_gpu.matmul(B_gpu)
    assert_true(C_cpu.all_close(C_gpu.to_cpu()))
    print("PASSED: CPU matmul == GPU matmul")

fn test_backward_grad_A_fidelity() raises:
    print("=== Test 4: Backward grad_A fidelity ===")
    var AA = Tensor[dtype].arange(9 * 30, requires_grad=True)
    var A = AA.reshape(9, 30)
    var B = Tensor[dtype].arange(30 * 5).reshape(30, 5)

    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()
    var C_cpu = A.matmul(B)
    var C_gpu = A_gpu.matmul(B_gpu)

    C_cpu.backward()

    assert_true(A_gpu.grad().all_close(Tensor[dtype].zeros(Shape(9, 30))))

    C_gpu.backward()

    A_gpu.grad().print()
    assert_true(AA.grad().reshape(Shape(9, 30)).all_close(A_gpu.grad() * 2))
    print("PASSED: GPU backward grad_A == CPU backward grad_A")


fn test_transposed_matmul_fidelity() raises:
    print("=== Test 5: Transposed matmul fidelity ===")
    var B = Tensor[dtype].rand(80, 20)
    var B_gpu = B.to_gpu()
    var BT_cpu = B.transpose(axes=IntArray(-1, -2))
    var BT_gpu = B_gpu.buffer.transpose(axes=IntArray(-1, -2))

    var grad_out = Tensor[dtype].ones(9, 20)
    var grad_out_gpu = grad_out.to_gpu()

    var grad_A_cpu = grad_out.matmul(BT_cpu)

    var grad_A_ndb = MatmulNdGpu[dtype].launch[tile_size=32](
        grad_out_gpu.buffer, BT_gpu
    )
    var grad_A_GPU = Tensor[dtype](grad_A_ndb^)
    var grad_A_gpu = grad_A_GPU.to_cpu()

    print("CPU grad_A row 0:")
    for i in range(min(8, grad_A_cpu.shape()[-1])):
        print(grad_A_cpu.buffer[[0, i]], end=" ")
    print()
    print("GPU grad_A row 0:")
    for i in range(min(8, grad_A_gpu.shape()[-1])):
        print(grad_A_gpu.buffer[[0, i]], end=" ")
    print()

    assert_true(grad_A_cpu.all_close(grad_A_gpu))
    print("PASSED: GPU transposed matmul == CPU")


fn test_ancestry_transposed_matmul_fidelity() raises:
    print("=== Test 6: B from ancestry transposed matmul ===")
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var B = Tensor[dtype].rand(80, 20)
    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()
    var C_gpu = A_gpu.matmul(B_gpu)

    var grad_out = Tensor[dtype].ones(9, 20)
    var grad_out_gpu = grad_out.to_gpu()

    var BT_cpu = B.transpose(axes=IntArray(-1, -2))
    var grad_A_cpu = grad_out.matmul(BT_cpu)

    var B_anc = C_gpu.ancestry().get(1)
    var BT_anc = B_anc.buffer.transpose(axes=IntArray(-1, -2))

    var grad_A_anc_ndb = MatmulNdGpu[dtype].launch[tile_size=32](
        grad_out_gpu.buffer, BT_anc
    )
    var grad_A_ANC = Tensor[dtype](grad_A_anc_ndb^)
    var grad_A_anc = grad_A_ANC.to_cpu()

    print("CPU grad_A row 0:")
    for i in range(min(8, grad_A_cpu.shape()[-1])):
        print(grad_A_cpu.buffer[[0, i]], end=" ")
    print()
    print("GPU grad_A from ancestry row 0:")
    for i in range(min(8, grad_A_anc.shape()[-1])):
        print(grad_A_anc.buffer[[0, i]], end=" ")
    print()

    assert_true(grad_A_cpu.all_close(grad_A_anc))
    print("PASSED: B from ancestry transposed matmul == CPU")

from common_utils import now
fn main() raises:
    var A_parent = Tensor[dtype].arange(9 * 30, requires_grad=True)
    var A = A_parent.reshape(1, 9, 30, requires_grad=True)
    var B = Tensor[dtype].arange(30 * 5).reshape(30, 5)

    var C_cpu = A.matmul(B)
    C_cpu.backward()

    A_parent.grad().print()

    @parameter
    if not has_accelerator():
        print("No GPU available — skipping tests")
        return
    else:
        test_gpu_transfer_fidelity()
        test_ancestry_storage_fidelity()
        test_forward_matmul_fidelity()
        test_ancestry_transposed_matmul_fidelity()
        test_transposed_matmul_fidelity()
        test_backward_grad_A_fidelity()

        print("\n=== ALL TESTS PASSED ===")

