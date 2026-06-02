from tenmo import Tensor
from tenmo.tensor import i, s
from std.sys import has_accelerator

def main() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32

        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var t = a_gpu.transpose()
        var w = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var prod = t * w
        var loss = prod.sum()
        loss.backward()

        print("a.grad().is_on_gpu:", a.grad().is_on_gpu())
        print("a.grad() shape:", a.grad().shape())
        print("a.grad() as_tensor:", a.grad().as_tensor())
        print("Expected: [[1.0, 3.0], [2.0, 4.0]]")

        # Also check a_gpu grad
        print("a_gpu.grad() shape:", a_gpu.grad().shape())
        print("a_gpu.grad() as_tensor (cpu):", a_gpu.grad().as_tensor().to_cpu())
    else:
        print("No GPU")
