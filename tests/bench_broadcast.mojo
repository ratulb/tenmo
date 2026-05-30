from std.time import perf_counter_ns
from tenmo.tensor import Tensor
from tenmo.shapes import Shape


def measure(
    label: String,
    ref a: Tensor[DType.float32],
    ref b: Tensor[DType.float32],
    num_iters: Int,
):
    for _ in range(3):
        _ = a + b

    var t0 = perf_counter_ns()
    for _ in range(num_iters):
        _ = a + b
    var t1 = perf_counter_ns()
    var avg_ms = Float64(t1 - t0) / 1_000_000.0 / Float64(num_iters)
    print(label, avg_ms, "ms")


def main() raises:
    comptime dtype = DType.float32

    print("=" * 60)
    print("BROADCAST PERFORMANCE BASELINE")
    print("=" * 60)

    var B = 256
    var N = 1024
    var M = 512

    print("\n--- Pattern 1: (B,N) + (N,)  [row broadcast, bias add] ---")
    var A1 = Tensor[dtype].rand(Shape([B, N]))
    var B1 = Tensor[dtype].rand(Shape([N]))
    measure("  (256,1024)+(1024,):", A1, B1, 500)

    print("\n--- Pattern 2: (B,N) + (B,1)  [column broadcast] ---")
    var A2 = Tensor[dtype].rand(Shape([B, N]))
    var B2 = Tensor[dtype].rand(Shape([B, 1]))
    measure("  (256,1024)+(256,1):", A2, B2, 500)

    print("\n--- Pattern 3: (M,1) + (1,N)  [outer broadcast] ---")
    var A3 = Tensor[dtype].rand(Shape([M, 1]))
    var B3 = Tensor[dtype].rand(Shape([1, N]))
    measure("  (512,1)+(1,1024):", A3, B3, 500)

    print("\n--- Pattern 4: (B,C,H,W) + (C,1,1)  [channel bias] ---")
    var C = 16
    var H = 16
    var W = 16
    var A4 = Tensor[dtype].rand(Shape([64, C, H, W]))
    var B4 = Tensor[dtype].rand(Shape([C, 1, 1]))
    measure("    (64,16,16,16)+(16,1,1):", A4, B4, 100)

    print("\n--- Pattern 5: (B,N) + (B,N)  [same shape, no broadcast — reference] ---")
    var A5 = Tensor[dtype].rand(Shape([B, N]))
    var B5 = Tensor[dtype].rand(Shape([B, N]))
    measure("  (256,1024)+(256,1024):", A5, B5, 500)

    print("\n--- Pattern 6: Medium batch bias ---")
    var L = 500
    var A6 = Tensor[dtype].rand(Shape([L, N]))
    var B6 = Tensor[dtype].rand(Shape([N]))
    measure("   (500,1024)+(1024,):", A6, B6, 200)

    print("\n========================================")
