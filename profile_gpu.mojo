"""GPU Profiling: bias_add dispatch + full forward + backward timing.

Usage on kaggle:
    cp arithmetic_ops_fixed_complete.mojo tenmo/kernels/binary_ops_kernel.mojo
    cp tenmo/tensor.mojo tenmo/tensor.mojo  # with backward profiling
    ./fire.sh profile_gpu.mojo

Tests:
  1. bias_add at each layer shape (sync=True), with warmup
  2. All 5 dispatch paths explicitly exercised
  3. matmul + relu baselines
  4. Full forward (sync=False) + backward with per-op breakdown
"""

from tenmo.tensor import Tensor
from tenmo.device import GPU
from std.time import perf_counter_ns
from std.sys import has_accelerator
from tenmo.net import Linear, ReLU, Sequential
from tenmo.crossentropy import CrossEntropyLoss


comptime FEATURE_DTYPE = DType.float32


def warmup_gpu(gpu: GPU) raises:
    """Run a tiny GPU kernel to warm up the device + compilation cache."""
    var a = Tensor[FEATURE_DTYPE].rand(Shape(4, 4))
    var b = Tensor[FEATURE_DTYPE].rand(Shape(4, 4))
    var a_gpu = a.to_gpu(gpu, sync=True)
    var b_gpu = b.to_gpu(gpu, sync=True)
    _ = a_gpu + b_gpu
    _ = a_gpu.matmul(b_gpu)


def time_bias_add(
    desc: String,
    a: Tensor[FEATURE_DTYPE],
    b: Tensor[FEATURE_DTYPE],
) raises:
    var t = perf_counter_ns()
    var r = a.__add__[track_grad=False, sync=True](b)
    var elapsed = Float64(perf_counter_ns() - t) / 1e6
    print("  ", desc, ":  ", elapsed, " ms  (shape=", String(r.shape()), ")")


def time_bias_add_warm(
    desc: String,
    a: Tensor[FEATURE_DTYPE],
    b: Tensor[FEATURE_DTYPE],
    iterations: Int = 10,
) raises:
    # First iteration (includes compilation)
    var t0 = perf_counter_ns()
    _ = a.__add__[track_grad=False, sync=True](b)
    var first_elapsed = Float64(perf_counter_ns() - t0) / 1e6
    print("  ", desc, " [first]:  ", first_elapsed, " ms")

    # Average of remaining iterations (warm)
    t0 = perf_counter_ns()
    for i in range(iterations - 1):
        _ = a.__add__[track_grad=False, sync=True](b)
    var total_ns = perf_counter_ns() - t0
    var avg_elapsed = Float64(total_ns) / Float64(iterations - 1) / 1e6
    print("  ", desc, " [avg ", iterations - 1, "]: ", avg_elapsed, " ms")


def main() raises:
    comptime if not has_accelerator():
        raise Error("No GPU accelerator found.")
    print("=" * 72)
    print("GPU Profiling: bias_add dispatch + forward/backward")
    print("=" * 72)
    var gpu = GPU()

    # Warmup
    print("\n--- Warmup ---")
    warmup_gpu(gpu)
    print("  Done.")

    # ================================================================
    # 1. Create GPU tensors at MNIST MLP shapes
    # ================================================================
    print("\n--- Creating tensors at MLP shapes ---")
    var h128 = Tensor[FEATURE_DTYPE].rand(Shape(64, 128))
    var b128 = Tensor[FEATURE_DTYPE].rand(Shape(128))
    var h32 = Tensor[FEATURE_DTYPE].rand(Shape(64, 32))
    var b32 = Tensor[FEATURE_DTYPE].rand(Shape(32))
    var h10 = Tensor[FEATURE_DTYPE].rand(Shape(64, 10))
    var b10 = Tensor[FEATURE_DTYPE].rand(Shape(10))

    var h128_gpu = h128.to_gpu(gpu, sync=True)
    var b128_gpu = b128.to_gpu(gpu, sync=True)
    var h32_gpu = h32.to_gpu(gpu, sync=True)
    var b32_gpu = b32.to_gpu(gpu, sync=True)
    var h10_gpu = h10.to_gpu(gpu, sync=True)
    var b10_gpu = b10.to_gpu(gpu, sync=True)
    print("  All tensors transferred to GPU.")

    # Additional shapes for dispatch-path testing
    var outer_b_gpu = Tensor[FEATURE_DTYPE].rand(Shape(64, 1)).to_gpu(gpu, sync=True)
    # Transposed views for strided path testing
    var h128_T_gpu = h128_gpu.transpose()
    var b128_1d_gpu = Tensor[FEATURE_DTYPE].rand(Shape(128)).to_gpu(gpu, sync=True)

    # ================================================================
    # 2. bias_add timing (sync=True) — one shot each
    # ================================================================
    print("\n=== Bias Add Timing (sync=True, track_grad=False) ===")
    time_bias_add("(64,128)+(128,)", h128_gpu, b128_gpu)
    time_bias_add("(64,32)+(32,)",   h32_gpu,  b32_gpu)
    time_bias_add("(64,10)+(10,)",   h10_gpu,  b10_gpu)

    # ================================================================
    # 3. bias_add with warmup loop (avg of N)
    # ================================================================
    print("\n=== Bias Add With Warmup (N=10) ===")
    time_bias_add_warm("(64,128)+(128,)", h128_gpu, b128_gpu, 10)
    time_bias_add_warm("(64,10)+(10,)", h10_gpu, b10_gpu, 10)

    # ================================================================
    # 4. Dispatch Path Tests — explicitly exercise each PATH
    # ================================================================
    print("\n=== Dispatch Path Tests ===")
    print("  (Watch for [DISPATCH] PATH X prints above)")

    # PATH 1: both_contiguous same_shape
    print("--- PATH 1: both_contiguous same shape ---")
    time_bias_add("(64,128)+(64,128)", h128_gpu, h128_gpu)

    # PATH 3: A_contiguous fills broadcast (bias_add)
    print("--- PATH 3: A contiguous fills shape <-- bias_add ---")
    time_bias_add("same as above", h128_gpu, b128_gpu)

    # PATH 4: B contiguous fills broadcast
    # B has the broadcast shape but A is non-contiguous (transposed)
    print("--- PATH 4: B contiguous fills shape ---")
    # h128_T_gpu shape (128,64), both operands same shape
    # h128_T_gpu is not contiguous (transposed)
    time_bias_add("(128,64)+(128,64) [B fills]", h128_T_gpu, h128_T_gpu)

    # PATH 2: both contiguous broadcast
    # (64,128)+(64,1) — both contiguous, shapes differ, neither fills broadcast
    print("--- PATH 2: both contiguous broadcast ---")
    time_bias_add("(64,128)+(64,1)", h128_gpu, outer_b_gpu)

    # PATH 5: both strided fallback
    # Transposed A + non-broadcast B that doesn't fill broadcast shape
    print("--- PATH 5: both_strided fallback ---")
    time_bias_add("(128,64)+(128,)", h128_T_gpu, b128_1d_gpu)

    # ================================================================
    # 5. matmul + relu baselines
    # ================================================================
    print("\n=== Matmul + ReLU Baselines (sync=True, track_grad=False) ===")
    var w1 = Tensor[FEATURE_DTYPE].rand(Shape(784, 128))
    var w1_gpu = w1.to_gpu(gpu, sync=True)
    var input_flat = Tensor[FEATURE_DTYPE].rand(Shape(64, 784))
    var input_gpu = input_flat.to_gpu(gpu, sync=True)

    var t = perf_counter_ns()
    var mm_result = input_gpu.matmul[track_grad=False](w1_gpu, sync=True)
    var mm_elapsed = Float64(perf_counter_ns() - t) / 1e6
    print("  matmul (64,784)x(784,128): ", mm_elapsed, " ms")

    t = perf_counter_ns()
    var biased = mm_result.__add__[track_grad=False, sync=True](b128_gpu)
    var bias_elapsed = Float64(perf_counter_ns() - t) / 1e6
    print("  bias_add (64,128)+(128,):   ", bias_elapsed, " ms")

    t = perf_counter_ns()
    var relu_out = biased.relu[track_grad=False](sync=True)
    var relu_elapsed = Float64(perf_counter_ns() - t) / 1e6
    print("  relu (64,128):              ", relu_elapsed, " ms")

    # ================================================================
    # 6. Full forward + backward profile (with autograd)
    # ================================================================
    print("\n=== Full Forward + Backward (with autograd, sync=False) ===")
    var model = Sequential[FEATURE_DTYPE]()
    model.append(
        Linear[FEATURE_DTYPE](784, 128, init_method="he", bias_zero=True).into(),
        ReLU[FEATURE_DTYPE]().into(),
        Linear[FEATURE_DTYPE](128, 32, init_method="he", bias_zero=True).into(),
        ReLU[FEATURE_DTYPE]().into(),
        Linear[FEATURE_DTYPE](32, 10, init_method="he", bias_zero=True).into(),
    )
    model = model.to_gpu(gpu, stop_grad=True)
    var criterion = CrossEntropyLoss[FEATURE_DTYPE]()
    var labels_cpu = Tensor[DType.int32].zeros(Shape(64))
    var features_gpu = input_flat.to_gpu(gpu, sync=True)
    var labels_gpu = labels_cpu.to_gpu(gpu, sync=True)

    # Forward (sync=False)
    t = perf_counter_ns()
    var pred = model(features_gpu, sync=False)
    var fwd_time = Float64(perf_counter_ns() - t) / 1e6
    t = perf_counter_ns()
    var loss = criterion(pred, labels_gpu, sync=False)
    var loss_time = Float64(perf_counter_ns() - t) / 1e6
    print("  Forward:     ", fwd_time, " ms")
    print("  Loss:        ", loss_time, " ms")

    # Backward (includes GPU sync inside backward())
    t = perf_counter_ns()
    loss.backward()
    var bwd_time = Float64(perf_counter_ns() - t) / 1e6
    print("  Backward:    ", bwd_time, " ms")
    print("  Fwd+Bwd:     ", fwd_time + loss_time + bwd_time, " ms")

    # ================================================================
    # 7. Forward per-op breakdown (sync=True)
    # ================================================================
    print("\n=== Forward Per-Op Breakdown (sync=True) ===")
    # Access weights via model.parameters()
    var params = model.parameters()
    var w1_m = params[0][]
    var b1_m = params[1][]
    var w2_m = params[2][]
    var b2_m = params[3][]
    var w3_m = params[4][]
    var b3_m = params[5][]

    var h = features_gpu.matmul[track_grad=True](w1_m, sync=True)
    print("  matmul (64,784)x(784,128): ", end="")

    t = perf_counter_ns()
    h = h + b1_m
    print(Float64(perf_counter_ns() - t) / 1e6, " ms")

    t = perf_counter_ns()
    h = h.relu[track_grad=True](sync=True)
    print("  relu 128:                   ", Float64(perf_counter_ns() - t) / 1e6, " ms")

    t = perf_counter_ns()
    h = h.matmul[track_grad=True](w2_m, sync=True)
    print("  matmul (64,128)x(128,32):  ", Float64(perf_counter_ns() - t) / 1e6, " ms")

    t = perf_counter_ns()
    h = h + b2_m
    print("  bias_add (64,32)+(32,):    ", Float64(perf_counter_ns() - t) / 1e6, " ms")

    t = perf_counter_ns()
    h = h.relu[track_grad=True](sync=True)
    print("  relu 32:                    ", Float64(perf_counter_ns() - t) / 1e6, " ms")

    t = perf_counter_ns()
    h = h.matmul[track_grad=True](w3_m, sync=True)
    print("  matmul (64,32)x(32,10):    ", Float64(perf_counter_ns() - t) / 1e6, " ms")

    t = perf_counter_ns()
    h = h + b3_m
    print("  bias_add (64,10)+(10,):    ", Float64(perf_counter_ns() - t) / 1e6, " ms")

    # Crossentropy with int32 labels
    t = perf_counter_ns()
    var loss2 = criterion(h, labels_gpu, sync=True)
    print("  crossentropy:               ", Float64(perf_counter_ns() - t) / 1e6, " ms")

    print("\n=== Done ===")
