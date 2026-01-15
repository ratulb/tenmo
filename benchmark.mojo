from time import perf_counter_ns
from random import random_float64
from tenmo import Tensor
from shapes import Shape


# ============================================
# NAIVE MATMUL - Pure scalar loops, no SIMD
# ============================================
fn matmul_naive[
    dtype: DType
](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
    """Naive triple-loop matmul with no optimizations."""
    var m = A.shape()[0]
    var n = A.shape()[1]
    var p = B.shape()[1]

    var C = Tensor[dtype].zeros(Shape([m, p]))

    # Pure scalar triple loop - i-j-k order
    for i in range(m):
        for j in range(p):
            var accumulator: Scalar[dtype] = 0
            for k in range(n):
                # Scalar loads only
                var a_ik = A.load[simdwidth=1, validated=False](i, k)
                var b_kj = B.load[simdwidth=1, validated=False](k, j)
                accumulator += a_ik * b_kj
            C.store[simdwidth=1, validated=False](i, j, accumulator)

    return C^


# ============================================
# BENCHMARK RUNNER
# ============================================
fn benchmark_matmul():
    alias dtype = DType.float32

    print("=" * 60)
    print("MATMUL BENCHMARK - Float32")
    print("=" * 60)

    # Test sizes
    var sizes = List[Int]()
    sizes.append(128)
    sizes.append(256)
    sizes.append(512)
    sizes.append(1024)

    for size_idx in range(len(sizes)):
        var size = sizes[size_idx]
        print("\n" + "─" * 60)
        print("Matrix Size:", size, "×", size)
        print("─" * 60)

        # Create test matrices
        var A = Tensor[dtype].rand(Shape([size, size]))
        var B = Tensor[dtype].rand(Shape([size, size]))

        # Warm-up (important for cache/CPU frequency scaling)
        _ = matmul_naive(A, B)
        _ = A.matmul[track_grad=False](B)

        # ========================================
        # NAIVE MATMUL BENCHMARK
        # ========================================
        print("\n[1] Naive Matmul (scalar, i-j-k order)")
        var t0 = perf_counter_ns()
        var C_naive = matmul_naive(A, B)
        var t1 = perf_counter_ns()
        var time_naive = (t1 - t0) / 1e9

        var flops = 2.0 * Float64(size) * Float64(size) * Float64(size)
        var gflops_naive = flops / time_naive / 1e9

        print("  Time:   ", time_naive, "seconds")
        print("  GFLOPS: ", gflops_naive)

        # ========================================
        # OPTIMIZED MATMUL BENCHMARK (no backward)
        # ========================================
        print("\n[2] Optimized Matmul (SIMD, hoisted metadata, no grad)")
        var t2 = perf_counter_ns()
        var C_opt = A.matmul[track_grad=False](B)
        var t3 = perf_counter_ns()
        var time_opt = (t3 - t2) / 1e9
        var gflops_opt = flops / time_opt / 1e9

        print("  Time:   ", time_opt, "seconds")
        print("  GFLOPS: ", gflops_opt)
        print("  Speedup:", time_naive / time_opt, "×")

        # ========================================
        # OPTIMIZED MATMUL WITH BACKWARD
        # ========================================
        print("\n[3] Optimized Matmul + Backward")

        var A_grad = Tensor[dtype].rand(Shape([size, size]))
        var B_grad = Tensor[dtype].rand(Shape([size, size]))
        A_grad.requires_grad_(True)
        B_grad.requires_grad_(True)

        var t4 = perf_counter_ns()
        var C_grad = A_grad.matmul(B_grad)
        var t5 = perf_counter_ns()
        var time_forward = (t5 - t4) / 1e9

        var t6 = perf_counter_ns()
        C_grad.backward()
        var t7 = perf_counter_ns()
        var time_backward = (t7 - t6) / 1e9
        var time_total = (t7 - t4) / 1e9

        print("  Forward:  ", time_forward, "seconds")
        print("  Backward: ", time_backward, "seconds")
        print("  Total:    ", time_total, "seconds")
        print("  Backward/Forward ratio:", time_backward / time_forward, "×")

        # ========================================
        # CORRECTNESS CHECK
        # ========================================
        print("\n[4] Correctness Check")
        var max_diff: Float32 = 0.0
        for i in range(min(size, 10)):  # Check first 10×10 submatrix
            for j in range(min(size, 10)):
                var naive_val = C_naive.load[simdwidth=1, validated=False](i, j)
                var opt_val = C_opt.load[simdwidth=1, validated=False](i, j)
                var diff = abs(naive_val - opt_val)
                if diff > max_diff:
                    max_diff = diff

        if max_diff < 1e-3:
            print("  ✓ Results match (max diff:", max_diff, ")")
        else:
            print("  ✗ Results differ (max diff:", max_diff, ")")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


# ============================================
# DETAILED TIMING BREAKDOWN
# ============================================
fn benchmark_backward_breakdown():
    alias dtype = DType.float32
    alias size = 1024

    print("\n" + "=" * 60)
    print("BACKWARD PASS DETAILED BREAKDOWN")
    print("=" * 60)

    var A = Tensor[dtype].rand(Shape([size, size]))
    var B = Tensor[dtype].rand(Shape([size, size]))
    A.requires_grad_(True)
    B.requires_grad_(True)

    # Forward pass
    print("\n[Forward Pass]")
    var t0 = perf_counter_ns()
    var C = A.matmul(B)
    var t1 = perf_counter_ns()
    print("Time:", (t1 - t0) / 1e9, "seconds")

    # Backward pass with instrumentation
    # (Make sure you have the timing code in Ancestor.backward())
    print("\n[Backward Pass - Detailed]")
    var t2 = perf_counter_ns()
    C.backward()
    var t3 = perf_counter_ns()
    print("Total time:", (t3 - t2) / 1e9, "seconds")

    # The detailed breakdown will be printed by the instrumented backward()
    # [Backward] Seed grad: ??? ms
    # [Backward] Build topology: ??? ms
    # [Backward] Refresh registry: ??? ms
    # [Backward] Execute backward: ??? ms


# ============================================
# MEMORY BANDWIDTH TEST
# ============================================
fn benchmark_memory_bandwidth():
    """Test if we're compute-bound or memory-bound."""
    alias dtype = DType.float32
    alias size = 1024

    print("\n" + "=" * 60)
    print("MEMORY BANDWIDTH TEST")
    print("=" * 60)

    var A = Tensor[dtype].rand(Shape([size, size]))

    # Measure pure memory copy speed
    var t0 = perf_counter_ns()
    var _B = A.copy()
    var t1 = perf_counter_ns()
    var copy_time = (t1 - t0) / 1e9

    var bytes_copied = Float64(size * size * 4)  # 4 bytes per float32
    var bandwidth_gbs = bytes_copied / copy_time / 1e9

    print("Matrix size:", size, "×", size)
    print("Copy time:  ", copy_time, "seconds")
    print("Bandwidth:  ", bandwidth_gbs, "GB/s")
    print("\nFor reference:")
    print("  - DDR4-3200: ~25 GB/s theoretical")
    print("  - DDR5-5600: ~45 GB/s theoretical")
    print("  - L3 cache:  ~100-200 GB/s")


# ============================================
# MAIN
# ============================================
fn main():
    print("Starting matmul benchmarks...\n")

    # Run main benchmark suite
    benchmark_matmul()

    # Run detailed backward breakdown
    benchmark_backward_breakdown()

    # Run memory bandwidth test
    benchmark_memory_bandwidth()

    print("\n✓ All benchmarks complete!")
