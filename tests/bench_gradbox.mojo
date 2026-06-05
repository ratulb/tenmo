from std.time import perf_counter_ns
from tenmo.tensor import Tensor
from tenmo.shapes import Shape


def measure(ns: UInt) -> Float64:
    return Float64(ns) / 1_000_000.0


# ── Benchmark 1: Tensor creation with requires_grad ─────────────────────────
def bench_create_gradbox():
    comptime dtype = DType.float32
    var sizes = List[Shape]()
    sizes.append(Shape([64]))
    sizes.append(Shape([256]))
    sizes.append(Shape([1024]))
    sizes.append(Shape([4096]))
    sizes.append(Shape([128, 128]))
    sizes.append(Shape([256, 256]))

    print("\n── Benchmark 1: Tensor creation with requires_grad ──")
    print("Shape                Total (ms)    Avg (ms)")
    print("--------------------------------------------")

    for s in range(len(sizes)):
        var shape = sizes[s]
        var num_iters = 100000

        var t0 = perf_counter_ns()
        for _ in range(num_iters):
            var t = Tensor[dtype](shape, requires_grad=True)
            _ = t.shape()
        var t1 = perf_counter_ns()
        var total_ms = measure(t1 - t0)
        var avg_ms = total_ms / Float64(num_iters)
        var shape_str = String(shape)
        while shape_str.byte_length() < 20:
            shape_str += " "
        print(shape_str, " ", total_ms, "       ", avg_ms)


# ── Benchmark 2: Forward graph construction ─────────────────────────────────
def bench_forward_graph():
    comptime dtype = DType.float32
    var depths = List[Int]()
    depths.append(10)
    depths.append(50)
    depths.append(100)
    depths.append(200)

    print("\n── Benchmark 2: Chain of adds (forward graph construction) ──")
    print("Depth      Total (ms)    Avg (ms)")
    print("----------------------------------")

    for d in range(len(depths)):
        var depth = depths[d]
        var num_iters = 2000

        var t0 = perf_counter_ns()
        for _ in range(num_iters):
            var a = Tensor[dtype](Shape([32]), requires_grad=True)
            var b = Tensor[dtype](Shape([32]), requires_grad=True)
            for _ in range(depth):
                a = a + b
        var t1 = perf_counter_ns()

        var total_ms = measure(t1 - t0)
        var avg_ms = total_ms / Float64(num_iters)
        print(String(depth), "        ", total_ms, "       ", avg_ms)


# ── Benchmark 3: Backward pass ─────────────────────────────────────────────
def bench_backward():
    comptime dtype = DType.float32
    var depths = List[Int]()
    depths.append(10)
    depths.append(50)
    depths.append(100)

    print("\n── Benchmark 3: Chain backward pass ──")
    print("Depth      Total (ms)    Avg (ms)")
    print("----------------------------------")

    for d in range(len(depths)):
        var depth = depths[d]
        var num_iters = 500

        var t0 = perf_counter_ns()
        for _ in range(num_iters):
            var a = Tensor[dtype](Shape([32]), requires_grad=True)
            var b = Tensor[dtype](Shape([32]), requires_grad=True)
            var out = a + b
            for _ in range(depth - 1):
                out = out + b
            out.backward()
        var t1 = perf_counter_ns()

        var total_ms = measure(t1 - t0)
        var avg_ms = total_ms / Float64(num_iters)
        print(String(depth), "        ", total_ms, "       ", avg_ms)


# ── Benchmark 4: Full pipeline ──────────────────────────────────────────────
def bench_full_pipeline():
    comptime dtype = DType.float32
    var sizes = List[Shape]()
    sizes.append(Shape([16]))
    sizes.append(Shape([64]))
    sizes.append(Shape([256]))

    print("\n── Benchmark 4: Full pipeline (create + forward + backward) ──")
    print("Shape       Total (ms)    Avg (ms)")
    print("------------------------------------")

    for s in range(len(sizes)):
        var shape = sizes[s]
        var num_iters = 2000

        var t0 = perf_counter_ns()
        for _ in range(num_iters):
            var a = Tensor[dtype](shape, requires_grad=True)
            var b = Tensor[dtype](shape, requires_grad=True)
            var c = a + b
            var d = c * a
            d.backward()
        var t1 = perf_counter_ns()

        var total_ms = measure(t1 - t0)
        var avg_ms = total_ms / Float64(num_iters)
        var shape_str = String(shape)
        while shape_str.byte_length() < 12:
            shape_str += " "
        print(shape_str, " ", total_ms, "       ", avg_ms)


# ── Benchmark 5: Fan-out graph ──────────────────────────────────────────────
def bench_fanout():
    comptime dtype = DType.float32

    print("\n── Benchmark 5: Fan-out (one root, many branches) ──")
    print("Branches    Total (ms)    Avg (ms)")
    print("------------------------------------")

    var branches = List[Int]()
    branches.append(10)
    branches.append(50)

    for b in range(len(branches)):
        var n = branches[b]
        var num_iters = 500

        var t0 = perf_counter_ns()
        for _ in range(num_iters):
            var root = Tensor[dtype](Shape([16]), requires_grad=True)
            var outs = List[Tensor[dtype]]()
            for _ in range(n):
                var leaf = Tensor[dtype](Shape([16]), requires_grad=True)
                outs.append(root + leaf)
            var sum_outs = outs[0]
            var i = 1
            while i < n:
                sum_outs = sum_outs + outs[i]
                i += 1
            sum_outs.backward()
        var t1 = perf_counter_ns()

        var total_ms = measure(t1 - t0)
        var avg_ms = total_ms / Float64(num_iters)
        print(String(n), "          ", total_ms, "       ", avg_ms)


# ── Benchmark 6: Gradbox alloc/free churn ──────────────────────────────────
def bench_gradbox_churn():
    comptime dtype = DType.float32
    var num_iters = 100000

    print("\n── Benchmark 6: Gradbox alloc/free churn ──")
    var t0 = perf_counter_ns()
    for _ in range(num_iters):
        var a = Tensor[dtype](Shape([64]), requires_grad=True)
        var b = Tensor[dtype](Shape([64]), requires_grad=False)
        _ = a.shape()
        _ = b.shape()
    var t1 = perf_counter_ns()
    var total_ms = measure(t1 - t0)
    var avg_ms = total_ms / Float64(num_iters)
    print("  ", String(num_iters), "iterations: ", total_ms, " ms total, ", avg_ms, " ms avg")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("GRADBOX BENCHMARK - gpu_synchronize")
    print("=" * 60)

    bench_create_gradbox()
    bench_forward_graph()
    bench_backward()
    bench_full_pipeline()
    bench_fanout()
    bench_gradbox_churn()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
