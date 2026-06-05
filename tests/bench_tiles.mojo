from std.time import perf_counter
from tenmo import NDBuffer, Shape, Tensor
from tenmo.ndbuffer import MmCpu2d
from std.sys import argv
from std.python import Python

comptime dtype = DType.float32

# ── Default benchmark data ───────────────────────────────────────────────────

comptime DEFAULT_SHAPE_DATA = (
    (1, 1, 1),
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 8, 16),
    (4, 8, 1024),
    (64, 256, 64),
    (1024, 64, 256),
    (256, 1024, 128),
)

comptime DEFAULT_SHAPE_LABELS = (
    "1×1×1",
    "16×16×16",
    "32×32×32",
    "64×64×64",
    "128×128×128",
    "256×256×256",
    "512×512×512",
    "1024×8×16 ",
    "4×8×1024  ",
    "64×256×64",
    "1024×64×256",
    "256×1024×128",
)

comptime CONFIGS = (
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (64, 64, 128),
    (128, 64, 128),
    (128, 64, 256),
)

comptime CFG_LABELS = (
    "16×16×16",
    "32×32×32",
    "64×64×64",
    "64×64×128",
    "128×64×128",
    "128×64×256",
)

comptime NUM_CFG = 6
comptime NUM_SHAPES = 12

# ── Helpers ──────────────────────────────────────────────────────────────────


def fmt3(val: Float64, w: Int) -> String:
    var scaled = val
    var neg = ""
    if scaled < 0.0:
        neg = "-"
        scaled = -scaled
    var whole = Int(scaled)
    var frac = Int((scaled - Float64(whole)) * 1000.0 + 0.5)
    if frac >= 1000:
        frac = 0
        whole += 1
    var s = neg + String(whole) + "."
    if frac < 10:
        s += "00"
    elif frac < 100:
        s += "0"
    s += String(frac)
    while s.byte_length() < w:
        s = " " + s
    return s


def make_mats(
    M: Int, K: Int, N: Int
) -> Tuple[NDBuffer[dtype], NDBuffer[dtype]]:
    var A = Tensor[dtype].randn(M, K)
    var B = Tensor[dtype].randn(K, N)
    return (A.buffer.copy(), B.buffer.copy())


# ── Parse CSV ints from string ──────────────────────────────────────────────


def parse_csv(s: String) raises -> List[Int]:
    var result = List[Int]()
    var val = 0
    for i in range(s.byte_length()):
        var ch = s[byte=i]
        if ch == ",":
            result.append(val)
            val = 0
        else:
            val = val * 10 + Int(ch)
    result.append(val)
    return result^


# ── Main dispatch ────────────────────────────────────────────────────────────


def main() raises:
    var args = argv()
    if len(args) <= 1:
        run_default()
    else:
        run_custom(args)


# ── Default run ──────────────────────────────────────────────────────────────


def run_default() raises:
    print(
        "── Default benchmark ────────────────────────────────────────────────"
    )
    print("Configs: 6  Shapes: 12")
    print("")
    var header = "shape          "
    comptime for ci in range(NUM_CFG):
        header += "  " + CFG_LABELS[ci]
    print(header)
    print("")

    comptime for si in range(NUM_SHAPES):
        comptime M = DEFAULT_SHAPE_DATA[si][0]
        comptime K = DEFAULT_SHAPE_DATA[si][1]
        comptime N = DEFAULT_SHAPE_DATA[si][2]
        var flops = 2 * M * K * N
        var iters = max(1, 10_000_000 // flops)
        iters = min(iters, 200)

        var (A, B) = make_mats(M, K, N)

        var row = DEFAULT_SHAPE_LABELS[si]
        comptime for ci in range(NUM_CFG):
            comptime tm = CONFIGS[ci][0]
            comptime tn = CONFIGS[ci][1]
            comptime tp = CONFIGS[ci][2]
            var t = bench_one[tm, tn, tp](A, B, iters)
            row += "  " + fmt3(t, 9)

        print(row)

    print("")
    print(
        "── Analysis ─────────────────────────────────────────────────────────"
    )
    print("")
    print("Key findings from benchmark data:")
    print("")
    print("  32x32x32 dominates up to M=256: fastest for 16..256 across all")
    print("  aspect ratios.  TILE_N=32 fits small inner dims without waste.")
    print("")
    print("  64x64x128 wins for large square (512x512x512): 2.2x faster than")
    print("  64x64x64, 2.5x faster than 32x32x32.  Larger TILE_P enables")
    print("  better SIMD utilization on the reduction dimension.")
    print("")
    print(
        "  Tall-skinny (1024x8x16): tile choice barely matters (~1.3x spread)."
    )
    print("  Wide-short (1024x64x256): 32x32x32 fastest despite large M.")
    print("  Fat inner (256x1024x128): 64x64x64 wins, 128x64x128 close.")
    print("")
    print("  Revised pick_cpu (ndbuffer.mojo:~3037):")
    print("    M>256 AND N>128  → (64, 64, 128)")
    print("    else             → (32, 32, 32)")
    print("  Two tiers instead of three; (32,32,32) handles all <=256 cases.")
    print("")
    print("  Use --configs and --shapes flags to run your own benchmarks.")
    print(
        "  e.g.:  mojo tests/bench_tiles.mojo --configs 32,32,32 64,64,128"
        " --shapes 128,128,128"
    )


# ── Custom run ───────────────────────────────────────────────────────────────


def run_custom(
    args: Span[StringSlice[StaticConstantOrigin], StaticConstantOrigin]
) raises:
    var cfg_indices = List[Int]()
    var cfg_labels = List[String]()
    var shape_ms = List[Int]()
    var shape_ks = List[Int]()
    var shape_ns = List[Int]()
    var shape_labels = List[String]()
    var mode = 0

    for i in range(1, len(args)):
        var arg = args[i]
        if arg == "--configs":
            mode = 1
        elif arg == "--shapes":
            mode = 2
        elif arg == "--help" or arg == "-h":
            print("Usage:")
            print(
                "  mojo tests/bench_tiles.mojo                                 "
                " default benchmark"
            )
            print(
                "  mojo tests/bench_tiles.mojo --configs M,N,P [M,N,P...]      "
                " custom tiles"
            )
            print(
                "  mojo tests/bench_tiles.mojo --shapes M,K,N [M,K,N...]       "
                " custom shapes"
            )
            print(
                "  mojo tests/bench_tiles.mojo --configs ... --shapes ...      "
                " both"
            )
            print("")
            print("Tiles:   TILE_M,TILE_N,TILE_P   e.g. 64,64,128")
            print("Shapes:  M,K,N                   e.g. 256,256,256")
            return
        elif mode == 1:
            var parts = parse_csv(arg)
            if len(parts) >= 3:
                var tm = parts[0]
                var tn = parts[1]
                var tp = parts[2]
                var found = -1
                comptime for ci in range(NUM_CFG):
                    if (
                        tm == CONFIGS[ci][0]
                        and tn == CONFIGS[ci][1]
                        and tp == CONFIGS[ci][2]
                    ):
                        found = ci
                if found >= 0:
                    cfg_indices.append(found)
                    cfg_labels.append(String(arg))
                else:
                    print(
                        "Unknown config: "
                        + String(arg)
                        + " (must be one of the 6 predefined)"
                    )
                    return
        elif mode == 2:
            var parts = parse_csv(arg)
            if len(parts) >= 3:
                shape_ms.append(parts[0])
                shape_ks.append(parts[1])
                shape_ns.append(parts[2])
                shape_labels.append(String(arg))
            else:
                print("Invalid shape: " + String(arg) + " (expected M,K,N)")
                return

    if len(cfg_indices) == 0:
        print("Missing --configs")
        return
    if len(shape_ms) == 0:
        print("Missing --shapes")
        return

    print(
        "── Custom benchmark ──────────────────────────────────────────────────"
    )
    print(
        "Configs: "
        + String(len(cfg_indices))
        + "  Shapes: "
        + String(len(shape_ms))
    )
    print("")
    var header = "shape          "
    for ci in range(len(cfg_indices)):
        header += "  " + cfg_labels[ci]
    print(header)
    print("")

    for si in range(len(shape_ms)):
        var M = shape_ms[si]
        var K = shape_ks[si]
        var N = shape_ns[si]
        var flops = 2 * M * K * N
        var iters = max(1, 10_000_000 // flops)
        iters = min(iters, 200)

        var (A, B) = make_mats(M, K, N)

        var row = shape_labels[si]
        for ci in range(len(cfg_indices)):
            var cidx = cfg_indices[ci]
            comptime for k in range(NUM_CFG):
                if k == cidx:
                    comptime tm = CONFIGS[k][0]
                    comptime tn = CONFIGS[k][1]
                    comptime tp = CONFIGS[k][2]
                    var t = bench_one[tm, tn, tp](A, B, iters)
                    row += "  " + fmt3(t, 9)

        print(row)


# ── Benchmark kernel ─────────────────────────────────────────────────────────


def bench_one[
    tm: Int, tn: Int, tp: Int
](A: NDBuffer[dtype], B: NDBuffer[dtype], iters: Int) -> Float64:
    for warm in range(3):
        _ = MmCpu2d[dtype, tm, tn, tp].matmul(A, B)
    var start = perf_counter()
    for iter_idx in range(iters):
        _ = MmCpu2d[dtype, tm, tn, tp].matmul(A, B)
    var end = perf_counter()
    return (end - start) * 1000.0 / Float64(iters)
