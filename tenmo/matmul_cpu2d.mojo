from tenmo.ndbuffer import NDBuffer, Shape, Strides, MatrixShapeValidator
from tenmo.kernels.matmul_kernel import MatmulNdGpu
from tenmo.common_utils import panic
from std.algorithm import parallelize
from std.sys import prefetch, PrefetchOptions, simd_width_of, has_accelerator, size_of
from std.sys.info import num_physical_cores
from std import math

# ─────────────────────────────────────────────────────────────────────────────
#  Tuning constants
#
#  PREFETCH_POLICY : 0 = Off (no prefetch instructions emitted),
#                    1 = Conservative (light prefetch, capped at 32 lines),
#                    2 = Aggressive (full prefetch, up to 128 lines)
#
#  GPU_TILE_SIZE : tile dimension for the GPU kernel (separate from CPU tiles)
#
#  CPU tiles — three independent dimensions:
#    TILE_M : rows of A/C per parallel chunk → sized to fit A-rows in L2
#    TILE_N : shared k-dimension strip       → sized to fit A k-strip in L1
#    TILE_P : columns of B/C per j-tile      → wide enough to saturate SIMD
#
#  UNROLL     : number of SIMD accumulators per j-strip inside the hot loop.
#               float32 with simdwidth=8 and UNROLL=4 → 32 columns per iter.
#               More unroll = better FMA pipeline utilisation, but more
#               register pressure. 4 is a good balance for most CPUs.
# ─────────────────────────────────────────────────────────────────────────────

comptime PREFETCH_POLICY = 1  # 0=Off, 1=Conservative, 2=Aggressive
comptime prefetch_opts = PrefetchOptions().for_read().high_locality().to_data_cache()
comptime MAX_PREFETCH_LINES = 32 if PREFETCH_POLICY <= 1 else 128
comptime GPU_TILE_SIZE = 32
comptime UNROLL = 4


struct MmCpu2d[
    dtype: DType, TILE_M: Int = 32, TILE_N: Int = 32, TILE_P: Int = 32
]:
    # ─────────────────────────────────────────────────────────────────────────
    #  matmul_2d
    #
    #  Entry point. Dispatches to GPU or CPU path at runtime, with the
    #  GPU branch compiled away entirely (comptime if) when no accelerator
    #  is present.
    #
    #  sync: relevant only for the GPU path. CPU ops are always synchronous
    #  and this parameter is silently ignored on the CPU path.
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def matmul_2d(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype], sync: Bool = True
    ) -> NDBuffer[Self.dtype]:
        ref A_shape = A.shape
        ref B_shape = B.shape
        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

        var C: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if A.is_on_gpu() and B.is_on_gpu():
                try:
                    C = MatmulNdGpu[Self.dtype].launch[tile_size=GPU_TILE_SIZE](
                        A, B, sync=sync
                    )
                except e:
                    print(e)
                    panic("NDBuffer matmul_2d → GPU operation failed")
                    C = NDBuffer[Self.dtype](Shape())  # unreachable
            elif (A.is_on_gpu() and B.is_on_cpu()) or (
                A.is_on_cpu() and B.is_on_gpu()
            ):
                panic(
                    "NDBuffer matmul_2d → both buffers must be on same device."
                    " A on gpu? ",
                    String(A.is_on_gpu()),
                    ", B on gpu? ",
                    String(B.is_on_gpu()),
                )
                C = NDBuffer[Self.dtype](Shape())  # unreachable
            else:
                # CPU path — sync parameter irrelevant, CPU is always synchronous
                C = MmCpu2d[Self.dtype].pick_cpu(A, B)
        else:
            # No accelerator — CPU path always. sync parameter ignored.
            C = MmCpu2d[Self.dtype].pick_cpu(A, B)

        return C^

    # ─────────────────────────────────────────────────────────────────────────
    #  pick_cpu
    #
    #  Dispatch tile config per-dimension independently.
    #
    #  Each dimension selects its own tile size based on its own size:
    #    TILE_M: driven by m  — controls row parallelism granularity
    #    TILE_N: driven by n  — controls A k-strip cache residency in L1
    #    TILE_P: driven by p  — MOST CRITICAL: must be >= simd_unroll
    #                           (simdwidth * UNROLL = 32 for float32/AVX2)
    #                           for the unrolled SIMD loop to fire at all.
    #                           A tall-narrow matrix (large m, small p) must
    #                           NOT use TILE_P=256 — the unroll loop never
    #                           fires and every iteration falls to scalar tail.
    #
    #  FIX Issue 1 (previous review): old code used OR across all dims,
    #  meaning (300, 10, 10) hit TILE_P=256 even with p=10.
    #
    #  FIX Issue 1 (this review): all 18 combinations are explicitly
    #  enumerated. The final else panics on any unanticipated combination
    #  rather than silently using wrong tile sizes. This makes dispatch bugs
    #  loud and immediate instead of subtle performance regressions.
    #
    #  Tile values:
    #    tile_m ∈ {32, 64, 128}
    #    tile_n ∈ {32, 64}
    #    tile_p ∈ {64, 128, 256}
    #    Total combinations: 3 × 2 × 3 = 18, all covered below.
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def pick_cpu(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        var m = A.shape[0]
        var n = A.shape[1]
        var p = B.shape[1]

        var tile_m = 128 if m > 256 else (64 if m > 64 else 32)
        var tile_n = 64  if n > 64  else 32
        var tile_p = 256 if p > 256 else (128 if p > 64 else 64)

        if tile_m == 128:
            if tile_n == 64:
                if tile_p == 256:
                    return MmCpu2d[Self.dtype, 128, 64, 256].matmul_2d_cpu(A, B)
                elif tile_p == 128:
                    return MmCpu2d[Self.dtype, 128, 64, 128].matmul_2d_cpu(A, B)
                else:
                    return MmCpu2d[Self.dtype, 128, 64, 64].matmul_2d_cpu(A, B)
            else:
                if tile_p == 256:
                    return MmCpu2d[Self.dtype, 128, 32, 256].matmul_2d_cpu(A, B)
                elif tile_p == 128:
                    return MmCpu2d[Self.dtype, 128, 32, 128].matmul_2d_cpu(A, B)
                else:
                    return MmCpu2d[Self.dtype, 128, 32, 64].matmul_2d_cpu(A, B)
        elif tile_m == 64:
            if tile_n == 64:
                if tile_p == 256:
                    return MmCpu2d[Self.dtype, 64, 64, 256].matmul_2d_cpu(A, B)
                elif tile_p == 128:
                    return MmCpu2d[Self.dtype, 64, 64, 128].matmul_2d_cpu(A, B)
                else:
                    return MmCpu2d[Self.dtype, 64, 64, 64].matmul_2d_cpu(A, B)
            else:
                if tile_p == 256:
                    return MmCpu2d[Self.dtype, 64, 32, 256].matmul_2d_cpu(A, B)
                elif tile_p == 128:
                    return MmCpu2d[Self.dtype, 64, 32, 128].matmul_2d_cpu(A, B)
                else:
                    return MmCpu2d[Self.dtype, 64, 32, 64].matmul_2d_cpu(A, B)
        else:
            if tile_n == 64:
                if tile_p == 256:
                    return MmCpu2d[Self.dtype, 32, 64, 256].matmul_2d_cpu(A, B)
                elif tile_p == 128:
                    return MmCpu2d[Self.dtype, 32, 64, 128].matmul_2d_cpu(A, B)
                else:
                    return MmCpu2d[Self.dtype, 32, 64, 64].matmul_2d_cpu(A, B)
            else:
                if tile_p == 256:
                    return MmCpu2d[Self.dtype, 32, 32, 256].matmul_2d_cpu(A, B)
                elif tile_p == 128:
                    return MmCpu2d[Self.dtype, 32, 32, 128].matmul_2d_cpu(A, B)
                else:
                    return MmCpu2d[Self.dtype, 32, 32, 64].matmul_2d_cpu(A, B)

    # ─────────────────────────────────────────────────────────────────────────
    #  matmul_2d_cpu
    #
    #  High-performance CPU matmul: C = A @ B
    #    A : (m, n)
    #    B : (n, p)
    #    C : (m, p)  — freshly allocated, zero-initialised
    #
    #  IMPORTANT: the k_tile==0 optimisation (skipping load from C) relies on
    #  C being zero-initialised on entry. This holds because C is always
    #  allocated via NDBuffer.zeros() above. Do not pass a pre-allocated C.
    #
    #  Three paths:
    #    1a. A contiguous,     B contiguous → SIMD + FMA + unroll + prefetch
    #    1b. A non-contiguous, B contiguous → SIMD + FMA + unroll + prefetch
    #    2.  B non-contiguous              → tiled scalar, fully tiled k×j
    #
    #  k_tile==0 split:
    #    In paths 1a and 1b the k_tile==0 branch sits inside the j loop.
    #    The compiler may hoist it but is not guaranteed to. For path 2
    #    (scalar, simpler structure) the branch is inside j which is inside
    #    k_tile — straightforward and correct.
    #    If profiling shows branch overhead in paths 1a/1b, split the j loop
    #    into a k_tile==0 copy and a k_tile>0 copy to guarantee hoisting.
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def matmul_2d_cpu(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        ref A_shape = A.shape
        ref B_shape = B.shape
        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

        comptime simdwidth    = simd_width_of[Self.dtype]()
        comptime simd_unroll  = simdwidth * UNROLL

        # Cache line = 64 bytes. Stride in elements is dtype-dependent:
        #   float16  → 32 elements,  float32 → 16,  float64 → 8
        comptime cache_line_elems = 64 // size_of[Scalar[Self.dtype]]()

        var m = A_shape[0]
        var n = A_shape[1]
        var p = B_shape[1]

        # C is zero-initialised. The k_tile==0 fast path (skip C load) relies
        # on this invariant. Do not change this allocation.
        var C = NDBuffer[Self.dtype].zeros(Shape([m, p]))

        # ── Hoist all pointer and stride metadata ─────────────────────────────
        # Accessing struct fields inside a hot loop forces repeated loads.
        # Storing them in locals lets the compiler keep them in registers.
        ref A_strides = A.strides
        var A_stride0 = A_strides[0]   # elements to advance one row in A
        var A_stride1 = A_strides[1]   # elements to advance one col in A
        var A_offset  = A.offset
        var A_data    = A.data_ptr()

        ref B_strides = B.strides
        var B_stride0 = B_strides[0]   # elements to advance one row in B
        var B_stride1 = B_strides[1]   # elements to advance one col in B (1 if contiguous)
        var B_offset  = B.offset
        var B_data    = B.data_ptr()

        var C_data = C.data_ptr()
        # C is always freshly allocated and contiguous:
        #   C_stride0 = p  (one full row)
        #   C_stride1 = 1  (adjacent columns)
        # These are inlined as literals below.

        if B.is_contiguous():
            var num_tiles_i = (m + Self.TILE_M - 1) // Self.TILE_M

            if A.is_contiguous():
                # ══════════════════════════════════════════════════════════
                #  Path 1a: A contiguous, B contiguous
                #
                #  Both strides are 1, so:
                #    a_addr = a_row_base + k          (no multiply)
                #    b_base = k * B_stride0 + B_offset + j  (B_stride1 = 1)
                #
                #  Loop order: k_tile → j_tile → i → j → k
                #    k outermost tile: A k-strip loads into L1 once per k_tile,
                #    reused across ALL j_tiles before eviction.
                #    j inside k_tile: B j-strip loads once per (k_tile, j_tile).
                #    i innermost loop: short, a_row_base computed once per row.
                # ══════════════════════════════════════════════════════════
                @parameter
                def process_contig_contig(tile_idx: Int):
                    var i_start = tile_idx * Self.TILE_M
                    var i_end   = min(i_start + Self.TILE_M, m)

                    for k_tile in range(0, n, Self.TILE_N):
                        var k_end = min(k_tile + Self.TILE_N, n)

                        # ── Prefetch next k-tile of B ──────────────────────
                        # Issue prefetch hints for the B k-strip needed in the
                        # next outer iteration while computing the current one.
                        # Placed outside j_tile: fired ONCE per k_tile, not
                        # once per (k_tile, j_tile) pair.
                        # Covers full column width in cache-line strides.
                        # Capped at MAX_PREFETCH_LINES to avoid flooding the
                        # prefetch queue.
                        comptime if PREFETCH_POLICY != 0:
                            var next_k = k_tile + Self.TILE_N
                            if next_k < n:
                                var next_k_end = min(next_k + Self.TILE_N, n)
                                var lines_issued = 0
                                for k_pre in range(next_k, next_k_end):
                                    if lines_issued >= MAX_PREFETCH_LINES:
                                        break
                                    var row_base = k_pre * B_stride0 + B_offset
                                    var cl = 0
                                    while (
                                        cl < p
                                        and lines_issued < MAX_PREFETCH_LINES
                                    ):
                                        prefetch[prefetch_opts](
                                            B_data + row_base + cl
                                        )
                                        cl += cache_line_elems
                                        lines_issued += 1

                        for j_tile in range(0, p, Self.TILE_P):
                            var j_end = min(j_tile + Self.TILE_P, p)

                            for i in range(i_start, i_end):
                                var a_row_base = i * A_stride0 + A_offset
                                var c_row_base = i * p
                                var j = j_tile

                                # ── Unrolled SIMD: UNROLL vectors per iter ─
                                # Processes simd_unroll = UNROLL * simdwidth
                                # columns per iteration (e.g. 32 for float32).
                                # UNROLL independent accumulators fill the FMA
                                # pipeline — prevents stalls from accumulator
                                # data dependency (FMA latency ~4 cycles).
                                # a_ik broadcast once, reused across all UNROLL
                                # accumulators — single load, 4 FMAs.
                                while j + simd_unroll <= j_end:
                                    var cj = c_row_base + j

                                    var acc0: SIMD[Self.dtype, simdwidth]
                                    var acc1: SIMD[Self.dtype, simdwidth]
                                    var acc2: SIMD[Self.dtype, simdwidth]
                                    var acc3: SIMD[Self.dtype, simdwidth]

                                    # k_tile==0: C is zeroed, skip the load.
                                    # Saves 4 vector reads on the first k pass.
                                    if k_tile == 0:
                                        acc0 = SIMD[Self.dtype, simdwidth](0)
                                        acc1 = SIMD[Self.dtype, simdwidth](0)
                                        acc2 = SIMD[Self.dtype, simdwidth](0)
                                        acc3 = SIMD[Self.dtype, simdwidth](0)
                                    else:
                                        acc0 = C_data.load[width=simdwidth](cj)
                                        acc1 = C_data.load[width=simdwidth](cj + simdwidth)
                                        acc2 = C_data.load[width=simdwidth](cj + simdwidth * 2)
                                        acc3 = C_data.load[width=simdwidth](cj + simdwidth * 3)

                                    for k in range(k_tile, k_end):
                                        # A contiguous: no multiply for a_addr
                                        var a_ik = SIMD[Self.dtype, simdwidth](
                                            A_data[a_row_base + k]
                                        )
                                        # B contiguous: B_stride1=1, no multiply for j
                                        var b_base = k * B_stride0 + B_offset + j
                                        acc0 = math.fma(a_ik, B_data.load[width=simdwidth](b_base),                 acc0)
                                        acc1 = math.fma(a_ik, B_data.load[width=simdwidth](b_base + simdwidth),     acc1)
                                        acc2 = math.fma(a_ik, B_data.load[width=simdwidth](b_base + simdwidth * 2), acc2)
                                        acc3 = math.fma(a_ik, B_data.load[width=simdwidth](b_base + simdwidth * 3), acc3)

                                    C_data.store[width=simdwidth](cj,                 acc0)
                                    C_data.store[width=simdwidth](cj + simdwidth,     acc1)
                                    C_data.store[width=simdwidth](cj + simdwidth * 2, acc2)
                                    C_data.store[width=simdwidth](cj + simdwidth * 3, acc3)
                                    j += simd_unroll

                                # ── Single-vector SIMD tail ────────────────
                                # Columns that didn't fill a full simd_unroll
                                # block (0 to simd_unroll-simdwidth columns).
                                while j + simdwidth <= j_end:
                                    var c_addr = c_row_base + j
                                    var acc: SIMD[Self.dtype, simdwidth]
                                    if k_tile == 0:
                                        acc = SIMD[Self.dtype, simdwidth](0)
                                    else:
                                        acc = C_data.load[width=simdwidth](c_addr)
                                    for k in range(k_tile, k_end):
                                        var a_ik = SIMD[Self.dtype, simdwidth](
                                            A_data[a_row_base + k]
                                        )
                                        var b_base = k * B_stride0 + B_offset + j
                                        acc = math.fma(
                                            a_ik,
                                            B_data.load[width=simdwidth](b_base),
                                            acc,
                                        )
                                    C_data.store[width=simdwidth](c_addr, acc)
                                    j += simdwidth

                                # ── Scalar tail ───────────────────────────
                                # Remaining 0 to simdwidth-1 columns that
                                # couldn't fill even a single SIMD vector.
                                while j < j_end:
                                    var c_addr = c_row_base + j
                                    var acc: Scalar[Self.dtype]
                                    if k_tile == 0:
                                        acc = 0
                                    else:
                                        acc = C_data[c_addr]
                                    for k in range(k_tile, k_end):
                                        var b_addr = k * B_stride0 + B_offset + j
                                        acc += A_data[a_row_base + k] * B_data[b_addr]
                                    C_data[c_addr] = acc
                                    j += 1

                parallelize[process_contig_contig](num_tiles_i, num_physical_cores())

            else:
                # ══════════════════════════════════════════════════════════
                #  Path 1b: A non-contiguous, B contiguous
                #
                #  A has non-unit column stride (e.g. transposed or sliced).
                #  a_addr = a_row_base + k * A_stride1  (multiply required)
                #  b_base = k * B_stride0 + B_offset + j  (B_stride1 = 1)
                #
                #  Structure identical to 1a except for the a_addr calculation.
                # ══════════════════════════════════════════════════════════
                @parameter
                def process_noncontig_contig(tile_idx: Int):
                    var i_start = tile_idx * Self.TILE_M
                    var i_end   = min(i_start + Self.TILE_M, m)

                    for k_tile in range(0, n, Self.TILE_N):
                        var k_end = min(k_tile + Self.TILE_N, n)

                        comptime if PREFETCH_POLICY != 0:
                            var next_k = k_tile + Self.TILE_N
                            if next_k < n:
                                var next_k_end = min(next_k + Self.TILE_N, n)
                                var lines_issued = 0
                                for k_pre in range(next_k, next_k_end):
                                    if lines_issued >= MAX_PREFETCH_LINES:
                                        break
                                    var row_base = k_pre * B_stride0 + B_offset
                                    var cl = 0
                                    while (
                                        cl < p
                                        and lines_issued < MAX_PREFETCH_LINES
                                    ):
                                        prefetch[prefetch_opts](
                                            B_data + row_base + cl
                                        )
                                        cl += cache_line_elems
                                        lines_issued += 1

                        for j_tile in range(0, p, Self.TILE_P):
                            var j_end = min(j_tile + Self.TILE_P, p)

                            for i in range(i_start, i_end):
                                var a_row_base = i * A_stride0 + A_offset
                                var c_row_base = i * p
                                var j = j_tile

                                while j + simd_unroll <= j_end:
                                    var cj = c_row_base + j

                                    var acc0: SIMD[Self.dtype, simdwidth]
                                    var acc1: SIMD[Self.dtype, simdwidth]
                                    var acc2: SIMD[Self.dtype, simdwidth]
                                    var acc3: SIMD[Self.dtype, simdwidth]

                                    if k_tile == 0:
                                        acc0 = SIMD[Self.dtype, simdwidth](0)
                                        acc1 = SIMD[Self.dtype, simdwidth](0)
                                        acc2 = SIMD[Self.dtype, simdwidth](0)
                                        acc3 = SIMD[Self.dtype, simdwidth](0)
                                    else:
                                        acc0 = C_data.load[width=simdwidth](cj)
                                        acc1 = C_data.load[width=simdwidth](cj + simdwidth)
                                        acc2 = C_data.load[width=simdwidth](cj + simdwidth * 2)
                                        acc3 = C_data.load[width=simdwidth](cj + simdwidth * 3)

                                    for k in range(k_tile, k_end):
                                        # A non-contiguous: stride multiply required
                                        var a_ik = SIMD[Self.dtype, simdwidth](
                                            A_data[a_row_base + k * A_stride1]
                                        )
                                        var b_base = k * B_stride0 + B_offset + j
                                        acc0 = math.fma(a_ik, B_data.load[width=simdwidth](b_base),                 acc0)
                                        acc1 = math.fma(a_ik, B_data.load[width=simdwidth](b_base + simdwidth),     acc1)
                                        acc2 = math.fma(a_ik, B_data.load[width=simdwidth](b_base + simdwidth * 2), acc2)
                                        acc3 = math.fma(a_ik, B_data.load[width=simdwidth](b_base + simdwidth * 3), acc3)

                                    C_data.store[width=simdwidth](cj,                 acc0)
                                    C_data.store[width=simdwidth](cj + simdwidth,     acc1)
                                    C_data.store[width=simdwidth](cj + simdwidth * 2, acc2)
                                    C_data.store[width=simdwidth](cj + simdwidth * 3, acc3)
                                    j += simd_unroll

                                while j + simdwidth <= j_end:
                                    var c_addr = c_row_base + j
                                    var acc: SIMD[Self.dtype, simdwidth]
                                    if k_tile == 0:
                                        acc = SIMD[Self.dtype, simdwidth](0)
                                    else:
                                        acc = C_data.load[width=simdwidth](c_addr)
                                    for k in range(k_tile, k_end):
                                        var a_ik = SIMD[Self.dtype, simdwidth](
                                            A_data[a_row_base + k * A_stride1]
                                        )
                                        var b_base = k * B_stride0 + B_offset + j
                                        acc = math.fma(
                                            a_ik,
                                            B_data.load[width=simdwidth](b_base),
                                            acc,
                                        )
                                    C_data.store[width=simdwidth](c_addr, acc)
                                    j += simdwidth

                                while j < j_end:
                                    var c_addr = c_row_base + j
                                    var acc: Scalar[Self.dtype]
                                    if k_tile == 0:
                                        acc = 0
                                    else:
                                        acc = C_data[c_addr]
                                    for k in range(k_tile, k_end):
                                        var b_addr = k * B_stride0 + B_offset + j
                                        acc += A_data[a_row_base + k * A_stride1] * B_data[b_addr]
                                    C_data[c_addr] = acc
                                    j += 1

                parallelize[process_noncontig_contig](num_tiles_i, num_physical_cores())

        else:
            # ════════════════════════════════════════════════════════════════
            #  Path 2: B non-contiguous
            #
            #  B has non-unit column stride (e.g. after transpose or slice).
            #  SIMD requires contiguous memory — scalar fallback.
            #  B_stride1 must always be respected — never collapsed to +j.
            #
            #  Loop order: i_tile (parallel) → k_tile → j_tile → i → j → k
            #
            #  FIX Issue 1 (previous review): k_tile is outermost inside the
            #  row tile, so A k-strip loads into L1 once per k_tile and is
            #  reused across all j_tiles before eviction.
            #
            #  FIX Issue 2 (this review): j is now tiled by TILE_P.
            #  Previous version iterated all p columns without tiling, meaning
            #  C was touched across its full width per k_tile iteration. For
            #  large p, C doesn't fit in cache and partial sums get evicted
            #  and reloaded between k_tile iterations, defeating the tiling.
            #  With j tiled, the C sub-block for one (k_tile, j_tile) pair
            #  fits in cache and partial sums stay hot.
            #
            #  Partial sum protocol:
            #    k_tile==0: acc starts at 0 (C is zeroed, skip the load)
            #    k_tile >0: acc loads existing partial sum from C_data
            #    After inner k loop: write acc back to C_data for next k_tile
            #
            #  Rows parallelised over physical cores via i_tile.
            # ════════════════════════════════════════════════════════════════
            var num_tiles_i = (m + Self.TILE_M - 1) // Self.TILE_M

            @parameter
            def process_noncontig_b(tile_idx: Int):
                var i_start = tile_idx * Self.TILE_M
                var i_end   = min(i_start + Self.TILE_M, m)

                for i in range(i_start, i_end):
                    var a_row_base = i * A_stride0 + A_offset
                    var c_row_base = i * p

                    # k_tile outermost: load A k-strip once, reuse across all
                    # j_tiles before the next k_tile evicts it from L1.
                    for k_tile in range(0, n, Self.TILE_N):
                        var k_end = min(k_tile + Self.TILE_N, n)

                        # j_tile inside k_tile: keeps the C sub-block for this
                        # (k_tile, j_tile) pair in cache across the inner k loop.
                        # FIX Issue 2: was "for j in range(p)" — untiled.
                        for j_tile in range(0, p, Self.TILE_P):
                            var j_end = min(j_tile + Self.TILE_P, p)

                            for j in range(j_tile, j_end):
                                var c_addr = c_row_base + j

                                # k_tile==0: C is zeroed, skip the load.
                                # k_tile >0: load partial sum from previous
                                #            k_tile iteration.
                                var acc: Scalar[Self.dtype]
                                if k_tile == 0:
                                    acc = 0
                                else:
                                    acc = C_data[c_addr]

                                for k in range(k_tile, k_end):
                                    var a_addr = a_row_base + k * A_stride1
                                    # B non-contiguous: B_stride1 != 1, must multiply
                                    var b_addr = k * B_stride0 + B_offset + j * B_stride1
                                    acc += A_data[a_addr] * B_data[b_addr]

                                # Write partial sum back for next k_tile
                                # iteration to load as its starting value.
                                C_data[c_addr] = acc

            parallelize[process_noncontig_b](num_tiles_i, num_physical_cores())

        return C^
