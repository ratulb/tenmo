from .matrixshapevalidator import MatrixShapeValidator
from .shared.shapes import Shape
from .shared.layout import Layout
from .shared.broadcasthelper import ShapeBroadcaster
from .shared.buffers import Buffer
from .shared.panic import panic
from max.algorithm import parallelize
from std.sys import prefetch, PrefetchOptions, simd_width_of, size_of
from std.sys.info import num_physical_cores
from std import math

#  Tuning constants
#
#  PREFETCH_POLICY : 0 = Off (no prefetch instructions emitted),
#                    1 = Conservative (light prefetch, capped at 32 lines),
#                    2 = Aggressive (full prefetch, up to 128 lines)
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

comptime PREFETCH_POLICY = 1  # 0=Off, 1=Conservative, 2=Aggressive
comptime prefetch_opts = PrefetchOptions().for_read().high_locality().to_data_cache()
comptime MAX_PREFETCH_LINES = 32 if PREFETCH_POLICY <= 1 else 128
comptime UNROLL = 4

comptime matmulFn[dtype: DType] = def(
    Layout, Buffer[dtype], Layout, Buffer[dtype]
) thin -> Tuple[Layout, Buffer[dtype]]

trait MatmulCpu:
    """ MatmulCpu trait.
     Common interface for CPU matmul implementations. Both 2D (MmCpu2d) and
     ND-batched (MmCpuNd) implement this trait. The per-shape tile-size
     dispatch — 3 × 2 × 3 = 18 (TILE_M, TILE_N, TILE_P) combinations — lives
     here ONCE in mm_fn(); each concrete struct only supplies:
       · comptime datatype      — its DType
       · matmul_for[TM, TN, TP]()  — comptime-specialized matmul() fn pointer
       · matmul()               — the actual tiled kernel for its own tiles
     tiled_matmul() on each struct then collapses to a thin wrapper:
       return Self.mm_fn(m, n, p)(A_layout, A_buffer, B_layout, B_buffer)
    """
    comptime datatype: DType

    # Return the comptime-specialized matmul() kernel (as a thin function
    # pointer) for the given tile sizes. Each conformer implements e.g.
    # `return MmCpu2d[Self.datatype, TM, TN, TP].matmul`. This is what lets
    # the shared dispatch reach the correct comptime-specialized kernel
    # without hard-coding a concrete struct name here.
    @staticmethod
    def matmul_for[TM: Int, TN: Int, TP: Int]() -> matmulFn[Self.datatype]:
        ...

    @staticmethod
    def mm_fn(m: Int, n: Int, p: Int) -> matmulFn[Self.datatype]:
        var tile_m = 128 if m > 256 else (64 if m > 64 else 32)
        var tile_n = 64 if n > 64 else 32
        var tile_p = 256 if p > 256 else (128 if p > 64 else 64)

        if tile_m == 128:
            if tile_n == 64:
                if tile_p == 256:
                    return Self.matmul_for[128, 64, 256]()
                elif tile_p == 128:
                    return Self.matmul_for[128, 64, 128]()
                else:
                    return Self.matmul_for[128, 64, 64]()
            else:
                if tile_p == 256:
                    return Self.matmul_for[128, 32, 256]()
                elif tile_p == 128:
                    return Self.matmul_for[128, 32, 128]()
                else:
                    return Self.matmul_for[128, 32, 64]()
        elif tile_m == 64:
            if tile_n == 64:
                if tile_p == 256:
                    return Self.matmul_for[64, 64, 256]()
                elif tile_p == 128:
                    return Self.matmul_for[64, 64, 128]()
                else:
                    return Self.matmul_for[64, 64, 64]()
            else:
                if tile_p == 256:
                    return Self.matmul_for[64, 32, 256]()
                elif tile_p == 128:
                    return Self.matmul_for[64, 32, 128]()
                else:
                    return Self.matmul_for[64, 32, 64]()
        else:
            if tile_n == 64:
                if tile_p == 256:
                    return Self.matmul_for[32, 64, 256]()
                elif tile_p == 128:
                    return Self.matmul_for[32, 64, 128]()
                else:
                    return Self.matmul_for[32, 64, 64]()
            else:
                if tile_p == 256:
                    return Self.matmul_for[32, 32, 256]()
                elif tile_p == 128:
                    return Self.matmul_for[32, 32, 128]()
                else:
                    return Self.matmul_for[32, 32, 64]()

    @staticmethod
    def matmul(
        A_layout: Layout,
        A_buffer: Buffer[Self.datatype],
        B_layout: Layout,
        B_buffer: Buffer[Self.datatype],
    ) -> Tuple[Layout, Buffer[Self.datatype]]:
        ...

def matmul_simd_tile[
    dtype: DType,
    TILE_N: Int,
    TILE_P: Int,
    A_STRIDED: Bool,
](
    A_data: Pointer[Scalar[dtype], MutAnyOrigin],
    B_data: Pointer[Scalar[dtype], MutAnyOrigin],
    C_data: Pointer[Scalar[dtype], MutAnyOrigin],
    A_base_off: Int,
    A_row_stride: Int,
    A_col_stride: Int,
    B_base_off: Int,
    B_row_stride: Int,
    C_base_off: Int,
    m: Int,
    n: Int,
    p: Int,
    i_start: Int,
    i_end: Int,
    C_stride: Int,
    k_origin: Int = 0,
    j_origin: Int = 0,
):
    """ Extracted tile kernels (shared by MmCpu2d && MmCpuNd)
     The "given data pointers + base offsets + strides + shapes, compute this
     i-tile of C = A @ B" logic. Both structs' matmul entries pass batch-adjusted
     base offsets in; nothing here knows about batches or which struct called.
     Deliberately a plain function extraction with explicit parameters — no trait
     dispatch — so the SIMD loop bodies exist exactly once.
     A_STRIDED is the comptime switch that collapses the two contiguous-B paths
     (1a vs 1b): their only difference is `a_row_base + k` vs
     `a_row_base + k * A_col_stride`. ONE SIMD body instead of the historic
     1a/1b pair per struct.
    """
    comptime simdwidth = simd_width_of[dtype]()
    comptime simd_unroll = simdwidth * UNROLL
    comptime cache_line_elems = 64 // size_of[Scalar[dtype]]()

    for k_tile in range(0, n, TILE_N):
        var k_end = min(k_tile + TILE_N, n)

        # Prefetch next k-tile of B (B is guaranteed contiguous here)
        comptime if PREFETCH_POLICY != 0:
            var next_k = k_tile + TILE_N
            if next_k < n:
                var lines_issued = 0
                for k_pre in range(next_k, next_k + TILE_N):
                    if lines_issued >= MAX_PREFETCH_LINES:
                        break
                    var row_base = k_pre * B_row_stride + B_base_off
                    var cl = 0
                    while cl < p and lines_issued < MAX_PREFETCH_LINES:
                        prefetch[prefetch_opts](
                            B_data.unsafe_offset(row_base + cl)
                        )
                        cl += cache_line_elems
                        lines_issued += 1

        for j_tile in range(0, p, TILE_P):
            var j_end = min(j_tile + TILE_P, p)

            for i in range(i_start, i_end):
                var a_row_base = A_base_off + i * A_row_stride
                var c_row_base = C_base_off + i * C_stride
                var j = j_tile

                # Unrolled SIMD: UNROLL vectors per iter
                while j + simd_unroll <= j_end:
                    var cj = c_row_base + j + j_origin

                    var acc0: SIMD[dtype, simdwidth]
                    var acc1: SIMD[dtype, simdwidth]
                    var acc2: SIMD[dtype, simdwidth]
                    var acc3: SIMD[dtype, simdwidth]

                    # k_tile==0: C is zeroed, skip the load.
                    if k_tile + k_origin == 0:
                        acc0 = SIMD[dtype, simdwidth](0)
                        acc1 = SIMD[dtype, simdwidth](0)
                        acc2 = SIMD[dtype, simdwidth](0)
                        acc3 = SIMD[dtype, simdwidth](0)
                    else:
                        acc0 = C_data.unsafe_load[width=simdwidth](cj)
                        acc1 = C_data.unsafe_load[width=simdwidth](cj + simdwidth)
                        acc2 = C_data.unsafe_load[width=simdwidth](
                            cj + simdwidth * 2
                        )
                        acc3 = C_data.unsafe_load[width=simdwidth](
                            cj + simdwidth * 3
                        )

                    for k in range(k_tile, k_end):
                        var a_col = k + k_origin
                        comptime if A_STRIDED:
                            a_col = (k + k_origin) * A_col_stride
                        var a_ik = SIMD[dtype, simdwidth](
                            A_data[unsafe_offset=a_row_base + a_col]
                        )
                        var b_base = k * B_row_stride + B_base_off + j
                        acc0 = math.fma(
                            a_ik,
                            B_data.unsafe_load[width=simdwidth](b_base),
                            acc0,
                        )
                        acc1 = math.fma(
                            a_ik,
                            B_data.unsafe_load[width=simdwidth](
                                b_base + simdwidth
                            ),
                            acc1,
                        )
                        acc2 = math.fma(
                            a_ik,
                            B_data.unsafe_load[width=simdwidth](
                                b_base + simdwidth * 2
                            ),
                            acc2,
                        )
                        acc3 = math.fma(
                            a_ik,
                            B_data.unsafe_load[width=simdwidth](
                                b_base + simdwidth * 3
                            ),
                            acc3,
                        )

                    C_data.unsafe_store[width=simdwidth](cj, acc0)
                    C_data.unsafe_store[width=simdwidth](cj + simdwidth, acc1)
                    C_data.unsafe_store[width=simdwidth](
                        cj + simdwidth * 2, acc2
                    )
                    C_data.unsafe_store[width=simdwidth](
                        cj + simdwidth * 3, acc3
                    )
                    j += simd_unroll

                # Single-vector SIMD tail
                while j + simdwidth <= j_end:
                    var c_addr = c_row_base + j + j_origin
                    var acc: SIMD[dtype, simdwidth]
                    if k_tile + k_origin == 0:
                        acc = SIMD[dtype, simdwidth](0)
                    else:
                        acc = C_data.unsafe_load[width=simdwidth](c_addr)
                    for k in range(k_tile, k_end):
                        var a_col = k + k_origin
                        comptime if A_STRIDED:
                            a_col = (k + k_origin) * A_col_stride
                        var a_ik = SIMD[dtype, simdwidth](
                            A_data[unsafe_offset=a_row_base + a_col]
                        )
                        var b_base = k * B_row_stride + B_base_off + j
                        acc = math.fma(
                            a_ik,
                            B_data.unsafe_load[width=simdwidth](b_base),
                            acc,
                        )
                    C_data.unsafe_store[width=simdwidth](c_addr, acc)
                    j += simdwidth

                # Scalar tail
                while j < j_end:
                    var c_addr = c_row_base + j + j_origin
                    var acc: Scalar[dtype]
                    if k_tile + k_origin == 0:
                        acc = 0
                    else:
                        acc = C_data[unsafe_offset=c_addr]
                    for k in range(k_tile, k_end):
                        var a_col = k + k_origin
                        comptime if A_STRIDED:
                            a_col = (k + k_origin) * A_col_stride
                        var b_addr = k * B_row_stride + B_base_off + j
                        acc += (
                            A_data[unsafe_offset=a_row_base + a_col]
                            * B_data[unsafe_offset=b_addr]
                        )
                    C_data[unsafe_offset=c_addr] = acc
                    j += 1


def matmul_scalar_tile[
    dtype: DType,
    TILE_N: Int,
    TILE_P: Int,
](
    A_data: Pointer[Scalar[dtype], MutAnyOrigin],
    B_data: Pointer[Scalar[dtype], MutAnyOrigin],
    C_data: Pointer[Scalar[dtype], MutAnyOrigin],
    A_base_off: Int,
    A_row_stride: Int,
    A_col_stride: Int,
    B_base_off: Int,
    B_row_stride: Int,
    B_col_stride: Int,
    C_base_off: Int,
    m: Int,
    n: Int,
    p: Int,
    i_start: Int,
    i_end: Int,
):
    """ Path 2: B non-contiguous — pack + SIMD
     SIMD needs contiguous B rows, so B's inner stride kills the fast path. This
     stays cheap by packing each (k_tile, j_tile) block of B into a ~TILE_N×TILE_P
     contiguous scratch tile ONCE per i-tile, then running the SAME fast SIMD
     kernel (matmul_simd_tile, with A_STRIDED comptime-selected and the original
     k_tile passed as k_origin so the "first k-tile zeroes C" protocol holds) over
     the packed tile. Pack cost is amortised across all i rows of the tile, so the
     SIMD+FMA win is recovered for the common transposed-B / strided-B shapes
     (every matmul backward's dL/dA = grad_out @ B^T).
    """
    var packed = Buffer[dtype](TILE_N * TILE_P)
    var packed_ptr = packed.unsafe_ptr()

    for k_tile in range(0, n, TILE_N):
        var k_end = min(k_tile + TILE_N, n)
        var klen = k_end - k_tile

        for j_tile in range(0, p, TILE_P):
            var j_end = min(j_tile + TILE_P, p)
            var jlen = j_end - j_tile

            # Pack B[k_tile:k_end, j_tile:j_end] → packed[0:klen, 0:jlen]
            for kk in range(klen):
                var row_base = (
                    B_base_off
                    + (k_tile + kk) * B_row_stride
                    + j_tile * B_col_stride
                )
                for jj in range(jlen):
                    packed_ptr[unsafe_offset=kk * jlen + jj] = B_data[
                        unsafe_offset=row_base + jj * B_col_stride
                    ]

            # SIMD over the packed tile, reusing the fast kernel
            # j_tile is passed as j_origin: the packed tile holds B-columns
            # [j_tile, j_tile+jlen), but the inner kernel iterates packed-local
            # j in [0, jlen) — without the origin it would store every j-tile
            # to C-columns [0, jlen), overwriting earlier tiles and leaving
            # columns past the first tile zero (silent corruption for any
            # strided-B matmul with p > TILE_P, e.g. every matmul backward
            # dL/dA = grad_out @ B^T with wide B).
            if A_col_stride == 1:
                matmul_simd_tile[dtype, TILE_N, TILE_P, False](
                    A_data,
                    packed_ptr,
                    C_data,
                    A_base_off,
                    A_row_stride,
                    A_col_stride,
                    0,
                    jlen,
                    C_base_off,
                    m,
                    klen,
                    jlen,
                    i_start,
                    i_end,
                    p,
                    k_tile,
                    j_tile,
                )
            else:
                matmul_simd_tile[dtype, TILE_N, TILE_P, True](
                    A_data,
                    packed_ptr,
                    C_data,
                    A_base_off,
                    A_row_stride,
                    A_col_stride,
                    0,
                    jlen,
                    C_base_off,
                    m,
                    klen,
                    jlen,
                    i_start,
                    i_end,
                    p,
                    k_tile,
                    j_tile,
                )


#  Path 2b: B non-contiguous — pack each j-panel ONCE per batch
#
#  matmul_scalar_tile (Path 2a) packs every (k_tile, j_tile) block inside
#  the i-tile loop, so each B block is re-packed once per i-tile
#  (ceil(m/TILE_M) times per call). The packed bytes are identical across
#  i-tiles — only the A rows and C rows differ. This helper packs the full
#  j-panel B[:, j_start:j_start+jlen] once into a contiguous n×jlen
#  thread-local buffer, then issues ONE matmul_simd_tile call over the full
#  i range [0, m): the kernel's own k/j/i loops do the rest, with
#  j_origin=j_start routing stores to the right C columns and k_origin=0
#  preserving the "first k-tile zeroes C" protocol (C arrives zeroed).
#  Per-element pack work drops from n·p·num_tiles_i to n·p; A traffic drops
#  too (each A row read once, not once per j-tile).
#
#  Callers engage this only when the i-loop actually re-packs
#  (num_tiles_i > 1) and the j-space covers min(i-workers, cores)
#  (adaptive rule — single-i-tile shapes pack each block exactly once
#  already, so panel would only add oversubscription there; they keep
#  per-tile packing bit-for-bit). Panels above PANEL_CAP_ELEMS fall back
#  to per-tile packing (bounded workspace). A_STRIDED follows the same
#  A_col_stride != 1 rule matmul_scalar_tile uses internally.
comptime PANEL_CAP_ELEMS = 1 << 20


def matmul_panel_tile[
    dtype: DType,
    TILE_N: Int,
    TILE_P: Int,
](
    A_data: Pointer[Scalar[dtype], MutAnyOrigin],
    B_data: Pointer[Scalar[dtype], MutAnyOrigin],
    C_data: Pointer[Scalar[dtype], MutAnyOrigin],
    A_base_off: Int,
    A_row_stride: Int,
    A_col_stride: Int,
    B_base_off: Int,
    B_row_stride: Int,
    B_col_stride: Int,
    C_base_off: Int,
    C_stride: Int,
    m: Int,
    n: Int,
    j_start: Int,
    jlen: Int,
    A_strided: Bool,
):
    var panel = Buffer[dtype](n * jlen)
    var panel_ptr = panel.unsafe_ptr()

    # Pack B[0:n, j_start:j_start+jlen] → panel (contiguous, row stride jlen)
    for kk in range(n):
        var row_base = (
            B_base_off + kk * B_row_stride + j_start * B_col_stride
        )
        for jj in range(jlen):
            panel_ptr[unsafe_offset=kk * jlen + jj] = B_data[
                unsafe_offset=row_base + jj * B_col_stride
            ]

    # One SIMD call over the full i range; j_origin routes C stores
    if A_strided:
        matmul_simd_tile[dtype, TILE_N, TILE_P, True](
            A_data,
            panel_ptr,
            C_data,
            A_base_off,
            A_row_stride,
            A_col_stride,
            0,
            jlen,
            C_base_off,
            m,
            n,
            jlen,
            0,
            m,
            C_stride,
            0,
            j_start,
        )
    else:
        matmul_simd_tile[dtype, TILE_N, TILE_P, False](
            A_data,
            panel_ptr,
            C_data,
            A_base_off,
            A_row_stride,
            A_col_stride,
            0,
            jlen,
            C_base_off,
            m,
            n,
            jlen,
            0,
            m,
            C_stride,
            0,
            j_start,
        )


struct MmCpu2d[
    dtype: DType, TILE_M: Int = 32, TILE_N: Int = 32, TILE_P: Int = 32
](MatmulCpu):
    comptime datatype = Self.dtype

    @staticmethod
    def matmul_for[TM: Int, TN: Int, TP: Int]() -> matmulFn[Self.datatype]:
        return MmCpu2d[Self.dtype, TM, TN, TP].matmul

    #  tiled_matmul
    #
    #  Per-shape tile dispatch now lives in the shared MatmulCpu.mm_fn(), which
    #  picks the (TILE_M, TILE_N, TILE_P) for the given m/n/p once — all 3 × 2
    #  × 3 = 18 combinations, dispatched per dimension — and returns the
    #  comptime-specialized matmul() kernel as a thin function pointer, which
    #  is applied here. Each dimension selects its own tile size based on its
    #  own size:
    #    TILE_M: driven by m  — controls row parallelism granularity
    #    TILE_N: driven by n  — controls A k-strip cache residency in L1
    #    TILE_P: driven by p  — MOST CRITICAL: must be >= simd_unroll
    #                           (simdwidth * UNROLL = 32 for float32/AVX2)
    #                           for the unrolled SIMD loop to fire at all.
    #
    #  All 18 combinations are explicitly enumerated. A final fallback
    #  panics on any unanticipated combination rather than silently
    #  using wrong tile sizes.
    @staticmethod
    def tiled_matmul(
        A_layout: Layout,
        A_buffer: Buffer[Self.dtype],
        B_layout: Layout,
        B_buffer: Buffer[Self.dtype],
    ) -> Tuple[Layout, Buffer[Self.dtype]]:
        var m = A_layout.shape[0]
        var n = A_layout.shape[1]
        var p = B_layout.shape[1]
        return Self.mm_fn(m, n, p)(A_layout, A_buffer, B_layout, B_buffer)

    #  matmul
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
    #    2a. B non-contiguous             → pack each tile per i-tile
    #    2b. B non-contiguous, j-space wide → pack each j-panel once (adaptive)
    #
    #  k_tile==0 split:
    #    In paths 1a and 1b the k_tile==0 branch sits inside the j loop.
    #    The compiler may hoist it but is not guaranteed to. For path 2
    #    (scalar, simpler structure) the branch is inside j which is inside
    #    k_tile — straightforward and correct.
    #    If profiling shows branch overhead in paths 1a/1b, split the j loop
    #    into a k_tile==0 copy and a k_tile>0 copy to guarantee hoisting.
    @staticmethod
    def matmul(
        A_layout: Layout,
        A_buffer: Buffer[Self.dtype],
        B_layout: Layout,
        B_buffer: Buffer[Self.dtype],
    ) -> Tuple[Layout, Buffer[Self.dtype]]:
        ref A_shape = A_layout.shape
        ref B_shape = B_layout.shape
        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

        var m = A_shape[0]
        var n = A_shape[1]
        var p = B_shape[1]

        # C is zero-initialised. The k_tile==0 fast path (skip C load) relies
        # on this invariant. Do not change this allocation.
        var C_storage = Buffer[Self.dtype].zeros(m * p)

        # Hoist all pointer and stride metadata
        # Accessing struct fields inside a hot loop forces repeated loads.
        # Storing them in locals lets the compiler keep them in registers.
        ref A_strides = A_layout.strides
        var A_stride0 = A_strides[0]  # elements to advance one row in A
        var A_stride1 = A_strides[1]  # elements to advance one col in A
        var A_offset = A_layout.offset
        var A_data = A_buffer.unsafe_ptr()

        ref B_strides = B_layout.strides
        var B_stride0 = B_strides[0]  # elements to advance one row in B
        var B_stride1 = B_strides[
            1
        ]  # elements to advance one col in B (1 if contiguous)
        var B_offset = B_layout.offset
        var B_data = B_buffer.unsafe_ptr()

        var C_data = C_storage.unsafe_ptr()
        # C is always freshly allocated and contiguous:
        #   C_stride0 = p  (one full row)
        #   C_stride1 = 1  (adjacent columns)
        # These are inlined as literals below.

        # Parallel space
        # Primary axis is rows (i-tiles), one thread per i-tile. When m is small
        # (e.g. m <= TILE_M so num_tiles_i == 1), a pure i-axis space leaves every
        # core but one idle no matter how large n/p are -- exactly the shape a
        # small-batch inference call or a narrow projection layer hits (both are
        # rank-2 matmuls routed to MmCpu2d). To recover MmCpuNd's core utilization
        # for that shape, we add a SECOND axis over j (columns), flattening
        # (i_tile x j_tile) into one parallel space. Each (i_tile, j_tile) cell of
        # C is fully independent -- the k-reduction is contained inside a single
        # kernel call -- so the split is embarrassingly parallel with no
        # cross-j-tile accumulation or reduction.
        #
        # Guardrail: only engage the j-axis when the i-axis alone can't fill the
        # machine (num_tiles_i < num_physical_cores()). For large m the extra
        # tile-decode division would be pure overhead on the hot path, so those
        # shapes keep the single-axis i-only space.
        var num_tiles_i = (m + Self.TILE_M - 1) // Self.TILE_M
        var num_tiles_j = (p + Self.TILE_P - 1) // Self.TILE_P
        var n_cores = num_physical_cores()
        var split_j = num_tiles_i < n_cores

        if B_layout.is_contiguous():
            # Paths 1a / 1b: B contiguous → SIMD + FMA + unroll + prefetch
            # A_STRIDED=False picks `a_row_base + k` (no multiply); True picks
            # `a_row_base + k * A_col_stride`. One kernel body, comptime-selected.
            if A_layout.is_contiguous():
                def process_contig_contig(tile_idx: Int) {imm}:
                    var i_start = tile_idx * Self.TILE_M
                    var i_end = min(i_start + Self.TILE_M, m)
                    matmul_simd_tile[
                        Self.dtype, Self.TILE_N, Self.TILE_P, False
                    ](
                        A_data,
                        B_data,
                        C_data,
                        A_offset,
                        A_stride0,
                        A_stride1,
                        B_offset,
                        B_stride0,
                        0,
                        m,
                        n,
                        p,
                        i_start,
                        i_end,
                        p,
                    )

                def process_contig_contig_j(tile_idx: Int) {imm}:
                    var i_tile = tile_idx // num_tiles_j
                    var j_tile = tile_idx % num_tiles_j
                    var j_start = j_tile * Self.TILE_P
                    var j_extent = min(j_start + Self.TILE_P, p) - j_start
                    var i_start = i_tile * Self.TILE_M
                    var i_end = min(i_start + Self.TILE_M, m)
                    matmul_simd_tile[
                        Self.dtype, Self.TILE_N, Self.TILE_P, False
                    ](
                        A_data,
                        B_data,
                        C_data,
                        A_offset,
                        A_stride0,
                        A_stride1,
                        B_offset + j_start,
                        B_stride0,
                        j_start,
                        m,
                        n,
                        j_extent,
                        i_start,
                        i_end,
                        p,
                    )

                if split_j:
                    parallelize(process_contig_contig_j, 
                        num_tiles_i * num_tiles_j, n_cores
                    )
                else:
                    parallelize(process_contig_contig, 
                        num_tiles_i, n_cores
                    )
            else:
                def process_noncontig_contig(tile_idx: Int) {imm}:
                    var i_start = tile_idx * Self.TILE_M
                    var i_end = min(i_start + Self.TILE_M, m)
                    matmul_simd_tile[
                        Self.dtype, Self.TILE_N, Self.TILE_P, True
                    ](
                        A_data,
                        B_data,
                        C_data,
                        A_offset,
                        A_stride0,
                        A_stride1,
                        B_offset,
                        B_stride0,
                        0,
                        m,
                        n,
                        p,
                        i_start,
                        i_end,
                        p,
                    )

                def process_noncontig_contig_j(tile_idx: Int) {imm}:
                    var i_tile = tile_idx // num_tiles_j
                    var j_tile = tile_idx % num_tiles_j
                    var j_start = j_tile * Self.TILE_P
                    var j_extent = min(j_start + Self.TILE_P, p) - j_start
                    var i_start = i_tile * Self.TILE_M
                    var i_end = min(i_start + Self.TILE_M, m)
                    matmul_simd_tile[
                        Self.dtype, Self.TILE_N, Self.TILE_P, True
                    ](
                        A_data,
                        B_data,
                        C_data,
                        A_offset,
                        A_stride0,
                        A_stride1,
                        B_offset + j_start,
                        B_stride0,
                        j_start,
                        m,
                        n,
                        j_extent,
                        i_start,
                        i_end,
                        p,
                    )

                if split_j:
                    parallelize(process_noncontig_contig_j, 
                        num_tiles_i * num_tiles_j, n_cores
                    )
                else:
                    parallelize(process_noncontig_contig, 
                        num_tiles_i, n_cores
                    )

        else:
            # Path 2a: B non-contiguous → pack each tile per i-tile;
            # Path 2b (panel-pack, one pack per j-tile) is selected by the
            # use_panel adaptive rule below
            def process_noncontig_b(tile_idx: Int) {imm}:
                var i_start = tile_idx * Self.TILE_M
                var i_end = min(i_start + Self.TILE_M, m)
                matmul_scalar_tile[
                    Self.dtype, Self.TILE_N, Self.TILE_P
                ](
                    A_data,
                    B_data,
                    C_data,
                    A_offset,
                    A_stride0,
                    A_stride1,
                    B_offset,
                    B_stride0,
                    B_stride1,
                    0,
                    m,
                    n,
                    p,
                    i_start,
                    i_end,
                )

            def process_noncontig_b_j(tile_idx: Int) {imm}:
                var i_tile = tile_idx // num_tiles_j
                var j_tile = tile_idx % num_tiles_j
                var j_start = j_tile * Self.TILE_P
                var j_extent = min(j_start + Self.TILE_P, p) - j_start
                var i_start = i_tile * Self.TILE_M
                var i_end = min(i_start + Self.TILE_M, m)
                matmul_scalar_tile[
                    Self.dtype, Self.TILE_N, Self.TILE_P
                ](
                    A_data,
                    B_data,
                    C_data,
                    A_offset,
                    A_stride0,
                    A_stride1,
                    B_offset + j_start * B_stride1,
                    B_stride0,
                    B_stride1,
                    j_start,
                    m,
                    n,
                    j_extent,
                    i_start,
                    i_end,
                )

            # Path 2b: panel-pack (one pack per j-tile, full-i SIMD)
            # Same gate as MmCpuNd: ≥2 j-tiles, the i-loop must actually
            # re-pack (num_tiles_i > 1), and the j-space must cover
            # min(workers, cores). Single-j/single-i shapes (incl. the
            # split_j small-m case) keep Path 2a bit-for-bit. Panel must
            # fit the workspace cap.
            var use_panel = False
            if num_tiles_j >= 2:
                if num_tiles_i > 1:
                    var panel_workers = num_tiles_i
                    if split_j:
                        panel_workers = num_tiles_i * num_tiles_j
                    var need = panel_workers
                    var have = num_tiles_j
                    if n_cores < need:
                        need = n_cores
                    if have >= need:
                        use_panel = True
            if n * Self.TILE_P > PANEL_CAP_ELEMS:
                use_panel = False

            def process_noncontig_b_panel(tile_idx: Int) {imm}:
                var j_start = tile_idx * Self.TILE_P
                var jlen = min(j_start + Self.TILE_P, p) - j_start
                matmul_panel_tile[Self.dtype, Self.TILE_N, Self.TILE_P](
                    A_data,
                    B_data,
                    C_data,
                    A_offset,
                    A_stride0,
                    A_stride1,
                    B_offset,
                    B_stride0,
                    B_stride1,
                    0,
                    p,
                    m,
                    n,
                    j_start,
                    jlen,
                    A_stride1 != 1,
                )

            if use_panel:
                parallelize(process_noncontig_b_panel,
                    num_tiles_j, n_cores
                )
            elif split_j:
                parallelize(process_noncontig_b_j, 
                    num_tiles_i * num_tiles_j, n_cores
                )
            else:
                parallelize(process_noncontig_b, 
                    num_tiles_i, n_cores
                )
        return (Layout(Shape(m, p)), C_storage)


struct MmCpuNd[
    dtype: DType,
    TILE_M: Int = 32,
    TILE_N: Int = 32,
    TILE_P: Int = 64,
](MatmulCpu):
    comptime datatype = Self.dtype

    @staticmethod
    def matmul_for[TM: Int, TN: Int, TP: Int]() -> matmulFn[Self.datatype]:
        return MmCpuNd[Self.dtype, TM, TN, TP].matmul

    #  tiled_matmul
    #
    #  Same shared dispatch as MmCpu2d (MatmulCpu.mm_fn); only the m/n/p
    #  extraction differs — they come from the inner two dims of the ND
    #  layouts. Batch dims never influence tile selection: they are
    #  parallelised at the outer loop level regardless of tile config.
    @staticmethod
    def tiled_matmul(
        A_layout: Layout,
        A_buffer: Buffer[Self.dtype],
        B_layout: Layout,
        B_buffer: Buffer[Self.dtype],
    ) -> Tuple[Layout, Buffer[Self.dtype]]:
        var A_rank = A_layout.shape.rank()
        var B_rank = B_layout.shape.rank()
        var m = A_layout.shape[A_rank - 2]
        var n = A_layout.shape[A_rank - 1]
        var p = B_layout.shape[B_rank - 1]
        return Self.mm_fn(m, n, p)(A_layout, A_buffer, B_layout, B_buffer)

    @staticmethod
    def matmul(
        A_layout: Layout,
        A_buffer: Buffer[Self.dtype],
        B_layout: Layout,
        B_buffer: Buffer[Self.dtype],
    ) -> Tuple[Layout, Buffer[Self.dtype]]:
        var A_shape = A_layout.shape
        var B_shape = B_layout.shape

        var A_rank = A_shape.rank()
        var B_rank = B_shape.rank()

        var m = A_shape[A_rank - 2]
        var k_A = A_shape[A_rank - 1]
        var k_B = B_shape[B_rank - 2]
        var p = B_shape[B_rank - 1]

        if k_A != k_B:
            panic(
                "NDBuffer → matmul_nd: inner dims must match, got "
                + String(k_A)
                + " and "
                + String(k_B)
            )

        var k = k_A

        # Batch shapes and broadcasting
        var A_batch_shape = A_shape[:-2]
        var B_batch_shape = B_shape[:-2]

        if not ShapeBroadcaster.broadcastable(A_batch_shape, B_batch_shape):
            panic(
                "NDBuffer → matmul_nd: batch shapes not broadcastable: "
                + String(A_batch_shape)
                + " vs "
                + String(B_batch_shape)
            )

        var batch_shape = ShapeBroadcaster.broadcast_shape(
            A_batch_shape, B_batch_shape
        )
        var total_batch = batch_shape.product()
        if total_batch == 0:
            total_batch = 1

        # Output
        var out_shape = batch_shape + Shape(m, p)
        var C_storage = Buffer[Self.dtype].zeros(out_shape.num_elements())

        # Hoist all metadata out of the parallel loop
        var A_batch_rank = A_batch_shape.rank()
        var B_batch_rank = B_batch_shape.rank()
        var batch_rank = batch_shape.rank()

        var A_batch_strides = A_layout.strides[:-2]
        var B_batch_strides = B_layout.strides[:-2]

        var A_row_stride = A_layout.strides[A_rank - 2]
        var A_col_stride = A_layout.strides[A_rank - 1]
        var B_row_stride = B_layout.strides[B_rank - 2]
        var B_col_stride = B_layout.strides[B_rank - 1]

        var A_offset = A_layout.offset
        var B_offset = B_layout.offset

        var A_data = A_buffer.unsafe_ptr()
        var B_data = B_buffer.unsafe_ptr()
        var C_data = C_storage.unsafe_ptr()
        # C always contiguous, offset 0, inner strides (p, 1)

        var B_contiguous = B_layout.is_contiguous()
        var A_contiguous = A_layout.is_contiguous()

        # Parallelise over batch × m-tiles
        var num_tiles_i = (m + Self.TILE_M - 1) // Self.TILE_M
        var num_tiles_j = (p + Self.TILE_P - 1) // Self.TILE_P
        var total_tiles = total_batch * num_tiles_i

        # Path 2b (panel-pack) engages iff B is strided, the j-space has
        # ≥2 tiles (single-j shapes pack each block exactly once already —
        # panel only adds oversubscription there), the i-loop actually
        # re-packs (num_tiles_i > 1), and the j-space covers min(i-workers,
        # cores) so the machine stays saturated. Anything else keeps Path
        # 2a bit-for-bit; the panel must fit the workspace cap.
        var use_panel = False
        if not B_contiguous:
            if num_tiles_j >= 2:
                if num_tiles_i > 1:
                    var need = total_batch * num_tiles_i
                    var have = total_batch * num_tiles_j
                    var cores = num_physical_cores()
                    if cores < need:
                        need = cores
                    if have >= need:
                        use_panel = True
        if k * Self.TILE_P > PANEL_CAP_ELEMS:
            use_panel = False

        def process_tile(flat_idx: Int) {imm}:
            var batch = flat_idx // num_tiles_i
            var tile_idx = flat_idx % num_tiles_i

            # Decode batch → A/B base offsets via pure arithmetic
            var A_base_off = A_offset
            var B_base_off = B_offset

            if batch_rank > 0:
                var remaining = batch
                var divisor = 1
                for d in range(batch_rank - 1, -1, -1):
                    var coord = (remaining // divisor) % batch_shape[d]

                    var A_d = d - (batch_rank - A_batch_rank)
                    if A_d >= 0 and A_batch_shape[A_d] > 1:
                        A_base_off += coord * A_batch_strides[A_d]

                    var B_d = d - (batch_rank - B_batch_rank)
                    if B_d >= 0 and B_batch_shape[B_d] > 1:
                        B_base_off += coord * B_batch_strides[B_d]

                    divisor *= batch_shape[d]

            var C_base_off = batch * m * p

            # Tiled matmul for this (batch, i-tile)
            var i_start = tile_idx * Self.TILE_M
            var i_end = min(i_start + Self.TILE_M, m)

            if B_contiguous:
                if A_contiguous:
                    # A_STRIDED=False picks `a_row_base + k`; True picks
                    # `a_row_base + k * A_col_stride`. One kernel body, comptime-selected.
                    matmul_simd_tile[Self.dtype, Self.TILE_N, Self.TILE_P, False](
                        A_data,
                        B_data,
                        C_data,
                        A_base_off,
                        A_row_stride,
                        A_col_stride,
                        B_base_off,
                        B_row_stride,
                        C_base_off,
                        m,
                        k,
                        p,
                        i_start,
                        i_end,
                        p,
                    )
                else:
                    matmul_simd_tile[Self.dtype, Self.TILE_N, Self.TILE_P, True](
                        A_data,
                        B_data,
                        C_data,
                        A_base_off,
                        A_row_stride,
                        A_col_stride,
                        B_base_off,
                        B_row_stride,
                        C_base_off,
                        m,
                        k,
                        p,
                        i_start,
                        i_end,
                        p,
                    )
            else:
                # Path 2a: B non-contiguous -> pack each tile per i-tile
                # (Path 2b panel-pack lives in the use_panel branch below)
                matmul_scalar_tile[Self.dtype, Self.TILE_N, Self.TILE_P](
                    A_data,
                    B_data,
                    C_data,
                    A_base_off,
                    A_row_stride,
                    A_col_stride,
                    B_base_off,
                    B_row_stride,
                    B_col_stride,
                    C_base_off,
                    m,
                    k,
                    p,
                    i_start,
                    i_end,
                )

        def process_panel(flat_idx: Int) {imm}:
            var batch = flat_idx // num_tiles_j
            var j_tile = flat_idx % num_tiles_j

            # Decode batch → A/B base offsets (same arithmetic as above)
            var A_base_off = A_offset
            var B_base_off = B_offset

            if batch_rank > 0:
                var remaining = batch
                var divisor = 1
                for d in range(batch_rank - 1, -1, -1):
                    var coord = (remaining // divisor) % batch_shape[d]

                    var A_d = d - (batch_rank - A_batch_rank)
                    if A_d >= 0 and A_batch_shape[A_d] > 1:
                        A_base_off += coord * A_batch_strides[A_d]

                    var B_d = d - (batch_rank - B_batch_rank)
                    if B_d >= 0 and B_batch_shape[B_d] > 1:
                        B_base_off += coord * B_batch_strides[B_d]

                    divisor *= batch_shape[d]

            var C_base_off = batch * m * p

            # One j-panel: pack once, single SIMD call over all i rows
            var j_start = j_tile * Self.TILE_P
            var jlen = min(j_start + Self.TILE_P, p) - j_start
            matmul_panel_tile[Self.dtype, Self.TILE_N, Self.TILE_P](
                A_data,
                B_data,
                C_data,
                A_base_off,
                A_row_stride,
                A_col_stride,
                B_base_off,
                B_row_stride,
                B_col_stride,
                C_base_off,
                p,
                m,
                k,
                j_start,
                jlen,
                A_col_stride != 1,
            )

        if use_panel:
            parallelize(
                process_panel, total_batch * num_tiles_j, num_physical_cores()
            )
        else:
            parallelize(process_tile, total_tiles, num_physical_cores())

        return (Layout(out_shape), C_storage)
