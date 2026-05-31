from tenmo import Tensor, NDBuffer, Shape
from std.testing import assert_true
from tenmo.common_utils import s, i
from std.sys.defines import get_defined_string
from tenmo.matrixshapevalidator import MatrixShapeValidator
from std.sys import has_accelerator
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from std.sys.intrinsics import PrefetchLocality
from tenmo.matmul_kernel import MatmulNdGpu

comptime dtype = DType.float32


def main() raises:
    _ = """comptime BLAS_PATH = get_defined_string[
        "BLAS_PATH", "/lib/x86_64-linux-gnu/libopenblas.so.0"
    ]()
    print("BLAS_PATH: ", BLAS_PATH)"""
    pass




# ─────────────────────────────────────────────────────────────
#  Tuning constants
#  Tune these for your hardware if benchmarking shows different
#  sweet spots. L1 cache line = 64 bytes → 16 float32 values.
# ─────────────────────────────────────────────────────────────
comptime TILE_M = 64   # row tile — fits well in L2
comptime TILE_N = 64   # shared-dim tile
comptime TILE_P = 128  # col tile — wider for SIMD reuse
comptime UNROLL = 4    # number of SIMD accumulators per j-strip
comptime PREFETCH_DIST = 2  # tiles ahead to prefetch

def matmul_2d(
    A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]
) -> NDBuffer[Self.dtype]:
    ref A_shape = A.shape
    ref B_shape = B.shape
    MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

    var C: NDBuffer[Self.dtype]

    comptime if has_accelerator():
        if A.is_on_gpu() and B.is_on_gpu():
            try:
                C = MatmulNdGpu[Self.dtype].launch[tile_size=TILE_SIZE](
                    A, B
                )
            except e:
                print(e)
                panic("NDBuffer matmul_2d → GPU operation failed")
                C = NDBuffer[Self.dtype](Shape())
        elif (A.is_on_gpu() and B.is_on_cpu()) or (
            A.is_on_cpu() and B.is_on_gpu()
        ):
            panic(
                " NDBuffer matmul_2d → both buffers must be on gpu. A"
                " is on gpu?",
                String(A.is_on_gpu()),
                ", B is on gpu?",
                String(B.is_on_gpu()),
            )
            C = NDBuffer[Self.dtype](Shape())
        else:
            C = A.matmul_2d_cpu(B)
    else:
        C = A.matmul_2d_cpu(B)

    return C^

def matmul_2d_cpu(
    A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]
) -> NDBuffer[Self.dtype]:
    ref A_shape = A.shape
    ref B_shape = B.shape
    MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

    comptime simdwidth = simd_width_of[Self.dtype]()
    # Process UNROLL * simdwidth columns of C per inner iteration.
    # e.g. float32: simdwidth=8, UNROLL=4 → 32 columns at once.
    comptime simd_unroll = simdwidth * UNROLL

    var m = A_shape[0]
    var n = A_shape[1]
    var p = B_shape[1]

    var C = NDBuffer[Self.dtype].zeros(Shape([m, p]))

    # ── Hoist all pointer/stride metadata ──────────────────────
    ref A_strides = A.strides
    var A_stride0 = A_strides[0]
    var A_stride1 = A_strides[1]
    var A_offset  = A.offset
    var A_data    = A.data_ptr()

    ref B_strides = B.strides
    var B_stride0 = B_strides[0]
    var B_stride1 = B_strides[1]
    var B_offset  = B.offset
    var B_data    = B.data_ptr()

    var C_data = C.data_ptr()

    if B.is_contiguous():
        # ════════════════════════════════════════════════════════
        #  OPTIMIZED TILED + UNROLLED + PREFETCHED MATMUL
        #
        #  Three-level tiling (TILE_M × TILE_N × TILE_P):
        #    - TILE_M rows of A/C fit in L2
        #    - TILE_N keeps the k-strip of A in L1
        #    - TILE_P wide enough to saturate SIMD across j
        #
        #  Inside the innermost i/k loop:
        #    - UNROLL accumulators per j-strip → fills FMA pipeline
        #    - Prefetch next k-tile of B while computing current
        #    - a_ik broadcast once, reused across all j accumulators
        # ════════════════════════════════════════════════════════
        var num_tiles_i = (m + TILE_M - 1) // TILE_M

        @parameter
        def process_row_tile(tile_idx: Int):
            var i_start = tile_idx * TILE_M
            var i_end   = min(i_start + TILE_M, m)

            for j_tile in range(0, p, TILE_P):
                var j_end = min(j_tile + TILE_P, p)

                for k_tile in range(0, n, TILE_N):
                    var k_end = min(k_tile + TILE_N, n)

                    # ── Prefetch the NEXT k-tile of B into cache ──
                    var next_k_tile = k_tile + TILE_N
                    if next_k_tile < n:
                        for k_pre in range(
                            next_k_tile,
                            min(next_k_tile + TILE_N, n),
                        ):
                            var b_pre_base = (
                                k_pre * B_stride0 + B_offset + j_tile
                            )
                            prefetch[
                                PrefetchLocality.HIGH,
                                PrefetchRW.READ,
                                PREFETCH_DIST,
                            ](B_data + b_pre_base)

                    for i in range(i_start, i_end):
                        var a_row_base = i * A_stride0 + A_offset
                        var c_row_base = i * p

                        var j = j_tile

                        # ── Unrolled SIMD: UNROLL vectors at once ──
                        while j + simd_unroll <= j_end:
                            # Load UNROLL accumulators from C
                            var acc0 = C_data.load[width=simdwidth](
                                c_row_base + j
                            )
                            var acc1 = C_data.load[width=simdwidth](
                                c_row_base + j + simdwidth
                            )
                            var acc2 = C_data.load[width=simdwidth](
                                c_row_base + j + simdwidth * 2
                            )
                            var acc3 = C_data.load[width=simdwidth](
                                c_row_base + j + simdwidth * 3
                            )

                            for k in range(k_tile, k_end):
                                # Load a_ik ONCE — broadcast across
                                # all UNROLL accumulator updates
                                var a_addr = a_row_base + k * A_stride1
                                var a_ik = SIMD[Self.dtype, simdwidth](
                                    A_data[a_addr]
                                )

                                var b_base = k * B_stride0 + B_offset + j

                                # FMA: acc += a_ik * b_vec
                                acc0 = math.fma(
                                    a_ik,
                                    B_data.load[width=simdwidth](b_base),
                                    acc0,
                                )
                                acc1 = math.fma(
                                    a_ik,
                                    B_data.load[width=simdwidth](
                                        b_base + simdwidth
                                    ),
                                    acc1,
                                )
                                acc2 = math.fma(
                                    a_ik,
                                    B_data.load[width=simdwidth](
                                        b_base + simdwidth * 2
                                    ),
                                    acc2,
                                )
                                acc3 = math.fma(
                                    a_ik,
                                    B_data.load[width=simdwidth](
                                        b_base + simdwidth * 3
                                    ),
                                    acc3,
                                )

                            # Store UNROLL accumulators back to C
                            C_data.store[width=simdwidth](
                                c_row_base + j, acc0
                            )
                            C_data.store[width=simdwidth](
                                c_row_base + j + simdwidth, acc1
                            )
                            C_data.store[width=simdwidth](
                                c_row_base + j + simdwidth * 2, acc2
                            )
                            C_data.store[width=simdwidth](
                                c_row_base + j + simdwidth * 3, acc3
                            )
                            j += simd_unroll

                        # ── Single-vector SIMD tail ─────────────
                        while j + simdwidth <= j_end:
                            var c_addr = c_row_base + j
                            var acc = C_data.load[width=simdwidth](c_addr)

                            for k in range(k_tile, k_end):
                                var a_addr = a_row_base + k * A_stride1
                                var a_ik = SIMD[Self.dtype, simdwidth](
                                    A_data[a_addr]
                                )
                                var b_base = k * B_stride0 + B_offset + j
                                acc = math.fma(
                                    a_ik,
                                    B_data.load[width=simdwidth](b_base),
                                    acc,
                                )

                            C_data.store[width=simdwidth](c_addr, acc)
                            j += simdwidth

                        # ── Scalar tail (remaining columns) ─────
                        while j < j_end:
                            var c_addr = c_row_base + j
                            var acc: Scalar[Self.dtype] = C_data[c_addr]

                            for k in range(k_tile, k_end):
                                var a_addr = a_row_base + k * A_stride1
                                var b_addr = (
                                    #k * B_stride0 + B_offset + j * B_stride1
                                    k * B_stride0 + B_offset + j
                                )
                                acc += A_data[a_addr] * B_data[b_addr]

                            C_data[c_addr] = acc
                            j += 1

        parallelize[process_row_tile](num_tiles_i, num_physical_cores())

    else:
        # ════════════════════════════════════════════════════════
        #  NON-CONTIGUOUS PATH — parallelized over row tiles
        #
        #  Original was fully serial. Now parallelized the same
        #  way as the contiguous path. No SIMD because strides
        #  mean elements aren't adjacent in memory.
        # ════════════════════════════════════════════════════════
        var num_tiles_i = (m + TILE_M - 1) // TILE_M

        @parameter
        def process_row_tile_noncontig(tile_idx: Int):
            var i_start = tile_idx * TILE_M
            var i_end   = min(i_start + TILE_M, m)

            for i in range(i_start, i_end):
                var a_row_base = i * A_stride0 + A_offset
                var c_row_base = i * p

                for j in range(p):
                    var acc: Scalar[Self.dtype] = 0

                    for k in range(n):
                        var a_addr = a_row_base + k * A_stride1
                        var b_addr = (
                            k * B_stride0 + B_offset + j * B_stride1
                        )
                        acc += A_data[a_addr] * B_data[b_addr]

                    C_data[c_row_base + j] = acc

        parallelize[process_row_tile_noncontig](
            num_tiles_i, num_physical_cores()
        )

    return C^
