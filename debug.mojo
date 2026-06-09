from tenmo import Tensor, NDBuffer, Shape, SGD, Buffer, Strides, IntArray
from std.testing import assert_true, TestSuite
from tenmo.common_utils import s, i, panic, Epsilon
from std.sys.defines import get_defined_string
from tenmo.matrixshapevalidator import MatrixShapeValidator
from std.sys import has_accelerator, simd_width_of
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from std.sys.intrinsics import PrefetchLocality
from tenmo.kernels.matmul_kernel import MatmulNdGpu
from std.sys import prefetch, PrefetchOptions
from std import math
from tenmo.device import GPU
from tenmo.mnemonics import (
    Multiply,
    Add,
    Subtract,
    ReverseSubtract,
    Divide,
)

comptime dtype = DType.float32

@fieldwise_init
struct IntAdder(ImplicitlyCopyable & Absable):
    var value: Int

    def __getitem__(self, index: Int, sync: Bool= True) -> Int:
        print("Sync in get: ", sync, "index: ", index)
        return self.value

    def __setitem__(mut self, index: Int, v: Int, sync: Bool= False):
        print("Sync in set: ", sync, "index: ", index)
        self.value = v



    def __add__[sync: Bool = True](self, other: Self) -> IntAdder:
        print("sync: ", sync)
        return IntAdder(self.value + other.value)

    def __abs__(self) -> Self:
        return Self(abs(self.value))

    def __iadd__[sync: Bool = True](mut self, other: Self):
        print("sync: ", sync)
        self.value += other.value


def test_ndb_addition_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype](1, 2, 3, 4)
        var a_gpu = a.to_gpu(gpu)
        var b = NDBuffer[dtype](1, 2, 3, 4)
        var b_gpu = b.to_gpu(gpu)
        var c_gpu = a_gpu.arithmetic_ops[Add](b_gpu)

        assert_true(c_gpu.to_cpu().all_close[atol=1e-5](a + b))

def test_ndb_inplace_addition_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype](1, 2, 3, 4)
        var a_gpu = a.to_gpu(gpu)
        var b = NDBuffer[dtype](1, 2, 3, 4)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu)
        a.inplace_ops[Add](b)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](a))
        print("Num devices: ", gpu[].number_of_devices())

def test_ndb_inter_gpu_copy_and_opeartion() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var gpu = GPU()
        var a = NDBuffer[dtype](1, 2, 3, 4)
        var a_gpu = a.to_gpu(gpu)
        var b = NDBuffer[dtype](1, 2, 3, 4)
        var b_gpu = b.to_gpu(gpu)
        a_gpu.inplace_ops[Add](b_gpu)
        a.inplace_ops[Add](b)
        assert_true(a_gpu.to_cpu().all_close[atol=1e-5](a))
        if gpu[].number_of_devices() > 1:
            var gpu_other = GPU(1)
            var a_in_other_gpu = a.to_gpu(gpu_other)
            var a_other = a_in_other_gpu.scalar_ops[Multiply](42)
            assert_true((a_gpu.to_gpu(gpu_other).scalar_ops[Multiply](42)) == a_other, "Inter gpu ndn operation failed")
            print("Inter gpu ndb op passed")
        print("Num devices: ", gpu[].number_of_devices())



def main() raises:
    #test_sgd_gpu_backward_integration()
    _="""var x = IntAdder(100)
    var y = IntAdder(200)
    var r = x.__add__[False](y)
    print(r.value)
    print(r[99, False])
    x.__setitem__(100,-800)
    print(x[88888])
    x += r
    print(x.value, abs(x).value)"""
    comptime dtype = DType.float32
    _="""var ndb = NDBuffer[dtype](1, 2, 3, 4)
    ndb.print()
    ndb = NDBuffer[DType.float32].arange(5, 30)
    ndb = ndb.reshape(Shape(5, 5)) # Reshape takes 'mut' self
    ndb.print()
    ndb.transpose().print() # Transpose takes 'mut' self

    var buffer = Buffer[dtype].arange(1, 13)
    ndb = NDBuffer[dtype](buffer, Shape(3, 4))
    ndb.print()
    ndb = NDBuffer[dtype](buffer^, Shape(4, 2), Strides(1, 2), offset=4)
    ndb.print()

    ndb = NDBuffer[dtype].full(Shape(2, 3), 42.0)
    ndb.print()

    ndb = NDBuffer[dtype].arange(1, 25)
    var shared = ndb.share(Shape(2, 2, 5), Strides(2, 1, 3), offset=4)
    shared.print()"""
    #test_ndb_addition_1d()
    #test_ndb_inplace_addition_1d()
    #test_ndb_inter_gpu_copy_and_opeartion()
    #var a = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 4, 6))
    _="""var b_base = NDBuffer[dtype].arange(1, 49).reshape(Shape(2, 6, 4))
    var b = b_base.transpose()  # non-contiguous (2,4,6)
    b.print()
    var c = b_base.transpose(IntArray(-1, -2))
    c.print()"""
    var a = SIMD[dtype, 4](1, 2, 3, 4)
    var b = SIMD[dtype, 4](0, 3, 9, 10)
    var r = a / (b + Epsilon[dtype].value())
    print(r)

def test_sgd_gpu_backward_integration() raises:
    var w = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var x = Tensor[dtype].d1([1.0, 1.0, 1.0])
    var params = List[UnsafePointer[Tensor[dtype], MutAnyOrigin]]()
    params.append(UnsafePointer(to=w))
    var sgd = SGD(params, lr=0.1)
    var loss = (w * x).sum()
    loss.backward()
    sgd.step()
    # grad_w = x = [1,1,1], w = w - 0.1*1 = [0.9, 1.9, 2.9]
    assert_true(w.all_close(Tensor[dtype].d1([0.9, 1.9, 2.9])))
    assert_true(w.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


comptime prefetch_opts = PrefetchOptions().for_read().high_locality().to_data_cache()

# ─────────────────────────────────────────────────────────────────────────────
#  Tuning constants
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
#
#  PREFETCH_DIST : kept for documentation; actual prefetch distance is
#                  controlled by the PrefetchLocality hint, not a scalar.
# ─────────────────────────────────────────────────────────────────────────────
comptime GPU_TILE_SIZE = 32  # GPU kernel tile — separate concern from CPU
comptime TILE_M = 64  # CPU row tile    — fits A rows in L2
comptime TILE_N = 64  # CPU k-dim tile  — fits A k-strip in L1
comptime TILE_P = 128  # CPU col tile    — wide for SIMD reuse
comptime UNROLL = 4  # SIMD accumulators per j-strip
comptime PREFETCH_DIST = 2  # informational only


struct MM[dtype: DType]:
    # ─────────────────────────────────────────────────────────────────────────
    #  matmul_2d
    #
    #  Entry point. Dispatches to GPU or CPU path at runtime, with the
    #  GPU branch compiled away entirely (comptime if) when no accelerator
    #  is present — zero overhead on CPU-only builds.
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def matmul_2d(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        ref A_shape = A.shape
        ref B_shape = B.shape
        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

        var C: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if A.is_on_gpu() and B.is_on_gpu():
                # Both operands on GPU — launch tiled GPU kernel.
                try:
                    C = MatmulNdGpu[Self.dtype].launch[tile_size=GPU_TILE_SIZE](
                        A, B
                    )
                except e:
                    print(e)
                    panic("NDBuffer matmul_2d → GPU operation failed")
                    C = NDBuffer[Self.dtype](
                        Shape()
                    )  # unreachable; satisfies compiler
            elif (A.is_on_gpu() and B.is_on_cpu()) or (
                A.is_on_cpu() and B.is_on_gpu()
            ):
                # Mixed device operands are not supported — data must be on the
                # same device before calling matmul.
                panic(
                    (
                        "NDBuffer matmul_2d → both buffers must be on same"
                        " device. A on gpu? "
                    ),
                    String(A.is_on_gpu()),
                    ", B on gpu? ",
                    String(B.is_on_gpu()),
                )
                C = NDBuffer[Self.dtype](Shape())  # unreachable
            else:
                # GPU present but both operands are on CPU — use CPU path.
                C = MM[Self.dtype].matmul_2d_cpu(A, B)
        else:
            # No accelerator compiled in — always CPU.
            C = MM[Self.dtype].matmul_2d_cpu(A, B)

        return C^

    # ─────────────────────────────────────────────────────────────────────────
    #  matmul_2d_cpu
    #
    #  High-performance CPU matmul: C = A @ B
    #    A : (m, n)
    #    B : (n, p)
    #    C : (m, p)  — freshly allocated, zero-initialised
    #
    #  Two paths:
    #    1. B contiguous  → tiled + parallelised + SIMD + FMA + prefetch
    #    2. B non-contig  → parallelised scalar (strides may not be unit)
    #
    #  The contiguous path adds a further fast lane when A is also
    #  contiguous, eliminating the remaining stride multiplications.
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def matmul_2d_cpu(
        A: NDBuffer[Self.dtype], B: NDBuffer[Self.dtype]
    ) -> NDBuffer[Self.dtype]:
        ref A_shape = A.shape
        ref B_shape = B.shape
        MatrixShapeValidator.validate_matrix_shapes_2d(A_shape, B_shape)

        # simdwidth  : how many dtype elements fit in one SIMD register
        #              e.g. 8 for float32 on AVX2, 16 for float16
        # simd_unroll: total columns processed per unrolled iteration
        #              = UNROLL × simdwidth  (e.g. 4 × 8 = 32 for float32)
        comptime simdwidth = simd_width_of[Self.dtype]()
        comptime simd_unroll = simdwidth * UNROLL

        var m = A_shape[0]  # rows of A
        var n = A_shape[1]  # shared dimension (cols of A = rows of B)
        var p = B_shape[1]  # cols of B

        # Output matrix — zero-initialised so partial k-tile accumulations
        # can safely load-accumulate-store across multiple k_tile iterations.
        var C = NDBuffer[Self.dtype].zeros(Shape([m, p]))

        # ── Hoist all pointer and stride metadata out of every loop ──────────
        # Accessing struct fields inside a hot loop forces repeated loads.
        # Storing them in locals lets the compiler keep them in registers.
        ref A_strides = A.strides
        var A_stride0 = A_strides[0]  # bytes/elems to advance one row in A
        var A_stride1 = A_strides[1]  # bytes/elems to advance one col in A
        var A_offset = A.offset  # base offset for non-zero-origin views
        var A_data = A.data_ptr()

        ref B_strides = B.strides
        var B_stride0 = B_strides[0]  # row stride of B
        var B_stride1 = B_strides[1]  # col stride of B (1 when contiguous)
        var B_offset = B.offset
        var B_data = B.data_ptr()

        var C_data = C.data_ptr()
        # C is freshly allocated and always contiguous:
        #   C_stride0 = p   (one full row)
        #   C_stride1 = 1   (adjacent columns)
        # These are inlined as literals below rather than stored in variables.

        # ════════════════════════════════════════════════════════════════════
        #  CONTIGUOUS B PATH
        #  When B is contiguous in memory its column stride is 1, so
        #  "j * B_stride1" collapses to just "j" — one fewer multiply per
        #  address calculation in the hot loop.
        # ════════════════════════════════════════════════════════════════════
        if B.is_contiguous():
            # Determine whether A is also contiguous.
            # When true, A_stride1 = 1, so "k * A_stride1" → "k".
            # This eliminates the last stride multiply in the innermost loop.
            var a_also_contiguous = A.is_contiguous()

            # Split the m-dimension into row tiles for parallelism.
            # Each tile is processed by one logical task (mapped to a core).
            var num_tiles_i = (m + TILE_M - 1) // TILE_M

            @parameter
            def process_row_tile(tile_idx: Int):
                var i_start = tile_idx * TILE_M
                var i_end = min(i_start + TILE_M, m)

                # ── Three-level tiling: k → j → i ────────────────────────
                #
                # Loop order matters for cache behaviour:
                #
                #   k outermost tile:
                #     Load a TILE_N-wide strip of A (rows i_start..i_end,
                #     cols k_tile..k_end) into L1 once per k_tile.
                #     Reuse that strip across ALL j_tiles before evicting.
                #     → A k-strip touches L1 exactly once per k_tile. ✓
                #
                #   j middle tile:
                #     Load a TILE_P-wide strip of B rows into cache.
                #     Wide enough to keep SIMD units busy.
                #
                #   i inner loop:
                #     Iterate over actual rows within the current i-tile.
                #     Short loop; a_row_base computed once per row.
                #
                # Previous order (j → k → i) was wrong: for each j_tile
                # the entire k loop ran, evicting A's k-strip from L1
                # between j iterations — defeating the tiling intent.
                # ─────────────────────────────────────────────────────────
                for k_tile in range(0, n, TILE_N):
                    var k_end = min(k_tile + TILE_N, n)

                    # ── Prefetch the NEXT k-tile of B into cache ──────────
                    # Issue prefetch instructions for the k-tile that will be
                    # needed in the next outer iteration while we compute the
                    # current one. This hides DRAM latency (typically 100+ns).
                    #
                    # We prefetch full cache lines (16 × float32 = 64 bytes)
                    # across the entire TILE_P column width so the hardware
                    # prefetcher has the data ready before we touch it.
                    #
                    # Placement: outside the j_tile loop so we issue the
                    # prefetch ONCE per k_tile, not once per (k_tile, j_tile).
                    var next_k = k_tile + TILE_N
                    if next_k < n:
                        var next_k_end = min(next_k + TILE_N, n)
                        for k_pre in range(next_k, next_k_end):
                            var row_base = k_pre * B_stride0 + B_offset
                            # Prefetch cache lines across the full column width.
                            # 16 float32 = one 64-byte cache line.
                            for cl in range(0, p, 16):
                                prefetch[prefetch_opts](B_data + row_base + cl)

                    for j_tile in range(0, p, TILE_P):
                        var j_end = min(j_tile + TILE_P, p)

                        for i in range(i_start, i_end):
                            # a_row_base: flat index of row i in A, accounting
                            # for non-zero offsets (views, slices).
                            var a_row_base = i * A_stride0 + A_offset

                            # c_row_base: flat index of row i in C.
                            # C is always contiguous so stride0 = p, offset = 0.
                            var c_row_base = i * p

                            var j = j_tile

                            # ── Unrolled SIMD: process UNROLL vectors/iter ──
                            #
                            # Each iteration handles simd_unroll = UNROLL *
                            # simdwidth columns of C simultaneously.
                            # For float32 / AVX2: 4 × 8 = 32 columns per iter.
                            #
                            # Why unroll?
                            #   A single SIMD FMA has latency ~4 cycles but
                            #   throughput ~0.5 cycles. With only one
                            #   accumulator the CPU stalls waiting for the
                            #   result. Four independent accumulators let the
                            #   CPU overlap four FMAs and approach peak
                            #   throughput.
                            #
                            # Pattern per k:
                            #   1. Load a_ik scalar → broadcast to SIMD vector
                            #      (one load, reused 4× across accumulators)
                            #   2. Load 4 consecutive B vectors (b_base + 0/8/16/24)
                            #   3. FMA: acc_n += a_ik * b_vec_n  (4 FMAs, independent)
                            # ───────────────────────────────────────────────
                            while j + simd_unroll <= j_end:
                                var cj = c_row_base + j

                                # Load existing C values. Necessary because
                                # previous k_tile iterations may have already
                                # accumulated partial sums here.
                                # Exception: first k_tile — C is still zero,
                                # so loading zeros is correct (though wasteful;
                                # see the k_tile==0 optimisation below).
                                var acc0: SIMD[Self.dtype, simdwidth]
                                var acc1: SIMD[Self.dtype, simdwidth]
                                var acc2: SIMD[Self.dtype, simdwidth]
                                var acc3: SIMD[Self.dtype, simdwidth]

                                if k_tile == 0:
                                    # First k-tile: C is zeroed, skip the load.
                                    # This avoids 4 unnecessary memory reads
                                    # on what is typically the first (and often
                                    # only) k_tile for small n.
                                    acc0 = SIMD[Self.dtype, simdwidth](0)
                                    acc1 = SIMD[Self.dtype, simdwidth](0)
                                    acc2 = SIMD[Self.dtype, simdwidth](0)
                                    acc3 = SIMD[Self.dtype, simdwidth](0)
                                else:
                                    # Subsequent k-tiles: load partial sums.
                                    acc0 = C_data.load[width=simdwidth](cj)
                                    acc1 = C_data.load[width=simdwidth](
                                        cj + simdwidth
                                    )
                                    acc2 = C_data.load[width=simdwidth](
                                        cj + simdwidth * 2
                                    )
                                    acc3 = C_data.load[width=simdwidth](
                                        cj + simdwidth * 3
                                    )

                                for k in range(k_tile, k_end):
                                    # a_ik: single element A[i, k].
                                    # Broadcast it to a full SIMD vector so
                                    # it can be multiplied against b_vec in
                                    # one FMA instruction.
                                    # Broadcast is free on modern CPUs (vbroadcastss).
                                    var a_addr: Int
                                    if a_also_contiguous:
                                        # A_stride1 = 1 — no multiply needed
                                        a_addr = a_row_base + k
                                    else:
                                        a_addr = a_row_base + k * A_stride1
                                    var a_ik = SIMD[Self.dtype, simdwidth](
                                        A_data[a_addr]
                                    )

                                    # b_base: start of B row k at column j.
                                    # B_stride0 = p when contiguous, but we
                                    # use the hoisted variable for correctness
                                    # with non-unit-stride views.
                                    # B_stride1 = 1 (B is contiguous) so
                                    # "+ j" replaces "+ j * B_stride1".
                                    var b_base = k * B_stride0 + B_offset + j

                                    # Four FMA operations, all independent.
                                    # The CPU can pipeline these because
                                    # acc0..acc3 do not depend on each other.
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

                                # Write all four accumulators back to C.
                                C_data.store[width=simdwidth](cj, acc0)
                                C_data.store[width=simdwidth](
                                    cj + simdwidth, acc1
                                )
                                C_data.store[width=simdwidth](
                                    cj + simdwidth * 2, acc2
                                )
                                C_data.store[width=simdwidth](
                                    cj + simdwidth * 3, acc3
                                )
                                j += simd_unroll

                            # ── Single-vector SIMD tail ──────────────────
                            # Handles the columns that didn't fill a full
                            # simd_unroll block (0 to simd_unroll-1 columns).
                            while j + simdwidth <= j_end:
                                var c_addr = c_row_base + j

                                var acc: SIMD[Self.dtype, simdwidth]
                                if k_tile == 0:
                                    acc = SIMD[Self.dtype, simdwidth](0)
                                else:
                                    acc = C_data.load[width=simdwidth](c_addr)

                                for k in range(k_tile, k_end):
                                    var a_addr: Int
                                    if a_also_contiguous:
                                        a_addr = a_row_base + k
                                    else:
                                        a_addr = a_row_base + k * A_stride1
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

                            # ── Scalar tail ───────────────────────────────
                            # Handles the remaining 0 to simdwidth-1 columns
                            # that couldn't fill even a single SIMD vector.
                            # B_stride1 = 1 (contiguous) → "+ j" not "+ j * B_stride1"
                            while j < j_end:
                                var c_addr = c_row_base + j

                                var acc: Scalar[Self.dtype]
                                if k_tile == 0:
                                    acc = 0
                                else:
                                    acc = C_data[c_addr]

                                for k in range(k_tile, k_end):
                                    var a_addr: Int
                                    if a_also_contiguous:
                                        a_addr = a_row_base + k
                                    else:
                                        a_addr = a_row_base + k * A_stride1
                                    var b_addr = k * B_stride0 + B_offset + j
                                    acc += A_data[a_addr] * B_data[b_addr]

                                C_data[c_addr] = acc
                                j += 1

            # Launch all row tiles in parallel across physical cores.
            # Each tile_idx maps to a disjoint set of rows → no data races.
            parallelize[process_row_tile](num_tiles_i, num_physical_cores())

        else:
            # ════════════════════════════════════════════════════════════════
            #  NON-CONTIGUOUS B PATH
            #
            #  B has non-unit column stride (e.g. after a transpose or slice).
            #  SIMD loads require contiguous memory, so we fall back to scalar.
            #  We still parallelise over row tiles to use all cores.
            #
            #  B_stride1 must be respected here — do NOT collapse to "+ j".
            # ════════════════════════════════════════════════════════════════
            var num_tiles_i = (m + TILE_M - 1) // TILE_M

            @parameter
            def process_row_tile_noncontig(tile_idx: Int):
                var i_start = tile_idx * TILE_M
                var i_end = min(i_start + TILE_M, m)

                for i in range(i_start, i_end):
                    var a_row_base = i * A_stride0 + A_offset
                    var c_row_base = i * p

                    for j in range(p):
                        var acc: Scalar[Self.dtype] = 0

                        for k in range(n):
                            var a_addr = a_row_base + k * A_stride1
                            # B_stride1 may not be 1 — must multiply.
                            var b_addr = (
                                k * B_stride0 + B_offset + j * B_stride1
                            )
                            acc += A_data[a_addr] * B_data[b_addr]

                        C_data[c_row_base + j] = acc

            parallelize[process_row_tile_noncontig](
                num_tiles_i, num_physical_cores()
            )

        return C^

