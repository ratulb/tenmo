# CPU Broadcast Engine
#
# All tensor‑tensor arithmetic operations that involve broadcasting land here.
# Broadcasting connects two NDBuffers with different but compatible shapes,
# producing a third NDBuffer of the broadcast shape.
#
# Three dispatch tiers, each optimised for a different memory‑layout pattern:
#
#   broadcast()                  ← public entry point
#    ├── broadcast_scalar()      ← one operand is effectively a scalar
#    └── broadcast_nd()          ← general ND broadcast
#         ├── Tier 1: both operands have unit stride in last dim
#         ├── Tier 2: one operand has unit stride, the other broadcasts (stride 0)
#         └── Tier 3: neither has unit stride — scalar odometer
#
# Correctness guarantee (proved by effective strides):
#   Every output coordinate maps to base_a + Σ coord[d] × eff_stride[d]
#   in operand A, where eff_stride[d] = 0 if the dimension is broadcast
#   (either because the operand lacks that dim or its size is 1) else the
#   original stride.  All three tiers evaluate *the same mapping*; only
#   the iteration strategy differs.

from .shared.constants import Epsilon
from .shared.panic import panic
from .shared.layout import Layout
from .shared.buffers import Buffer
from .shared.intarray import IntArray
from .shared.broadcasthelper import ShapeBroadcaster
from .shared.indexhelper import IndexIterator, IndexCalculator
from .shared.scalar_ops import (
    simd_op,
    scalar_op,
    unary_op,
    unary_op_simd,
    float_unary_op,
    float_unary_op_simd,
)
from .shared.constants import Epsilon
from .shared.mnemonics import (
    Subtract,
    ReverseSubtract,
    Divide,
    ReverseDivide,
    POW,
)
from std.sys import simd_width_of
from std.sys.intrinsics import strided_load
from max.algorithm import parallelize
from std.sys.info import num_physical_cores


struct CpuArithmeticOps[dtype: DType](
    ImplicitlyCopyable & Equatable & Writable
):
    # Public entry point: choose scalar or ND path
    # A buffer is treated as "scalar‑like" when its shape has rank ≤ 1
    # and it holds exactly one element.  This includes true 0‑d scalars
    # (Shape()) and 1‑d tensors with shape (1,).

    @staticmethod
    @always_inline
    def broadcast[
        op_code: Int,
    ](
        a: Layout,
        a_buffer: Buffer[Self.dtype],
        b: Layout,
        b_buffer: Buffer[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> Tuple[Layout, Buffer[Self.dtype]]:
        var a_is_scalar = a.shape.rank() <= 1 and a.numel() == 1
        var b_is_scalar = b.shape.rank() <= 1 and b.numel() == 1
        if a_is_scalar or b_is_scalar:
            if a_is_scalar:
                # Non‑commutative ops (Subtract, Divide) need reversed
                # semantics when the scalar is on the left:
                #   scalar - tensor  →  ReverseSubtract
                #   scalar / tensor  →  ReverseDivide
                comptime if op_code == Subtract or op_code == Divide:
                    if op_code == Subtract:
                        return CpuArithmeticOps.broadcast_scalar[
                            ReverseSubtract
                        ](a, a_buffer, b, b_buffer, a_is_scalar, epsilon)
                    else:
                        return CpuArithmeticOps.broadcast_scalar[ReverseDivide](
                            a, a_buffer, b, b_buffer, a_is_scalar, epsilon
                        )

            return CpuArithmeticOps.broadcast_scalar[op_code](
                a, a_buffer, b, b_buffer, a_is_scalar, epsilon
            )
        else:
            return CpuArithmeticOps.broadcast_nd[op_code](
                a, a_buffer, b, b_buffer, epsilon
            )

    # Scalar path  —  one operand is scalar‑like
    # When the scalar side is contiguous we can use the SIMD‑vectorised
    # arithmetic_ops_scalar on a Buffer slice, avoiding a per‑element
    # loop.  When non‑contiguous we fall back to index‑iterator walk.

    @staticmethod
    @always_inline
    def broadcast_scalar[
        op_code: Int
    ](
        a: Layout,
        a_buffer: Buffer[Self.dtype],
        b: Layout,
        b_buffer: Buffer[Self.dtype],
        a_is_scalar: Bool,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> Tuple[Layout, Buffer[Self.dtype]]:
        var result_shape = ShapeBroadcaster.broadcast_shape(a.shape, b.shape)
        var is_contiguous = (
            b.is_contiguous() if a_is_scalar else a.is_contiguous()
        )
        var item = a_buffer[a.offset] if a_is_scalar else b_buffer[b.offset]
        var buffer: Buffer[Self.dtype]
        if is_contiguous:
            # Fast contiguous path: broadcast the SIMD‑vectorised scalar op
            # directly on the non‑scalar operand's data range via
            # arithmetic_ops_scalar(start, end).  This reads the range in
            # one SIMD pass and writes a fresh contiguous result Buffer —
            # avoids an unnecessary alloc+unsafe_memcpy that the old
            # .copied().arithmetic_ops_scalar() incurred.
            var offset = b.offset if a_is_scalar else a.offset
            var numels = b.numel() if a_is_scalar else a.numel()
            buffer = b_buffer.arithmetic_ops_scalar[op_code](
                item, offset, offset + numels
            ) if a_is_scalar else a_buffer.arithmetic_ops_scalar[op_code](
                item, offset, offset + numels
            )

        else:
            # Slow non‑contiguous path: row-parallel walk of the non-scalar
            # operand's logical elements via IndexIterator (peek/skip, no
            # raises) with a SIMD inner loop over the last dimension.
            # ReverseSubtract and ReverseDivide swap the argument order.
            var o_layout = b if a_is_scalar else a
            var o_storage = b_buffer if a_is_scalar else a_buffer
            buffer = Buffer[Self.dtype](result_shape.num_elements())
            var total = result_shape.num_elements()
            comptime simd_width = simd_width_of[
                Self.dtype
            ]() if Self.dtype != DType.bool else 1

            var rank = result_shape.rank()
            var last_dim = result_shape[rank - 1] if rank >= 1 else 0

            if rank < 1 or last_dim < 1:
                # Degenerate shape — serial fallback. Non-raising
                # __has_next__/peek/skip walk (no StopIteration machinery).
                var index = 0
                var o_shape = o_layout.shape
                var o_strides = o_layout.strides
                var it = IndexIterator(
                    shape=Pointer(to=o_shape).as_imm(),
                    strides=Pointer(to=o_strides).as_imm(),
                    start_offset=o_layout.offset,
                )
                while it.__has_next__():
                    var idx = it.peek()
                    if a_is_scalar:
                        comptime if op_code == ReverseSubtract or op_code == ReverseDivide:
                            buffer[index] = scalar_op[op_code, Self.dtype](
                                o_storage[idx], item
                            )
                        else:
                            buffer[index] = scalar_op[op_code, Self.dtype](
                                item, o_storage[idx]
                            )
                    else:
                        buffer[index] = scalar_op[op_code, Self.dtype](
                            o_storage[idx], item
                        )
                    index += 1
                    it.skip(1)

            else:
                var rows = total // last_dim
                var n_threads = num_physical_cores()
                var n_segments = 1
                if total >= n_threads * 4096 and rows > 1:
                    n_segments = rows if rows < n_threads else n_threads

                var r_data = buffer.data.unsafe_value()
                var o_data = o_storage.data.unsafe_value()
                var o_offset = o_layout.offset
                var o_inner = o_layout.strides[rank - 1]
                var o_shape = o_layout.shape
                var o_strides = o_layout.strides

                def worker_broadcast_scalar(seg: Int) {imm}:
                    var r0 = seg * rows // n_segments
                    var r1 = (seg + 1) * rows // n_segments
                    var it = IndexIterator(
                        shape=Pointer(to=o_shape).as_imm(),
                        strides=Pointer(to=o_strides).as_imm(),
                        start_offset=o_offset,
                    )
                    it.skip(r0 * last_dim)
                    var flat = r0 * last_dim
                    for _ in range(r0, r1):
                        var row_off = it.peek()
                        var j = 0
                        comptime if simd_width > 1:
                            var item_splat = SIMD[Self.dtype, simd_width](item)
                            while j + simd_width <= last_dim:
                                var bv = strided_load[simd_width](
                                    o_data.unsafe_offset(row_off + j * o_inner),
                                    o_inner,
                                )
                                var rv: SIMD[Self.dtype, simd_width]
                                if a_is_scalar:
                                    comptime if op_code == ReverseSubtract or op_code == ReverseDivide:
                                        rv = simd_op[
                                            op_code, Self.dtype, simd_width
                                        ](bv, item_splat, epsilon)
                                    else:
                                        rv = simd_op[
                                            op_code, Self.dtype, simd_width
                                        ](item_splat, bv, epsilon)
                                else:
                                    rv = simd_op[
                                        op_code, Self.dtype, simd_width
                                    ](bv, item_splat, epsilon)
                                r_data.unsafe_store[width=simd_width](
                                    flat + j, rv
                                )
                                j += simd_width
                        for k in range(j, last_dim):
                            if a_is_scalar:
                                comptime if op_code == ReverseSubtract or op_code == ReverseDivide:
                                    r_data[unsafe_offset=flat + k] = scalar_op[
                                        op_code, Self.dtype
                                    ](
                                        o_data[
                                            unsafe_offset=row_off + k * o_inner
                                        ],
                                        item,
                                        epsilon,
                                    )
                                else:
                                    r_data[unsafe_offset=flat + k] = scalar_op[
                                        op_code, Self.dtype
                                    ](
                                        item,
                                        o_data[
                                            unsafe_offset=row_off + k * o_inner
                                        ],
                                        epsilon,
                                    )
                            else:
                                r_data[unsafe_offset=flat + k] = scalar_op[
                                    op_code, Self.dtype
                                ](
                                    o_data[unsafe_offset=row_off + k * o_inner],
                                    item,
                                    epsilon,
                                )
                        flat += last_dim
                        it.skip(last_dim)

                if n_segments > 1:
                    parallelize(worker_broadcast_scalar, n_segments, n_threads)
                else:
                    worker_broadcast_scalar(0)

        return (Layout(result_shape), buffer^)

    # ND broadcast path  —  both operands have rank ≥ 1
    # Three‑tier dispatch based on the last‑dimension effective stride:
    #
    #   Tier 1  —  both strides == 1   → SIMD‑SIMD tile
    #   Tier 2  —  one stride 1, one 0 → splat scalar + SIMD
    #   Tier 3  —  anything else        → scalar odometer
    #
    # Effective strides are the key abstraction: for each result dimension
    # d, an operand's effective stride is either 0 (if that dimension is
    # broadcast) or the original stride (if the dimension maps directly).
    # A dimension is broadcast when the operand lacks it (prepended 1s)
    # or when its size is 1 while the result size is larger.
    #
    # With effective strides, the ad‑hoc per‑coordinate translation falls
    # away — we just accumulate offsets linearly, and stride‑0 dimensions
    # naturally keep re‑reading the same memory location.

    @staticmethod
    @always_inline
    def broadcast_nd[
        op_code: Int
    ](
        a: Layout,
        a_buffer: Buffer[Self.dtype],
        b: Layout,
        b_buffer: Buffer[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> Tuple[Layout, Buffer[Self.dtype]]:
        var result_shape = ShapeBroadcaster.broadcast_shape(a.shape, b.shape)

        # 1. Compute effective strides
        var rank = result_shape.rank()
        var a_rank = a.shape.rank()
        var b_rank = b.shape.rank()
        var extra_a = rank - a_rank  # leading dims `a` doesn't have
        var extra_b = rank - b_rank  # leading dims `b` doesn't have

        var a_eff = IntArray.with_capacity(rank)
        var b_eff = IntArray.with_capacity(rank)

        for i in range(rank):
            var a_i = i - extra_a
            if a_i < 0:
                a_eff.append(0)  # prepended dim → broadcast
            elif a.shape[a_i] == 1 and result_shape[i] > 1:
                a_eff.append(0)  # size‑1 dim stretched → broadcast
            else:
                a_eff.append(a.strides[a_i])

            var b_i = i - extra_b
            if b_i < 0:
                b_eff.append(0)
            elif b.shape[b_i] == 1 and result_shape[i] > 1:
                b_eff.append(0)
            else:
                b_eff.append(b.strides[b_i])

        # 2. Allocate output
        var buffer = Buffer[Self.dtype](result_shape.num_elements())
        var total = buffer.size

        # SIMD width: 1 for bool (stored as uint8, no SIMD benefit),
        # hardware native for everything else.
        comptime simd_width = simd_width_of[
            Self.dtype
        ]() if Self.dtype != DType.bool else 1

        # TIER 1  —  both operands have unit stride in the last dimension
        # The last dimension is dense in both buffers.  Tiling it with
        # SIMD loads/stores gives maximum throughput.
        #
        # Outer rows are tiled independently: each worker reconstructs its
        # operand offsets from effective strides (stride‑0 dimensions stay
        # put naturally), so rows run in parallel.

        if (
            simd_width > 1
            and rank >= 1
            and a_eff[rank - 1] == 1
            and b_eff[rank - 1] == 1
        ):
            var last_dim = result_shape[rank - 1]
            var outer_rank = rank - 1
            var outer_count = total // last_dim
            var n_threads = num_physical_cores()
            var a_eff_c = a_eff
            var b_eff_c = b_eff
            var result_shape_c = result_shape
            var a_off_base = a.offset
            var b_off_base = b.offset

            def worker_tier1(outer_idx: Int) {imm}:
                # Per-row offsets instead of a serial odometer
                var coords = IndexCalculator.index_to_coord(
                    result_shape_c, outer_idx * last_dim
                )
                var a_off = a_off_base
                var b_off = b_off_base
                for d in range(outer_rank):
                    a_off += coords[d] * a_eff_c[d]
                    b_off += coords[d] * b_eff_c[d]
                var out_base = outer_idx * last_dim

                # SIMD tile the last dimension
                var j = 0
                while j + simd_width <= last_dim:
                    var a_v = a_buffer.load[simdwidth=simd_width](a_off + j)
                    var b_v = b_buffer.load[simdwidth=simd_width](b_off + j)
                    var op_result: SIMD[Self.dtype, simd_width]

                    # Path 3b: SIMD vector op via shared helper
                    op_result = simd_op[op_code, Self.dtype, simd_width](
                        a_v, b_v, epsilon
                    )

                    buffer.store[simdwidth=simd_width](out_base + j, op_result)
                    j += simd_width

                # Scalar remainder (last_dim not a multiple of simd_width)
                for k in range(j, last_dim):
                    buffer[out_base + k] = scalar_op[op_code, Self.dtype](
                        a_buffer[a_off + k],
                        b_buffer[b_off + k],
                        epsilon,
                    )

            if outer_count >= n_threads and total >= n_threads * 32768:
                parallelize(worker_tier1, outer_count, n_threads)
            else:
                for outer_idx in range(outer_count):
                    worker_tier1(outer_idx)

        # TIER 2  —  one operand broadcasts in the last dim, the other
        #            has unit stride
        # The broadcasting operand reads the same scalar for every
        # position in the last dimension (effective stride == 0).
        # We splat it to a SIMD register once per outer row, then
        # SIMD‑load from the contiguous operand and vector‑op.
        #
        # Operand roles are determined by an `a_broadcasts_last` flag.

        elif (
            simd_width > 1
            and rank >= 1
            and (
                (a_eff[rank - 1] == 1 and b_eff[rank - 1] == 0)
                or (b_eff[rank - 1] == 1 and a_eff[rank - 1] == 0)
            )
        ):
            var a_broadcasts_last = a_eff[rank - 1] != 1
            var last_dim = result_shape[rank - 1]
            var outer_rank = rank - 1
            var outer_count = total // last_dim
            var n_threads = num_physical_cores()
            var a_eff_c = a_eff
            var b_eff_c = b_eff
            var result_shape_c = result_shape
            var a_off_base = a.offset
            var b_off_base = b.offset

            def worker_tier2(outer_idx: Int) {imm}:
                # Per-row offsets instead of a serial odometer
                var coords = IndexCalculator.index_to_coord(
                    result_shape_c, outer_idx * last_dim
                )
                var a_off = a_off_base
                var b_off = b_off_base
                for d in range(outer_rank):
                    a_off += coords[d] * a_eff_c[d]
                    b_off += coords[d] * b_eff_c[d]
                var out_base = outer_idx * last_dim

                # Read the scalar that will be broadcast across this row
                var scalar_v = a_buffer[
                    a_off
                ] if a_broadcasts_last else b_buffer[b_off]

                var scalar_vec = SIMD[Self.dtype, simd_width](scalar_v)
                var j = 0
                while j + simd_width <= last_dim:
                    # SIMD load from the *non‑broadcasting* side
                    var vec = b_buffer.load[simdwidth=simd_width](
                        b_off + j
                    ) if a_broadcasts_last else a_buffer.load[
                        simdwidth=simd_width
                    ](
                        a_off + j
                    )
                    var op_result: SIMD[Self.dtype, simd_width]

                    # The scalar is always the *first* operand in the
                    # comptime branch; the flag controls which side is
                    # the scalar and which is the vector.
                    if a_broadcasts_last:
                        op_result = simd_op[op_code, Self.dtype, simd_width](
                            scalar_vec, vec, epsilon
                        )
                    else:
                        op_result = simd_op[op_code, Self.dtype, simd_width](
                            vec, scalar_vec, epsilon
                        )

                    buffer.store[simdwidth=simd_width](out_base + j, op_result)
                    j += simd_width

                # Scalar remainder
                for k in range(j, last_dim):
                    var a_i = a_off if a_broadcasts_last else (a_off + k)
                    var b_i = b_off + k if a_broadcasts_last else b_off
                    buffer[out_base + k] = scalar_op[op_code, Self.dtype](
                        a_buffer[a_i],
                        b_buffer[b_i],
                        epsilon,
                    )

            if outer_count >= n_threads and total >= n_threads * 32768:
                parallelize(worker_tier2, outer_count, n_threads)
            else:
                for outer_idx in range(outer_count):
                    worker_tier2(outer_idx)

        # TIER 3  —  general scalar odometer
        # Neither operand has unit stride in the last dimension.
        # This happens when:
        #   - both are transposed views (non‑unit last stride)
        #   - both broadcast in the last dim (both stride == 0)
        #   - dtype is bool (simd_width forced to 1)
        #
        # We walk every element with a full rank‑dimensional odometer,
        # reading through effective strides.  No SIMD, but correct
        # for any valid broadcast pair.

        else:
            var a_last = a_eff[rank - 1] if rank >= 1 else 0
            var b_last = b_eff[rank - 1] if rank >= 1 else 0
            var last_dim = result_shape[rank - 1] if rank >= 1 else 0

            if rank < 1 or last_dim < 1:
                # Degenerate shape — original serial odometer.
                var a_off = a.offset
                var b_off = b.offset
                var coords = IntArray.filled(rank, 0)

                for i in range(total):
                    buffer[i] = scalar_op[op_code, Self.dtype](
                        a_buffer[a_off], b_buffer[b_off], epsilon
                    )

                    for d in range(rank - 1, -1, -1):
                        coords[d] += 1
                        if coords[d] < result_shape[d]:
                            a_off += a_eff[d]
                            b_off += b_eff[d]
                            break
                        else:
                            a_off -= (result_shape[d] - 1) * a_eff[d]
                            b_off -= (result_shape[d] - 1) * b_eff[d]
                            coords[d] = 0

            else:
                var rows = total // last_dim
                var n_threads = num_physical_cores()
                var n_segments = 1
                if total >= n_threads * 4096 and rows > 1:
                    n_segments = rows if rows < n_threads else n_threads

                var a_eff_c = a_eff
                var b_eff_c = b_eff
                var result_shape_c = result_shape

                def worker_tier3(seg: Int) {imm}:
                    var r0 = seg * rows // n_segments
                    var r1 = (seg + 1) * rows // n_segments
                    for r in range(r0, r1):
                        var coords = IndexCalculator.index_to_coord(
                            result_shape_c, r * last_dim
                        )
                        var a_off = a.offset
                        var b_off = b.offset
                        for d in range(rank):
                            a_off += coords[d] * a_eff_c[d]
                            b_off += coords[d] * b_eff_c[d]
                        var out_base = r * last_dim
                        for k in range(last_dim):
                            buffer[out_base + k] = scalar_op[
                                op_code, Self.dtype
                            ](a_buffer[a_off], b_buffer[b_off], epsilon)
                            a_off += a_last
                            b_off += b_last

                if n_segments > 1:
                    parallelize(worker_tier3, n_segments, n_threads)
                else:
                    worker_tier3(0)

        return (Layout(result_shape), buffer^)

    @staticmethod
    @always_inline
    def compute[
        op_code: Int,
    ](
        self: Layout,
        self_buffer: Buffer[Self.dtype],
        other: Layout,
        other_buffer: Buffer[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> Tuple[Layout, Buffer[Self.dtype]]:
        # Handle broadcasting case
        if self.shape != other.shape:
            return Self.broadcast[op_code](
                self, self_buffer, other, other_buffer, epsilon
            )
        # Same shape
        if self.is_contiguous() and other.is_contiguous():
            var self_start = self.offset
            var self_end = self_start + self.numel()
            var other_start = other.offset
            var other_end = other_start + other.numel()
            var result_buffer = self_buffer.arithmetic_ops[op_code=op_code](
                other_buffer,
                self_start,
                self_end,
                other_start,
                other_end,
                epsilon=epsilon,
            )
            return (Layout(self.shape), result_buffer^)

        else:
            var result_buffer = Buffer[Self.dtype](self.numel())
            var total = self.numel()
            comptime simd_width = simd_width_of[
                Self.dtype
            ]() if Self.dtype != DType.bool else 1

            var rank = self.shape.rank()
            var last_dim = self.shape[rank - 1] if rank >= 1 else 0

            if rank < 1 or last_dim < 1:
                # Degenerate shape (rank-0 / empty) — serial fallback.
                var index = 0
                var o_shape = other.shape
                var o_strides = other.strides
                var it = IndexIterator(
                    shape=Pointer(to=o_shape).as_imm(),
                    strides=Pointer(to=o_strides).as_imm(),
                    start_offset=other.offset,
                )
                while it.__has_next__():
                    var idx = it.peek()
                    result_buffer[index] = scalar_op[op_code, Self.dtype](
                        self_buffer[self.offset + index],
                        other_buffer[idx],
                        epsilon,
                    )
                    index += 1
                    it.skip(1)

            else:
                # Row-parallel path: split the shared logical rows across
                # workers.  Each worker positions a fresh IndexIterator at its
                # first row via skip() (O(rank) direct computation), then per
                # row reads the row base with peek() and walks the last
                # dimension with direct stride math.  No raises, no
                # per-element odometer.
                var rows = total // last_dim
                var n_threads = num_physical_cores()
                var n_segments = 1
                if total >= n_threads * 4096 and rows > 1:
                    n_segments = rows if rows < n_threads else n_threads

                var r_data = result_buffer.data.unsafe_value()
                var s_data = self_buffer.data.unsafe_value()
                var s_offset = self.offset
                var s_inner = self.strides[rank - 1]
                var o_data = other_buffer.data.unsafe_value()
                var o_offset = other.offset
                var o_inner = other.strides[rank - 1]

                if self.is_contiguous() and not other.is_contiguous():
                    var o_shape = other.shape
                    var o_strides = other.strides

                    def worker_contig_self(seg: Int) {imm}:
                        var r0 = seg * rows // n_segments
                        var r1 = (seg + 1) * rows // n_segments
                        var it = IndexIterator(
                            shape=Pointer(to=o_shape).as_imm(),
                            strides=Pointer(to=o_strides).as_imm(),
                            start_offset=o_offset,
                        )
                        it.skip(r0 * last_dim)
                        var flat = r0 * last_dim
                        for _ in range(r0, r1):
                            var row_off = it.peek()
                            var j = 0
                            comptime if simd_width > 1:
                                while j + simd_width <= last_dim:
                                    var av = s_data.unsafe_load[
                                        width=simd_width
                                    ](s_offset + flat + j)
                                    var bv = strided_load[simd_width](
                                        o_data.unsafe_offset(
                                            row_off + j * o_inner
                                        ),
                                        o_inner,
                                    )
                                    r_data.unsafe_store[width=simd_width](
                                        flat + j,
                                        simd_op[
                                            op_code, Self.dtype, simd_width
                                        ](av, bv, epsilon),
                                    )
                                    j += simd_width
                            for k in range(j, last_dim):
                                r_data[unsafe_offset=flat + k] = scalar_op[
                                    op_code, Self.dtype
                                ](
                                    s_data[unsafe_offset=s_offset + flat + k],
                                    o_data[unsafe_offset=row_off + k * o_inner],
                                    epsilon,
                                )
                            flat += last_dim
                            it.skip(last_dim)

                    if n_segments > 1:
                        parallelize(worker_contig_self, n_segments, n_threads)
                    else:
                        worker_contig_self(0)

                elif not self.is_contiguous() and other.is_contiguous():
                    var s_shape = self.shape
                    var s_strides = self.strides

                    def worker_strided_self(seg: Int) {imm}:
                        var r0 = seg * rows // n_segments
                        var r1 = (seg + 1) * rows // n_segments
                        var it = IndexIterator(
                            shape=Pointer(to=s_shape).as_imm(),
                            strides=Pointer(to=s_strides).as_imm(),
                            start_offset=s_offset,
                        )
                        it.skip(r0 * last_dim)
                        var flat = r0 * last_dim
                        for _ in range(r0, r1):
                            var row_off = it.peek()
                            var j = 0
                            comptime if simd_width > 1:
                                while j + simd_width <= last_dim:
                                    var av = strided_load[simd_width](
                                        s_data.unsafe_offset(
                                            row_off + j * s_inner
                                        ),
                                        s_inner,
                                    )
                                    var bv = o_data.unsafe_load[
                                        width=simd_width
                                    ](o_offset + flat + j)
                                    r_data.unsafe_store[width=simd_width](
                                        flat + j,
                                        simd_op[
                                            op_code, Self.dtype, simd_width
                                        ](av, bv, epsilon),
                                    )
                                    j += simd_width
                            for k in range(j, last_dim):
                                r_data[unsafe_offset=flat + k] = scalar_op[
                                    op_code, Self.dtype
                                ](
                                    s_data[unsafe_offset=row_off + k * s_inner],
                                    o_data[unsafe_offset=o_offset + flat + k],
                                    epsilon,
                                )
                            flat += last_dim
                            it.skip(last_dim)

                    if n_segments > 1:
                        parallelize(worker_strided_self, n_segments, n_threads)
                    else:
                        worker_strided_self(0)

                else:
                    var s_shape = self.shape
                    var s_strides = self.strides
                    var o_shape = other.shape
                    var o_strides = other.strides

                    def worker_both_strided(seg: Int) {imm}:
                        var r0 = seg * rows // n_segments
                        var r1 = (seg + 1) * rows // n_segments
                        var it_s = IndexIterator(
                            shape=Pointer(to=s_shape).as_imm(),
                            strides=Pointer(to=s_strides).as_imm(),
                            start_offset=s_offset,
                        )
                        var it_o = IndexIterator(
                            shape=Pointer(to=o_shape).as_imm(),
                            strides=Pointer(to=o_strides).as_imm(),
                            start_offset=o_offset,
                        )
                        it_s.skip(r0 * last_dim)
                        it_o.skip(r0 * last_dim)
                        var flat = r0 * last_dim
                        for _ in range(r0, r1):
                            var s_row = it_s.peek()
                            var o_row = it_o.peek()
                            var j = 0
                            comptime if simd_width > 1:
                                while j + simd_width <= last_dim:
                                    var av = strided_load[simd_width](
                                        s_data.unsafe_offset(
                                            s_row + j * s_inner
                                        ),
                                        s_inner,
                                    )
                                    var bv = strided_load[simd_width](
                                        o_data.unsafe_offset(
                                            o_row + j * o_inner
                                        ),
                                        o_inner,
                                    )
                                    r_data.unsafe_store[width=simd_width](
                                        flat + j,
                                        simd_op[
                                            op_code, Self.dtype, simd_width
                                        ](av, bv, epsilon),
                                    )
                                    j += simd_width
                            for k in range(j, last_dim):
                                r_data[unsafe_offset=flat + k] = scalar_op[
                                    op_code, Self.dtype
                                ](
                                    s_data[unsafe_offset=s_row + k * s_inner],
                                    o_data[unsafe_offset=o_row + k * o_inner],
                                    epsilon,
                                )
                            flat += last_dim
                            it_s.skip(last_dim)
                            it_o.skip(last_dim)

                    if n_segments > 1:
                        parallelize(worker_both_strided, n_segments, n_threads)
                    else:
                        worker_both_strided(0)

            return (Layout(self.shape), result_buffer^)

    @staticmethod
    @always_inline
    def compute[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](
        self: Layout,
        self_buffer: Buffer[Self.dtype],
        scalar: Scalar[Self.dtype],
    ) -> Tuple[Layout, Buffer[Self.dtype]]:
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numel()
            var result_buffer: Buffer[Self.dtype]

            comptime if op_code == POW:
                result_buffer = self_buffer[start:end] ** scalar
            else:
                result_buffer = self_buffer.arithmetic_ops_scalar[op_code](
                    scalar, start, end
                )
            return (Layout(self.shape), result_buffer^)

        else:
            var total = self.numel()
            var result_buffer = Buffer[Self.dtype](total)
            comptime simd_width = simd_width_of[
                Self.dtype
            ]() if Self.dtype != DType.bool else 1

            var rank = self.shape.rank()
            var last_dim = self.shape[rank - 1] if rank >= 1 else 0

            if rank < 1 or last_dim < 1:
                var index = 0
                var s_shape = self.shape
                var s_strides = self.strides
                var it = IndexIterator(
                    shape=Pointer(to=s_shape).as_imm(),
                    strides=Pointer(to=s_strides).as_imm(),
                    start_offset=self.offset,
                )
                while it.__has_next__():
                    var idx = it.peek()
                    result_buffer[index] = scalar_op[op_code, Self.dtype](
                        self_buffer[idx], scalar, epsilon
                    )
                    index += 1
                    it.skip(1)
            else:
                var rows = total // last_dim
                var n_threads = num_physical_cores()
                var n_segments = 1
                if total >= n_threads * 4096 and rows > 1:
                    n_segments = rows if rows < n_threads else n_threads

                var r_data = result_buffer.data.unsafe_value()
                var s_data = self_buffer.data.unsafe_value()
                var s_offset = self.offset
                var s_inner = self.strides[rank - 1]
                var s_shape = self.shape
                var s_strides = self.strides

                def worker_compute_scalar(seg: Int) {imm}:
                    var r0 = seg * rows // n_segments
                    var r1 = (seg + 1) * rows // n_segments
                    var it = IndexIterator(
                        shape=Pointer(to=s_shape).as_imm(),
                        strides=Pointer(to=s_strides).as_imm(),
                        start_offset=s_offset,
                    )
                    it.skip(r0 * last_dim)
                    var flat = r0 * last_dim
                    for _ in range(r0, r1):
                        var row_off = it.peek()
                        var j = 0
                        comptime if simd_width > 1:
                            while j + simd_width <= last_dim:
                                var av = strided_load[simd_width](
                                    s_data.unsafe_offset(row_off + j * s_inner),
                                    s_inner,
                                )
                                r_data.unsafe_store[width=simd_width](
                                    flat + j,
                                    simd_op[op_code, Self.dtype, simd_width](
                                        av, scalar, epsilon
                                    ),
                                )
                                j += simd_width
                        for k in range(j, last_dim):
                            r_data[unsafe_offset=flat + k] = scalar_op[
                                op_code, Self.dtype
                            ](
                                s_data[unsafe_offset=row_off + k * s_inner],
                                scalar,
                                epsilon,
                            )
                        flat += last_dim
                        it.skip(last_dim)

                if n_segments > 1:
                    parallelize(worker_compute_scalar, n_segments, n_threads)
                else:
                    worker_compute_scalar(0)

            return (Layout(self.shape), result_buffer^)

    @staticmethod
    @always_inline
    def unary_ops[
        op_code: Int,
    ](self: Layout, self_buffer: Buffer[Self.dtype]) -> Tuple[
        Layout, Buffer[Self.dtype]
    ]:
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numel()
            var result_buffer = self_buffer.unary_ops[op_code](start, end)

            return (Layout(self.shape), result_buffer^)

        else:
            var total = self.numel()
            var result_buffer = Buffer[Self.dtype](total)

            var rank = self.shape.rank()
            var last_dim = self.shape[rank - 1] if rank >= 1 else 0

            if rank < 1 or last_dim < 1:
                var index = 0
                var s_shape = self.shape
                var s_strides = self.strides
                var it = IndexIterator(
                    shape=Pointer(to=s_shape).as_imm(),
                    strides=Pointer(to=s_strides).as_imm(),
                    start_offset=self.offset,
                )
                while it.__has_next__():
                    var idx = it.peek()
                    result_buffer[index] = unary_op[op_code, Self.dtype](
                        self_buffer[idx]
                    )
                    index += 1
                    it.skip(1)
            else:
                var rows = total // last_dim
                var n_threads = num_physical_cores()
                var n_segments = 1
                if total >= n_threads * 4096 and rows > 1:
                    n_segments = rows if rows < n_threads else n_threads

                var r_data = result_buffer.data.unsafe_value()
                var s_data = self_buffer.data.unsafe_value()
                var s_offset = self.offset
                var s_inner = self.strides[rank - 1]
                var s_shape = self.shape
                var s_strides = self.strides

                def worker_unary(seg: Int) {imm}:
                    var r0 = seg * rows // n_segments
                    var r1 = (seg + 1) * rows // n_segments
                    var it = IndexIterator(
                        shape=Pointer(to=s_shape).as_imm(),
                        strides=Pointer(to=s_strides).as_imm(),
                        start_offset=s_offset,
                    )
                    it.skip(r0 * last_dim)
                    var flat = r0 * last_dim
                    comptime simd_width = (
                        1
                        if Self.dtype == DType.bool
                        else simd_width_of[Self.dtype]()
                    )
                    for _ in range(r0, r1):
                        var row_off = it.peek()
                        var j = 0
                        comptime if simd_width > 1:
                            while j + simd_width <= last_dim:
                                var srcv = strided_load[simd_width](
                                    s_data.unsafe_offset(
                                        row_off + j * s_inner
                                    ),
                                    s_inner,
                                )
                                r_data.unsafe_store[width=simd_width](
                                    flat + j,
                                    unary_op_simd[
                                        op_code, Self.dtype, simd_width
                                    ](srcv),
                                )
                                j += simd_width
                        for k in range(j, last_dim):
                            r_data[unsafe_offset=flat + k] = unary_op[
                                op_code, Self.dtype
                            ](s_data[unsafe_offset=row_off + k * s_inner])
                        flat += last_dim
                        it.skip(last_dim)

                if n_segments > 1:
                    parallelize(worker_unary, n_segments, n_threads)
                else:
                    worker_unary(0)

            return (Layout(self.shape), result_buffer^)

    @staticmethod
    @always_inline
    def unary_ops_constrained[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](self: Layout, self_buffer: Buffer[Self.dtype]) -> Tuple[
        Layout, Buffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numel()
            var result_buffer = self_buffer.float_unary_ops[op_code, epsilon](
                start, end
            )
            return (Layout(self.shape), result_buffer^)
        else:
            var total = self.numel()
            var result_buffer = Buffer[Self.dtype](total)

            var rank = self.shape.rank()
            var last_dim = self.shape[rank - 1] if rank >= 1 else 0

            if rank < 1 or last_dim < 1:
                var index = 0
                var s_shape = self.shape
                var s_strides = self.strides
                var it = IndexIterator(
                    shape=Pointer(to=s_shape).as_imm(),
                    strides=Pointer(to=s_strides).as_imm(),
                    start_offset=self.offset,
                )
                while it.__has_next__():
                    var idx = it.peek()
                    result_buffer[index] = float_unary_op[
                        op_code, Self.dtype, epsilon
                    ](self_buffer[idx])
                    index += 1
                    it.skip(1)
            else:
                var rows = total // last_dim
                var n_threads = num_physical_cores()
                var n_segments = 1
                if total >= n_threads * 4096 and rows > 1:
                    n_segments = rows if rows < n_threads else n_threads

                var r_data = result_buffer.data.unsafe_value()
                var s_data = self_buffer.data.unsafe_value()
                var s_offset = self.offset
                var s_inner = self.strides[rank - 1]
                var s_shape = self.shape
                var s_strides = self.strides

                def worker_unary_float(seg: Int) {imm}:
                    var r0 = seg * rows // n_segments
                    var r1 = (seg + 1) * rows // n_segments
                    var it = IndexIterator(
                        shape=Pointer(to=s_shape).as_imm(),
                        strides=Pointer(to=s_strides).as_imm(),
                        start_offset=s_offset,
                    )
                    it.skip(r0 * last_dim)
                    var flat = r0 * last_dim
                    comptime simd_width = simd_width_of[Self.dtype]()
                    for _ in range(r0, r1):
                        var row_off = it.peek()
                        var j = 0
                        while j + simd_width <= last_dim:
                            var srcv = strided_load[simd_width](
                                s_data.unsafe_offset(row_off + j * s_inner),
                                s_inner,
                            )
                            r_data.unsafe_store[width=simd_width](
                                flat + j,
                                float_unary_op_simd[
                                    op_code, Self.dtype, simd_width, epsilon
                                ](srcv),
                            )
                            j += simd_width
                        for k in range(j, last_dim):
                            r_data[unsafe_offset=flat + k] = float_unary_op[
                                op_code, Self.dtype, epsilon
                            ](s_data[unsafe_offset=row_off + k * s_inner])
                        flat += last_dim
                        it.skip(last_dim)

                if n_segments > 1:
                    parallelize(worker_unary_float, n_segments, n_threads)
                else:
                    worker_unary_float(0)

            return (Layout(self.shape), result_buffer^)

    @staticmethod
    @always_inline
    def inplace_ops[
        op_code: Int,
    ](
        self: Layout,
        self_buffer: Buffer[Self.dtype],
        other: Layout,
        other_buffer: Buffer[Self.dtype],
    ):

        # Handle broadcasting case
        if self.shape != other.shape:
            var broadcast_shape = ShapeBroadcaster.broadcast_shape(
                self.shape, other.shape
            )

            # PyTorch's rule: broadcasted shape must match receiver shape
            if broadcast_shape != self.shape:
                panic(
                    "NDBuffer → inplace_ops: broadcasted shape "
                    + String(broadcast_shape)
                    + " must match receiver shape "
                    + String(self.shape)
                )

            var broadcast_result = CpuArithmeticOps[Self.dtype].broadcast[
                op_code
            ](self, self_buffer, other, other_buffer)
            var src_shape = broadcast_result[0].shape
            var src_strides = broadcast_result[0].strides
            var dst_shape = self.shape
            var dst_strides = self.strides
            var it_src = IndexIterator(
                shape=Pointer(to=src_shape).as_imm(),
                strides=Pointer(to=src_strides).as_imm(),
                start_offset=broadcast_result[0].offset,
            )
            var it_dst = IndexIterator(
                shape=Pointer(to=dst_shape).as_imm(),
                strides=Pointer(to=dst_strides).as_imm(),
                start_offset=self.offset,
            )
            while it_dst.__has_next__():
                self_buffer[it_dst.peek()] = broadcast_result[1][it_src.peek()]
                it_dst.skip(1)
                it_src.skip(1)

        else:
            # Same shape case
            if self.is_contiguous() and other.is_contiguous():
                var self_start = self.offset
                var self_end = self_start + self.numel()
                var other_start = other.offset
                var other_end = other_start + other.numel()
                self_buffer.inplace_ops[op_code](
                    other_buffer, self_start, self_end, other_start, other_end
                )

            elif self.is_contiguous() and not other.is_contiguous():
                var total = self.numel()
                comptime simd_width = simd_width_of[
                    Self.dtype
                ]() if Self.dtype != DType.bool else 1

                var rank = self.shape.rank()
                var last_dim = self.shape[rank - 1] if rank >= 1 else 0

                if rank < 1 or last_dim < 1:
                    var index = self.offset
                    var o_shape = other.shape
                    var o_strides = other.strides
                    var it = IndexIterator(
                        shape=Pointer(to=o_shape).as_imm(),
                        strides=Pointer(to=o_strides).as_imm(),
                        start_offset=other.offset,
                    )
                    while it.__has_next__():
                        var idx = it.peek()
                        self_buffer[index] = scalar_op[op_code, Self.dtype](
                            self_buffer[index], other_buffer[idx]
                        )
                        index += 1
                        it.skip(1)
                else:
                    var rows = total // last_dim
                    var n_threads = num_physical_cores()
                    var n_segments = 1
                    if total >= n_threads * 4096 and rows > 1:
                        n_segments = rows if rows < n_threads else n_threads

                    var s_data = self_buffer.data.unsafe_value()
                    var s_offset = self.offset
                    var o_data = other_buffer.data.unsafe_value()
                    var o_offset = other.offset
                    var o_inner = other.strides[rank - 1]
                    var o_shape = other.shape
                    var o_strides = other.strides

                    def worker_inplace_contig_self(seg: Int) {imm}:
                        var r0 = seg * rows // n_segments
                        var r1 = (seg + 1) * rows // n_segments
                        var it = IndexIterator(
                            shape=Pointer(to=o_shape).as_imm(),
                            strides=Pointer(to=o_strides).as_imm(),
                            start_offset=o_offset,
                        )
                        it.skip(r0 * last_dim)
                        var flat = r0 * last_dim
                        for _ in range(r0, r1):
                            var row_off = it.peek()
                            var j = 0
                            comptime if simd_width > 1:
                                while j + simd_width <= last_dim:
                                    var av = s_data.unsafe_load[
                                        width=simd_width
                                    ](s_offset + flat + j)
                                    var bv = strided_load[simd_width](
                                        o_data.unsafe_offset(
                                            row_off + j * o_inner
                                        ),
                                        o_inner,
                                    )
                                    s_data.unsafe_store[width=simd_width](
                                        s_offset + flat + j,
                                        simd_op[
                                            op_code, Self.dtype, simd_width
                                        ](av, bv),
                                    )
                                    j += simd_width
                            for k in range(j, last_dim):
                                s_data[
                                    unsafe_offset=s_offset + flat + k
                                ] = scalar_op[op_code, Self.dtype](
                                    s_data[unsafe_offset=s_offset + flat + k],
                                    o_data[unsafe_offset=row_off + k * o_inner],
                                )
                            flat += last_dim
                            it.skip(last_dim)

                    if n_segments > 1:
                        parallelize(worker_inplace_contig_self, 
                            n_segments, n_threads
                        )
                    else:
                        worker_inplace_contig_self(0)

            elif not self.is_contiguous() and other.is_contiguous():
                var total = self.numel()
                var rank = self.shape.rank()
                var last_dim = self.shape[rank - 1] if rank >= 1 else 0

                if rank < 1 or last_dim < 1:
                    var index = other.offset
                    var s_shape = self.shape
                    var s_strides = self.strides
                    var it = IndexIterator(
                        shape=Pointer(to=s_shape).as_imm(),
                        strides=Pointer(to=s_strides).as_imm(),
                        start_offset=self.offset,
                    )
                    while it.__has_next__():
                        var idx = it.peek()
                        self_buffer[idx] = scalar_op[op_code, Self.dtype](
                            self_buffer[idx], other_buffer[index]
                        )
                        index += 1
                        it.skip(1)
                else:
                    var rows = total // last_dim
                    var n_threads = num_physical_cores()
                    var n_segments = 1
                    if total >= n_threads * 4096 and rows > 1:
                        n_segments = rows if rows < n_threads else n_threads

                    var s_data = self_buffer.data.unsafe_value()
                    var s_inner = self.strides[rank - 1]
                    var s_shape = self.shape
                    var s_strides = self.strides
                    var o_data = other_buffer.data.unsafe_value()
                    var o_offset = other.offset

                    def worker_inplace_strided_self(seg: Int) {imm}:
                        var r0 = seg * rows // n_segments
                        var r1 = (seg + 1) * rows // n_segments
                        var it = IndexIterator(
                            shape=Pointer(to=s_shape).as_imm(),
                            strides=Pointer(to=s_strides).as_imm(),
                            start_offset=self.offset,
                        )
                        it.skip(r0 * last_dim)
                        var flat = r0 * last_dim
                        for _ in range(r0, r1):
                            var row_off = it.peek()
                            for k in range(last_dim):
                                s_data[
                                    unsafe_offset=row_off + k * s_inner
                                ] = scalar_op[op_code, Self.dtype](
                                    s_data[unsafe_offset=row_off + k * s_inner],
                                    o_data[unsafe_offset=o_offset + flat + k],
                                )
                            flat += last_dim
                            it.skip(last_dim)

                    if n_segments > 1:
                        parallelize(worker_inplace_strided_self, 
                            n_segments, n_threads
                        )
                    else:
                        worker_inplace_strided_self(0)

            else:
                var total = self.numel()
                var rank = self.shape.rank()
                var last_dim = self.shape[rank - 1] if rank >= 1 else 0

                if rank < 1 or last_dim < 1:
                    var s_shape = self.shape
                    var s_strides = self.strides
                    var o_shape = other.shape
                    var o_strides = other.strides
                    var it_self = IndexIterator(
                        shape=Pointer(to=s_shape).as_imm(),
                        strides=Pointer(to=s_strides).as_imm(),
                        start_offset=self.offset,
                    )
                    var it_other = IndexIterator(
                        shape=Pointer(to=o_shape).as_imm(),
                        strides=Pointer(to=o_strides).as_imm(),
                        start_offset=other.offset,
                    )
                    while it_self.__has_next__():
                        var index = it_self.peek()
                        var next_index = it_other.peek()
                        self_buffer[index] = scalar_op[op_code, Self.dtype](
                            self_buffer[index], other_buffer[next_index]
                        )
                        it_self.skip(1)
                        it_other.skip(1)
                else:
                    var rows = total // last_dim
                    var n_threads = num_physical_cores()
                    var n_segments = 1
                    if total >= n_threads * 4096 and rows > 1:
                        n_segments = rows if rows < n_threads else n_threads

                    var s_data = self_buffer.data.unsafe_value()
                    var s_inner = self.strides[rank - 1]
                    var s_shape = self.shape
                    var s_strides = self.strides
                    var o_data = other_buffer.data.unsafe_value()
                    var o_inner = other.strides[rank - 1]
                    var o_shape = other.shape
                    var o_strides = other.strides

                    def worker_inplace_both_strided(seg: Int) {imm}:
                        var r0 = seg * rows // n_segments
                        var r1 = (seg + 1) * rows // n_segments
                        var it_s = IndexIterator(
                            shape=Pointer(to=s_shape).as_imm(),
                            strides=Pointer(to=s_strides).as_imm(),
                            start_offset=self.offset,
                        )
                        var it_o = IndexIterator(
                            shape=Pointer(to=o_shape).as_imm(),
                            strides=Pointer(to=o_strides).as_imm(),
                            start_offset=other.offset,
                        )
                        it_s.skip(r0 * last_dim)
                        it_o.skip(r0 * last_dim)
                        for _ in range(r0, r1):
                            var s_row = it_s.peek()
                            var o_row = it_o.peek()
                            for k in range(last_dim):
                                s_data[
                                    unsafe_offset=s_row + k * s_inner
                                ] = scalar_op[op_code, Self.dtype](
                                    s_data[unsafe_offset=s_row + k * s_inner],
                                    o_data[unsafe_offset=o_row + k * o_inner],
                                )
                            it_s.skip(last_dim)
                            it_o.skip(last_dim)

                    if n_segments > 1:
                        parallelize(worker_inplace_both_strided, 
                            n_segments, n_threads
                        )
                    else:
                        worker_inplace_both_strided(0)

    @staticmethod
    @always_inline
    def inplace_scalar_ops[
        op_code: Int,
    ](
        self: Layout,
        self_buffer: Buffer[Self.dtype],
        scalar: Scalar[Self.dtype],
    ):
        comptime if op_code == Divide:
            if scalar == Scalar[Self.dtype](0):
                panic("NDBuffer → inplace_scalar_ops: cannot divide by zero")

        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numel()
            self_buffer.inplace_ops_scalar[op_code](scalar, start, end)

        else:
            var total = self.numel()
            var rank = self.shape.rank()
            var last_dim = self.shape[rank - 1] if rank >= 1 else 0

            if rank < 1 or last_dim < 1:
                var s_shape = self.shape
                var s_strides = self.strides
                var it = IndexIterator(
                    shape=Pointer(to=s_shape).as_imm(),
                    strides=Pointer(to=s_strides).as_imm(),
                    start_offset=self.offset,
                )
                while it.__has_next__():
                    var index = it.peek()
                    self_buffer[index] = scalar_op[op_code, Self.dtype](
                        self_buffer[index], scalar
                    )
                    it.skip(1)
            else:
                var rows = total // last_dim
                var n_threads = num_physical_cores()
                var n_segments = 1
                if total >= n_threads * 4096 and rows > 1:
                    n_segments = rows if rows < n_threads else n_threads

                var s_data = self_buffer.data.unsafe_value()
                var s_inner = self.strides[rank - 1]
                var s_shape = self.shape
                var s_strides = self.strides

                def worker_inplace_scalar(seg: Int) {imm}:
                    var r0 = seg * rows // n_segments
                    var r1 = (seg + 1) * rows // n_segments
                    var it = IndexIterator(
                        shape=Pointer(to=s_shape).as_imm(),
                        strides=Pointer(to=s_strides).as_imm(),
                        start_offset=self.offset,
                    )
                    it.skip(r0 * last_dim)
                    for _ in range(r0, r1):
                        var row_off = it.peek()
                        for k in range(last_dim):
                            s_data[
                                unsafe_offset=row_off + k * s_inner
                            ] = scalar_op[op_code, Self.dtype](
                                s_data[unsafe_offset=row_off + k * s_inner],
                                scalar,
                            )
                        it.skip(last_dim)

                if n_segments > 1:
                    parallelize(worker_inplace_scalar, n_segments, n_threads)
                else:
                    worker_inplace_scalar(0)
