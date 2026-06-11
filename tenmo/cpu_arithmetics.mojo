# === CPU Broadcast Engine ===
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

from tenmo.ndbuffer import NDBuffer
from tenmo.buffers import Buffer
from tenmo.strides import Strides
from tenmo.intarray import IntArray
from tenmo.broadcasthelper import ShapeBroadcaster
from tenmo.shared.scalar_ops import simd_op, scalar_op, unary_op, float_unary_op
from tenmo.common_utils import Epsilon
from tenmo.mnemonics import (
    Subtract,
    ReverseSubtract,
    Divide,
    ReverseDivide,
    POW,
)
from std.sys import simd_width_of


struct CpuArithmeticOps[dtype: DType](
    ImplicitlyCopyable & Movable & Equatable & Writable
):
    # ------------------------------------------------------------------
    # Public entry point: choose scalar or ND path
    # ------------------------------------------------------------------
    # A buffer is treated as "scalar‑like" when its shape has rank ≤ 1
    # and it holds exactly one element.  This includes true 0‑d scalars
    # (Shape()) and 1‑d tensors with shape (1,).

    @staticmethod
    @always_inline
    def broadcast[
        op_code: Int,
    ](
        a: NDBuffer[Self.dtype],
        b: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> NDBuffer[Self.dtype]:
        var a_is_scalar = a.shape.rank() <= 1 and a.numels() == 1
        var b_is_scalar = b.shape.rank() <= 1 and b.numels() == 1
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
                        ](a, b, a_is_scalar, epsilon)
                    else:
                        return CpuArithmeticOps.broadcast_scalar[ReverseDivide](
                            a, b, a_is_scalar, epsilon
                        )

            return CpuArithmeticOps.broadcast_scalar[op_code](
                a, b, a_is_scalar, epsilon
            )
        else:
            return CpuArithmeticOps.broadcast_nd[op_code](a, b, epsilon)

    # ------------------------------------------------------------------
    # Scalar path  —  one operand is scalar‑like
    # ------------------------------------------------------------------
    # When the scalar side is contiguous we can use the SIMD‑vectorised
    # arithmetic_ops_scalar on a Buffer slice, avoiding a per‑element
    # loop.  When non‑contiguous we fall back to index‑iterator walk.

    @staticmethod
    @always_inline
    def broadcast_scalar[
        op_code: Int
    ](
        a: NDBuffer[Self.dtype],
        b: NDBuffer[Self.dtype],
        a_is_scalar: Bool,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> NDBuffer[Self.dtype]:
        var result_shape = ShapeBroadcaster.broadcast_shape(a.shape, b.shape)
        var is_contiguous = (
            b.is_contiguous() if a_is_scalar else a.is_contiguous()
        )
        var item = a.item() if a_is_scalar else b.item()
        var buffer: Buffer[Self.dtype]
        if is_contiguous:
            # Fast contiguous path: broadcast the SIMD‑vectorised scalar op
            # directly on the non‑scalar operand's data range via
            # arithmetic_ops_scalar(start, end).  This reads the range in
            # one SIMD pass and writes a fresh contiguous result Buffer —
            # avoids an unnecessary alloc+memcpy that the old
            # .copied().arithmetic_ops_scalar() incurred.
            var offset = b.offset if a_is_scalar else a.offset
            var numels = b.numels() if a_is_scalar else a.numels()
            buffer = b.buffer.arithmetic_ops_scalar[op_code](
                item, offset, offset + numels
            ) if a_is_scalar else a.buffer.arithmetic_ops_scalar[op_code](
                item, offset, offset + numels
            )

        else:
            # Slow non‑contiguous path: walk the non‑scalar operand's
            # logical elements via its index_iterator and broadcast the
            # scalar function element‑by‑element.  ReverseSubtract and
            # ReverseDivide swap the argument order.
            buffer = Buffer[Self.dtype](result_shape.num_elements())
            var index = 0

            if a_is_scalar:
                for idx in b.index_iterator():
                    comptime if op_code == ReverseSubtract or op_code == ReverseDivide:
                        buffer[index] = scalar_op[op_code, Self.dtype](
                            b.buffer[idx], item
                        )
                    else:
                        buffer[index] = scalar_op[op_code, Self.dtype](
                            item, b.buffer[idx]
                        )
                    index += 1
            else:
                for idx in a.index_iterator():
                    buffer[index] = scalar_op[op_code, Self.dtype](
                        a.buffer[idx], item
                    )
                    index += 1

        return NDBuffer[Self.dtype](buffer^, result_shape)

    # ------------------------------------------------------------------
    # ND broadcast path  —  both operands have rank ≥ 1
    # ------------------------------------------------------------------
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
        a: NDBuffer[Self.dtype],
        b: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> NDBuffer[Self.dtype]:
        var result_shape = ShapeBroadcaster.broadcast_shape(a.shape, b.shape)

        # ---- 1. Compute effective strides  --------------------------
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

        # ---- 2. Allocate output  ------------------------------------
        var buffer = Buffer[Self.dtype](result_shape.num_elements())
        var total = buffer.size

        # SIMD width: 1 for bool (stored as uint8, no SIMD benefit),
        # hardware native for everything else.
        comptime simd_width = simd_width_of[
            Self.dtype
        ]() if Self.dtype != DType.bool else 1

        # ================================================================
        # TIER 1  —  both operands have unit stride in the last dimension
        # ================================================================
        # The last dimension is dense in both buffers.  Tiling it with
        # SIMD loads/stores gives maximum throughput.
        #
        # Outer dimensions are iterated with an odometer that advances
        # a_off and b_off using effective strides — stride‑0 dimensions
        # stay put, stride‑1 dimensions advance naturally.

        if (
            simd_width > 1
            and rank >= 1
            and a_eff[rank - 1] == 1
            and b_eff[rank - 1] == 1
        ):
            var last_dim = result_shape[rank - 1]
            var outer_rank = rank - 1
            var outer_count = total // last_dim

            var outer_coords = IntArray.filled(outer_rank, 0)
            var a_off = a.offset
            var b_off = b.offset

            for outer_idx in range(outer_count):
                var out_base = outer_idx * last_dim

                # --- SIMD tile the last dimension ---
                var j = 0
                while j + simd_width <= last_dim:
                    var a_v = a.buffer.load[simdwidth=simd_width](a_off + j)
                    var b_v = b.buffer.load[simdwidth=simd_width](b_off + j)
                    var op_result: SIMD[Self.dtype, simd_width]

                    # --- Path 3b: SIMD vector op via shared helper ---
                    op_result = simd_op[op_code, Self.dtype, simd_width](
                        a_v, b_v, epsilon
                    )

                    buffer.store[simdwidth=simd_width](out_base + j, op_result)
                    j += simd_width

                # Scalar remainder (last_dim not a multiple of simd_width)
                for k in range(j, last_dim):
                    buffer[out_base + k] = scalar_op[op_code, Self.dtype](
                        a.buffer[a_off + k],
                        b.buffer[b_off + k],
                        epsilon,
                    )

                # --- Advance outer dims (odometer) ---
                if outer_idx + 1 < outer_count:
                    for d in range(outer_rank - 1, -1, -1):
                        outer_coords[d] += 1
                        if outer_coords[d] < result_shape[d]:
                            a_off += a_eff[d]
                            b_off += b_eff[d]
                            break
                        else:
                            # Carry: subtract what we added for this dim
                            # and reset coordinate to 0
                            a_off -= (result_shape[d] - 1) * a_eff[d]
                            b_off -= (result_shape[d] - 1) * b_eff[d]
                            outer_coords[d] = 0

        # ================================================================
        # TIER 2  —  one operand broadcasts in the last dim, the other
        #            has unit stride
        # ================================================================
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

            var outer_coords = IntArray.filled(outer_rank, 0)
            var a_off = a.offset
            var b_off = b.offset

            for outer_idx in range(outer_count):
                var out_base = outer_idx * last_dim

                # Read the scalar that will be broadcast across this row
                var scalar_v = a.buffer[
                    a_off
                ] if a_broadcasts_last else b.buffer[b_off]

                var scalar_vec = SIMD[Self.dtype, simd_width](scalar_v)
                var j = 0
                while j + simd_width <= last_dim:
                    # SIMD load from the *non‑broadcasting* side
                    var vec = b.buffer.load[simdwidth=simd_width](
                        b_off + j
                    ) if a_broadcasts_last else a.buffer.load[
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
                        a.buffer[a_i],
                        b.buffer[b_i],
                        epsilon,
                    )

                # Advance outer dims (same odometer as Tier 1)
                if outer_idx + 1 < outer_count:
                    for d in range(outer_rank - 1, -1, -1):
                        outer_coords[d] += 1
                        if outer_coords[d] < result_shape[d]:
                            a_off += a_eff[d]
                            b_off += b_eff[d]
                            break
                        else:
                            a_off -= (result_shape[d] - 1) * a_eff[d]
                            b_off -= (result_shape[d] - 1) * b_eff[d]
                            outer_coords[d] = 0

        # ================================================================
        # TIER 3  —  general scalar odometer
        # ================================================================
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
            var a_off = a.offset
            var b_off = b.offset
            var coords = IntArray.filled(rank, 0)

            for i in range(total):
                buffer[i] = scalar_op[op_code, Self.dtype](
                    a.buffer[a_off], b.buffer[b_off], epsilon
                )

                # Odometer: least‑significant dimension (index rank‑1)
                # ticks fastest, carries propagate left.
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

        return NDBuffer[Self.dtype](buffer^, result_shape)

    @staticmethod
    @always_inline
    def compute[
        op_code: Int,
    ](
        self: NDBuffer[Self.dtype],
        other: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> NDBuffer[Self.dtype]:
        # Handle broadcasting case
        if self.shape != other.shape:
            return Self.broadcast[op_code](self, other, epsilon)
        # Same shape
        if self.is_contiguous() and other.is_contiguous():
            self_start = self.offset
            self_end = self_start + self.numels()
            other_start = other.offset
            other_end = other_start + other.numels()
            var result_buffer = self.buffer.arithmetic_ops[op_code=op_code](
                other.buffer,
                self_start,
                self_end,
                other_start,
                other_end,
                epsilon=epsilon,
            )
            return NDBuffer[Self.dtype](result_buffer^, self.shape)

        else:
            var result_buffer = Buffer[Self.dtype](self.numels())
            var index = 0

            if self.is_contiguous() and not other.is_contiguous():
                var offset = self.offset
                for idx in other.index_iterator():
                    result_buffer[index] = scalar_op[op_code, Self.dtype](
                        self.buffer[offset + index], other.buffer[idx], epsilon
                    )
                    index += 1

            elif not self.is_contiguous() and other.is_contiguous():
                var offset = other.offset
                for idx in self.index_iterator():
                    result_buffer[index] = scalar_op[op_code, Self.dtype](
                        self.buffer[idx], other.buffer[offset + index], epsilon
                    )
                    index += 1

            else:
                var iterator = other.index_iterator()
                for idx in self.index_iterator():
                    var next_index = -1
                    try:
                        next_index = iterator.__next__()
                    except e:
                        print(e)
                        panic(
                            "Raised StopIteration in NDBuffer → arithmetic_ops"
                        )

                    result_buffer[index] = scalar_op[op_code, Self.dtype](
                        self.buffer[idx], other.buffer[next_index], epsilon
                    )
                    index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.shape)

    @staticmethod
    @always_inline
    def compute[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]) -> NDBuffer[
        Self.dtype
    ]:
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            var result_buffer: Buffer[Self.dtype]

            comptime if op_code == POW:
                result_buffer = self.buffer[start:end] ** scalar
            else:
                result_buffer = self.buffer.arithmetic_ops_scalar[op_code](
                    scalar, start, end
                )
            return NDBuffer[Self.dtype](result_buffer^, self.shape)

        else:
            var index = 0
            var result_buffer = Buffer[Self.dtype](self.numels())

            for idx in self.index_iterator():
                result_buffer[index] = scalar_op[op_code, Self.dtype](
                    self.buffer[idx], scalar, epsilon
                )
                index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.shape)

    @staticmethod
    @always_inline
    def unary_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype]) -> NDBuffer[Self.dtype]:
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            var result_buffer = self.buffer.unary_ops[op_code](start, end)

            return NDBuffer[Self.dtype](result_buffer^, self.shape)

        else:
            var index = 0
            var result_buffer = Buffer[Self.dtype](self.numels())

            for idx in self.index_iterator():
                result_buffer[index] = unary_op[op_code, Self.dtype](
                    self.buffer[idx]
                )
                index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.shape)

    @staticmethod
    @always_inline
    def unary_ops_constrained[
        op_code: Int, epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value()
    ](self: NDBuffer[Self.dtype]) -> NDBuffer[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        if self.is_contiguous():
            var start = self.offset
            var end = start + self.numels()
            var result_buffer = self.buffer.float_unary_ops[op_code, epsilon](
                start, end
            )
            return NDBuffer[Self.dtype](result_buffer^, self.shape)
        else:
            var index = 0
            var result_buffer = Buffer[Self.dtype](self.numels())

            for idx in self.index_iterator():
                result_buffer[index] = float_unary_op[
                    op_code, Self.dtype, epsilon
                ](self.buffer[idx])
                index += 1

            return NDBuffer[Self.dtype](result_buffer^, self.shape)

    @staticmethod
    @always_inline
    def inplace_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], other: NDBuffer[Self.dtype]):
        _ = """# Broadcast validation
        if not ShapeBroadcaster.broadcastable(self.shape, other.shape):
            panic(
                "NDBuffer → inplace_ops: dimension mismatch: "
                + String(self.shape)
                + ", "
                + String(other.shape)
            )"""

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
            ](self, other)
            self.copy_from_alike[overwrite=True, validate=False](
                broadcast_result^
            )

        else:
            # Same shape case
            if self.is_contiguous() and other.is_contiguous():
                self_start = self.offset
                self_end = self_start + self.numels()
                other_start = other.offset
                other_end = other_start + other.numels()
                self.buffer.inplace_ops[op_code](
                    other.buffer, self_start, self_end, other_start, other_end
                )

            elif self.is_contiguous() and not other.is_contiguous():
                var index = self.offset
                for idx in other.index_iterator():
                    self.buffer[index] = scalar_op[op_code, Self.dtype](
                        self.buffer[index], other.buffer[idx]
                    )
                    index += 1

            elif not self.is_contiguous() and other.is_contiguous():
                var index = other.offset
                for idx in self.index_iterator():
                    self.buffer[idx] = scalar_op[op_code, Self.dtype](
                        self.buffer[idx], other.buffer[index]
                    )
                    index += 1
            else:
                var iterator = other.index_iterator()
                for index in self.index_iterator():
                    var next_index = -1
                    try:
                        next_index = iterator.__next__()
                    except e:
                        print(e)
                        panic("Raised StopIteration in NDBuffer → inplace_ops")

                    self.buffer[index] = scalar_op[op_code, Self.dtype](
                        self.buffer[index], other.buffer[next_index]
                    )

    @staticmethod
    @always_inline
    def inplace_scalar_ops[
        op_code: Int,
    ](self: NDBuffer[Self.dtype], scalar: Scalar[Self.dtype]):
        comptime if op_code == Divide:
            if scalar == Scalar[Self.dtype](0):
                panic("NDBuffer → inplace_scalar_ops: cannot divide by zero")

        if self.is_contiguous():
            start = self.offset
            end = start + self.numels()
            self.buffer.inplace_ops_scalar[op_code](scalar, start, end)

        else:
            for index in self.index_iterator():
                self.buffer[index] = scalar_op[op_code, Self.dtype](
                    self.buffer[index], scalar
                )
