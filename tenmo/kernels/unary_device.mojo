"""Unary GPU kernels: scalar/bool unary op bodies.

Kernel *bodies* only — they take raw `DeviceBuffer` pointers. The host-side
launch wrappers (`UnaryOpsKernel`, `unary_ops_with_mask`) stay in
`tenmo/kernels/unary_ops_kernel.mojo`, which imports these bodies from here.
"""

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of
from std.math import log2, exp2, rsqrt, round, floor

from ..shared.constants import Epsilon, LOG2E, LN2, _GELU_K0, _GELU_C, _GELU_C3
from ..shared.mnemonics import (
    LOG,
    EXP,
    SQRT,
    TANH_FORWARD,
    NEGATE,
    SIGMOID_FORWARD,
    RELU_FORWARD,
    GELU_FORWARD,
    INVERT,
    ROUND,
    FLOOR,
)

# log2(e) and ln(2): used to lower exp()/log() to the hardware lg2/ex2
# instructions (exp2(x * LOG2E), ln2 * log2(x)). std.math.exp/log on this
# nightly lower to libdevice device calls (__nv_expf/__nv_logf) that this
# toolchain's launch path never completes -> synchronize() hangs (verified on
# Tesla T4, mojo 1.0.0b3.dev2026080206).


# Invert DType.bool


def invert_bool[
    simd_width: Int = simd_width_of[DType.uint8](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[DType.uint8], MutAnyOrigin],
    A: Pointer[Scalar[DType.uint8], ImmutAnyOrigin],
    size_: Int64,
):
    """Logical NOT for bool stored as uint8. 0 -> 1, 1 -> 0."""
    var size = Int(size_)
    var gtid = thread_idx.x + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width
            if i + simd_width <= size:
                var vec_a = A.unsafe_load[width=simd_width](i)
                # logical NOT: 0->1, anything else->0
                var vec_result = (
                    vec_a.eq(SIMD[DType.uint8, simd_width](0))
                ).cast[DType.uint8]()
                result.unsafe_store[width=simd_width](i, vec_result)
            elif i < size:
                for j in range(size - i):
                    result[unsafe_offset=i + j] = UInt8(1) if A[
                        unsafe_offset=i + j
                    ] == UInt8(0) else UInt8(0)
        base_idx += stride * CHUNK_SIZE


def unary_ops[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
):
    """Generic unary ops kernel — SQRT, NEGATE, ABS, RELU.
    LOG, EXP, TANH, SIGMOID are handled by float_unary_ops.
    RELU = max(x, 0) — pure arithmetic, safe for any dtype.

    Generic unary ops kernel (SQRT, NEGATE, ABS, RELU)
    Works for any dtype — no floating point constraint needed.
    LOG, EXP, TANH, SIGMOID live in float_unary_ops below.
    """
    var size = Int(size_)
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.unsafe_load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width]

                comptime if op_code == SQRT:
                    vec_result = SIMD[dtype, simd_width](1) / rsqrt(
                        max(SIMD[dtype, simd_width](0), vec_a)
                    )
                elif op_code == NEGATE:
                    vec_result = -vec_a
                elif op_code == INVERT:
                    vec_result = vec_a.__invert__()

                elif op_code == RELU_FORWARD:
                    vec_result = max(vec_a, SIMD[dtype, simd_width](0))
                else:  # ABS
                    # max(x, -x) instead of abs(): abs() lowers to the
                    # llvm.nvvm.fabs intrinsic, which this toolchain's NVPTX
                    # backend cannot select.
                    vec_result = max(vec_a, -vec_a)

                result.unsafe_store[width=simd_width](i, vec_result)

            elif i < size:
                for j in range(size - i):
                    var val = A[unsafe_offset=i + j]
                    var res: Scalar[dtype]

                    comptime if op_code == SQRT:
                        res = Scalar[dtype](1) / rsqrt(
                            max(Scalar[dtype](0), val)
                        )
                    elif op_code == NEGATE:
                        res = -val
                    elif op_code == INVERT:
                        res = val.__invert__()
                    elif op_code == RELU_FORWARD:
                        res = max(val, Scalar[dtype](0))
                    else:  # ABS
                        res = max(val, -val)

                    result[unsafe_offset=i + j] = res

        base_idx += stride * CHUNK_SIZE


def float_unary_ops[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
    epsilon: Scalar[dtype] = Epsilon[dtype].value(),
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
) where dtype.is_floating_point():
    """Floating point unary ops kernel — LOG, EXP, TANH, SIGMOID.

    NOTE: the `where` clause alone guards the floating-point constraint.
    Do NOT re-add a `comptime if dtype.is_floating_point():` wrapper around
    this body — a where-clause + comptime-if wrapper + exp2/log2 body in an
    imported module hangs at `ctx.synchronize()` on this toolchain (Tesla T4,
    mojo 1.0.0b3.dev2026080206; verified by the copycat/where/fix probes in
    tests/gpu/). The plain form below is the proven-passing structure.

    Floating point unary ops kernel (LOG, EXP, TANH, SIGMOID)
    Single merged kernel — requires dtype.is_floating_point().
    Supported in Mojo 0.26.2+; previously crashed the compiler.
    tanh notes:
      tanh() requires PTX ISA 7.0+ — implemented via the exp2() identity:
      tanh(x) = (e^2x - 1) / (e^2x + 1)  — works on all PTX ISA versions.
    log notes:
      epsilon-clamped: log(max(x, epsilon)) to avoid log(0) = -inf.
      epsilon defaults differ by dtype:
        float32 → 1e-7  (1e-12 flushes to 0.0 in float32 — silent breakage)
        float64 → 1e-12
      Both log/exp lower to exp2(x*LOG2E) / ln2*log2(x) — see the LOG2E/LN2
      comment at the top of the file for why (libdevice exp/log hang launch).
    """
    var size = Int(size_)
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.unsafe_load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width]
                var one = SIMD[dtype, simd_width](1.0)

                comptime if op_code == LOG:
                    vec_result = (
                        log2(max(vec_a, SIMD[dtype, simd_width](epsilon))) * LN2
                    )
                elif op_code == EXP:
                    vec_result = exp2(vec_a * LOG2E)
                elif op_code == TANH_FORWARD:
                    var e2x = exp2((vec_a + vec_a) * LOG2E)
                    vec_result = (e2x - one) / (e2x + one)
                elif op_code == ROUND:
                    vec_result = round(vec_a)
                elif op_code == FLOOR:
                    vec_result = floor(vec_a)
                else:  # SIGMOID_FORWARD
                    vec_result = one / (one + exp2(-vec_a * LOG2E))

                result.unsafe_store[width=simd_width](i, vec_result)

            elif i < size:
                for j in range(size - i):
                    var x = A[unsafe_offset=i + j]
                    var res: Scalar[dtype]

                    comptime if op_code == LOG:
                        res = log2(max(x, epsilon)) * LN2
                    elif op_code == EXP:
                        res = exp2(x * LOG2E)
                    elif op_code == TANH_FORWARD:
                        var e2x = exp2((x + x) * LOG2E)
                        res = (e2x - 1.0) / (e2x + 1.0)
                    elif op_code == ROUND:
                        res = round(x)
                    elif op_code == FLOOR:
                        res = floor(x)
                    else:  # SIGMOID_FORWARD
                        res = 1.0 / (1.0 + exp2(-x * LOG2E))

                    result[unsafe_offset=i + j] = res

        base_idx += stride * CHUNK_SIZE


def float_unary_ops_with_mask[
    op_code: Int,
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    mask: Pointer[Scalar[dtype], MutAnyOrigin],
    A: Pointer[Scalar[dtype], ImmutAnyOrigin],
    size_: Int64,
) where dtype.is_floating_point():
    """Floating point unary-with-mask kernel — currently GELU only.

    Writes two output buffers in a single pass, same shape as
    unary_ops_with_mask:
        result — activated values
        mask   — gradient multiplier (a real derivative here, not a 0/1 gate)

    NOTE: mirrors float_unary_ops's structure exactly — do not add a
    `comptime if dtype.is_floating_point():` wrapper on top of the `where`
    clause; see float_unary_ops's docstring for why that combination hangs
    synchronize() on this toolchain.

    Floating point unary-with-mask ops kernel (GELU)
    Two-output sibling of float_unary_ops, mirroring how unary_ops_with_mask
    is the two-output sibling of unary_ops. Needed because GELU_FORWARD's
    tanh-approx math requires dtype.is_floating_point() (exp2 below cannot be
    proven correct for a fully-generic dtype), unlike RELU_FORWARD which is
    valid for any dtype and stays in the unconstrained unary_ops_with_mask.
    tanh notes: same as float_unary_ops — tanh(x) computed via the exp2
    identity (e^2x - 1)/(e^2x + 1), NOT std.math.tanh, which requires PTX
    ISA 7.0+. See the LOG2E/LN2 comment at the top of this file.
    GELU (tanh approximation):
      y  = 0.5*x*(1 + tanh(k0*(x + c*x^3)))
      dy/dx = 0.5*(1+t) + 0.5*x*(1-t^2)*k0*(1+3c*x^2), t = tanh(k0*(x+c*x^3))
    where k0 = sqrt(2/pi), c = 0.044715.
    """
    var size = Int(size_)
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    var zero_vec = SIMD[dtype, simd_width](0)
    var one_vec = SIMD[dtype, simd_width](1)
    var half_vec = SIMD[dtype, simd_width](0.5)
    var k0_vec = SIMD[dtype, simd_width](Scalar[dtype](_GELU_K0))
    var c_vec = SIMD[dtype, simd_width](Scalar[dtype](_GELU_C))
    var c3_vec = SIMD[dtype, simd_width](Scalar[dtype](_GELU_C3))

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a = A.unsafe_load[width=simd_width](i)
                var vec_result: SIMD[dtype, simd_width]
                var vec_mask: SIMD[dtype, simd_width]

                comptime if op_code == GELU_FORWARD:
                    var x2 = vec_a * vec_a
                    var u = k0_vec * (vec_a + c_vec * vec_a * x2)
                    var e2u = exp2((u + u) * SIMD[dtype, simd_width](LOG2E))
                    var t = (e2u - one_vec) / (e2u + one_vec)

                    vec_result = half_vec * vec_a * (one_vec + t)
                    vec_mask = half_vec * (one_vec + t) + half_vec * vec_a * (
                        one_vec - t * t
                    ) * (k0_vec * (one_vec + c3_vec * x2))
                else:
                    # Extend here for other float ops that need a mask.
                    vec_result = vec_a
                    vec_mask = one_vec

                result.unsafe_store[width=simd_width](i, vec_result)
                mask.unsafe_store[width=simd_width](i, vec_mask)

            elif i < size:
                for j in range(size - i):
                    var x = A[unsafe_offset=i + j]
                    var res: Scalar[dtype]
                    var msk: Scalar[dtype]

                    comptime if op_code == GELU_FORWARD:
                        var k0_s = Scalar[dtype](_GELU_K0)
                        var c_s = Scalar[dtype](_GELU_C)
                        var c3_s = Scalar[dtype](_GELU_C3)
                        var half_s = Scalar[dtype](0.5)
                        var one_s = Scalar[dtype](1)

                        var x2 = x * x
                        var u = k0_s * (x + c_s * x * x2)
                        var e2u = exp2((u + u) * Scalar[dtype](LOG2E))
                        var t = (e2u - one_s) / (e2u + one_s)

                        res = half_s * x * (one_s + t)
                        msk = half_s * (one_s + t) + half_s * x * (
                            one_s - t * t
                        ) * (k0_s * (one_s + c3_s * x2))
                    else:
                        res = x
                        msk = Scalar[dtype](1)

                    result[unsafe_offset=i + j] = res
                    mask[unsafe_offset=i + j] = msk

        base_idx += stride * CHUNK_SIZE
