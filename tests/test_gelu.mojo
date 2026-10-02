from tenmo.tensor import Tensor
from tenmo.gelu import GeLU
from tenmo.shared.buffers import Buffer
from std.sys import has_accelerator
from std.testing import assert_true, assert_equal, TestSuite
from tenmo.shared.shapes import Shape


# NOTE: expected values below are the GeLU tanh-approximation
#   y  = 0.5*x*(1 + tanh(k0*(x + c*x^3))), k0 = sqrt(2/pi), c = 0.044715
#   dy = 0.5*(1+t) + 0.5*x*(1-t^2)*k0*(1+3*c*x^2)
# computed independently in float64 and rounded to 8 dp — NOT a 0/1 mask
# like ReLU. Kept as separate constants (rather than re-deriving inline)
# so a broken kernel can't accidentally validate itself.


def test_gelu_basic() raises:
    comptime dtype = DType.float32

    var t = Tensor[dtype].d1([-1.0, 0.0, 1.0, 2.0])
    t.requires_grad_(True)
    var out = GeLU[dtype].forward[True](t)
    var s = out.sum()
    s.backward()

    assert_true(
        out.all_close(
            Tensor[dtype].d1(
                [-0.15880801, 0.0, 0.84119199, 1.95459769]
            )
        )
    )
    assert_true(
        t.grad().all_close(
            Tensor[dtype].d1(
                [-0.08296408, 0.5, 1.08296408, 1.08609926]
            )
        )
    )


def test_gelu_multidim() raises:
    comptime dtype = DType.float32

    # 2×3 input tensor
    var t = Tensor[dtype].d2(
        [
            [-1.0, 2.0, 0.0],
            [3.0, -4.0, 5.0],
        ]
    )
    t.requires_grad_(True)

    # Apply GeLU
    var out = t.gelu()
    assert_true(
        out.all_close(
            Tensor[dtype].d2(
                [
                    [-0.15880801, 1.95459769, 0.0],
                    [2.99636261, -0.00007025, 4.99999977],
                ]
            )
        )
    )

    # Backward on sum of outputs
    var s = out.sum()
    s.backward()

    var expected_grad = Tensor[dtype].d2(
        [
            [-0.08296408, 1.08609926, 0.5],
            [1.01158417, -0.00033512, 1.00000155],
        ]
    )
    assert_true(t.grad().all_close(expected_grad))


def test_gelu_cpu_fwd_4d() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].full([2, 2, 2, 2], 3.0)
    var out = a.gelu()
    assert_true(out.all_close(Tensor[dtype].full([2, 2, 2, 2], 2.99636261)))


def test_gelu_cpu_fwd_scalar() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(-5.0)
    var out = a.gelu()
    assert_true(out.all_close(Tensor[dtype].scalar(-0.00000023)))


# =============================================================================
# ── SECTION 2: CPU BACKWARD (derivative correctness) ─────────────────────────
# =============================================================================


def test_gelu_cpu_bwd_all_positive() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var out = a.gelu()
    var loss = out.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d1([1.08296408, 1.08609926, 1.01158417])
        )
    )


def test_gelu_cpu_bwd_all_negative() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-1.0, -2.0, -3.0], requires_grad=True)
    var out = a.gelu()
    var loss = out.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d1([-0.08296408, -0.08609926, -0.01158417])
        )
    )


# =============================================================================
# ── SECTION 3: CPU GRAD FLOW ─────────────────────────────────────────────────
# =============================================================================


def test_gelu_cpu_grad_chain_gelu_mul() raises:
    comptime dtype = DType.float32
    # y = gelu(a) * 2  →  dy/da = 2 * dgelu(a)
    var a = Tensor[dtype].d1([-1.0, 2.0, 3.0], requires_grad=True)
    var r = a.gelu()
    var out = r * Tensor[dtype].full([3], 2.0)
    var loss = out.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d1([-0.16592816, 2.17219852, 2.02316834])
        )
    )


# =============================================================================
# ── SECTION 4: CPU NON-CONTIGUOUS ────────────────────────────────────────────
# =============================================================================


def test_gelu_cpu_noncontig_transposed() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    var t = a.transpose()  # non-contiguous view
    var out = t.gelu()
    var loss = out.sum()
    loss.backward()
    # transpose forward: gelu([[-1,3],[2,-4]])
    assert_true(
        out.contiguous().all_close(
            Tensor[dtype].d2(
                [[-0.15880801, 2.99636261], [1.95459769, -0.00007025]]
            )
        )
    )


def test_gelu_cpu_noncontig_slice_bwd() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-2.0, 1.0, -1.0, 3.0, -5.0], requires_grad=True)
    # Slice [1:4] → [1.0, -1.0, 3.0]
    var s = a.slice(1, 4)
    var out = s.gelu()
    var loss = out.sum()
    loss.backward()
    assert_true(
        out.all_close(
            Tensor[dtype].d1([0.84119199, -0.15880801, 2.99636261])
        )
    )


# =============================================================================
# ── SECTION 5: GPU FORWARD ───────────────────────────────────────────────────
# =============================================================================

def test_gelu_gpu_fwd_large() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # Exercises multi-block dispatch in the kernel
        var a = Tensor[dtype].full([131072], 2.0).to_gpu()
        var out = a.gelu()
        assert_true(
            out.to_cpu().all_close(Tensor[dtype].full([131072], 1.95459769))
        )

def test_gelu_gpu_fwd_dtype_float64() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var a = Tensor[dtype].d1([-1.0, 0.0, 1.0]).to_gpu()
        var out = a.gelu()
        assert_true(
            out.to_cpu().all_close(
                Tensor[dtype].d1([-0.15880801, 0.0, 0.84119199])
            )
        )


# =============================================================================
# ── SECTION 6: GPU BACKWARD (derivative buffer stays on device) ─────────────
# =============================================================================


def test_gelu_gpu_bwd_all_positive() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var out = a.gelu()
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(
                Tensor[dtype].d1([1.08296408, 1.08609926, 1.01158417])
            )
        )


def test_gelu_gpu_bwd_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[-1.0, 2.0], [3.0, -4.0]], requires_grad=True
        )
        var a = a_cpu.to_gpu()
        var out = a.gelu()
        var loss = out.sum()
        loss.backward()
        assert_true(
            a_cpu.grad().all_close(
                Tensor[dtype].d2(
                    [[-0.08296408, 1.08609926], [1.01158417, -0.00033512]]
                )
            )
        )


# =============================================================================
# ── SECTION 7: GPU GRAD FLOW ─────────────────────────────────────────────────
# =============================================================================

def test_gelu_gpu_grad_chain_gelu_gelu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, 2.0, 3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var r1 = a.gelu()
        var r2 = r1.gelu()
        var loss = r2.sum()
        loss.backward()
        # chain rule: dL/da = gelu'(gelu(a)) * gelu'(a)
        assert_true(
            a_cpu.grad().all_close(
                Tensor[dtype].d1([-0.03105793, 1.18491713, 1.02341988])
            )
        )


def test_gelu_gpu_grad_chain_add_gelu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([-1.0, 1.0], requires_grad=True)
        var b_cpu = Tensor[dtype].d1([2.0, -3.0], requires_grad=True)
        var a = a_cpu.to_gpu()
        var b = b_cpu.to_gpu()
        var out = (a + b).gelu()
        var loss = out.sum()
        loss.backward()
        # a+b = [1.0, -2.0] → dgelu = [1.08296408, -0.08609926]
        assert_true(
            a_cpu.grad().all_close(
                Tensor[dtype].d1([1.08296408, -0.08609926])
            )
        )
        assert_true(
            b_cpu.grad().all_close(
                Tensor[dtype].d1([1.08296408, -0.08609926])
            )
        )

# =============================================================================
# ── SECTION 9: CPU / GPU PARITY ──────────────────────────────────────────────
# =============================================================================


def test_gelu_parity_fwd_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var data = Tensor[dtype].d1([-3.0, -1.0, 0.0, 1.0, 3.0])
        var cpu_out = data.gelu()
        var gpu_out = data.to_gpu().gelu().to_cpu()
        assert_true(cpu_out.all_close(gpu_out))


# =============================================================================
# ── BUFFER-LEVEL TESTS ──────────────────────────────────────────────────────
# =============================================================================


def test_gelu_buffer_forward_with_mask() raises:
    var buffer = Buffer[DType.float32](10)
    for i in range(10):
        buffer[i] = Float32(i - 5)  # Values from -5 to 4

    var result = GeLU[DType.float32]._buffer_forward(buffer)
    var output = result[0]
    var deriv = result[1]

    # x = -5 .. 4, in order
    var expected_output: List[Float32] = [
        -0.00000023,
        -0.00007025,
        -0.00363739,
        -0.04540231,
        -0.15880801,
        0.0,
        0.84119199,
        1.95459769,
        2.99636261,
        3.99992975,
        ]
    var expected_deriv: List[Float32] = [
        -0.00000155,
        -0.00033512,
        -0.01158417,
        -0.08609926,
        -0.08296408,
        0.5,
        1.08296408,
        1.08609926,
        1.01158417,
        1.00033512,
        ]

    var output_correct = True
    for i in range(10):
        if abs(output[i] - expected_output[i]) > 1e-5:
            output_correct = False
            break

    var deriv_correct = True
    for i in range(10):
        if abs(deriv[i] - expected_deriv[i]) > 1e-5:
            deriv_correct = False
            break

    assert_true(
        output_correct, "gelu_buffer_forward_with_mask: output incorrect"
    )
    assert_true(
        deriv_correct, "gelu_buffer_forward_with_mask: derivative incorrect"
    )



# =============================================================================
# GeLU — supplementary QA suite
#
# All tests here use the `test_gelu_qa_` prefix so they cannot collide with
# names in the existing GeLU test file.
#
# Design principle: every expected value below is either
#   (a) an EXACT closed-form result (x=0, or |x| large enough that tanh has
#       fully saturated to +/-1.0 in the given dtype — both give exact
#       0 / x / 0.0 / 1.0 results with no rounding ambiguity), or
#   (b) a constant already verified in the project's existing GeLU test file,
#       reused here in a new structural context (3D shapes, float64, chained
#       composition, non-contiguous views, cross-device parity), or
#   (c) a cross-check between two independently-computed results (CPU vs GPU,
#       float32 vs float64) rather than a hand-typed decimal.
#
# This avoids introducing any new hand-derived tanh/cubic arithmetic, which
# is exactly the kind of thing that's easy to get subtly wrong to the 8th
# decimal place and then chase as a phantom kernel bug.
#
# ASSUMPTIONS FLAGGED INLINE where an API surface is inferred rather than
# directly observed (e.g. `.requires_grad` as a field vs. method) — please
# adjust those specific tests if your actual signatures differ.
# =============================================================================


# =============================================================================
# ── SECTION A: EXACT EDGE VALUES (zero, negative zero, saturation) ──────────
# =============================================================================


def test_gelu_qa_zero_input_forward_exact() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(0.0)
    var out = a.gelu[track_grad=False]()
    # gelu(0) = 0.5*0*(1+tanh(0)) = 0 exactly
    assert_true(out.all_close(Tensor[dtype].scalar(0.0)))


def test_gelu_qa_negative_zero_input_forward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].scalar(-0.0)
    var out = a.gelu[track_grad=False]()
    assert_true(out.all_close(Tensor[dtype].scalar(0.0)))


def test_gelu_qa_zero_input_backward_exact() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([0.0], requires_grad=True)
    var out = a.gelu()
    var loss = out.sum()
    loss.backward()
    # gelu'(0) = 0.5*(1+tanh(0)) + 0.5*0*(...) = 0.5 exactly
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.5])))


def test_gelu_qa_large_positive_saturation_exact() raises:
    comptime dtype = DType.float32
    # tanh(k0*(x + c*x^3)) saturates to exactly 1.0 in float32 well before
    # these magnitudes, so gelu(x) = 0.5*x*(1+1) = x exactly, and
    # gelu'(x) = 0.5*(1+1) + 0.5*x*(1-1)*(...) = 1.0 exactly.
    var a = Tensor[dtype].d1([20.0, 30.0, 50.0], requires_grad=True)
    var out = a.gelu()
    assert_true(out.all_close(Tensor[dtype].d1([20.0, 30.0, 50.0])))
    var loss = out.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 1.0, 1.0])))


def test_gelu_qa_large_negative_saturation_exact() raises:
    comptime dtype = DType.float32
    # Symmetric case: tanh saturates to exactly -1.0, so gelu(x) = 0 exactly
    # and gelu'(x) = 0 exactly.
    var a = Tensor[dtype].d1([-20.0, -30.0, -50.0], requires_grad=True)
    var out = a.gelu()
    assert_true(out.all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))
    var loss = out.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


# =============================================================================
# ── SECTION B: SIMD REMAINDER / CHUNK BOUNDARY COVERAGE ─────────────────────
# =============================================================================


def test_gelu_qa_remainder_boundary_odd_size_cpu() raises:
    comptime dtype = DType.float32
    # 100003 is not a multiple of 4, 8, or 16 — guarantees the SIMD
    # remainder loop in _buffer_forward actually executes, regardless of
    # this machine's simd_width. Reuses the already-verified gelu(2.0)
    # constant from the existing GPU large-tensor test.
    var n = 100003
    var a = Tensor[dtype].full([n], 2.0)
    var out = a.gelu[track_grad=False]()
    assert_true(out.all_close(Tensor[dtype].full([n], 1.95459769)))


def test_gelu_qa_gpu_fwd_odd_size_remainder() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # The existing GPU large-tensor test uses 131072 = 2^17, which is
        # suspiciously block-aligned and may never exercise the GPU
        # kernel's tail/remainder handling. This deliberately awkward size
        # closes that gap.
        var n = 100003
        var a = Tensor[dtype].full([n], 2.0).to_gpu()
        var out = a.gelu()
        assert_true(out.to_cpu().all_close(Tensor[dtype].full([n], 1.95459769)))


# =============================================================================
# ── SECTION C: HIGHER-RANK / TRANSFORMER-SHAPED TENSORS ─────────────────────
# =============================================================================


def test_gelu_qa_transformer_shaped_3d_forward() raises:
    comptime dtype = DType.float32
    # (batch=2, seq=3, hidden=4) — a realistic FFN activation shape.
    var a = Tensor[dtype].full([2, 3, 4], 2.0)
    var out = a.gelu[track_grad=False]()
    assert_true(out.all_close(Tensor[dtype].full([2, 3, 4], 1.95459769)))


def test_gelu_qa_transformer_shaped_3d_backward() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].full([2, 3, 4], 2.0)
    a.requires_grad_(True)
    var out = a.gelu()
    var loss = out.sum()
    loss.backward()
    # deriv(2.0) = 1.08609926 — already verified in the existing
    # test_gelu_cpu_bwd_all_positive test, reused here at higher rank.
    assert_true(a.grad().all_close(Tensor[dtype].full([2, 3, 4], 1.08609926)))


def test_gelu_qa_noncontig_transpose_3x3_exact_saturation() raises:
    comptime dtype = DType.float32
    # Larger (3x3) non-contiguous transpose than the existing 2x2 case,
    # using only saturating values so every expected entry is exact
    # (20.0 or 0.0 forward; 1.0 or 0.0 backward) — no new tanh arithmetic
    # needed, while still exercising the strided odometer path at a
    # different shape/size than the existing coverage.
    var a = Tensor[dtype].d2(
        [
            [20.0, -20.0, 20.0],
            [-20.0, 20.0, -20.0],
            [20.0, -20.0, 20.0],
        ],
        requires_grad=True,
    )
    var t = a.transpose()
    var out = t.gelu()

    var expected_out = Tensor[dtype].d2([[20.0, 0.0, 20.0], [0.0, 20.0, 0.0], [20.0, 0.0, 20.0]])
    assert_true(out.contiguous().all_close(expected_out))

    var loss = out.sum()
    loss.backward()

    var expected_grad = Tensor[dtype].d2(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    assert_true(a.grad().all_close(expected_grad))


# =============================================================================
# ── SECTION D: DTYPE COVERAGE (float64) ──────────────────────────────────────
# =============================================================================


def test_gelu_qa_dtype_float64_fwd_reference_match() raises:
    comptime dtype = DType.float64
    # The 1.95459769 constant in the existing suite is documented as
    # "computed independently in float64 and rounded to 8 dp" — so a
    # float64 computation of gelu(2.0) should match it comfortably within
    # float64's default all_close tolerance.
    var a = Tensor[dtype].d1([2.0])
    var out = a.gelu[track_grad=False]()
    assert_true(out.all_close(Tensor[dtype].d1([1.95459769])))


def test_gelu_qa_dtype_float64_bwd_reference_match() raises:
    comptime dtype = DType.float64
    var a = Tensor[dtype].d1([2.0], requires_grad=True)
    var out = a.gelu()
    var loss = out.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([1.08609926])))


def test_gelu_qa_dtype_float64_saturation_exact() raises:
    comptime dtype = DType.float64
    # Same exact-saturation argument as the float32 case, just confirming
    # it holds for float64 too (tanh saturates even earlier there is not
    # true — float64 needs a larger |x| to fully saturate to +/-1.0 due to
    # its wider mantissa — so this uses a bigger magnitude than the
    # float32 saturation tests).
    var a = Tensor[dtype].d1([100.0, -100.0], requires_grad=True)
    var out = a.gelu()
    assert_true(out.all_close(Tensor[dtype].d1([100.0, 0.0])))
    var loss = out.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([1.0, 0.0])))


# =============================================================================
# ── SECTION E: AUTOGRAD WIRING (track_grad, requires_grad override) ─────────
# =============================================================================


def test_gelu_qa_track_grad_false_no_autograd() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, -1.0, 2.0], requires_grad=True)
    var out = a.gelu[track_grad=False]()
    # ASSUMPTION: `.requires_grad` is a directly-readable field/property on
    # Tensor (its presence in the printed repr strongly suggests this).
    # If it's actually a method, change to `out.requires_grad()`.
    assert_true(not out.requires_grad)


def test_gelu_qa_requires_grad_explicit_override_enables() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, -2.0, 3.0], requires_grad=False)
    var out = a.gelu(requires_grad=True)
    assert_true(out.requires_grad)
    # Should build ancestry and complete a backward pass without raising,
    # even though `a` itself never had requires_grad=True.
    var loss = out.sum()
    loss.backward()


def test_gelu_qa_requires_grad_explicit_override_disables() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, -2.0, 3.0], requires_grad=True)
    var out = a.gelu(requires_grad=False)
    assert_true(not out.requires_grad)


# =============================================================================
# ── SECTION F: REPEAT-BACKWARD ACCUMULATION ──────────────────────────────────
# =============================================================================


def test_gelu_qa_double_backward_accumulates_grad() raises:
    # Intermediates always clear once consumed, so each pass flows fresh
    # values: two backward() calls double (not triple) the single-call result
    # verified in test_gelu_cpu_bwd_all_negative
    # ([-0.08296408, -0.08609926, -0.01158417]).
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-1.0, -2.0, -3.0], requires_grad=True)
    var out = a.gelu()
    var loss = out.sum()

    loss.backward()
    print("after first backward:\n")
    a.grad().buffer().print()
    loss.backward()
    print("after second backward:\n")
    a.grad().buffer().print()

    # Gradients accumulate via AddTensor on every backward() call.
    var expected_doubled = 2 * Tensor[dtype].d1(
            [-0.08296408, -0.08609926, -0.01158417]
    )
    assert_true(a.grad().all_close(expected_doubled))


# =============================================================================
# ── SECTION G: CPU COMPOSITION (chain, previously GPU-only) ─────────────────
# =============================================================================


def test_gelu_qa_chain_gelu_twice_cpu() raises:
    comptime dtype = DType.float32
    # Same input and expected values as the existing GPU-only
    # test_gelu_gpu_grad_chain_gelu_gelu — proves CPU and GPU agree on
    # double composition, and gives this composition CPU coverage for the
    # first time.
    var a = Tensor[dtype].d1([-1.0, 2.0, 3.0], requires_grad=True)
    var r1 = a.gelu()
    var r2 = r1.gelu()
    var loss = r2.sum()
    loss.backward()
    assert_true(
        a.grad().all_close(
            Tensor[dtype].d1([-0.03105793, 1.18491713, 1.02341988])
        )
    )


# =============================================================================
# ── SECTION H: CROSS-DEVICE / CROSS-DTYPE PARITY ────────────────────────────
# =============================================================================


def test_gelu_qa_parity_bwd_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_gpu_src = Tensor[dtype].d1(
            [-3.0, -1.0, 0.0, 1.0, 3.0], requires_grad=True
        )
        var a_gpu = a_gpu_src.to_gpu()
        var out_gpu = a_gpu.gelu()
        var ss= out_gpu.sum()
        ss.backward()

        var a_cpu = Tensor[dtype].d1(
            [-3.0, -1.0, 0.0, 1.0, 3.0], requires_grad=True
        )
        var out_cpu = a_cpu.gelu()
        var sss = out_cpu.sum()
        sss.backward()

        assert_true(a_gpu_src.grad().all_close(a_cpu.grad()))


def test_gelu_qa_gpu_bwd_float64_matches_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var a_gpu_src = Tensor[dtype].d1([-1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a_gpu_src.to_gpu()
        var out_gpu = a_gpu.gelu()
        var ss = out_gpu.sum()
        ss.backward()

        var a_cpu = Tensor[dtype].d1([-1.0, 2.0, 3.0], requires_grad=True)
        var out_cpu = a_cpu.gelu()
        var sss = out_cpu.sum()
        sss.backward()

        assert_true(a_gpu_src.grad().all_close(a_cpu.grad()))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll gelu_qa tests passed!")
