from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite

# ===----------------------------------------------------------------------=== #
# round / floor tests — prefix: round_ / floor_
# Covers: forward (CPU), half-to-even tie semantics, leaves, contiguity,
#         contiguity matrix, dtype coverage.
#
# Expectations are HALF-TO-EVEN: Mojo's round is banker's rounding, NOT C's
# half-away-from-zero. So round(2.5)=2 and round(-2.5)=-2. A test written to
# the C convention would be the test's bug.
# ===----------------------------------------------------------------------=== #

comptime F32 = DType.float32
comptime F64 = DType.float64


# ===----------------------------------------------------------------------=== #
# CPU — round: tie semantics (the whole point)
# ===----------------------------------------------------------------------=== #


def test_round_cpu_scalar_ties() raises:
    var x = Tensor[F32].d1([2.5, 3.5, -2.5, -3.5, 0.5, -0.5])
    var y = x.round()
    var expected = Tensor[F32].d1([2.0, 4.0, -2.0, -4.0, 0.0, 0.0])
    assert_true(y.all_close(expected))


def test_round_cpu_non_ties() raises:
    var x = Tensor[F32].d1([2.4, 2.6, -2.4, -2.6, 0.4, -0.4])
    var y = x.round()
    var expected = Tensor[F32].d1([2.0, 3.0, -2.0, -3.0, 0.0, 0.0])
    assert_true(y.all_close(expected))


def test_round_cpu_integers_are_identity() raises:
    var x = Tensor[F32].d1([3.0, -3.0, 0.0, 100.0, -100.0])
    var y = x.round()
    assert_true(y.all_close(x))


def test_round_cpu_f64_ties() raises:
    var x = Tensor[F64].d1([2.5, -2.5, 3.5, -3.5])
    var y = x.round()
    var expected = Tensor[F64].d1([2.0, -2.0, 4.0, -4.0])
    assert_true(y.all_close(expected))


# ===----------------------------------------------------------------------=== #
# CPU — floor: always toward -inf
# ===----------------------------------------------------------------------=== #


def test_floor_cpu_ties_and_non_ties() raises:
    var x = Tensor[F32].d1([2.5, 3.5, -2.5, -3.5, 2.4, -2.4, 0.5, -0.5])
    var y = x.floor()
    var expected = Tensor[F32].d1([2.0, 3.0, -3.0, -4.0, 2.0, -3.0, 0.0, -1.0])
    assert_true(y.all_close(expected))


def test_floor_cpu_integers_are_identity() raises:
    var x = Tensor[F32].d1([3.0, -3.0, 0.0, 7.0, -7.0])
    var y = x.floor()
    assert_true(y.all_close(x))


def test_floor_cpu_f64() raises:
    var x = Tensor[F64].d1([2.5, -2.5, 1.9, -1.9])
    var y = x.floor()
    var expected = Tensor[F64].d1([2.0, -3.0, 1.0, -2.0])
    assert_true(y.all_close(expected))


# ===----------------------------------------------------------------------=== #
# round vs floor — the documented relationship
# ===----------------------------------------------------------------------=== #


def test_round_and_floor_agree_on_non_integers() raises:
    # round == floor exactly when both truncate the same way: positive inputs
    # with frac < 0.5, negative inputs with |frac| > 0.5. Everywhere else
    # (including all ties) they differ. Values below verified by hand.
    var agree = Tensor[F32].d1([0.1, 1.2, 2.4, -0.6, -1.7, -2.6])
    assert_true(agree.round().all_close(agree.floor()))
    var expected = Tensor[F32].d1([0.0, 1.0, 2.0, -1.0, -2.0, -3.0])
    assert_true(agree.round().all_close(expected))
    assert_true(agree.floor().all_close(expected))


def test_round_and_floor_differ_at_positive_tie() raises:
    # round(2.5) = 2 (to even), floor(2.5) = 2. They agree here...
    var t = Tensor[F32].scalar(2.5)
    assert_true(t.round().all_close(t.floor()))

    # ...and round(3.5) = 4 while floor(3.5) = 3. This is the case that
    # detects the convention: a half-up implementation would give 4 for
    # BOTH ties, which would fail the 2.5 assertion above.
    var t2 = Tensor[F32].scalar(3.5)
    assert_true(t2.round().all_close(Tensor[F32].scalar(4.0)))
    assert_true(t2.floor().all_close(Tensor[F32].scalar(3.0)))


def test_round_is_not_floor_x_plus_half() raises:
    # floor(x + 0.5) rounds half UP: it gives 3 where round gives 2.
    # This is why the STE must not use the substitution.
    var x = Tensor[F32].scalar(2.5)
    var substitution = (x + Tensor[F32].scalar(0.5)).floor()
    var true_round = x.round()
    assert_true(substitution.all_close(Tensor[F32].scalar(3.0)))
    assert_true(true_round.all_close(Tensor[F32].scalar(2.0)))
    assert_true(not substitution.all_close(true_round))


# ===----------------------------------------------------------------------=== #
# Signed zero — round(-0.5) = -0.0
# ===----------------------------------------------------------------------=== #


def test_round_negative_zero() raises:
    var x = Tensor[F32].scalar(-0.5)
    var y = x.round()
    # -0.0 == 0.0 is true, so compare by value and note the sign separately.
    assert_true(y.all_close(Tensor[F32].scalar(0.0)))
    assert_true(y.item() == Scalar[F32](0.0))


# ===----------------------------------------------------------------------=== #
# Leaves — the defining property of these ops
# ===----------------------------------------------------------------------=== #


def test_round_returns_leaf_even_from_grad_input() raises:
    var x = Tensor[F32].d1([1.4, 2.5], requires_grad=True)
    var y = x.round()
    assert_true(not y.has_ancestry())
    assert_true(not y.is_leaf())  # requires_grad is False, so not a leaf node
    assert_true(not y.requires_grad)


def test_floor_returns_leaf_even_from_grad_input() raises:
    var x = Tensor[F32].d1([1.4, 2.5], requires_grad=True)
    var y = x.floor()
    assert_true(not y.has_ancestry())
    assert_true(not y.requires_grad)


def test_round_from_grad_input_does_not_break_backward_elsewhere() raises:
    # The quantizer severs its own edge but must not poison the rest of the
    # graph: w (downstream of round) still receives gradient.
    var w = Tensor[F32].d1([2.0, 3.0], requires_grad=True)
    var x = Tensor[F32].d1([1.4, 2.5], requires_grad=True)
    var y = (x.round() * w).sum()
    y.backward()
    assert_true(w.grad().all_close[atol=1e-6](Tensor[F32].d1([1.0, 2.0])))


# ===----------------------------------------------------------------------=== #
# Shapes and strides
# ===----------------------------------------------------------------------=== #


def test_round_cpu_1d() raises:
    var x = Tensor[F32].d1([1.4, 2.5, -3.7, 0.0])
    var y = x.round()
    assert_true(y.all_close(Tensor[F32].d1([1.0, 2.0, -4.0, 0.0])))


def test_round_cpu_2d() raises:
    var x = Tensor[F32].d2([[1.4, 2.5], [-3.7, -0.5]])
    var y = x.round()
    var expected = Tensor[F32].d2([[1.0, 2.0], [-4.0, -0.0]])
    assert_true(y.all_close(expected))


def test_round_cpu_3d() raises:
    var x = Tensor[F32].d3([[[1.4, 2.5]], [[-3.7, -0.5]]])
    var y = x.round()
    var expected = Tensor[F32].d3([[[1.0, 2.0]], [[-4.0, -0.0]]])
    assert_true(y.all_close(expected))


def test_round_cpu_scalar_tensor() raises:
    var x = Tensor[F32].scalar(3.5)
    var y = x.round()
    assert_true(y.all_close(Tensor[F32].scalar(4.0)))


def test_floor_cpu_2d() raises:
    var x = Tensor[F32].d2([[1.9, -1.9], [2.5, -2.5]])
    var y = x.floor()
    var expected = Tensor[F32].d2([[1.0, -2.0], [2.0, -3.0]])
    assert_true(y.all_close(expected))


# ===----------------------------------------------------------------------=== #
# Contiguity matrix — exercises the index_iterator fallback
# ===----------------------------------------------------------------------=== #


def test_round_non_contiguous_transposed_view() raises:
    var x = Tensor[F32].d2([[1.4, 2.6], [3.5, -4.7]])
    var t = x.transpose(1, 0)
    var y = t.round()
    # After transpose the logical matrix is [[1.4, 3.5], [2.6, -4.7]]
    var expected = Tensor[F32].d2([[1.0, 4.0], [3.0, -5.0]])
    assert_true(y.all_close(expected))


def test_round_contiguous_2d() raises:
    var x = Tensor[F32].d2([[1.4, 2.6], [3.5, -4.7]])
    var y = x.round()
    var expected = Tensor[F32].d2([[1.0, 3.0], [4.0, -5.0]])
    assert_true(y.all_close(expected))


def test_round_strided_slice() raises:
    var x = Tensor[F32].d1([1.4, 9.9, 2.5, 9.9, -3.7])
    var s = x[0:5:2]  # strided view: 1.4, 2.5, -3.7
    var y = s.round()
    assert_true(y.all_close(Tensor[F32].d1([1.0, 2.0, -4.0])))


def test_floor_non_contiguous_transposed_view() raises:
    var x = Tensor[F32].d2([[1.9, 3.9], [2.5, -4.7]])
    var t = x.transpose(1, 0)
    var y = t.floor()
    var expected = Tensor[F32].d2([[1.0, 2.0], [3.0, -5.0]])
    assert_true(y.all_close(expected))


# ===----------------------------------------------------------------------=== #
# Quantization-relevant behaviour (the STE's use case)
# ===----------------------------------------------------------------------=== #


def test_round_levels_for_int8_scale() raises:
    # s = 1/127, x = 0.37 -> x/s = 46.99 -> round -> 47 -> dequant 47*s
    var s = Tensor[F32].scalar(1.0 / 127.0)
    var x = Tensor[F32].scalar(0.37)
    var level = (x / s).round()
    assert_true(level.all_close(Tensor[F32].scalar(47.0)))
    var dequant = level * s
    var expected = Tensor[F32].scalar(47.0 / 127.0)
    assert_true(dequant.all_close[atol=1e-6](expected))


def test_round_ties_reachable_with_dyadic_scale() raises:
    # A power-of-two scale makes x/s exactly a .5, which is how a tie arises in
    # practice. s = 0.5, x = 0.75 -> 1.5 -> half-to-even -> 2.
    var s = Tensor[F32].scalar(0.5)
    var x = Tensor[F32].scalar(0.75)
    var level = (x / s).round()
    assert_true(level.all_close(Tensor[F32].scalar(2.0)))


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
