from tenmo.tensor import Tensor
from std.testing import assert_true, assert_false, TestSuite
from tenmo.shapes import Shape
from tenmo.common_utils import log_warning
from tenmo.gradbox import Gradbox
from tenmo.intarray import IntArray
from std.sys import has_accelerator
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.shared import Reduction
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE
from std.math import log, exp
from std.utils.numerics import min_finite


@always_inline("nodebug")
def inf[dtype: DType]() -> Scalar[dtype]:
    """Gets a +inf value for the given dtype.

    Constraints:
        Can only be used for FP dtypes.

    Parameters:
        dtype: The value dtype.

    Returns:
        The +inf value of the given dtype.
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Only floating point dtypes support +inf."

    return Scalar[dtype](1.0) / Scalar[dtype](0.0)


def isinf[dtype: DType, //](value: Scalar[dtype]) -> Bool:
    return inf[dtype]() == value


def isnan[dtype: DType, //](value: Scalar[dtype]) -> Bool:
    return nan[dtype]() == value


@always_inline("nodebug")
def nan[dtype: DType]() -> Scalar[dtype]:
    """Gets a NaN value for the given dtype.

    Constraints:
        Can only be used for FP dtypes.

    Parameters:
        dtype: The value dtype.

    Returns:
        The NaN value of the given dtype.
    """
    comptime assert (
        dtype.is_floating_point()
    ), "Only floating point dtypes support NaN."

    return Scalar[dtype](0.0) / Scalar[dtype](0.0)


# ============================================================================
# Test Utilities
# ============================================================================


def assert_close(
    actual: Tensor[DType.float32],
    expected: Tensor[DType.float32],
    msg: String = "Assertion failed",
) raises:
    """Assert two tensors are close within tolerance."""
    comptime rtol: Float32 = 1e-5
    comptime atol: Float32 = 1e-5

    if not actual.all_close[rtol=rtol, atol=atol](expected):
        print("Expected:")
        print()
        expected.print()
        print()
        print("Actual:")
        actual.print()
        print()
        print("Difference: (actual - expected)")
        (actual - expected).print()
        log_warning(msg)


def assert_close(
    actual: Gradbox[DType.float32],
    expected: Tensor[DType.float32],
    msg: String = "Assertion failed",
) raises:
    """Assert two tensors are close within tolerance."""
    comptime rtol: Float32 = 1e-5
    comptime atol: Float32 = 1e-5

    if not actual.all_close[rtol=rtol, atol=atol](expected):
        print("Expected:")
        print()
        expected.print()
        print()
        print("Actual:")
        actual.print()
        print()
        print("Difference: (actual - expected)")
        (actual - expected).print()
        log_warning(msg)


# ============================================================================
# Basic Functionality Tests
# ============================================================================


# ============================================================================
# Reduction Type Tests
# ============================================================================


# ============================================================================
# Ignore Index Tests
# ============================================================================


# ============================================================================
# Label Smoothing Tests
# ============================================================================


# ============================================================================
# Spatial Dimension Tests
# ============================================================================


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


# ============================================================================
# Gradient Correctness Tests
# ============================================================================


# ============================================================================
# Equivalence Tests
# ============================================================================


# ============================================================================
# Performance/Optimization Tests
# ============================================================================


def assert_approx_equal(
    actual: Float32,
    expected: Float32,
    rel_tol: Float32 = 0.2,
    abs_tol: Float32 = 1e-6,
) -> Bool:
    """Check if two values are approximately equal with relative tolerance."""
    var diff = abs(actual - expected)
    var max_val = max(abs(actual), abs(expected))

    if max_val < abs_tol:
        return diff < abs_tol
    else:
        return diff / max_val < rel_tol


def _ce_gradients_basic_uu() raises:
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]], requires_grad=True).float()
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    loss.backward()

    # Gradients should exist and be non-zero
    assert_true(logits.grad().sum().item() != 0.0)
    # Specific gradient pattern check
    var softmax = logits.softmax(axes=IntArray(1))
    assert_true(abs(logits.grad()[0, 0] - (softmax[0, 0] - 1.0)) < 1e-6)


# Negative validation tests (should not panic if validation is correct)
def _ce_validation_wrong_target_dims() raises:
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor[DEFAULT_INDEX_DTYPE].d2(
        [[0, 1], [1, 0]]
    )  # Wrong: should be 1D

    # This should be caught by validation before any computation
    # (Test framework should handle the panic)
    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    _loss = loss_fn(logits, target)


def _ce_validation_class_out_of_bounds() raises:
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()  # 2 classes
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1(
        [0, 2]
    )  # Class 2 is out of bounds"""

    # Should be caught by validation

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    _loss = loss_fn(logits, target)


def _ce_validation_spatial_mismatch() raises:
    var logits = Tensor.d4([[[[2.0, 1.0], [1.0, 2.0]]]]).float()  # (1, 2, 2, 2)
    var target = Tensor[DEFAULT_INDEX_DTYPE].d3(
        [[[0, 1, 0]]]
    )  # Wrong spatial dim: (1, 1, 3)

    # Should be caught by spatial dimension validation
    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    _loss = loss_fn(logits, target)


def _ce_validation_batch_size_mismatch() raises:
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()  # batch size 2
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0, 1, 0])  # batch size 3

    # Should be caught by batch size validation

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    _loss = loss_fn(logits, target)


# ═════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═════════════════════════════════════════════════════════════════════════════


def softmax_1d(
    logits: Tensor[DType.float32], row: Int, C: Int
) -> List[Float32]:
    """Compute softmax for a single row manually."""
    var max_val = logits[[row, 0]]
    for c in range(1, C):
        if logits[[row, c]] > max_val:
            max_val = logits[[row, c]]
    var exp_sum = Float32(0)
    var exps = List[Float32]()
    for c in range(C):
        var e = exp(logits[[row, c]] - max_val)
        exps.append(e)
        exp_sum += e
    var result = List[Float32]()
    for c in range(C):
        result.append(exps[c] / exp_sum)
    return result^


def log_softmax_val(
    logits: Tensor[DType.float32], row: Int, cls: Int, C: Int
) -> Float32:
    """Compute log_softmax[row, cls] manually."""
    var probs = softmax_1d(logits, row, C)
    return log(probs[cls])


def nll_loss(log_prob: Float32) -> Float32:
    return -log_prob


def allclose(a: Float32, b: Float32, atol: Float32 = 1e-4) -> Bool:
    return abs(a - b) < atol


# ═════════════════════════════════════════════════════════════════════════════
# GROUP A: Basic Forward — Class Indices, 2D, CPU
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GROUP B: Ignore Index — CPU
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GROUP C: Label Smoothing — CPU
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GROUP D: Multi-dimensional — CPU
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GROUP E: Probability Targets — CPU
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GROUP F: Backward — Class Indices, CPU
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GROUP G: Backward — Probability Targets, CPU
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GROUP H: Train/Eval mode
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# GROUP I: GPU Forward — Class Indices
# ═════════════════════════════════════════════════════════════════════════════


def test_ce_gpu_ci_basic_mean() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = (
            Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]]).to_gpu()
        )
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0, 1])
        var target_gpu = Tensor[DEFAULT_INDEX_DTYPE].d1([0, 1]).to_gpu()
        var ce = CrossEntropyLoss[dtype](reduction="mean")
        var loss = ce(logits, target_gpu)
        var logits_cpu = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]])
        var ce2 = CrossEntropyLoss[dtype](reduction="mean")
        var loss_cpu = ce2(logits_cpu, target)
        assert_true(allclose(loss.item(), loss_cpu.item()))


def test_ce_gpu_ci_basic_sum() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = (
            Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]]).to_gpu()
        )
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0, 1]).to_gpu()
        var target_cpu = Tensor[DEFAULT_INDEX_DTYPE].d1([0, 1])
        var ce = CrossEntropyLoss[dtype](reduction="sum")
        var loss = ce(logits, target)
        var logits_cpu = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]])
        var loss_cpu = CrossEntropyLoss[dtype](reduction="sum")(
            logits_cpu, target_cpu
        )
        assert_true(allclose(loss.item(), loss_cpu.item()))


def test_ce_gpu_ci_ignore_index() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = (
            Tensor[dtype]
            .d2([[2.0, 1.0, 0.5], [1.0, 2.0, 0.1], [3.0, 1.0, 0.2]])
            .to_gpu()
        )
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0, -100, 2]).to_gpu()
        var target_cpu = Tensor[DEFAULT_INDEX_DTYPE].d1([0, -100, 2])
        var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
        var loss = ce(logits, target)
        var logits_cpu = Tensor[dtype].d2(
            [[2.0, 1.0, 0.5], [1.0, 2.0, 0.1], [3.0, 1.0, 0.2]]
        )
        var loss_cpu = CrossEntropyLoss[dtype](
            ignore_index=-100, reduction="mean"
        )(logits_cpu, target_cpu)
        assert_true(allclose(loss.item(), loss_cpu.item()))


def test_ce_gpu_ci_label_smoothing() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5]]).to_gpu()
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0])

        var ce = CrossEntropyLoss[dtype](
            reduction="mean", label_smoothing=Scalar[dtype](0.1)
        )
        var loss = ce(logits, target.to_gpu())
        var logits_cpu = Tensor[dtype].d2([[2.0, 1.0, 0.5]])
        var loss_cpu = CrossEntropyLoss[dtype](
            reduction="mean", label_smoothing=Scalar[dtype](0.1)
        )(logits_cpu, target)
        assert_true(allclose(loss.item(), loss_cpu.item()))


def test_ce_gpu_ci_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32

        # Define data with correct shape (N, C, H)
        var logits = Tensor[dtype].d3(
            [
                # Batch 0
                [
                    [2.0, 1.0],  # Class 0: spatial positions 0, 1
                    [1.0, 2.0],  # Class 1: spatial positions 0, 1
                    [0.5, 1.5],  # Class 2: spatial positions 0, 1
                ],
                # Batch 1
                [
                    [1.5, 2.0],  # Class 0: spatial positions 0, 1
                    [2.0, 1.0],  # Class 1: spatial positions 0, 1
                    [1.0, 2.0],  # Class 2: spatial positions 0, 1
                ],
            ],
            requires_grad=True,
        )

        var target = Tensor[DEFAULT_INDEX_DTYPE].d2(
            [
                [0, 2],  # Batch 0: spatial0→class0, spatial1→class2
                [1, 0],
            ]  # Batch 1: spatial0→class1, spatial1→class0
        )

        # GPU forward
        var logits_gpu = logits.to_gpu()
        var target_gpu = target.to_gpu()
        var ce = CrossEntropyLoss[dtype](reduction="mean")
        var loss_gpu = ce(logits_gpu, target_gpu)

        # CPU forward for comparison
        var loss_cpu = ce(
            logits, target
        )  # Use same criterion instance or create new

        # Compare results
        assert_true(allclose(loss_gpu.to_cpu().item(), loss_cpu.item(), 1e-5))

        # Optional: Test backward
        loss_gpu.backward()
        loss_cpu.backward()

        # Compare gradients
        assert_true(logits.grad().all_close(2 * logits_gpu.grad().to_cpu()))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP J: GPU Backward — Class Indices
# ═════════════════════════════════════════════════════════════════════════════


def test_ce_gpu_bwd_ci_grad_shape() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0, 1])
        var ce = CrossEntropyLoss[dtype](reduction="mean")
        var loss = ce(a_gpu, target.to_gpu())
        loss.backward()
        assert_true(a.grad().shape() == Shape(2, 3))


def test_ce_gpu_bwd_ci_parity() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True
        )
        var a_gpu = (
            Tensor[dtype]
            .d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True)
            .to_gpu()
        )
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0, 1])
        var ce = CrossEntropyLoss[dtype](reduction="mean")
        var loss_cpu = ce(a_cpu, target)
        loss_cpu.backward()
        var loss_gpu = ce(a_gpu, target.to_gpu())
        loss_gpu.backward()
        for i in range(2):
            for c in range(3):
                assert_true(
                    allclose(
                        a_cpu.grad()[[i, c]], a_gpu.grad()[[i, c]], atol=1e-4
                    )
                )


def test_ce_gpu_bwd_ci_ignore_zeros_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[2.0, 1.0, 0.5], [1.0, 2.0, 0.1], [3.0, 1.0, 0.2]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0, -100, 2])
        var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
        var loss = ce(a_gpu, target.to_gpu())
        loss.backward()
        # Ignored row should have zero gradient
        for c in range(3):
            assert_true(abs(a.grad()[[1, c]]) < Float32(1e-6))


def test_ce_gpu_bwd_ci_3d_ignore() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [
                [[2.0, 1.0, 0.5], [1.0, 2.0, 1.5], [0.5, 0.5, 2.0]],
                [[1.5, 2.0, 1.0], [2.0, 1.0, 2.0], [0.5, 0.5, 0.5]],
            ],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var target = Tensor[DEFAULT_INDEX_DTYPE].d2(
            [[0, -100, 2], [1, 0, -100]]
        )
        var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
        var loss = ce(a_gpu, target.to_gpu())
        loss.backward()
        # Batch 0, spatial pos 1 ignored
        for c in range(3):
            assert_true(abs(a.grad()[[0, c, 1]]) < Float32(1e-6))
        # Batch 1, spatial pos 2 ignored
        for c in range(3):
            assert_true(abs(a.grad()[[1, c, 2]]) < Float32(1e-6))


def test_ce_gpu_bwd_ci_label_smoothing() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True)
        var a_gpu = (
            Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True).to_gpu()
        )
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0])
        var ce = CrossEntropyLoss[dtype](
            reduction="mean", label_smoothing=Scalar[dtype](0.1)
        )
        var loss_cpu = ce(a_cpu, target)
        loss_cpu.backward()
        var loss_gpu = ce(a_gpu, target.to_gpu())
        loss_gpu.backward()
        for c in range(3):
            assert_true(
                allclose(a_cpu.grad()[[0, c]], a_gpu.grad()[[0, c]], atol=1e-4)
            )


# ═════════════════════════════════════════════════════════════════════════════
# GROUP K: GPU — Probability Targets
# ═════════════════════════════════════════════════════════════════════════════


def test_ce_gpu_prob_forward_parity() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var logits_cpu = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]])
        var logits_gpu = logits_cpu.to_gpu()
        var target = Tensor[dtype].d2([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
        var ce = CrossEntropyLoss[dtype](reduction="mean")
        var loss_cpu = ce(logits_cpu, target)
        var loss_gpu = ce(logits_gpu, target.to_gpu())
        assert_true(allclose(loss_cpu.item(), loss_gpu.item()))


def test_ce_gpu_prob_backward_parity() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True
        )
        var a_gpu = (
            Tensor[dtype]
            .d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True)
            .to_gpu()
        )
        var target = Tensor[dtype].d2([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
        var ce = CrossEntropyLoss[dtype](reduction="mean")
        var loss_cpu = ce(a_cpu, target)
        loss_cpu.backward()
        var loss_gpu = ce(a_gpu, target.to_gpu())
        loss_gpu.backward()
        for i in range(2):
            for c in range(3):
                assert_true(
                    allclose(
                        a_cpu.grad()[[i, c]], a_gpu.grad()[[i, c]], atol=1e-4
                    )
                )


def test_ce_gpu_prob_label_smoothing_parity() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True)
        var a_gpu = (
            Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True).to_gpu()
        )
        var target = Tensor[dtype].d2([[0.6, 0.3, 0.1]])
        var ce = CrossEntropyLoss[dtype](
            reduction="mean", label_smoothing=Scalar[dtype](0.1)
        )
        var loss_cpu = ce(a_cpu, target)
        loss_cpu.backward()
        var loss_gpu = ce(a_gpu, target.to_gpu())
        loss_gpu.backward()
        for c in range(3):
            assert_true(
                allclose(a_cpu.grad()[[0, c]], a_gpu.grad()[[0, c]], atol=1e-4)
            )


# ═════════════════════════════════════════════════════════════════════════════
# GROUP L: Edge Cases and Robustness
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
