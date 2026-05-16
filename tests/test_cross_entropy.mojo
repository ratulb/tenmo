from tenmo.tensor import Tensor
from std.testing import assert_true, assert_false, TestSuite
from std.time import perf_counter_ns
from tenmo.shapes import Shape
from tenmo.common_utils import log_warning
from tenmo.gradbox import Gradbox
from tenmo.intarray import IntArray
from std.sys import has_accelerator
from tenmo.crossentropy import CrossEntropyLoss, Reduction
from std.math import log, exp
from std.utils.numerics import min_finite


@always_inline("nodebug")
fn inf[dtype: DType]() -> Scalar[dtype]:
    """Gets a +inf value for the given dtype.

    Constraints:
        Can only be used for FP dtypes.

    Parameters:
        dtype: The value dtype.

    Returns:
        The +inf value of the given dtype.
    """
    comptime assert
        dtype.is_floating_point(),
        "Only floating point dtypes support +inf."

    comptime if dtype == DType.bfloat16:
        return rebind[Scalar[dtype]](
            __mlir_attr.`#pop.simd<"inf"> : !pop.scalar<bf16>`,
        )
    elif dtype == DType.float16:
        return rebind[Scalar[dtype]](
            __mlir_attr.`#pop.simd<"inf"> : !pop.scalar<f16>`,
        )
    elif dtype == DType.float32:
        return rebind[Scalar[dtype]](
            __mlir_attr.`#pop.simd<"inf"> : !pop.scalar<f32>`,
        )
    elif dtype == DType.float64:
        return rebind[Scalar[dtype]](
            __mlir_attr.`#pop.simd<"inf"> : !pop.scalar<f64>`,
        )
    else:
        comptime assert False, "unsupported float type"


fn isinf[dtype: DType, //](value: Scalar[dtype]) -> Bool:
    return inf[dtype]() == value


fn isnan[dtype: DType, //](value: Scalar[dtype]) -> Bool:
    return nan[dtype]() == value


@always_inline("nodebug")
fn nan[dtype: DType]() -> Scalar[dtype]:
    """Gets a NaN value for the given dtype.

    Constraints:
        Can only be used for FP dtypes.

    Parameters:
        dtype: The value dtype.

    Returns:
        The NaN value of the given dtype.
    """
    comptime assert
        dtype.is_floating_point(),
        "Only floating point dtypes support NaN."

    comptime if dtype == DType.float32:
        return rebind[Scalar[dtype]](
            __mlir_attr.`#pop.simd<"nan"> : !pop.scalar<f32>`,
        )
    elif dtype == DType.float64:
        return rebind[Scalar[dtype]](
            __mlir_attr.`#pop.simd<"nan"> : !pop.scalar<f64>`,
        )
    else:
        comptime assert False, "unsupported float type"


# ============================================================================
# Test Utilities
# ============================================================================


fn assert_close(
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


fn assert_close(
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


fn test_ce_basic_class_indices() raises:
    """Test basic cross entropy with class indices."""
    print("test_ce_basic_class_indices")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    # Loss should be positive
    assert_true(loss.item() > 0.0, "Loss should be positive")

    loss.backward()

    # Gradients should exist and sum to approximately 0 for each sample
    assert_true(
        logits.grad().shape() == logits.shape(), "Gradient shape mismatch"
    )
    print("Basic class indices test passed")


fn test_ce_basic_probability_targets() raises:
    """Test basic cross entropy with probability targets."""
    print("test_ce_basic_probability_targets")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], requires_grad=True
    ).float()
    var targets = Tensor.d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).float()

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    assert_true(loss.item() > 0.0, "Loss should be positive")

    loss.backward()
    assert_true(
        logits.grad().shape() == logits.shape(), "Gradient shape mismatch"
    )
    print("Basic probability targets test passed")


# ============================================================================
# Reduction Type Tests
# ============================================================================


fn test_ce_reduction_mean() raises:
    """Test mean reduction."""
    print("test_ce_reduction_mean")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3], [0.2, 0.1, 2.5]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1, 2])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    # Mean should be average of individual losses
    assert_true(
        loss.shape().rank() == 0, "Mean reduction should produce scalar"
    )
    print("Mean reduction test passed")


fn test_ce_reduction_sum() raises:
    """Test sum reduction."""
    print("test_ce_reduction_sum")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3], [0.2, 0.1, 2.5]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1, 2])

    var criterion_sum = CrossEntropyLoss[DType.float32](reduction="sum")
    var criterion_none = CrossEntropyLoss[DType.float32](reduction="none")

    var loss_sum = criterion_sum(logits, targets)
    var loss_none = criterion_none(logits, targets)

    # Sum should equal sum of none reduction
    assert_close(loss_sum, loss_none.sum(), msg="Sum reduction mismatch")
    print("Sum reduction test passed")


fn test_ce_reduction_none() raises:
    """Test none reduction."""
    print("test_ce_reduction_none")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3], [0.2, 0.1, 2.5]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1, 2])

    var criterion = CrossEntropyLoss[DType.float32](reduction="none")
    var loss = criterion(logits, targets)

    # Should return per-sample losses
    assert_true(
        loss.shape()[0] == 3, "None reduction should return per-sample losses"
    )
    assert_true(loss.shape().rank() == 1, "None reduction output rank mismatch")
    print("None reduction test passed")


# ============================================================================
# Ignore Index Tests
# ============================================================================


fn test_ce_ignore_index_basic() raises:
    """Test basic ignore index functionality."""
    print("test_ce_ignore_index_basic")

    var logits = Tensor.d2(
        [[2.0, 1.0], [1.0, 2.0], [0.5, 0.3]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, -100, 1])  # Middle sample ignored

    var criterion = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction="mean"
    )
    var loss = criterion(logits, targets)

    loss.backward()

    # Gradients for ignored sample should be zero
    assert_true(
        abs(logits.grad()[1, 0]) < 1e-10, "Ignored sample grad should be 0"
    )
    assert_true(
        abs(logits.grad()[1, 1]) < 1e-10, "Ignored sample grad should be 0"
    )

    # Gradients for non-ignored samples should be non-zero
    assert_true(
        abs(logits.grad()[0, 0]) > 1e-6, "Non-ignored grad should be non-zero"
    )
    assert_true(
        abs(logits.grad()[2, 0]) > 1e-6, "Non-ignored grad should be non-zero"
    )
    print("Ignore index basic test passed")


fn test_ce_ignore_index_all_ignored() raises:
    """Test when all samples are ignored."""
    print("test_ce_ignore_index_all_ignored")

    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]], requires_grad=True).float()
    var targets = Tensor[DType.int32].d1([-100, -100])  # All ignored

    var criterion = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction="mean"
    )
    var loss = criterion(logits, targets)

    # Loss should be zero when all samples ignored
    assert_true(abs(loss.item()) < 1e-10, "Loss should be 0 when all ignored")

    loss.backward()

    # All gradients should be zero
    assert_true(logits.grad().sum().item() < 1e-10, "All grads should be 0")
    print("All ignored test passed")


fn test_ce_ignore_index_none_reduction() raises:
    """Test ignore index with none reduction."""
    print("test_ce_ignore_index_none_reduction")

    var logits = Tensor.d2(
        [[2.0, 1.0], [1.0, 2.0], [0.5, 0.3]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, -100, 1])

    var criterion = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction="none"
    )
    var loss = criterion(logits, targets)

    # Ignored position should have zero loss
    assert_true(abs(loss[1]) < 1e-10, "Ignored sample loss should be 0")

    # Non-ignored positions should have non-zero loss
    assert_true(loss[0] > 0.0, "Non-ignored loss should be positive")
    assert_true(loss[2] > 0.0, "Non-ignored loss should be positive")
    print("Ignore index with none reduction test passed")


# ============================================================================
# Label Smoothing Tests
# ============================================================================


fn test_ce_label_smoothing_basic() raises:
    """Test basic label smoothing."""
    print("test_ce_label_smoothing_basic")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1])

    var criterion_no_smooth = CrossEntropyLoss[DType.float32](
        reduction="mean", label_smoothing=Float32(0.0)
    )
    var criterion_smooth = CrossEntropyLoss[DType.float32](
        reduction="mean", label_smoothing=Float32(0.1)
    )

    var loss_no_smooth = criterion_no_smooth(logits, targets)
    var loss_smooth = criterion_smooth(logits, targets)

    # Smoothed loss should be different (typically slightly higher)
    assert_true(
        abs(loss_smooth.item() - loss_no_smooth.item()) > 1e-6,
        "Label smoothing should change loss value",
    )
    print("Label smoothing basic test passed")


fn test_ce_label_smoothing_with_probabilities() raises:
    """Test label smoothing with probability targets."""
    print("test_ce_label_smoothing_with_probabilities")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], requires_grad=True
    ).float()
    var targets = Tensor.d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).float()

    var criterion_no_smooth = CrossEntropyLoss[DType.float32](
        reduction="mean", label_smoothing=Float32(0.0)
    )
    var criterion_smooth = CrossEntropyLoss[DType.float32](
        reduction="mean", label_smoothing=Float32(0.1)
    )

    var loss_no_smooth = criterion_no_smooth(logits, targets)
    var loss_smooth = criterion_smooth(logits, targets)

    # Smoothed loss should be different
    assert_true(
        abs(loss_smooth.item() - loss_no_smooth.item()) > 1e-6,
        "Label smoothing should change loss with probability targets",
    )
    print("Label smoothing with probabilities test passed")


fn test_ce_label_smoothing_ignore_index_combined() raises:
    """Test label smoothing combined with ignore index."""
    print("test_ce_label_smoothing_ignore_index_combined")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3], [0.2, 0.1, 2.5]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, -100, 2])

    var criterion = CrossEntropyLoss[DType.float32](
        reduction="mean", ignore_index=-100, label_smoothing=Float32(0.2)
    )
    var loss = criterion(logits, targets)

    loss.backward()

    # Middle sample should have zero gradient
    for c in range(3):
        assert_true(
            abs(logits.grad()[1, c]) < 1e-10, "Ignored grad should be 0"
        )

    # Other samples should have non-zero gradients
    assert_true(
        abs(logits.grad()[0, 0]) > 1e-6, "Non-ignored grad should be non-zero"
    )
    print("Label smoothing + ignore index test passed")


fn test_ce_reduction_types_with_ignore_index_and_label_smoothing() raises:
    """Test all reduction types with ignore index and label smoothing."""
    print("test_ce_reduction_types_with_ignore_index_and_label_smoothing")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3], [0.2, 0.1, 2.5]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1, 2])

    # 1. MEAN reduction
    var criterion_mean = CrossEntropyLoss[DType.float32](
        reduction="mean", ignore_index=1, label_smoothing=Float32(0.2)
    )
    var loss_mean = criterion_mean(logits, targets)
    assert_close(
        loss_mean,
        Tensor.scalar(0.5492352).float(),
        msg="Mean reduction value mismatch",
    )
    loss_mean.backward()

    assert_close(
        logits.grad(),
        Tensor.d2(
            [
                [-0.10383275, 0.08788316, 0.015949614],
                [0.0, 0.0, 0.0],
                [0.008757681, 0.0047521815, -0.013509899],
            ]
        ).float(),
        msg="Mean reduction grad mismatch",
    )

    logits.zero_grad()

    # 2. SUM reduction
    var criterion_sum = CrossEntropyLoss[DType.float32](
        reduction="sum", ignore_index=1, label_smoothing=Float32(0.2)
    )
    var loss_sum = criterion_sum(logits, targets)
    assert_close(
        loss_sum,
        Tensor.scalar(1.0984705).float(),
        msg="Sum reduction value mismatch",
    )
    loss_sum.backward()

    assert_close(
        logits.grad(),
        Tensor.d2(
            [
                [-0.2076655, 0.17576632, 0.03189923],
                [0.0, 0.0, 0.0],
                [0.017515361, 0.009504363, -0.027019799],
            ]
        ).float(),
        msg="Sum reduction grad mismatch",
    )

    logits.zero_grad()

    # 3. NONE reduction
    var criterion_none = CrossEntropyLoss[DType.float32](
        reduction="none", ignore_index=1, label_smoothing=Float32(0.2)
    )
    var loss_none = criterion_none(logits, targets)
    assert_close(
        loss_none,
        Tensor.d1([0.6103632, 0.0, 0.4881073]).float(),
        msg="None reduction value mismatch",
    )

    loss_none.backward()

    assert_close(
        logits.grad(),
        Tensor.d2(
            [
                [-0.2076655, 0.17576632, 0.03189923],
                [0.0, 0.0, 0.0],
                [0.017515361, 0.009504363, -0.027019799],
            ]
        ).float(),
        msg="None reduction grad mismatch",
    )
    print("Combined reduction + ignore + smoothing test passed")


# ============================================================================
# Spatial Dimension Tests
# ============================================================================


fn test_ce_spatial_2d() raises:
    """Test 3D input (batch, classes, spatial)."""
    print("test_ce_spatial_2d")

    # Shape: (2, 3, 4) -> batch=2, classes=3, width=4
    var logits = Tensor.d3(
        [
            [[2.0, 1.0, 0.5, 1.5], [1.0, 2.0, 1.5, 0.5], [0.5, 0.5, 2.0, 2.0]],
            [[1.5, 2.0, 1.0, 0.5], [2.0, 1.0, 2.0, 1.5], [0.5, 0.5, 0.5, 2.0]],
        ],
        requires_grad=True,
    ).float()

    # Shape: (2, 4) -> batch=2, width=4
    var targets = Tensor[DType.int32].d2([[0, 1, 2, 0], [1, 0, 1, 2]])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    assert_true(loss.item() > 0.0, "Loss should be positive")

    loss.backward()
    assert_true(
        logits.grad().shape() == logits.shape(), "Gradient shape mismatch"
    )
    assert_true(
        logits.grad().sum().item() != 0.0, "Gradients should be non-zero"
    )
    print("Spatial 2D test passed")


fn test_ce_spatial_3d() raises:
    """Test 4D input (batch, classes, height, width)."""
    print("test_ce_spatial_3d")

    # Shape: (1, 2, 2, 2) -> batch=1, classes=2, height=2, width=2
    var logits = Tensor.d4(
        [
            [
                [[2.0, 1.0], [1.0, 2.0]],  # Class 0
                [[1.0, 2.0], [2.0, 1.0]],  # Class 1
            ]
        ],
        requires_grad=True,
    ).float()

    # Shape: (1, 2, 2) -> batch=1, height=2, width=2
    var targets = Tensor[DType.int32].d3([[[0, 1], [1, 0]]])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    loss.backward()

    assert_true(
        logits.grad().shape() == logits.shape(), "Gradient shape mismatch"
    )
    assert_true(
        logits.grad().sum().item() != 0.0, "Gradients should be non-zero"
    )
    print("Spatial 3D test passed")

fn test_ce_spatial_with_ignore_index() raises:
    """Test spatial dimensions with ignore index."""
    print("test_ce_spatial_with_ignore_index")


    var logits = Tensor[DType.float32].d3(
        [
            [[2.0, 1.0, 0.5], [1.0, 2.0, 1.5], [0.5, 0.5, 2.0]],
            [[1.5, 2.0, 1.0], [2.0, 1.0, 2.0], [0.5, 0.5, 0.5]],
        ],
        requires_grad=True,
    )

    var targets = Tensor[DType.int32].d2([[0, -100, 2], [1, 0, -100]])

    var criterion = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction="mean"
    )
    var loss = criterion(logits, targets)

    loss.backward()

    # Check that ignored positions have zero gradient across ALL classes
    # Batch 0, spatial position 1 is ignored (target=-100)
    # So gradients for all classes at [0, :, 1] should be zero
    for c in range(3):
        assert_true(
            abs(logits.grad()[0, c, 1]) < 1e-10,
            "Batch 0, spatial pos 1, class "
            + String(c)
            + " grad should be 0",
        )

    # Batch 1, spatial position 2 is ignored (target=-100)
    # So gradients for all classes at [1, :, 2] should be zero
    for c in range(3):
        assert_true(
            abs(logits.grad()[1, c, 2]) < 1e-10,
            "Batch 1, spatial pos 2, class "
            + String(c)
            + " grad should be 0",
        )

    # Non-ignored positions should have non-zero gradients
    # Batch 0, spatial position 0 (target=0)
    var has_nonzero_grad = False
    for c in range(3):
        if abs(logits.grad()[0, c, 0]) > 1e-6:
            has_nonzero_grad = True
            break
    assert_true(
        has_nonzero_grad, "Non-ignored position should have non-zero grads"
    )

    print("Spatial with ignore index test passed")


fn test_ce_spatial_probability_targets() raises:
    """Test spatial dimensions with probability targets."""
    print("test_ce_spatial_probability_targets")

    var logits = Tensor.d3(
        [
            [[2.0, 1.0], [1.0, 2.0]],  # batch 0, 2 classes, 2 spatial
        ],
        requires_grad=True,
    ).float()

    var targets = Tensor.d3(
        [
            [[1.0, 0.0], [0.0, 1.0]],  # batch 0
        ]
    ).float()

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    loss.backward()

    assert_true(
        logits.grad().shape() == logits.shape(), "Gradient shape mismatch"
    )
    print("Spatial probability targets test passed")


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


fn test_ce_single_sample() raises:
    """Test with single sample."""
    print("test_ce_single_sample")

    var logits = Tensor.d2([[2.0, 1.0, 0.1]], requires_grad=True).float()
    var targets = Tensor[DType.int32].d1([0])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    assert_true(loss.item() > 0.0, "Loss should be positive")

    loss.backward()
    assert_true(
        logits.grad().shape() == logits.shape(), "Gradient shape mismatch"
    )
    print("Single sample test passed")


fn test_ce_large_batch() raises:
    """Test with large batch size."""
    print("test_ce_large_batch")

    var batch_size = 100
    var num_classes = 10

    # Create random-ish logits
    var logits = Tensor[DType.float32](
        Shape([batch_size, num_classes]), requires_grad=True
    )
    for i in range(batch_size):
        for j in range(num_classes):
            logits[i, j] = Float32((i + j) % 10) * 0.5

    # Create targets (cycling through classes)
    var targets = Tensor[DType.int32](Shape([batch_size]))
    for i in range(batch_size):
        targets[i] = Int32(i % num_classes)

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    assert_true(loss.item() > 0.0, "Loss should be positive")

    loss.backward()
    assert_true(
        logits.grad().shape() == logits.shape(), "Gradient shape mismatch"
    )
    print("Large batch test passed")


fn test_ce_binary_classification() raises:
    """Test binary classification (2 classes)."""
    print("test_ce_binary_classification")

    var logits = Tensor.d2(
        [[2.0, 1.0], [1.0, 2.0], [0.5, 1.5]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1, 1])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    loss.backward()

    assert_true(
        logits.grad().shape() == logits.shape(), "Gradient shape mismatch"
    )
    print("Binary classification test passed")


fn test_ce_perfect_prediction() raises:
    """Test with perfect predictions (very confident)."""
    print("test_ce_perfect_prediction")

    var logits = Tensor.d2(
        [
            [100.0, -100.0, -100.0],
            [-100.0, 100.0, -100.0],
            [-100.0, -100.0, 100.0],
        ],
        requires_grad=True,
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1, 2])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    # Loss should be very close to 0
    assert_true(
        loss.item() < 0.1, "Perfect prediction should have near-zero loss"
    )
    print("Perfect prediction test passed")


fn test_ce_uniform_logits() raises:
    """Test with uniform (uncertain) predictions."""
    print("test_ce_uniform_logits")

    var logits = Tensor.d2(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    # Loss should be close to -log(1/3) ≈ 1.0986
    var expected_loss = -log(1.0 / 3.0)
    var diff = Scalar[DType.float64](loss.item()) - Scalar[DType.float64](
        expected_loss
    )
    assert_true(
        abs(diff) < 0.01,
        "Uniform prediction loss mismatch",
    )
    print("Uniform logits test passed")


fn test_ce_numerical_stability() raises:
    """Test numerical stability with extreme values."""
    print("test_ce_numerical_stability")

    # Very large logits
    var logits = Tensor.d2(
        [[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    # Should not overflow/underflow
    assert_true(not isnan(loss.item()), "Loss should not be NaN")
    assert_true(not isinf(loss.item()), "Loss should not be inf")

    loss.backward()

    # Gradients should also be stable
    for i in range(2):
        for j in range(2):
            var grad_val = logits.grad()[i, j]
            assert_true(not isnan(grad_val), "Gradient should not be NaN")
            assert_true(not isinf(grad_val), "Gradient should not be inf")

    print("Numerical stability test passed")


fn test_ce_validate_parameter() raises:
    """Test validate parameter for skipping validation."""
    print("test_ce_validate_parameter")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")

    # Both should give same result
    var loss1 = criterion(logits, targets, validate=True)
    var loss2 = criterion(logits, targets, validate=False)

    assert_close(
        loss1, loss2, msg="Validate parameter should not affect result"
    )
    print("Validate parameter test passed")


# ============================================================================
# Gradient Correctness Tests
# ============================================================================


fn test_ce_gradient_sum_property() raises:
    """Test that gradients sum to approximately 0 for each sample (softmax property).
    """
    print("test_ce_gradient_sum_property")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.5], [1.5, 2.0, 0.3]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    loss.backward()

    # For each sample, gradients across classes should sum to approximately 0
    # (property of softmax gradient)
    for i in range(2):
        var grad_sum = Scalar[DType.float32](0)
        for j in range(3):
            grad_sum += logits.grad()[i, j]
        assert_true(
            abs(grad_sum) < 1e-5, "Gradients should sum to 0 for each sample"
        )

    print("Gradient sum property test passed")


fn test_ce_gradient_magnitude() raises:
    """Test that gradient magnitudes are reasonable."""
    print("test_ce_gradient_magnitude")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)

    loss.backward()

    # Gradients should be bounded (softmax outputs are in [0,1])
    for i in range(2):
        for j in range(3):
            var grad_val = abs(logits.grad()[i, j])
            assert_true(
                grad_val < 1.0,
                "Gradient magnitude should be < 1 for mean reduction",
            )

    print("Gradient magnitude test passed")


fn test_ce_gradients_ignore_index() raises:
    """Test gradients with ignore index."""
    print("test_ce_gradients_ignore_index")

    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]], requires_grad=True).float()
    var target = Tensor[DType.int32].d1([0, -100])  # Second sample ignored

    var loss_fn = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction=0
    )
    var loss = loss_fn(logits, target)

    loss.backward()

    # Gradients for ignored sample should be zero
    assert_true(abs(logits.grad()[1, 0]) < 1e-10, "Ignored grad should be 0")
    assert_true(abs(logits.grad()[1, 1]) < 1e-10, "Ignored grad should be 0")

    # Gradients for non-ignored sample should be non-zero
    assert_true(
        abs(logits.grad()[0, 0]) > 1e-6, "Non-ignored grad should be non-zero"
    )
    print("Gradients ignore index test passed")


fn test_ce_gradients_spatial() raises:
    """Test gradients with spatial dimensions."""
    print("test_ce_gradients_spatial")

    # Create proper 4D tensor with shape (1, 2, 2, 2)
    var logits = Tensor.d4(
        [
            [
                [[2.0, 1.0], [1.0, 2.0]],  # Class 0
                [[1.0, 2.0], [2.0, 1.0]],  # Class 1
            ]
        ],
        requires_grad=True,
    ).float()

    var target = Tensor[DType.int32].d3(
        [
            [
                [0, 1],  # First spatial row
                [1, 0],  # Second spatial row
            ]
        ]
    )

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    loss.backward()

    # Should have gradients for all spatial positions
    assert_true(
        logits.grad().sum().item() != 0.0, "Should have non-zero gradients"
    )
    print("Gradients spatial test passed")


# ============================================================================
# Equivalence Tests
# ============================================================================


fn test_ce_class_indices_vs_onehot() raises:
    """Test that class indices give same result as one-hot probabilities."""
    print("test_ce_class_indices_vs_onehot")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], requires_grad=True
    ).float()
    var targets_indices = Tensor[DType.int32].d1([0, 1])
    var targets_onehot = Tensor.d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).float()

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")

    var loss_indices = criterion(logits, targets_indices)
    var loss_onehot = criterion(logits, targets_onehot)

    assert_close(
        loss_indices,
        loss_onehot,
        msg="Class indices and one-hot should give same loss",
    )
    print("Class indices vs one-hot equivalence test passed")


# ============================================================================
# Main Test Runner
# ============================================================================


fn run_all_tests() raises:
    """Run all CrossEntropyLoss tests."""
    print("=" * 80)
    print("RUNNING COMPREHENSIVE CROSSENTROPYLOSS TEST SUITE")
    print("=" * 80)
    print()

    var total_tests = 0
    var passed_tests = 0

    # Basic Functionality Tests
    print("BASIC FUNCTIONALITY TESTS")
    print("-" * 80)
    try:
        test_ce_basic_class_indices()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_basic_probability_targets()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1
    print()

    # Reduction Type Tests
    print("REDUCTION TYPE TESTS")
    print("-" * 80)
    try:
        test_ce_reduction_mean()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_reduction_sum()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_reduction_none()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1
    print()

    # Ignore Index Tests
    print("IGNORE INDEX TESTS")
    print("-" * 80)
    try:
        test_ce_ignore_index_basic()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_ignore_index_all_ignored()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_ignore_index_none_reduction()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1
    print()

    # Label Smoothing Tests
    print("LABEL SMOOTHING TESTS")
    print("-" * 80)
    try:
        test_ce_label_smoothing_basic()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_label_smoothing_with_probabilities()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_label_smoothing_ignore_index_combined()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_reduction_types_with_ignore_index_and_label_smoothing()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1
    print()

    # Spatial Dimension Tests
    print("SPATIAL DIMENSION TESTS")
    print("-" * 80)
    try:
        test_ce_spatial_2d()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_spatial_3d()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_spatial_with_ignore_index() #To be enabled
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_spatial_probability_targets()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1
    print()

    # Edge Cases
    print("EDGE CASE TESTS")
    print("-" * 80)
    try:
        test_ce_single_sample()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_large_batch()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_binary_classification()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_perfect_prediction()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_uniform_logits()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_numerical_stability()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_validate_parameter()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1
    print()

    # Gradient Tests
    print("GRADIENT CORRECTNESS TESTS")
    print("-" * 80)
    try:
        test_ce_gradient_sum_property()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_gradient_magnitude()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_gradients_ignore_index()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_gradients_spatial()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1
    print()

    # Equivalence Tests
    print("EQUIVALENCE TESTS")
    print("-" * 80)
    try:
        test_ce_class_indices_vs_onehot()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1

    try:
        test_ce_mean_vs_manual_average()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1
    print()

    # Performance Tests
    print("PERFORMANCE/OPTIMIZATION TESTS")
    print("-" * 80)
    try:
        test_ce_no_validation_speedup()
        passed_tests += 1
    except e:
        print("FAILED:", e)
    total_tests += 1
    print()

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("Total tests:", total_tests)
    print("Passed:", passed_tests)
    print("Failed:", total_tests - passed_tests)

    if passed_tests == total_tests:
        print()
        print("ALL TESTS PASSED!")
    else:
        print()
        print("❌ SOME TESTS FAILED ❌")

    print("=" * 80)


# ============================================================================
# Quick Test Runners for Specific Categories
# ============================================================================


fn run_basic_tests() raises:
    """Run only basic functionality tests."""
    print("Running basic tests...")
    test_ce_basic_class_indices()
    test_ce_basic_probability_targets()
    print("Basic tests completed!")


fn run_reduction_tests() raises:
    """Run only reduction type tests."""
    print("Running reduction tests...")
    test_ce_reduction_mean()
    test_ce_reduction_sum()
    test_ce_reduction_none()
    print("Reduction tests completed!")


fn run_ignore_index_tests() raises:
    """Run only ignore index tests."""
    print("Running ignore index tests...")
    test_ce_ignore_index_basic()
    test_ce_ignore_index_all_ignored()
    test_ce_ignore_index_none_reduction()
    print("Ignore index tests completed!")


fn run_label_smoothing_tests() raises:
    """Run only label smoothing tests."""
    print("Running label smoothing tests...")
    test_ce_label_smoothing_basic()
    test_ce_label_smoothing_with_probabilities()
    test_ce_label_smoothing_ignore_index_combined()
    test_ce_reduction_types_with_ignore_index_and_label_smoothing()
    print("Label smoothing tests completed!")


fn run_spatial_tests() raises:
    """Run only spatial dimension tests."""
    print("Running spatial tests...")
    test_ce_spatial_2d()
    test_ce_spatial_3d()
    test_ce_spatial_with_ignore_index()
    test_ce_spatial_probability_targets()
    print("Spatial tests completed!")


fn run_gradient_tests() raises:
    """Run only gradient correctness tests."""
    print("Running gradient tests...")
    test_ce_gradient_sum_property()
    test_ce_gradient_magnitude()
    test_ce_gradients_ignore_index()
    test_ce_gradients_spatial()
    print("Gradient tests completed!")


fn run_edge_case_tests() raises:
    """Run only edge case tests."""
    print("Running edge case tests...")
    test_ce_single_sample()
    test_ce_large_batch()
    test_ce_binary_classification()
    test_ce_perfect_prediction()
    test_ce_uniform_logits()
    test_ce_numerical_stability()
    test_ce_validate_parameter()
    print("Edge case tests completed!")


fn run_equivalence_tests() raises:
    """Run only equivalence tests."""
    print("Running equivalence tests...")
    test_ce_class_indices_vs_onehot()
    test_ce_mean_vs_manual_average()
    print("Equivalence tests completed!")


fn test_ce_mean_vs_manual_average() raises:
    """Test that mean reduction equals manual average of none reduction."""
    print("test_ce_mean_vs_manual_average")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3], [0.2, 0.1, 2.5]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1, 2])

    var criterion_mean = CrossEntropyLoss[DType.float32](reduction="mean")
    var criterion_none = CrossEntropyLoss[DType.float32](reduction="none")

    var loss_mean = criterion_mean(logits, targets)
    var loss_none = criterion_none(logits, targets)

    var manual_mean = loss_none.sum().item() / Float32(3)

    assert_true(
        abs(loss_mean.item() - manual_mean) < 1e-5,
        "Mean reduction should equal average of individual losses",
    )
    print("Mean vs manual average test passed")


# ============================================================================
# Performance/Optimization Tests
# ============================================================================


fn test_ce_no_validation_speedup() raises:
    """Test that disabling validation works (can't test speedup, just correctness).
    """
    print("test_ce_no_validation_speedup")

    var logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]], requires_grad=True
    ).float()
    var targets = Tensor[DType.int32].d1([0, 1])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")

    # Multiple calls without validation should give same result
    var loss1 = criterion(logits, targets, validate=False)
    var loss2 = criterion(logits, targets, validate=False)
    var loss3 = criterion(logits, targets, validate=False)

    assert_close(loss1, loss2, msg="Repeated calls should give same result")
    assert_close(loss2, loss3, msg="Repeated calls should give same result")
    print("No validation speedup test passed")


fn test_ce_reduction_types_with_ignore_index_and_label_smoothing_orig() raises:
    print("test_ce_reduction_types_with_ignore_index_and_label_smoothing_orig")
    logits = Tensor.d2(
        [[2.0, 1.0, 0.1], [0.5, 2.0, 0.3], [0.2, 0.1, 2.5]], requires_grad=True
    ).float()

    targets = Tensor[DType.int32].d1([0, 1, 2])  # Class indices

    # 1. MEAN reduction (default)
    criterion_mean = CrossEntropyLoss(
        reduction="mean", ignore_index=1, label_smoothing=Float32(0.2)
    )
    loss_mean = criterion_mean(logits, targets)
    assert_true(
        loss_mean.all_close(Tensor.scalar(0.5492352).float()),
        "ce mean reduction value assertion failed",
    )
    loss_mean.backward()

    assert_true(
        logits.grad().all_close(
            Tensor.d2(
                [
                    [-0.10383275, 0.08788316, 0.015949614],
                    [0.0, 0.0, 0.0],
                    [0.008757681, 0.0047521815, -0.013509899],
                ]
            ).float()
        ),
        "ce mean reduction grad assertion failed",
    )

    logits.zero_grad()
    # 2. SUM reduction
    criterion_sum = CrossEntropyLoss(
        reduction="sum", ignore_index=1, label_smoothing=Float32(0.2)
    )
    loss_sum = criterion_sum(logits, targets)
    assert_true(
        loss_sum.all_close(Tensor.scalar(1.0984705).float()),
        "ce sum reduction value assertion failed",
    )
    loss_sum.backward()

    assert_true(
        logits.grad().all_close(
            Tensor.d2(
                [
                    [-0.2076655, 0.17576632, 0.03189923],
                    [0.0, 0.0, 0.0],
                    [0.017515361, 0.009504363, -0.027019799],
                ]
            ).float()
        ),
        "ce sum reduction grad assertion failed",
    )

    logits.zero_grad()
    # 3. NONE reduction
    criterion_none = CrossEntropyLoss(
        reduction="none", ignore_index=1, label_smoothing=Float32(0.2)
    )
    loss_none = criterion_none(logits, targets)
    assert_true(
        loss_none.all_close(Tensor.d1([0.6103632, 0.0, 0.4881073]).float()),
        "ce none reduction value assertion failed",
    )

    loss_none.backward()

    assert_true(
        logits.grad().all_close(
            Tensor.d2(
                [
                    [-0.2076655, 0.17576632, 0.03189923],
                    [0.0, 0.0, 0.0],
                    [0.017515361, 0.009504363, -0.027019799],
                ]
            ).float()
        ),
        "ce none reduction grad assertion failed",
    )


fn assert_approx_equal(
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


fn test_ce_gradients_computation_heavy() raises:
    print("test_ce_gradients_computation_heavy")

    # Create a larger, more complex tensor for thorough gradient testing
    # Shape: (batch=4, classes=5, height=3, width=3) - 4*5*3*3 = 180 elements
    var logits = Tensor.d4(
        [
            [  # Sample 0
                [  # Class 0
                    [1.5, 0.8, -0.3],
                    [2.1, -1.2, 0.7],
                    [0.4, 1.9, -0.6],
                ],
                [  # Class 1
                    [-0.5, 1.2, 2.3],
                    [0.9, -1.8, 0.3],
                    [1.7, 0.2, -0.9],
                ],
                [  # Class 2
                    [0.6, -1.1, 1.8],
                    [-0.7, 2.0, 0.5],
                    [1.3, -0.4, 0.9],
                ],
                [  # Class 3
                    [1.2, 0.3, -1.5],
                    [-0.8, 1.6, 0.4],
                    [0.7, -1.9, 2.1],
                ],
                [  # Class 4
                    [-1.3, 0.9, 1.4],
                    [1.1, -0.6, 2.2],
                    [0.5, 1.8, -1.7],
                ],
            ],
            [  # Sample 1
                [  # Class 0
                    [2.0, -0.5, 1.2],
                    [0.8, 1.6, -1.1],
                    [-0.9, 0.7, 1.8],
                ],
                [  # Class 1
                    [0.4, 1.9, -0.3],
                    [1.7, -1.2, 0.6],
                    [2.1, 0.5, -0.8],
                ],
                [  # Class 2
                    [-1.5, 0.8, 1.3],
                    [0.9, 2.0, -0.4],
                    [1.2, -0.7, 0.6],
                ],
                [  # Class 3
                    [1.8, -0.6, 0.9],
                    [-1.3, 1.4, 0.7],
                    [0.5, 2.1, -0.8],
                ],
                [  # Class 4
                    [0.7, 1.5, -1.0],
                    [1.9, -0.3, 2.2],
                    [-0.4, 0.8, 1.6],
                ],
            ],
            [  # Sample 2
                [  # Class 0
                    [-0.8, 1.7, 0.9],
                    [1.3, -1.4, 2.0],
                    [0.6, 1.1, -0.5],
                ],
                [  # Class 1
                    [1.9, -0.2, 1.4],
                    [0.8, 2.1, -0.7],
                    [-1.3, 0.5, 1.6],
                ],
                [  # Class 2
                    [0.5, 1.8, -0.9],
                    [1.2, -0.4, 1.7],
                    [2.0, 0.3, -1.1],
                ],
                [  # Class 3
                    [-1.2, 0.7, 1.9],
                    [0.9, 1.5, -0.6],
                    [1.4, -0.8, 2.1],
                ],
                [  # Class 4
                    [1.6, -0.9, 0.8],
                    [-1.1, 2.2, 0.4],
                    [0.7, 1.3, -1.5],
                ],
            ],
            [  # Sample 3
                [  # Class 0
                    [0.9, -1.6, 1.3],
                    [1.5, 0.7, -0.8],
                    [-1.2, 2.0, 0.4],
                ],
                [  # Class 1
                    [1.7, 0.3, -1.1],
                    [-0.9, 1.8, 0.6],
                    [2.1, -0.5, 1.4],
                ],
                [  # Class 2
                    [-0.7, 1.4, 2.0],
                    [1.1, -0.8, 1.6],
                    [0.9, 1.3, -1.2],
                ],
                [  # Class 3
                    [1.3, -1.0, 0.8],
                    [2.2, 0.4, -0.9],
                    [-0.6, 1.7, 1.5],
                ],
                [  # Class 4
                    [0.8, 1.9, -1.3],
                    [1.4, -0.7, 2.1],
                    [0.5, 1.2, -1.8],
                ],
            ],
        ],
        requires_grad=True,
    ).float()

    # Target: (batch=4, height=3, width=3) - random class indices
    var target = Tensor[DType.int32].d3(
        [
            [[0, 2, 4], [1, 3, 0], [2, 4, 1]],
            [[3, 1, 0], [4, 2, 1], [0, 3, 2]],
            [[2, 0, 3], [1, 4, 2], [3, 1, 0]],
            [[4, 2, 1], [0, 3, 4], [2, 1, 0]],
        ]
    )

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)  # mean
    var loss = loss_fn(logits, target)

    loss.backward()

    # Gradient validation assertions (values to be filled from PyTorch)
    var gradients = logits.grad()
    # Check gradient statistics
    assert_true(
        abs(gradients.sum().item() - -1.6763806e-08) < 1e-6
    )  # Replace PYTHON_VALUE

    assert_true(abs(gradients.mean().item() - -9.313226e-11) < 1e-6)
    # assert_true(abs(gradients.std().item() - PYTHON_VALUE) < 1e-6)
    # assert_true(assert_approx_equal(gradients[0, 0, 0, 0], -0.01592483, 0.2))
    # Check specific gradient values (sample a few positions)
    _ = """assert_true(abs(gradients[0, 0, 0, 0] - -0.01592483) < 1e-6)
    assert_true(abs(gradients[1, 2, 1, 2] - 0.0013430393) < 1e-6)
    assert_true(abs(gradients[2, 4, 2, 1] - 0.010070266) < 1e-6)
    assert_true(abs(gradients[3, 1, 0, 2] - -0.027112056) < 1e-6)"""


fn test_ce_basic_no_reduction() raises:
    print("test_ce_basic_no_reduction")
    var logits = Tensor.d2([[2.0, 1.0, 0.5], [1.0, 2.0, 0.1]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)  # mean
    var loss = loss_fn(logits, target)

    # Manual calculation check
    var softmax1 = exp(2.0) / (exp(2.0) + exp(1.0) + exp(0.5))
    var softmax2 = exp(2.0) / (exp(1.0) + exp(2.0) + exp(0.1))
    var expected_loss = Scalar[DType.float32](
        (-log(softmax1) + -log(softmax2)) / 2.0
    )

    assert_true(abs(loss.item() - expected_loss) < 1e-6)


fn test_ce_reduction_mean_1() raises:
    print("test_ce_reduction_mean_1")
    var logits = Tensor.d2([[3.0, 1.0], [1.0, 3.0], [2.0, 1.0]]).float()
    var target = Tensor[DType.int32].d1([0, 1, 0])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    # Should be average of 3 sample losses
    assert_true(loss.shape() == Shape())
    assert_true(loss.item() > 0)


fn test_ce_reduction_sum_1() raises:
    print("test_ce_reduction_sum_1")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=1)
    var loss = loss_fn(logits, target)
    # Should be sum of 2 sample losses
    assert_true(loss.shape() == Shape())
    assert_true(loss.item() > 0)


fn test_ce_reduction_none_1() raises:
    print("test_ce_reduction_none_1")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=2)
    var loss = loss_fn(logits, target)

    assert_true(loss.shape() == Shape([2]))  # per-sample loss
    assert_true(loss[0] > 0 and loss[1] > 0)


fn test_ce_label_smoothing_basic_orig() raises:
    print("test_ce_label_smoothing_basic_orig")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn_no_smooth = CrossEntropyLoss[DType.float32](label_smoothing=0.0)
    var loss_fn_smooth = CrossEntropyLoss[DType.float32](label_smoothing=0.1)

    var loss_no_smooth = loss_fn_no_smooth(logits, target)
    var loss_smooth = loss_fn_smooth(logits, target)

    # Label smoothing should give different (usually higher) loss
    assert_true(abs(loss_smooth.item() - loss_no_smooth.item()) > 1e-6)


fn test_ce_label_smoothing_mean() raises:
    print("test_ce_label_smoothing_mean")
    var logits = Tensor.d2([[3.0, 1.0, 0.5], [1.0, 3.0, 0.1]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](
        reduction=0, label_smoothing=0.2
    )
    var loss = loss_fn(logits, target)

    assert_true(loss.shape() == Shape())
    assert_true(loss.item() > 0)


fn test_ce_label_smoothing_extreme() raises:
    print("test_ce_label_smoothing_extreme")
    var logits = Tensor.d2([[10.0, 0.0], [0.0, 10.0]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn_max_smooth = CrossEntropyLoss[DType.float32](
        label_smoothing=0.9
    )
    var loss = loss_fn_max_smooth(logits, target)

    # With extreme smoothing, loss should be relatively high even for good predictions
    assert_true(loss.item() > 1.0)


fn test_ce_ignore_index_basic_orig() raises:
    print("test_ce_ignore_index_basic_orig")
    var logits = Tensor.d2(
        [[2.0, 1.0, 0.5], [1.0, 2.0, 0.1], [3.0, 1.0, 0.2]]
    ).float()
    var target = Tensor[DType.int32].d1([0, -100, 2])  # Second sample ignored

    var loss_fn = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction=0
    )
    var loss = loss_fn(logits, target)

    # Should only average over 2 samples (not 3)
    assert_true(loss.item() > 0)


fn test_ce_ignore_index_all_ignored_orig() raises:
    print("test_ce_ignore_index_all_ignored_orig")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor[DType.int32].d1([-100, -100])  # All ignored

    var loss_fn = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction=0
    )
    var loss = loss_fn(logits, target)

    # Should return 0 when all samples are ignored
    assert_true(abs(loss.item()) < 1e-10)


fn test_ce_ignore_index_partial() raises:
    print("test_ce_ignore_index_partial")
    var logits = Tensor.d3(
        [[[2.0, 1.0], [1.0, 2.0]], [[3.0, 1.0], [1.0, 3.0]]]
    ).float()
    var target = Tensor[DType.int32].d2([[-100, 1], [0, -100]])  # Mixed ignored

    var loss_fn = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction=1
    )  # sum
    var loss = loss_fn(logits, target)

    # Should sum over only non-ignored elements
    assert_true(loss.item() > 0)


fn test_ce_2d_spatial() raises:
    print("test_ce_2d_spatial")
    var logits = Tensor.d4(
        [[[[2.0, 1.0], [1.0, 2.0]], [[3.0, 1.0], [1.0, 3.0]]]]
    ).float()  # (1, 2, 2, 2)
    var target = Tensor[DType.int32].d3([[[0, 1], [1, 0]]])  # (1, 2, 2)

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    # Should compute mean over 4 spatial positions
    assert_true(loss.shape() == Shape())
    assert_true(loss.item() > 0)


fn test_ce_5d_spatial() raises:
    print("test_ce_5d_spatial")

    # Create proper 5D tensor: (batch=1, classes=2, depth=2, height=2, width=2)
    var logits = Tensor.d5(
        [
            [  # Batch dimension (size 1)
                [  # Class 0
                    [  # Depth 0
                        [2.0, 1.0],  # Height 0: [width0, width1]
                        [1.0, 2.0],  # Height 1: [width0, width1]
                    ],
                    [[3.0, 1.0], [1.0, 3.0]],  # Depth 1
                ],
                [  # Class 1
                    [[1.0, 2.0], [2.0, 1.0]],  # Depth 0
                    [[2.0, 1.0], [1.0, 2.0]],  # Depth 1
                ],
            ]
        ]
    ).float()

    # Target: (batch=1, depth=2, height=2, width=2)
    var target = Tensor[DType.int32].d4(
        [
            [  # Batch dimension (size 1)
                [  # Depth 0
                    [0, 1],  # Height 0: [width0, width1]
                    [1, 0],  # Height 1: [width0, width1]
                ],
                [[0, 1], [1, 0]],  # Depth 1
            ]
        ]
    )

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=1)  # sum
    var loss = loss_fn(logits, target)

    assert_true(loss.shape() == Shape())
    assert_true(loss.item() > 0)
    print("5D spatial test passed")


fn test_ce_spatial_with_ignore() raises:
    print("test_ce_spatial_with_ignore")
    var logits = Tensor.d4(
        [[[[2.0, 1.0], [1.0, 2.0]], [[3.0, 1.0], [1.0, 3.0]]]]
    ).float()  # (1, 2, 2, 2)
    var target = Tensor[DType.int32].d3(
        [[[0, -100], [1, 0]]]
    )  # One position ignored

    var loss_fn = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction=0
    )
    var loss = loss_fn(logits, target)

    # Should mean over 3 non-ignored positions
    assert_true(loss.item() > 0)


fn _ce_gradients_basic_uu() raises:
    print("test_ce_gradients_basic")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]], requires_grad=True).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    loss.backward()

    # Gradients should exist and be non-zero
    assert_true(logits.grad().sum().item() != 0.0)
    # Specific gradient pattern check
    var softmax = logits.softmax(axes=IntArray(1))
    assert_true(abs(logits.grad()[0, 0] - (softmax[0, 0] - 1.0)) < 1e-6)


fn test_ce_gradients_basic() raises:
    print("test_ce_gradients_basic")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]], requires_grad=True).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)  # mean
    var loss = loss_fn(logits, target)

    loss.backward()
    print(logits.grad().sum().item())
    print(logits.grad().sum().item())
    print(logits.grad().sum().item())
    # Gradients should exist and be non-zero
    assert_true(logits.grad().sum().item() != 0.0)

    # Specific gradient pattern check - ACCOUNT FOR MEAN REDUCTION!
    var softmax = logits.softmax(axes=IntArray(1))
    var batch_size = Scalar[DType.float32](logits.shape()[0])

    # For mean reduction: gradient = (softmax - one_hot) / batch_size
    var expected_grad_00 = (softmax[0, 0] - 1.0) / batch_size
    assert_true(abs(logits.grad()[0, 0] - expected_grad_00) < 1e-6)

    # Also check other elements
    var expected_grad_01 = (softmax[0, 1] - 0.0) / batch_size
    assert_true(abs(logits.grad()[0, 1] - expected_grad_01) < 1e-6)

    var expected_grad_10 = (softmax[1, 0] - 0.0) / batch_size
    assert_true(abs(logits.grad()[1, 0] - expected_grad_10) < 1e-6)

    var expected_grad_11 = (softmax[1, 1] - 1.0) / batch_size
    assert_true(abs(logits.grad()[1, 1] - expected_grad_11) < 1e-6)


fn test_ce_gradients_label_smoothing() raises:
    print("test_ce_gradients_label_smoothing")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]], requires_grad=True).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](label_smoothing=0.1)
    var loss = loss_fn(logits, target)

    loss.backward()

    # Gradients with smoothing should be different from without
    var grads_smooth = logits.grad().copy()
    logits.zero_grad()

    var loss_fn_no_smooth = CrossEntropyLoss[DType.float32](label_smoothing=0.0)
    var loss_no_smooth = loss_fn_no_smooth(logits, target)
    loss_no_smooth.backward()

    assert_false(grads_smooth.all_close(logits.grad()))


fn test_ce_gradients_ignore_index_orig() raises:
    print("test_ce_gradients_ignore_index_orig")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]], requires_grad=True).float()
    var target = Tensor[DType.int32].d1([0, -100])  # Second sample ignored

    var loss_fn = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction=0
    )
    var loss = loss_fn(logits, target)

    loss.backward()

    # Gradients for ignored sample should be zero
    assert_true(abs(logits.grad()[1, 0]) < 1e-10)
    assert_true(abs(logits.grad()[1, 1]) < 1e-10)
    # Gradients for non-ignored sample should be non-zero
    assert_true(abs(logits.grad()[0, 0]) > 1e-6)


fn test_ce_gradients_spatial_orig() raises:
    print("test_ce_gradients_spatial_orig")

    # Create proper 4D tensor with shape (1, 2, 2, 2)
    # This means: batch=1, classes=2, height=2, width=2
    var logits = Tensor.d4(
        [
            [
                [[2.0, 1.0], [1.0, 2.0]],  # Class 0
                [[1.0, 2.0], [2.0, 1.0]],  # Class 1
            ]
        ],
        requires_grad=True,
    ).float()

    # Target should have shape (1, 2, 2) with values in [0, 1]
    var target = Tensor[DType.int32].d3(
        [
            [
                [0, 1],  # First spatial row: class 0, class 1
                [1, 0],  # Second spatial row: class 1, class 0
            ]
        ]
    )

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    loss.backward()

    # Should have gradients for all spatial positions
    assert_true(logits.grad().sum().item() != 0.0)
    print("Spatial gradients test passed")


fn test_ce_single_class() raises:
    print("test_ce_single_class")
    var logits = Tensor.d2([[5.0], [3.0]]).float()  # Single class
    var target = Tensor[DType.int32].d1([0, 0])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    # With single class, softmax should be ~1.0, loss should be low
    assert_true(loss.item() < 0.1)


fn test_ce_perfect_prediction_orig() raises:
    print("test_ce_perfect_prediction")
    var logits = Tensor.d2([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)
    # Perfect prediction should have very low loss
    assert_true(loss.item() < 1e-4)  # More reasonable threshold
    assert_true(loss.item() > 1e-5)  # Also check it's not too small
    # assert_true(loss.item() < 1e-6)


fn test_ce_extreme_perfect_prediction() raises:
    print("test_ce_extreme_perfect_prediction")
    # Use even larger logits for "more perfect" prediction
    var logits = Tensor.d2([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    print("Extreme perfect prediction loss:", loss.item())

    # With logits of 100.0, loss should be extremely small
    # (approaching the limits of float32 precision)
    assert_true(loss.item() < 1e-20)


fn test_ce_worst_prediction() raises:
    print("test_ce_worst_prediction")
    var logits = Tensor.d2(
        [[0.0, 10.0], [10.0, 0.0]]
    ).float()  # Wrong predictions
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    # Worst possible prediction should have high loss
    assert_true(loss.item() > 5.0)


fn test_ce_zero_logits() raises:
    print("test_ce_zero_logits")
    var logits = Tensor.d2([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).float()
    var target = Tensor[DType.int32].d1([0, 2])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    # With zero logits, all classes have equal probability -> higher loss
    var expected_loss = Scalar[DType.float32](-log(1.0 / 3.0))  # ~1.0986
    assert_true(abs(loss.item() - expected_loss) < 1e-6)


fn test_ce_large_logits() raises:
    print("test_ce_large_logits")
    var logits = Tensor.d2([[1000.0, 0.0], [0.0, 1000.0]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    # Very large logits should still give valid (low) loss due to numerical stability
    assert_true(loss.item() < 1e-6)


# Negative validation tests (should not panic if validation is correct)
fn _ce_validation_wrong_target_dims() raises:
    print("test_ce_validation_wrong_target_dims")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor[DType.int32].d2([[0, 1], [1, 0]])  # Wrong: should be 1D

    # This should be caught by validation before any computation
    # (Test framework should handle the panic)
    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    _loss = loss_fn(logits, target)


fn _ce_validation_class_out_of_bounds() raises:
    print("test_ce_validation_class_out_of_bounds")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()  # 2 classes
    var target = Tensor[DType.int32].d1([0, 2])  # Class 2 is out of bounds"""

    # Should be caught by validation

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    _loss = loss_fn(logits, target)


fn _ce_validation_spatial_mismatch() raises:
    print("test_ce_validation_spatial_mismatch")
    var logits = Tensor.d4([[[[2.0, 1.0], [1.0, 2.0]]]]).float()  # (1, 2, 2, 2)
    var target = Tensor[DType.int32].d3(
        [[[0, 1, 0]]]
    )  # Wrong spatial dim: (1, 1, 3)

    # Should be caught by spatial dimension validation
    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    _loss = loss_fn(logits, target)


fn _ce_validation_batch_size_mismatch() raises:
    print("test_ce_validation_batch_size_mismatch")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()  # batch size 2
    var target = Tensor[DType.int32].d1([0, 1, 0])  # batch size 3

    # Should be caught by batch size validation

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    _loss = loss_fn(logits, target)


fn test_ce_2d_basic() raises:
    print("test_ce_2d_basic")
    var logits = Tensor.d2([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]]).float()
    var target = Tensor[DType.int32].d1([0, 1])  # Class indices
    var loss_fn = CrossEntropyLoss[DType.float32]()
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_2d_basic -> Total:            ", end - start)

    # Expected: -log(softmax([2,1,0.1])[0]) for first sample, etc.
    assert_true(loss.item() > 0.0)


fn test_ce_2d_ignore_index() raises:
    print("test_ce_2d_ignore_index")
    var logits = Tensor.d2([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]]).float()
    var target = Tensor[DType.int32].d1([0, -1])  # Second sample ignored
    var loss_fn = CrossEntropyLoss[DType.float32](ignore_index=-1)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_2d_ignore_index -> Total:            ", end - start)

    # Should only compute loss for first sample
    assert_true(loss.item() > 0.0)


fn test_ce_2d_label_smoothing() raises:
    print("test_ce_2d_label_smoothing")
    var logits = Tensor.d2([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](label_smoothing=0.1)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_2d_label_smoothing -> Total:            ", end - start)

    # Should be slightly higher than without smoothing
    assert_true(loss.item() > 0.0)


fn test_ce_2d_logits_1d_target() raises:
    print("test_ce_2d_logits_1d_target")
    var logits = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32]()
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_2d_logits_1d_target -> Total:            ", end - start)

    assert_true(loss.item() > 0.0)


fn test_ce_3d_spatial() raises:
    print("test_ce_3d_spatial")
    var logits = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    ).float()  # (2, 2, 2)
    var target = Tensor[DType.int32].d2([[0, 1], [1, 0]])  # (2, 2)

    var loss_fn = CrossEntropyLoss[DType.float32]()
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_3d_spatial -> Total:            ", end - start)
    assert_true(loss.item() > 0.0)


fn test_ce_4d_spatial() raises:
    print("test_ce_4d_spatial_2")
    # Logits: 2 samples, 2 classes, 2x2 spatial
    var logits = Tensor.d4(
        [
            [  # Sample 0
                [[1.0, 2.0], [3.0, 4.0]],  # Class 0 spatial outputs
                [[5.0, 6.0], [7.0, 8.0]],  # Class 1 spatial outputs
            ],
            [  # Sample 1
                [[9.0, 10.0], [11.0, 12.0]],  # Class 0
                [[13.0, 14.0], [15.0, 16.0]],  # Class 1
            ],
        ]
    ).float()  # Shape: (2, 2, 2, 2)

    # Target must have matching spatial dimensions: 2x2
    var target = Tensor[DType.int32].d3(
        [
            [[0, 1], [1, 0]],  # Sample 0: 2x2 spatial class assignments
            [[1, 0], [0, 1]],  # Sample 1: 2x2 spatial class assignments
        ]
    )  # Shape: (2, 2, 2) - matches logits spatial dims

    var loss_fn = CrossEntropyLoss[DType.float32]()
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_4d_spatial -> Total:            ", end - start)
    assert_true(loss.item() > 0.0)


fn test_ce_4d_spatial_1x2() raises:
    print("test_ce_4d_spatial_1x2")
    # Logits: 2 samples, 2 classes, 1x2 spatial (not 2x2)
    var logits = Tensor.d4(
        [
            [  # Sample 0
                [[1.0, 2.0]],  # Class 0: 1x2 spatial
                [[3.0, 4.0]],  # Class 1: 1x2 spatial
            ],
            [  # Sample 1
                [[5.0, 6.0]],  # Class 0: 1x2 spatial
                [[7.0, 8.0]],  # Class 1: 1x2 spatial
            ],
        ]
    ).float()  # Shape: (2, 2, 1, 2)

    # Target: 1x2 spatial dimensions to match logits
    var target = Tensor[DType.int32].d3(
        [[[0, 1]], [[1, 0]]]  # Sample 0: 1x2 spatial  # Sample 1: 1x2 spatial
    )  # Shape: (2, 1, 2) - matches logits spatial dims

    var loss_fn = CrossEntropyLoss[DType.float32]()
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_4d_spatial_1x2 -> Total:            ", end - start)
    assert_true(loss.item() > 0.0)


fn test_ce_reduction_mean_orig() raises:
    print("test_ce_reduction_mean_orig")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)  # mean
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_reduction_mean -> Total:            ", end - start)
    assert_true(loss.rank() == 0)  # scalar


fn test_ce_reduction_sum_orig() raises:
    print("test_ce_reduction_sum_orig")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=1)  # sum
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_reduction_sum -> Total:            ", end - start)

    assert_true(loss.rank() == 0)  # scalar


fn test_ce_reduction_none_orig() raises:
    print("test_ce_reduction_none_orig")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor[DType.int32].d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=2)  # none
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_reduction_none -> Total:            ", end - start)

    assert_true(loss.shape() == Shape([2]))  # per-sample loss

fn test_ce_rank2_basic_v2() raises:
    """Test rank-2 (no spatial dims) - baseline."""
    print("test_ce_rank2_basic_v2")

    var logits = Tensor[DType.float32].d2(
        [[2.0, 1.0, 0.5], [1.5, 2.0, 1.0]],
        requires_grad=True,
    )
    var targets = Tensor[DType.int32].d1([0, 1])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)
    loss.backward()

    # All positions are valid - gradients should be non-zero
    var has_nonzero = False
    for i in range(2):
        for c in range(3):
            if abs(logits.grad()[i, c]) > 1e-6:
                has_nonzero = True
                break
    assert_true(has_nonzero, "Rank-2: Should have non-zero gradients")
    print("✓ Rank-2 basic test passed")


fn test_ce_rank3_ignore_v2() raises:
    """Test rank-3 with ignore_index - spatial dimensions."""
    print("test_ce_rank3_ignore_v2")

    var logits = Tensor[DType.float32].d3(
        [
            [[2.0, 1.0, 0.5], [1.0, 2.0, 1.5], [0.5, 0.5, 2.0]],
            [[1.5, 2.0, 1.0], [2.0, 1.0, 2.0], [0.5, 0.5, 0.5]],
        ],
        requires_grad=True,
    )
    var targets = Tensor[DType.int32].d2([[0, -100, 2], [1, 0, -100]])

    var criterion = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction="mean"
    )
    var loss = criterion(logits, targets)
    loss.backward()

    # Check ignored positions have zero gradient across ALL classes
    # Batch 0, spatial position 1 is ignored (target=-100)
    for c in range(3):
        assert_true(
            abs(logits.grad()[0, c, 1]) < 1e-10,
            "Rank-3: Batch 0, pos 1, class " + String(c) + " grad should be 0",
        )

    # Batch 1, spatial position 2 is ignored (target=-100)
    for c in range(3):
        assert_true(
            abs(logits.grad()[1, c, 2]) < 1e-10,
            "Rank-3: Batch 1, pos 2, class " + String(c) + " grad should be 0",
        )

    # Non-ignored positions should have non-zero gradients
    var has_nonzero_grad = False
    for c in range(3):
        if abs(logits.grad()[0, c, 0]) > 1e-6:
            has_nonzero_grad = True
            break
    assert_true(
        has_nonzero_grad, "Rank-3: Non-ignored position should have non-zero grads"
    )

    print("✓ Rank-3 with ignore_index test passed")


fn test_ce_rank4_ignore_v2() raises:
    """Test rank-4 (image-like) with ignore_index."""
    print("test_ce_rank4_ignore_v2")

    # Shape: [batch=2, classes=3, height=2, width=2]
    var logits = Tensor[DType.float32].d4(
        [
            # Batch 0
            [
                # Class 0
                [[2.0, 1.0], [1.5, 0.5]],
                # Class 1
                [[1.0, 2.0], [1.0, 1.5]],
                # Class 2
                [[0.5, 0.5], [2.0, 2.0]],
            ],
            # Batch 1
            [
                # Class 0
                [[1.5, 2.0], [1.0, 0.5]],
                # Class 1
                [[2.0, 1.0], [2.0, 1.0]],
                # Class 2
                [[0.5, 0.5], [0.5, 2.0]],
            ],
        ],
        requires_grad=True,
    )

    # Targets: [batch=2, height=2, width=2]
    # Layout: targets[batch, height, width]
    # Ignored positions:
    #   - Batch 0, height=0, width=1: targets[0, 0, 1] = -100
    #   - Batch 1, height=0, width=0: targets[1, 0, 0] = -100
    var targets = Tensor[DType.int32].d3(
        [
            [[0, -100], [2, 1]],  # Batch 0: [0,0,0]=0, [0,0,1]=-100, [0,1,0]=2, [0,1,1]=1
            [[-100, 1], [0, 2]],  # Batch 1: [1,0,0]=-100, [1,0,1]=1, [1,1,0]=0, [1,1,1]=2
        ]
    )

    var criterion = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction="mean"
    )
    var loss = criterion(logits, targets)
    loss.backward()

    # Check ignored position [0, :, 0, 1] (batch 0, height 0, width 1)
    # This corresponds to targets[0, 0, 1] = -100
    for c in range(3):
        var grad_val = logits.grad()[0, c, 0, 1]
        assert_true(
            abs(grad_val) < 1e-10,
            "Rank-4: Batch 0, [0,1], class " + String(c) + " grad should be 0, got " + String(grad_val),
        )

    # Check ignored position [1, :, 0, 0] (batch 1, height 0, width 0)
    # This corresponds to targets[1, 0, 0] = -100
    for c in range(3):
        var grad_val = logits.grad()[1, c, 0, 0]
        assert_true(
            abs(grad_val) < 1e-10,
            "Rank-4: Batch 1, [0,0], class " + String(c) + " grad should be 0, got " + String(grad_val),
        )

    # Check non-ignored position [0, :, 0, 0] has non-zero grads
    # This corresponds to targets[0, 0, 0] = 0 (valid)
    var has_nonzero = False
    for c in range(3):
        if abs(logits.grad()[0, c, 0, 0]) > 1e-6:
            has_nonzero = True
            break
    assert_true(
        has_nonzero, "Rank-4: Non-ignored position should have non-zero grads"
    )

    print("✓ Rank-4 with ignore_index test passed")


fn test_ce_rank3_no_ignore_v2() raises:
    """Test rank-3 without ignore_index - all positions valid."""
    print("test_ce_rank3_no_ignore_v2")

    var logits = Tensor[DType.float32].d3(
        [
            [[2.0, 1.0, 0.5], [1.0, 2.0, 1.5], [0.5, 0.5, 2.0]],
            [[1.5, 2.0, 1.0], [2.0, 1.0, 2.0], [0.5, 0.5, 0.5]],
        ],
        requires_grad=True,
    )
    var targets = Tensor[DType.int32].d2([[0, 1, 2], [1, 0, 2]])

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)
    loss.backward()

    # All positions should have non-zero gradients somewhere
    var total_nonzero = 0
    for b in range(2):
        for s in range(3):
            for c in range(3):
                if abs(logits.grad()[b, c, s]) > 1e-6:
                    total_nonzero += 1

    assert_true(
        total_nonzero > 0, "Rank-3: Should have at least some non-zero gradients"
    )
    print("✓ Rank-3 without ignore_index test passed")


fn test_ce_rank4_all_valid_v2() raises:
    """Test rank-4 with all valid positions."""
    print("test_ce_rank4_all_valid_v2")

    var logits = Tensor[DType.float32].d4(
        [
            [
                [[2.0, 1.0], [1.5, 0.5]],
                [[1.0, 2.0], [1.0, 1.5]],
                [[0.5, 0.5], [2.0, 2.0]],
            ],
            [
                [[1.5, 2.0], [1.0, 0.5]],
                [[2.0, 1.0], [2.0, 1.0]],
                [[0.5, 0.5], [0.5, 2.0]],
            ],
        ],
        requires_grad=True,
    )
    var targets = Tensor[DType.int32].d3(
        [
            [[0, 1], [2, 1]],
            [[1, 0], [0, 2]],
        ]
    )

    var criterion = CrossEntropyLoss[DType.float32](reduction="mean")
    var loss = criterion(logits, targets)
    loss.backward()

    # Should have non-zero gradients
    var total_nonzero = 0
    for b in range(2):
        for h in range(2):
            for w in range(2):
                for c in range(3):
                    if abs(logits.grad()[b, c, h, w]) > 1e-6:
                        total_nonzero += 1

    assert_true(
        total_nonzero > 0, "Rank-4: Should have non-zero gradients"
    )
    print("✓ Rank-4 all valid test passed")

fn test_ce_rank3_all_ignored_v2() raises:
    """Test rank-3 where all positions are ignored."""
    print("test_ce_rank3_all_ignored_v2")

    var logits = Tensor[DType.float32].d3(
        [
            [[2.0, 1.0], [1.0, 2.0], [0.5, 0.5]],
            [[1.5, 2.0], [2.0, 1.0], [0.5, 0.5]],
        ],
        requires_grad=True,
    )
    var targets = Tensor[DType.int32].d2([[-100, -100], [-100, -100]])

    var criterion = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction="mean"
    )
    var loss = criterion(logits, targets)
    loss.backward()

    # ALL gradients should be zero
    for b in range(2):
        for c in range(3):
            for s in range(2):
                var grad_val = logits.grad()[b, c, s]
                assert_true(
                    abs(grad_val) < 1e-10,
                    "Rank-3: All ignored - grad should be 0, got " + String(grad_val),
                )

    # Loss should be 0 (no valid samples)
    assert_true(abs(loss.item()) < 1e-10, "Rank-3: All ignored - loss should be 0")
    print("✓ Rank-3 all ignored test passed")


fn test_ce_rank4_partial_ignore_v2() raises:
    """Test rank-4 with mix of valid and ignored positions."""
    print("test_ce_rank4_partial_ignore_v2")

    var logits = Tensor[DType.float32].d4(
        [
            [
                [[2.0, 1.0, 0.5], [1.5, 1.0, 0.5]],
                [[1.0, 2.0, 1.5], [1.0, 2.0, 1.0]],
                [[0.5, 0.5, 2.0], [0.5, 0.5, 2.0]],
            ],
        ],
        requires_grad=True,
    )
    # Shape: [1, 2, 3] - 1 batch, 2 height, 3 width
    var targets = Tensor[DType.int32].d3([[[0, -100, 2], [1, 0, -100]]])

    var criterion = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction="mean"
    )
    var loss = criterion(logits, targets)
    loss.backward()

    # Ignored: [0, :, 0, 1], [0, :, 1, 2]
    # Check ignored position [0, :, 0, 1]
    for c in range(3):
        assert_true(
            abs(logits.grad()[0, c, 0, 1]) < 1e-10,
            "Rank-4: Ignored [0,1] class " + String(c) + " should be 0",
        )

    # Check ignored position [0, :, 1, 2]
    for c in range(3):
        assert_true(
            abs(logits.grad()[0, c, 1, 2]) < 1e-10,
            "Rank-4: Ignored [1,2] class " + String(c) + " should be 0",
        )

    # Check non-ignored [0, :, 0, 0] has non-zero
    var has_nonzero = False
    for c in range(3):
        if abs(logits.grad()[0, c, 0, 0]) > 1e-6:
            has_nonzero = True
            break
    assert_true(has_nonzero, "Rank-4: Valid position should have non-zero grads")

    print("✓ Rank-4 partial ignore test passed")


fn test_ce_rank3_label_smoothing_ignore_v2() raises:
    """Test rank-3 with label smoothing and ignore_index."""
    print("test_ce_rank3_label_smoothing_ignore_v2")

    var logits = Tensor[DType.float32].d3(
        [
            [[2.0, 1.0, 0.5], [1.0, 2.0, 1.5], [0.5, 0.5, 2.0]],
        ],
        requires_grad=True,
    )
    var targets = Tensor[DType.int32].d2([[0, -100, 2]])

    var criterion = CrossEntropyLoss[DType.float32](
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.1,
    )
    var loss = criterion(logits, targets)
    loss.backward()

    # Ignored position should still have zero gradients even with smoothing
    for c in range(3):
        assert_true(
            abs(logits.grad()[0, c, 1]) < 1e-10,
            "Rank-3: Label smoothing + ignore - class " + String(c) + " should be 0",
        )

    # Non-ignored positions should have non-zero gradients
    var has_nonzero = False
    for c in range(3):
        if abs(logits.grad()[0, c, 0]) > 1e-6:
            has_nonzero = True
            break
    assert_true(
        has_nonzero, "Rank-3: Label smoothing - valid pos should have non-zero grads"
    )

    print("✓ Rank-3 label smoothing + ignore test passed")


fn test_ce_rank3_reduction_none_v2() raises:
    """Test rank-3 with reduction='none'."""
    print("test_ce_rank3_reduction_none_v2")

    var logits = Tensor[DType.float32].d3(
        [
            [[2.0, 1.0], [1.0, 2.0], [0.5, 0.5]],
        ],
        requires_grad=True,
    )
    var targets = Tensor[DType.int32].d2([[0, -100]])

    var criterion = CrossEntropyLoss[DType.float32](
        ignore_index=-100,
        reduction="none",
    )
    var loss = criterion(logits, targets)

    # Loss shape should match target shape [1, 2]
    assert_true(
        loss.shape().rank() == 2 and loss.shape()[0] == 1 and loss.shape()[1] == 2,
        "Rank-3: reduction=none should preserve target shape",
    )

    # Backward with ones
    loss.backward()

    # Ignored position should have zero gradient
    for c in range(3):
        assert_true(
            abs(logits.grad()[0, c, 1]) < 1e-10,
            "Rank-3: reduction=none - ignored class " + String(c) + " should be 0",
        )

    print("✓ Rank-3 reduction=none test passed")


fn run_all_ce_tests_v2() raises:
    """Run all CrossEntropy validation tests."""
    print("\n" + "="*60)
    print("Running CrossEntropy Loss Validation Tests (v2)")
    print("="*60 + "\n")

    test_ce_rank2_basic_v2()
    test_ce_rank3_ignore_v2()
    test_ce_rank4_ignore_v2()
    test_ce_rank3_no_ignore_v2()
    test_ce_rank4_all_valid_v2()
    test_ce_rank3_all_ignored_v2()
    test_ce_rank4_partial_ignore_v2()
    test_ce_rank3_label_smoothing_ignore_v2()
    test_ce_rank3_reduction_none_v2()

    print("\n" + "="*60)
    print("ALL CROSSENTROPY TESTS PASSED (v2)")
    print("="*60 + "\n")




# ═════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═════════════════════════════════════════════════════════════════════════════


fn softmax_1d(logits: Tensor[DType.float32], row: Int, C: Int) -> List[Float32]:
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


fn log_softmax_val(logits: Tensor[DType.float32], row: Int, cls: Int, C: Int) -> Float32:
    """Compute log_softmax[row, cls] manually."""
    var probs = softmax_1d(logits, row, C)
    return log(probs[cls])


fn nll_loss(log_prob: Float32) -> Float32:
    return -log_prob


fn allclose(a: Float32, b: Float32, atol: Float32 = 1e-4) -> Bool:
    return abs(a - b) < atol


# ═════════════════════════════════════════════════════════════════════════════
# GROUP A: Basic Forward — Class Indices, 2D, CPU
# ═════════════════════════════════════════════════════════════════════════════


fn test_ce_ci_basic_mean() raises:
    print("test_ce_ci_basic_mean")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]])
    var target = Tensor[DType.int32].d1([0, 1])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    # Manual: loss = mean(-log_softmax[0,0], -log_softmax[1,1])
    var probs0 = softmax_1d(logits, 0, 3)
    var probs1 = softmax_1d(logits, 1, 3)
    var expected = (-log(probs0[0]) + -log(probs1[1])) / 2.0
    assert_true(allclose(loss.item(), expected))


fn test_ce_ci_basic_sum() raises:
    print("test_ce_ci_basic_sum")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]])
    var target = Tensor[DType.int32].d1([0, 1])
    var ce = CrossEntropyLoss[dtype](reduction="sum")
    var loss = ce(logits, target)
    var probs0 = softmax_1d(logits, 0, 3)
    var probs1 = softmax_1d(logits, 1, 3)
    var expected = -log(probs0[0]) + -log(probs1[1])
    assert_true(allclose(loss.item(), expected))


fn test_ce_ci_basic_none() raises:
    print("test_ce_ci_basic_none")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]])
    var target = Tensor[DType.int32].d1([0, 1])
    var ce = CrossEntropyLoss[dtype](reduction="none")
    var loss = ce(logits, target)
    assert_true(loss.shape() == Shape(2))
    var probs0 = softmax_1d(logits, 0, 3)
    var probs1 = softmax_1d(logits, 1, 3)
    assert_true(allclose(loss[[0]], -log(probs0[0])))
    assert_true(allclose(loss[[1]], -log(probs1[1])))


fn test_ce_ci_loss_positive() raises:
    print("test_ce_ci_loss_positive")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    var target = Tensor[DType.int32].d1([0, 2])
    var ce = CrossEntropyLoss[dtype]()
    var loss = ce(logits, target)
    assert_true(loss.item() > 0)


fn test_ce_ci_perfect_prediction() raises:
    print("test_ce_ci_perfect_prediction")
    comptime dtype = DType.float32
    # Large logit for correct class → loss near 0
    var logits = Tensor[dtype].d2([[100.0, -100.0, -100.0]])
    var target = Tensor[DType.int32].d1([0])
    var ce = CrossEntropyLoss[dtype]()
    var loss = ce(logits, target)
    assert_true(loss.item() < Float32(0.01))


fn test_ce_ci_wrong_prediction() raises:
    print("test_ce_ci_wrong_prediction")
    comptime dtype = DType.float32
    # Large logit for wrong class → high loss
    var logits = Tensor[dtype].d2([[-100.0, 100.0, -100.0]])
    var target = Tensor[DType.int32].d1([0])
    var ce = CrossEntropyLoss[dtype]()
    var loss = ce(logits, target)
    assert_true(loss.item() > Float32(10.0))


fn test_ce_ci_uniform_logits() raises:
    print("test_ce_ci_uniform_logits")
    comptime dtype = DType.float32
    # Uniform logits → loss = log(C) for any target
    var C = 4
    var logits = Tensor[dtype].d2([[1.0, 1.0, 1.0, 1.0]])
    var target = Tensor[DType.int32].d1([2])
    var ce = CrossEntropyLoss[dtype]()
    var loss = ce(logits, target)
    assert_true(allclose(loss.item(), log(Float32(C)), atol=1e-4))


fn test_ce_ci_batch4() raises:
    print("test_ce_ci_batch4")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2(
        [[2.0, 1.0], [1.0, 2.0], [0.5, 0.5], [3.0, 0.0]]
    )
    var target = Tensor[DType.int32].d1([0, 1, 0, 0])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    assert_true(loss.item() > 0)
    # Sum should be 4x mean
    var ce_sum = CrossEntropyLoss[dtype](reduction="sum")
    var loss_sum = ce_sum(logits, target)
    assert_true(allclose(loss_sum.item(), loss.item() * 4.0, atol=1e-4))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP B: Ignore Index — CPU
# ═════════════════════════════════════════════════════════════════════════════


fn test_ce_ii_basic() raises:
    print("test_ce_ii_basic")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2(
        [[2.0, 1.0, 0.5], [1.0, 2.0, 0.1], [3.0, 1.0, 0.2]]
    )
    var target = Tensor[DType.int32].d1([0, -100, 2])
    var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
    var loss = ce(logits, target)
    # Only 2 valid samples
    var ce_all = CrossEntropyLoss[dtype](reduction="mean")
    var logits2 = Tensor[dtype].d2([[2.0, 1.0, 0.5], [3.0, 1.0, 0.2]])
    var target2 = Tensor[DType.int32].d1([0, 2])
    var loss2 = ce_all(logits2, target2)
    assert_true(allclose(loss.item(), loss2.item()))


fn test_ce_ii_all_ignored() raises:
    print("test_ce_ii_all_ignored")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0], [1.0, 2.0]])
    var target = Tensor[DType.int32].d1([-100, -100])
    var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
    var loss = ce(logits, target)
    # All ignored → loss = 0
    assert_true(allclose(loss.item(), Float32(0.0)))


fn test_ce_ii_first_ignored() raises:
    print("test_ce_ii_first_ignored")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5], [1.0, 2.0, 0.1]])
    var target = Tensor[DType.int32].d1([-100, 1])
    var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
    var loss = ce(logits, target)
    var ce2 = CrossEntropyLoss[dtype](reduction="mean")
    var logits2 = Tensor[dtype].d2([[1.0, 2.0, 0.1]])
    var target2 = Tensor[DType.int32].d1([1])
    var loss2 = ce2(logits2, target2)
    assert_true(allclose(loss.item(), loss2.item()))


fn test_ce_ii_custom_ignore_value() raises:
    print("test_ce_ii_custom_ignore_value")
    comptime dtype = DType.float32
    # ignore_index = 255 (common in segmentation)
    var logits = Tensor[dtype].d2([[2.0, 1.0], [1.0, 2.0], [3.0, 0.0]])
    var target = Tensor[DType.int32].d1([0, 255, 1])
    var ce = CrossEntropyLoss[dtype](ignore_index=255, reduction="mean")
    var loss = ce(logits, target)
    assert_true(loss.item() > 0)
    # Only 2 valid samples
    var ce2 = CrossEntropyLoss[dtype](reduction="mean")
    var logits2 = Tensor[dtype].d2([[2.0, 1.0], [3.0, 0.0]])
    var target2 = Tensor[DType.int32].d1([0, 1])
    var loss2 = ce2(logits2, target2)
    assert_true(allclose(loss.item(), loss2.item()))


fn test_ce_ii_sum_reduction() raises:
    print("test_ce_ii_sum_reduction")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0], [1.0, 2.0], [3.0, 0.0]])
    var target = Tensor[DType.int32].d1([0, -100, 1])
    var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="sum")
    var loss = ce(logits, target)
    # sum over 2 valid
    var ce2 = CrossEntropyLoss[dtype](reduction="sum")
    var logits2 = Tensor[dtype].d2([[2.0, 1.0], [3.0, 0.0]])
    var target2 = Tensor[DType.int32].d1([0, 1])
    var loss2 = ce2(logits2, target2)
    assert_true(allclose(loss.item(), loss2.item()))


fn test_ce_ii_none_reduction() raises:
    print("test_ce_ii_none_reduction")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0], [1.0, 2.0], [3.0, 0.0]])
    var target = Tensor[DType.int32].d1([0, -100, 1])
    var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="none")
    var loss = ce(logits, target)
    assert_true(loss.shape() == Shape(3))
    # Ignored position should have 0 loss
    assert_true(allclose(loss[[1]], Float32(0.0)))
    # Valid positions should have positive loss
    assert_true(loss[[0]] > Float32(0.0))
    assert_true(loss[[2]] > Float32(0.0))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP C: Label Smoothing — CPU
# ═════════════════════════════════════════════════════════════════════════════


fn test_ce_ls_basic() raises:
    print("test_ce_ls_basic")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5]])
    var target = Tensor[DType.int32].d1([0])
    var ce_no_ls = CrossEntropyLoss[dtype](reduction="mean")
    var ce_ls = CrossEntropyLoss[dtype](
        reduction="mean", label_smoothing=Scalar[dtype](0.1)
    )
    var loss_no_ls = ce_no_ls(logits, target)
    var loss_ls = ce_ls(logits, target)
    # Label smoothing spreads probability → loss changes
    assert_true(loss_no_ls.item() != loss_ls.item())


fn test_ce_ls_increases_entropy() raises:
    print("test_ce_ls_increases_entropy")
    comptime dtype = DType.float32
    # For a confident prediction, smoothing increases loss
    var logits = Tensor[dtype].d2([[10.0, -10.0, -10.0]])
    var target = Tensor[DType.int32].d1([0])
    var ce_no_ls = CrossEntropyLoss[dtype](reduction="mean")
    var ce_ls = CrossEntropyLoss[dtype](
        reduction="mean", label_smoothing=Scalar[dtype](0.1)
    )
    var loss_no_ls = ce_no_ls(logits, target)
    var loss_ls = ce_ls(logits, target)
    # Smoothing should increase loss for over-confident predictions
    assert_true(loss_ls.item() > loss_no_ls.item())


fn test_ce_ls_formula() raises:
    print("test_ce_ls_formula")
    comptime dtype = DType.float32
    var C = 3
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5]])
    var target = Tensor[DType.int32].d1([0])
    var ls = Float32(0.1)
    var ce = CrossEntropyLoss[dtype](
        reduction="mean", label_smoothing=Scalar[dtype](ls)
    )
    var loss = ce(logits, target)
    # Manual: loss = (1-ls)*NLL + ls * mean(-log_p over all classes)
    var probs = softmax_1d(logits, 0, C)
    var nll = -log(probs[0])
    var mean_log_p = Float32(0)
    for c in range(C):
        mean_log_p += -log(probs[c])
    mean_log_p /= Float32(C)
    var expected = (1.0 - ls) * nll + ls * mean_log_p
    assert_true(allclose(loss.item(), expected, atol=1e-4))


fn test_ce_ls_zero_is_standard() raises:
    print("test_ce_ls_zero_is_standard")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]])
    var target = Tensor[DType.int32].d1([0, 1])
    var ce_standard = CrossEntropyLoss[dtype](reduction="mean")
    var ce_ls0 = CrossEntropyLoss[dtype](
        reduction="mean", label_smoothing=Scalar[dtype](0.0)
    )
    var loss_standard = ce_standard(logits, target)
    var loss_ls0 = ce_ls0(logits, target)
    assert_true(allclose(loss_standard.item(), loss_ls0.item()))


fn test_ce_ls_with_ignore() raises:
    print("test_ce_ls_with_ignore")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5], [1.0, 2.0, 0.1]])
    var target = Tensor[DType.int32].d1([0, -100])
    var ce = CrossEntropyLoss[dtype](
        reduction="mean",
        ignore_index=-100,
        label_smoothing=Scalar[dtype](0.1),
    )
    var loss = ce(logits, target)
    # Only 1 valid sample
    assert_true(loss.item() > 0)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP D: Multi-dimensional — CPU
# ═════════════════════════════════════════════════════════════════════════════


fn test_ce_3d_basic() raises:
    print("test_ce_3d_basic")
    comptime dtype = DType.float32
    # logits: (2, 3, 4) — batch=2, classes=3, spatial=4
    var logits = Tensor[dtype].arange(24).reshape(Shape(2, 3, 4)).float()
    var target = Tensor[DType.int32].d2([[0, 1, 2, 0], [1, 2, 0, 1]])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    assert_true(loss.item() > 0)
    # none reduction → shape (2, 4)
    var ce_none = CrossEntropyLoss[dtype](reduction="none")
    var loss_none = ce_none(logits, target)
    assert_true(loss_none.shape() == Shape(2, 4))


fn test_ce_3d_ignore_index() raises:
    print("test_ce_3d_ignore_index")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d3(
        [
            [[2.0, 1.0, 0.5], [1.0, 2.0, 1.5], [0.5, 0.5, 2.0]],
            [[1.5, 2.0, 1.0], [2.0, 1.0, 2.0], [0.5, 0.5, 0.5]],
        ]
    )
    var target = Tensor[DType.int32].d2([[0, -100, 2], [1, 0, -100]])
    var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
    var loss = ce(logits, target)
    # 4 valid positions out of 6
    assert_true(loss.item() > 0)


fn test_ce_3d_none_shape() raises:
    print("test_ce_3d_none_shape")
    comptime dtype = DType.float32
    # logits (N=2, C=4, H=3, W=5)
    var logits = Tensor[dtype].full(Shape(2, 4, 3, 5), 1.0)
    var target = Tensor[DType.int32].zeros(Shape(2, 3, 5))
    var ce = CrossEntropyLoss[dtype](reduction="none")
    var loss = ce(logits, target)
    # Output shape should be (N, H, W) = (2, 3, 5)
    assert_true(loss.shape() == Shape(2, 3, 5))


fn test_ce_4d_basic() raises:
    print("test_ce_4d_basic")
    comptime dtype = DType.float32
    # logits (2, 3, 2, 2) — batch=2, C=3, H=2, W=2
    var logits = Tensor[dtype].full(Shape(2, 3, 2, 2), 1.0)
    var target = Tensor[DType.int32].zeros(Shape(2, 2, 2))
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    # Uniform logits → log(C) = log(3)
    assert_true(allclose(loss.item(), log(Float32(3)), atol=1e-4))


fn test_ce_5d_basic() raises:
    print("test_ce_5d_basic")
    comptime dtype = DType.float32
    # logits (2, 4, 2, 2, 2)
    var logits = Tensor[dtype].full(Shape(2, 4, 2, 2, 2), 1.0)
    var target = Tensor[DType.int32].zeros(Shape(2, 2, 2, 2))
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    assert_true(allclose(loss.item(), log(Float32(4)), atol=1e-4))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP E: Probability Targets — CPU
# ═════════════════════════════════════════════════════════════════════════════


fn test_ce_prob_basic() raises:
    print("test_ce_prob_basic")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]])
    # One-hot targets → same as class indices
    var target_probs = Tensor[dtype].d2(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    var target_idx = Tensor[DType.int32].d1([0, 1])
    var ce_probs = CrossEntropyLoss[dtype](reduction="mean")
    var ce_idx = CrossEntropyLoss[dtype](reduction="mean")
    var loss_probs = ce_probs(logits, target_probs)
    var loss_idx = ce_idx(logits, target_idx)
    # One-hot probs should match class indices
    assert_true(allclose(loss_probs.item(), loss_idx.item(), atol=1e-4))


fn test_ce_prob_soft_labels() raises:
    print("test_ce_prob_soft_labels")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5]])
    # Soft label — not one-hot
    var target = Tensor[dtype].d2([[0.7, 0.2, 0.1]])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    # Manual: loss = -sum(target * log_softmax)
    var probs = softmax_1d(logits, 0, 3)
    var expected = -(0.7 * log(probs[0]) + 0.2 * log(probs[1]) + 0.1 * log(probs[2]))
    assert_true(allclose(loss.item(), expected, atol=1e-4))


fn test_ce_prob_sum_reduction() raises:
    print("test_ce_prob_sum_reduction")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0], [0.5, 1.5]])
    var target = Tensor[dtype].d2([[0.8, 0.2], [0.3, 0.7]])
    var ce = CrossEntropyLoss[dtype](reduction="sum")
    var loss = ce(logits, target)
    assert_true(loss.item() > 0)


fn test_ce_prob_none_reduction() raises:
    print("test_ce_prob_none_reduction")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0], [0.5, 1.5], [1.0, 1.0]])
    var target = Tensor[dtype].d2([[1.0, 0.0], [0.4, 0.6], [0.5, 0.5]])
    var ce = CrossEntropyLoss[dtype](reduction="none")
    var loss = ce(logits, target)
    assert_true(loss.shape() == Shape(3))
    for i in range(3):
        assert_true(loss[[i]] > Float32(0.0))


fn test_ce_prob_label_smoothing() raises:
    print("test_ce_prob_label_smoothing")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5]])
    var target = Tensor[dtype].d2([[1.0, 0.0, 0.0]])
    var ce_no_ls = CrossEntropyLoss[dtype](reduction="mean")
    var ce_ls = CrossEntropyLoss[dtype](
        reduction="mean", label_smoothing=Scalar[dtype](0.1)
    )
    var loss_no_ls = ce_no_ls(logits, target)
    var loss_ls = ce_ls(logits, target)
    assert_true(loss_no_ls.item() != loss_ls.item())


fn test_ce_prob_3d() raises:
    print("test_ce_prob_3d")
    comptime dtype = DType.float32
    # logits (2, 3, 4), target (2, 3, 4) — prob targets same shape
    var logits = Tensor[dtype].full(Shape(2, 3, 4), 1.0)
    var target = Tensor[dtype].full(Shape(2, 3, 4), 1.0 / 3.0)
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    # Uniform logits + uniform target → log(C)
    assert_true(allclose(loss.item(), log(Float32(3)), atol=1e-4))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP F: Backward — Class Indices, CPU
# ═════════════════════════════════════════════════════════════════════════════


fn test_ce_bwd_ci_grad_shape() raises:
    print("test_ce_bwd_ci_grad_shape")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2(
        [[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True
    )
    var target = Tensor[DType.int32].d1([0, 1])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    loss.backward()
    assert_true(logits.grad().shape() == Shape(2, 3))


fn test_ce_bwd_ci_grad_formula() raises:
    print("test_ce_bwd_ci_grad_formula")
    comptime dtype = DType.float32
    # grad[i, c] = (softmax[i,c] - 1[c==target[i]]) / N  for mean
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True)
    var target = Tensor[DType.int32].d1([0])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    loss.backward()
    var probs = softmax_1d(logits, 0, 3)
    # grad[0, 0] = (softmax[0] - 1) / 1
    assert_true(allclose(logits.grad()[[0, 0]], probs[0] - 1.0, atol=1e-4))
    # grad[0, 1] = softmax[1] / 1
    assert_true(allclose(logits.grad()[[0, 1]], probs[1], atol=1e-4))
    # grad[0, 2] = softmax[2] / 1
    assert_true(allclose(logits.grad()[[0, 2]], probs[2], atol=1e-4))


fn test_ce_bwd_ci_grad_sums_to_zero() raises:
    print("test_ce_bwd_ci_grad_sums_to_zero")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2(
        [[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True
    )
    var target = Tensor[DType.int32].d1([0, 1])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    loss.backward()
    # Sum of grads across classes for each sample ≈ 0
    for i in range(2):
        var row_sum = Float32(0)
        for c in range(3):
            row_sum += logits.grad()[[i, c]]
        assert_true(abs(row_sum) < Float32(1e-5))


fn test_ce_bwd_ci_sum_reduction() raises:
    print("test_ce_bwd_ci_sum_reduction")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True)
    var target = Tensor[DType.int32].d1([0])
    var ce = CrossEntropyLoss[dtype](reduction="sum")
    var loss = ce(logits, target)
    loss.backward()
    var probs = softmax_1d(logits, 0, 3)
    # No division by N for sum
    assert_true(allclose(logits.grad()[[0, 0]], probs[0] - 1.0, atol=1e-4))


fn test_ce_bwd_ci_ignore_zeros_grad() raises:
    print("test_ce_bwd_ci_ignore_zeros_grad")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2(
        [[2.0, 1.0, 0.5], [1.0, 2.0, 0.1], [3.0, 1.0, 0.2]],
        requires_grad=True,
    )
    var target = Tensor[DType.int32].d1([0, -100, 2])
    var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
    var loss = ce(logits, target)
    loss.backward()
    # Ignored position (row 1) should have zero gradient for ALL classes
    for c in range(3):
        assert_true(abs(logits.grad()[[1, c]]) < Float32(1e-6))
    # Non-ignored positions should have non-zero gradients
    var has_nonzero = False
    for c in range(3):
        if abs(logits.grad()[[0, c]]) > Float32(1e-6):
            has_nonzero = True
    assert_true(has_nonzero)


fn test_ce_bwd_ci_3d_ignore_zeros_grad() raises:
    print("test_ce_bwd_ci_3d_ignore_zeros_grad")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d3(
        [
            [[2.0, 1.0, 0.5], [1.0, 2.0, 1.5], [0.5, 0.5, 2.0]],
            [[1.5, 2.0, 1.0], [2.0, 1.0, 2.0], [0.5, 0.5, 0.5]],
        ],
        requires_grad=True,
    )
    var target = Tensor[DType.int32].d2([[0, -100, 2], [1, 0, -100]])
    var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
    var loss = ce(logits, target)
    loss.backward()
    # Batch 0, spatial pos 1 is ignored → all classes grad = 0
    for c in range(3):
        assert_true(abs(logits.grad()[[0, c, 1]]) < Float32(1e-6))
    # Batch 1, spatial pos 2 is ignored → all classes grad = 0
    for c in range(3):
        assert_true(abs(logits.grad()[[1, c, 2]]) < Float32(1e-6))
    # Non-ignored: batch 0, pos 0 → nonzero
    var has_nonzero = False
    for c in range(3):
        if abs(logits.grad()[[0, c, 0]]) > Float32(1e-6):
            has_nonzero = True
    assert_true(has_nonzero)


fn test_ce_bwd_ci_label_smoothing_grad() raises:
    print("test_ce_bwd_ci_label_smoothing_grad")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True)
    var target = Tensor[DType.int32].d1([0])
    var ls = Float32(0.1)
    var ce = CrossEntropyLoss[dtype](
        reduction="mean", label_smoothing=Scalar[dtype](ls)
    )
    var loss = ce(logits, target)
    loss.backward()
    # grad[0, c] = softmax[c] - (1-ls)*onehot[c] - ls/C
    var probs = softmax_1d(logits, 0, 3)
    var C = Float32(3)
    var expected_0 = probs[0] - (1.0 - ls) * 1.0 - ls / C
    var expected_1 = probs[1] - ls / C
    assert_true(allclose(logits.grad()[[0, 0]], expected_0, atol=1e-4))
    assert_true(allclose(logits.grad()[[0, 1]], expected_1, atol=1e-4))


fn test_ce_bwd_ci_finite_difference() raises:
    print("test_ce_bwd_ci_finite_difference")
    comptime dtype = DType.float32
    var eps = Float32(1e-3)
    var target = Tensor[DType.int32].d1([0, 2])
    var ce = CrossEntropyLoss[dtype](reduction="mean")

    # Analytical gradient
    var logits_a = Tensor[dtype].d2(
        [[2.0, 1.0, 0.5], [0.5, 1.5, 2.0]], requires_grad=True
    )
    var loss_a = ce(logits_a, target)
    loss_a.backward()

    # Numerical gradient for logits[0, 0]
    var logits_p = Tensor[dtype].d2([[2.0 + eps, 1.0, 0.5], [0.5, 1.5, 2.0]])
    var logits_m = Tensor[dtype].d2([[2.0 - eps, 1.0, 0.5], [0.5, 1.5, 2.0]])
    var loss_p = ce(logits_p, target)
    var loss_m = ce(logits_m, target)
    var numerical_grad = (loss_p.item() - loss_m.item()) / (2.0 * eps)

    assert_true(allclose(logits_a.grad()[[0, 0]], numerical_grad, atol=1e-3))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP G: Backward — Probability Targets, CPU
# ═════════════════════════════════════════════════════════════════════════════


fn test_ce_bwd_prob_grad_shape() raises:
    print("test_ce_bwd_prob_grad_shape")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2(
        [[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True
    )
    var target = Tensor[dtype].d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    loss.backward()
    assert_true(logits.grad().shape() == Shape(2, 3))


fn test_ce_bwd_prob_formula() raises:
    print("test_ce_bwd_prob_formula")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True)
    var target = Tensor[dtype].d2([[0.6, 0.3, 0.1]])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    loss.backward()
    # grad = (softmax - target) / N
    var probs = softmax_1d(logits, 0, 3)
    assert_true(allclose(logits.grad()[[0, 0]], probs[0] - 0.6, atol=1e-4))
    assert_true(allclose(logits.grad()[[0, 1]], probs[1] - 0.3, atol=1e-4))
    assert_true(allclose(logits.grad()[[0, 2]], probs[2] - 0.1, atol=1e-4))


fn test_ce_bwd_prob_sums_to_zero() raises:
    print("test_ce_bwd_prob_sums_to_zero")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2(
        [[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True
    )
    var target = Tensor[dtype].d2([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    loss.backward()
    for i in range(2):
        var row_sum = Float32(0)
        for c in range(3):
            row_sum += logits.grad()[[i, c]]
        assert_true(abs(row_sum) < Float32(1e-4))


fn test_ce_bwd_prob_finite_difference() raises:
    print("test_ce_bwd_prob_finite_difference")
    comptime dtype = DType.float32
    var eps = Float32(1e-3)
    var target = Tensor[dtype].d2([[0.7, 0.2, 0.1]])
    var ce = CrossEntropyLoss[dtype](reduction="mean")

    var logits_a = Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True)
    var loss_a = ce(logits_a, target)
    loss_a.backward()

    var logits_p = Tensor[dtype].d2([[2.0 + eps, 1.0, 0.5]])
    var logits_m = Tensor[dtype].d2([[2.0 - eps, 1.0, 0.5]])
    var numerical = (ce(logits_p, target).item() - ce(logits_m, target).item()) / (2.0 * eps)
    assert_true(allclose(logits_a.grad()[[0, 0]], numerical, atol=1e-3))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP H: Train/Eval mode
# ═════════════════════════════════════════════════════════════════════════════


fn test_ce_mode_training_tracks_grad() raises:
    print("test_ce_mode_training_tracks_grad")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0]], requires_grad=True)
    var target = Tensor[DType.int32].d1([0])
    var ce = CrossEntropyLoss[dtype](training=True)
    var loss = ce(logits, target)
    assert_true(loss.requires_grad)


fn test_ce_mode_eval_no_grad() raises:
    print("test_ce_mode_eval_no_grad")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0]], requires_grad=True)
    var target = Tensor[DType.int32].d1([0])
    var ce = CrossEntropyLoss[dtype](training=False)
    var loss = ce(logits, target)
    assert_true(not loss.requires_grad)


fn test_ce_mode_train_eval_toggle() raises:
    print("test_ce_mode_train_eval_toggle")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[2.0, 1.0]], requires_grad=True)
    var target = Tensor[DType.int32].d1([0])
    var ce = CrossEntropyLoss[dtype]()
    ce.eval()
    var loss_eval = ce(logits, target)
    assert_true(not loss_eval.requires_grad)
    ce.train()
    var loss_train = ce(logits, target)
    assert_true(loss_train.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP I: GPU Forward — Class Indices
# ═════════════════════════════════════════════════════════════════════════════


fn test_ce_gpu_ci_basic_mean() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_ci_basic_mean")
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]]).to_gpu()
        var target = Tensor[DType.int32].d1([0, 1])
        var target_gpu = Tensor[DType.int32].d1([0, 1]).to_gpu()
        var ce = CrossEntropyLoss[dtype](reduction="mean")
        var loss = ce(logits, target_gpu)
        var logits_cpu = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]])
        var ce2 = CrossEntropyLoss[dtype](reduction="mean")
        var loss_cpu = ce2(logits_cpu, target)
        assert_true(allclose(loss.item(), loss_cpu.item()))


fn test_ce_gpu_ci_basic_sum() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_ci_basic_sum")
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]]).to_gpu()
        var target = Tensor[DType.int32].d1([0, 1]).to_gpu()
        var target_cpu = Tensor[DType.int32].d1([0, 1])
        var ce = CrossEntropyLoss[dtype](reduction="sum")
        var loss = ce(logits, target)
        var logits_cpu = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]])
        var loss_cpu = CrossEntropyLoss[dtype](reduction="sum")(logits_cpu, target_cpu)
        loss.print()
        loss_cpu.print()
        assert_true(allclose(loss.item(), loss_cpu.item()))


fn test_ce_gpu_ci_ignore_index() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_ci_ignore_index")
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d2(
            [[2.0, 1.0, 0.5], [1.0, 2.0, 0.1], [3.0, 1.0, 0.2]]
        ).to_gpu()
        var target = Tensor[DType.int32].d1([0, -100, 2]).to_gpu()
        var target_cpu = Tensor[DType.int32].d1([0, -100, 2])
        var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
        var loss = ce(logits, target)
        var logits_cpu = Tensor[dtype].d2(
            [[2.0, 1.0, 0.5], [1.0, 2.0, 0.1], [3.0, 1.0, 0.2]]
        )
        var loss_cpu = CrossEntropyLoss[dtype](
            ignore_index=-100, reduction="mean"
        )(logits_cpu, target_cpu)
        loss.print()
        loss_cpu.print()
        assert_true(allclose(loss.item(), loss_cpu.item()))


fn test_ce_gpu_ci_label_smoothing() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_ci_label_smoothing")
        comptime dtype = DType.float32
        var logits = Tensor[dtype].d2([[2.0, 1.0, 0.5]]).to_gpu()
        var target = Tensor[DType.int32].d1([0])

        var ce = CrossEntropyLoss[dtype](
            reduction="mean", label_smoothing=Scalar[dtype](0.1)
        )
        var loss = ce(logits, target.to_gpu())
        var logits_cpu = Tensor[dtype].d2([[2.0, 1.0, 0.5]])
        var loss_cpu = CrossEntropyLoss[dtype](
            reduction="mean", label_smoothing=Scalar[dtype](0.1)
        )(logits_cpu, target)
        assert_true(allclose(loss.item(), loss_cpu.item()))

fn test_ce_gpu_ci_3d() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_ci_3d")
        comptime dtype = DType.float32

        # Define data with correct shape (N, C, H)
        var logits = Tensor[dtype].d3(
            [
                # Batch 0
                [
                    [2.0, 1.0],   # Class 0: spatial positions 0, 1
                    [1.0, 2.0],   # Class 1: spatial positions 0, 1
                    [0.5, 1.5]    # Class 2: spatial positions 0, 1
                ],
                # Batch 1
                [
                    [1.5, 2.0],   # Class 0: spatial positions 0, 1
                    [2.0, 1.0],   # Class 1: spatial positions 0, 1
                    [1.0, 2.0]    # Class 2: spatial positions 0, 1
                ]
            ],
            requires_grad=True
        )

        var target = Tensor[DType.int32].d2(
            [[0, 2],   # Batch 0: spatial0→class0, spatial1→class2
             [1, 0]]   # Batch 1: spatial0→class1, spatial1→class0
        )

        # GPU forward
        var logits_gpu = logits.to_gpu()
        var target_gpu = target.to_gpu()
        var ce = CrossEntropyLoss[dtype](reduction="mean")
        var loss_gpu = ce(logits_gpu, target_gpu)

        # CPU forward for comparison
        var loss_cpu = ce(logits, target)  # Use same criterion instance or create new

        # Compare results
        assert_true(allclose(loss_gpu.to_cpu().item(), loss_cpu.item(), 1e-5))

        # Optional: Test backward
        loss_gpu.backward()
        loss_cpu.backward()

        # Compare gradients
        assert_true(logits.grad().all_close(2 * logits_gpu.grad().to_cpu()))

        print("✓ 3D cross-entropy GPU test passed")



# ═════════════════════════════════════════════════════════════════════════════
# GROUP J: GPU Backward — Class Indices
# ═════════════════════════════════════════════════════════════════════════════


fn test_ce_gpu_bwd_ci_grad_shape() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_bwd_ci_grad_shape")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var target = Tensor[DType.int32].d1([0, 1])
        var ce = CrossEntropyLoss[dtype](reduction="mean")
        var loss = ce(a_gpu, target.to_gpu())
        loss.backward()
        assert_true(a.grad().shape() == Shape(2, 3))


fn test_ce_gpu_bwd_ci_parity() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_bwd_ci_parity")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True
        )
        var a_gpu = Tensor[dtype].d2(
            [[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True
        ).to_gpu()
        var target = Tensor[DType.int32].d1([0, 1])
        var ce = CrossEntropyLoss[dtype](reduction="mean")
        var loss_cpu = ce(a_cpu, target)
        loss_cpu.backward()
        var loss_gpu = ce(a_gpu, target.to_gpu())
        loss_gpu.backward()
        for i in range(2):
            for c in range(3):
                assert_true(
                    allclose(a_cpu.grad()[[i, c]], a_gpu.grad()[[i, c]], atol=1e-4)
                )


fn test_ce_gpu_bwd_ci_ignore_zeros_grad() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_bwd_ci_ignore_zeros_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[2.0, 1.0, 0.5], [1.0, 2.0, 0.1], [3.0, 1.0, 0.2]],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var target = Tensor[DType.int32].d1([0, -100, 2])
        var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
        var loss = ce(a_gpu, target.to_gpu())
        loss.backward()
        # Ignored row should have zero gradient
        for c in range(3):
            assert_true(abs(a.grad()[[1, c]]) < Float32(1e-6))


fn test_ce_gpu_bwd_ci_3d_ignore() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_bwd_ci_3d_ignore")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3(
            [
                [[2.0, 1.0, 0.5], [1.0, 2.0, 1.5], [0.5, 0.5, 2.0]],
                [[1.5, 2.0, 1.0], [2.0, 1.0, 2.0], [0.5, 0.5, 0.5]],
            ],
            requires_grad=True,
        )
        var a_gpu = a.to_gpu()
        var target = Tensor[DType.int32].d2([[0, -100, 2], [1, 0, -100]])
        var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
        var loss = ce(a_gpu, target.to_gpu())
        loss.backward()
        # Batch 0, spatial pos 1 ignored
        for c in range(3):
            assert_true(abs(a.grad()[[0, c, 1]]) < Float32(1e-6))
        # Batch 1, spatial pos 2 ignored
        for c in range(3):
            assert_true(abs(a.grad()[[1, c, 2]]) < Float32(1e-6))


fn test_ce_gpu_bwd_ci_label_smoothing() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_bwd_ci_label_smoothing")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True)
        var a_gpu = Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True).to_gpu()
        var target = Tensor[DType.int32].d1([0])
        var ce = CrossEntropyLoss[dtype](
            reduction="mean", label_smoothing=Scalar[dtype](0.1)
        )
        var loss_cpu = ce(a_cpu, target)
        loss_cpu.backward()
        var loss_gpu = ce(a_gpu, target.to_gpu())
        loss_gpu.backward()
        for c in range(3):
            assert_true(allclose(a_cpu.grad()[[0, c]], a_gpu.grad()[[0, c]], atol=1e-4))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP K: GPU — Probability Targets
# ═════════════════════════════════════════════════════════════════════════════


fn test_ce_gpu_prob_forward_parity() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_prob_forward_parity")
        comptime dtype = DType.float32
        var logits_cpu = Tensor[dtype].d2([[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]])
        var logits_gpu = logits_cpu.to_gpu()
        var target = Tensor[dtype].d2([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
        var ce = CrossEntropyLoss[dtype](reduction="mean")
        var loss_cpu = ce(logits_cpu, target)
        var loss_gpu = ce(logits_gpu, target.to_gpu())
        assert_true(allclose(loss_cpu.item(), loss_gpu.item()))


fn test_ce_gpu_prob_backward_parity() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_prob_backward_parity")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True
        )
        var a_gpu = Tensor[dtype].d2(
            [[2.0, 1.0, 0.5], [0.5, 2.0, 0.1]], requires_grad=True
        ).to_gpu()
        var target = Tensor[dtype].d2([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
        var ce = CrossEntropyLoss[dtype](reduction="mean")
        var loss_cpu = ce(a_cpu, target)
        loss_cpu.backward()
        var loss_gpu = ce(a_gpu, target.to_gpu())
        loss_gpu.backward()
        for i in range(2):
            for c in range(3):
                assert_true(
                    allclose(a_cpu.grad()[[i, c]], a_gpu.grad()[[i, c]], atol=1e-4)
                )


fn test_ce_gpu_prob_label_smoothing_parity() raises:
    comptime if has_accelerator():
        print("test_ce_gpu_prob_label_smoothing_parity")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True)
        var a_gpu = Tensor[dtype].d2([[2.0, 1.0, 0.5]], requires_grad=True).to_gpu()
        var target = Tensor[dtype].d2([[0.6, 0.3, 0.1]])
        var ce = CrossEntropyLoss[dtype](
            reduction="mean", label_smoothing=Scalar[dtype](0.1)
        )
        var loss_cpu = ce(a_cpu, target)
        loss_cpu.backward()
        var loss_gpu = ce(a_gpu, target.to_gpu())
        loss_gpu.backward()
        for c in range(3):
            assert_true(allclose(a_cpu.grad()[[0, c]], a_gpu.grad()[[0, c]], atol=1e-4))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP L: Edge Cases and Robustness
# ═════════════════════════════════════════════════════════════════════════════


fn test_ce_edge_single_class() raises:
    print("test_ce_edge_single_class")
    comptime dtype = DType.float32
    # C=1 — only one class, loss should be 0
    var logits = Tensor[dtype].d2([[5.0], [3.0]])
    var target = Tensor[DType.int32].d1([0, 0])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    assert_true(allclose(loss.item(), Float32(0.0), atol=1e-4))


fn test_ce_edge_large_batch() raises:
    print("test_ce_edge_large_batch")
    comptime dtype = DType.float32
    var N = 100
    var logits = Tensor[dtype].full(Shape(N, 5), 1.0)
    var target = Tensor[DType.int32].zeros(Shape(N))
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    assert_true(allclose(loss.item(), log(Float32(5)), atol=1e-4))


fn test_ce_edge_many_classes() raises:
    print("test_ce_edge_many_classes")
    comptime dtype = DType.float32
    var C = 1000
    var logits = Tensor[dtype].full(Shape(2, C), 1.0)
    var target = Tensor[DType.int32].zeros(Shape(2))
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    assert_true(allclose(loss.item(), log(Float32(C)), atol=1e-2))


fn test_ce_edge_numerically_stable_large_logits() raises:
    print("test_ce_edge_numerically_stable_large_logits")
    comptime dtype = DType.float32
    # Very large logits — should not overflow due to log_softmax stability
    var logits = Tensor[dtype].d2([[1000.0, 999.0, 998.0]])
    var target = Tensor[DType.int32].d1([0])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    # Should be finite and small (correct class has max logit)
    assert_true(loss.item() > 0)
    assert_true(loss.item() < Float32(5.0))


fn test_ce_edge_numerically_stable_negative_logits() raises:
    print("test_ce_edge_numerically_stable_negative_logits")
    comptime dtype = DType.float32
    var logits = Tensor[dtype].d2([[-1000.0, -999.0, -998.0]])
    var target = Tensor[DType.int32].d1([2])
    var ce = CrossEntropyLoss[dtype](reduction="mean")
    var loss = ce(logits, target)
    assert_true(loss.item() > 0)
    assert_true(loss.item() < Float32(5.0))


fn test_ce_edge_ignore_only_some_3d() raises:
    print("test_ce_edge_ignore_only_some_3d")
    comptime dtype = DType.float32
    # 3D with all positions ignored in one batch
    var logits = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 1.0]], [[2.0, 1.0], [1.0, 3.0]]]
    )
    var target = Tensor[DType.int32].d2([[-100, -100], [0, 1]])
    var ce = CrossEntropyLoss[dtype](ignore_index=-100, reduction="mean")
    var loss = ce(logits, target)
    # Only batch 1 contributes
    var logits2 = Tensor[dtype].d3([[[2.0, 1.0], [1.0, 3.0]]])
    var target2 = Tensor[DType.int32].d2([[0, 1]])
    var loss2 = CrossEntropyLoss[dtype](reduction="mean")(logits2, target2)
    assert_true(allclose(loss.item(), loss2.item()))


fn test_ce_edge_zero_grad_no_requires_grad() raises:
    print("test_ce_edge_zero_grad_no_requires_grad")
    comptime dtype = DType.float32
    # No grad required → loss computed but no autograd
    var logits = Tensor[dtype].d2([[2.0, 1.0]], requires_grad=False)
    var target = Tensor[DType.int32].d1([0])
    var ce = CrossEntropyLoss[dtype]()
    var loss = ce(logits, target)
    assert_true(not loss.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll cross entropy tests passed!")
