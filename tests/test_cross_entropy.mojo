from tensors import Tensor
from crossentropy import CrossEntropyLoss
from testing import assert_true
from testing import assert_false
from time import perf_counter_ns
from math import log, exp
from shapes import Shape
from intlist import IntList


fn main() raises:
    test_ce_gradients_computation_heavy()
    test_ce_reduction_none_1()
    # Basic functionality tests
    test_ce_basic_no_reduction()
    test_ce_reduction_mean()
    test_ce_reduction_sum()
    test_ce_reduction_none()

    # Label smoothing tests
    test_ce_label_smoothing_basic()
    test_ce_label_smoothing_mean()
    test_ce_label_smoothing_extreme()

    # Ignore index tests
    test_ce_ignore_index_basic()
    test_ce_ignore_index_all_ignored()
    test_ce_ignore_index_partial()

    # Spatial dimensions tests
    test_ce_2d_spatial()
    test_ce_3d_spatial()
    test_ce_spatial_with_ignore()
    test_ce_5d_spatial()
    # Gradient validation tests
    test_ce_gradients_basic()
    test_ce_gradients_label_smoothing()
    test_ce_gradients_ignore_index()
    test_ce_gradients_spatial()

    # Edge case tests
    test_ce_single_class()
    test_ce_perfect_prediction()
    test_ce_worst_prediction()
    test_ce_zero_logits()
    test_ce_large_logits()

    # Validation tests (negative)
    # test_ce_validation_wrong_target_dims()
    # test_ce_validation_class_out_of_bounds()
    # test_ce_validation_spatial_mismatch()
    # test_ce_validation_batch_size_mismatch()

    test_ce_extreme_perfect_prediction()
    test_ce_reduction_sum_1()
    test_ce_2d_basic()
    test_ce_2d_ignore_index()
    test_ce_2d_label_smoothing()
    test_ce_2d_logits_1d_target()
    test_ce_3d_spatial()
    test_ce_4d_spatial()
    test_ce_4d_spatial_1x2()
    test_ce_reduction_mean()
    test_ce_reduction_sum()
    test_ce_reduction_none()
    test_ce_reduction_mean_1()


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
    var target = Tensor.d3(
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
    var gradients = logits.gradbox[]
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
    var target = Tensor.d1([0, 1])

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
    var target = Tensor.d1([0, 1, 0])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    # Should be average of 3 sample losses
    assert_true(loss.shape == Shape())
    assert_true(loss.item() > 0)


fn test_ce_reduction_sum_1() raises:
    print("test_ce_reduction_sum_1")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=1)
    var loss = loss_fn(logits, target)
    # Should be sum of 2 sample losses
    assert_true(loss.shape == Shape([1]))
    assert_true(loss.item() > 0)


fn test_ce_reduction_none_1() raises:
    print("test_ce_reduction_none_1")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=2)
    var loss = loss_fn(logits, target)

    assert_true(loss.shape == Shape([2]))  # per-sample loss
    assert_true(loss[0] > 0 and loss[1] > 0)


fn test_ce_label_smoothing_basic() raises:
    print("test_ce_label_smoothing_basic")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn_no_smooth = CrossEntropyLoss[DType.float32](label_smoothing=0.0)
    var loss_fn_smooth = CrossEntropyLoss[DType.float32](label_smoothing=0.1)

    var loss_no_smooth = loss_fn_no_smooth(logits, target)
    var loss_smooth = loss_fn_smooth(logits, target)

    # Label smoothing should give different (usually higher) loss
    assert_true(abs(loss_smooth.item() - loss_no_smooth.item()) > 1e-6)


fn test_ce_label_smoothing_mean() raises:
    print("test_ce_label_smoothing_mean")
    var logits = Tensor.d2([[3.0, 1.0, 0.5], [1.0, 3.0, 0.1]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](
        reduction=0, label_smoothing=0.2
    )
    var loss = loss_fn(logits, target)

    assert_true(loss.shape == Shape())
    assert_true(loss.item() > 0)


fn test_ce_label_smoothing_extreme() raises:
    print("test_ce_label_smoothing_extreme")
    var logits = Tensor.d2([[10.0, 0.0], [0.0, 10.0]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn_max_smooth = CrossEntropyLoss[DType.float32](
        label_smoothing=0.9
    )
    var loss = loss_fn_max_smooth(logits, target)

    # With extreme smoothing, loss should be relatively high even for good predictions
    assert_true(loss.item() > 1.0)


fn test_ce_ignore_index_basic() raises:
    print("test_ce_ignore_index_basic")
    var logits = Tensor.d2(
        [[2.0, 1.0, 0.5], [1.0, 2.0, 0.1], [3.0, 1.0, 0.2]]
    ).float()
    var target = Tensor.d1([0, -100, 2])  # Second sample ignored

    var loss_fn = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction=0
    )
    var loss = loss_fn(logits, target)

    # Should only average over 2 samples (not 3)
    assert_true(loss.item() > 0)


fn test_ce_ignore_index_all_ignored() raises:
    print("test_ce_ignore_index_all_ignored")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor.d1([-100, -100])  # All ignored

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
    var target = Tensor.d2([[-100, 1], [0, -100]])  # Mixed ignored

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
    var target = Tensor.d3([[[0, 1], [1, 0]]])  # (1, 2, 2)

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    # Should compute mean over 4 spatial positions
    assert_true(loss.shape == Shape())
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
    var target = Tensor.d4(
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

    assert_true(loss.shape == Shape([1]))
    assert_true(loss.item() > 0)
    print("5D spatial test passed")


fn test_ce_spatial_with_ignore() raises:
    print("test_ce_spatial_with_ignore")
    var logits = Tensor.d4(
        [[[[2.0, 1.0], [1.0, 2.0]], [[3.0, 1.0], [1.0, 3.0]]]]
    ).float()  # (1, 2, 2, 2)
    var target = Tensor.d3([[[0, -100], [1, 0]]])  # One position ignored

    var loss_fn = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction=0
    )
    var loss = loss_fn(logits, target)

    # Should mean over 3 non-ignored positions
    assert_true(loss.item() > 0)


fn test_ce_gradients_basic_uu() raises:
    print("test_ce_gradients_basic")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]], requires_grad=True).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    loss.backward()

    # Gradients should exist and be non-zero
    assert_true(logits.gradbox[].sum().item() != 0.0)
    # Specific gradient pattern check
    var softmax = logits.softmax(axes=IntList(1))
    var batch_size = Scalar[DType.float32](logits.shape[0])
    assert_true(abs(logits.gradbox[][0, 0] - (softmax[0, 0] - 1.0)) < 1e-6)


fn test_ce_gradients_basic() raises:
    print("test_ce_gradients_basic")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]], requires_grad=True).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)  # mean
    var loss = loss_fn(logits, target)

    loss.backward()

    # Gradients should exist and be non-zero
    assert_true(logits.gradbox[].sum().item() != 0.0)

    # Specific gradient pattern check - ACCOUNT FOR MEAN REDUCTION!
    var softmax = logits.softmax(axes=IntList(1))
    var batch_size = Scalar[DType.float32](logits.shape[0])

    # For mean reduction: gradient = (softmax - one_hot) / batch_size
    var expected_grad_00 = (softmax[0, 0] - 1.0) / batch_size
    assert_true(abs(logits.gradbox[][0, 0] - expected_grad_00) < 1e-6)

    # Also check other elements
    var expected_grad_01 = (softmax[0, 1] - 0.0) / batch_size
    assert_true(abs(logits.gradbox[][0, 1] - expected_grad_01) < 1e-6)

    var expected_grad_10 = (softmax[1, 0] - 0.0) / batch_size
    assert_true(abs(logits.gradbox[][1, 0] - expected_grad_10) < 1e-6)

    var expected_grad_11 = (softmax[1, 1] - 1.0) / batch_size
    assert_true(abs(logits.gradbox[][1, 1] - expected_grad_11) < 1e-6)


fn test_ce_gradients_label_smoothing() raises:
    print("test_ce_gradients_label_smoothing")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]], requires_grad=True).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](label_smoothing=0.1)
    var loss = loss_fn(logits, target)

    loss.backward()

    # Gradients with smoothing should be different from without
    var grads_smooth = logits.gradbox[].copy()
    logits.zero_grad()

    var loss_fn_no_smooth = CrossEntropyLoss[DType.float32](label_smoothing=0.0)
    var loss_no_smooth = loss_fn_no_smooth(logits, target)
    loss_no_smooth.backward()

    assert_false(grads_smooth.all_close(logits.gradbox[]))


fn test_ce_gradients_ignore_index() raises:
    print("test_ce_gradients_ignore_index")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]], requires_grad=True).float()
    var target = Tensor.d1([0, -100])  # Second sample ignored

    var loss_fn = CrossEntropyLoss[DType.float32](
        ignore_index=-100, reduction=0
    )
    var loss = loss_fn(logits, target)

    loss.backward()

    # Gradients for ignored sample should be zero
    assert_true(abs(logits.gradbox[][1, 0]) < 1e-10)
    assert_true(abs(logits.gradbox[][1, 1]) < 1e-10)
    # Gradients for non-ignored sample should be non-zero
    assert_true(abs(logits.gradbox[][0, 0]) > 1e-6)


fn test_ce_gradients_spatial() raises:
    print("test_ce_gradients_spatial")

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
    var target = Tensor.d3(
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
    assert_true(logits.gradbox[].sum().item() != 0.0)
    print("Spatial gradients test passed")


fn test_ce_single_class() raises:
    print("test_ce_single_class")
    var logits = Tensor.d2([[5.0], [3.0]]).float()  # Single class
    var target = Tensor.d1([0, 0])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    # With single class, softmax should be ~1.0, loss should be low
    assert_true(loss.item() < 0.1)


fn test_ce_perfect_prediction() raises:
    print("test_ce_perfect_prediction")
    var logits = Tensor.d2([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]).float()
    var target = Tensor.d1([0, 1])

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
    var target = Tensor.d1([0, 1])

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
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    # Worst possible prediction should have high loss
    assert_true(loss.item() > 5.0)


fn test_ce_zero_logits() raises:
    print("test_ce_zero_logits")
    var logits = Tensor.d2([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).float()
    var target = Tensor.d1([0, 2])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    # With zero logits, all classes have equal probability -> higher loss
    var expected_loss = Scalar[DType.float32](-log(1.0 / 3.0))  # ~1.0986
    assert_true(abs(loss.item() - expected_loss) < 1e-6)


fn test_ce_large_logits() raises:
    print("test_ce_large_logits")
    var logits = Tensor.d2([[1000.0, 0.0], [0.0, 1000.0]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    var loss = loss_fn(logits, target)

    # Very large logits should still give valid (low) loss due to numerical stability
    assert_true(loss.item() < 1e-6)


# Negative validation tests (should not panic if validation is correct)
fn test_ce_validation_wrong_target_dims() raises:
    print("test_ce_validation_wrong_target_dims")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor.d2([[0, 1], [1, 0]])  # Wrong: should be 1D

    # This should be caught by validation before any computation
    # (Test framework should handle the panic)
    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    _loss = loss_fn(logits, target)


fn test_ce_validation_class_out_of_bounds() raises:
    print("test_ce_validation_class_out_of_bounds")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()  # 2 classes
    var target = Tensor.d1([0, 2])  # Class 2 is out of bounds"""

    # Should be caught by validation

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    _loss = loss_fn(logits, target)


fn test_ce_validation_spatial_mismatch() raises:
    print("test_ce_validation_spatial_mismatch")
    var logits = Tensor.d4([[[[2.0, 1.0], [1.0, 2.0]]]]).float()  # (1, 2, 2, 2)
    var target = Tensor.d3([[[0, 1, 0]]])  # Wrong spatial dim: (1, 1, 3)

    # Should be caught by spatial dimension validation
    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    _loss = loss_fn(logits, target)


fn test_ce_validation_batch_size_mismatch() raises:
    print("test_ce_validation_batch_size_mismatch")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()  # batch size 2
    var target = Tensor.d1([0, 1, 0])  # batch size 3

    # Should be caught by batch size validation

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)
    _loss = loss_fn(logits, target)


fn test_ce_2d_basic() raises:
    print("test_ce_2d_basic")
    var logits = Tensor.d2([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]]).float()
    var target = Tensor.d1([0, 1])  # Class indices
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
    var target = Tensor.d1([0, -1])  # Second sample ignored
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
    var target = Tensor.d1([0, 1])

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
    var target = Tensor.d1([0, 1])

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
    var target = Tensor.d2([[0, 1], [1, 0]])  # (2, 2)

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
    var target = Tensor.d3(
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
    var target = Tensor.d3(
        [[[0, 1]], [[1, 0]]]  # Sample 0: 1x2 spatial  # Sample 1: 1x2 spatial
    )  # Shape: (2, 1, 2) - matches logits spatial dims

    var loss_fn = CrossEntropyLoss[DType.float32]()
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_4d_spatial_1x2 -> Total:            ", end - start)
    assert_true(loss.item() > 0.0)


fn test_ce_reduction_mean() raises:
    print("test_ce_reduction_mean")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=0)  # mean
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_reduction_mean -> Total:            ", end - start)
    assert_true(loss.shape.rank() == 0)  # scalar


fn test_ce_reduction_sum() raises:
    print("test_ce_reduction_sum")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=1)  # sum
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_reduction_sum -> Total:            ", end - start)

    assert_true(loss.shape.rank() == 1)  # scalar


fn test_ce_reduction_none() raises:
    print("test_ce_reduction_none")
    var logits = Tensor.d2([[2.0, 1.0], [1.0, 2.0]]).float()
    var target = Tensor.d1([0, 1])

    var loss_fn = CrossEntropyLoss[DType.float32](reduction=2)  # none
    # var loss = loss_fn(logits, target)
    start = perf_counter_ns()
    var loss = loss_fn(logits, target)
    end = perf_counter_ns()
    print("test_ce_reduction_none -> Total:            ", end - start)

    assert_true(loss.shape == Shape([2]))  # per-sample loss
