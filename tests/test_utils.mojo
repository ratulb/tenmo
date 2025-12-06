from tenmo import Tensor
from common_utils import accuracy, now
from testing import assert_true


fn main() raises:
    run_all_accuracy_tests()


fn test_accuracy_perfect_predictions() raises:
    """Test accuracy with perfect predictions."""
    print("test_accuracy_perfect_predictions")
    alias dtype = DType.float32

    var pred = Tensor.d2([[0.9], [0.1], [0.8], [0.2]])
    var target = Tensor.d2([[1.0], [0.0], [1.0], [0.0]])

    var correct, total = accuracy(pred, target)
    assert_true(correct == 4)
    assert_true(total == 4)


fn test_accuracy_all_wrong_predictions() raises:
    """Test accuracy with all wrong predictions."""
    print("test_accuracy_all_wrong_predictions")
    alias dtype = DType.float32

    var pred = Tensor.d2([[0.1], [0.9], [0.2], [0.8]])
    var target = Tensor.d2([[1.0], [0.0], [1.0], [0.0]])

    var correct, total = accuracy(pred, target)
    assert_true(correct == 0)
    assert_true(total == 4)


fn test_accuracy_mixed_predictions() raises:
    """Test accuracy with some correct and some wrong."""
    print("test_accuracy_mixed_predictions")
    alias dtype = DType.float32

    var pred = Tensor.d2([[0.9], [0.1], [0.2], [0.8]])  # [1, 0, 0, 1]
    var target = Tensor.d2([[1.0], [0.0], [1.0], [0.0]])  # [1, 0, 1, 0]

    var correct, total = accuracy(pred, target)
    assert_true(correct == 2)  # First two are correct
    assert_true(total == 4)


fn test_accuracy_threshold_boundary() raises:
    """Test accuracy at threshold boundary (0.5)."""
    print("test_accuracy_threshold_boundary")
    alias dtype = DType.float32

    var pred = Tensor.d2([[0.5], [0.50001], [0.49999]])
    var target = Tensor.d2([[1.0], [1.0], [0.0]])

    var correct, total = accuracy(pred, target)
    # 0.5 should round down to 0 (not correct)
    # 0.50001 should be 1 (correct)
    # 0.49999 should be 0 (correct)
    assert_true(correct == 2)
    assert_true(total == 3)


fn test_accuracy_custom_threshold() raises:
    """Test accuracy with custom threshold."""
    print("test_accuracy_custom_threshold")
    alias dtype = DType.float32

    var pred = Tensor.d2([[0.7], [0.3], [0.6], [0.4]])
    var target = Tensor.d2([[1.0], [0.0], [1.0], [0.0]])

    # With threshold 0.6
    var correct, total = accuracy[threshold=0.6](pred, target)
    # 0.7 > 0.6 → 1 (correct)
    # 0.3 > 0.6 → 0 (correct)
    # 0.6 > 0.6 → 0 (wrong, should be 1)
    # 0.4 > 0.6 → 0 (correct)
    assert_true(correct == 3)
    assert_true(total == 4)


fn test_accuracy_single_sample() raises:
    """Test accuracy with single sample."""
    print("test_accuracy_single_sample")
    alias dtype = DType.float32

    var pred = Tensor.d2([[0.9]])
    var target = Tensor.d2([[1.0]])

    var correct, total = accuracy(pred, target)
    assert_true(correct == 1)
    assert_true(total == 1)


fn test_accuracy_large_batch() raises:
    """Test accuracy with larger batch."""
    print("test_accuracy_large_batch")
    alias dtype = DType.float32

    # Create batch of 100 samples, all correct
    var pred_list = List[Scalar[dtype]]()
    var target_list = List[Scalar[dtype]]()

    for _ in range(50):
        pred_list.append(0.9)  # Predict 1
        target_list.append(1.0)  # True 1
    for _ in range(50):
        pred_list.append(0.1)  # Predict 0
        target_list.append(0.0)  # True 0

    var pred = Tensor[dtype](pred_list).reshape(-1, 1)
    var target = Tensor[dtype](target_list).reshape(-1, 1)

    var correct, total = accuracy(pred, target)
    assert_true(correct == 100)
    assert_true(total == 100)


fn test_accuracy_float64() raises:
    """Test accuracy with float64 dtype."""
    print("test_accuracy_float64")
    alias dtype = DType.float64

    var pred = Tensor.d2([[0.9], [0.1], [0.8]])
    var target = Tensor.d2([[1.0], [0.0], [1.0]])

    var correct, total = accuracy(pred, target)
    assert_true(correct == 3)
    assert_true(total == 3)


fn run_all_accuracy_tests() raises:
    """Run all accuracy tests."""
    print("\n=== Running Accuracy Test Suite ===\n")
    start = now()
    test_accuracy_perfect_predictions()
    test_accuracy_all_wrong_predictions()
    test_accuracy_mixed_predictions()
    test_accuracy_threshold_boundary()
    test_accuracy_custom_threshold()
    test_accuracy_single_sample()
    test_accuracy_large_batch()
    test_accuracy_float64()
    print("Accuracy tests completed in ", now() - start, " seconds")
    print("\n=== All Accuracy Tests Passed! ===\n")
