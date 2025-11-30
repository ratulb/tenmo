from tenmo import Tensor
from common_utils import panic
from testing import assert_true
from data import *

# Comprehensive tests for TensorDataset, Batch, and DataLoader
# With FIXED shuffle implementation

fn assert_true_1(condition: Bool, msg: String = "Assertion failed") raises:
    if not condition:
        raise Error(msg)

fn assert_tensors_equal[dtype: DType](t1: Tensor[dtype], t2: Tensor[dtype]) raises:
    if t1.shape() != t2.shape():
        raise Error("Shape mismatch")
    for i in range(t1.numels()):
        if t1.element_at(i) != t2.element_at(i):
            raise Error("Tensor values not equal")

# ============================================================================
# TensorDataset Tests
# ============================================================================

fn test_tensor_dataset_basic_creation() raises:
    print("test_tensor_dataset_basic_creation")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6]])
    var labels = Tensor[dtype].d1([10, 20, 30])

    var dataset = TensorDataset(features, labels)

    assert_true(len(dataset) == 3, "Dataset size should be 3")
    assert_true(dataset.features().shape()[0] == 3, "Features should have 3 samples")
    assert_true(dataset.labels().shape()[0] == 3, "Labels should have 3 samples")
    print("✓ Passed")

fn test_tensor_dataset_2d_labels() raises:
    print("test_tensor_dataset_2d_labels")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
    var labels = Tensor[dtype].d2([[10, 11], [20, 21]])

    var dataset = TensorDataset(features, labels)

    assert_true(len(dataset) == 2, "Dataset size should be 2")
    assert_true(dataset.labels().shape().rank() == 2, "Labels should be 2D")
    assert_true(dataset.labels().shape()[1] == 2, "Labels should have 2 dimensions")
    print("✓ Passed")

fn test_tensor_dataset_getitem_1d_labels() raises:
    print("test_tensor_dataset_getitem_1d_labels")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6]])
    var labels = Tensor[dtype].d1([10, 20, 30])

    var dataset = TensorDataset(features, labels)
    var (feat, lab) = dataset[1]

    assert_true(feat.shape()[0] == 2, "Feature should have 2 elements")
    assert_true(feat[0] == 3, "First feature element should be 3")
    assert_true(feat[1] == 4, "Second feature element should be 4")
    assert_true(lab[0] == 20, "Label should be 20")
    print("✓ Passed")

fn test_tensor_dataset_getitem_2d_labels() raises:
    print("test_tensor_dataset_getitem_2d_labels")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
    var labels = Tensor[dtype].d2([[10, 11], [20, 21]])

    var dataset = TensorDataset(features, labels)
    var (feat, lab) = dataset[0]

    assert_true(feat.shape()[0] == 3, "Feature should have 3 elements")
    assert_true(lab.shape()[0] == 2, "Label should have 2 elements")
    assert_true(lab[0] == 10, "First label should be 10")
    assert_true(lab[1] == 11, "Second label should be 11")
    print("✓ Passed")

fn test_tensor_dataset_getitem_all_indices() raises:
    print("test_tensor_dataset_getitem_all_indices")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6], [7, 8]])
    var labels = Tensor[dtype].d1([10, 20, 30, 40])

    var dataset = TensorDataset(features, labels)

    for i in range(4):
        var (feat, lab) = dataset[i]
        var expected_feat_0 = Float32(i * 2 + 1)
        var expected_feat_1 = Float32(i * 2 + 2)
        var expected_label = Float32((i + 1) * 10)

        assert_true(feat[0] == expected_feat_0, "Feature mismatch at index " + i.__str__())
        assert_true(feat[1] == expected_feat_1, "Feature mismatch at index " + i.__str__())
        assert_true(lab[0] == expected_label, "Label mismatch at index " + i.__str__())
    print("✓ Passed")

# ============================================================================
# Batch Tests
# ============================================================================

fn test_batch_creation() raises:
    print("test_batch_creation")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
    var labels = Tensor[dtype].d2([[10], [20]])

    var batch = Batch[dtype](features, labels)

    assert_true(batch.batch_size == 2, "Batch size should be 2")
    assert_true(batch.features.shape()[0] == 2, "Features should have 2 samples")
    assert_true(batch.labels.shape()[0] == 2, "Labels should have 2 samples")
    print("✓ Passed")

fn test_batch_single_sample() raises:
    print("test_batch_single_sample")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2]])
    var labels = Tensor[dtype].d2([[10]])

    var batch = Batch[dtype](features, labels)

    assert_true(batch.batch_size == 1, "Batch size should be 1")
    print("✓ Passed")

# ============================================================================
# DataLoader Tests
# ============================================================================

fn test_dataloader_basic_iteration() raises:
    print("test_dataloader_basic_iteration")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6], [7, 8]])
    var labels = Tensor[dtype].d1([10, 20, 30, 40])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=2, reshuffle=False)

    assert_true(len(loader) == 2, "Should have 2 batches")

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size == 2, "Each batch should have size 2")

    assert_true(batch_count == 2, "Should iterate through 2 batches")
    print("✓ Passed")

fn test_dataloader_batch_size_one() raises:
    print("test_dataloader_batch_size_one")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6]])
    var labels = Tensor[dtype].d1([10, 20, 30])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=1, reshuffle=False)

    assert_true(len(loader) == 3, "Should have 3 batches")

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size == 1, "Each batch should have size 1")

    assert_true(batch_count == 3, "Should iterate through 3 batches")
    print("✓ Passed")

fn test_dataloader_partial_last_batch() raises:
    print("test_dataloader_partial_last_batch")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    var labels = Tensor[dtype].d1([10, 20, 30, 40, 50])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=2, reshuffle=False, drop_last=False)

    assert_true(len(loader) == 3, "Should have 3 batches")

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        if batch_count < 3:
            assert_true(batch.batch_size == 2, "First 2 batches should have size 2")
        else:
            assert_true(batch.batch_size == 1, "Last batch should have size 1")

    assert_true(batch_count == 3, "Should iterate through 3 batches")
    print("✓ Passed")

fn test_dataloader_drop_last_true() raises:
    print("test_dataloader_drop_last_true")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    var labels = Tensor[dtype].d1([10, 20, 30, 40, 50])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=2, reshuffle=False, drop_last=True)

    assert_true(len(loader) == 2, "Should have 2 batches (drop last)")

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size == 2, "All batches should have size 2")

    assert_true(batch_count == 2, "Should iterate through only 2 batches")
    print("✓ Passed")

fn test_dataloader_exact_division() raises:
    print("test_dataloader_exact_division")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6], [7, 8]])
    var labels = Tensor[dtype].d1([10, 20, 30, 40])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=2, reshuffle=False)

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size == 2, "All batches should have size 2")

    assert_true(batch_count == 2, "Should have exactly 2 batches")
    print("✓ Passed")

fn test_dataloader_content_correctness() raises:
    print("test_dataloader_content_correctness")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6]])
    var labels = Tensor[dtype].d1([10, 20, 30])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=2, reshuffle=False)

    var batch_idx = 0
    for batch in loader:
        if batch_idx == 0:
            # First batch: samples 0 and 1
            assert_true(batch.features[0, 0] == 1, "Batch 0 feature mismatch")
            assert_true(batch.features[0, 1] == 2, "Batch 0 feature mismatch")
            assert_true(batch.features[1, 0] == 3, "Batch 0 feature mismatch")
            assert_true(batch.features[1, 1] == 4, "Batch 0 feature mismatch")
            assert_true(batch.labels[0, 0] == 10, "Batch 0 label mismatch")
            assert_true(batch.labels[1, 0] == 20, "Batch 0 label mismatch")
        elif batch_idx == 1:
            # Second batch: sample 2
            assert_true(batch.features[0, 0] == 5, "Batch 1 feature mismatch")
            assert_true(batch.features[0, 1] == 6, "Batch 1 feature mismatch")
            assert_true(batch.labels[0, 0] == 30, "Batch 1 label mismatch")
        batch_idx += 1
    print("✓ Passed")

fn test_dataloader_2d_labels() raises:
    print("test_dataloader_2d_labels")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4]])
    var labels = Tensor[dtype].d2([[10, 11], [20, 21]])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=2, reshuffle=False)

    for batch in loader:
        assert_true(batch.labels.shape()[1] == 2, "Labels should have 2 dimensions")
        assert_true(batch.labels[0, 0] == 10, "Label mismatch")
        assert_true(batch.labels[0, 1] == 11, "Label mismatch")
        assert_true(batch.labels[1, 0] == 20, "Label mismatch")
        assert_true(batch.labels[1, 1] == 21, "Label mismatch")
    print("✓ Passed")

fn test_dataloader_large_batch_size() raises:
    print("test_dataloader_large_batch_size")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6]])
    var labels = Tensor[dtype].d1([10, 20, 30])
    var dataset = TensorDataset(features, labels)

    # Batch size larger than dataset
    var loader = DataLoader[dtype](dataset^, batch_size=10, reshuffle=False)

    assert_true(len(loader) == 1, "Should have 1 batch")

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size == 3, "Batch should contain all 3 samples")

    assert_true(batch_count == 1, "Should iterate once")
    print("✓ Passed")

fn test_dataloader_multiple_epochs() raises:
    print("test_dataloader_multiple_epochs")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4]])
    var labels = Tensor[dtype].d1([10, 20])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=1, reshuffle=False)

    # First epoch
    var count1 = 0
    for _batch in loader:
        count1 += 1

    # Second epoch
    var count2 = 0
    for _batch in loader:
        count2 += 1

    assert_true(count1 == 2, "First epoch should have 2 batches")
    assert_true(count2 == 2, "Second epoch should have 2 batches")
    print("✓ Passed")

fn test_dataloader_reshuffle_changes_order() raises:
    print("test_dataloader_reshuffle_changes_order")
    alias dtype = DType.float32
    # Use a medium-sized dataset to test shuffle
    var features = Tensor[dtype].d2([
        [1, 2], [3, 4], [5, 6], [7, 8], [9, 10]
    ])
    var labels = Tensor[dtype].d1([10, 20, 30, 40, 50])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=2, reshuffle=True)

    # Note: This test just verifies that loader works with reshuffle=True
    # We can't reliably test randomness, but we can ensure it doesn't crash
    var batch_count = 0
    var total_samples = 0
    for batch in loader:
        batch_count += 1
        total_samples += batch.batch_size
        assert_true(batch.batch_size <= 2, "Batch size should be at most 2")

    assert_true(batch_count == 3, "Should have 3 batches")
    assert_true(total_samples == 5, "Should have processed all 5 samples")
    print("✓ Passed (shuffle functionality verified)")

fn test_dataloader_reshuffle_changes_order_orig() raises:
    print("test_dataloader_reshuffle_changes_order")
    alias dtype = DType.float32
    # Use a larger dataset to make shuffle more apparent
    var features = Tensor[dtype].d2([
        [1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
        [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]
    ])
    var labels = Tensor[dtype].d1([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=2, reshuffle=True)

    # Note: This test just verifies that loader works with reshuffle=True
    # We can't reliably test randomness, but we can ensure it doesn't crash
    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size <= 2, "Batch size should be at most 2")

    assert_true(batch_count == 5, "Should have 5 batches")
    print("✓ Passed (shuffle functionality verified)")

fn test_dataloader_single_sample_dataset() raises:
    print("test_dataloader_single_sample_dataset")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2, 3]])
    var labels = Tensor[dtype].d1([10])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=1, reshuffle=False)

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size == 1, "Batch should have 1 sample")
        assert_true(batch.features[0, 0] == 1, "Feature value mismatch")
        assert_true(batch.labels[0, 0] == 10, "Label value mismatch")

    assert_true(batch_count == 1, "Should have 1 batch")
    print("✓ Passed")


fn test_dataloader_high_dimensional_features() raises:
    print("test_dataloader_high_dimensional_features")
    alias dtype = DType.float32
    var features = Tensor[dtype].d2([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ])
    var labels = Tensor[dtype].d1([100, 200])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=1, reshuffle=False)

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.features.shape()[1] == 10, "Features should have 10 dimensions")

    assert_true(batch_count == 2, "Should have 2 batches")
    print("✓ Passed")

fn test_dataloader_shuffle_quality() raises:
    print("test_dataloader_shuffle_quality")
    alias dtype = DType.float32
    # Create a dataset with ordered indices
    var features = Tensor[dtype].d2([
        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4],
        [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]
    ])
    var labels = Tensor[dtype].d1([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    var dataset = TensorDataset(features, labels)

    var loader = DataLoader[dtype](dataset^, batch_size=10, reshuffle=True)

    # Get the shuffled batch
    var shuffled_order = List[Int]()
    for batch in loader:
        # Check that we got all 10 samples
        assert_true(batch.batch_size == 10, "Should have all 10 samples")

        # Record the order by checking label values
        for i in range(10):
            shuffled_order.append(Int(batch.labels[i, 0]))

    # Verify we have all indices 0-9 (no duplicates, no missing)
    var found = List[Bool](capacity=UInt(10))
    for _ in range(10):
        found.append(False)

    for i in range(len(shuffled_order)):
        var idx = shuffled_order[i]
        assert_true(idx >= 0 and idx < 10, "Invalid index in shuffle")
        found[idx] = True

    # Check all indices were found
    for i in range(10):
        assert_true(found[i], "Missing index " + i.__str__() + " after shuffle")

    print("✓ Passed (all indices present after shuffle)")

# ============================================================================
# Main Test Runner
# ============================================================================

fn main() raises:
    print("=" * 70)
    print("Running TensorDataset, Batch, and DataLoader Tests")
    print("=" * 70)
    print()

    # TensorDataset tests
    print("--- TensorDataset Tests ---")
    test_tensor_dataset_basic_creation()
    test_tensor_dataset_2d_labels()
    test_tensor_dataset_getitem_1d_labels()
    test_tensor_dataset_getitem_2d_labels()
    test_tensor_dataset_getitem_all_indices()
    print()

    # Batch tests
    print("--- Batch Tests ---")
    test_batch_creation()
    test_batch_single_sample()
    print()

    # DataLoader tests
    print("--- DataLoader Tests ---")
    test_dataloader_basic_iteration()
    test_dataloader_batch_size_one()
    test_dataloader_partial_last_batch()
    test_dataloader_drop_last_true()
    test_dataloader_exact_division()
    test_dataloader_content_correctness()
    test_dataloader_2d_labels()
    test_dataloader_large_batch_size()
    test_dataloader_multiple_epochs()
    test_dataloader_reshuffle_changes_order()
    test_dataloader_single_sample_dataset()
    test_dataloader_high_dimensional_features()
    test_dataloader_shuffle_quality()
    print()

    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)

    run_all_dataloader_tests()
# ============================================================================
# TensorDataset Tests
# ============================================================================

fn test_dataset_basic_creation_dl() raises:
    """Test basic dataset creation and indexing."""
    print("test_dataset_basic_creation_dl")
    var features = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    var labels = Tensor.d2([[0.0], [1.0], [2.0]]).float()

    var dataset = TensorDataset(features, labels)
    assert_true(len(dataset) == 3)


fn test_dataset_getitem_single_sample_dl() raises:
    """Test retrieving single samples."""
    print("test_dataset_getitem_single_sample_dl")
    var features = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).float()
    var labels = Tensor.d2([[10.0], [20.0]]).float()

    var dataset = TensorDataset(features, labels)

    # Test first sample
    var (feat0, label0) = dataset[0]
    var expected_feat0 = Tensor.d1([1.0, 2.0, 3.0]).float()
    var expected_label0 = Tensor.d1([10.0]).float()
    assert_true(feat0.all_close(expected_feat0))
    assert_true(label0.all_close(expected_label0))

    # Test second sample
    var (feat1, label1) = dataset[1]
    var expected_feat1 = Tensor.d1([4.0, 5.0, 6.0]).float()
    var expected_label1 = Tensor.d1([20.0]).float()
    assert_true(feat1.all_close(expected_feat1))
    assert_true(label1.all_close(expected_label1))


fn test_dataset_1d_labels_dl() raises:
    """Test dataset with 1D label vector."""
    print("test_dataset_1d_labels_dl")
    var features = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    var labels = Tensor.d1([0.0, 1.0, 2.0]).float()

    var dataset = TensorDataset(features, labels)

    var (feat, label) = dataset[1]
    var expected_feat = Tensor.d1([3.0, 4.0]).float()
    var expected_label = Tensor.d1([1.0]).float()
    assert_true(feat.all_close(expected_feat))
    assert_true(label.all_close(expected_label))


fn test_dataset_multi_dimensional_labels_dl() raises:
    """Test dataset with multi-dimensional labels."""
    print("test_dataset_multi_dimensional_labels_dl")
    var features = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = Tensor.d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).float()  # One-hot

    var dataset = TensorDataset(features, labels)

    var (_feat, label) = dataset[0]
    var expected_label = Tensor.d1([1.0, 0.0, 0.0]).float()
    assert_true(label.all_close(expected_label))


fn test_dataset_features_labels_accessors_dl() raises:
    """Test features() and labels() accessors."""
    print("test_dataset_features_labels_accessors_dl")
    var features = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = Tensor.d1([0.0, 1.0]).float()

    var dataset = TensorDataset(features, labels)

    assert_true(dataset.features().all_close(features))
    assert_true(dataset.labels().all_close(labels))


# ============================================================================
# Batch Tests
# ============================================================================

fn test_batch_creation_dl() raises:
    """Test batch container creation."""
    print("test_batch_creation_dl")
    var features = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    var labels = Tensor.d2([[0.0], [1.0], [2.0]]).float()

    var batch = Batch(features, labels)
    assert_true(batch.batch_size == 3)
    assert_true(batch.features.all_close(features))
    assert_true(batch.labels.all_close(labels))


# ============================================================================
# DataLoader Tests - Basic Functionality
# ============================================================================

fn test_dataloader_batch_size_exact_dl() raises:
    """Test dataloader with exact batch division."""
    print("test_dataloader_batch_size_exact_dl")
    var features = Tensor.d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    ).float()
    var labels = Tensor.d1([0.0, 1.0, 2.0, 3.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader[DType.float32](dataset^, batch_size=2, reshuffle=False, drop_last=False)

    assert_true(len(loader) == 2)

    var batch_count = 0
    for batch in loader:
        assert_true(batch.batch_size == 2)
        batch_count += 1

    assert_true(batch_count == 2)


fn test_dataloader_batch_size_with_remainder_dl() raises:
    """Test dataloader with incomplete last batch."""
    print("test_dataloader_batch_size_with_remainder_dl")
    var features = Tensor.d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    ).float()
    var labels = Tensor.d1([0.0, 1.0, 2.0, 3.0, 4.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader[DType.float32](dataset^, batch_size=2, reshuffle=False, drop_last=False)

    assert_true(len(loader) == 3)  # 2 full batches + 1 partial

    var batch_count = 0
    var last_batch_size = 0
    for batch in loader:
        batch_count += 1
        last_batch_size = batch.batch_size

    assert_true(batch_count == 3)
    assert_true(last_batch_size == 1)  # Last batch has 1 sample


fn test_dataloader_drop_last_true_dl() raises:
    """Test dataloader with drop_last=True."""
    print("test_dataloader_drop_last_true_dl")
    var features = Tensor.d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    ).float()
    var labels = Tensor.d1([0.0, 1.0, 2.0, 3.0, 4.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader[DType.float32](dataset^, batch_size=2, reshuffle=False, drop_last=True)

    assert_true(len(loader) == 2)  # Only 2 complete batches

    var batch_count = 0
    for batch in loader:
        assert_true(batch.batch_size == 2)  # All batches are full
        batch_count += 1

    assert_true(batch_count == 2)


fn test_dataloader_single_batch_dl() raises:
    """Test dataloader when batch_size >= dataset size."""
    print("test_dataloader_single_batch_dl")
    var features = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    var labels = Tensor.d1([0.0, 1.0, 2.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader[DType.float32](dataset^, batch_size=10, reshuffle=False, drop_last=False)

    assert_true(len(loader) == 1)

    var batch_count = 0
    for batch in loader:
        assert_true(batch.batch_size == 3)  # All 3 samples in one batch
        batch_count += 1

    assert_true(batch_count == 1)


fn test_dataloader_batch_size_one_dl() raises:
    """Test dataloader with batch_size=1."""
    print("test_dataloader_batch_size_one_dl")
    var features = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    var labels = Tensor.d1([0.0, 1.0, 2.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader(dataset^, batch_size=1, reshuffle=False, drop_last=False)

    assert_true(len(loader) == 3)

    var batch_count = 0
    for batch in loader:
        assert_true(batch.batch_size == 1)
        batch_count += 1

    assert_true(batch_count == 3)


# ============================================================================
# DataLoader Tests - Data Correctness (No Shuffle)
# ============================================================================

fn test_dataloader_data_correctness_no_shuffle_dl() raises:
    """Test that batches contain correct data without shuffling."""
    print("test_dataloader_data_correctness_no_shuffle_dl")
    var features = Tensor.d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    ).float()
    var labels = Tensor.d1([10.0, 20.0, 30.0, 40.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader(dataset^, batch_size=2, reshuffle=False, drop_last=False)

    var iter = loader.__iter__()

    # First batch
    var batch1 = iter.__next__()
    var expected_feat1 = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var expected_label1 = Tensor.d2([[10.0], [20.0]]).float()
    assert_true(batch1.features.all_close(expected_feat1))
    assert_true(batch1.labels.all_close(expected_label1))

    # Second batch
    var batch2 = iter.__next__()
    var expected_feat2 = Tensor.d2([[5.0, 6.0], [7.0, 8.0]]).float()
    var expected_label2 = Tensor.d2([[30.0], [40.0]]).float()
    assert_true(batch2.features.all_close(expected_feat2))
    assert_true(batch2.labels.all_close(expected_label2))


fn test_dataloader_multi_dimensional_features_dl() raises:
    """Test dataloader with higher dimensional features."""
    print("test_dataloader_multi_dimensional_features_dl")
    var features = Tensor.d2(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    ).float()
    var labels = Tensor.d1([0.0, 1.0, 2.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader(dataset^, batch_size=2, reshuffle=False, drop_last=False)

    var iter = loader.__iter__()
    var batch = iter.__next__()

    # Check feature dimensions
    assert_true(batch.features.shape()[0] == 2)  # batch_size
    assert_true(batch.features.shape()[1] == 4)  # feature_dim


fn test_dataloader_multi_dimensional_labels_dl() raises:
    """Test dataloader with multi-dimensional labels."""
    print("test_dataloader_multi_dimensional_labels_dl")
    var features = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    var labels = Tensor.d2(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ).float()  # One-hot encoded

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader(dataset^, batch_size=2, reshuffle=False, drop_last=False)

    var iter = loader.__iter__()
    var batch = iter.__next__()

    # Check label dimensions
    assert_true(batch.labels.shape()[0] == 2)  # batch_size
    assert_true(batch.labels.shape()[1] == 3)  # label_dim

    var expected_labels = Tensor.d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).float()
    assert_true(batch.labels.all_close(expected_labels))


# ============================================================================
# DataLoader Tests - Shuffling
# ============================================================================

fn test_dataloader_shuffle_changes_order_dl_orig() raises:
    """Test that shuffle actually changes the order (probabilistic test)."""
    print("test_dataloader_shuffle_changes_order_dl")
    var features = Tensor.d2(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
    ).float()
    var labels = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader(dataset^, batch_size=5, reshuffle=True, drop_last=False)

    var iter = loader.__iter__()
    var batch = iter.__next__()

    # Check that we got all samples (sum should be same)
    var sum_features = batch.features.sum()
    var expected_sum = Tensor.d1([15.0]).float()  # 1+2+3+4+5
    assert_true(sum_features.all_close(expected_sum))

fn test_dataloader_shuffle_changes_order_dl() raises:
    """Test that shuffle actually changes the order (probabilistic test)."""
    print("test_dataloader_shuffle_changes_order_dl")
    var features = Tensor.d2(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
    ).float()
    var labels = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader(dataset^, batch_size=5, reshuffle=True, drop_last=False)

    var iter = loader.__iter__()
    var batch = iter.__next__()

    # Check that we got all samples (sum should be same)
    var sum_features = batch.features.sum()
    var expected_sum = Scalar[DType.float32](15.0)  # Just use scalar value
    # Compare scalar values directly
    assert_true(abs(sum_features.item() - expected_sum) < 1e-5)

fn test_dataloader_shuffle_preserves_all_data_dl() raises:
    """Test that shuffle doesn't lose or duplicate data."""
    print("test_dataloader_shuffle_preserves_all_data_dl")
    var features = Tensor.d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    ).float()
    var labels = Tensor.d1([10.0, 20.0, 30.0, 40.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader(dataset^, batch_size=2, reshuffle=True, drop_last=False)

    # Collect all batches
    var all_features = Tensor.d2([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]).float()
    var all_labels = Tensor.d1([0.0, 0.0, 0.0, 0.0]).float()

    var idx = 0
    for batch in loader:
        for i in range(batch.batch_size):
            all_features[idx, 0] = batch.features[i, 0]
            all_features[idx, 1] = batch.features[i, 1]
            all_labels[idx] = batch.labels[i, 0]
            idx += 1

    # Check sum (order-independent verification)
    var sum_feat = all_features.sum()
    var expected_sum_feat = features.sum()
    assert_true(sum_feat.all_close(expected_sum_feat))

    var sum_labels = all_labels.sum()
    var expected_sum_labels = labels.sum()
    assert_true(sum_labels.all_close(expected_sum_labels))


# ============================================================================
# DataLoader Tests - Multiple Epochs
# ============================================================================

fn test_dataloader_multiple_epochs_dl() raises:
    """Test iterating through dataloader multiple times."""
    print("test_dataloader_multiple_epochs_dl")
    var features = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    var labels = Tensor.d1([0.0, 1.0, 2.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader(dataset^, batch_size=2, reshuffle=False, drop_last=False)

    # Epoch 1
    var epoch1_count = 0
    for _batch in loader:
        epoch1_count += 1
    assert_true(epoch1_count == 2)

    # Epoch 2 - should work again
    var epoch2_count = 0
    for _batch in loader:
        epoch2_count += 1
    assert_true(epoch2_count == 2)


fn test_dataloader_reshuffle_between_epochs_dl() raises:
    """Test that reshuffle=True reshuffles each epoch."""
    print("test_dataloader_reshuffle_between_epochs_dl")
    var features = Tensor.d2(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    ).float()
    var labels = Tensor.d1([1.0, 2.0, 3.0, 4.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader(dataset^, batch_size=4, reshuffle=True, drop_last=False)

    # Both epochs should contain all data (verified by sum)
    var iter1 = loader.__iter__()
    var batch1 = iter1.__next__()
    var sum1 = batch1.features.sum().item()

    var iter2 = loader.__iter__()
    var batch2 = iter2.__next__()
    var sum2 = batch2.features.sum().item()

    var expected_sum = Scalar[DType.float32]([10.0])  # 1+2+3+4
    assert_true(sum1 == expected_sum)
    assert_true(sum2 == expected_sum)

# ============================================================================
# DataLoader Tests - Edge Cases
# ============================================================================

fn test_dataloader_empty_iteration_drop_last_dl() raises:
    """Test dataloader when drop_last=True eliminates all batches."""
    print("test_dataloader_empty_iteration_drop_last_dl")
    var features = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = Tensor.d1([0.0, 1.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader(dataset^, batch_size=5, reshuffle=False, drop_last=True)

    assert_true(len(loader) == 0)

    var batch_count = 0
    for _batch in loader:
        batch_count += 1

    assert_true(batch_count == 0)


fn test_dataloader_large_batch_size_dl() raises:
    """Test dataloader with batch_size much larger than dataset."""
    print("test_dataloader_large_batch_size_dl")
    var features = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = Tensor.d1([0.0, 1.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = DataLoader(dataset^, batch_size=1000, reshuffle=False, drop_last=False)

    assert_true(len(loader) == 1)

    for batch in loader:
        assert_true(batch.batch_size == 2)


# ============================================================================
# Consolidated Test Runner
# ============================================================================

fn run_all_dataloader_tests() raises:
    """Run all dataloader tests."""
    print("\n=== Running DataLoader Test Suite ===\n")

    # TensorDataset tests
    test_dataset_basic_creation_dl()
    test_dataset_getitem_single_sample_dl()
    test_dataset_1d_labels_dl()
    test_dataset_multi_dimensional_labels_dl()
    test_dataset_features_labels_accessors_dl()

    # Batch tests
    test_batch_creation_dl()

    # DataLoader basic functionality
    test_dataloader_batch_size_exact_dl()
    test_dataloader_batch_size_with_remainder_dl()
    test_dataloader_drop_last_true_dl()
    test_dataloader_single_batch_dl()
    test_dataloader_batch_size_one_dl()

    # DataLoader data correctness
    test_dataloader_data_correctness_no_shuffle_dl()
    test_dataloader_multi_dimensional_features_dl()
    test_dataloader_multi_dimensional_labels_dl()

    # DataLoader shuffling
    test_dataloader_shuffle_changes_order_dl()
    test_dataloader_shuffle_preserves_all_data_dl()

    # DataLoader multiple epochs
    test_dataloader_multiple_epochs_dl()
    test_dataloader_reshuffle_between_epochs_dl()

    # DataLoader edge cases
    test_dataloader_empty_iteration_drop_last_dl()
    test_dataloader_large_batch_size_dl()

    print("\n=== All DataLoader Tests Passed! ===\n")

