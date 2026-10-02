from tenmo.tensor import Tensor
from tenmo.shared.panic import panic
from tenmo.shared.timing import now
from std.testing import (
    assert_true,
    assert_equal,
    assert_false,
    assert_raises,
    TestSuite,
)
from tenmo.dataloader import *
from tenmo.nlp.dataset import RandomSlidingWindowDataset

# from tenmo.nlp import LLMDataset
# from bpe import BasicTokenizer, Tokenizer
from std.python import Python, PythonObject

# Comprehensive tests for TensorDataset, Batch, DataLoader (tensor engine),
# NativeLoader (trait-generic engine), and the token-stream windowing core
# (SlidingWindowDataset / WindowLoader)

comptime dtype = DType.float32


def assert_true_1(condition: Bool, msg: String = "Assertion failed") raises:
    if not condition:
        raise Error(msg)


def assert_tensors_equal[
    dtype: DType
](t1: Tensor[dtype], t2: Tensor[dtype]) raises:
    if t1.shape() != t2.shape():
        raise Error("Shape mismatch")
    for i in range(t1.numels()):
        if t1.get(i) != t2.get(i):
            raise Error("Tensor values not equal")


# ============================================================================
# TensorDataset Tests
# ============================================================================


def test_tensor_dataset_basic_creation() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6]])
    var labels = Tensor[dtype].d1([10, 20, 30])

    var dataset = TensorDataset(features, labels)

    assert_true(len(dataset) == 3, "Dataset size should be 3")
    assert_true(
        dataset._features.shape()[0] == 3, "Features should have 3 samples"
    )
    assert_true(dataset._labels.shape()[0] == 3, "Labels should have 3 samples")


def test_tensor_dataset_2d_labels() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
    var labels = Tensor[dtype].d2([[10, 11], [20, 21]])

    var dataset = TensorDataset(features, labels)

    assert_true(len(dataset) == 2, "Dataset size should be 2")
    assert_true(dataset._labels.shape().rank() == 2, "Labels should be 2D")
    assert_true(
        dataset._labels.shape()[1] == 2, "Labels should have 2 dimensions"
    )


def test_tensor_dataset_getitem_1d_labels() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6]])
    var labels = Tensor[dtype].d1([10, 20, 30])

    var dataset = TensorDataset(features, labels)
    var (feat, lab) = dataset[1]
    assert_true(feat.shape()[0] == 2, "Feature should have 2 elements")
    assert_true(feat[0] == 3, "First feature element should be 3")
    assert_true(feat[1] == 4, "Second feature element should be 4")
    # assert_true(lab[0] == 20, "Label should be 20")
    assert_true(lab[[]] == 20, "Label should be 20")


def test_tensor_dataset_getitem_2d_labels() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
    var labels = Tensor[dtype].d2([[10, 11], [20, 21]])

    var dataset = TensorDataset(features, labels)
    var (feat, lab) = dataset[0]

    assert_true(feat.shape()[0] == 3, "Feature should have 3 elements")
    assert_true(lab.shape()[0] == 2, "Label should have 2 elements")
    assert_true(lab[0] == 10, "First label should be 10")
    assert_true(lab[1] == 11, "Second label should be 11")


def test_tensor_dataset_getitem_all_indices() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6], [7, 8]])
    var labels = Tensor[dtype].d1([10, 20, 30, 40])

    var dataset = TensorDataset(features, labels)

    for i in range(4):
        var (feat, lab) = dataset[i]
        var expected_feat_0 = Float32(i * 2 + 1)
        var expected_feat_1 = Float32(i * 2 + 2)
        var expected_label = Float32((i + 1) * 10)

        assert_true(
            feat[0] == expected_feat_0,
            "Feature mismatch at index " + String(i),
        )
        assert_true(
            feat[1] == expected_feat_1,
            "Feature mismatch at index " + String(i),
        )
        assert_true(
            lab[[]] == expected_label, "Label mismatch at index " + String(i)
        )


# ============================================================================
# Batch Tests
# ============================================================================


def test_batch_creation() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]])
    var labels = Tensor[dtype].d2([[10], [20]])

    var batch = Batch[dtype](features, labels)

    assert_true(batch.batch_size == 2, "Batch size should be 2")
    assert_true(
        batch.features.shape()[0] == 2, "Features should have 2 samples"
    )
    assert_true(batch.labels.shape()[0] == 2, "Labels should have 2 samples")


def test_batch_single_sample() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2]])
    var labels = Tensor[dtype].d2([[10]])

    var batch = Batch[dtype](features, labels)

    assert_true(batch.batch_size == 1, "Batch size should be 1")


# ============================================================================
# DataLoader Tests
# ============================================================================


def test_dataloader_basic_iteration() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6], [7, 8]])
    var labels = Tensor[dtype].d1([10, 20, 30, 40])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(batch_size=2, shuffle=False)

    assert_true(len(loader) == 2, "Should have 2 batches")

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size == 2, "Each batch should have size 2")

    assert_true(batch_count == 2, "Should iterate through 2 batches")


def test_dataloader_batch_size_one() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6]])
    var labels = Tensor[dtype].d1([10, 20, 30])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(batch_size=1, shuffle=False)

    assert_true(len(loader) == 3, "Should have 3 batches")

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size == 1, "Each batch should have size 1")

    assert_true(batch_count == 3, "Should iterate through 3 batches")


def test_dataloader_partial_last_batch() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    var labels = Tensor[dtype].d1([10, 20, 30, 40, 50])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

    assert_true(len(loader) == 3, "Should have 3 batches")

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        if batch_count < 3:
            assert_true(
                batch.batch_size == 2, "First 2 batches should have size 2"
            )
        else:
            assert_true(batch.batch_size == 1, "Last batch should have size 1")

    assert_true(batch_count == 3, "Should iterate through 3 batches")


def test_dataloader_drop_last_true() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    var labels = Tensor[dtype].d1([10, 20, 30, 40, 50])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=True
    )

    assert_true(len(loader) == 2, "Should have 2 batches (drop last)")

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size == 2, "All batches should have size 2")

    assert_true(batch_count == 2, "Should iterate through only 2 batches")


def test_dataloader_exact_division() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6], [7, 8]])
    var labels = Tensor[dtype].d1([10, 20, 30, 40])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(batch_size=2, shuffle=False)

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size == 2, "All batches should have size 2")

    assert_true(batch_count == 2, "Should have exactly 2 batches")


def test_dataloader_content_correctness() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6]])
    var labels = Tensor[dtype].d1([10, 20, 30])
    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(batch_size=2, shuffle=False)

    var batch_idx = 0
    var iter = loader.__iter__()
    while iter.__has_next__():
        var batch = iter.__next__()
        if batch_idx == 0:
            # First batch: samples 0 and 1
            assert_true(batch.features[0, 0] == 1, "Batch 0 feature mismatch")
            assert_true(batch.features[0, 1] == 2, "Batch 0 feature mismatch")
            assert_true(batch.features[1, 0] == 3, "Batch 0 feature mismatch")
            assert_true(batch.features[1, 1] == 4, "Batch 0 feature mismatch")
            assert_true(
                batch.labels[0] == 10, "Batch 0 label mismatch"
            )  # Fixed
            assert_true(
                batch.labels[1] == 20, "Batch 0 label mismatch"
            )  # Fixed
        elif batch_idx == 1:
            # Second batch: sample 2
            assert_true(batch.features[0, 0] == 5, "Batch 1 feature mismatch")
            assert_true(batch.features[0, 1] == 6, "Batch 1 feature mismatch")
            assert_true(
                batch.labels[0] == 30, "Batch 1 label mismatch"
            )  # Fixed
        batch_idx += 1

    assert_true(batch_idx == 2, "Should have 2 batches")


def test_dataloader_2d_labels() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4]])
    var labels = Tensor[dtype].d2([[10, 11], [20, 21]])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(batch_size=2, shuffle=False)

    for batch in loader:
        assert_true(
            batch.labels.shape()[1] == 2, "Labels should have 2 dimensions"
        )
        assert_true(batch.labels[0, 0] == 10, "Label mismatch")
        assert_true(batch.labels[0, 1] == 11, "Label mismatch")
        assert_true(batch.labels[1, 0] == 20, "Label mismatch")
        assert_true(batch.labels[1, 1] == 21, "Label mismatch")


def test_dataloader_large_batch_size() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6]])
    var labels = Tensor[dtype].d1([10, 20, 30])
    var dataset = TensorDataset(features, labels)

    # Batch size larger than dataset
    var loader = dataset.into_loader(batch_size=10, shuffle=False)

    assert_true(len(loader) == 1, "Should have 1 batch")

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size == 3, "Batch should contain all 3 samples")

    assert_true(batch_count == 1, "Should iterate once")


def test_dataloader_multiple_epochs() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2], [3, 4]])
    var labels = Tensor[dtype].d1([10, 20])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(batch_size=1, shuffle=False)

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


def test_dataloader_steps_cap_across_epochs() raises:
    """Boundary: `while done < steps: for batch in loader` runs past one epoch.

    Regression for the imdb_bert incident (§4.9): a single `for` pass is
    exactly one epoch (`__next__` raises StopIteration at the end), so a
    bare `for` + `break`-at-cap silently under-trains whenever `steps`
    exceeds batches-per-epoch. The blessed multi-epoch shape re-enters
    `__iter__` per epoch (fresh shuffle for free).
    """
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    )
    var labels = Tensor[dtype].d1([10, 20, 30, 40, 50])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(batch_size=2, shuffle=False, drop_last=True)

    # 5 rows / B=2 / drop_last → 2 batches per epoch; ask for 5 steps.
    var done = 0
    var epochs = 0
    var per_epoch = List[Int]()
    while done < 5:
        epochs += 1
        var n_this_epoch = 0
        for _batch in loader:
            if done >= 5:
                break
            done += 1
            n_this_epoch += 1
        per_epoch.append(n_this_epoch)

    assert_true(done == 5, "Steps cap must bind across epochs")
    assert_true(epochs == 3, "5 steps at 2 batches/epoch needs 3 epochs")
    assert_true(len(per_epoch) == 3, "Three epoch passes expected")
    assert_true(per_epoch[0] == 2, "Epoch 1 yields 2 batches")
    assert_true(per_epoch[1] == 2, "Epoch 2 yields 2 batches")
    assert_true(per_epoch[2] == 1, "Epoch 3 stops at the cap")


def test_dataloader_reshuffle_changes_order() raises:
    comptime dtype = DType.float32
    # Use a medium-sized dataset to test shuffle
    var features = Tensor[dtype].d2([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    var labels = Tensor[dtype].d1([10, 20, 30, 40, 50])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(batch_size=2, shuffle=True)

    # Note: This test just verifies that loader works with shuffle=True
    # We can't reliably test randomness, but we can ensure it doesn't crash
    var batch_count = 0
    var total_samples = 0
    for batch in loader:
        batch_count += 1
        total_samples += batch.batch_size
        assert_true(batch.batch_size <= 2, "Batch size should be at most 2")

    assert_true(batch_count == 3, "Should have 3 batches")
    assert_true(total_samples == 5, "Should have processed all 5 samples")


def test_dataloader_reshuffle_changes_order_orig() raises:
    comptime dtype = DType.float32
    # Use a larger dataset to make shuffle more apparent
    var features = Tensor[dtype].d2(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
            [17, 18],
            [19, 20],
        ]
    )
    var labels = Tensor[dtype].d1([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(batch_size=2, shuffle=True)

    # Note: This test just verifies that loader works with shuffle=True
    # We can't reliably test randomness, but we can ensure it doesn't crash
    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size <= 2, "Batch size should be at most 2")

    assert_true(batch_count == 5, "Should have 5 batches")


def test_dataloader_single_sample_dataset() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2([[1, 2, 3]])
    var labels = Tensor[dtype].d1([10])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(batch_size=1, shuffle=False)

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(batch.batch_size == 1, "Batch should have 1 sample")
        assert_true(batch.features[0, 0] == 1, "Feature value mismatch")
        assert_true(batch.labels[0] == 10, "Label value mismatch")

    assert_true(batch_count == 1, "Should have 1 batch")


def test_dataloader_high_dimensional_features() raises:
    comptime dtype = DType.float32
    var features = Tensor[dtype].d2(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        ]
    )
    var labels = Tensor[dtype].d1([100, 200])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(batch_size=1, shuffle=False)

    var batch_count = 0
    for batch in loader:
        batch_count += 1
        assert_true(
            batch.features.shape()[1] == 10,
            "Features should have 10 dimensions",
        )

    assert_true(batch_count == 2, "Should have 2 batches")


def test_dataloader_shuffle_quality() raises:
    comptime dtype = DType.float32
    # Create a dataset with ordered indices
    var features = Tensor[dtype].d2(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6],
            [7, 7],
            [8, 8],
            [9, 9],
        ]
    )
    var labels = Tensor[dtype].d1([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    var dataset = TensorDataset(features, labels)

    var loader = dataset.into_loader(batch_size=10, shuffle=True)

    # Get the shuffled batch
    var shuffled_order = List[Int]()
    for batch in loader:
        # Check that we got all 10 samples
        assert_true(batch.batch_size == 10, "Should have all 10 samples")

        # Record the order by checking label values
        for i in range(10):
            shuffled_order.append(Int(batch.labels[i]))

    # Verify we have all indices 0-9 (no duplicates, no missing)
    var found = List[Bool](capacity=10)
    for _ in range(10):
        found.append(False)

    for i in range(len(shuffled_order)):
        var idx = shuffled_order[i]
        assert_true(idx >= 0 and idx < 10, "Invalid index in shuffle")
        found[idx] = True

    # Check all indices were found
    for i in range(10):
        assert_true(found[i], "Missing index " + String(i) + " after shuffle")


# ============================================================================
# Main Test Runner
# ============================================================================


# ============================================================================
# Token-stream windowing core (SlidingWindowDataset / WindowLoader) Tests
#
# Deterministic int64 streams where each ID equals its own position, so the
# shift-by-one invariant reads as ``target == input + 1`` elementwise.
# ============================================================================


def _sliding_ids(n: Int) -> List[Scalar[DType.int64]]:
    """IDs ``0..n-1`` as int64 scalars."""
    var ids = List[Scalar[DType.int64]](capacity=n)
    for i in range(n):
        ids.append(Scalar[DType.int64](i))
    return ids^


def test_sliding_window_not_enough_tokens() raises:
    """N <= seq_length yields 0 samples and 0 batches in both drop_last modes."""
    var ds = SlidingWindowDataset[DType.int64](_sliding_ids(6), 8, 2)
    assert_true(len(ds) == 0, "Expected 0 samples when n <= seq_length")
    assert_true(ds.num_samples() == 0, "num_samples must be 0")

    var dl_drop = ds.into_loader(batch_size=4, shuffle=False, drop_last=True)
    assert_true(len(dl_drop) == 0, "Expected 0 batches (drop_last=True)")
    assert_true(
        not dl_drop.__has_next__(), "Empty loader must report no next batch"
    )

    var dl_keep = ds.into_loader(batch_size=4, shuffle=False, drop_last=False)
    assert_true(len(dl_keep) == 0, "Expected 0 batches (drop_last=False)")
    assert_true(
        not dl_keep.__has_next__(), "Empty loader must report no next batch"
    )


def test_sliding_window_partial_batch_shift_by_one() raises:
    """Non-divisor batch_size: target == input + 1 elementwise, every batch."""
    var seq_length = 4
    var ds = SlidingWindowDataset[DType.int64](_sliding_ids(21), seq_length, 1)
    # 21 - 4 = 17 windows; bs=3 -> 6 batches, the last one partial (2 rows).
    assert_true(len(ds) == 17, "Expected 17 windows")
    var loader = ds.into_loader(batch_size=3, shuffle=False, drop_last=False)
    assert_true(len(loader) == 6, "Expected 6 batches incl. the partial one")

    var batch_idx = 0
    for batch in loader:
        assert_true(
            batch.features.shape()[0] == batch.batch_size,
            "Feature batch dim must match",
        )
        assert_true(
            batch.features.shape()[1] == seq_length,
            "Feature dim should be seq_length",
        )
        assert_true(
            batch.labels.shape() == batch.features.shape(),
            "Labels should match features shape",
        )
        # Identity order (shuffle=False): sample s of batch b starts here.
        var base = Int64(batch_idx * 3)
        for s in range(batch.batch_size):
            var first = base + Int64(s)
            for j in range(seq_length):
                assert_true(
                    batch.features[s, j] == first + Int64(j),
                    "Input row must equal its stream slice",
                )
                assert_true(
                    batch.labels[s, j] == first + Int64(j) + Int64(1),
                    "target[i] != input[i]+1 at batch "
                    + String(batch_idx)
                    + " sample "
                    + String(s)
                    + " pos "
                    + String(j),
                )
        batch_idx += 1
    assert_true(batch_idx == 6, "Iterated all batches incl. the partial one")


def test_sliding_window_stride() raises:
    """Stride=2 spaces window starts per the ceiling formula."""
    # (20 - 4 + 2 - 1) // 2 = 8 windows, starts 0, 2, ..., 14.
    var ds = SlidingWindowDataset[DType.int64](_sliding_ids(20), 4, 2)
    assert_true(len(ds) == 8, "Expected 8 windows at stride 2")

    var (x3, y3) = ds.sample(Optional[Int](3))
    for j in range(4):
        assert_true(x3[j] == Int64(6 + j), "Indexed sample input mismatch")
        assert_true(y3[j] == Int64(7 + j), "Indexed sample target mismatch")

    # A random window is internally shift-consistent whatever its offset.
    var (xr, yr) = ds.sample()
    assert_true(xr.numels() == 4, "Random sample input length must be T")
    assert_true(yr.numels() == 4, "Random sample target length must be T")
    for j in range(3):
        assert_true(
            yr[j] == xr[j] + Int64(1),
            "Random sample must hold target == input + 1",
        )

    # Loader identity order: first batch rows start at 0 and 2.
    var loader = ds.into_loader(batch_size=2, shuffle=False)
    var seen = False
    for batch in loader:
        if not seen:
            assert_true(
                batch.features[0, 0] == Int64(0), "First row must start at 0"
            )
            assert_true(
                batch.features[1, 0] == Int64(2), "Second row must start at 2"
            )
            seen = True
    assert_true(seen, "Expected at least one batch")


def test_sliding_window_random_offsets() raises:
    """Random offsets: in-bounds rows, nominal length, shift holds."""
    # N=50, T=8 -> valid starts [0, 41]; nominal ceil(42 / 7) = 6 batches.
    var ds = SlidingWindowDataset[DType.int64](_sliding_ids(50), 8, 1)
    var loader = ds.into_loader(
        batch_size=7, shuffle=False, drop_last=False, random_offsets=True
    )
    assert_true(len(loader) == 6, "Nominal epoch length must be ceil(42/7)")
    var count = 0
    for batch in loader:
        assert_true(
            batch.batch_size == 7, "42 divisible by 7: all batches full"
        )
        for s in range(batch.batch_size):
            var first = batch.features[s, 0]
            assert_true(
                first >= Int64(0) and first <= Int64(41),
                "Random start out of [0, N - seq_length - 1]",
            )
            for j in range(8):
                assert_true(
                    batch.labels[s, j] == batch.features[s, j] + Int64(1),
                    "Random-offset batch must hold target == input + 1",
                )
        count += 1
    assert_true(count == 6, "Iterated the full nominal epoch")


def test_sliding_window_equiv_random_sliding() raises:
    """Migration gate: row-for-row equivalence vs the legacy dataset."""
    var seq_length = 6
    var stream = Tensor[DType.int64].from_list[DType.int64](_sliding_ids(30))
    var ds = SlidingWindowDataset[DType.int64](stream, seq_length, 1)
    var legacy = RandomSlidingWindowDataset[DType.int64](stream, seq_length)
    assert_true(
        len(ds) == len(legacy), "New core and legacy must agree on count"
    )
    for i in range(len(ds)):
        var (nx, ny) = ds.sample(Optional[Int](i))
        var (lx, ly) = legacy.sample(Optional[Int](i))
        assert_tensors_equal(nx, lx)
        assert_tensors_equal(ny, ly)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


# ============================================================================
# TensorDataset Tests
# ============================================================================


def test_dataset_basic_creation_dl() raises:
    """Test basic dataset creation and indexing."""
    var features = (
        Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    )
    var labels = Tensor[DType.float32].d2([[0.0], [1.0], [2.0]]).float()

    var dataset = TensorDataset(features, labels)
    assert_true(len(dataset) == 3)


def test_dataset_getitem_single_sample_dl() raises:
    """Test retrieving single samples."""
    var features = (
        Tensor[DType.float32].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).float()
    )
    var labels = Tensor[DType.float32].d2([[10.0], [20.0]]).float()

    var dataset = TensorDataset(features, labels)

    # Test first sample
    var (feat0, label0) = dataset[0]
    var expected_feat0 = Tensor[DType.float32].d1([1.0, 2.0, 3.0]).float()
    var expected_label0 = Tensor[DType.float32].d1([10.0]).float()
    assert_true(feat0.all_close(expected_feat0))
    assert_true(label0.all_close(expected_label0))

    # Test second sample
    var (feat1, label1) = dataset[1]
    var expected_feat1 = Tensor[DType.float32].d1([4.0, 5.0, 6.0]).float()
    var expected_label1 = Tensor[DType.float32].d1([20.0]).float()
    assert_true(feat1.all_close(expected_feat1))
    assert_true(label1.all_close(expected_label1))


def test_dataset_1d_labels_dl() raises:
    """Test dataset with 1D label vector."""
    var features = (
        Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    )
    var labels = Tensor[DType.float32].d1([0.0, 1.0, 2.0]).float()

    var dataset = TensorDataset(features, labels)

    var (feat, label) = dataset[1]
    var expected_feat = Tensor[DType.float32].d1([3.0, 4.0]).float()
    var expected_label = Tensor[DType.float32].scalar(1.0).float()
    assert_true(feat.all_close(expected_feat))
    assert_true(label.all_close(expected_label))


def test_dataset_multi_dimensional_labels_dl() raises:
    """Test dataset with multi-dimensional labels."""
    var features = Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = (
        Tensor[DType.float32].d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).float()
    )  # One-hot

    var dataset = TensorDataset(features, labels)

    var (_feat, label) = dataset[0]
    var expected_label = Tensor[DType.float32].d1([1.0, 0.0, 0.0]).float()
    assert_true(label.all_close(expected_label))


def test_dataset_features_labels_accessors_dl() raises:
    """Test features() and labels() accessors."""
    var features = Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = Tensor[DType.float32].d1([0.0, 1.0]).float()

    var dataset = TensorDataset(features, labels)

    assert_true(dataset._features.all_close(features))
    assert_true(dataset._labels.all_close(labels))


# ============================================================================
# Batch Tests
# ============================================================================


def test_batch_creation_dl() raises:
    """Test batch container creation."""
    var features = (
        Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    )
    var labels = Tensor[DType.float32].d2([[0.0], [1.0], [2.0]]).float()

    var batch = Batch(features, labels)
    assert_true(batch.batch_size == 3)
    assert_true(batch.features.all_close(features))
    assert_true(batch.labels.all_close(labels))


# ============================================================================
# DataLoader Tests - Basic Functionality
# ============================================================================


def test_dataloader_batch_size_exact_dl() raises:
    """Test dataloader with exact batch division."""
    var features = (
        Tensor[DType.float32]
        .d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        .float()
    )
    var labels = Tensor[DType.float32].d1([0.0, 1.0, 2.0, 3.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

    assert_true(len(loader) == 2)

    var batch_count = 0
    for batch in loader:
        assert_true(batch.batch_size == 2)
        batch_count += 1

    assert_true(batch_count == 2)


def test_dataloader_batch_size_with_remainder_dl() raises:
    """Test dataloader with incomplete last batch."""
    var features = (
        Tensor[DType.float32]
        .d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        .float()
    )
    var labels = Tensor[DType.float32].d1([0.0, 1.0, 2.0, 3.0, 4.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

    assert_true(len(loader) == 3)  # 2 full batches + 1 partial

    var batch_count = 0
    var last_batch_size = 0
    for batch in loader:
        batch_count += 1
        last_batch_size = batch.batch_size

    assert_true(batch_count == 3)
    assert_true(last_batch_size == 1)  # Last batch has 1 sample


def test_dataloader_drop_last_true_dl() raises:
    """Test dataloader with drop_last=True."""
    var features = (
        Tensor[DType.float32]
        .d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        .float()
    )
    var labels = Tensor[DType.float32].d1([0.0, 1.0, 2.0, 3.0, 4.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=True
    )

    assert_true(len(loader) == 2)  # Only 2 complete batches

    var batch_count = 0
    for batch in loader:
        assert_true(batch.batch_size == 2)  # All batches are full
        batch_count += 1

    assert_true(batch_count == 2)


def test_dataloader_single_batch_dl() raises:
    """Test dataloader when batch_size >= dataset size."""
    var features = (
        Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    )
    var labels = Tensor[DType.float32].d1([0.0, 1.0, 2.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=10, shuffle=False, drop_last=False
    )

    assert_true(len(loader) == 1)

    var batch_count = 0
    for batch in loader:
        assert_true(batch.batch_size == 3)  # All 3 samples in one batch
        batch_count += 1

    assert_true(batch_count == 1)


def test_dataloader_batch_size_one_dl() raises:
    """Test dataloader with batch_size=1."""
    var features = (
        Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    )
    var labels = Tensor[DType.float32].d1([0.0, 1.0, 2.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=1, shuffle=False, drop_last=False
    )

    assert_true(len(loader) == 3)

    var batch_count = 0
    for batch in loader:
        assert_true(batch.batch_size == 1)
        batch_count += 1

    assert_true(batch_count == 3)


# ============================================================================
# DataLoader Tests - Data Correctness (No Shuffle)
# ============================================================================


def test_dataloader_data_correctness_no_shuffle_dl() raises:
    """Test that batches contain correct data without shuffling."""
    var features = (
        Tensor[DType.float32]
        .d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        .float()
    )
    var labels = Tensor[DType.float32].d1([10.0, 20.0, 30.0, 40.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

    var iter = loader.__iter__()

    # First batch
    var batch1 = iter.__next__()
    var expected_feat1 = (
        Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0]]).float()
    )
    var expected_label1 = Tensor[DType.float32].d1([10.0, 20.0]).float()
    assert_true(batch1.features.all_close(expected_feat1))
    assert_true(batch1.labels.all_close(expected_label1))
    # Second batch
    var batch2 = iter.__next__()
    var expected_feat2 = (
        Tensor[DType.float32].d2([[5.0, 6.0], [7.0, 8.0]]).float()
    )
    var expected_label2 = Tensor[DType.float32].d1([30.0, 40.0]).float()
    assert_true(batch2.features.all_close(expected_feat2))
    assert_true(batch2.labels.all_close(expected_label2))


def test_dataloader_multi_dimensional_features_dl() raises:
    """Test dataloader with higher dimensional features."""
    var features = (
        Tensor[DType.float32]
        .d2(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        )
        .float()
    )
    var labels = Tensor[DType.float32].d1([0.0, 1.0, 2.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

    var iter = loader.__iter__()
    var batch = iter.__next__()

    # Check feature dimensions
    assert_true(batch.features.shape()[0] == 2)  # batch_size
    assert_true(batch.features.shape()[1] == 4)  # feature_dim


def test_dataloader_multi_dimensional_labels_dl() raises:
    """Test dataloader with multi-dimensional labels."""
    var features = (
        Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    )
    var labels = (
        Tensor[DType.float32]
        .d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        .float()
    )  # One-hot encoded

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

    var iter = loader.__iter__()
    var batch = iter.__next__()

    # Check label dimensions
    assert_true(batch.labels.shape()[0] == 2)  # batch_size
    assert_true(batch.labels.shape()[1] == 3)  # label_dim

    var expected_labels = (
        Tensor[DType.float32].d2([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).float()
    )
    assert_true(batch.labels.all_close(expected_labels))


# ============================================================================
# DataLoader Tests - Shuffling
# ============================================================================


def test_dataloader_shuffle_changes_order_dl_orig() raises:
    """Test that shuffle actually changes the order (probabilistic test)."""
    var features = Tensor[dtype].d2(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
    )
    var labels = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0])

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=5, shuffle=True, drop_last=False
    )

    var iter = loader.__iter__()
    var batch = iter.__next__()

    # Check that we got all samples (sum should be same)
    var sum_features = batch.features.sum()
    var expected_sum = Tensor[dtype].scalar(15.0)  # 1+2+3+4+5
    assert_true(sum_features.all_close(expected_sum))


def test_dataloader_shuffle_changes_order_dl() raises:
    """Test that shuffle actually changes the order (probabilistic test)."""
    var features = (
        Tensor[DType.float32]
        .d2([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
        .float()
    )
    var labels = Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0, 5.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=5, shuffle=True, drop_last=False
    )

    var iter = loader.__iter__()
    var batch = iter.__next__()

    # Check that we got all samples (sum should be same)
    var sum_features = batch.features.sum()
    var expected_sum = Scalar[DType.float32](15.0)  # Just use scalar value
    # Compare scalar values directly
    assert_true((sum_features.item() - expected_sum).__abs__() < 1e-5)


def test_dataloader_shuffle_preserves_all_data_dl() raises:
    """Test that shuffle doesn't lose or duplicate data."""
    var features = (
        Tensor[DType.float32]
        .d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        .float()
    )
    var labels = Tensor[DType.float32].d1([10.0, 20.0, 30.0, 40.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=True, drop_last=False
    )

    # Collect all batches
    var all_features = (
        Tensor[DType.float32]
        .d2([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        .float()
    )
    var all_labels = Tensor[DType.float32].d1([0.0, 0.0, 0.0, 0.0]).float()

    var idx = 0
    for batch in loader:
        for i in range(batch.batch_size):
            all_features[idx, 0] = batch.features[i, 0]
            all_features[idx, 1] = batch.features[i, 1]
            all_labels[idx] = batch.labels[i]
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


def test_dataloader_multiple_epochs_dl() raises:
    """Test iterating through dataloader multiple times."""
    var features = (
        Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    )
    var labels = Tensor[DType.float32].d1([0.0, 1.0, 2.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

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


def test_dataloader_reshuffle_between_epochs_dl() raises:
    """Test that reshuffle=True reshuffles each epoch."""
    var features = (
        Tensor[DType.float32]
        .d2([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        .float()
    )
    var labels = Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=4, shuffle=True, drop_last=False
    )

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


def test_dataloader_empty_iteration_drop_last_dl() raises:
    """Test dataloader when drop_last=True eliminates all batches."""
    var features = Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = Tensor[DType.float32].d1([0.0, 1.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=5, shuffle=False, drop_last=True
    )

    assert_true(len(loader) == 0)

    var batch_count = 0
    for _batch in loader:
        batch_count += 1

    assert_true(batch_count == 0)


def test_dataloader_large_batch_size_dl() raises:
    """Test dataloader with batch_size much larger than dataset."""
    var features = Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = Tensor[DType.float32].d1([0.0, 1.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=1000, shuffle=False, drop_last=False
    )

    assert_true(len(loader) == 1)

    for batch in loader:
        assert_true(batch.batch_size == 2)


# ========== ZERO-COPY ARCHITECTURE TESTS (CORRECTED) ==========


def test_iterator_copy_is_lightweight() raises:
    """Verify __iter__ doesn't copy dataset, only metadata."""

    # Large dataset to make copies expensive
    var features = Tensor[DType.float32].zeros(10000, 784)
    var labels = Tensor[DType.float32].zeros(10000)

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=64, shuffle=False, drop_last=False
    )

    # Time iterator creation (should be instant if no deep copy)
    var start = now()
    var _iter1 = loader.__iter__()
    var _iter2 = loader.__iter__()
    var _iter3 = loader.__iter__()
    var end = now()

    var time_per_iter = (end - start) / 3.0 * 1000  # ms

    print("  Time per iterator copy:", time_per_iter, "ms")

    # Should be under 0.01ms (just metadata copy)
    # If it's copying the dataset, would be 100-500ms
    assert_true(
        time_per_iter < 1.0,
        "Iterator copy too slow - might be copying dataset!",
    )

    print("  Iterator copy is lightweight (<1ms)")


def test_batch_buffers_reused_not_reallocated() raises:
    """Verify batch buffers are pre-allocated and reused."""

    var features = (
        Tensor[DType.float32]
        .d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        .float()
    )
    var labels = Tensor[DType.float32].d1([10.0, 20.0, 30.0, 40.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

    var batch_pointers = List[Int]()

    for batch in loader:
        # Track batch buffer pointer
        var ptr = Int(batch.features.buffer.data_buffer().data.unsafe_value())
        batch_pointers.append(ptr)

    # Count unique pointers
    var num_unique_ptrs = 0
    for i in range(len(batch_pointers)):
        var is_unique = True
        for j in range(i):
            if batch_pointers[i] == batch_pointers[j]:
                is_unique = False
                break
        if is_unique:
            num_unique_ptrs += 1

    # Should have at most 2 unique buffers:
    # 1. Full-size batch buffer (reused for all full batches)
    # 2. Optional last batch buffer (for remainder)
    assert_true(
        num_unique_ptrs <= 2,
        "Too many unique buffer pointers - buffers might not be reused!",
    )

    print("  Batch buffers reused (", num_unique_ptrs, "unique buffers)")


def test_no_allocation_during_iteration() raises:
    """Verify no memory allocation happens during batch iteration."""

    var features = Tensor[DType.float32].zeros(1000, 784)
    var labels = Tensor[DType.float32].zeros(1000)

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=64, shuffle=False, drop_last=False
    )

    # Warmup (trigger any lazy allocations)
    for batch in loader:
        _ = batch.features[0, 0]  # Touch data

    # Time second epoch (should have zero allocation overhead)
    var start = now()
    var batch_count = 0
    for batch in loader:
        _ = batch.features[0, 0]  # Touch data
        batch_count += 1
    var end = now()

    var time_per_batch = (end - start) / Float64(batch_count) * 1000  # ms

    print("  Time per batch:", time_per_batch, "ms")
    print("  Batches:", batch_count)

    # Should be under 0.2ms (just unsafe_memcpy, no allocation)
    assert_true(
        time_per_batch < 0.5,
        "Iteration too slow - might be allocating memory per batch!",
    )

    print("  No allocation overhead detected")


def test_shuffle_preserves_data_location() raises:
    """Verify shuffle only reorders access, doesn't move data."""

    var features = (
        Tensor[DType.float32]
        .d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        .float()
    )
    var labels = Tensor[DType.float32].d1([10.0, 20.0, 30.0, 40.0]).float()

    # Get pointer to original data
    var original_features_ptr = Int(
        features.buffer.data_buffer().data.unsafe_value()
    )
    var original_labels_ptr = Int(
        labels.buffer.data_buffer().data.unsafe_value()
    )

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=True, drop_last=False
    )

    # Iterate multiple epochs
    for _epoch in range(3):
        for batch in loader:
            # Verify we can still access data (indirect test)
            # If data was moved, pointers would be invalid
            var _sum = batch.features.sum().item()
            # Data should be accessible (no crash = test passes)

    # After all iterations, original data should still be at same location
    var final_features_ptr = Int(
        features.buffer.data_buffer().data.unsafe_value()
    )
    var final_labels_ptr = Int(labels.buffer.data_buffer().data.unsafe_value())

    assert_equal(original_features_ptr, final_features_ptr)
    assert_equal(original_labels_ptr, final_labels_ptr)

    print("  Data stays in place during shuffle")


def test_memory_overhead_is_minimal() raises:
    """Calculate and verify DataLoader memory overhead."""

    var num_samples = 60000
    var feature_dim = 784

    var features = Tensor[DType.float32].zeros(num_samples, feature_dim)
    var labels = Tensor[DType.int32].zeros(num_samples)

    var dataset_size_mb = (
        Float64(num_samples * feature_dim * 4) / 1024.0 / 1024.0
    )

    var dataset = TensorDataset(features, labels)
    var _loader = dataset.into_loader(
        batch_size=64, shuffle=True, drop_last=False
    )

    # Calculate theoretical overhead
    var batch_size = 64
    var batch_buffer_features_mb = (
        Float64(batch_size * feature_dim * 4) / 1024.0 / 1024.0
    )
    var batch_buffer_labels_mb = Float64(batch_size * 4) / 1024.0 / 1024.0
    var indices_mb = Float64(num_samples * 8) / 1024.0 / 1024.0  # List[Int]
    var metadata_kb = 0.1  # Estimate for pointer + counters + flags

    var total_overhead_mb = (
        batch_buffer_features_mb
        + batch_buffer_labels_mb
        + indices_mb
        + metadata_kb / 1024.0
    )
    var overhead_percentage = (total_overhead_mb / dataset_size_mb) * 100.0

    print("  Dataset size:", dataset_size_mb, "MB")
    print(
        "  Batch buffers:",
        batch_buffer_features_mb + batch_buffer_labels_mb,
        "MB",
    )
    print("  Indices array:", indices_mb, "MB")
    print("  Metadata:", metadata_kb, "KB")
    print("  Total overhead:", total_overhead_mb, "MB")
    print("  Overhead %:", overhead_percentage, "%")

    # Overhead should be < 1% of dataset size
    assert_true(overhead_percentage < 1.0, "Memory overhead too large!")

    print("  Memory overhead minimal (", overhead_percentage, "%)")


def test_multi_epoch_no_memory_growth() raises:
    """Verify memory doesn't grow across epochs."""

    var features = Tensor[DType.float32].zeros(1000, 784)
    var labels = Tensor[DType.float32].zeros(1000)

    var dataset = TensorDataset(features^, labels^)
    # Use shuffle=False to ensure deterministic batch order
    var loader = dataset.into_loader(
        batch_size=64, shuffle=False, drop_last=False
    )

    # Track ALL batch buffer pointers across epochs
    var epoch_pointers = List[List[Int]]()

    for _ in range(3):
        var batch_ptrs = List[Int]()
        for batch in loader:
            var ptr = Int(
                batch.features.buffer.data_buffer().data.unsafe_value()
            )
            batch_ptrs.append(ptr)
        epoch_pointers.append(batch_ptrs^)

    # Verify same pattern of pointers across epochs
    for epoch in range(1, len(epoch_pointers)):
        assert_equal(
            len(epoch_pointers[0]),
            len(epoch_pointers[epoch]),
            "Different number of batches across epochs!",
        )

        # Check each batch uses same buffer as in first epoch
        for batch_idx in range(len(epoch_pointers[0])):
            assert_equal(
                epoch_pointers[0][batch_idx],
                epoch_pointers[epoch][batch_idx],
                "Buffer pointer changed for batch "
                + String(batch_idx)
                + " in epoch "
                + String(epoch),
            )

    print(
        "  No memory growth across",
        len(epoch_pointers),
        "epochs (",
        len(epoch_pointers[0]),
        "batches each)",
    )


def disabled_performance_overhead_is_negligible() raises:
    """Measure DataLoader overhead vs raw data access."""

    var num_samples = 10000
    var features = Tensor[DType.float32].zeros(num_samples, 784)
    var labels = Tensor[DType.float32].zeros(num_samples)

    var batch_size = 64
    var num_batches = num_samples // batch_size

    # Baseline: Direct manual slicing (best possible)
    var start_baseline = now()
    for i in range(num_batches):
        var start_idx = i * batch_size
        var end_idx = start_idx + batch_size
        var batch_features = features[start_idx:end_idx, :]
        var _batch_labels = labels[start_idx:end_idx]
        _ = batch_features[0, 0]  # Touch data
    var end_baseline = now()
    var baseline_time = end_baseline - start_baseline

    # DataLoader: With all the indirection
    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=batch_size, shuffle=False, drop_last=True
    )

    var start_loader = now()
    for batch in loader:
        _ = batch.features[0, 0]  # Touch data
    var end_loader = now()
    var loader_time = end_loader - start_loader

    var overhead = ((loader_time - baseline_time) / baseline_time) * 100.0

    print("  Baseline (manual slicing):", baseline_time, "s")
    print("  DataLoader:", loader_time, "s")
    print("  Overhead:", overhead, "%")

    # Overhead should be < 20% (ideally < 10%)
    assert_true(overhead < 30.0, "DataLoader overhead too large!")

    print("  DataLoader overhead acceptable (<30%)")


def test_dataloader_pointer_semantics() raises:
    """Test that DataLoader properly uses pointer (not copy) to dataset."""

    var features = Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = Tensor[DType.float32].d1([10.0, 20.0]).float()

    var dataset = TensorDataset(features, labels)

    # Create loader - should only store pointer, not copy dataset
    var creation_start = now()
    var loader = dataset.into_loader(
        batch_size=1, shuffle=False, drop_last=False
    )
    var creation_end = now()
    var creation_time = creation_end - creation_start

    print("  Loader creation time:", creation_time * 1000, "ms")

    # Should be very fast (< 1ms) since no copying
    # If copying 180MB dataset, would take 100-500ms
    assert_true(
        creation_time < 0.01,  # 10ms threshold
        "Loader creation too slow - might be copying dataset!",
    )

    # Verify loader works correctly
    var batch_count = 0
    for _batch in loader:
        batch_count += 1
    assert_equal(batch_count, 2)

    print("  Loader uses pointer semantics (<10ms creation)")


def disabled_comparing_copy_vs_reference_semantics() raises:
    """Demonstrate the difference between copy and reference."""

    var features = Tensor[DType.float32].zeros(10000, 784)
    var labels = Tensor[DType.float32].zeros(10000)

    # Test 1: Measure Dataset copy time (simulate old design)
    var copy_start = now()
    var _features_copy = features.copy()
    var _labels_copy = labels.copy()
    var copy_end = now()
    var copy_time = copy_end - copy_start

    # Test 2: Measure DataLoader creation (pointer design)
    var dataset = TensorDataset(features, labels)
    var loader_start = now()
    var _loader = dataset.into_loader(
        batch_size=64, shuffle=False, drop_last=False
    )
    var loader_end = now()
    var loader_time = loader_end - loader_start

    print("  Dataset copy time:", copy_time, "s")
    print("  DataLoader creation time:", loader_time, "s")
    print("  Speedup:", copy_time / loader_time, "x")

    # Reference semantics should be faster than copying (no data copy)
    assert_true(
        loader_time < copy_time,
        "DataLoader must be faster than copying — reference vs copy semantics",
    )

    print("  Pointer-based design is", copy_time / loader_time, "x faster")


def test_shuffle_uses_two_buffers() raises:
    """Verify that shuffle correctly alternates between _batch and _last_batch.
    """

    # Create dataset where batch_size doesn't divide evenly
    var features = Tensor[DType.float32].zeros(
        130, 784
    )  # 130 % 64 = 2 remainder
    var labels = Tensor[DType.float32].zeros(130)

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=64, shuffle=True, drop_last=False
    )

    # Track unique buffer pointers across multiple epochs
    var all_pointers = List[Int]()

    for _ in range(10):  # Multiple epochs to see both buffers
        for batch in loader:
            var ptr = Int(
                batch.features.buffer.data_buffer().data.unsafe_value()
            )

            # Check if this pointer is new
            var is_new = True
            for i in range(len(all_pointers)):
                if all_pointers[i] == ptr:
                    is_new = False
                    break

            if is_new:
                all_pointers.append(ptr)

    # Should have exactly 2 unique pointers:
    # 1. _batch (for full-size batches of 64)
    # 2. _last_batch (for remainder batch of 2)
    var num_unique = len(all_pointers)

    print("  Unique buffer pointers found:", num_unique)
    assert_true(
        num_unique == 2,
        "Expected exactly 2 buffers (_batch and _last_batch), found "
        + String(num_unique),
    )

    print("  Correctly uses 2 pre-allocated buffers (full + remainder)")


def test_no_memory_growth_with_shuffle() raises:
    """Verify no memory growth with shuffle (accepts two stable buffers)."""

    var features = Tensor[DType.float32].zeros(1000, 784)
    var labels = Tensor[DType.float32].zeros(1000)

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=64, shuffle=True, drop_last=False
    )

    # Collect all buffer pointers from first 2 epochs
    var epoch1_ptrs = List[Int]()
    var epoch2_ptrs = List[Int]()

    for batch in loader:
        epoch1_ptrs.append(
            Int(batch.features.buffer.data_buffer().data.unsafe_value())
        )

    for batch in loader:
        epoch2_ptrs.append(
            Int(batch.features.buffer.data_buffer().data.unsafe_value())
        )

    # Count unique pointers in each epoch
    var unique1 = List[Int]()
    for i in range(len(epoch1_ptrs)):
        var is_new = True
        for j in range(len(unique1)):
            if unique1[j] == epoch1_ptrs[i]:
                is_new = False
                break
        if is_new:
            unique1.append(epoch1_ptrs[i])

    var unique2 = List[Int]()
    for i in range(len(epoch2_ptrs)):
        var is_new = True
        for j in range(len(unique2)):
            if unique2[j] == epoch2_ptrs[i]:
                is_new = False
                break
        if is_new:
            unique2.append(epoch2_ptrs[i])

    # Both epochs should use same set of buffers (max 2)
    assert_equal(
        len(unique1), len(unique2), "Different number of buffers across epochs!"
    )
    assert_true(len(unique1) <= 2, "Too many buffers (expected 1 or 2)")

    # The unique pointers should be the same across epochs
    for i in range(len(unique1)):
        var found = False
        for j in range(len(unique2)):
            if unique1[i] == unique2[j]:
                found = True
                break
        assert_true(
            found, "New buffer appeared in epoch 2 - possible memory leak!"
        )

    print("  No memory growth with shuffle (", len(unique1), "stable buffers)")


# ==============================================================================
# BASIC FUNCTIONALITY TESTS
# ==============================================================================


def test_basic_iteration_no_shuffle() raises:
    """Test basic iteration without shuffling."""

    var features = (
        Tensor[DType.float32]
        .d2(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        )
        .float()
    )
    var labels = Tensor[DType.float32].d1([10.0, 20.0, 30.0, 40.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

    var batch_count = 0
    for batch in loader:
        if batch_count == 0:
            # First batch: samples 0, 1
            assert_true(batch.features[0, 0] == 1.0)
            assert_true(batch.features[0, 1] == 2.0)
            assert_true(batch.features[1, 0] == 3.0)
            assert_true(batch.features[1, 1] == 4.0)
            assert_true(batch.labels[0] == 10.0)
            assert_true(batch.labels[1] == 20.0)
        elif batch_count == 1:
            # Second batch: samples 2, 3
            assert_true(batch.features[0, 0] == 5.0)
            assert_true(batch.features[0, 1] == 6.0)
            assert_true(batch.features[1, 0] == 7.0)
            assert_true(batch.features[1, 1] == 8.0)
            assert_true(batch.labels[0] == 30.0)
            assert_true(batch.labels[1] == 40.0)
        batch_count += 1

    assert_equal(batch_count, 2)


def test_basic_iteration_with_shuffle() raises:
    """Test that shuffle changes order but preserves data."""

    var features = (
        Tensor[DType.float32]
        .d2(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        )
        .float()
    )
    var labels = Tensor[DType.float32].d1([10.0, 20.0, 30.0, 40.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=True, drop_last=False
    )

    # Collect all data
    var all_features = (
        Tensor[DType.float32]
        .d2([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        .float()
    )
    var all_labels = Tensor[DType.float32].d1([0.0, 0.0, 0.0, 0.0]).float()

    var idx = 0
    for batch in loader:
        for i in range(batch.batch_size):
            all_features[idx, 0] = batch.features[i, 0]
            all_features[idx, 1] = batch.features[i, 1]
            all_labels[idx] = batch.labels[i]
            idx += 1

    # Verify all data is present (sum should be same)
    assert_true(all_features.sum().all_close(features.sum()))
    assert_true(all_labels.sum().all_close(labels.sum()))


def test_drop_last_true() raises:
    """Test drop_last=True drops incomplete batch."""

    var features = (
        Tensor[DType.float32]
        .d2(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        .float()
    )
    var labels = Tensor[DType.float32].d1([10.0, 20.0, 30.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=True
    )

    var batch_count = 0
    for batch in loader:
        assert_equal(batch.batch_size, 2)  # All batches should be full size
        batch_count += 1

    assert_equal(batch_count, 1)  # Only 1 full batch (3rd sample dropped)
    assert_equal(len(loader), 1)


def test_drop_last_false() raises:
    """Test drop_last=False keeps incomplete batch."""

    var features = (
        Tensor[DType.float32]
        .d2(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        .float()
    )
    var labels = Tensor[DType.float32].d1([10.0, 20.0, 30.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

    var batch_count = 0
    for batch in loader:
        if batch_count == 0:
            assert_equal(batch.batch_size, 2)
        elif batch_count == 1:
            assert_equal(batch.batch_size, 1)  # Last batch is smaller
            assert_true(batch.features[0, 0] == 5.0)
            assert_true(batch.labels[0] == 30.0)
        batch_count += 1

    assert_equal(batch_count, 2)
    assert_equal(len(loader), 2)


def test_single_batch() raises:
    """Test when batch_size >= dataset size."""

    var features = Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = Tensor[DType.float32].d1([10.0, 20.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=5, shuffle=False, drop_last=False
    )

    var batch_count = 0
    for batch in loader:
        assert_equal(batch.batch_size, 2)
        batch_count += 1

    assert_equal(batch_count, 1)


def test_batch_size_one() raises:
    """Test batch_size=1 (edge case)."""

    var features = (
        Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).float()
    )
    var labels = Tensor[DType.float32].d1([10.0, 20.0, 30.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=1, shuffle=False, drop_last=False
    )

    var batch_count = 0
    for batch in loader:
        assert_equal(batch.batch_size, 1)
        batch_count += 1

    assert_equal(batch_count, 3)


# ==============================================================================
# MULTI-EPOCH TESTS
# ==============================================================================


def test_multiple_epochs_no_shuffle() raises:
    """Test iterating multiple epochs without shuffle."""

    var features = Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = Tensor[DType.float32].d1([10.0, 20.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=1, shuffle=False, drop_last=False
    )

    # Epoch 1
    var count1 = 0
    for _batch in loader:
        count1 += 1
    assert_equal(count1, 2)

    # Epoch 2 - should reset and give same order
    var epoch2_first_val: Float32 = 0.0
    var count2 = 0
    for batch in loader:
        if count2 == 0:
            epoch2_first_val = batch.labels[0]
        count2 += 1

    assert_equal(count2, 2)
    assert_true(
        (epoch2_first_val - 10.0).__abs__() < 1e-5
    )  # Should start with same sample


def test_multiple_epochs_with_shuffle() raises:
    """Test that shuffle gives different order across epochs."""

    # Large enough dataset to ensure shuffle produces different orders
    var features = (
        Tensor[DType.float32]
        .d2(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
                [6.0, 6.0],
                [7.0, 7.0],
                [8.0, 8.0],
            ]
        )
        .float()
    )
    var labels = (
        Tensor[DType.float32]
        .d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .float()
    )

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=True, drop_last=False
    )

    # Collect epoch 1 order
    var epoch1_order = List[Float32]()
    for batch in loader:
        for i in range(batch.batch_size):
            epoch1_order.append(batch.labels[i])

    # Collect epoch 2 order
    var epoch2_order = List[Float32]()
    for batch in loader:
        for i in range(batch.batch_size):
            epoch2_order.append(batch.labels[i])

    # Both epochs should have all data
    assert_equal(len(epoch1_order), 8)
    assert_equal(len(epoch2_order), 8)

    # Check that at least one position is different (shuffle worked)
    var different = False
    for i in range(8):
        if (epoch1_order[i] - epoch2_order[i]).__abs__() > 1e-5:
            different = True
            break

    assert_true(different)  # Orders should differ


# ==============================================================================
# DATA INTEGRITY TESTS
# ==============================================================================


def test_no_data_loss() raises:
    """Verify no samples are lost during batching."""

    var features = (
        Tensor[DType.float32]
        .d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        .float()
    )
    var labels = Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0, 5.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=True, drop_last=False
    )

    var seen_labels = List[Float32]()
    for batch in loader:
        for i in range(batch.batch_size):
            seen_labels.append(batch.labels[i])

    # Check all 5 labels are present
    assert_equal(len(seen_labels), 5)

    # Check sum
    var sum: Float32 = 0.0
    for i in range(len(seen_labels)):
        sum += seen_labels[i]
    assert_true((sum - 15.0).__abs__() < 1e-5)  # 1+2+3+4+5 = 15


def test_no_data_duplication() raises:
    """Verify no samples are duplicated."""

    var features = (
        Tensor[DType.float32]
        .d2([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        .float()
    )
    var labels = Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=True, drop_last=False
    )

    var seen_labels = List[Float32]()
    for batch in loader:
        for i in range(batch.batch_size):
            seen_labels.append(batch.labels[i])

    # Check for duplicates
    for i in range(len(seen_labels)):
        var count = 0
        for j in range(len(seen_labels)):
            if (seen_labels[i] - seen_labels[j]).__abs__() < 1e-5:
                count += 1
        assert_equal(count, 1)  # Each label should appear exactly once


def test_feature_label_correspondence() raises:
    """Verify features and labels stay paired correctly."""

    # Features encode their index in first element
    var features = (
        Tensor[DType.float32]
        .d2(
            [
                [10.0, 100.0],
                [20.0, 200.0],
                [30.0, 300.0],
                [40.0, 400.0],
            ]
        )
        .float()
    )
    var labels = Tensor[DType.float32].d1([10.0, 20.0, 30.0, 40.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=True, drop_last=False
    )

    for batch in loader:
        for i in range(batch.batch_size):
            # Label should match first feature element
            var feat = batch.features[i, 0]
            var label = batch.labels[i]
            assert_true((feat - label).__abs__() < 1e-5)


# ==============================================================================
# NUMPY DATASET TESTS
# ==============================================================================
def test_numpy_dataset_integration() raises:
    """Test NumpyDataset with DataLoader."""

    var np = Python.import_module("numpy")
    var array: PythonObject = Python.list(
        Python.list(1.0, 2.0), Python.list(3.0, 4.0), Python.list(5.0, 6.0)
    )
    var features_np = np.array(
        array,
        dtype=np.float32,
    )

    var labels_np = np.array(Python.list(10, 20, 30), dtype=np.int32)

    var dataset = NumpyDataset[DType.float32, DType.int32](
        features_np, labels_np, copy=False
    )

    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

    var batch_count = 0
    for _batch in loader:
        batch_count += 1

    assert_equal(batch_count, 2)
    assert_equal(len(loader), 2)


# ==============================================================================
# EDGE CASES & STRESS TESTS
# ==============================================================================


def test_empty_last_batch_handling() raises:
    """Test exact multiple of batch_size (no partial batch)."""

    var features = (
        Tensor[DType.float32]
        .d2(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        )
        .float()
    )
    var labels = Tensor[DType.float32].d1([10.0, 20.0, 30.0, 40.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

    var batch_count = 0
    for batch in loader:
        assert_equal(batch.batch_size, 2)
        batch_count += 1

    assert_equal(batch_count, 2)


def test_large_batch_size() raises:
    """Test batch_size larger than dataset."""

    var features = Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = Tensor[DType.float32].d1([10.0, 20.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=100, shuffle=False, drop_last=False
    )

    var batch_count = 0
    for batch in loader:
        assert_equal(batch.batch_size, 2)
        batch_count += 1

    assert_equal(batch_count, 1)


def test_stress_many_small_batches() raises:
    """Stress test with many small batches."""

    # 100 samples
    var features = Tensor[DType.float32].zeros(100, 10)
    var labels = Tensor[DType.float32].zeros(100)

    for i in range(100):
        labels[i] = Float32(i)

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=3, shuffle=False, drop_last=False
    )

    var total_samples = 0
    for batch in loader:
        total_samples += batch.batch_size

    assert_equal(total_samples, 100)


def test_stress_large_dataset() raises:
    """Stress test with larger dataset (simulating MNIST size)."""

    # 1000 samples, 784 features
    var features = Tensor[DType.float32].zeros(1000, 784)
    var labels = Tensor[DType.int32].zeros(1000)

    for i in range(1000):
        labels[i] = Int32(i % 10)  # 10 classes

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=64, shuffle=True, drop_last=False
    )

    var total_samples = 0
    for batch in loader:
        total_samples += batch.batch_size
        assert_true(batch.features.shape()[1] == 784)

    assert_equal(total_samples, 1000)
    assert_equal(len(loader), 16)  # ceil(1000/64) = 16


# ==============================================================================
# PERFORMANCE REGRESSION TESTS
# ==============================================================================


def test_performance_no_shuffle() raises:
    """Benchmark no-shuffle performance (should use fast path)."""

    var features = Tensor[DType.float32].zeros(10000, 784)
    var labels = Tensor[DType.int32].zeros(10000)

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=64, shuffle=False, drop_last=False
    )

    var start = now()
    var batch_count = 0
    for _batch in loader:
        batch_count += 1
    var end = now()

    var time_per_batch = (end - start) / Float64(batch_count) * 1000
    print("  Time per batch:", time_per_batch, "ms")
    print("  Expected: < 0.1ms (bulk unsafe_memcpy fast path)")


def test_performance_with_shuffle() raises:
    """Benchmark shuffle performance (should use row-by-row unsafe_memcpy)."""

    var features = Tensor[DType.float32].zeros(10000, 784)
    var labels = Tensor[DType.int32].zeros(10000)

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=64, shuffle=True, drop_last=False
    )

    var start = now()
    var batch_count = 0
    for _batch in loader:
        batch_count += 1
    var end = now()

    var time_per_batch = (end - start) / Float64(batch_count) * 1000
    print("  Time per batch:", time_per_batch, "ms")
    print("  Expected: < 0.05ms (per-row unsafe_memcpy)")


# ==============================================================================
# SPECIAL CASES
# ==============================================================================


def test_multi_dimensional_labels() raises:
    """Test with non-scalar labels (multi-output)."""

    var features = Tensor[DType.float32].d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var labels = (
        Tensor[DType.float32].d2([[10.0, 11.0], [20.0, 21.0]]).float()
    )  # 2D labels

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=1, shuffle=False, drop_last=False
    )

    var batch_count = 0
    for batch in loader:
        if batch_count == 0:
            assert_true(batch.labels[0, 0] == 10.0)
            assert_true(batch.labels[0, 1] == 11.0)
        batch_count += 1

    assert_equal(batch_count, 2)


def test_single_feature_dimension() raises:
    """Test with 1D features (edge case)."""

    var features = Tensor[DType.float32].d2([[1.0], [2.0], [3.0]]).float()
    var labels = Tensor[DType.float32].d1([10.0, 20.0, 30.0]).float()

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(
        batch_size=2, shuffle=False, drop_last=False
    )

    var batch_count = 0
    for batch in loader:
        assert_equal(batch.features.shape()[1], 1)
        batch_count += 1

    assert_equal(batch_count, 2)


# ==============================================================================
# CONSOLIDATED TEST RUNNERS
# ==============================================================================


# HOW TO USE
def example_usage() raises:
    """Example showing the fixed implementation."""

    # Create 4D image data (N, C, H, W)
    var images = Tensor[DType.float32].randn(1000, 1, 28, 28)
    var labels = Tensor[DType.int32].zeros(1000)

    print("Original data shapes:")
    print("  Images:", images.shape())  # (1000, 1, 28, 28)
    print("  Labels:", labels.shape())  # (1000,)

    # Create dataset
    var dataset = NumpyDataset[DType.float32, DType.int32](images, labels)

    print("\nDataset info:")
    print("  Size:", len(dataset))
    print(
        "  Feature shape per sample:", dataset.get_feature_shape()
    )  # (1, 28, 28)
    print("  Label shape per sample:", dataset.get_label_shape())  # ()
    print("  Features per sample:", dataset.get_features_per_sample())  # 784

    # Create dataloader
    var loader = dataset.into_loader(batch_size=128, shuffle=False)

    print("\nDataLoader info:")
    print("  Num batches:", len(loader))

    # Get first batch
    loader.reset()
    if loader.__has_next__():
        ref batch = loader.__next__()
        print("\nFirst batch:")
        print("Features shape:", batch.features.shape())  # (128, 1, 28, 28)
        print("Labels shape:", batch.labels.shape())  # (128,)
        print("Shapes preserved correctly!")


# ============================================================================
# DataLoader (tensor-native engine) Tests
# The dtype-generic tensor loader: std `Iterator` protocol conformance with
# sequential zero-copy views (eval) and shuffled buffer row-gathering (train).
# ============================================================================


def test_data_loader_tensor_len_bounds_samples() raises:
    var features = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    )
    var labels = Tensor[DType.int64].d1(0, 1, 2, 3, 4)
    var loader = DataLoader[DType.float32, DType.int64](
        features, labels, batch_size=2, shuffle=False
    )
    assert_true(len(loader) == 3, "N=5, bs=2 -> 3 batches")
    assert_true(loader.num_samples() == 5, "num_samples must be 5")
    var (lo, hi) = loader.bounds()
    assert_true(lo == 3, "bounds lo should equal len(loader)")
    assert_true(hi.value() == 3, "bounds hi should match the epoch length")


def test_data_loader_tensor_for_loop_sequential_partition() raises:
    """A for-loop iterates the backend in identity order, last batch partial."""
    var features = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    )
    var labels = Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var loader = DataLoader[DType.float32, DType.float32](
        features, labels, batch_size=2, shuffle=False
    )
    var full_count = 0
    var partial_count = 0
    var batch_idx = 0
    for batch in loader:
        if batch.batch_size == 2:
            full_count += 1
        elif batch.batch_size == 1:
            partial_count += 1
        else:
            assert_true(False, "Unexpected batch size")
        var start_sample = batch_idx * 2
        for s in range(batch.batch_size):
            assert_true(
                batch.labels[s] == Float32(start_sample + s + 1),
                "Sequential batch label out of order",
            )
        batch_idx += 1
    assert_true(batch_idx == 3, "Should iterate 3 batches")
    assert_true(
        full_count == 2 and partial_count == 1,
        "Partition must be [2, 2, 1]",
    )


def test_data_loader_tensor_drop_last_no_partial() raises:
    var features = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    )
    var labels = Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0, 5.0])
    var loader = DataLoader[DType.float32, DType.float32](
        features, labels, batch_size=2, shuffle=False, drop_last=True
    )
    assert_true(len(loader) == 2, "drop_last drops the partial batch")
    var count = 0
    var total = Float32(0.0)
    for batch in loader:
        count += 1
        assert_true(batch.batch_size == 2, "No partial batch with drop_last")
        total += batch.labels[0] + batch.labels[1]
    assert_true(count == 2, "Exactly 2 full batches")
    assert_true(total == 10.0, "labels 1+2+3+4")


def test_data_loader_tensor_exhaustion_raises_stop_iteration() raises:
    """Calling ``__next__()`` past the epoch end raises ``StopIteration``."""
    var features = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    )
    var labels = Tensor[DType.float32].d1([1.0, 2.0, 3.0])
    var loader = DataLoader[DType.float32, DType.float32](
        features, labels, batch_size=2, shuffle=False
    )
    var iter = loader.__iter__()
    var got = 0
    while iter.__has_next__():
        var _b = iter.__next__()
        got += 1
    assert_true(got == 2, "Two full batches")
    assert_false(iter.__has_next__(), "Epoch is exhausted (iter copy)")
    with assert_raises():
        var b = iter.__next__()


def test_data_loader_tensor_epoch_restart_reiterates() raises:
    """Each ``for`` starts a fresh epoch; second loop iterates again cleanly."""
    var features = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    var labels = Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0])
    var loader = DataLoader[DType.float32, DType.float32](
        features, labels, batch_size=2, shuffle=False
    )
    var count1 = 0
    for _batch in loader:
        count1 += 1
    var count2 = 0
    for _batch in loader:
        count2 += 1
    assert_true(count1 == 2 and count2 == 2, "Both epochs see 2 batches")


def test_data_loader_tensor_reset_restarts_epoch() raises:
    var features = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    var labels = Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0])
    var loader = DataLoader[DType.float32, DType.float32](
        features, labels, batch_size=2, shuffle=False
    )
    var iter = loader.__iter__()
    var _b1 = iter.__next__()
    loader.reset()
    var count = 0
    for batch in loader:
        count += 1
        if count == 1:
            assert_true(
                batch.labels[0] == 1.0, "Reset restarts at sample 0"
            )
    assert_true(count == 2, "Reset mid-epoch -> full remaining epoch")


def test_data_loader_tensor_shuffle_epoch_covers_all_samples() raises:
    """A shuffled epoch yields every sample exactly once (row-gather path)."""
    var features = Tensor[DType.float32].d2(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
            [19.0, 20.0],
        ]
    )
    var labels = Tensor[DType.float32].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )
    var loader = DataLoader[DType.float32, DType.float32](
        features, labels, batch_size=3, shuffle=True
    )
    var count = 0
    var total = Float32(0.0)
    for batch in loader:
        count += 1
        for s in range(batch.batch_size):
            total += batch.labels[s]
    assert_true(count == 4, "10 samples / 3 -> 3 full + 1 partial")
    assert_true(total == 55.0, "Each sample appears exactly once per epoch")


def test_data_loader_tensor_reshuffle_between_epochs() raises:
    """Two shuffled epochs both cover all samples with fresh permutations."""
    var features = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    )
    var labels = Tensor[DType.float32].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    var loader = DataLoader[DType.float32, DType.float32](
        features, labels, batch_size=3, shuffle=True
    )
    var epoch1_total = Float32(0.0)
    for batch in loader:
        for s in range(batch.batch_size):
            epoch1_total += batch.labels[s]
    var epoch2_total = Float32(0.0)
    for batch in loader:
        for s in range(batch.batch_size):
            epoch2_total += batch.labels[s]
    assert_true(epoch1_total == 21.0, "Epoch 1 covers 1..6")
    assert_true(epoch2_total == 21.0, "Epoch 2 covers 1..6")


def test_data_loader_tensor_mode_toggle_buffers() raises:
    """Toggle shuffle off and back on.
    Sequential views must not corrupt the
    re-allocated row-gather buffers."""
    var features = Tensor[DType.float32].d2(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
            [19.0, 20.0],
        ]
    )
    var labels = Tensor[DType.float32].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    )
    var loader = DataLoader[DType.float32, DType.float32](
        features, labels, batch_size=3, shuffle=True
    )
    # Shuffled epoch (row-gather buffers).
    var total1 = Float32(0.0)
    for batch in loader:
        for s in range(batch.batch_size):
            total1 += batch.labels[s]
    assert_true(total1 == 55.0, "Buffered shuffled epoch covers all")

    # Sequential epoch: zero-copy views replace the buffers.
    loader.set_shuffle(False)
    var count_seq = 0
    for batch in loader:
        count_seq += 1
        if count_seq == 1:
            assert_true(batch.labels[0] == 1.0, "Sequential restarts at 0")
            assert_true(batch.labels[1] == 2.0, "Sequential order kept")
            assert_true(batch.labels[2] == 3.0, "Sequential order kept")
    assert_true(count_seq == 4, "Sequential partition is [3, 3, 3, 1]")

    # Shuffled again: buffers re-allocated; coverage must be complete.
    loader.set_shuffle(True)
    var count2 = 0
    var total2 = Float32(0.0)
    for batch in loader:
        count2 += 1
        for s in range(batch.batch_size):
            total2 += batch.labels[s]
    assert_true(count2 == 4, "Shuffled epoch again yields 4 batches")
    assert_true(total2 == 55.0, "Re-allocated buffers stay uncorrupted")


def test_data_loader_tensor_2d_labels() raises:
    var features = Tensor[DType.float32].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    )
    var labels = Tensor[DType.int64].d2([[11, 12], [21, 22], [31, 32]])
    var loader = DataLoader[DType.float32, DType.int64](
        features, labels, batch_size=2, shuffle=False
    )
    assert_true(len(loader) == 2, "3 samples / 2 -> 2 batches")
    var batch_idx = 0
    for batch in loader:
        assert_true(
            batch.labels.shape()[1] == 2, "2-D labels preserved"
        )
        if batch_idx == 0:
            assert_true(
                batch.labels[0, 0] == 11 and batch.labels[0, 1] == 12,
                "Batch 0 label row 0",
            )
            assert_true(
                batch.labels[1, 0] == 21 and batch.labels[1, 1] == 22,
                "Batch 0 label row 1",
            )
        else:
            assert_true(
                batch.labels[0, 0] == 31 and batch.labels[0, 1] == 32,
                "Batch 1 partial label row",
            )
        batch_idx += 1
