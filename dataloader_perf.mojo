from tenmo import Tensor
from dataloader import NumpyDataset, TensorDataset
from testing import assert_true
from common_utils import now


fn test_validation_faster_than_training() raises:
    """Test that validation batching (no shuffle) is faster than training (shuffled).
    """
    print("\n" + "=" * 80)
    print("Testing: Validation DataLoader Speed vs Training DataLoader Speed")
    print("=" * 80)

    # Create a reasonably large dataset to see timing differences
    var num_samples = 10000
    var feature_dim = 784

    print(
        "Creating dataset:", num_samples, "samples ×", feature_dim, "features"
    )

    var features = Tensor[DType.float32].zeros(num_samples, feature_dim)
    var labels = Tensor[DType.int32].zeros(num_samples)

    # Fill with some data (values don't matter for timing test)
    for i in range(num_samples):
        for j in range(feature_dim):
            features[i, j] = Float32(i + j)
        labels[i] = i % 10

    var dataset = TensorDataset(features, labels)

    var batch_size = 64
    var num_batches = num_samples // batch_size

    print("Batch size:", batch_size)
    print("Number of batches:", num_batches)
    print()

    # Test 1: Training-style (with shuffle) - row-by-row memcpy
    print("Testing TRAINING mode (shuffle=True, row-by-row memcpy)...")
    var train_loader = dataset.into_loader(
        batch_size=batch_size,
        shuffle=True,  # Shuffled = row-by-row memcpy
        drop_last=True,
    )

    var train_start = now()
    var train_iterations = 5  # Multiple iterations for better timing

    for _iteration in range(train_iterations):
        var iter = train_loader.__iter__()
        var batch_count = 0
        while iter.__has_next__():
            var batch = iter.__next__()
            # Simulate minimal processing (just access the data)
            var _ = batch.features[0, 0]
            batch_count += 1

    var train_end = now()
    var train_total_time = train_end - train_start
    var train_time_per_epoch = train_total_time / train_iterations
    var train_time_per_batch = train_time_per_epoch / num_batches

    print(
        "  Total time (",
        train_iterations,
        "epochs):",
        train_total_time,
        "seconds",
    )
    print("  Time per epoch:", train_time_per_epoch, "seconds")
    print("  Time per batch:", train_time_per_batch, "seconds")
    print()

    # Test 2: Validation-style (no shuffle) - bulk memcpy
    print("Testing VALIDATION mode (shuffle=False, bulk memcpy)...")
    var val_loader = dataset.into_loader(
        batch_size=batch_size,
        shuffle=False,  # Not shuffled = bulk memcpy
        drop_last=True,
    )

    var val_start = now()
    var val_iterations = 5

    for _iteration in range(val_iterations):
        var iter = val_loader.__iter__()
        var batch_count = 0
        while iter.__has_next__():
            var batch = iter.__next__()
            # Simulate minimal processing (just access the data)
            var _ = batch.features[0, 0]
            batch_count += 1

    var val_end = now()
    var val_total_time = val_end - val_start
    var val_time_per_epoch = val_total_time / val_iterations
    var val_time_per_batch = val_time_per_epoch / num_batches

    print(
        "  Total time (", val_iterations, "epochs):", val_total_time, "seconds"
    )
    print("  Time per epoch:", val_time_per_epoch, "seconds")
    print("  Time per batch:", val_time_per_batch, "seconds")
    print()

    # Analysis
    print("=" * 80)
    print("RESULTS:")
    print("=" * 80)

    var speedup = train_time_per_epoch / val_time_per_epoch
    var time_saved = train_time_per_epoch - val_time_per_epoch
    var percent_faster = (
        (train_time_per_epoch - val_time_per_epoch) / train_time_per_epoch
    ) * 100

    print("Training (shuffled):  ", train_time_per_epoch, "seconds/epoch")
    print("Validation (no shuffle):", val_time_per_epoch, "seconds/epoch")
    print()
    print("Speedup:", speedup, "x faster")
    print("Time saved:", time_saved, "seconds per epoch")
    print("Percentage faster:", percent_faster, "%")

    # Verification
    print()
    print("=" * 80)
    if val_time_per_epoch < train_time_per_epoch:
        print("✅ PASS: Validation mode is faster than training mode!")
        print("   Bulk memcpy optimization is working correctly.")
    else:
        print("⚠️  WARNING: Validation mode is not faster.")
        print("   This might indicate an issue with bulk memcpy optimization.")

    print("=" * 80)
    print()

    # Assert that validation is faster (with some tolerance for noise)
    # We expect at least 5% speedup from bulk memcpy
    assert_true(
        val_time_per_epoch < train_time_per_epoch * 0.95,
        (
            "Validation should be at least 5% faster than training due to bulk"
            " memcpy"
        ),
    )


fn test_bulk_memcpy_scales_with_batch_size() raises:
    """Test that bulk memcpy advantage increases with larger batch sizes."""
    print()
    print("=" * 80)
    print("Testing: Bulk Memcpy Scaling with Batch Size")
    print("=" * 80)
    print()

    var num_samples = 5000
    var feature_dim = 784

    var features = Tensor[DType.float32].zeros(num_samples, feature_dim)
    var labels = Tensor[DType.int32].zeros(num_samples)

    # Fill with data
    for i in range(num_samples):
        for j in range(feature_dim):
            features[i, j] = Float32(i + j)
        labels[i] = i % 10

    var dataset = TensorDataset(features, labels)

    print("Dataset:", num_samples, "samples ×", feature_dim, "features")
    print()
    print("Batch Size | Shuffled Time | Sequential Time | Speedup")
    print("-" * 60)

    var batch_sizes = List[Int]()
    batch_sizes.append(32)
    batch_sizes.append(64)
    batch_sizes.append(128)
    batch_sizes.append(256)

    for i in range(len(batch_sizes)):
        var batch_size = batch_sizes[i]
        var iterations = 3

        # Training (shuffled)
        var train_loader = dataset.into_loader(
            batch_size, shuffle=True, drop_last=True
        )
        var train_start = now()
        for _ in range(iterations):
            var iter = train_loader.__iter__()
            while iter.__has_next__():
                var batch = iter.__next__()
                var _ = batch.features[0, 0]
        var train_time = (now() - train_start) / iterations

        # Validation (sequential)
        var val_loader = dataset.into_loader(
            batch_size, shuffle=False, drop_last=True
        )
        var val_start = now()
        for _ in range(iterations):
            var iter = val_loader.__iter__()
            while iter.__has_next__():
                var batch = iter.__next__()
                var _ = batch.features[0, 0]
        var val_time = (now() - val_start) / iterations

        var speedup = train_time / val_time

        print(
            "   ",
            batch_size,
            "     |   ",
            train_time,
            "s    |     ",
            val_time,
            "s     |  ",
            speedup,
            "x",
        )

    print()
    print("=" * 80)
    print("Expected: Speedup should increase with larger batch sizes")
    print("Reason: Bulk memcpy copies more data in single operation")
    print("=" * 80)
    print()


fn test_dataloader_memory_efficiency() raises:
    """Verify that DataLoader doesn't allocate during iteration."""
    print()
    print("=" * 80)
    print("Testing: DataLoader Memory Efficiency")
    print("=" * 80)
    print()

    var num_samples = 1000
    var feature_dim = 784

    var features = Tensor[DType.float32].zeros(num_samples, feature_dim)
    var labels = Tensor[DType.int32].zeros(num_samples)

    for i in range(num_samples):
        labels[i] = i % 10

    var dataset = TensorDataset(features, labels)
    var loader = dataset.into_loader(batch_size=64, shuffle=True)

    print("This test verifies pre-allocation by timing consistency:")
    print(
        "If batches are pre-allocated, all iterations should take similar time."
    )
    print()

    # Time first 10 batches
    var times = List[Float64]()
    var iter = loader.__iter__()

    for batch_idx in range(10):
        if not iter.__has_next__():
            break
        var batch_start = now()
        var batch = iter.__next__()
        var _ = batch.features[0, 0]  # Access to prevent optimization
        var batch_time = now() - batch_start
        times.append(batch_time)
        print("Batch", batch_idx, ":", batch_time, "seconds")

    # Calculate variance
    var mean_time = Float64(0)
    for i in range(len(times)):
        mean_time += times[i]
    mean_time /= len(times)

    var variance = Float64(0)
    for i in range(len(times)):
        var diff = times[i] - mean_time
        variance += diff * diff
    variance /= len(times)

    var std_dev = variance**0.5
    var coefficient_of_variation = (std_dev / mean_time) * 100

    print()
    print("Mean time:", mean_time, "seconds")
    print("Std deviation:", std_dev, "seconds")
    print("Coefficient of variation:", coefficient_of_variation, "%")

    if coefficient_of_variation < 20:
        print()
        print("✅ PASS: Low variance indicates pre-allocated buffers")
        print("   (No allocations happening during iteration)")
    else:
        print()
        print("⚠️  High variance might indicate allocations during iteration")

    print("=" * 80)
    print()


fn run_all_performance_tests() raises:
    """Run all DataLoader performance tests."""
    print()
    print("=" * 80)
    print("DATALOADER PERFORMANCE TEST SUITE")
    print("=" * 80)

    test_validation_faster_than_training()
    test_bulk_memcpy_scales_with_batch_size()
    test_dataloader_memory_efficiency()

    print()
    print("=" * 80)
    print("ALL PERFORMANCE TESTS COMPLETED")
    print("=" * 80)
    print()


fn main() raises:
    run_all_performance_tests()
