"""
CIFAR-10 Training with Mojo - Profiled Version
Includes detailed timing measurements to identify bottlenecks.
"""

from tenmo import Tensor
from net import LinearBLAS, ReLU, SequentialBLAS, SGD, Dropout
from crossentropy import CrossEntropyLoss
from python import Python
from numpy_interop import from_ndarray, numpy_dtype
from dataloader import NumpyDataset
from time import perf_counter_ns


fn train_cifar_10() raises:
    """Train a neural network on CIFAR-10 dataset."""
    print("=" * 80)
    print("CIFAR-10 Training (Profiled)")
    print("=" * 80 + "\n")

    # ========== Data Loading ==========
    var t0 = perf_counter_ns()
    print("Loading CIFAR-10 dataset...")
    var cifar = Python.import_module("pure_cifar_10")
    var loader = cifar.CIFAR10(folder="/tmp")

    var train_data = loader.load()
    var train_images = train_data[0]
    var train_labels = train_data[1]

    var test_data = loader.load(train=False)
    var test_images = test_data[0]
    var test_labels = test_data[1]

    var t1 = perf_counter_ns()
    print("  [TIME] Data loading:", (t1 - t0) / 1e9, "s")
    print("  Train samples:", len(train_images))
    print("  Test samples:", len(test_images), "\n")

    # ========== Data Preparation ==========
    t0 = perf_counter_ns()
    alias FEATURE_DTYPE = DType.float32
    alias LABEL_DTYPE = DType.int32

    train_images = train_images.astype(numpy_dtype(FEATURE_DTYPE))
    train_labels = train_labels.astype(numpy_dtype(LABEL_DTYPE))
    test_images = test_images.astype(numpy_dtype(FEATURE_DTYPE))
    test_labels = test_labels.astype(numpy_dtype(LABEL_DTYPE))

    var X_train = from_ndarray[FEATURE_DTYPE](train_images, copy=True)
    var y_train = from_ndarray[LABEL_DTYPE](train_labels, copy=True)
    var X_test = from_ndarray[FEATURE_DTYPE](test_images, copy=True)
    var y_test = from_ndarray[LABEL_DTYPE](test_labels, copy=True)

    t1 = perf_counter_ns()
    print("[TIME] NumPy to Mojo conversion:", (t1 - t0) / 1e9, "s")

    # Normalize to [-1, 1]
    t0 = perf_counter_ns()
    X_train = X_train.__truediv__[track_grad=False](255.0)
    X_test = X_test.__truediv__[track_grad=False](255.0)
    X_train = X_train.__sub__[track_grad=False](0.5)
    X_test = X_test.__sub__[track_grad=False](0.5)
    X_train = X_train.__mul__[track_grad=False](2)
    X_test = X_test.__mul__[track_grad=False](2)

    t1 = perf_counter_ns()
    print("[TIME] Normalization:", (t1 - t0) / 1e9, "s")

    print("\nData shapes:")
    print("  X_train:", X_train.shape())
    print("  y_train:", y_train.shape())
    print("  X_test:", X_test.shape())
    print("  y_test:", y_test.shape())

    t0 = perf_counter_ns()
    X_train = X_train.reshape[False](50000, 3072)
    X_test = X_test.reshape[False](10000, 3072)
    t1 = perf_counter_ns()
    print("[TIME] Reshape:", (t1 - t0) / 1e9, "s\n")

    # ========== DataLoaders ==========
    t0 = perf_counter_ns()
    var train_batch_size = 128
    var test_batch_size = 128

    var train_dataset = NumpyDataset[FEATURE_DTYPE, LABEL_DTYPE](X_train, y_train)
    var test_dataset = NumpyDataset[FEATURE_DTYPE, LABEL_DTYPE](X_test, y_test)

    var train_loader = train_dataset.into_loader(
        batch_size=train_batch_size, shuffle=True, drop_last=False
    )
    var test_loader = test_dataset.into_loader(
        batch_size=test_batch_size, shuffle=False, drop_last=False
    )

    t1 = perf_counter_ns()
    print("[TIME] DataLoader creation:", (t1 - t0) / 1e9, "s")
    print("DataLoaders:")
    print("  Train batches:", len(train_loader))
    print("  Test batches:", len(test_loader), "\n")

    # ========== Model Architecture ==========
    t0 = perf_counter_ns()
    print("Building model...")
    var model = SequentialBLAS[FEATURE_DTYPE]()
    model.append(
        LinearBLAS[FEATURE_DTYPE](3072, 256, bias_zero=True).into(),
        ReLU[FEATURE_DTYPE]().into(),
        #Dropout[FEATURE_DTYPE]().into(),
        LinearBLAS[FEATURE_DTYPE](256, 64, bias_zero=True).into(),
        ReLU[FEATURE_DTYPE]().into(),
        LinearBLAS[FEATURE_DTYPE](64, 10, bias_zero=True).into(),
    )
    t1 = perf_counter_ns()
    print("  [TIME] Model construction:", (t1 - t0) / 1e9, "s")
    print("  Architecture: 3072 -> 256 -> 128 -> 64 -> 10")
    print("  Total parameters:", model.num_parameters(), "\n")

    # ========== Training Setup ==========
    var num_epochs = 60
    var learning_rate = Scalar[FEATURE_DTYPE](0.001075)
    var momentum = Scalar[FEATURE_DTYPE](0.9)

    var criterion = CrossEntropyLoss[FEATURE_DTYPE]()
    var optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    print("Training configuration:")
    print("  Epochs:", num_epochs)
    print("  Batch size:", train_batch_size)
    print("  Learning rate:", learning_rate)
    print("  Momentum:", momentum, "\n")

    print("=" * 80)
    var training_start = perf_counter_ns()

    # ========== Training Loop ==========
    for epoch in range(num_epochs):
        var epoch_start = perf_counter_ns()

        # --- Training Phase ---
        model.train()
        criterion.train()
        var train_loss = Scalar[FEATURE_DTYPE](0.0)
        var train_correct: Int64 = 0
        var train_total = 0

        var train_forward_time: Float64 = 0
        var train_backward_time: Float64 = 0
        var train_optimizer_time: Float64 = 0
        var train_accuracy_time: Float64 = 0

        train_loader.reset()
        var batch_count = 0
        while train_loader.__has_next__():
            ref batch = train_loader.__next__()

            # Forward pass
            t0 = perf_counter_ns()
            var pred = model(batch.features)
            var loss = criterion(pred, batch.labels)
            t1 = perf_counter_ns()
            train_forward_time += (t1 - t0) / 1e9

            # Backward pass
            t0 = perf_counter_ns()
            optimizer.zero_grad()
            loss.backward()
            t1 = perf_counter_ns()
            train_backward_time += (t1 - t0) / 1e9

            # Optimizer step
            t0 = perf_counter_ns()
            optimizer.step()
            t1 = perf_counter_ns()
            train_optimizer_time += (t1 - t0) / 1e9

            # Accumulate loss
            train_loss += loss.item() * batch.batch_size

            # Accuracy computation
            t0 = perf_counter_ns()
            train_correct += compute_accuracy(pred, batch.labels)
            t1 = perf_counter_ns()
            train_accuracy_time += (t1 - t0) / 1e9

            train_total += batch.batch_size
            batch_count += 1

        # --- Validation Phase ---
        model.eval()
        criterion.eval()
        var val_loss = Scalar[FEATURE_DTYPE](0.0)
        var val_correct: Int64 = 0
        var val_total = 0

        var val_forward_time: Float64 = 0
        var val_accuracy_time: Float64 = 0

        test_loader.reset()
        while test_loader.__has_next__():
            ref batch = test_loader.__next__()

            t0 = perf_counter_ns()
            var pred = model(batch.features)
            var loss = criterion(pred, batch.labels)
            t1 = perf_counter_ns()
            val_forward_time += (t1 - t0) / 1e9

            val_loss += loss.item() * batch.batch_size

            t0 = perf_counter_ns()
            val_correct += compute_accuracy(pred, batch.labels)
            t1 = perf_counter_ns()
            val_accuracy_time += (t1 - t0) / 1e9

            val_total += batch.batch_size

        # --- Epoch Report ---
        var epoch_end = perf_counter_ns()
        var epoch_time = (epoch_end - epoch_start) / 1e9

        var avg_train_loss = train_loss / train_total
        var train_acc = 100.0 * Float64(train_correct) / train_total
        var avg_val_loss = val_loss / val_total
        var val_acc = 100.0 * Float64(val_correct) / val_total

        print(
            "Epoch", epoch + 1, "/", num_epochs,
            "| Time:", epoch_time, "s",
            "| Train Loss:", avg_train_loss, "Acc:", train_acc, "%",
            "| Val Loss:", avg_val_loss, "Acc:", val_acc, "%"
        )

        # Detailed timing for first epoch
        if epoch == 0:
            print("  [PROFILE] Train forward:", train_forward_time, "s")
            print("  [PROFILE] Train backward:", train_backward_time, "s")
            print("  [PROFILE] Train optimizer:", train_optimizer_time, "s")
            print("  [PROFILE] Train accuracy:", train_accuracy_time, "s")
            print("  [PROFILE] Val forward:", val_forward_time, "s")
            print("  [PROFILE] Val accuracy:", val_accuracy_time, "s")
            print("  [PROFILE] Total train batches:", batch_count)

    var total_time = (perf_counter_ns() - training_start) / 1e9
    print("\n" + "=" * 80)
    print("Training completed in", total_time, "seconds")
    print("Average time per epoch:", total_time / num_epochs, "seconds")
    print("=" * 80)


fn compute_accuracy[dtype: DType](
    pred: Tensor[dtype], target: Tensor[DType.int32]
) -> Int64:
    """Compute classification accuracy by comparing argmax predictions to targets."""
    pred_classes = pred.argmax(axis=1)
    correct = (
        pred_classes.eq(target)
        .to_dtype[DType.int64]()
        .sum[track_grad=False]()
        .item()
    )
    return correct


fn main() raises:
    train_cifar_10()
