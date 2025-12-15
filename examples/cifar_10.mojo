"""
MNIST Training with Mojo
A simple neural network implementation for MNIST digit classification.
"""

from tenmo import Tensor
from net import Linear, ReLU, Sequential, SGD
from crossentropy import CrossEntropyLoss
from python import Python
from numpy_interop import from_ndarray, numpy_dtype
from dataloader import NumpyDataset
from time import perf_counter_ns


fn train_cifar_10() raises:
    """Train a neural network on MNIST dataset."""
    print("=" * 80)
    print("MNIST Training")
    print("=" * 80 + "\n")

    # ========== Data Loading ==========
    print("Loading cifar_10 dataset...")
    var cifar = Python.import_module("pure_cifar_10")
    var loader = cifar.CIFAR10(folder="/tmp")

    var train_data = loader.load()
    var train_images = train_data[0]
    var train_labels = train_data[1]

    var test_data = loader.load(train=False)
    var test_images = test_data[0]
    var test_labels = test_data[1]

    print("  Train samples:", len(train_images))
    print("  Test samples:", len(test_images), "\n")

    # ========== Data Preparation ==========
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
    # Normalize to [0, 1]
    X_train = X_train.__truediv__[track_grad=False](255.0)
    X_test = X_test.__truediv__[track_grad=False](255.0)

    print("Check point 0*********")

    X_train = X_train.__sub__[track_grad=False](0.5)
    X_test = X_test.__sub__[track_grad=False](0.5)

    print("Check point 1*********")

    X_train = X_train.__mul__[track_grad=False](2)
    X_test = X_test.__mul__[track_grad=False](2)

    print("Check point 2*********")

    print("Data shapes:")
    print("  X_train:", X_train.shape())
    print("  y_train:", y_train.shape())
    print("  X_test:", X_test.shape())
    print("  y_test:", y_test.shape(), "\n")

    X_train = X_train.reshape[False](50000, 3072)
    X_test = X_test.reshape[False](10000, 3072)

    # ========== DataLoaders ==========
    var train_batch_size = 128
    var test_batch_size = 128

    var train_dataset = NumpyDataset[FEATURE_DTYPE, LABEL_DTYPE](
        X_train, y_train
    )
    var test_dataset = NumpyDataset[FEATURE_DTYPE, LABEL_DTYPE](X_test, y_test)

    var train_loader = train_dataset.into_loader(
        batch_size=train_batch_size, shuffle=True, drop_last=False
    )
    var test_loader = test_dataset.into_loader(
        batch_size=test_batch_size, shuffle=False, drop_last=False
    )

    print("DataLoaders:")
    print("  Train batches:", len(train_loader))
    print("  Test batches:", len(test_loader), "\n")

    # ========== Model Architecture ==========
    print("Building model...")
    var model = Sequential[FEATURE_DTYPE]()
    model.append(
        Linear[FEATURE_DTYPE](3072, 256, bias_zero=True).into(),
        ReLU[FEATURE_DTYPE]().into(),
        Linear[FEATURE_DTYPE](256, 128, bias_zero=True).into(),
        ReLU[FEATURE_DTYPE]().into(),
        Linear[FEATURE_DTYPE](128, 64, bias_zero=True).into(),
        ReLU[FEATURE_DTYPE]().into(),
        Linear[FEATURE_DTYPE](64, 10, bias_zero=True).into()
    )
    print("  Architecture: 3072 -> 256 -> 64 -> 10")
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
        _ = """if epoch == 8:
            optimizer.set_lr(optimizer.lr/5)
        # Learning rate decay at epoch 5
        if epoch == 15:
            optimizer.set_lr(optimizer.lr / 5)"""

        # --- Training Phase ---
        model.train()
        criterion.train()
        var train_loss = Scalar[FEATURE_DTYPE](0.0)
        var train_correct: Int64 = 0
        var train_total = 0

        train_loader.reset()
        while train_loader.__has_next__():
            ref batch = train_loader.__next__()
            var pred = model(batch.features)
            var loss = criterion(pred, batch.labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.batch_size
            train_correct += compute_accuracy(pred, batch.labels)
            train_total += batch.batch_size

        # --- Validation Phase ---
        model.eval()
        criterion.eval()
        var val_loss = Scalar[FEATURE_DTYPE](0.0)
        var val_correct: Int64 = 0
        var val_total = 0

        test_loader.reset()
        while test_loader.__has_next__():
            ref batch = test_loader.__next__()
            var pred = model(batch.features)
            var loss = criterion(pred, batch.labels)

            val_loss += loss.item() * batch.batch_size
            val_correct += compute_accuracy(pred, batch.labels)
            val_total += batch.batch_size

        # --- Epoch Report ---
        var epoch_time = (perf_counter_ns() - epoch_start) / 1e9
        var avg_train_loss = train_loss / train_total
        var train_acc = 100.0 * Float64(train_correct) / train_total
        var avg_val_loss = val_loss / val_total
        var val_acc = 100.0 * Float64(val_correct) / val_total

        print(
            "Epoch",
            epoch + 1,
            "/",
            num_epochs,
            "| Time:",
            epoch_time,
            "s",
            "| Train Loss:",
            avg_train_loss,
            "Acc:",
            train_acc,
            "%",
            "| Val Loss:",
            avg_val_loss,
            "Acc:",
            val_acc,
            "%",
        )

    var total_time = (perf_counter_ns() - training_start) / 1e9
    print("=" * 80)
    print("Training completed in", total_time, "seconds")
    print("=" * 80)


fn compute_accuracy_orig[
    dtype: DType
](pred: Tensor[dtype], target: Tensor[DType.int32]) -> Int:
    """Compute classification accuracy by comparing argmax predictions to targets.
    """
    var correct = 0
    var batch_size = pred.shape()[0]
    var num_classes = pred.shape()[1]

    for i in range(batch_size):
        var max_idx = 0
        var max_val = pred[i, 0]
        for j in range(1, num_classes):
            if pred[i, j] > max_val:
                max_val = pred[i, j]
                max_idx = j

        if max_idx == Int(target[i]):
            correct += 1

    return correct


fn compute_accuracy[
    dtype: DType, //
](pred: Tensor[dtype], target: Tensor[DType.int32]) -> Int64:
    """
    Compute classification accuracy by comparing argmax predictions to targets.

    Args:
        pred: Prediction tensor of shape (batch_size, num_classes).
        target: Target tensor of shape (batch_size,) with integer class labels.

    Returns:
        Number of correct predictions in the batch.
    """
    # Get predicted class indices (argmax along class dimension)
    pred_classes = pred.argmax(axis=1)

    # Compare predictions to targets
    correct = (
        pred_classes.eq(target)
        .to_dtype[DType.int64]()
        .sum[track_grad=False]()
        .item()
    )

    return correct


fn main() raises:
    train_cifar_10()
