from tenmo import Tensor
from net import Linear, ReLU, Sequential, SGD
from crossentropy import CrossEntropyLoss
from testing import assert_true
from python import Python, PythonObject
from numpy_interop import from_ndarray, numpy_dtype
from common_utils import id

# from data import *
from dataloader import *
from time import perf_counter_ns


fn train_mnist() raises:
    print("=" * 80)
    print("MNIST Training with NumPy Dataset & DataLoader")
    print("=" * 80 + "\n")

    # Load data
    print("Loading MNIST data...")
    var mnist = Python.import_module("mnist_datasets")
    var loader = mnist.MNISTLoader(dataset_type="fashion", folder="/tmp")
    var np = Python.import_module("numpy")

    var train_data = loader.load()
    var train_images_np = train_data[0]
    var train_labels_np = train_data[1]
    print("Train data loaded:", len(train_images_np), "samples")

    var test_data = loader.load(train=False)
    var test_images_np = test_data[0]
    var test_labels_np = test_data[1]
    print("Test data loaded:", len(test_images_np), "samples\n")

    # Convert to Mojo tensors
    print("Converting to Mojo tensors and normalizing...")
    alias feature_dtype = DType.float32
    alias label_dtype = DType.int32

    # ✅ KEY: Convert labels to int32, keep features as float32
    _ = """train_images_np = train_images_np.astype(np.float32)
    train_labels_np = train_labels_np.astype(np.int32)
    test_images_np = test_images_np.astype(np.float32)
    test_labels_np = test_labels_np.astype(np.int32)"""

    # ✅ KEY: Convert labels to int32, keep features as float32
    train_images_np = train_images_np.astype(numpy_dtype(feature_dtype))
    train_labels_np = train_labels_np.astype(numpy_dtype(label_dtype))
    test_images_np = test_images_np.astype(numpy_dtype(feature_dtype))
    test_labels_np = test_labels_np.astype(numpy_dtype(label_dtype))

    var X_train = from_ndarray[feature_dtype](train_images_np, copy=True)
    var y_train = from_ndarray[label_dtype](train_labels_np, copy=True)
    var X_test = from_ndarray[feature_dtype](test_images_np, copy=True)
    var y_test = from_ndarray[label_dtype](test_labels_np, copy=True)

    # Normalize features
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # ✅ KEY: Keep labels as (N,) NOT (N, 1) for CrossEntropyLoss
    # y_train and y_test already have shape (N,) from NumPy

    print("Data shapes:")
    print("  X_train:", X_train.shape().__str__())
    print("  y_train:", y_train.shape().__str__())
    print("  X_test:", X_test.shape().__str__())
    print("  y_test:", y_test.shape().__str__(), "\n")

    # Create datasets with heterogeneous dtypes
    var train_dataset = NumpyDataset[feature_dtype, label_dtype](
        X_train, y_train
    )
    var test_dataset = NumpyDataset[feature_dtype, label_dtype](X_test, y_test)

    # Create DataLoaders
    _ = """var train_loader = DataLoader(
        train_dataset^, batch_size=64, shuffle=True, drop_last=False
    )"""
    var train_loader = train_dataset.into_loader(
        batch_size=64, shuffle=True, drop_last=False
    )

    _ = """var test_loader = DataLoader(
        test_dataset^,
        batch_size=64,
        shuffle=False,
        drop_last=False,
    )"""
    var test_loader = test_dataset.into_loader(
        batch_size=64,
        shuffle=False,
        drop_last=False,
    )

    print("DataLoaders created:")
    print("  Train batches:", len(train_loader))
    print("  Test batches:", len(test_loader), "\n")

    # Build model
    print("Building model...")
    var model = Sequential[feature_dtype]()
    _ = """model.append(
        Linear[feature_dtype](784, 128, xavier=True).into(),
        ReLU[feature_dtype]().into(),
        Linear[feature_dtype](128, 32, xavier=True).into(),
        ReLU[feature_dtype]().into(),
        Linear[feature_dtype](32, 10, xavier=True).into(),
        )"""
    model.append(
        Linear[feature_dtype](784, 256, xavier=True).into(),
        ReLU[feature_dtype]().into(),
        Linear[feature_dtype](256, 128, xavier=True).into(),
        ReLU[feature_dtype]().into(),
        Linear[feature_dtype](128, 64, xavier=True).into(),
        ReLU[feature_dtype]().into(),
        Linear[feature_dtype](64, 32, xavier=True).into(),
        ReLU[feature_dtype]().into(),
        Linear[feature_dtype](32, 10, xavier=True).into(),
    )

    print("Model architecture:")
    print("  Input: 784 (28x28 flattened)")
    print("  Hidden: 128 -> 32")
    print("  Output: 10 (digit classes)")
    print("  Total parameters:", model.num_parameters(), "\n")

    # Setup training
    var criterion = CrossEntropyLoss[feature_dtype]()
    var optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    var num_epochs = 20

    print("Training configuration:")
    print("  Epochs:", num_epochs)
    # print("  Batch size: 256")
    print("  Batch size: 512")
    print("  Learning rate: 0.001")
    print("  Momentum: 0.9\n")

    var start_training = perf_counter_ns()

    # Training loop
    for epoch in range(num_epochs):
        var epoch_start = perf_counter_ns()
        if epoch == 10:
            optimizer.set_lr(optimizer.lr / 10)
        if epoch == 15:
            optimizer.set_lr(optimizer.lr / 10)

        # Training phase
        model.train()
        criterion.train()

        var epoch_train_loss = Scalar[feature_dtype](0.0)
        var epoch_train_correct = 0
        var epoch_train_total = 0

        for train_batch in train_loader:
            var train_pred = model(train_batch.features)
            # ✅ labels are already DType.int32 and shape (batch_size,)
            var train_loss = criterion(train_pred, train_batch.labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_train_loss += train_loss.item() * train_batch.batch_size
            var batch_correct = compute_accuracy_multiclass(
                train_pred, train_batch.labels
            )
            epoch_train_correct += batch_correct
            epoch_train_total += train_batch.batch_size

        # Validation phase
        model.eval()
        criterion.eval()

        var epoch_val_loss = Scalar[feature_dtype](0.0)
        var epoch_val_correct = 0
        var epoch_val_total = 0

        for val_batch in test_loader:
            var val_pred = model(val_batch.features)
            var val_loss = criterion(val_pred, val_batch.labels)

            epoch_val_loss += val_loss.item() * val_batch.batch_size
            var val_correct = compute_accuracy_multiclass(
                val_pred, val_batch.labels
            )
            epoch_val_correct += val_correct
            epoch_val_total += val_batch.batch_size

        # Reporting
        var epoch_end = perf_counter_ns()
        var epoch_time = (epoch_end - epoch_start) / 1e9

        var avg_train_loss = epoch_train_loss / epoch_train_total
        var train_accuracy = 100.0 * epoch_train_correct / epoch_train_total
        var avg_val_loss = epoch_val_loss / epoch_val_total
        var val_accuracy = 100.0 * epoch_val_correct / epoch_val_total

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
            "Train Acc:",
            train_accuracy,
            "%",
            "| Val Loss:",
            avg_val_loss,
            "Val Acc:",
            val_accuracy,
            "%",
        )

    var end_training = perf_counter_ns()
    var total_time = (end_training - start_training) / 1e9

    print("\n" + "=" * 80)
    print("Training completed in:", total_time, "seconds")
    print("=" * 80)


fn compute_accuracy_multiclass[
    dtype: DType
](pred: Tensor[dtype], target: Tensor[DType.int32]) -> Int:
    """Compute accuracy - target can be any dtype now."""
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

        # ✅ target is shape (batch_size,) not (batch_size, 1)
        var true_label = Int(target[i])

        if max_idx == true_label:
            correct += 1

    return correct


fn main() raises:
    train_mnist()
