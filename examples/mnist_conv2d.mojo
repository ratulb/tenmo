"""
MNIST Training with Mojo with Convolution layers
A simple neural network implementation for MNIST digit classification.
"""

from tenmo import Tensor
from sgd import SGD
from net import Linear, ReLU, Sequential, Conv2D, Flatten
from forwards import MaxPool2d
from crossentropy import CrossEntropyLoss
from python import Python
from numpy_interop import from_ndarray, numpy_dtype
from dataloader import NumpyDataset
from forwards import Padding
from common_utils import now


fn train_mnist() raises:
    """Train a neural network on MNIST dataset."""
    print("=" * 80)
    print("MNIST Training")
    print("=" * 80 + "\n")

    # ========== Data Loading ==========
    print("Loading MNIST dataset...")
    var mnist = Python.import_module("mnist_datasets")
    var loader = mnist.MNISTLoader(folder="/tmp")

    var train_data = loader.load()
    var train_images = train_data[0]
    var train_labels = train_data[1]
    train_images = train_images.reshape(-1, 1, 28, 28)
    var test_data = loader.load(train=False)
    var test_images = test_data[0]
    test_images = test_images.reshape(-1, 1, 28, 28)
    var test_labels = test_data[1]

    print("Train samples:", len(train_images))
    print("Test samples:", len(test_images), "\n")

    # ========== Data Preparation ==========
    alias dtype = DType.float32
    alias label_dtype = DType.int32

    train_images = train_images.astype(numpy_dtype(dtype))
    train_labels = train_labels.astype(numpy_dtype(label_dtype))
    test_images = test_images.astype(numpy_dtype(dtype))
    test_labels = test_labels.astype(numpy_dtype(label_dtype))

    var X_train = from_ndarray[dtype](train_images, copy=True)
    var y_train = from_ndarray[label_dtype](train_labels, copy=True)
    var X_test = from_ndarray[dtype](test_images, copy=True)
    var y_test = from_ndarray[label_dtype](test_labels, copy=True)

    # Normalize to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print("Data shapes:")
    print("  X_train:", X_train.shape())
    print("  y_train:", y_train.shape())
    print("  X_test:", X_test.shape())
    print("  y_test:", y_test.shape(), "\n")
    # ========== DataLoaders ==========
    var train_batch_size = 128
    var test_batch_size = 128

    var train_dataset = NumpyDataset[dtype, label_dtype](X_train, y_train)
    var test_dataset = NumpyDataset[dtype, label_dtype](X_test, y_test)

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
    var model = Sequential[dtype]()

    model.append(
        # Convolutional layers (work on 4D tensors)
        Conv2D[dtype](
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=Padding("same"),
            init_method="he",
        ).into(),
        ReLU[dtype]().into(),
        MaxPool2d[dtype](kernel_size=2, stride=2).into(),
        Conv2D[dtype](
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=Padding("same"),
            init_method="he",
        ).into(),
        ReLU[dtype]().into(),
        MaxPool2d[dtype](kernel_size=2, stride=2).into(),
        # Transition from 4D to 2D
        Flatten[dtype]().into(),
        # Fully connected layers
        Linear[dtype](
            3136, 32, init_method="he", bias_zero=True  # 64 × 28 × 28
        ).into(),
        ReLU[dtype]().into(),
        Linear[dtype](32, 10, init_method="he", bias_zero=True).into(),
    )
    print("Total parameters:", model.num_parameters(), "\n")

    # ========== Training Setup ==========
    var num_epochs = 15
    var learning_rate = Scalar[dtype](0.01)
    var momentum = Scalar[dtype](0.9)
    var weight_decay = Scalar[dtype](1e-4)
    var clip_norm = Scalar[dtype](1)
    var clip_value = Scalar[dtype](0.5)

    var criterion = CrossEntropyLoss[dtype]()
    var optimizer = SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
        clip_value=clip_value,
    )

    print("Training configuration:")
    print("Epochs:", num_epochs)
    print("Batch size:", train_batch_size)
    print("Learning rate:", learning_rate)
    print("Momentum:", momentum, "\n")

    print("=" * 80)
    var training_start = now()

    # ========== Training Loop ==========
    for epoch in range(num_epochs):
        var epoch_start = now()

        # Learning rate decay
        if epoch == 10:
            optimizer.set_lr(optimizer.lr / 10)
        model.train()
        criterion.train()
        var train_loss = Scalar[dtype](0.0)
        var train_correct = 0
        var train_total = 0

        train_loader.reset()
        var batch_num = 0
        # while train_loader.__has_next__():
        for batch in train_loader:
            # ref batch = train_loader.__next__()
            var pred = model(batch.features)
            var loss = criterion(pred, batch.labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.batch_size
            train_correct += compute_accuracy(pred, batch.labels)
            train_total += batch.batch_size
            batch_num += 1

            if batch_num % 10 == 0:
                print("Epoch: ", epoch + 1, "batch num: ", batch_num)
        # --- Validation ---
        model.eval()
        criterion.eval()
        var val_loss = Scalar[dtype](0.0)
        var val_correct = 0
        var val_total = 0

        test_loader.reset()
        # while test_loader.__has_next__():
        for batch in test_loader:
            # ref batch = test_loader.__next__()
            var pred = model(batch.features)
            var loss = criterion(pred, batch.labels)

            val_loss += loss.item() * batch.batch_size
            val_correct += compute_accuracy(pred, batch.labels)
            val_total += batch.batch_size

        # --- Epoch Report ---
        var epoch_time = now() - epoch_start
        var avg_train_loss = train_loss / train_total
        var train_acc = 100.0 * train_correct / train_total
        var avg_val_loss = val_loss / val_total
        var val_acc = 100.0 * val_correct / val_total
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

    var total_time = now() - training_start
    print("=" * 80)
    print("Training completed in", total_time, "seconds")
    print("=" * 80)


fn compute_accuracy[
    dtype: DType
](pred: Tensor[dtype], target: Tensor[DType.int32]) -> Int:
    """
    Compute accuracy by comparing argmax predictions to targets.
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


fn main() raises:
    train_mnist()
