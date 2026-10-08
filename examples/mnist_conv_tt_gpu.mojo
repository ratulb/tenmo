"""
MNIST Training with Mojo with TileTensor Convolution layers — GPU.
A simple CNN for MNIST digit classification, running end-to-end on GPU
after a one-time parameter transfer. Mirrors
examples/mnist_conv2d_gpu.mojo except the convolutions run the zero-copy
TileTensor path (`ConvTT2D`) and downsampling uses real max-pooling
(`MaxPool2d`, k=2) — the thing the ConvGpu path could not do (MaxPool2d
has no GPU kernel, so the sibling uses stride-2 convolutions instead).

Spatial path: 28 → conv(same) → 28 → pool → 14 → conv(same) → 14 →
pool → 7; flatten width 64×7×7=3136, same as the sibling.

The model moves to GPU once (`MixedSequential.to_gpu`, in place; layers
transfer with stop_grad=True, i.e. native GPU leaves); per-batch
features/labels transfer with sync=False. Forward, backward, argmax-mask
scatter, and the optimizer step all stay on-device — no per-op transfers.
"""

from tenmo.tensor import Tensor
from tenmo.optim import SGD
from tenmo.net import MixedSequential, Linear, ReLU, Flatten
from tenmo.conv_tt import ConvTT2D
from tenmo.pooling import MaxPool2d
from tenmo.crossentropy import CrossEntropyLoss
from std.python import Python
from tenmo.numpy_interop import from_ndarray, numpy_dtype
from tenmo.dataloader import NumpyDataset, MNIST_MEAN, MNIST_STD
from tenmo.shared.timing import now
from tenmo.accuracy import Accuracy
from tenmo.gpu.device import GPU
from std.sys import has_accelerator


def train_mnist() raises:
    """Train a TileTensor CNN on MNIST digit classification on GPU."""
    comptime if not has_accelerator():
        raise Error(
            "No GPU accelerator found. Use mnist_conv2d.mojo for CPU training."
        )

    print("=" * 80)
    print("MNIST TileTensor CNN Training — GPU")
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
    comptime dtype = DType.float32
    comptime label_dtype = DType.int32

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

    # Normalize once, eagerly: the loader serves data as-is.
    train_dataset = train_dataset.normalized(
        Float32(MNIST_MEAN), Float32(MNIST_STD)
    )
    test_dataset = test_dataset.normalized(
        Float32(MNIST_MEAN), Float32(MNIST_STD)
    )

    var train_loader = train_dataset.into_loader(
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=False,
    )
    var test_loader = test_dataset.into_loader(
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )

    print("DataLoaders:")
    print("  Train batches:", len(train_loader))
    print("  Test batches:", len(test_loader), "\n")

    # ========== Model Architecture ==========
    print("Building model...")
    var model = MixedSequential()
    model.append(
        ConvTT2D[dtype](
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1,
            init_method="he",
        )
    )
    model.append(ReLU[dtype]())
    model.append(MaxPool2d[dtype](kernel_size=2))
    model.append(
        ConvTT2D[dtype](
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            init_method="he",
        )
    )
    model.append(ReLU[dtype]())
    model.append(MaxPool2d[dtype](kernel_size=2))
    # Transition from 4D to 2D
    model.append(Flatten[dtype]())
    # Fully connected layers (64 x 7 x 7 = 3136)
    model.append(
        Linear[dtype](3136, 32, init_method="he", bias_zero=True)
    )
    model.append(ReLU[dtype]())
    model.append(Linear[dtype](32, 10, init_method="he", bias_zero=True))
    print("Total parameters:", model.num_parameters(), "\n")

    # ========== Transfer Model to GPU ==========
    # Must happen before constructing the optimizer so that
    # model.parameters_of() returns GPU leaves, not the CPU originals.
    # Layers transfer with stop_grad=True: native GPU leaves.
    print("Transferring model parameters to GPU...")
    var gpu = GPU()
    model.to_gpu(gpu)
    print("  Model is now resident on GPU\n")

    # ========== Training Setup ==========
    var num_epochs = 15
    var learning_rate = Scalar[dtype](0.01)
    var momentum = Scalar[dtype](0.9)
    var weight_decay = Scalar[dtype](1e-4)
    var clip_norm = Scalar[dtype](1)
    var clip_value = Scalar[dtype](0.5)

    var criterion = CrossEntropyLoss[dtype, label_dtype]()
    var optimizer = SGD[dtype](
        model.parameters_of[dtype](),
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
        var train_correct = Float64(0.0)
        var train_total = 0

        train_loader.reset()
        var batch_num = 0
        for batch in train_loader:
            # Async data transfer — GPU queues the copy, returns immediately.
            var features_gpu = batch.features.to_gpu(gpu, sync=False)
            var labels_gpu = batch.labels.to_gpu(gpu, sync=False)

            var pred = model.forward[dtype, dtype](features_gpu)
            var loss = criterion(pred, labels_gpu)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss.item() syncs GPU (reads scalar value back)
            train_loss += loss.item() * Float32(batch.batch_size)
            train_correct += Accuracy[dtype, label_dtype].compute(
                pred, labels_gpu
            ) * Float64(batch.labels.shape()[0])
            train_total += batch.batch_size
            batch_num += 1

        # --- Validation ---
        model.eval()
        criterion.eval()
        var val_loss = Scalar[dtype](0.0)
        var val_correct = Float64(0.0)
        var val_total = 0

        test_loader.reset()
        for batch in test_loader:
            var features_gpu = batch.features.to_gpu(gpu, sync=False)
            var labels_gpu = batch.labels.to_gpu(gpu, sync=False)

            var pred = model.forward[dtype, dtype](features_gpu)
            var loss = criterion(pred, labels_gpu)

            val_loss += loss.item() * Float32(batch.batch_size)
            val_correct += Accuracy[dtype, label_dtype].compute(
                pred, labels_gpu
            ) * Float64(batch.labels.shape()[0])
            val_total += batch.batch_size

        # --- Epoch Report ---
        var epoch_time = now() - epoch_start
        var avg_train_loss = train_loss / Float32(train_total)
        var train_acc = 100.0 * train_correct / Float64(train_total)
        var avg_val_loss = val_loss / Float32(val_total)
        var val_acc = 100.0 * val_correct / Float64(val_total)
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

    # ========== Transfer Trained Weights Back to CPU ==========
    print("Transferring trained model parameters back to CPU...")
    model.to_cpu()
    print("  Model weights saved to CPU")
    print("=" * 80)


def main() raises:
    train_mnist()
