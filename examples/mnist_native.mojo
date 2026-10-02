"""
MNIST Training — pure Mojo, tensor-native `DataLoader`.

Exact per-batch twin of `examples/mnist_dataloader.py` but with every
training construct expressed in Mojo — the dtype-generic
`DataLoader[float32, int64]` (the same engine behind the Python
`tenmo.DataLoader` facade), per-batch forward/backward in-process, and
zero calls through the Python binding boundary:

  * train  — shuffled row-gathers (std.random permutation per epoch,
             same shuffle source as the trait-generic NativeLoader)
  * eval   — sequential zero-copy view slices of the source tensors
  * loss   — `CrossEntropyLoss` on int64 class-index labels
  * metric — `Accuracy.compute` over each argumentmax

Batches are read-only; a shuffled batch aliases the loader's reused
buffer and is valid only until the next batch.

Run from the repo root:
    pixi run mojo -I . examples/mnist_native.mojo

Timing note (Linux, one CPU core, batch 64, 8 epochs): ~46-47 s total
(~5.9 s/epoch) versus ~53-56 s total (~6.8 s/epoch) for the per-batch
Python twin (examples/mnist_dataloader.py) — the binding boundary costs
roughly ~15-16% over full training. Both reach ~97.5% test accuracy.
"""

from tenmo.tensor import Tensor
from tenmo.optim import SGD
from tenmo.net import Linear, ReLU, Sequential
from tenmo.crossentropy import CrossEntropyLoss
from tenmo.dataloader import DataLoader, MNIST_MEAN, MNIST_STD
from tenmo.accuracy import Accuracy
from tenmo.shared.mnemonics import DEFAULT_INDEX_DTYPE as LABEL_DTYPE
from std.python import Python
from tenmo.numpy_interop import from_ndarray, numpy_dtype
from std.time import perf_counter_ns


def train_mnist() raises:
    comptime FEATURE_DTYPE = DType.float32
    comptime LOADER_LABEL_DTYPE = DType.int64

    print("=" * 80)
    print("MNIST Training (pure Mojo — tensor-native DataLoader)")
    print("=" * 80 + "\n")

    # ========== Data Loading (dataset acquisition only) ==========
    print("Loading MNIST dataset...")
    var mnist = Python.import_module("mnist_datasets")
    var loader = mnist.MNISTLoader(folder="/tmp")

    var train_data = loader.load()
    var train_images = train_data[0]
    var train_labels = train_data[1]
    var test_data = loader.load(train=False)
    var test_images = test_data[0]
    var test_labels = test_data[1]

    train_images = train_images.astype(numpy_dtype(FEATURE_DTYPE))
    test_images = test_images.astype(numpy_dtype(FEATURE_DTYPE))
    train_labels = train_labels.astype(numpy_dtype(DType.int64))
    test_labels = test_labels.astype(numpy_dtype(DType.int64))

    # ========== Data Preparation (normalize like the Python twin) ==========
    var X_train = from_ndarray[FEATURE_DTYPE](train_images, copy=True)
    var X_test = from_ndarray[FEATURE_DTYPE](test_images, copy=True)
    var y_train = from_ndarray[DType.int64](train_labels, copy=True)
    var y_test = from_ndarray[DType.int64](test_labels, copy=True)

    X_train = (
        X_train / 255.0 - Float32(MNIST_MEAN)
    ) / Float32(MNIST_STD)
    X_test = (X_test / 255.0 - Float32(MNIST_MEAN)) / Float32(MNIST_STD)

    print("Data shapes:")
    print("  X_train:", X_train.shape(), " y_train:", y_train.shape())
    print("  X_test:", X_test.shape(), "   y_test:", y_test.shape(), "\n")

    # ========== Model Architecture ==========
    print("Building model...")
    var model = Sequential[FEATURE_DTYPE]()
    model.append(
        Linear[FEATURE_DTYPE](784, 128, init_method="he", bias_zero=True)
        .into(),
        ReLU[FEATURE_DTYPE]().into(),
        Linear[FEATURE_DTYPE](128, 32, init_method="he", bias_zero=True).into(),
        ReLU[FEATURE_DTYPE]().into(),
        Linear[FEATURE_DTYPE](32, 10, init_method="he", bias_zero=True).into(),
    )
    print("  Architecture: 784 -> 128 -> 32 -> 10\n")

    # ========== Training Setup (mirrors examples/mnist_dataloader.py) ======
    var num_epochs = 15
    var batch_size = 64
    var learning_rate = Scalar[FEATURE_DTYPE](0.01)
    var momentum = Scalar[FEATURE_DTYPE](0.9)

    var criterion = CrossEntropyLoss[FEATURE_DTYPE]()
    var optimizer = SGD[FEATURE_DTYPE](
        model.parameters(), lr=learning_rate, momentum=momentum
    )

    # ========== DataLoaders (the tensor-native engine) ==========
    var train_loader = DataLoader[
        FEATURE_DTYPE, LOADER_LABEL_DTYPE
    ](
        X_train, y_train, batch_size, shuffle=True, drop_last=False,
    )
    var eval_loader = DataLoader[
        FEATURE_DTYPE, LOADER_LABEL_DTYPE
    ](
        X_test, y_test, batch_size, shuffle=False, drop_last=False,
    )
    print(
        "Batches per epoch:",
        len(train_loader),
        "(train),",
        len(eval_loader),
        "(eval)",
    )
    print("Epochs:", num_epochs, "| Batch size:", batch_size, "\n")

    print("=" * 80)
    var training_start = perf_counter_ns()

    # ========== Training Loop ==========
    for epoch in range(num_epochs):
        var epoch_start = perf_counter_ns()

        # --- Training Phase (shuffled row-gathers) ---
        model.train()
        criterion.train()
        var train_loss = Scalar[FEATURE_DTYPE](0.0)
        var train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            var pred = model(batch.features)
            var loss = criterion(pred, batch.labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * Float32(batch.batch_size)
            train_total += batch.batch_size

        # --- Validation Phase (sequential zero-copy views) ---
        model.eval()
        criterion.eval()
        var val_correct = Float64(0.0)
        var val_total = 0

        for batch in eval_loader:
            var pred = model(batch.features)
            val_correct += Accuracy[FEATURE_DTYPE].compute(
                pred, batch.labels
            ) * Float64(batch.labels.shape()[0])
            val_total += batch.labels.shape()[0]

        # --- Epoch Report ---
        var epoch_time = Float64(perf_counter_ns() - epoch_start) / 1e9
        var avg_train_loss = train_loss / Float32(train_total)
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
            "| Val Acc:",
            val_acc,
            "%",
        )

    var total_time = Float64(perf_counter_ns() - training_start) / 1e9
    print("=" * 80)
    print("Training completed in", total_time, "seconds")
    print("=" * 80)


def main() raises:
    train_mnist()
