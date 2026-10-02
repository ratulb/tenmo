"""
MNIST Training with Mojo — mixed-precision variant of examples/mnist_mixed.mojo.

Same MLP as mnist.mojo, but the bottleneck runs in float16:
    784[f32] -> 128[f32] -> (f32->f16 cast) -> 32[f16] -> (f16->f32 cast) -> 10[f32]

The container inserts grad-tracked boundary casts whenever adjacent layers'
dtypes differ; the loss stays in f32 for softmax/CE stability. Training uses
one SGD per dtype segment (parameters_of[dtype]); gradients cross dtype
boundaries through the casts, parameter updates stay dtype-local.

Container-API differences vs the legacy Sequential[dtype]:
  - no dtype parameter on the container; layers carry their own dtypes
  - bare append(layer) per layer (generic append[M: LayerTrait]; no .into())
  - forward is model.forward[in_dt, out_dt](x) (annotations runtime-checked)
"""

from tenmo.tensor import Tensor
from tenmo.optim import SGD
from tenmo.net import Linear, ReLU
from tenmo.net import MixedSequential
from tenmo.crossentropy import CrossEntropyLoss
from std.python import Python
from tenmo.numpy_interop import from_ndarray, numpy_dtype
from tenmo.dataloader import NumpyDataset, MNIST_MEAN, MNIST_STD
from std.time import perf_counter_ns
from tenmo.accuracy import Accuracy
from tenmo.shared.mnemonics import DEFAULT_INDEX_DTYPE as LABEL_DTYPE


def train_mnist() raises:
    """Train a neural network on MNIST dataset."""
    print("=" * 80)
    print("MNIST Training (MixedSequential)")
    print("=" * 80 + "\n")

    # ========== Data Loading ==========
    print("Loading MNIST dataset...")
    var mnist = Python.import_module("mnist_datasets")
    var loader = mnist.MNISTLoader(folder="/tmp")

    var train_data = loader.load()
    var train_images = train_data[0]
    var train_labels = train_data[1]

    var test_data = loader.load(train=False)
    var test_images = test_data[0]
    var test_labels = test_data[1]

    print("  Train samples:", len(train_images))
    print("  Test samples:", len(test_images), "\n")

    # ========== Data Preparation ==========
    comptime FEATURE_DTYPE = DType.float32

    train_images = train_images.astype(numpy_dtype(FEATURE_DTYPE))
    train_labels = train_labels.astype(numpy_dtype(LABEL_DTYPE))
    test_images = test_images.astype(numpy_dtype(FEATURE_DTYPE))
    test_labels = test_labels.astype(numpy_dtype(LABEL_DTYPE))

    var X_train = from_ndarray[FEATURE_DTYPE](train_images, copy=True)
    var y_train = from_ndarray[LABEL_DTYPE](train_labels, copy=True)
    var X_test = from_ndarray[FEATURE_DTYPE](test_images, copy=True)
    var y_test = from_ndarray[LABEL_DTYPE](test_labels, copy=True)

    # Normalize to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print("Data shapes:")
    print("  X_train:", X_train.shape())
    print("  y_train:", y_train.shape())
    print("  X_test:", X_test.shape())
    print("  y_test:", y_test.shape(), "\n")

    # ========== DataLoaders ==========
    var train_batch_size = 64
    var test_batch_size = 64

    var train_dataset = NumpyDataset[FEATURE_DTYPE, LABEL_DTYPE](
        X_train, y_train
    )
    var test_dataset = NumpyDataset[FEATURE_DTYPE, LABEL_DTYPE](X_test, y_test)

    var train_loader = train_dataset.into_loader(
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=False,
        normalize_mean=Float32(MNIST_MEAN),
        normalize_std=Float32(MNIST_STD),
    )
    var test_loader = test_dataset.into_loader(
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        normalize_mean=Float32(MNIST_MEAN),
        normalize_std=Float32(MNIST_STD),
    )

    print("DataLoaders:")
    print("  Train batches:", len(train_loader))
    print("  Test batches:", len(test_loader), "\n")

    # ========== Model Architecture ==========
    # Mixed-precision chain: f32 stem -> f16 bottleneck -> f32 logits.
    # Adjacent layers with different dtypes make the container insert
    # grad-tracked boundary casts at append time (f32->f16 and back);
    # the loss stays in f32 for softmax/CE stability.
    comptime HALF_DTYPE = DType.float16
    print("Building model...")
    var model = MixedSequential()
    model.append(Linear[FEATURE_DTYPE](784, 128, init_method="he", bias_zero=True))
    model.append(ReLU[FEATURE_DTYPE]())
    model.append(Linear[HALF_DTYPE](128, 32, init_method="he", bias_zero=True))
    model.append(ReLU[HALF_DTYPE]())
    model.append(Linear[FEATURE_DTYPE](32, 10, init_method="he", bias_zero=True))
    print("  Architecture: 784[f32] -> 128[f32] -> [cast] -> 32[f16] -> [cast] -> 10[f32]")
    var p32 = model.parameters_of[FEATURE_DTYPE]()
    var p16 = model.parameters_of[HALF_DTYPE]()
    print("  Parameter tensors: ", len(p32), " x f32 + ", len(p16), " x f16")
    print("  Total parameters:", model.num_parameters(), "\n")

    # ========== Training Setup ==========
    var num_epochs = 15
    var learning_rate = Scalar[FEATURE_DTYPE](0.01)
    var momentum = Scalar[FEATURE_DTYPE](0.9)
    var weight_decay = Scalar[FEATURE_DTYPE](1e-4)
    var clip_norm = Scalar[FEATURE_DTYPE](1)
    var clip_value = Scalar[FEATURE_DTYPE](0.5)

    var criterion = CrossEntropyLoss[FEATURE_DTYPE]()
    var optimizer = SGD[FEATURE_DTYPE](
        p32^,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        clip_norm=clip_norm,
        clip_value=clip_value,
    )
    # One optimizer per dtype segment; gradients cross dtype boundaries
    # through the container-inserted casts, updates stay dtype-local.
    var optimizer16 = SGD[HALF_DTYPE](
        p16^,
        lr=Scalar[HALF_DTYPE](0.01),
        momentum=Scalar[HALF_DTYPE](0.9),
        weight_decay=Scalar[HALF_DTYPE](1e-4),
        clip_norm=Scalar[HALF_DTYPE](1),
        clip_value=Scalar[HALF_DTYPE](0.5),
    )

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

        # Learning rate decay at epoch 10
        if epoch == 10:
            optimizer.set_lr(optimizer.lr / 10)
            optimizer16.set_lr(optimizer16.lr / 10)

        # --- Training Phase ---
        model.train()
        criterion.train()
        var train_loss = Scalar[FEATURE_DTYPE](0.0)
        var train_correct = Float64(0.0)
        var train_total = 0

        train_loader.reset()
        while train_loader.__has_next__():
            ref batch = train_loader.__next__()
            var pred = model.forward[FEATURE_DTYPE, FEATURE_DTYPE](
                batch.features
            )
            var loss = criterion(pred, batch.labels)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer16.step()

            train_loss += loss.item() * Float32(batch.batch_size)
            train_correct += Accuracy[FEATURE_DTYPE].compute(
                pred, batch.labels
            ) * Float64(batch.labels.shape()[0])
            train_total += batch.batch_size

        # --- Validation Phase ---
        model.eval()
        criterion.eval()
        var val_loss = Scalar[FEATURE_DTYPE](0.0)
        var val_correct = Float64(0.0)
        var val_total = 0

        test_loader.reset()
        while test_loader.__has_next__():
            ref batch = test_loader.__next__()
            var pred = model.forward[FEATURE_DTYPE, FEATURE_DTYPE](
                batch.features
            )
            var loss = criterion(pred, batch.labels)

            val_loss += loss.item() * Float32(batch.batch_size)
            val_correct += Accuracy[FEATURE_DTYPE].compute(
                pred, batch.labels
            ) * Float64(batch.labels.shape()[0])
            val_total += batch.batch_size

        # --- Epoch Report ---
        var epoch_time = Float64(perf_counter_ns() - epoch_start) / 1e9
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

    var total_time = Float64(perf_counter_ns() - training_start) / 1e9
    print("=" * 80)
    print("Training completed in", total_time, "seconds")
    print("=" * 80)


def main() raises:
    train_mnist()
