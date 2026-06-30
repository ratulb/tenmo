"""
MNIST Training — runs on CPU or GPU automatically.
Architecture: 784 -> 128 -> ReLU -> 32 -> ReLU -> 10.
"""

from tenmo.tensor import Tensor
from tenmo.optim import SGD
from tenmo.net import Linear, ReLU, Sequential
from tenmo.crossentropy import CrossEntropyLoss
from std.python import Python
from tenmo.numpy_interop import from_ndarray, numpy_dtype
from tenmo.dataloader import NumpyDataset, MNIST_MEAN, MNIST_STD
from std.time import perf_counter_ns
from std.sys import has_accelerator
from tenmo.accuracy import Accuracy
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE as LABEL_DTYPE


def train_mnist() raises:
    comptime dtype = DType.float32

    comptime if has_accelerator():
        print("Device: GPU")
    else:
        print("Device: CPU")

    # ── Data ──
    var mnist = Python.import_module("mnist_datasets")
    var train_data = mnist.MNISTLoader(folder="/tmp").load()
    var test_data = mnist.MNISTLoader(folder="/tmp").load(train=False)

    var X_train = from_ndarray[dtype](train_data[0].astype(numpy_dtype(dtype)), copy=True) / 255.0
    var y_train = from_ndarray[LABEL_DTYPE](train_data[1].astype(numpy_dtype(LABEL_DTYPE)), copy=True)
    var X_test = from_ndarray[dtype](test_data[0].astype(numpy_dtype(dtype)), copy=True) / 255.0
    var y_test = from_ndarray[LABEL_DTYPE](test_data[1].astype(numpy_dtype(LABEL_DTYPE)), copy=True)

    # ── DataLoaders ──
    var batch_size = 64
    var train_loader = NumpyDataset[dtype, LABEL_DTYPE](X_train, y_train).into_loader(
        batch_size=batch_size, shuffle=True,
        normalize_mean=Float32(MNIST_MEAN), normalize_std=Float32(MNIST_STD),
    )
    var test_loader = NumpyDataset[dtype, LABEL_DTYPE](X_test, y_test).into_loader(
        batch_size=batch_size,
        normalize_mean=Float32(MNIST_MEAN), normalize_std=Float32(MNIST_STD),
    )

    # ── Model ──
    var model = Sequential[dtype]()
    model.append(
        Linear[dtype](784, 128, init_method="he", bias_zero=True).into(),
        ReLU[dtype]().into(),
        Linear[dtype](128, 32, init_method="he", bias_zero=True).into(),
        ReLU[dtype]().into(),
        Linear[dtype](32, 10, init_method="he", bias_zero=True).into(),
    )
    print("Params:", model.num_parameters())

    var loss_fn = CrossEntropyLoss[dtype]()

    # ── GPU Transfer ──
    comptime if has_accelerator():
        model = model.to_gpu(stop_grad=True)

    var opt = SGD(
        model.parameters(), lr=0.01, momentum=0.9,
        weight_decay=1e-4, clip_norm=1, clip_value=0.5,
    )

    # ── Train ──
    var epochs = 15
    var total_start = perf_counter_ns()
    for epoch in range(epochs):
        var epoch_start = perf_counter_ns()

        # Learning rate decay
        if epoch == 10:
            opt.set_lr(opt.lr / 10)
        if epoch == 15:
            opt.set_lr(opt.lr / 10)

        # Training phase
        model.train()
        loss_fn.train()
        var train_loss = Scalar[dtype](0.0)
        var train_correct = 0
        var train_total = 0

        train_loader.reset()
        while train_loader.__has_next__():
            ref batch = train_loader.__next__()
            var x = batch.features
            var y = batch.labels
            comptime if has_accelerator():
                x = x.to_gpu(sync=False)
                y = y.to_gpu(sync=False)

            var pred = model(x)
            var loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * Float32(batch.batch_size)
            train_correct += Accuracy[dtype].compute(pred, y)
            train_total += batch.batch_size

        # Validation phase
        model.eval()
        loss_fn.eval()
        var val_loss = Scalar[dtype](0.0)
        var val_correct = 0
        var val_total = 0

        test_loader.reset()
        while test_loader.__has_next__():
            ref batch = test_loader.__next__()
            var x = batch.features
            var y = batch.labels
            comptime if has_accelerator():
                x = x.to_gpu(sync=False)
                y = y.to_gpu(sync=False)

            var pred = model(x)
            var loss = loss_fn(pred, y)

            val_loss += loss.item() * Float32(batch.batch_size)
            val_correct += Accuracy[dtype].compute(pred, y)
            val_total += batch.batch_size

        # Report
        var epoch_time = Float64(perf_counter_ns() - epoch_start) / 1e9
        print(
            epoch + 1, "/", epochs,
            "| Loss:", train_loss / Float32(train_total),
            "Train:", 100 * Float64(train_correct) / Float64(train_total), "%",
            "Val:", 100 * Float64(val_correct) / Float64(val_total), "%",
            "Time:", epoch_time, "s",
        )

    var total_time = Float64(perf_counter_ns() - total_start) / 1e9
    print("Total:", total_time, "s")

    # Transfer back to CPU on GPU
    comptime if has_accelerator():
        _ = model.to_cpu()


def main() raises:
    train_mnist()
