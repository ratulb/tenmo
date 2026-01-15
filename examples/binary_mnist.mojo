from tenmo import Tensor
from net import Linear, ReLU, Sequential, BCELoss, Sigmoid
from sgd import SGD
from python import Python, PythonObject
from numpy_interop import from_ndarray
from common_utils import now
from dataloader import *


fn train_mnist_binary() raises:
    """Train binary classifier: digit 0 vs digit 1."""
    print("=" * 80)
    print("MNIST Binary Classification (0 vs 1)")
    print("=" * 80 + "\n")

    # Load data
    var mnist = Python.import_module("mnist_datasets")
    var loader = mnist.MNISTLoader(folder="/tmp")

    var train_data = loader.load()
    var train_images_np = train_data[0]
    var train_labels_np = train_data[1]

    # Filter only 0s and 1s
    var np = Python.import_module("numpy")
    var mask = (train_labels_np == 0) | (train_labels_np == 1)
    var filtered_images = train_images_np[mask]
    var filtered_labels = train_labels_np[mask]

    filtered_images = filtered_images.astype(np.float32)
    filtered_labels = filtered_labels.astype(np.float32)

    print("Filtered to", len(filtered_labels), "samples (0s and 1s only)\n")

    # Convert to Mojo
    alias dtype = DType.float32
    var X_train = from_ndarray[dtype](filtered_images, copy=True) / 255.0
    var y_train = from_ndarray[dtype](filtered_labels, copy=True)
    y_train = y_train.reshape(-1, 1)

    # Create dataset and loader
    var train_dataset = NumpyDataset[dtype](X_train, y_train)
    var train_loader = train_dataset.into_loader(
        batch_size=128, shuffle=True, drop_last=False
    )

    # Simple model
    var model = Sequential[dtype]()
    model.append(
        Linear[dtype](784, 32, init_method="xavier").into(),
        ReLU[dtype]().into(),
        Linear[dtype](32, 8, init_method="xavier").into(),
        ReLU[dtype]().into(),
        Linear[dtype](8, 1, init_method="xavier").into(),
        Sigmoid[dtype]().into(),
    )

    var criterion = BCELoss[dtype]()
    var optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training
    for epoch in range(10):
        model.train()
        criterion.train()
        var epoch_start = now()

        var epoch_loss = Scalar[dtype](0.0)
        var epoch_correct = 0
        var epoch_total = 0
        var batch_num = 0

        for batch in train_loader:
            var pred = model(batch.features)
            var loss = criterion(pred, batch.labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            var loss_val = loss.item()
            epoch_loss += loss_val * batch.batch_size
            epoch_correct += accuracy(pred, batch.labels)
            epoch_total += batch.batch_size

            batch_num += 1

        var epoch_time = now() - epoch_start
        print("\n" + "=" * 80)
        print("EPOCH", epoch+1, "COMPLETED")
        print("Total epoch time:", epoch_time, "seconds")
        print("Avg time per batch:", epoch_time / batch_num, "seconds")
        print(
            "Loss:",
            epoch_loss / epoch_total,
            "| Acc:",
            100.0 * epoch_correct / epoch_total,
            "%",
        )
        print("=" * 80 + "\n")


fn accuracy[
    dtype: DType, //
](pred: Tensor[dtype], target: Tensor[dtype]) -> Int:
    var preds = pred.gt(Scalar[dtype](0.5)).to_dtype[dtype]()
    return preds.eq(target).count(Scalar[DType.bool](True))


fn main() raises:
    train_mnist_binary()
