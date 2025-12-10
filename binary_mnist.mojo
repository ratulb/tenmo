from tenmo import Tensor
from net import Linear, ReLU, Sequential, SGD, BCELoss, Sigmoid
from python import Python, PythonObject
from numpy_interop import from_ndarray
from common_utils import id, now, accuracy
from data import *


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
    var y_train = from_ndarray[dtype](filtered_labels, copy=True).reshape(-1, 1)

    # Create dataset and loader
    var train_dataset = NumpyDataset[dtype](X_train, y_train)
    var train_loader = DataLoader[dtype](
        train_dataset^,
        # batch_size=32,
        batch_size=128,
        reshuffle=True,
    )

    # Simple model
    var model = Sequential[dtype]()
    model.append(
        Linear[dtype](784, 32, xavier=True).into(),  # 64 → 32 (2x faster!)
        ReLU[dtype]().into(),
        Linear[dtype](32, 8, xavier=True).into(),
        ReLU[dtype]().into(),
        Linear[dtype](8, 1, xavier=True).into(),
        Sigmoid[dtype]().into(),
    )

    var criterion = BCELoss[dtype]()
    var optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop (Simple to debug)
    for epoch in range(1):
        model.train()
        criterion.train()

        var epoch_start = now()
        print("\n" + "=" * 80)
        print("EPOCH", epoch, "STARTED at", epoch_start, "seconds")
        print("=" * 80)

        var epoch_loss = Scalar[dtype](0.0)
        var epoch_correct = 0
        var epoch_total = 0
        var batch_num = 0

        var loader_start = now()
        var last_batch_end = loader_start

        for batch in train_loader:
            var batch_prep_time = now() - last_batch_end

            var batch_start = now()
            print("\n--- Batch", batch_num, "---")
            print("Prep time:", batch_prep_time, "seconds")

            # Forward
            var forward_start = now()
            var pred = model(batch.features)
            var forward_time = now() - forward_start

            # Loss
            var loss_start = now()
            var loss = criterion(pred, batch.labels)
            var loss_time = now() - loss_start

            # Zero grad
            var zero_start = now()
            optimizer.zero_grad()
            var zero_time = now() - zero_start

            # Backward
            var back_start = now()
            loss.backward()
            var back_time = now() - back_start

            # Step
            var step_start = now()
            optimizer.step()
            var step_time = now() - step_start

            # Metrics
            var metrics_start = now()
            var loss_val = loss.item()
            epoch_loss += loss_val * batch.batch_size
            var correct, _ = accuracy(pred, batch.labels)
            epoch_correct += correct
            epoch_total += batch.batch_size
            var metrics_time = now() - metrics_start

            var batch_total = now() - batch_start

            print(
                "Forward:",
                forward_time,
                "| Loss:",
                loss_time,
                "| Zero:",
                zero_time,
                "| Back:",
                back_time,
                "| Step:",
                step_time,
                "| Metrics:",
                metrics_time,
            )
            print("BATCH TOTAL:", batch_total, "seconds")
            print(
                "Cumulative batches:",
                batch_num + 1,
                "| Time since epoch start:",
                (now() - epoch_start),
                "seconds",
            )

            batch_num += 1
            last_batch_end = now()

            # Emergency exit for debugging
            if batch_num >= 10:
                print("\n!!! STOPPING AT 1 BATCH FOR DEBUG !!!")
                break

        var epoch_time = now() - epoch_start
        print("\n" + "=" * 80)
        print("EPOCH", epoch, "COMPLETED")
        print("Batches processed:", batch_num)
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


fn compute_accuracy_multiclass[
    dtype: DType, //
](pred: Tensor[dtype], target: Tensor[dtype]) -> Int:
    """Compute accuracy for multiclass classification.

    Args:
        pred: Predictions of shape (batch_size, num_classes).
        target: True labels of shape (batch_size, 1) or (batch_size,).

    Returns:
        Number of correct predictions.
    """
    var correct = 0
    var batch_size = pred.shape()[0]
    var num_classes = pred.shape()[1]

    for i in range(batch_size):
        # Find argmax of predictions
        var max_idx = 0
        var max_val = pred[i, 0]
        for j in range(1, num_classes):
            if pred[i, j] > max_val:
                max_val = pred[i, j]
                max_idx = j

        # Get true label
        var true_label = Int(target[i, 0])

        if max_idx == true_label:
            correct += 1

    return correct


fn main() raises:
    train_mnist_binary()
