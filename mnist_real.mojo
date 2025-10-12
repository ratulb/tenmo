from tensors import Tensor
from layers import Linear, ReLU, Sequential
from crossentropy import CrossEntropyLoss
from sgd import SGD
from testing import assert_true
from python import Python, PythonObject
from numpy_interop import from_ndarray
from common_utils import id


fn load_mnist_subset(
    num_train: Int = 1000, num_test: Int = 200
) raises -> (
    Tensor[DType.float32],
    Tensor[DType.int32],
    Tensor[DType.float32],
    Tensor[DType.int32],
):
    np = Python.import_module("numpy")
    mnist = Python.import_module("mnist_datasets")
    loader = mnist.MNISTLoader(folder="/tmp")

    var train_data: PythonObject = loader.load()
    # Load test dataset
    var test_data: PythonObject = loader.load(train=False)
    # Load train
    train_images, train_labels = train_data[0], train_data[1]
    train_images_small = train_images[:num_train]
    train_labels_small = train_labels[:num_train]

    # Load test
    test_images, test_labels = test_data[0], test_data[1]
    test_images_small = test_images[:num_test]
    test_labels_small = test_labels[:num_test]

    train_images_small = train_images_small.astype(np.float32)
    test_images_small = test_images_small.astype(np.float32)
    train_images_small /= 255.0
    test_images_small /= 255.0

    train_labels_small = train_labels_small.astype(np.int32)
    test_labels_small = test_labels_small.astype(np.int32)

    # Convert to Mojo Tensor
    var train_x = from_ndarray[DType.float32](train_images_small, True)
    var train_y = from_ndarray[DType.int32](train_labels_small, True)
    var test_x = from_ndarray[DType.float32](test_images_small, True)
    var test_y = from_ndarray[DType.int32](test_labels_small, True)

    # Flatten each image to row vector
    _ = """train_x = train_x.reshape([num_train, 28*28])
    test_x = test_x.reshape([num_test, 28*28])

    # Make sure labels are column vectors or 1D
    train_y = train_y.reshape([num_train])
    test_y = test_y.reshape([num_test])"""

    return (train_x, train_y, test_x, test_y)


fn get_batches(
    x: Tensor[DType.float32], y: Tensor[DType.int32], batch_size: Int
) -> List[(Tensor[DType.float32], Tensor[DType.int32])]:
    n = x.shape[0]
    num_batches = (n + batch_size - 1) // batch_size
    batches = List[(Tensor[DType.float32], Tensor[DType.int32])](
        capacity=num_batches
    )
    xs = x
    ys = y
    i = 0
    while i < n:
        end = min(i + batch_size, n)

        x_batch = xs.slice(i, end).contiguous[
            track_grad=False
        ]()  # shape: (batch_size, 784)
        y_batch = ys.slice(i, end).contiguous[
            track_grad=False
        ]()  # shape: (batch_size,)
        batches.append((x_batch^, y_batch^))
        i += batch_size

    return batches

fn main() raises:
    num_train = 8000
    num_test = 1
    batch_size = 32
    epochs = 25
    lr = Scalar[DType.float32](0.006)

    var (train_x, train_y, test_x, test_y) = load_mnist_subset(
        num_train, num_test
    )

    # Define model
    model = Sequential[DType.float32]()
    model.append(Linear(784, 128).into())
    model.append(ReLU().into())
    model.append(Linear(128, 10).into())

    # Loss + optimizer
    var criterion = CrossEntropyLoss[DType.float32]()
    var optimizer = SGD[DType.float32](model.parameters_ptrs(), lr, False)

    # Training
    for epoch in range(epochs):
        var epoch_loss: Float32 = 0.0
        var correct: Int = 0
        var total: Int = 0

        # Create batches for this epoch
        var batches = get_batches(train_x, train_y, batch_size)

        for xb, yb in batches:
            xs = xb
            var logits = model(xs)
            var loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * yb.shape[0]
            var preds = logits.argmax(axis=1)
            correct += (
                (preds == yb).to_dtype[DType.int32]().sum().item().__int__()
            )
            total += yb.shape[0]

            # Free up temporaries
            preds.free()
            loss.free()
            xb.free()
            yb.free()
            logits.free()

        # CRITICAL: Free the batches list and its contents
        for i in range(batches.__len__()):
            var batch = batches[i]
            batch[0].free()  # Free x_batch
            batch[1].free()  # Free y_batch
        batches.clear()  # Free the list itself

        print(
            "Epoch",
            epoch,
            "Loss:",
            epoch_loss / total,
            "Accuracy:",
            Float32(correct) / Float32(total),
        )

    # Free main datasets
    train_x.free()
    train_y.free()
    test_x.free()
    test_y.free()
