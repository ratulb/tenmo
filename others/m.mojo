from tensors import Tensor
from shapes import Shape
from intlist import IntList
from layers import Linear, ReLU, Sequential
from crossentropy import CrossEntropyLoss
from sgd import SGD
from testing import assert_true
from utils.numerics import max_finite
from common_utils import log_debug, s, i as Row


# Synthetic dataset generator
fn make_synthetic_mnist_dataset(
    n_samples: Int,
) -> (Tensor[DType.float32], Tensor[DType.int32]):
    height = 28
    width = 28
    num_classes = 10

    # Allocate tensors
    var X = Tensor[DType.float32](Shape([n_samples, height * width]))
    var y = Tensor[DType.int32](Shape(n_samples))

    for i in range(n_samples):
        # Pick class
        label = Tensor.randint([1], 0, num_classes).item()
        y[i] = label.__int__()

        # Create blank canvas
        var img = Tensor[DType.float32](Shape([height, width]))
        img.fill(0.0)

        # Draw synthetic shape depending on label
        if label == 0:  # vertical bar
            for r in range(height):
                img[r, width // 2] = 1.0
        elif label == 1:  # horizontal bar
            for c in range(width):
                img[height // 2, c] = 1.0
        elif label == 2:  # diagonal
            for d in range(min(height, width)):
                img[d, d] = 1.0
        elif label == 3:  # cross
            for r in range(height):
                img[r, width // 2] = 1.0
            for c in range(width):
                img[height // 2, c] = 1.0
        elif label == 4:  # square border
            for r in range(height):
                img[r, 0] = 1.0
                img[r, width - 1] = 1.0
            for c in range(width):
                img[0, c] = 1.0
                img[height - 1, c] = 1.0
        elif label == 5:  # dot
            img[height // 2, width // 2] = 1.0
        elif label == 6:  # two vertical bars
            for r in range(height):
                img[r, width // 3] = 1.0
                img[r, 2 * width // 3] = 1.0
        elif label == 7:  # two horizontal bars
            for c in range(width):
                img[height // 3, c] = 1.0
                img[2 * height // 3, c] = 1.0
        elif label == 8:  # X pattern
            for d in range(min(height, width)):
                img[d, d] = 1.0
                img[d, width - 1 - d] = 1.0
        elif label == 9:  # checkerboard small patch
            for r in range(0, height, 4):
                for c in range(0, width, 4):
                    img[r, c] = 1.0

        # Flatten into X
        img_flattened = img.flatten[track_grad=False]()
        X.set(img_flattened, Row(i), s())

    return (X, y)


# Batching utility
fn get_batches(
    x: Tensor[DType.float32], y: Tensor[DType.int32], batch_size: Int
) -> List[(Tensor[DType.float32], Tensor[DType.int32])]:
    n = x.shape[0]
    num_batch = (n + batch_size - 1) // batch_size
    batches = List[(Tensor[DType.float32], Tensor[DType.int32])](
        capacity=num_batch
    )
    xs = x
    ys = y
    i = 0
    while i < n:
        end = min(i + batch_size, n)
        x_batch = xs.slice(i, end).contiguous()
        y_batch = ys.slice(i, end).contiguous()
        batches.append((x_batch^, y_batch^))
        i += batch_size
    return batches


# Training loop
fn fit(
    model: Sequential[DType.float32],
    optimizer: SGD[DType.float32],
    criterion: CrossEntropyLoss[DType.float32],
    train_x: Tensor[DType.float32],
    train_y: Tensor[DType.float32],
    epochs: Int,
    batch_size: Int,
):
        _="""for epoch in range(epochs):
        var epoch_loss: Float32 = 0.0
        var correct: Int = 0
        var total: Int = 0
        batches = get_batches(train_x, train_y, batch_size)
        for xb, yb in batches:
            # Forward
            var logits = model(xb)
            var loss = criterion(logits, yb)

            # Backward
            loss.backward()

            # Step
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()
            # Accuracy
            var preds = logits.argmax(axis=1).float()
            correct += Int((preds == yb).float().sum().item())
            total += yb.shape[0]

        print(
            "Epoch",
            epoch,
            "Loss:",
            epoch_loss / total,
            "Accuracy:",
            Float32(correct) / Float32(total),
        )"""
        pass


fn main():
    # Make synthetic dataset (like 1000 "images")
    num_samples = 100
    (train_x, train_y) = make_synthetic_mnist_dataset(num_samples)
    # Define model
    model = Sequential[DType.float32]()
    model.append(Linear(784, 128).into())
    model.append(ReLU().into())
    model.append(Linear(128, 10).into())

    # Loss + Optimizer
    var criterion = CrossEntropyLoss[DType.float32]()
    var optimizer = SGD[DType.float32](model.parameters_ptrs(), lr=0.1)
    # Train
    # fit(model, optimizer, criterion, train_x, train_y, epochs=40, batch_size=8)
    epochs = 20
    batch_size = 32
    for epoch in range(epochs):
        var epoch_loss: Float32 = 0.0
        var correct: Int = 0
        var total: Int = 0
        batches = get_batches(train_x, train_y, batch_size)
        for xb, yb in batches:
            # Forward
            var logits = model(xb)
            var loss = criterion(logits, yb)

            # Backward
            loss.backward()

            # Step
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()
            preds = logits.argmax(axis=1)
            result = preds == yb

            correct += result.to_dtype[DType.int32]().sum_all().__int__()
            total += yb.shape[0]

        print(
            "Epoch",
            epoch,
            "Loss:",
            epoch_loss / total,
            "Accuracy:",
            Float32(correct) / Float32(total),
        )
