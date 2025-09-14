from tensors import Tensor
from shapes import Shape
from intlist import IntList
from layers import *
from crossentropy import CrossEntropyLoss
from sgd import *

fn make_synthetic_dataset(
    n_samples: Int,
) -> Tuple[Tensor[DType.float32], Tensor[DType.int64]]:
    # Images: n_samples × 1 × 8 × 8  (simple 8×8 grayscale blobs)
    var X = Tensor[DType.float32].zeros(Shape([n_samples, 1, 8, 8]))
    var y = Tensor[DType.int64].zeros(Shape(n_samples))

    for i in range(n_samples):
        # draw a random label 0–1
        var label = Tensor.randint([1], 0, 2)
        y[IntList([i])] = label[0]

        # simple pattern generator
        if label[0] == 0:
            # horizontal bar
            for r in range(3, 5):
                for c in range(8):
                    X[IntList([i, 0, r, c])] = 1.0
        else:
            # vertical bar
            for r in range(8):
                for c in range(3, 5):
                    X[IntList([i, 0, r, c])] = 1.0

    # flatten to (n_samples × 64)
    var X_flat = X.flatten(start_dim=1)  # squash everything but batch dim
    return (X_flat, y)



fn test_tiny_training_smoke() raises:
    print("test_tiny_training_smoke")

    var (X, y) = make_synthetic_dataset(20)  # small dataset
    var model = Sequential[DType.float32]([
        Linear[DType.float32](64, 32).into(),
        ReLU[DType.float32]().into(),
        Linear[DType.float32](32, 2).into()
    ])

    #var criterion = CrossEntropyLoss[DType.float32]()
    var criterion = CrossEntropyLoss()
    var optim = SGD(model.parameters_ptrs(), lr=0.1)

    for epoch in range(20):
        # forward
        var logits = model(X)
        var loss = criterion(logits, y.to_dtype[DType.float32]())

        # backward
        #optim.zero_grad()
        loss.backward()
        optim.step()

        print("epoch", epoch, "loss", loss.item())





fn main() raises:
    test_tiny_training_smoke()
