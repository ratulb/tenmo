from tensors import Tensor
from shapes import Shape


fn main() raises:
    # Training data (x: inputs, y: targets)
    var x = Tensor.of(1.0, 2.0, 3.0, 4.0).to_dtype[DType.float32]()
    var y = Tensor.of(5.0, 7.0, 9.0, 11.0).to_dtype[
        DType.float32
    ]()  # y = 2x + 3

    # Parameters to learn (initialized arbitrarily)
    var w = Tensor.rand(1, requires_grad=True)
    var b = Tensor.rand(1, requires_grad=True)

    var learning_rate: Scalar[DType.float32] = 0.01555

    for epoch in range(1500):  # More epochs → better convergence
        # Forward pass: predict
        y_pred = x * w + b

        # Loss: mean squared error
        loss = ((y_pred - y) ** 2).mean()

        # Backward pass
        loss.backward()

        # Print progress
        if epoch % 100 == 0:
            print("Loss:")
            print(loss.item())

        # SGD update
        w.buffer[0] -= learning_rate * w.gradbox[].buffer[0]
        b.buffer[0] -= learning_rate * b.gradbox[].buffer[0]

        # Zero gradients for next step
        w.zero_grad()
        b.zero_grad()

    print()
    print()
    print()
    print()

    print("w:")
    w.print()
    b.print()
    print("b:")
