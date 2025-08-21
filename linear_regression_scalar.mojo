from tensors import Tensor


fn main() raises:
    # Training data (x: inputs, y: targets)
    var x = Tensor.of(1.0, 2.0, 3.0, 4.0)
    var y = Tensor.of(5.0, 7.0, 9.0, 11.0)  # y = 2x + 3

    # Parameters to learn (initialized arbitrarily)
    var w = Tensor.scalar(0.955, requires_grad=True)
    var b = Tensor.scalar(0.955, requires_grad=True)

    var learning_rate = 0.01555

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
            loss.print()

        # SGD update
        w = w - learning_rate * w.gradbox[]
        b = b - learning_rate * b.gradbox[]

        # Zero gradients for next step
        w.gradbox[].fill(0.0)
        b.gradbox[].fill(0.0)

    print()
    print()
    print()

    print("w:")
    w.print()
    print()
    print("b:")
    b.print()
