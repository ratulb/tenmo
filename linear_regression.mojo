from tensors import Tensor
from shapes import Shape


fn main() raises:
    _="""w = Tensor.scalar(2.0, requires_grad=True)
    x = Tensor.of(1.0, 2.0, 3.0, 4.0)
    y = w * x
    loss = y.sum()
    Tensor.walk_backward(loss)
    w.grad[].print()  # Should be 1 + 2 + 3 + 4 = 10.0

    a = Tensor.of(1.0, 2.0, 3.0, requires_grad=True)
    b = a.reshape(Shape.of(1, 3))
    c = b * 2
    loss = c.sum()
    loss.backward()

    a.grad[].print()  # Should print: [2.0, 2.0, 2.0]"""



    # Training data (x: inputs, y: targets)
    var x = Tensor.of(1.0, 2.0, 3.0, 4.0)
    var y = Tensor.of(5.0, 7.0, 9.0, 11.0) # y = 2x + 3

    # Parameters to learn (initialized arbitrarily)
    var w = Tensor.scalar(0.955, requires_grad=True)
    var b = Tensor.scalar(0.955, requires_grad=True)

    learning_rate = 0.01555

    for epoch in range(10000):  # More epochs → better convergence
        # Forward pass: predict
        y_pred = x * w + b

        # Loss: mean squared error
        loss = ((y_pred - y) ** 2).mean()

        # Backward pass
        Tensor.walk_backward(loss)

        # Print progress
        if epoch % 100 == 0:
            #print("Epoch:", epoch)
            print("Loss:")
            loss.print()
            #print("w:")
            #w.print()
            #print("b:")
            #b.print()

        # SGD update
        #w = w - learning_rate * w.grad[]
        #b = b - learning_rate * b.grad[]
        w.data[] -= learning_rate * w.grad[].data[]
        b.data[] -= learning_rate * b.grad[].data[]

        # Zero gradients for next step
        w.grad[].fill(0.0)
        b.grad[].fill(0.0)

    print()
    print()
    print()
    print()

    print("w:")
    w.print()
    print("b:")
    b.print()


