from tensors import Tensor
from intlist import IntList
from shapes import Shape
from os import abort
from testing import assert_true


fn test_training_convergence() raises:
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var y = Tensor.d2([[13.0], [23.0], [33.0]])

    var w = Tensor.rand(2, 1, requires_grad=True)
    var b = Tensor.rand(1, requires_grad=True)

    for epoch in range(10000):
        var y_pred = x.matmul(w) + b
        var loss = ((y_pred - y) ** 2).mean()
        #print("loss: ", loss.item())
        Tensor.walk_backward(loss)

        # SGD
        w.data[] -= 0.011 * w.grad[].data[]
        b.data[] -= 0.011 * b.grad[].data[]
        w.zero_grad()
        b.zero_grad()

    # After training
    w.print()
    b.print()
    # assert_true(w.all_close(Tensor.d2([[2.0], [3.0]])))
    # assert_true(b.all_close(Tensor[DType.float32].d1([5.0])))
    (x.matmul(w) + b).print()


fn main() raises:
    test_training_convergence()
    _ = """x = Tensor.d2([[1.0, 2.0]])
    y = Tensor.d2([[13.0]])

    w = Tensor.d2([[1.0], [1.0]], requires_grad=True)
    b = Tensor.d1([1], requires_grad=True)

    y_pred = x.matmul(w) + b
    loss = ((y_pred - y) ** 2).mean()

    print("\nLoss:\n")
    loss.print()

    Tensor.walk_backward(loss)

    print("\nw.grad:\n")
    w.grad[].print()
    print("\nb.grad:\n")
    b.grad[].print()"""

    _="""
    y_pred = x.matmul(w) + b
        = [[1*1 + 2*1]] + 1 = [[3.0]] + 1 = [[4.0]]
    loss = (4 - 13)^2 = 81.0"""

    _="""
    dL/dy_pred = 2 * (y_pred - y) / N = 2 * (4 - 13) = -18.0

    dL/dw = x.T.matmul(dL/dy_pred) = [[1], [2]] * -18.0 = [[-18], [-36]]
    dL/db = sum(-18.0) = -18.0"""




