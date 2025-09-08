from tensors import Tensor
from layers import *
from sgd import SGD
from crossentropy import CrossEntropyLoss

fn test_sgd() raises:

    var model = Sequential()
    model.append(Linear(784, 128).into())
    model.append(ReLU().into())
    model.append(Linear(128, 10).into())

    # collect params
    var params = model.parameters_ptrs()

    # make optimizer
    var optimizer = SGD(params)

    # forward
    var x = Tensor.rand([4, 784], requires_grad=True)
    var y = Tensor.d1([1, 3, 0, 2])

    var logits = model(x)
    var criterion = CrossEntropyLoss()
    var loss = criterion(logits, y)

    # backward
    loss.backward()

    # optimizer step
    optimizer.step()
    x.gradbox[].print()

fn main() raises:

    test_sgd()
