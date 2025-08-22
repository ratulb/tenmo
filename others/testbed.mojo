from tensors import Tensor
from views import TensorView
from intlist import IntList
from shapes import Shape
from os import abort
from testing import assert_true
from common_utils import id
from shared import TensorLike
from utils import Variant

fn main() raises:
    s = Shape.Void
    print(len(s), s.rank())
    a = Tensor.arange(6, requires_grad=True).reshape(2, 3)
    # Every other column
    z = a[:, ::2]
    print(z.strides, a.strides, a.load(1, 1), z.load[1](1,0))  # (3, 2) - stride[1] is now 2
    values = SIMD[DType.float32, 1](9)
    z.store[1](1, 1, values)
    z.print()

    a.print()
    print()
    a.grad[].print()
    r = a.data.load[width=2](3)

    print(r)

    g = G()
    g[1:5:3]


alias IndexArg = Variant[Int, Slice]


fn test_reshape() raises:
    print("test_reshape")
    tensor = Tensor.rand(3, 3)
    reshaped = tensor.reshape(9)
    assert_true(
        tensor[2, 2] == reshaped[8], "reshape __getitem__ assertion 1 failed"
    )
    assert_true(
        tensor.reshape(1, 9)[0, 8] == tensor[2, 2],
        "reshape __getitem__ assertion 2 failed",
    )
    assert_true(
        tensor.reshape(9, 1)[0, 0] == tensor[0, 0],
        "reshape __getitem__ assertion 3 failed",
    )

    tensor = Tensor.of(42)
    assert_true(
        tensor.shape == Shape.Unit, "Unit tensor shape assertion failure"
    )
    reshaped = tensor.reshape(1, 1)
    assert_true(
        reshaped.shape == Shape.of(1, 1) and reshaped[0, 0] == tensor[0],
        "post reshape shape and get assertion failed",
    )
    tensor = Tensor.scalar(42)
    reshaped = tensor.reshape(1, 1)
    assert_true(
        reshaped.shape == Shape.of(1, 1) and reshaped[0, 0] == tensor.item(),
        "post reshape shape and get assertion failed for scalar tensor",
    )
    reshaped = tensor.reshape(1)
    assert_true(
        reshaped.shape == Shape.Unit and reshaped[0] == tensor.item(),
        "post reshape 2 - shape and get assertion failed for scalar tensor",
    )
    assert_true(
        reshaped.reshape(1, 1, 1, 1)[0, 0, 0, 0] == tensor.item(),
        "post reshape 3 - item assertion failed for scalar tensor",
    )

    tensor = Tensor.rand(1, 1)
    reshaped = tensor.reshape()
    assert_true(
        reshaped.shape == Shape.Void and reshaped.item() == tensor[0, 0],
        "post reshape random tensor - shape and get assertion failed",
    )
    tensor = Tensor.scalar(42, requires_grad=True)
    result = tensor * 3
    # result.backward()
    result.backward()
    assert_true(tensor.grad[].item() == 3.0)
    tensor2 = tensor.reshape(1)
    tensor.gprint()
    result = tensor2 * 42
    tensor.gprint()
    tensor2.gprint()
    # result.backward()
    result.backward()
    tensor.gprint()
    tensor2.gprint()
    # tensor3 = tensor2.reshape(1,1,1,1,1)
    tensor3 = tensor2.reshape(1, 1, 1, 1, 1)
    result = tensor3 * 12
    # result.backward()
    result.backward()

    tensor3.gprint()
    tensor2.gprint()
    tensor.gprint()

fn take_a_ptr(ptr: UnsafePointer[String]):
    print("The length is: ", len(ptr[]))


struct Slicer:
    @staticmethod
    fn slice(
        slice: Slice, end: Int, start: Int = 0, step: Int = 1
    ) -> (Int, Int, Int):
        _start, _end, _step = (
            slice.start.or_else(start),
            slice.end.or_else(end),
            slice.step.or_else(step),
        )
        return _start, _end, _step


@fieldwise_init
struct G:
    fn __getitem__(self, slice: Slice):
        print(slice.__str__())
        start, end, step = Slicer.slice(slice, 100)
        print(start, end, step)




fn test_tensor_reuse_mixed() raises:
    print("test_tensor_reuse_mixed")
    var x = Tensor.scalar(3.0, requires_grad=True)
    var y = x * x + x  # 3 * 3 + 3 = 12

    y.backward()

    assert_true(y.item() == 12.0, "Value check")
    assert_true(x.grad[].item() == 2 * 3 + 1, "∂y/∂x = 2x + 1 = 6 + 1 = 7")


fn test_mean_with_keepdims() raises:
    _ = """a = Tensor.d2([[1, 2], [3, 4]], requires_grad=True)
    s = a.sum()
    s.backward()
    assert_true(s.item() == 10.0)
    assert_true(a.grad[].all_close(Tensor.d2([[1, 1], [1, 1]])))"""

    _ = """var A = Tensor.scalar(3.0, requires_grad=True)
    var B = Tensor.scalar(4.0, requires_grad=True)
    var C = A + B
    C.backward()
    assert_true(C.item() == 7.0)
    assert_true(A.grad[].item() == 1.0)
    assert_true(B.grad[].item() == 1.0)"""

    _ = """fn test_training_convergence() raises:
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var y = Tensor.d2([[13.0], [23.0], [33.0]])

    var w = Tensor.rand(2, 1, requires_grad=True)
    var b = Tensor.rand(1, requires_grad=True)

    for epoch in range(1000):
        var y_pred = x.matmul(w) + b
        var loss = ((y_pred - y) ** 2).mean()
        #print("loss: ", loss.item())
        loss.backward()

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

fn test_step_once() raises:

    x = Tensor.d2([[1.0, 2.0]])
    y = Tensor.d2([[13.0]])

    w = Tensor.d2([[1.0], [1.0]], requires_grad=True)
    b = Tensor.d1([1], requires_grad=True)

    y_pred = x.matmul(w) + b
    loss = ((y_pred - y) ** 2).mean()

    print("\nLoss:\n")
    loss.print()

    loss.backward()

    print("\nw.grad:\n")
    w.grad[].print()
    print("\nb.grad:\n")
    b.grad[].print()"""

    _ = """
    y_pred = x.matmul(w) + b
        = [[1*1 + 2*1]] + 1 = [[3.0]] + 1 = [[4.0]]
    loss = (4 - 13)^2 = 81.0"""

    _ = """
    dL/dy_pred = 2 * (y_pred - y) / N = 2 * (4 - 13) = -18.0

    dL/dw = x.T.matmul(dL/dy_pred) = [[1], [2]] * -18.0 = [[-18], [-36]]
    dL/db = sum(-18.0) = -18.0"""
