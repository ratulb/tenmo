# run_tensor.mojo  (sits at repo root, outside tenmo/)
from tenmo.tensor import Tensor

fn main_1() raises:
    # whatever quick test you want
    var t = Tensor[DType.float32].ones(3, 3)
    t.print()


fn main() raises:
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a * 2
    var c = a * 3
    var d = b + c
    var loss = d.sum()
    loss.backward()
    # a.grad() == [5.0, 5.0, 5.0]
    a.grad().print()
