from tensors import Tensor
from shared import TensorLite
from operators import AddTensor, SubtractTensor
from os import abort
from strides import Strides
from shapes import Shape
from testing import assert_true

fn main() raises:
    a = Tensor.rand(4, 4, requires_grad=True)
    t = a.transpose(0, 1)
    p = t.permute([1, 0])
    c = p.contiguous()
    s = c.sum()
    s.backward(42)
    assert_true(
        Strides.default(a.gradbox[].shape) == Strides.default(a.shape)
    )
    a.gradbox[].print()
    print()
    expected = Tensor.full(Shape.of(4, 4), 42)
    print("going to compare")
    result = expected == a.gradbox[]
    for idx in result.shape:
        #print(expected[idx])
        pass
    print("compared")
    #print(a.gradbox[].is_contiguous(), expected.is_contiguous())
    assert_true(
        (result).all_true(),
        "grad propagation through contiguous failed",
    )
    _ = s
    _ = c
    _ = p
    _ = t
    _ = a
    _ = expected
    _="""x3 = Tensor.rand(3, 3, requires_grad=True)
    v3 = x3.into_view()
    y3 = v3.permute([0, 1])
    loss3 = y3.sum()
    loss3.backward()
    assert_true(x3.gradbox[].all_close(Tensor.ones(3, 3)))
    loss3.free()
    y3.free()
    v3.free()
    x3.free()"""




