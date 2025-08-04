from tensors import Tensor
from views import TensorView
from intlist import IntList
from shared import TensorLike

from testing import assert_true, assert_false


fn test_equality_when_inner_is_tensor() raises:
    a = Tensor.d1([1, 2, 3])
    tl1 = TensorLike(UnsafePointer(to=a))
    tl2 = TensorLike(UnsafePointer(to=a))
    assert_true(
        tl1 == tl2, "Equality assertion failed when pointing to same lvalue"
    )
    b = a
    tl3 = TensorLike(UnsafePointer(to=b))
    assert_false(
        tl1 == tl3,
        "Inequality assertion failed when pointing to different lvalues",
    )
    assert_true(
        tl1.inner_id() == tl2.inner_id(),
        "Inner id equality assertion failed when pointing to same lvalue",
    )

    assert_false(
        tl1.inner_id() == tl3.inner_id(),
        (
            "Inner id inequality assertion failed when pointing to different"
            " lvalues"
        ),
    )


fn main() raises:
    test_equality_when_inner_is_tensor()
