from tensors import Tensor
from intlist import IntList
from shared import TensorLite

from testing import assert_true, assert_false


fn test_equality_when_inner_is_tensor() raises:
    a = Tensor.d1([1, 2, 3])
    tl1 = TensorLite.of(a)
    tl2 = TensorLite.of(a)
    assert_true(
        tl1 == tl2, "Equality assertion failed when pointing to same lvalue"
    )
    b = a
    tl3 = TensorLite.of(b)
    assert_true(
        tl1 == tl3,
        "Inequality assertion failed when pointing to assigned lvalues",
    )
    assert_true(
        tl1.inner_id() == tl2.inner_id(),
        "Inner id equality assertion failed when pointing to same lvalue",
    )

    assert_true(
        tl1.inner_id() == tl3.inner_id(),
        (
            "Inner id inequality assertion failed when pointing to assigned"
            " lvalues"
        ),
    )


fn main() raises:
    test_equality_when_inner_is_tensor()
