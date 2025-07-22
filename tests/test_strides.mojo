from shapes import Shape
from strides import Strides
from testing import assert_true
from intlist import IntList

fn test_compute_default_strides() raises:
    shape = Shape.of(2, 3, 4)
    strides = Strides.default(shape)
    assert_true(
        strides == Strides(IntList(12, 4, 1)),
        "stride compute assertion 1 failed",
    )


fn main() raises:
    print("Running strides tests")
    test_compute_default_strides()


