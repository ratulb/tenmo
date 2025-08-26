from shapes import Shape
from intlist import IntList
from validators import Validator


from testing import assert_true


fn test_validate_new_shape() raises:
    print("test_validate_new_shape")
    curr_dims = IntList.new([3, 4, 5])
    new_dims = IntList.new([2, -1, 10])
    concrete_shape = Validator.validate_new_shape(curr_dims, new_dims)
    assert_true(
        concrete_shape == Shape.of(2, 3, 10),
        "validate_new_shape assertion 1 failed",
    )
    new_dims = IntList.new([-1])
    concrete_shape = Validator.validate_new_shape(curr_dims, new_dims)
    assert_true(
        concrete_shape == Shape.of(60), "validate_new_shape assertion 2 failed"
    )


fn main() raises:
    test_validate_new_shape()
