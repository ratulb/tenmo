from tensors import Tensor
from testing import assert_true


fn test_iteration() raises:
    """
    Tests tensor iterator implementation.
    """
    tensor = Tensor.arange(10).reshape[2](5, 2)
    i = 0
    for each in tensor:
        assert_true(each[] == i, "Tensor iterator failed")
        i += 1


fn test_mutation() raises:
    """
    Tests mutation via iteration.
    """
    tensor = Tensor.arange(10).reshape[2](5, 2)
    for each in tensor:
        each[] *= 2
    value = 0
    for i in range(tensor.shape[0]):
        assert_true(
            tensor[i, 0] == value and tensor[i, 1] == tensor[i, 0] + 2,
            "Tensor iterator failed",
        )
        value += 4

fn main() raises:
    test_iteration()
    test_mutation()
