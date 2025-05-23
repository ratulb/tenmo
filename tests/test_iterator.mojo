from tensors import Tensor
from testing import assert_true


fn test_tensor_iterator() raises:
    """
    Tests tensor iterator implementation.
    """
    tensor = Tensor.arange(10).reshape[2](5,2)
    i = 0
    for each in tensor:
        assert_true(each[] == i, "Tensor iterator failed")
        i += 1

fn main() raises:
    test_tensor_iterator()
