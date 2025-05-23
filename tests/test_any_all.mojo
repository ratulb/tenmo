from tensors import Tensor
from testing import assert_true


fn test_tensor_any_1() raises:
    tensor = Tensor[dtype = DType.bool].ones(4)
    assert_true(tensor.any_true(), "Tensor any_true failed")


fn test_tensor_all_1() raises:
    tensor = Tensor[dtype = DType.bool].ones(4)
    assert_true(tensor.all_true(), "Tensor all_true failed")


fn main() raises:
    test_tensor_any_1()
    test_tensor_all_1()
