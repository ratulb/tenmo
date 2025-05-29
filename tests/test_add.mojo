from tensors import Tensor
from testing import assert_true


fn test_tensor_add_scalar() raises:
    t1 = Tensor(4)
    for i in range(4):
        t1[i] = i
    added_3 = t1 + 3
    expected = Tensor.arange[7, 3]().to_dtype[DType.float32]()
    expected.print()
    result = added_3 == expected
    result.print()
    assert_true(result.all_true(), "Tensor scalar addition failed")


fn test_tensor_add_tensor() raises:
    t1 = Tensor(4)
    for i in range(4):
        t1[i] = i
    t2 = Tensor(4)
    for i in range(4):
        t2[i] = i * 2
    summed = t1 + t2
    tensor = Tensor.arange[4]().to_dtype[DType.float32]()
    expected = tensor * 3
    result = summed == expected
    assert_true(result.all_true(), "Tensor tensor addition failed")


fn main() raises:
    test_tensor_add_scalar()
    test_tensor_add_tensor()
