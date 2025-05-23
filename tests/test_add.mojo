from tensors import Tensor
from testing import assert_true


fn all_true(yes: Scalar[DType.bool]) -> Bool:
    return yes == True


fn test_tensor_add_scalar() raises:
    t1 = Tensor(4)
    for i in range(4):
        t1[i] = i
    added_3 = t1 + 3
    expected = Tensor.arange(7, start=3)
    result = added_3 == expected
    assert_true(result.for_all(all_true), "Tensor scalar addition failed")


fn test_tensor_add_tensor() raises:
    t1 = Tensor(4)
    for i in range(4):
        t1[i] = i
    t2 = Tensor(4)
    for i in range(4):
        t2[i] = i * 2
    summed = t1 + t2
    expected = Tensor.arange(4) * 3
    result = summed == expected
    assert_true(result.for_all(all_true), "Tensor tensor addition failed")


fn main() raises:
    test_tensor_add_scalar()
    test_tensor_add_tensor()
