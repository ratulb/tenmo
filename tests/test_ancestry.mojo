from testing import assert_true
from tensors import Tensor
from ancestry import Ancestors
from shared import TensorLike


fn test_contains() raises:
    a = Tensor.scalar(10)
    tl1 = TensorLike(UnsafePointer(to=a))
    tl2 = TensorLike(UnsafePointer(to=a))
    ancestors = Ancestors[DType.float32].untracked()
    ancestors.append(UnsafePointer(to=tl1))
    assert_true(ancestors.__contains__(tl2))
    for each in ancestors:
        _ = each[].inner_id()


fn main() raises:
    test_contains()
