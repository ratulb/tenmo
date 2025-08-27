from testing import assert_true
from tensors import Tensor
from ancestry import Ancestors
from shared import TensorLite


fn test_contains() raises:
    a = Tensor.scalar(10)
    tl1 = TensorLite.of(a)
    tl2 = TensorLite.of(a)
    ancestors = Ancestors[DType.float32].untracked()
    ancestors.append(UnsafePointer(to=tl1))
    assert_true(ancestors.__contains__(tl2))
    for each in ancestors:
        _ = each[].inner_id()


fn main() raises:
    test_contains()
