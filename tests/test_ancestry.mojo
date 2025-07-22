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
    print(ancestors.__contains__(tl2))
    # print(ancestors.get(0)[].inner_id())
    # print(tl1 == tl2)
    for each in ancestors:
        print(each[].inner_id())


fn main() raises:
    test_contains()
