from utils import StaticTuple
from memory import UnsafePointer
from tensors import Tensor


struct Ancestors[dtype: DType = DType.float32](Sized & Copyable & Movable):
    var ancestors: StaticTuple[UnsafePointer[Tensor[dtype]], 2]

    fn __init__(out self):
        self.ancestors = StaticTuple[UnsafePointer[Tensor[dtype]], 2](
            UnsafePointer[Tensor[dtype]]()
        )

    fn __init__(
        out self, _ancestors: StaticTuple[UnsafePointer[Tensor[dtype]], 2]
    ):
        self.ancestors = _ancestors

    fn __copyinit__(out self, other: Self):
        self.ancestors = other.ancestors

    fn __moveinit__(out self, owned existing: Self):
        self.ancestors = existing.ancestors

    @always_inline
    fn capacity(self) -> Int:
        return len(self.ancestors)

    fn __len__(self) -> Int:
        length = 0
        for i in range(self.capacity()):
            if self.ancestors[i].__as_bool__() == False:
                break
            length += 1
        return length

    fn get(self, index: Int) -> Optional[UnsafePointer[Tensor[dtype]]]:
        if index < 0 or index >= len(self):
            print("Ancestors -> get(index) invalid index: " + String(index))
            return None
        if self.ancestors[index].__as_bool__() == False:
            print("Ancestors -> get(index) invalid pointer")
            return None
        return Optional(self.ancestors[index])

    fn set(mut self, ptr: UnsafePointer[Tensor[dtype]]):
        self.ancestors[0] = ptr
        print("After setting single ptr: ", self.ancestors[0].__str__())

    fn set(mut self, tensor: Tensor[dtype]):
        self.set(UnsafePointer(to=tensor))

    fn set(
        mut self,
        tensor1: Tensor[dtype],
        tensor2: Tensor[dtype],
    ):
        self.set(UnsafePointer(to=tensor1), UnsafePointer(to=tensor2))

    fn set(
        mut self,
        ptr1: UnsafePointer[Tensor[dtype]],
        ptr2: UnsafePointer[Tensor[dtype]],
    ):
        self.ancestors[0] = ptr1
        self.ancestors[1] = ptr2

    fn __del__(owned self):
        print(
            "Deleting Ancestry with length: ",
            len(self),
            "and capacity: ",
            self.capacity(),
        )
        _ = self.ancestors
        _ = self^


from testing import assert_true


fn test_ancestors_set() raises:
    ancestors = Ancestors()
    tensor = Tensor.ones(10)
    ancestors.set(tensor)
    assert_true(ancestors.get(0) is not None, "Get assertion failed")


# Test that pointer obtained at callsite and evaluated at the callee is same
fn test_is_same_pointer[
    dtype: DType
](ptr: UnsafePointer[Tensor[dtype]], tensor: Tensor[dtype]) raises:
    ptr2 = UnsafePointer(to=tensor)
    assert_true(ptr == ptr2, "Same pointer assertion failed")
    assert_true(
        len(ptr[]) == len(ptr2[]),
        "Same pointer length equqlity assertion failed",
    )
    assert_true(
        ptr[].numels() == 1024 and ptr2[].numels() == 1024,
        "Same pointer numels equqlity assertion failed",
    )


fn main() raises:
    tensor = Tensor(32, 32)
    ptr = UnsafePointer(to=tensor)
    # 1
    test_is_same_pointer(ptr, tensor)
    # 2
    test_ancestors_set()
