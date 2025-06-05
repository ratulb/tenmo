from utils import StaticTuple
from memory import UnsafePointer
from tensors import Tensor


struct Ancestors[dtype: DType = DType.float32](
    Sized & Copyable & Movable & EqualityComparable
):
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

    fn __ne__(self, other: Self) -> Bool:
        return ~(self == other)

    fn __eq__(self, other: Self) -> Bool:
        if self.capacity() != other.capacity() or len(self) != len(other):
            return False
        if self.capacity() == other.capacity() and len(self) == len(other):
            for i in range(len(self)):
                if self.ancestors[i] != other.ancestors[i]:
                    return False
        return True

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

    fn free(owned self):
    #fn __del__(owned self):
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
    assert_true(
        ancestors.get(0).value()[].numels() == 10,
        "Get numels 10 assertion failed",
    )
    assert_true(
        ancestors.get(1) == None,
        "Accessing beyond current length assertion failed",
    )
    # Now length would be 2
    ancestors.set(tensor, tensor)

    assert_true(
        ancestors.get(1) is not None,
        "Accessing within length assertion failed",
    )
    tensor1 = ancestors.get(0).value()[]
    tensor2 = ancestors.get(1).value()[]
    same_tensor = (tensor == tensor1).all_true() and (
        tensor == tensor2
    ).all_true()

    assert_true(
        same_tensor,
        "Same tensor assertion failed",
    )

    assert_true(
        ancestors.get(0).value()[].shape.num_elements() == 10,
        "Get shape num elements 10 assertion failed",
    )


fn test_ancestor_equality_check() raises:
    ancestors = Ancestors()
    tensor = Tensor.ones(32, 128, requires_grad=True)
    assert_true(
        tensor.ancestors is None, "Tensor ancestors None assertion failed"
    )
    output = 100 * tensor
    assert_true(
        output.ancestors is not None,
        "Tensor ancestors not None assertion failed",
    )
    not_same = output.ancestors.value() != ancestors
    assert_true(not_same, "Tensor ancestors unequality assertion failed")
    ancestors.set(tensor)
    same = output.ancestors.value() == ancestors
    assert_true(same, "Tensor ancestors equality assertion failed")


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
    # 3
    test_ancestor_equality_check()
