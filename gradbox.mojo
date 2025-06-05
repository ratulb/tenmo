from tensors import Tensor
from shapes import Shape
from memory import UnsafePointer, memset_zero


struct GradBox[dtype: DType = DType.float32](
    Sized & Copyable & Movable & EqualityComparable
):
    var storage: UnsafePointer[Tensor[dtype]]

    fn __init__(out self, shape: Shape) raises:
        _storage = Tensor[dtype](shape, requires_grad=False)
        memset_zero(_storage.data, _storage.numels())
        self.storage = UnsafePointer[__type_of(_storage)].alloc(1)
        self.storage.init_pointee_move(_storage^)

    fn __init__(out self, tensor: Tensor[dtype]) raises:
        shape = tensor.shape
        self = Self(shape)

    fn shape(self) -> Shape:
        return self.storage[].shape

    fn gradients(ref self) -> ref [self.storage[]] Tensor[dtype]:
        return self.storage[]

    fn __copyinit__(out self, other: Self):
        self.storage = other.storage

    fn __moveinit__(out self, owned existing: Self):
        self.storage = existing.storage

    fn __ne__(self, other: Self) -> Bool:
        return ~(self == other)

    fn __eq__(self, other: Self) -> Bool:
        try:
            return (
                self.storage[].shape == other.storage[].shape
                and (self.storage[] == other.storage[]).all_true()
            )
        except e:
            print("GradBox __eq__ thrown up", e)
            return False

    fn __len__(self) -> Int:
        return len(self.storage[])

    fn __str__(self) -> String:
        return "GradBox" + self.storage[].__str__()

    # fn __del__(owned self):
    fn free(owned self):
        print(
            "Deleting GradBox with length: ",
            len(self),
        )
        for i in range(len(self)):
            (self.storage[].unsafe_ptr() + i).destroy_pointee()
        _ = self.storage
        _ = self^


from testing import assert_true


_ = """fn test_check_update() raises:
    tensor = Tensor.ones(3, 3)
    tensor.print()

    gradbox = GradBox(tensor)
    print(gradbox.__str__())
    inner = gradbox.get()
    with_inner = inner + 10.0
    with_inner.print()
    # result = tensor + Scalar[DType.float32](10.0)
    result = tensor + 10.0
    result.print()


fn main() raises:
    test_check_update()
"""
