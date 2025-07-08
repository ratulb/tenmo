from tensors import Tensor
from shapes import Shape
from intlist import IntList
from strides import Strides
from memory import UnsafePointer
from os import abort
from shared import TensorLike
from ancestry import Ancestors

fn main():
    pass


struct TensorView[dtype: DType = DType.float32](Copyable & Movable):
    var base_tensor: UnsafePointer[Tensor[dtype]]
    var shape: Shape
    var strides: Strides
    var offset: Int  # default 0

    fn __init__(
        out self,
        base_tensor: UnsafePointer[Tensor[dtype]],
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
    ):
        self.base_tensor = base_tensor
        self.shape = shape
        self.strides = strides
        self.offset = offset

    fn __moveinit__(out self, owned other: Self):
        self.base_tensor = other.base_tensor
        self.shape = other.shape
        self.strides = other.strides
        self.offset = other.offset

    fn __copyinit__(out self, other: Self):
        self.base_tensor = other.base_tensor
        self.shape = other.shape
        self.strides = other.strides
        self.offset = other.offset

    fn is_contiguous(self) -> Bool:
        return self.offset == 0 and self.strides.is_contiguous(self.shape)

    fn into_tensorlike(self) -> TensorLike[dtype]:
        return TensorLike[dtype](self.address())

    # Index calculation: flat offset into underlying tensor's data[]
    fn index_offset(self, indices: IntList) -> Int:
        if not indices.len() == self.shape.rank():
            abort("TensorView → index_offset → rank mismatch")
        var flat_idx = self.offset
        for i in range(indices.len()):
            flat_idx += indices[i] * self.strides[i]
        return flat_idx

    # Element access
    fn __getitem__(self, indices: IntList) -> Scalar[dtype]:
        return self.base_tensor[].data.load[volatile=True](
            self.index_offset(indices)
        )

    fn __setitem__(self, indices: IntList, value: Scalar[dtype]):
        self.base_tensor[].data.store[volatile=True](
            self.index_offset(indices), value
        )
    fn has_grad(self) -> Bool:
        return self.base_tensor[].has_grad()

    fn _requires_grad(self) -> Bool:
        return self.base_tensor[]._requires_grad()

    fn address(self) -> UnsafePointer[Self]:
        return UnsafePointer(to=self)

    fn id(self) -> Int:
        return Int(self.address())

    fn ancestry(self) -> Ancestors[dtype]:
        return Ancestors[dtype].untracked()

    fn seed_grad(self, value: Scalar[dtype]):
            self.base_tensor[].seed_grad(value)

    fn invoke_grad_fn(self, verbose: Bool = False) raises -> None:
        print("Will do it for sure!")
