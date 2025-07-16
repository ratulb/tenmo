from tensors import Tensor
from shapes import Shape
from intlist import IntList
from strides import Strides
from os import abort
from shared import TensorLike
fn main():
    pass

struct TensorView[dtype: DType = DType.float32](
    Copyable & Movable
):
    alias Blank: TensorView[dtype] = Self(
        UnsafePointer[Tensor[dtype]](), Shape.Void, Strides(IntList.Empty), 0
    )
    var base_tensor: UnsafePointer[Tensor[dtype]]
    var shape: Shape
    var strides: Strides
    var offset: Int

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

    _="""fn backward(self, start_grad: Scalar[dtype] = 1.0):
        graph = Graph[dtype]()
        graph.walk_backward(self.into_tensorlike(), start_grad)

    fn backward(self, with_tensor: Tensor[dtype]):
        graph = Graph[dtype]()
        graph.walk_backward(self.into_tensorlike(), with_tensor)"""

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

    fn __getitem__(self, *indices: Int) -> Scalar[dtype]:
        return self.base_tensor[].data.load[volatile=True](
            self.index_offset(IntList(indices))
        )

    fn __setitem__(self, *indices: Int, value: Scalar[dtype]):
        self.base_tensor[].data.store[volatile=True](
            self.index_offset(IntList(indices)), value
        )

    fn has_grad(self) -> Bool:
        return self.base_tensor[].has_grad()

    fn _requires_grad(self) -> Bool:
        return self.base_tensor[]._requires_grad()

    fn address(self) -> UnsafePointer[Self]:
        return UnsafePointer(to=self)

    fn id(self) -> Int:
        return Int(self.address())

    fn is_view(self) -> Bool:
        return True

    fn is_tensor(self) -> Bool:
        return False

    fn into_view(self) -> TensorView[dtype]:
        return self

    fn into_tensor(self) -> Tensor[dtype]:
        abort("TensorView -> into_tensor(self) - not supported")
        return Tensor[dtype]([])

    fn seed_grad(self, value: Scalar[dtype]):
        self.base_tensor[].seed_grad(value)

    fn seed_grad(self, with_tensor: Tensor[dtype]):
        self.base_tensor[].seed_grad(with_tensor)

    fn invoke_grad_fn(self, verbose: Bool = False) raises -> None:
        print("Will do it for sure!")
