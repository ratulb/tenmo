from tensors import Tensor
from views import TensorView
from utils import Variant
from memory import UnsafePointer
from ancestry import Ancestors


trait Differentiable:
    alias datatype: DType

    fn id(self) -> Int:
        ...

    fn _requires_grad(self) -> Bool:
        ...

    fn ancestry(self) -> Ancestors[datatype]:
        ...

    fn seed_grad(self, value: Scalar[datatype]):
        ...
    fn into_tensorlike(self) -> TensorLike[datatype]:
        ...

# struct TensorLike[dtype: DType](Copyable & Movable):
@register_passable
struct TensorLike[dtype: DType](Copyable):
    alias TensorAddress = UnsafePointer[Tensor[dtype]]
    alias ViewAddress = UnsafePointer[TensorView[dtype]]
    var kind: Int
    var tensor_address: Self.TensorAddress
    var view_address: Self.ViewAddress

    fn __init__(out self, _tensor_address: UnsafePointer[Tensor[dtype]]):
        self.kind = 0
        self.tensor_address = _tensor_address
        self.view_address = Self.ViewAddress()

    fn __init__(out self, _view_address: UnsafePointer[TensorView[dtype]]):
        self.kind = 1
        self.tensor_address = Self.TensorAddress()
        self.view_address = _view_address

    _ = """fn __moveinit__(out self, owned other: Self):
        self.pointee = other.pointee"""

    fn __copyinit__(out self, other: Self):
        self.kind = other.kind
        self.tensor_address = other.tensor_address
        self.view_address = other.view_address

    fn is_view(self) -> Bool:
        return self.kind == 1

    fn is_tensor(self) -> Bool:
        return self.kind == 0

    fn tensor(self) -> Tensor[dtype]:
        return self.tensor_address[]

    fn view(self) -> TensorView[dtype]:
        return self.view_address[]

    fn address(self) -> UnsafePointer[Self]:
        return UnsafePointer(to=self)

    fn id(self) -> Int:
        return self.tensor().id() if self.is_tensor() else self.view().id()

    fn ancestry(self) -> Ancestors[dtype]:
        return self.tensor().ancestry() if self.is_tensor() else self.view().ancestry()


    fn _requires_grad(self) -> Bool:
        if self.is_view():
            return self.view()._requires_grad()
        else:
            return self.tensor()._requires_grad()

    fn invoke_grad_fn(self, verbose: Bool = False) raises -> None:
        if self.is_view():
            self.view().invoke_grad_fn(verbose)
        else:
            self.tensor().invoke_grad_fn(verbose)


fn main():
    print("Starting of the begining!")
