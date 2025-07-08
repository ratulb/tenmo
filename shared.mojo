from tensors import Tensor
from views import TensorView
from memory import UnsafePointer
from ancestry import Ancestors


_="""trait Differentiable:
    fn id(self) -> Int:
        ...

    fn _requires_grad(self) -> Bool:
        ...

    fn ancestry[datatype: DType](self) -> Ancestors[datatype]:
        ...

    fn seed_grad[datatype: DType](self, value: Scalar[datatype]):
        ...

    fn into_tensorlike[datatype: DType](self) -> TensorLike[datatype]:
        ..."""


struct TensorLike[dtype: DType](Copyable & Movable):
    alias TensorAddress = UnsafePointer[Tensor[dtype]]
    alias ViewAddress = UnsafePointer[TensorView[dtype]]
    var kind: Int
    var tensor_address: Self.TensorAddress
    var view_address: Self.ViewAddress

    fn __init__(out self, _tensor_address: UnsafePointer[Tensor[dtype]]):
        self.kind = 0
        self.tensor_address = _tensor_address[].address()
        self.view_address = Self.ViewAddress()

    fn __init__(out self, _view_address: UnsafePointer[TensorView[dtype]]):
        self.kind = 1
        self.tensor_address = Self.TensorAddress()
        self.view_address = _view_address[].address()

    fn __moveinit__(out self, owned other: Self):
        self.kind = 1
        self.tensor_address = other.tensor_address
        self.view_address = other.view_address

    fn __copyinit__(out self, other: Self):
        self.kind = other.kind
        self.tensor_address = other.tensor_address
        self.view_address = other.view_address

    fn address(self) -> UnsafePointer[Self]:
        return UnsafePointer(to=self)

    fn is_view(self) -> Bool:
        return self.address()[].kind == 1

    fn is_tensor(self) -> Bool:
        return self.address()[].kind == 0

    fn tensor(self) -> Tensor[dtype]:
        return self.address()[].tensor_address[]

    fn view(self) -> TensorView[dtype]:
        return self.address()[].view_address[]

    fn id(self) -> Int:
        return (
            self.address()[]
            .tensor_address[]
            .id() if self.address()[]
            .is_tensor() else self.address()[]
            .view_address[]
            .id()
        )

    fn ancestry(self) -> Ancestors[dtype]:
        _ancestry = (
            self.address()[]
            .tensor()
            .ancestry() if self.address()[]
            .is_tensor() else self.address()[]
            .view()
            .ancestry()
        )
        print("TensorLike returning ancestry: ")
        _ancestry.print()
        print("============================")
        return _ancestry

    fn _requires_grad(self) -> Bool:
        if self.address()[].is_view():
            return self.address()[].view()._requires_grad()
        else:
            return self.address()[].tensor()._requires_grad()

    fn invoke_grad_fn(self, verbose: Bool = False) raises -> None:
        print("Are you invoking?")
        if self.address()[].is_view():
            self.address()[].view().invoke_grad_fn(verbose)
        else:
            self.address()[].tensor().invoke_grad_fn(verbose)


fn main():
    print("Starting of the begining!")
