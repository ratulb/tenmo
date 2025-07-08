from tensors import Tensor
from views import TensorView
from memory import UnsafePointer
from ancestry import Ancestors


_ = """trait Differentiable:
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


_ = """struct TensorLike[dtype: DType](Copyable & Movable):
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
        return UnsafePointer(to=self)[].kind == 1

    fn is_tensor(self) -> Bool:
        return UnsafePointer(to=self)[].kind == 0

    fn tensor_ptr(self) -> UnsafePointer[Tensor[dtype]]:
        return UnsafePointer(to=self)[].tensor_address

    fn view_ptr(self) -> UnsafePointer[TensorView[dtype]]:
        return UnsafePointer(to=self)[].view_address

    fn tensor(self) -> Tensor[dtype]:
        #return UnsafePointer(to=self)[].tensor_address[]
        return self.tensor_ptr()[]

    fn view(self) -> TensorView[dtype]:
        #return UnsafePointer(to=self)[].view_address[]
        return self.view_ptr()[]

    fn id(self) -> Int:
        return (
            UnsafePointer(to=self)[]
            .tensor_address[]
            .id() if UnsafePointer(to=self)[]
            .is_tensor() else UnsafePointer(to=self)[]
            .view_address[]
            .id()
        )

    fn ancestry(self) -> Ancestors[dtype]:
        _ancestry = (
            UnsafePointer(to=self)[]
            .tensor()
            .ancestry() if UnsafePointer(to=self)[]
            .is_tensor() else UnsafePointer(to=self)[]
            .view()
            .ancestry()
        )
        print("TensorLike returning ancestry: ")
        _ancestry.print()
        print("============================")
        return _ancestry

    fn _requires_grad(self) -> Bool:
        if UnsafePointer(to=self)[].is_view():
            return UnsafePointer(to=self)[].view()._requires_grad()
        else:
            return UnsafePointer(to=self)[].tensor()._requires_grad()

    fn invoke_grad_fn(self, verbose: Bool = False) raises -> None:
        print("Are you invoking?")
        if UnsafePointer(to=self)[].is_view():
            UnsafePointer(to=self)[].view().invoke_grad_fn(verbose)
        else:
            UnsafePointer(to=self)[].tensor().invoke_grad_fn(verbose)"""


struct TensorLike[dtype: DType](Copyable & Movable):
    alias TensorAddress = UnsafePointer[Tensor[dtype]]
    alias ViewAddress = UnsafePointer[TensorView[dtype]]

    var kind: Int
    var tensor_address: Self.TensorAddress
    var view_address: Self.ViewAddress

    fn __init__(out self, tensor_ptr: Self.TensorAddress):
        self.kind = 0
        self.tensor_address = tensor_ptr
        self.view_address = Self.ViewAddress()  # null

    fn __init__(out self, view_ptr: Self.ViewAddress):
        self.kind = 1
        self.tensor_address = Self.TensorAddress()  # null
        self.view_address = view_ptr

    fn __copyinit__(out self, other: Self):
        self.kind = other.kind
        self.tensor_address = other.tensor_address
        self.view_address = other.view_address

    fn __moveinit__(out self, owned other: Self):
        self.kind = other.kind
        self.tensor_address = other.tensor_address
        self.view_address = other.view_address

    fn address(self) -> UnsafePointer[Self]:
        return UnsafePointer(to=self)

    fn is_view(self) -> Bool:
        return self.kind == 1

    fn is_tensor(self) -> Bool:
        return self.kind == 0

    fn tensor_ptr(self) -> Self.TensorAddress:
        return self.tensor_address

    fn view_ptr(self) -> Self.ViewAddress:
        return self.view_address

    fn tensor(self) -> Tensor[dtype]:
        return self.tensor_address[]

    fn view(self) -> TensorView[dtype]:
        return self.view_address[]

    _ = """fn id(self) -> Int:
        return self.tensor().id() if self.is_tensor() else self.view().id()"""

    fn id(self) -> Int:
        if self.kind == 0:
            return self.tensor_address[].id()
        else:
            return self.view_address[].id()

    fn ancestry(self) -> Ancestors[dtype]:
        ancestry = (
            self.tensor_address[].ancestry() if self.kind
            == 0 else self.view_address[].ancestry()
        )
        print("TensorLike returning ancestry: ")
        ancestry.print()
        print("============================")
        return ancestry

    fn _requires_grad(self) -> Bool:
        return (
            self.view()
            ._requires_grad() if self.is_view() else self.tensor()
            ._requires_grad()
        )

    fn invoke_grad_fn(self, verbose: Bool = False) raises:
        if self.is_view():
            self.view().invoke_grad_fn(verbose)
        else:
            self.tensor().invoke_grad_fn(verbose)


fn main():
    print("Starting of the begining!")
