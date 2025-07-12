from tensors import Tensor
from views import TensorView
from memory import UnsafePointer
from ancestry import Ancestors
from utils import Variant

fn main():
    print("Starting of the begining!")
    tensor = Tensor.scalar(1)
    _p = Possible[Tensor](tensor.address())



trait Differentiable:
    alias datatype: DType

    fn id(self) -> Int:
        ...

    fn into_tensor(self) -> Tensor[datatype]:
        ...
    fn is_tensor(self) -> Bool:
        ...
    fn is_view(self) -> Bool:
        ...

    fn _requires_grad(self) -> Bool:
        ...

    fn into_tensorlike(self) -> TensorLike[datatype]:
        ...
    fn seed_grad(self, value: Scalar[datatype]):
        ...


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

    fn id(self) -> Int:
        return Int(self.address())

    fn inner_id(self) -> Int:
        if self.kind == 0:
            return self.tensor_address[].id()
        else:
            return self.view_address[].id()

    fn ancestry(self) -> Ancestors[dtype]:
        return (
            self.tensor_address[].ancestry() if self.kind
            == 0 else self.view_address[].ancestry()
        )

    fn requires_grad(self) -> Bool:
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

_="""struct TensorLike[dtype: DType](Copyable & Movable):
    alias TensorAddress = UnsafePointer[Tensor[dtype]]
    alias ViewAddress = UnsafePointer[TensorView[dtype]]

    alias Inner = Variant[Self.TensorAddress, Self.ViewAddress]
    var pointee: Self.Inner

    fn __init__(out self, tensor_addr: Self.TensorAddress):
        self.pointee = Self.Inner(tensor_addr)

    fn __init__(out self, tensor_view_addr: Self.ViewAddress):
        self.pointee = Self.Inner(tensor_view_addr)

    fn __copyinit__(out self, other: Self):
        self.pointee = other.pointee

    fn __moveinit__(out self, owned other: Self):
        self.pointee = other.pointee

    fn id(self) -> Int:
        if self.is_view():
            return self.view().id()
        else:
            return self.tensor().id()

    fn ancestry(self) -> Ancestors[dtype]:
        is_it_a_tensor = self.is_tensor()
        print("is_it_a_tensor: ", is_it_a_tensor)
        var ancestors: Ancestors[dtype]
        if is_it_a_tensor:
            tensor = self.tensor()
            tensor.print()
            ancestors = tensor.ancestry()
            ancestors.print()
            print("Got tensor ancestry")
            # return self.tensor().ancestry()
        else:
            ancestors = self.view().ancestry()
            print("Got view ancestry")
            # return self.view().ancestry()
        return ancestors

    fn is_tensor(self) -> Bool:
        return self.pointee.isa[Self.TensorAddress]()

    fn is_view(self) -> Bool:
        return self.pointee.isa[Self.ViewAddress]()

    fn tensor(self) -> Tensor[dtype]:
        return self.pointee[Self.TensorAddress][]

    fn view(self) -> TensorView[dtype]:
        return self.pointee[Self.ViewAddress][]

    fn address(self) -> UnsafePointer[Self]:
        return UnsafePointer(to=self)

    fn requires_grad(self) -> Bool:
        if self.is_view():
            return self.view().base_tensor[].requires_grad
        else:
            return self.tensor().requires_grad

    fn invoke_grad_fn(self, verbose: Bool = False) raises -> None:
        if self.is_view():
            self.view().base_tensor[].invoke_grad_fn(verbose)
        else:
            self.tensor().invoke_grad_fn(verbose)"""

struct Possible[dtype: DType, //, T: Differentiable](Copyable & Movable):
    alias Address = UnsafePointer[T]
    alias TensorAddress = UnsafePointer[Tensor[dtype]]
    alias ViewAddress = UnsafePointer[TensorView[dtype]]
    var address: Self.Address

    fn __init__(out self, tensor_addr: Self.TensorAddress):
        self.address = rebind[Self.Address](tensor_addr)
        print("Do a reconstruction")
        self.address[].into_tensor().print()

