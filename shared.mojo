from tensors import Tensor
from shapes import Shape
from views import TensorView


fn main():
    print("Starting of the begining!")


struct Ancestor[dtype: DType](Copyable & Movable):
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

    fn has_grad(self) -> Bool:
        return (
            self.tensor_address[]
            .has_grad() if self.is_tensor() else self.view_address[]
            .base_tensor[]
            .has_grad()
        )

    fn has_grad_fn(self) -> Bool:
        return (
            self.tensor_address[]
            .has_grad_fn() if self.is_tensor() else self.view_address[]
            .base_tensor[]
            .has_grad_fn()
        )

    fn grad_fn(self) -> UnsafePointer[Tensor[dtype].BackwardFn]:
        return (
            self.tensor_address[]
            .backward_fn() if self.is_tensor() else self.view_address[]
            .base_tensor[]
            .backward_fn()
        )

    fn gradients(self) -> UnsafePointer[Tensor[dtype]]:
        return (
            self.tensor_address[]
            .gradients() if self.is_tensor() else self.view_address[]
            .base_tensor[]
            .gradients()
        )

    fn id(self) -> Int:
        return Int(self.address())

    fn inner_id(self) -> Int:
        if self.kind == 0:
            return self.tensor_address[].id()
        else:
            return self.view_address[].id()

    fn shape(self) -> Shape:
        if self.kind == 0:
            return self.tensor_address[].shape
        else:
            return self.view_address[].shape

    fn seed_grad(self, value: Scalar[dtype]):
        if self.is_tensor():
            self.tensor().seed_grad(value)
        else:
            self.view().seed_grad(value)

    fn update_grad[opcode: Int](self, incoming: Tensor[dtype]):
        if self.is_tensor():
            self.tensor().update_grad[opcode](incoming)
        else:
            self.view().base_tensor[].update_grad[opcode](incoming)

    fn seed_grad(self, with_tensor: Tensor[dtype]):
        if self.is_tensor():
            self.tensor().seed_grad(with_tensor)
        else:
            self.view().seed_grad(with_tensor)

    fn requires_grad(self) -> Bool:
        return (
            self.view()
            ._requires_grad() if self.is_view() else self.tensor()
            ._requires_grad()
        )

