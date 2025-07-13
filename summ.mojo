from memory import UnsafePointer
from tensors import Tensor
from shapes import Shape
from intlist import IntList
from operators import AddTensor
from common_utils import is_null


fn main():
    pass


struct SumGradFn[dtype: DType](Copyable & Movable):
    alias Address = UnsafePointer[Tensor[dtype]]
    alias Axes = UnsafePointer[IntList]
    var in_addr: Self.Address
    var out_addr: Self.Address
    var keepdims: Bool
    var axes: Self.Axes

    fn __init__(
        out self,
        in_addr: Self.Address,
        out_addr: Self.Address,
        keepdims: Bool,
        axes: Self.Axes,
    ):
        self.in_addr = in_addr
        self.out_addr = out_addr
        self.keepdims = keepdims
        self.axes = axes
        print("I have been created")

    fn __moveinit__(out self, owned other: Self):
        self.in_addr = other.in_addr
        self.out_addr = other.out_addr
        self.keepdims = other.keepdims
        self.axes = other.axes

    fn __copyinit__(out self, other: Self):
        self.in_addr = other.in_addr
        self.out_addr = other.out_addr
        self.keepdims = other.keepdims
        self.axes = other.axes

    fn __call__(self):
        print("You are reaching here alright")
        print(
            "Inside call check 1", is_null(self.out_addr), is_null(self.in_addr)
        )
        var result = self.out_addr[]
        print("Inside call check 2", result.shape)
        this = self.in_addr[]

        print("Inside call check 5")
        var outstream_grad = result.grad[]
        var original_shape = this.shape
        var grad_contrib: Tensor[dtype]
        var axes = self.axes[]
        if outstream_grad.shape == Shape.Void:
            grad_contrib = Tensor[dtype].full(
                original_shape, outstream_grad.item(), requires_grad=False
            )
        else:
            if not self.keepdims:
                var inserted_axes = outstream_grad.shape.intlist().insert(
                    # self.axes, IntList.with_capacity(len(self.axes), 1)
                    axes,
                    IntList.with_capacity(len(axes), 1),
                )
                var unsqueezed_shape = Shape(inserted_axes)

                var unsqueezed_grad = outstream_grad.reshape(unsqueezed_shape)
                grad_contrib = unsqueezed_grad.broadcast_to(original_shape)
            else:
                grad_contrib = outstream_grad.broadcast_to(original_shape)

        grad_contrib.requires_grad = False
        this.update_grad[AddTensor](grad_contrib)
