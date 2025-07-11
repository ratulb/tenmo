from memory import UnsafePointer
from tensors import Tensor
from shapes import Shape
from intlist import IntList
from operators import AddTensor
from common_utils import is_null


fn main():
    pass


struct SumGradFn[dtype: DType](Copyable & Movable):
    var input: Tensor[dtype]
    var output: Tensor[dtype]
    var keepdims: Bool
    var axes: IntList

    fn __init__(
        out self,
        input: Tensor[dtype],
        output: Tensor[dtype],
        keepdims: Bool,
        axes: IntList,
    ):
        self.input = input
        self.output = output
        self.keepdims = keepdims
        self.axes = axes
        print("I have been created")


    fn __moveinit__(out self, owned other: Self):
        self.input = other.input
        self.output = other.output
        self.keepdims = other.keepdims
        self.axes = other.axes


    fn __copyinit__(out self, other: Self):
        self.input = other.input
        self.output = other.output
        self.keepdims = other.keepdims
        self.axes = other.axes


    fn __call__(self):
        print("You are reaching here alright")
        print("Inside call check 1", is_null(self.output.address()), is_null(self.input.address()))
        var result = self.output
        print("Inside call check 2", result.shape)
        #dependencies = result.dependencies[]
        #print("Inside call check 3", dependencies.count)
        #var this_pointer = dependencies.lookup(1)
        print("Inside call check 4")
        this = self.input

        print("Inside call check 5")
        var outstream_grad = result.grad[]
        var original_shape = this.shape
        var grad_contrib: Tensor[dtype]

        if outstream_grad.shape == Shape.Void:
            grad_contrib = Tensor[dtype].full(
                original_shape, outstream_grad.item(), requires_grad=False
            )
        else:
            if not self.keepdims:
                var inserted_axes = outstream_grad.shape.intlist().insert(
                    self.axes, IntList.with_capacity(len(self.axes), 1)
                )
                var unsqueezed_shape = Shape(inserted_axes)

                var unsqueezed_grad = outstream_grad.reshape(unsqueezed_shape)
                grad_contrib = unsqueezed_grad.broadcast_to(original_shape)
            else:
                grad_contrib = outstream_grad.broadcast_to(original_shape)

        grad_contrib.requires_grad = False
        this.update_grad[AddTensor](grad_contrib)


_="""struct SumGradFn[dtype: DType](Copyable & Movable):
    alias Address = UnsafePointer[Tensor[dtype]]
    var in_addr: Self.Address
    var out_addr: Self.Address
    var keepdims: Bool
    var axes: IntList

    fn __init__(
        out self,
        in_addr: Self.Address,
        out_addr: Self.Address,
        keepdims: Bool,
        axes: IntList,
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
        print("Inside call check 1", is_null(self.out_addr), is_null(self.in_addr))
        var result = self.out_addr[]
        print("Inside call check 2", result.shape)
        dependencies = result.dependencies[]
        print("Inside call check 3", dependencies.count)
        var this_pointer = dependencies.lookup(1)
        print("Inside call check 4")
        this = this_pointer[]

        print("Inside call check 5")
        var outstream_grad = result.grad[]
        var original_shape = this.shape
        var grad_contrib: Tensor[dtype]

        if outstream_grad.shape == Shape.Void:
            grad_contrib = Tensor[dtype].full(
                original_shape, outstream_grad.item(), requires_grad=False
            )
        else:
            if not self.keepdims:
                var inserted_axes = outstream_grad.shape.intlist().insert(
                    self.axes, IntList.with_capacity(len(self.axes), 1)
                )
                var unsqueezed_shape = Shape(inserted_axes)

                var unsqueezed_grad = outstream_grad.reshape(unsqueezed_shape)
                grad_contrib = unsqueezed_grad.broadcast_to(original_shape)
            else:
                grad_contrib = outstream_grad.broadcast_to(original_shape)

        grad_contrib.requires_grad = False
        this.update_grad[AddTensor](grad_contrib)"""


