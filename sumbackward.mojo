from tensors import Tensor
from intlist import IntList
from operators import AddTensor
from shared import TensorLike
from shapes import Shape
from backpropagation import Delegate, BackwardFn


struct SumBackward[dtype: DType](Copyable & Movable):
    var axes: IntList
    var keepdims: Bool

    fn __init__(out self, axes: IntList=IntList.Empty, keepdims: Bool = False):
        self.axes = axes
        self.keepdims = keepdims

    fn __moveinit__(out self, var other: Self):
        self.axes = other.axes
        self.keepdims = other.keepdims

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes
        self.keepdims = other.keepdims

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.grad[]
        ancestor = output.ancestors.get(0)[]
        rank = ancestor.rank()
        if rank == 0:
            return [(ancestor, gradients, AddTensor)]
        shape = ancestor.shape()

        var grad_contrib: Tensor[dtype]

        # Handle scalar gradient case (sum reduced to scalar)
        if gradients.shape == Shape.Void:
            grad_contrib = Tensor[dtype].full(
                shape,
                gradients.item(),
                requires_grad=False,
            )
        else:
            # Handle keepdims=False case (need to reshape gradient)
            if not self.keepdims:
                # Determine axes/unsqueeze (insert dims of size 1)
                axes = gradients.shape.intlist().insert(
                    self.axes,
                    IntList.with_capacity(len(self.axes), 1),
                )
                unsqueezed_shape = Shape(axes)

                unsqueezed_grad = gradients.reshape(unsqueezed_shape)
                grad_contrib = unsqueezed_grad.broadcast_to(shape)
            else:
                # keepdims=True: shapes match except for broadcasting
                grad_contrib = gradients.broadcast_to(shape)
        grad_contrib.requires_grad = False
        return [
            (
                ancestor,
                grad_contrib,
                AddTensor,
            )
        ]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


fn main():
    print("passes")
