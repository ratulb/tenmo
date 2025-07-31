from tensors import Tensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList


@fieldwise_init
@register_passable
struct TBackward[dtype: DType](Copyable & Movable & Stringable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[TensorLike[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        return [
            (
                ancestor,
                gradients.transpose().into_tensor(),
                AddTensor,
            )
        ]

    fn __str__(self) -> String:
        return "TBackward"


@fieldwise_init
@register_passable
struct TransposeBackward[dtype: DType](Copyable & Movable & Stringable):
    var axes: IntList

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[TensorLike[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        inverted_axes = IntList.invert_permutation(self.axes)
        grad_transposed = gradients.transpose(inverted_axes)
        grad_output = grad_transposed.into_tensor()
        return [
            (
                ancestor,
                grad_output,
                AddTensor,
            )
        ]

    fn __str__(self) -> String:
        return "TransposeBackward"


fn main():
    pass
