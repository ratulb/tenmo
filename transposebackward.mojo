from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList


@fieldwise_init
@register_passable
struct TransposeBackward[dtype: DType](Copyable & Movable & Stringable):
    var axes: IntList

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        inverted_axes = IntList.invert_permutation(self.axes)
        grad_transposed = gradients.transpose(inverted_axes)
        return [
            (
                ancestor,
                grad_transposed,
                AddTensor,
            )
        ]

    fn __str__(self) -> String:
        return "TransposeBackward"


fn main():
    pass
