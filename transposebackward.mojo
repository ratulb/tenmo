from tensors import Tensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList


@fieldwise_init
struct TBackward[dtype: DType](Copyable & Movable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.grad[]
        var grad_outputs: List[
            Tuple[TensorLike[dtype], Tensor[dtype], Int]
        ] = []
        ancestor = output.ancestors.get(0)[]
        return [
            (
                ancestor,
                gradients.T(),
                AddTensor,
            )
        ]


@fieldwise_init
struct TransposeBackward[dtype: DType](Copyable & Movable):
    var axes: IntList

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], TensorLike[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.grad[]
        var grad_outputs: List[
            Tuple[TensorLike[dtype], TensorLike[dtype], Int]
        ] = []
        ancestor = output.ancestors.get(0)[]
        inverted_axes = IntList.invert_permutation(self.axes)
        grad_transposed = gradients.transpose(inverted_axes)
        grad_output = TensorLike(UnsafePointer(to=grad_transposed))
        return [
            (
                ancestor,
                grad_output,
                AddTensor,
            )
        ]


fn main():
    pass
