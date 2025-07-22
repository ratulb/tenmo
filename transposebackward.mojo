from tensors import Tensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn
from operators import AddTensor


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


fn main():
    pass
