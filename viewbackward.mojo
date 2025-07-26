from tensors import Tensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList


@fieldwise_init
@register_passable
struct ViewBackward[dtype: DType](Copyable & Movable):
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
                gradients.reshape(ancestor.shape()),
                AddTensor,
            )
        ]

fn main():
    pass
