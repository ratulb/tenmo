from tensors import Tensor
from shared import TensorLite
from operators import AddTensor, ZeroGrad
from backpropagation import BackwardFn, Delegate


@fieldwise_init
@register_passable
struct ReshapeBackward[dtype: DType](Copyable):
    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        reshaped = gradients.reshape(ancestor.shape())

        return [
            (ancestor, reshaped, AddTensor),
            (output, gradients, ZeroGrad),
        ]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

fn main():
    print("Yes")
