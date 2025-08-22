from tensors import Tensor
from shared import TensorLite
from operators import AddTensor, SubtractTensor
from backpropagation import BackwardFn, Delegate


@fieldwise_init
struct ReshapeBackward[dtype: DType](Copyable & Movable & Stringable):
    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        #print("ReshapeBackward -> gradients")
        #gradients.print()
        ancestor = output.ancestry().get(0)[]
        reshaped = gradients.reshape(ancestor.shape())

        #print("ReshapeBackward -> reshaped")
        #reshaped.print()

        return [(ancestor, reshaped, AddTensor), (output, gradients, SubtractTensor)]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn __str__(self) -> String:
        return "ReshapeBackward"

fn main():
    print("Yes")
