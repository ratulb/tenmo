from tensors import Tensor
from operators import AddTensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn


@fieldwise_init
@register_passable
struct ExpandBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        print("\nWhat did I receive here?\n")
        gradients.print()
        ancestor = output.ancestry().get(0)[]
        recipient_shape = ancestor.shape()
        reduced_grad = Tensor[dtype].sum_over_broadcasted_axes(
            gradients, recipient_shape
        )

        return [(ancestor, reduced_grad, AddTensor)]


fn main():
    print("passes")
