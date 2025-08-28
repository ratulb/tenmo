from tensors import Tensor
from operators import AddTensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn


@fieldwise_init
@register_passable
struct AddBackwardScalar[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        if ancestor.shape() != gradients.shape:
            gradients = gradients.reshape(ancestor.shape())
        # Gradient of addition is 1 → just pass through incoming grad
        return [(ancestor, gradients, AddTensor)]


@fieldwise_init
@register_passable
struct AddBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        count = len(output.ancestry())
        grad_outputs = List[Tuple[TensorLite[dtype], Tensor[dtype], Int]](
            capacity=count
        )
        for i in range(count):
            ancestor = output.ancestry().get(i)[]
            grad_outputs.append((ancestor, gradients, AddTensor))
        return grad_outputs


fn main():
    print("passes")
