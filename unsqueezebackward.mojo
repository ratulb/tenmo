from tensors import Tensor
from operators import AddTensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn


@fieldwise_init
@register_passable
struct UnsqueezeBackward[dtype: DType](Copyable & Movable):
    var axis: Int  # where axis was inserted

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        # Remove the axis we had inserted
        gradients_squeezed = gradients.squeeze(self.axis)
        ancestor = output.ancestry().get(0)[]
        return [(ancestor, gradients_squeezed, AddTensor)]


fn main():
    print("passes")
