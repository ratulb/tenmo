from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import (
    AddTensor,
)


@fieldwise_init
@register_passable
struct DotBackward[dtype: DType](Copyable & Movable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[].item()  # Scalar
        var grad_outputs: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]

        if ancestor_1.requires_grad():
            tensor = ancestor_2.tensor()
            buffer = tensor.buffer if tensor.owns_data else tensor.contiguous(
                requires_grad=False
            ).buffer
            buffer = gradients * buffer
            outgoing = Tensor[dtype](ancestor_1.shape(), buffer, False)
            grad_outputs.append(
                (
                    ancestor_1,
                    outgoing^,
                    AddTensor,
                )
            )

        if ancestor_2.requires_grad():
            tensor = ancestor_1.tensor()
            buffer = tensor.buffer if tensor.owns_data else tensor.contiguous(
                requires_grad=False
            ).buffer
            buffer = gradients * buffer
            outgoing = Tensor[dtype](ancestor_2.shape(), buffer, False)

            grad_outputs.append(
                (
                    ancestor_2,
                    outgoing^,
                    AddTensor,
                )
            )

        return grad_outputs


fn main():
    pass
