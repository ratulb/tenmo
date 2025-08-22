from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import (
    AddTensor,
)


@fieldwise_init
struct MulBackwardScalar[dtype: DType](Copyable & Movable):
    var factor: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        var value: Scalar[dtype] = rebind[Scalar[dtype]](self.factor)
        ancestor = output.ancestry().get(0)[]
        scaled_gradients = gradients * value
        return [
            (
                ancestor,
                scaled_gradients,
                AddTensor,
            )
        ]


@fieldwise_init
struct MultiplyBackward[dtype: DType](Copyable & Movable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        var grad_outputs: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]

        if ancestor_1.requires_grad():
            product = gradients * ancestor_2.tensor()

            product.requires_grad = False
            grad_outputs.append(
                (
                    ancestor_1,
                    product,
                    AddTensor,
                )
            )

        if ancestor_2.requires_grad():
            product = gradients * ancestor_1.tensor()
            product.requires_grad = False
            grad_outputs.append(
                (
                    ancestor_2,
                    product,
                    AddTensor,
                )
            )

        return grad_outputs


fn main():
    pass
