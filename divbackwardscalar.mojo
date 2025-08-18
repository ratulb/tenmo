from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import (
    AddTensor,
    SubtractTensor,
)


@fieldwise_init
struct TrueDivBackwardScalar[dtype: DType](Copyable & Movable):
    var factor: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        var divisor: Scalar[dtype] = rebind[Scalar[dtype]](self.factor)
        ancestor = output.ancestry().get(0)[]
        # ∂(x / s)/∂x = 1/s → incoming_grad / scalar
        var divided = gradients / divisor
        return [
            (
                ancestor,
                divided,
                AddTensor,
            )
        ]


@fieldwise_init
struct RightTrueDivBackwardScalar[dtype: DType](Copyable & Movable):
    var scalar: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        var scalar: Scalar[dtype] = rebind[Scalar[dtype]](self.scalar)
        ancestor = output.ancestry().get(0)[]
        squared = ancestor.tensor().__pow__(2)
        squared_reciprocal = 1.0 / squared
        grad = (gradients * scalar) * squared_reciprocal

        return [
            (
                ancestor,
                grad,
                SubtractTensor,
            )
        ]


fn main():
    pass
