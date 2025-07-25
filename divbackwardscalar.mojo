from tensors import Tensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn
from operators import (
    __tensor_op_scalar__,
    AddTensor,
    DivideScalar,
    SubtractTensor,
)


@fieldwise_init
struct TrueDivBackwardScalar[dtype: DType](Copyable & Movable):
    var factor: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[TensorLike[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
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
    ](self, out_ptr: UnsafePointer[TensorLike[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.gradients()[]
        var scalar: Scalar[dtype] = rebind[Scalar[dtype]](self.scalar)
        ancestor = output.ancestry().get(0)[]
        squared = ancestor.tensor().__pow__(2)
        squared_reciprocal = __tensor_op_scalar__[dtype, DivideScalar](
            squared, 1.0
        )
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
