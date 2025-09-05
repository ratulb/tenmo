from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import (
    AddTensor,
    SubtractTensor,
)


@fieldwise_init
@register_passable
struct TrueDivBackwardScalar[dtype: DType](Copyable):
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
@register_passable
struct RightTrueDivBackwardScalar[dtype: DType](Copyable):
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


@fieldwise_init
@register_passable
struct DivideBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]

        outgoing_grads = List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]()

        if ancestor_1.requires_grad():
            tensor_2 = ancestor_2.tensor()
            tensor_2.requires_grad_(False)
            tensor_2_reciprocal = 1 / tensor_2
            tensor_1_shape = ancestor_1.shape()
            tensor_1_grad = gradients * tensor_2_reciprocal
            if tensor_1_grad.shape != tensor_1_shape:
                tensor_1_grad = Tensor[dtype].sum_over_broadcasted_axes(
                    tensor_1_grad, tensor_1_shape
                )
            outgoing_grads.append((ancestor_1, tensor_1_grad, AddTensor))

        if ancestor_2.requires_grad():
            tensor_1 = ancestor_1.tensor()
            tensor_2 = ancestor_2.tensor()
            tensor_1.requires_grad_(False)
            tensor_2.requires_grad_(False)
            tensor_2_squared = tensor_2 * tensor_2
            tensor_2_squared_reciprocal = 1 / tensor_2_squared
            tensor_2_grad = tensor_1 * tensor_2_squared_reciprocal
            tensor_2_grad = gradients * tensor_2_grad
            if tensor_2_grad.shape != tensor_2.shape:
                tensor_2_grad = Tensor[dtype].sum_over_broadcasted_axes(
                    tensor_2_grad, tensor_2.shape
                )
            outgoing_grads.append((ancestor_2, tensor_2_grad, SubtractTensor))

        return outgoing_grads


fn main():
    pass
