from tensors import Tensor
from shared import TensorLite
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from operators import AddTensor


@fieldwise_init
@register_passable
struct VectorMatrixMMBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]

        gradients = output.gradients()[]
        grad_reshaped = gradients.reshape(1, -1)

        tensor_1 = ancestor_1.tensor()
        tensor_2 = ancestor_2.tensor()
        tensor_1_reshaped = tensor_1.reshape(1, -1)
        tensor_1_reshaped_transposed = tensor_1_reshaped.transpose(
            requires_grad=False
        )
        tensor_2_transposed = tensor_2.transpose(requires_grad=False)
        ancestor_1_reshaped_grad = Tensor[dtype].matmul_2d(
            UnsafePointer(to=grad_reshaped),
            UnsafePointer(to=tensor_2_transposed),
            track_grad=False,
        )

        ancestor_1_grad = ancestor_1_reshaped_grad.reshape(
            ancestor_1_reshaped_grad.shape[1], requires_grad=False
        )

        ancestor_2_grad = Tensor[dtype].matmul_2d(
            UnsafePointer(to=tensor_1_reshaped_transposed),
            UnsafePointer(to=grad_reshaped),
            track_grad=False,
        )

        var outgoing_grads: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []

        if ancestor_1.requires_grad():
            outgoing_grads.append(
                (
                    ancestor_1,
                    ancestor_1_grad,
                    AddTensor,
                )
            )
        if ancestor_2.requires_grad():
            outgoing_grads.append(
                (
                    ancestor_2,
                    ancestor_2_grad,
                    AddTensor,
                )
            )

        return outgoing_grads


fn main():
    pass
