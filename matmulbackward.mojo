from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import AddTensor


@fieldwise_init
struct MatmulBackward[dtype: DType](Copyable & Movable & Stringable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        print("gradients")
        gradients.print()
        var grad_outputs: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]
        if ancestor_1.requires_grad():
            ancestor_2_tensor = ancestor_2.tensor()
            ancestor_2_transposed = ancestor_2_tensor.transpose(
                requires_grad=False
            ).contiguous()
            ancestor_1_grad_share = gradients.matmul(ancestor_2_transposed)
            grad_outputs.append(
                (
                    ancestor_1,
                    ancestor_1_grad_share,
                    AddTensor,
                )
            )

        if ancestor_2.requires_grad():
            ancestor_1_tensor = ancestor_1.tensor()
            ancestor_1_transposed = ancestor_1_tensor.transpose(
                requires_grad=False
            ).contiguous()
            ancestor_2_grad_share = ancestor_1_transposed.matmul(gradients)
            grad_outputs.append(
                (
                    ancestor_2,
                    ancestor_2_grad_share,
                    AddTensor,
                )
            )

        return grad_outputs

    fn __str__(self) -> String:
        return "MatmulBackward"


fn main():
    pass
