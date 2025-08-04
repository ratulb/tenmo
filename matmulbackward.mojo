from tensors import Tensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn
from operators import AddTensor


@fieldwise_init
struct MatmulBackward[dtype: DType](Copyable & Movable & Stringable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[TensorLike[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.gradients()[]
        var grad_outputs: List[
            Tuple[TensorLike[dtype], Tensor[dtype], Int]
        ] = []
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]
        if ancestor_1.requires_grad():
            ancestor_2_transposed = (
                ancestor_2.tensor() if ancestor_2.is_tensor() else ancestor_2.view().into_tensor()
            ).transpose()
            ancestor_1_grad_share = gradients.matmul(ancestor_2_transposed)
            ancestor_1_grad_share.requires_grad = False
            grad_outputs.append(
                (
                    ancestor_1,
                    ancestor_1_grad_share,
                    AddTensor,
                )
            )

        if ancestor_2.requires_grad():
            ancestor_1_transposed = (
                ancestor_1.tensor() if ancestor_1.is_tensor() else ancestor_1.view().into_tensor()
            ).transpose()
            ancestor_2_grad_share = ancestor_1_transposed.matmul(gradients)
            ancestor_2_grad_share.requires_grad = False
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
