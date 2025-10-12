from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn


@fieldwise_init
@register_passable
struct BroadcastBackward[
    dtype: DType, Tensor_Op_First: Int, Tensor_Op_Second: Int, Multiply: Bool
](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.grad()
        var grad_outputs: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []
        ancestors = output.ancestry()
        ancestor_1 = ancestors.get(0)
        ancestor_2 = ancestors.get(1)

        if ancestor_1.requires_grad():
            temp_tensor_1 = ancestor_1.tensor()
            temp_tensor_2 = ancestor_2.tensor()
            ancestor_1_share = temp_tensor_1.backward_contribution(
                temp_tensor_2, gradients, Multiply
            )
            grad_outputs.append(
                (
                    ancestor_1,
                    ancestor_1_share^,
                    Tensor_Op_First,
                )
            )
            temp_tensor_1.free()
            temp_tensor_2.free()

        if ancestor_2.requires_grad():
            temp_tensor_2 =ancestor_2.tensor()
            temp_tensor_1 = ancestor_1.tensor()
            ancestor_2_share = temp_tensor_2.backward_contribution(
                temp_tensor_1, gradients, Multiply
            )
            grad_outputs.append(
                (
                    ancestor_2,
                    ancestor_2_share^,
                    Tensor_Op_Second,
                )
            )
            temp_tensor_2.free()
            temp_tensor_1.free()
        gradients.free()
        return grad_outputs


fn main():
    pass
