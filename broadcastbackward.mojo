from tensors import Tensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn


@fieldwise_init
struct BroadcastBackward[
    dtype: DType, Tensor_Op_First: Int, Tensor_Op_Second: Int, Multiply: Bool
](Copyable & Movable & Stringable):
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
        ancestors = output.ancestry()
        ancestor_1 = ancestors.get(0)[]
        ancestor_2 = ancestors.get(1)[]

        if ancestor_1.requires_grad():
            ancestor_1_share = ancestor_1.tensor().backward_contribution(
                ancestor_2.tensor(), gradients, Multiply
            )
            grad_outputs.append(
                (
                    ancestor_1,
                    ancestor_1_share,
                    Tensor_Op_First,
                )
            )

        if ancestor_2.requires_grad():
            ancestor_2_share = ancestor_2.tensor().backward_contribution(
                ancestor_1.tensor(), gradients, Multiply
            )
            grad_outputs.append(
                (
                    ancestor_2,
                    ancestor_2_share,
                    Tensor_Op_Second,
                )
            )
        return grad_outputs

    fn __str__(self) -> String:
        return "BroadcastBackward"


fn main():
    pass
