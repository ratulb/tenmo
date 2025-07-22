from tensors import Tensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn
from operators import AddTensor, MulTensor, __tensor_op_tensor__


@fieldwise_init
struct MultiplyBackward[dtype: DType](Copyable & Movable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.grad[]
        var grad_outputs: List[
            Tuple[TensorLike[dtype], Tensor[dtype], Int]
        ] = []
        ancestor_1 = output.ancestors.get(0)[]
        ancestor_2 = output.ancestors.get(1)[]

        if ancestor_1.requires_grad():
            product = __tensor_op_tensor__[dtype, MulTensor](
                gradients, ancestor_2.tensor()
            )

            product.requires_grad = False
            grad_outputs.append(
                (
                    ancestor_1,
                    product,
                    AddTensor,
                )
            )

        if ancestor_2.requires_grad():
            product = __tensor_op_tensor__[dtype, MulTensor](
                gradients, ancestor_1.tensor()
            )
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
