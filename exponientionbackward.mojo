from tensors import Tensor
from shared import TensorLike
from backpropagation import BackwardFn, Delegate
from operators import (
    __tensor_op_tensor__,
    AddTensor,
    SubtractTensor,
    MulTensor,
    __tensor_op_scalar__,
    MulScalar,
    Power,
)


@fieldwise_init
struct ExponientionBackward[dtype: DType](Copyable & Movable):
    var exponent: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[TensorLike[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.gradients()[]
        var exponent: Scalar[dtype] = rebind[Scalar[dtype]](self.exponent)
        ancestor = output.ancestry().get(0)[]

        # ∂(x**n)/∂x = n * x**(n-1)
        # Need to see if base_pow gets a grad_fn or not - we don't want it to have one!
        # var base_pow = self ** (scalar - 1.0)
        base_pow = __tensor_op_scalar__[dtype, Power](
            ancestor.tensor(), (exponent - 1.0)
        )
        base_pow.requires_grad = False
        var local_grad = __tensor_op_scalar__[dtype, MulScalar](
            base_pow, exponent
        )
        product = __tensor_op_tensor__[dtype, MulTensor](gradients, local_grad)
        return [
            (
                ancestor,
                product,
                AddTensor,
            )
        ]


fn main():
    pass
