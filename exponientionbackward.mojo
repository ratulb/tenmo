from tensors import Tensor
from shared import TensorLite
from backpropagation import BackwardFn, Delegate
from operators import AddTensor


@fieldwise_init
struct ExponientionBackward[dtype: DType](Copyable & Movable & Stringable):
    var exponent: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        var exponent: Scalar[dtype] = rebind[Scalar[dtype]](self.exponent)
        ancestor = output.ancestry().get(0)[]

        # ∂(x**n)/∂x = n * x**(n-1)
        # Need to see if base_pow gets a grad_fn or not - we don't want it to have one!
        # var base_pow = self ** (scalar - 1.0)
        base_pow = ancestor.tensor() ** (exponent - 1.0)
        base_pow.requires_grad = False
        var local_grad = base_pow * exponent
        product = gradients * local_grad
        return [
            (
                ancestor,
                product,
                AddTensor,
            )
        ]

    fn __str__(self) -> String:
        return "AddBackward"

fn main():
    pass
