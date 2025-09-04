from tensors import Tensor
from shared import TensorLite
from operators import AddTensor
from backpropagation import Delegate, BackwardFn


@fieldwise_init
@register_passable
struct ReLUBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        this = output.tensor()
        shape = this.shape
        grad_share = Tensor[dtype].ones(shape)
        zero = Scalar[dtype](0)
        for indices in shape:
            grad_share[indices] = (
                gradients[indices] if this[indices] > zero else zero
            )
        return [(ancestor, grad_share, AddTensor)]


@fieldwise_init
@register_passable
struct ReLUForward[dtype: DType]:
    @staticmethod
    fn relu(
        self: Tensor[dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = self.shape
        out = Tensor[dtype].zeros(shape)
        zero = Scalar[dtype](0)
        for indices in shape:
            out[indices] = self[indices] if self[indices] > zero else zero
        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )
        if grad_required:
            out.requires_grad_(True)
            backward_fn = ReLUBackward[dtype]().into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite[dtype].of(self))

        return out


fn main():
    a = Tensor.arange(5, requires_grad=True)
    out = a.relu()
    out.print()
    print()
    out.backward()
    a.gradbox[].print()
    print()
    print("passes")
