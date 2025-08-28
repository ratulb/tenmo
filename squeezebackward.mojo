from tensors import Tensor
from operators import AddTensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn


@fieldwise_init
@register_passable
struct SqueezeBackward[dtype: DType](Copyable & Movable):
    var axis: Int  # axis removed; if axis == -1 means "all singleton axes"

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        if self.axis == -1:
            # Re-insert all singleton axes (unsqueeze all dims that were size 1)
            original_shape = ancestor.shape()
            grad_unsqueezed = gradients
            for i in range(original_shape.rank()):
                if original_shape[i] == 1:
                    grad_unsqueezed = grad_unsqueezed.unsqueeze(i)
        else:
            # Re-insert single axis
            grad_unsqueezed = gradients.unsqueeze(axis=self.axis)

        return [(ancestor, grad_unsqueezed, AddTensor)]


fn main():
    print("passes")
