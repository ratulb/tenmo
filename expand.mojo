from tensors import Tensor
from operators import AddTensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from shapes import Shape
from intlist import IntList
from strides import Strides

@fieldwise_init
@register_passable
struct ExpandBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        gradients.print()
        ancestor = output.ancestry().get(0)[]
        recipient_shape = ancestor.shape()
        reduced_grad = Tensor[dtype].sum_over_broadcasted_axes(
            gradients, recipient_shape
        )

        return [(ancestor, reduced_grad, AddTensor)]

@register_passable
struct Expand[dtype: DType]:
    @staticmethod
    fn forward(
        tensor: Tensor[dtype],
        target: Shape,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        exp_shape = Shape.broadcast_shape(tensor.shape, target)

        ndim_diff = len(exp_shape) - len(tensor.shape)
        padded_shape = Shape.Unit * ndim_diff + tensor.shape
        padded_strides = IntList(0) * ndim_diff + tensor.strides.strides

        exp_strides_list = IntList.Empty
        for i in range(len(exp_shape)):
            if padded_shape[i] == 1 and exp_shape[i] > 1:
                # Broadcasted dimension → stride 0
                exp_strides_list.append(0)
            else:
                exp_strides_list.append(padded_strides[i])

        strides = Strides(exp_strides_list)

        base_addr = tensor.address() if tensor.owns_data else tensor.base.copy()
        offset = tensor.offset  # keep same as current tensor

        grad_required = (
            requires_grad.value() if requires_grad else tensor.requires_grad
        )

        var out = Tensor[dtype](
            exp_shape, base_addr, strides, offset, grad_required
        )

        if grad_required:
            out.requires_grad_()
            var bfn = ExpandBackward[dtype]().into_backward_fn()
            out.backwardFn = Optional(bfn)
            out.add_ancestry(TensorLite.of(tensor))

        return out


fn main():
    print("passes")
