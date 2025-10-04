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
        gradients = output.grad()
        ancestor = output.ancestry().get(0)[]
        recipient_shape = ancestor.shape()
        reduced_grad = Tensor[dtype].sum_over_broadcasted_axes(
            gradients, recipient_shape
        )

        return [(ancestor, reduced_grad, AddTensor)]

@register_passable
struct Expand[dtype: DType]:
    @staticmethod
    fn forward[track_grad: Bool=True](
        mut tensor: Tensor[dtype],
        target: Shape,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        exp_shape = Shape.broadcast_shape(tensor.shape, target)

        ndim_diff = len(exp_shape) - len(tensor.shape)
        #padded_shape = Shape(1) * ndim_diff + tensor.shape
        padded_shape = Shape.Unit * ndim_diff + tensor.shape
        #padded_strides = IntList(0) * ndim_diff + tensor.strides.strides
        padded_strides = IntList.filled(ndim_diff, 0) + tensor.strides.strides

        exp_strides_list = IntList.Empty
        #exp_strides_list = IntList()
        for i in range(len(exp_shape)):
            if padded_shape[i] == 1 and exp_shape[i] > 1:
                # Broadcasted dimension → stride 0
                exp_strides_list.append(0)
            else:
                exp_strides_list.append(padded_strides[i])

        strides = Strides(exp_strides_list)

        offset = tensor.offset  # keep same as current tensor

        out = tensor.build_view(exp_shape, strides, offset, False)

        @parameter
        if track_grad:        

            grad_required = (
                requires_grad.value() if requires_grad else tensor.requires_grad
            )


            if grad_required:
                out.requires_grad_()
                var bfn = ExpandBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(bfn)
                out.add_ancestry(TensorLite.of(tensor))

        return out


fn main():
    print("passes")
