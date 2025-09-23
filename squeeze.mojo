from tensors import Tensor
from operators import AddTensor
from shared import TensorLite
from intlist import IntList
from backpropagation import Delegate, BackwardFn
from common_utils import panic
from shapes import Shape
from strides import Strides


@fieldwise_init
@register_passable
struct SqueezeBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        ancestor = output.ancestry().get(0)[]
        gradients = output.gradients()[]

        var original_shape = ancestor.shape()

        # Create gradient with the same shape as original tensor
        var final_grad = gradients.reshape(original_shape, requires_grad=False)
        return [(ancestor, final_grad, AddTensor)]


struct Squeeze[dtype: DType]:
    # Squeeze specified axes or all dims of size 1 if no axes provided
    @staticmethod
    fn squeeze(
        tensor: Tensor[dtype],
        axes: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        """
        Squeeze dimensions of size 1.

        Args:
            tensor: The tensor being squeezed.
            axes: Optional list of axes to squeeze. If None, squeeze all dims of size 1.
            requires_grad: Optional override for gradient requirement.

        Returns:
            Tensor with specified dimensions squeezed.
        """
        if tensor.shape.count_axes_of_size(1) == 0:
            return tensor
        rank = tensor.shape.rank()

        # Determine which axes to squeeze
        var axes_to_squeeze: IntList
        if not axes == IntList():
            # Use the specified axes after validation
            axes_to_squeeze = IntList.with_capacity(rank)
            seen = IntList.with_capacity(len(axes))
            for axis in axes:
                normalized = axis if axis >= 0 else axis + rank
                if normalized < 0 or normalized >= rank:
                    panic(
                        "Tensor → squeeze: axis ",
                        axis.__str__(),
                        " out of range",
                    )
                if tensor.shape[normalized] != 1:
                    panic(
                        "Tensor → squeeze: cannot squeeze axis ",
                        axis.__str__(),
                        " with size ",
                        tensor.shape[normalized].__str__(),
                    )
                if normalized in seen:
                    panic("Tensor → squeeze dupliacte axis", axis.__str__())
                seen.append(normalized)
                axes_to_squeeze.append(normalized)

            axes_to_squeeze.sort()

        else:
            axes_to_squeeze = tensor.shape.indices_of_axes_with_size(1)

        if len(axes_to_squeeze) == 0:
            return tensor

        new_size = rank - len(axes_to_squeeze)
        new_shape = IntList.with_capacity(new_size)
        new_strides = IntList.with_capacity(new_size)

        for i in range(rank):
            if i not in axes_to_squeeze:
                new_shape.append(tensor.shape[i])
                new_strides.append(tensor.strides[i])

        shape = Shape(new_shape)
        strides = Strides(new_strides)
        grad_required = (
            requires_grad.value() if requires_grad else tensor.requires_grad
        )

        base_addr = tensor.address() if tensor.owns_data else tensor.base.copy()

        var out = Tensor[dtype](
            shape, base_addr, strides, tensor.offset, grad_required
        )

        if grad_required:
            out.requires_grad_()
            bfn = SqueezeBackward[dtype]().into_backward_fn()
            out.backwardFn = Optional(bfn)
            out.add_ancestry(TensorLite.of(tensor))

        return out


fn main():
    print("passes")
