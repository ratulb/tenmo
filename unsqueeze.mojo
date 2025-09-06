from tensors import Tensor
from operators import AddTensor
from shared import TensorLite
from intlist import IntList
from shapes import Shape
from strides import Strides
from backpropagation import Delegate, BackwardFn
from squeeze import Squeeze
from common_utils import panic


@fieldwise_init
@register_passable
struct UnsqueezeBackward[dtype: DType](Copyable):
    var axes: IntList  # where axes were inserted

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        # Remove the axis we had inserted
        gradients_squeezed = Squeeze[dtype].squeeze(
            gradients, self.axes, requires_grad=False
        )
        ancestor = output.ancestry().get(0)[]
        return [(ancestor, gradients_squeezed, AddTensor)]


@register_passable
struct Unsqueeze[dtype: DType]:
    @staticmethod
    fn unsqueeze(
        tensor: Tensor[dtype],
        axes: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        """Unsqueeze multiple axes by inserting dimensions of size 1."""
        rank = tensor.shape.rank()
        new_axes_count = len(axes)

        if new_axes_count == 0:
            return tensor

        new_rank = rank + new_axes_count

        normalized_axes = IntList.with_capacity(new_axes_count)
        seen = IntList.with_capacity(new_axes_count)

        for axis in axes:
            normalized = axis if axis >= 0 else new_rank + axis
            if normalized < 0 or normalized >= new_rank:
                panic(
                    "Tensor → unsqueeze: axis", axis.__str__(), "out of range"
                )

            # Check for duplicates
            if normalized in seen:
                panic("Tensor → unsqueeze: duplicate axis", axis.__str__())
            seen.append(normalized)
            normalized_axes.append(normalized)

        # Sort axes for efficient insertion
        normalized_axes.sort()

        # Pre-allocate with exact capacity
        new_rank = rank + new_axes_count
        var new_shape = IntList.with_capacity(new_rank)
        var new_strides = IntList.with_capacity(new_rank)

        var orig_axis_index = 0
        var new_axis_index = 0

        # Build new shape and strides by inserting 1's at specified positions
        for i in range(new_rank):
            if (
                new_axis_index < new_axes_count
                and i == normalized_axes[new_axis_index]
            ):
                # Insert new dimension
                new_shape.append(1)

                # Calculate stride for inserted dimension
                # If inserting before existing dimensions, use stride of next dimension
                # If inserting at the end, use stride 1
                insert_stride = (
                    tensor.strides[orig_axis_index] if orig_axis_index
                    < rank else 1
                )

                new_strides.append(insert_stride)

                new_axis_index += 1
            else:
                # Copy existing dimension
                new_shape.append(tensor.shape[orig_axis_index])
                new_strides.append(tensor.strides[orig_axis_index])
                orig_axis_index += 1

        # Create the unsqueezed tensor
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
            bfn = UnsqueezeBackward[dtype](
                axes=normalized_axes
            ).into_backward_fn()
            out.backwardFn = Optional(bfn)
            out.add_ancestry(TensorLite.of(tensor))

        return out


fn main():
    print("passes")
