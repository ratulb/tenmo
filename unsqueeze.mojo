from tenmo import Tensor
from operators import AddTensor, ZeroGrad
from intarray import IntArray
from shapes import Shape
from strides import Strides
from backpropagation import Delegate, BackwardFn
from squeeze import Squeeze
from common_utils import panic
from gradbox import Gradbox
from ancestry import Ancestor


@fieldwise_init
@register_passable
struct UnsqueezeBackward[dtype: DType](ImplicitlyCopyable):
    var axes: IntArray  # where axes were inserted

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var gradbox = output.grad()
        # Remove the axis we had inserted
        squeezed_gradbox = gradbox.squeeze(self.axes)

        ancestor = output.ancestry().get(0)
        return [
            (ancestor^, squeezed_gradbox^, AddTensor),
            (Ancestor(output), gradbox^, ZeroGrad),
        ]


@register_passable
struct Unsqueeze[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        """Unsqueeze multiple axes by inserting dimensions of size 1."""
        rank = tensor.rank()
        original_shape = tensor.shape()
        original_strides = tensor.strides()
        new_axes_count = len(axes)

        if new_axes_count == 0:
            return tensor.copy()

        new_rank = rank + new_axes_count

        normalized_axes = IntArray.with_capacity(new_axes_count)
        seen = IntArray.with_capacity(new_axes_count)

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
        var new_shape = IntArray.with_capacity(new_rank)
        var new_strides = IntArray.with_capacity(new_rank)

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
                    original_strides[orig_axis_index] if orig_axis_index
                    < rank else 1
                )

                new_strides.append(insert_stride)

                new_axis_index += 1
            else:
                # Copy existing dimension
                new_shape.append(original_shape[orig_axis_index])
                new_strides.append(original_strides[orig_axis_index])
                orig_axis_index += 1

        # Create the unsqueezed tensor
        shape = Shape(new_shape)
        strides = Strides(new_strides)
        out = Tensor[dtype].build_view(
            tensor.address(),
            shape,
            strides,
            tensor.offset(),
            requires_grad=False,
        )

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                bfn = UnsqueezeBackward[dtype](
                    axes=normalized_axes
                ).into_backward_fn()
                out.backwardFn = Optional(bfn^)
                out.add_ancestry(tensor)

        return out^


from testing import assert_true


fn main() raises:
    pass
