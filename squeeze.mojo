from tenmo import Tensor
from operators import AddTensor, ZeroGrad
from intarray import IntArray
from backpropagation import Delegate, BackwardFn, BACKWARD_SQUEEZE
from common_utils import panic
from shapes import Shape
from strides import Strides
from gradbox import Gradbox
from ancestry import Ancestor


@fieldwise_init
@register_passable
struct SqueezeBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_SQUEEZE
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ancestor = output.ancestry().get(0)
        var gradbox = output.grad()

        var original_shape = ancestor.shape()

        var gradbox_ancestor = gradbox.reshape(original_shape)
        return [
            (ancestor^, gradbox_ancestor^, AddTensor),
            (
                Ancestor(output),
                gradbox^,
                ZeroGrad,
            ),  # Send out a signal to this output of squeeze op to zero out its grad(No accumulation of grad for view)
        ]


struct Squeeze[dtype: DType]:
    # Squeeze specified axes or all dims of size 1 if no axes provided
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        """
        Squeeze dimensions of size 1.

        Note: This operation returns a view sharing storage with the input tensor.
        Gradients are properly propagated through the squeeze operation.

        Args:
            tensor: The tensor being squeezed.
            axes: Optional list of axes to squeeze. If empty, squeeze all dims of size 1.
            requires_grad: Optional override for gradient requirement.

        Returns:
            Tensor with specified dimensions squeezed (view of original data).
        """
        shape = tensor.shape()
        if shape.count_axes_of_size(1) == 0:
            return tensor.copy()
        rank = tensor.rank()
        # Determine which axes to squeeze
        var axes_to_squeeze: IntArray
        if not axes == IntArray():
            # Use the specified axes after validation
            axes_to_squeeze = IntArray.with_capacity(rank)
            seen = IntArray.with_capacity(len(axes))
            for axis in axes:
                normalized = axis if axis >= 0 else axis + rank
                if normalized < 0 or normalized >= rank:
                    panic(
                        "Tensor → squeeze: axis ",
                        axis.__str__(),
                        " out of range",
                    )
                if shape[normalized] != 1:
                    panic(
                        "Tensor → squeeze: cannot squeeze axis ",
                        axis.__str__(),
                        " with size ",
                        shape[normalized].__str__(),
                    )
                if normalized in seen:
                    panic("Tensor → squeeze duplicate axis", axis.__str__())
                seen.append(normalized)
                axes_to_squeeze.append(normalized)

            axes_to_squeeze.sort()

        else:
            axes_to_squeeze = shape.indices_of_axes_with_size(1)

        if len(axes_to_squeeze) == 0:
            return tensor.copy()

        new_size = rank - len(axes_to_squeeze)
        new_shape = IntArray.with_capacity(new_size)
        new_strides = IntArray.with_capacity(new_size)

        for i in range(rank):
            if i not in axes_to_squeeze:
                new_shape.append(shape[i])
                new_strides.append(tensor.strides()[i])

        squeezed_shape = Shape(new_shape)
        strides = Strides(new_strides)
        offset = tensor.offset()

        out = Tensor[dtype].build_view(
            tensor.address(),
            squeezed_shape,
            strides,
            offset,
            requires_grad=False,
        )

        @parameter
        if (
            track_grad
        ):  # comptime parameter - disaable generation of autograd code if False
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                bfn = SqueezeBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(bfn^)
                out.add_ancestry(tensor)

        return out^


from testing import assert_true


fn main() raises:
    print("passes")
