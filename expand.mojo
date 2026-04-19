from tenmo import Tensor
from mnemonics import AddTensor
from backpropagation import BackwardFnArg, BACKWARD_EXPAND
from shapes import Shape
from intarray import IntArray
from strides import Strides
from gradbox import Gradbox
from broadcasthelper import ShapeBroadcaster
from views import View
from ancestry import Ancestor


@fieldwise_init
struct ExpandBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        ancestor = output.ancestry().get(0)
        parent_shape = ancestor.shape()
        gradbox_contracted = gradbox.sum_over_broadcasted_axes(parent_shape)

        return [(ancestor, gradbox_contracted^, AddTensor)]


@fieldwise_init
struct Expand[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut tensor: Tensor[Self.dtype],
        target_shape: Shape,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        curr_shape = tensor.shape()
        shape_expanded = ShapeBroadcaster.broadcast_shape(
            curr_shape, target_shape
        )

        extra_dims = len(shape_expanded) - len(curr_shape)
        unit_shape = Shape.Unit()  # Shape(1)
        shape_padded = unit_shape * extra_dims + curr_shape
        padded_strides = (
            IntArray.filled(extra_dims, 0) + tensor.strides().intarray()
        )

        strides_expanded = IntArray.with_capacity(len(padded_strides))
        for i in range(len(shape_expanded)):
            if shape_padded[i] == 1 and shape_expanded[i] > 1:
                # Broadcasted dimension → stride 0
                strides_expanded.append(0)
            else:
                strides_expanded.append(padded_strides[i])

        var strides = Strides(strides_expanded)

        var offset = tensor.offset()  # keep same as current tensor

        var out = View[Self.dtype].forward[track_grad=False](
            tensor,
            shape_expanded,
            strides,
            offset,
            requires_grad=False,
            validated=True,
        )

        comptime if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_()
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_EXPAND
                )
                out.add_ancestry(backwardFnArg^, tensor)

        return out^
