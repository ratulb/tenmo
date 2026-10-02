from .tensor import Tensor
from .shared.mnemonics import AddTensor
from .backpropagation import BackwardFn, BackwardFnType

from .shared.shapes import Shape
from .shared.intarray import IntArray
from .shared.strides import Strides
from .shared.broadcasthelper import ShapeBroadcaster
from .views import View
from .ancestry import Ancestor


@fieldwise_init
struct ExpandBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref gradbox = output.gradients()
        var ancestor = output.ancestry().get(0)
        var parent_shape = ancestor.shape()
        var gradbox_contracted = gradbox.sum_over_broadcasted_axes(parent_shape)

        ancestor.update_grad(gradbox_contracted^, AddTensor, None)

        parent_ids.append(ancestor._id)
        # View conduit: grad already forwarded to base above; always cleared.
        gradbox.zero_grad()


@fieldwise_init
struct Expand[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        target_shape: Shape,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var curr_shape = tensor.shape()
        var shape_expanded = ShapeBroadcaster.broadcast_shape(
            curr_shape, target_shape
        )

        var extra_dims = len(shape_expanded) - len(curr_shape)
        var unit_shape = Shape.Unit()  # Shape(1)
        var shape_padded = unit_shape * extra_dims + curr_shape
        var padded_strides = (
            IntArray.filled(extra_dims, 0) + tensor.strides().intarray()
        )

        var strides_expanded = IntArray.with_capacity(len(padded_strides))
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
            sync=sync,
        )

        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_()
                var backwardFn = BackwardFn.null_arg[Self.dtype](
                    ExpandBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, tensor)

        return out^
