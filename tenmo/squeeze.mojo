from .tensor import Tensor
from .shared.mnemonics import AddTensor
from .shared.intarray import IntArray
from .backpropagation import BackwardFn, BackwardFnType

from .gradbox import Gradbox
from .shared.shapes import Shape
from .ancestry import Ancestor


@fieldwise_init
struct SqueezeBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var ancestor = output.ancestry().get(0)
        ref gradbox = output.gradients()
        var ancestor_gradbox: Gradbox[Self.dtype]
        var original_shape = ancestor.shape()
        if ancestor.requires_grad:
            if gradbox.shape() == Shape():
                ancestor_gradbox = Gradbox[Self.dtype].full(
                    original_shape,
                    gradbox.item(),
                    device=gradbox.device(),
                )
            else:
                ancestor_gradbox = gradbox.reshape(original_shape)
            ancestor.update_grad(ancestor_gradbox^, AddTensor, None)
        parent_ids.append(ancestor._id)
        # View conduit: always cleared.
        gradbox.zero_grad()


@fieldwise_init
struct Squeeze[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    # Squeeze specified axes or all dims of size 1 if no axes provided
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var shape = tensor.shape()
        # Honor an explicit requires_grad override even on these no-op
        # paths: a disagreeing override needs the normal wiring below
        # (tracked view + backward), not these raw aliases. The alias
        # itself is intentional identity (shared storage/_id, zero-copy).
        var tracked = requires_grad.or_else(tensor.requires_grad)
        if shape.count_axes_of_size(1) == 0 and tracked == tensor.requires_grad:
            return tensor

        var squeezed_ndb = tensor.buffer.squeeze(axes)
        if squeezed_ndb.shape == tensor.buffer.shape and tracked == tensor.requires_grad:
            return tensor

        var out = Tensor[Self.dtype](squeezed_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.null_arg[Self.dtype](
                    SqueezeBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, tensor)

        return out^
