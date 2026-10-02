from .tensor import Tensor
from .shared.mnemonics import AddTensor
from .shared.intarray import IntArray
from .backpropagation import BackwardFn, IntArrayArg, BackwardFnType

from .ancestry import Ancestor


@fieldwise_init
struct UnsqueezeBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var axes = output.ancestry().backward_fn().get[IntArrayArg]().array
        ref gradbox = output.gradients()
        # Remove the axis we had inserted
        var squeezed_gradbox = gradbox.squeeze(axes)

        var ancestor = output.ancestry().get(0)
        if ancestor.requires_grad:
            ancestor.update_grad(squeezed_gradbox^, AddTensor, None)
        parent_ids.append(ancestor._id)
        # View conduit: always cleared.
        gradbox.zero_grad()


@fieldwise_init
struct Unsqueeze[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        # Honor an explicit requires_grad override even on this no-op path:
        # a disagreeing override needs the normal wiring below (tracked
        # view + backward), not this raw alias. The alias itself is
        # intentional identity (shared storage/_id, zero-copy).
        var tracked = requires_grad.or_else(tensor.requires_grad)
        if len(axes) == 0 and tracked == tensor.requires_grad:
            return tensor

        var unsqueezed_ndb = tensor.buffer.unsqueeze(axes)  # always a view
        var out = Tensor[Self.dtype](unsqueezed_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var rank = tensor.rank()
                var new_rank = rank + len(axes)
                var normalized = IntArray.with_capacity(len(axes))
                for axis in axes:
                    var n = axis if axis >= 0 else new_rank + axis
                    normalized.append(n)
                normalized.sort()
                var backwardFn = BackwardFn.from_intarray[Self.dtype](
                    normalized,
                    UnsqueezeBackward[Self.dtype](),
                )

                out.add_ancestry(backwardFn^, tensor)

        return out^
