from .tensor import Tensor
from .mnemonics import AddTensor
from .intarray import IntArray
from .backpropagation import BackwardFnArg, BACKWARD_SQUEEZE
from .gradbox import Gradbox
from .shapes import Shape
from .common_utils import panic
from .ancestry import Ancestor


@fieldwise_init
struct SqueezeBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref ancestor = output.ancestry().get(0)
        ref gradbox = output.gradients()[]
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
        gradbox.zero_grad()


@fieldwise_init
struct Squeeze[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    # Squeeze specified axes or all dims of size 1 if no axes provided
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        mut tensor: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var shape = tensor.shape()
        if shape.count_axes_of_size(1) == 0:
            return tensor

        var squeezed_ndb = tensor.buffer.squeeze(axes, shared=True)
        if squeezed_ndb.shape == tensor.buffer.shape:
            return tensor

        var out = Tensor[Self.dtype](squeezed_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_SQUEEZE
                )
                out.add_ancestry(backwardFnArg^, tensor)

        return out^
