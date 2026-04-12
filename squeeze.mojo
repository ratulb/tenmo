from tenmo import Tensor
from mnemonics import AddTensor, ZeroGrad
from intarray import IntArray
from backpropagation import FnArg, BACKWARD_SQUEEZE
from gradbox import Gradbox
from shapes import Shape
from common_utils import panic


@fieldwise_init
struct SqueezeBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ancestor = output.ancestry().get(0)
        var gradbox = output.gradients()[]
        var ancestor_gradbox: Gradbox[Self.dtype]
        var original_shape = ancestor.shape()
        if gradbox.shape() == Shape():
            ancestor_gradbox = Gradbox[Self.dtype].full(
                original_shape,
                gradbox.item(),
                share=False,
                device=gradbox.device(),
            )
        else:
            ancestor_gradbox = gradbox.reshape(original_shape)
        return [
            (ancestor^, ancestor_gradbox^, AddTensor),
            (
                output,
                gradbox^,
                ZeroGrad,
            ),  # Send out a signal to this output of squeeze op to zero out its grad(No accumulation of grad for view)
        ]

@fieldwise_init
struct Squeeze[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    # Squeeze specified axes or all dims of size 1 if no axes provided
    @staticmethod
    fn forward[
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
                out.fnArg = Optional(FnArg[Self.dtype].null(BACKWARD_SQUEEZE))
                out.add_ancestry(tensor)

        return out^
