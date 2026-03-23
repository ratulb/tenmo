from tenmo import Tensor
from mnemonics import AddTensor, ZeroGrad
from intarray import IntArray
from backpropagation import Delegate, BackwardFn, BACKWARD_SQUEEZE
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct SqueezeBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_SQUEEZE

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ancestor = output.ancestry().get(0)
        var gradbox = output.gradients()[]

        var original_shape = ancestor.shape()

        var gradbox_ancestor = gradbox.reshape(original_shape)
        return [
            (ancestor^, gradbox_ancestor^, AddTensor),
            (
                output,
                gradbox^,
                ZeroGrad,
            ),  # Send out a signal to this output of squeeze op to zero out its grad(No accumulation of grad for view)
        ]


@register_passable
struct Squeeze[dtype: DType]:
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

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var bfn = SqueezeBackward[Self.dtype]().into_backward_fn()
                out.backwardFn = Optional(bfn^)
                out.add_ancestry(tensor)

        return out^
