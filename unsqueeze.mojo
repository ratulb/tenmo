from tenmo import Tensor
from mnemonics import AddTensor, ZeroGrad
from intarray import IntArray
from backpropagation import IntArrayArg, BACKWARD_UNSQUEEZE
from squeeze import Squeeze
from gradbox import Gradbox


@fieldwise_init
struct UnsqueezeBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var axes = output.fn_arg().arg[IntArrayArg].axes
        var gradbox = output.gradients()[]
        # Remove the axis we had inserted
        var squeezed_gradbox = gradbox.squeeze(axes)

        var ancestor = output.ancestry().get(0)
        return [
            (ancestor^, squeezed_gradbox^, AddTensor),
            (output, gradbox^, ZeroGrad),
        ]


@fieldwise_init
struct Unsqueeze[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut tensor: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        if len(axes) == 0:
            return tensor.copy()

        var unsqueezed_ndb = tensor.buffer.unsqueeze(
            axes
        )  # shared=True default
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
                var bwd_arg = IntArrayArg(
                    axes=normalized
                ).into_arg[Self.dtype](BACKWARD_UNSQUEEZE)
                out.fnArg = Optional(bwd_arg^)
                out.add_ancestry(tensor)

        return out^
