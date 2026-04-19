from tenmo import Tensor
from mnemonics import AddTensor, ZeroGrad
from intarray import IntArray
from backpropagation import BackwardFnArg, IntArrayArg, BACKWARD_UNSQUEEZE
from squeeze import Squeeze
from gradbox import Gradbox
from ancestry import Ancestor


@fieldwise_init
struct UnsqueezeBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var axes = output.ancestry().backward_fn_arg().get[IntArrayArg]().array
        var gradbox = output.gradients()[]
        # Remove the axis we had inserted
        var squeezed_gradbox = gradbox.squeeze(axes)

        var ancestor = output.ancestry().get(0)
        return [
            (ancestor, squeezed_gradbox^, AddTensor),
            (output, gradbox^, ZeroGrad),
        ]


@fieldwise_init
struct Unsqueeze[dtype: DType](ImplicitlyCopyable, RegisterPassable):
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
                var backwardFnArg = BackwardFnArg[Self.dtype].from_intarray(
                    BACKWARD_UNSQUEEZE, normalized
                )

                out.add_ancestry(backwardFnArg^, tensor)

        return out^
