from tenmo import Tensor
from mnemonics import AddTensor, ZeroGrad
from intarray import IntArray
from backpropagation import Delegate, BackwardFn, BACKWARD_UNSQUEEZE
from squeeze import Squeeze
from gradbox import Gradbox


@fieldwise_init
struct UnsqueezeBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    comptime TAG = BACKWARD_UNSQUEEZE
    var axes: IntArray  # where axes were inserted

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var gradbox = output.gradients()[]
        # Remove the axis we had inserted
        var squeezed_gradbox = gradbox.squeeze(self.axes)

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
                var bfn = UnsqueezeBackward[Self.dtype](
                    axes=normalized
                ).into_backward_fn()
                out.backwardFn = Optional(bfn^)
                out.add_ancestry(tensor)

        return out^

    @staticmethod
    fn forward_unshared(
        tensor: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        if len(axes) == 0:
            return tensor.copy()

        var buffer = tensor.buffer.copy()
        var unsqueezed_ndb = buffer.unsqueeze(axes, shared=False)
        return Tensor[Self.dtype](
            unsqueezed_ndb^, requires_grad=requires_grad.or_else(False)
        )
