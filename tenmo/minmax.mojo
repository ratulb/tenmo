from .tensor import Tensor
from .mnemonics import AddTensor
from .shapes import Shape
from .backpropagation import BackwardFnArg, MinMaxArg, BACKWARD_MINMAX
from .validators import Validator
from .intarray import IntArray
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .ancestry import Ancestor


@fieldwise_init
struct MinMaxBackward[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = (
            output.ancestry().backward_fn_arg().get[MinMaxArg[Self.dtype]]()
        )
        var (axes, keepdims, mask) = (
            bwd_arg.axes,
            bwd_arg.keepdims,
            bwd_arg.mask,
        )
        var gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        var shape = ancestor.shape()
        var mask_grad = Gradbox[Self.dtype](mask, share=False)

        if shape.rank() == 0:
            return [(ancestor, mask_grad^, AddTensor)]

        var grad_expanded: Gradbox[Self.dtype]
        if gradbox.shape() == Shape():
            grad_expanded = Gradbox[Self.dtype].full(
                shape, gradbox.item(), share=False, device=mask.device()
            )
        elif not keepdims:
            grad_expanded = gradbox.unsqueeze(axes).broadcast_to(
                shape, share=False
            )
        else:
            grad_expanded = gradbox.broadcast_to(shape, share=False)

        var grad_contrib = grad_expanded * mask_grad
        return [(ancestor, grad_contrib^, AddTensor)]


@fieldwise_init
struct MinMax[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        max: Bool, track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var shape = self.shape()
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        var tracking_grad = track_grad and requires_grad.or_else(
            self.requires_grad
        )
        var (result_ndb, mask_ndb) = self.buffer.minmax[is_max=max](
            normalized_axes, keepdims, tracking_grad
        )
        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_MINMAX,
                    MinMaxArg[Self.dtype](normalized_axes, keepdims, mask_ndb^),
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^


fn main() raises:
    pass
