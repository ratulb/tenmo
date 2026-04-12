from tenmo import Tensor
from mnemonics import AddTensor
from shapes import Shape
from backpropagation import (
    MinMaxArg,
)
from validators import Validator
from intarray import IntArray
from gradbox import Gradbox
from ndbuffer import NDBuffer


@fieldwise_init
struct MinMaxBackward[dtype: DType](ImplicitlyCopyable & Movable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = output.fn_arg().arg[MinMaxArg[Self.dtype]]
        var gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        var shape = ancestor.shape()
        var mask_grad = Gradbox[Self.dtype](bwd_arg.mask, share=False)

        if shape.rank() == 0:
            return [(ancestor^, mask_grad^, AddTensor)]

        var grad_expanded: Gradbox[Self.dtype]
        if gradbox.shape() == Shape():
            grad_expanded = Gradbox[Self.dtype].full(
                shape, gradbox.item(), share=False, device=bwd_arg.mask.device()
            )
        elif not bwd_arg.keepdims:
            grad_expanded = gradbox.unsqueeze(bwd_arg.axes).broadcast_to(
                shape, share=False
            )
        else:
            grad_expanded = gradbox.broadcast_to(shape, share=False)

        var grad_contrib = grad_expanded * mask_grad
        return [(ancestor^, grad_contrib^, AddTensor)]


@fieldwise_init
struct MinMax[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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
                var bwd_arg = MinMaxArg[Self.dtype](
                    normalized_axes, keepdims, mask_ndb^
                ).into_arg()
                out.fnArg = Optional(bwd_arg^)
                out.add_ancestry(self)

        return out^

fn main() raises:
    pass
