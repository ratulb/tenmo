from tenmo import Tensor
from mnemonics import AddTensor
from shapes import Shape
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_MINMAX,
    BACKWARD_MINMAX_GPU,
)
from validators import Validator
from intarray import IntArray
from gradbox import Gradbox
from ndbuffer import NDBuffer


@fieldwise_init
struct MinMaxBackward[dtype: DType](ImplicitlyCopyable & Movable):
    comptime TAG = BACKWARD_MINMAX
    var axes: IntArray
    var keepdims: Bool
    var mask: NDBuffer[Self.dtype]  # shape == ancestor.shape, contiguous

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        var shape = ancestor.shape()
        var mask_grad = Gradbox[Self.dtype](self.mask)

        if shape.rank() == 0:
            return [(ancestor^, mask_grad^, AddTensor)]

        var grad_expanded: Gradbox[Self.dtype]
        if gradbox.shape() == Shape():
            grad_expanded = Gradbox[Self.dtype].full(
                shape, gradbox.item(), share=False
            )
        elif not self.keepdims:
            grad_expanded = gradbox.unsqueeze(self.axes).broadcast_to(
                shape, share=False
            )
        else:
            grad_expanded = gradbox.broadcast_to(shape, share=False)

        var grad_contrib = grad_expanded * mask_grad
        return [(ancestor^, grad_contrib^, AddTensor)]

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


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
                var var backward_fn: BackwardFn[Self.dtype]
                if self.is_on_gpu():
                    var mask_gradbox = Gradbox[Self.dtype](
                        mask_ndb^, share=False
                    )
                    backward_fn = MinMaxBackwardGPU[Self.dtype](
                        mask_gradbox^, normalized_axes, keepdims
                    ).into_backward_fn()
                else:
                    backward_fn = MinMaxBackward[Self.dtype](
                        normalized_axes, keepdims, mask_ndb^
                    ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


@fieldwise_init
struct MinMaxBackwardGPU[dtype: DType](ImplicitlyCopyable & Movable):
    comptime TAG = BACKWARD_MINMAX_GPU
    var mask: Gradbox[Self.dtype]  # on GPU, shape == ancestor.shape
    var axes: IntArray
    var keepdims: Bool

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        var shape = ancestor.shape()
        var grad_expanded: Gradbox[Self.dtype]
        if gradbox.shape() == Shape():
            # scalar upstream grad — full tensor on same device as mask
            grad_expanded = Gradbox[Self.dtype].full(
                shape,
                gradbox.item(),
                share=False,
                device=self.mask.device(),
            )
        elif not self.keepdims:
            grad_expanded = gradbox.unsqueeze(self.axes).broadcast_to(
                shape, share=False
            )
        else:
            grad_expanded = gradbox.broadcast_to(shape, share=False)
        return [(ancestor^, grad_expanded * self.mask, AddTensor)]

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


fn main() raises:
    pass
