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
from minmax_reducer import MinMaxReducer
from ndbuffer import NDBuffer
from sys import has_accelerator
from common_utils import panic
from minmax_kernel import ReductionMinMax


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
@register_passable
struct MinMax[dtype: DType = DType.float32]:
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

        @parameter
        if has_accelerator():
            if self.buffer.is_on_gpu():
                try:
                    var (result_ndb, mask_ndb) = ReductionMinMax[
                        Self.dtype
                    ].launch[is_max=max](self.buffer, normalized_axes, keepdims)
                    var result = Tensor[Self.dtype](
                        result_ndb^, requires_grad=False
                    )

                    @parameter
                    if track_grad:
                        var grad_required = requires_grad.or_else(
                            self.requires_grad
                        )
                        if grad_required:
                            result.requires_grad_(True)
                            # Wrap mask NDBuffer in a Gradbox (contiguous, on GPU)
                            var mask_gradbox = Gradbox[Self.dtype](
                                mask_ndb^, share=False
                            )
                            var backward_fn = MinMaxBackwardGPU[Self.dtype](
                                mask_gradbox^, normalized_axes, keepdims
                            ).into_backward_fn()
                            result.backwardFn = Optional(backward_fn^)
                            result.add_ancestry(self)

                    return result^
                except e:
                    panic("MinMax.forward GPU path failed: " + e.__str__())
                    # unreachable
                    return Tensor[Self.dtype](Shape())

        # CPU path — unchanged
        var result_ndb = MinMaxReducer[Self.dtype].reduce_minmax[max](
            self.buffer, normalized_axes, keepdims
        )
        var result = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                result.requires_grad_(True)
                var mask_ndb = MinMaxReducer[Self.dtype].build_minmax_mask[max](
                    self.buffer, result.buffer, normalized_axes, keepdims
                )
                var backward_fn = MinMaxBackward[Self.dtype](
                    normalized_axes, keepdims, mask_ndb^
                ).into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                result.add_ancestry(self)

        return result^


@fieldwise_init
struct MinMaxBackwardGPU[dtype: DType](ImplicitlyCopyable & Movable):
    comptime TAG = BACKWARD_MINMAX_GPU
    var mask: Gradbox[Self.dtype]  # on GPU, shape == ancestor.shape
    var axes: IntArray
    var keepdims: Bool

    fn backward(
        self, read output: Tensor[Self.dtype]
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
