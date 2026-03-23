from tenmo import Tensor
from mnemonics import AddTensor
from intarray import IntArray
from shapes import Shape
from backpropagation import Delegate, BackwardFn, BACKWARD_SUM
from validators import Validator
from gradbox import Gradbox
from common_utils import panic


@fieldwise_init
@register_passable
struct SumBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_SUM
    var axes: IntArray
    var keepdims: Bool

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        print("In sum backward")
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        shape = ancestor.shape()
        var grad_contrib: Gradbox[Self.dtype]
        # SumBackward.backward — already raises-capable via panic pattern
        if gradbox.shape() == Shape():
            grad_contrib = Gradbox[Self.dtype].full(
                shape, gradbox.item(), share=False, device=gradbox.device()
            )
        else:
            # Handle keepdims=False case (need to reshape gradient)
            if not self.keepdims:
                # Determine axes/unsqueeze (insert dims of size 1)
                axes = (
                    gradbox.shape()
                    .intarray()
                    .insert(
                        self.axes,
                        IntArray.filled(len(self.axes), 1),
                    )
                )
                unsqueezed_shape = Shape(axes)

                unsqueezed_grad = gradbox.reshape(unsqueezed_shape)
                grad_contrib = unsqueezed_grad.broadcast_to(shape, share=False)
            else:
                # keepdims=True: shapes match except for broadcasting
                grad_contrib = gradbox.broadcast_to(shape, share=False)
        print("Out of sumbackward")

        return [
            (
                ancestor^,
                grad_contrib^,
                AddTensor,
            )
        ]

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct Summer[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var shape = tensor.shape()
        var reduction_axes = Validator.normalize_reduction_axes(shape, axes)
        var nd_buffer = tensor.buffer.reduce(reduction_axes, keepdims)
        var out = Tensor[Self.dtype](nd_buffer^, requires_grad=False)

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                backward_fn = SumBackward[Self.dtype](
                    reduction_axes, keepdims
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(tensor)

        return out^


fn main() raises:
    comptime dtype = DType.float32

    var a = Tensor[dtype].arange(12, requires_grad=True)
    var a_gpu = a.to_gpu()
    var s = a_gpu.sum()
    s.backward()
    a.grad().print()

    var b = (
        Tensor[dtype].arange(12, requires_grad=True).reshape(3, 4).contiguous()
    )
    var c = b * 42
    var c_gpu = c.to_gpu()
    var ss = c.sum()

    ss.backward()
    b.grad().print()
