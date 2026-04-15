from tenmo import Tensor
from mnemonics import AddTensor
from intarray import IntArray
from shapes import Shape
from backpropagation import BackwardFnArg, ReductionArg, BACKWARD_SUM
from validators import Validator
from gradbox import Gradbox


@fieldwise_init
struct SumBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Tensor[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = output.backward_fn_arg().get[ReductionArg]()
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        shape = ancestor.shape()
        var grad_contrib: Gradbox[Self.dtype]
        if gradbox.shape() == Shape():
            grad_contrib = Gradbox[Self.dtype].full(
                shape, gradbox.item(), share=False, device=gradbox.device()
            )
        else:
            # Handle keepdims=False case (need to reshape gradient)
            if not bwd_arg.keepdims:
                # Determine axes/unsqueeze (insert dims of size 1)
                axes = (
                    gradbox.shape()
                    .intarray()
                    .insert(
                        bwd_arg.axes,
                        IntArray.filled(len(bwd_arg.axes), 1),
                    )
                )
                unsqueezed_shape = Shape(bwd_arg.axes)

                unsqueezed_grad = gradbox.reshape(unsqueezed_shape)
                grad_contrib = unsqueezed_grad.broadcast_to(shape, share=False)
            else:
                # keepdims=True: shapes match except for broadcasting
                grad_contrib = gradbox.broadcast_to(shape, share=False)

        return [
            (
                ancestor^,
                grad_contrib^,
                AddTensor,
            )
        ]


@fieldwise_init
struct Summer[dtype: DType](ImplicitlyCopyable, RegisterPassable):
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

        comptime if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                out.backwardFnArg = Optional(
                    BackwardFnArg[Self.dtype](
                        BACKWARD_SUM, ReductionArg(reduction_axes, keepdims)
                    )
                )
                out.add_ancestry(tensor)

        return out^


fn main():
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(10, requires_grad=True)
    var s = a.sum()
    s.backward()
    a.grad().print()
