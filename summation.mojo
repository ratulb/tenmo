from tenmo import Tensor
from mnemonics import AddTensor
from intarray import IntArray
from shapes import Shape
from backpropagation import BackwardFnArg, ReductionArg, BACKWARD_SUM
from validators import Validator
from gradbox import Gradbox
from ancestors_newest import AncestorRef

@fieldwise_init
struct SumBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Tensor[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref bwd_arg = output.backward_fn_arg().get[ReductionArg]()
        var (axes, keepdims) = bwd_arg.axes, bwd_arg.keepdims
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
            if not keepdims:
                # Determine axes/unsqueeze (insert dims of size 1)
                axes = (
                    gradbox.shape()
                    .intarray()
                    .insert(
                        axes,
                        IntArray.filled(len(axes), 1),
                    )
                )
                unsqueezed_shape = Shape(axes)

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

    @staticmethod
    fn backward(
        output: AncestorRef[Self.dtype],
    ) -> List[Tuple[AncestorRef[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref bwd_arg = output.backward_fn_arg().get[ReductionArg]()
        var (axes, keepdims) = bwd_arg.axes, bwd_arg.keepdims
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        ref shape = ancestor.buffer().shape

        var grad_contrib: Gradbox[Self.dtype]

        if gradbox.shape() == Shape():
            grad_contrib = Gradbox[Self.dtype].full(
                shape, gradbox.item(), share=False, device=gradbox.device()
            )
        else:
            # Handle keepdims=False case (need to reshape gradient)
            if not keepdims:
                # Determine axes/unsqueeze (insert dims of size 1)
                axes = (
                    gradbox.shape()
                    .intarray()
                    .insert(
                        axes,
                        IntArray.filled(len(axes), 1),
                    )
                )
                unsqueezed_shape = Shape(axes)

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
                var backwardFnArg =  BackwardFnArg[Self.dtype](
                        BACKWARD_SUM, ReductionArg(reduction_axes, keepdims)

                )
                out.add_ancestry(backwardFnArg^, tensor)

        return out^



fn main() raises:
    test_sum_2d_backward_axis0_cpu()


from std.testing import assert_true

fn test_sum_2d_backward_axis0_cpu() raises:
    """Test backward of sum along axis 0 on CPU."""
    print("test_sum_2d_backward_axis0_cpu")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    var loss = a.sum(axes=[0])
    loss.backward_new()
    var expected = Tensor[dtype].d2([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert_true(a.grad().all_close(expected))
    print("✓ CPU 2D sum backward axis0 passed")
