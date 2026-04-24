from .intarray import IntArray
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .backpropagation import (
    BackwardFnArg,
    SoftmaxArg,
    BACKWARD_SOFTMAX,
    BACKWARD_LOG_SOFTMAX,
)
from .mnemonics import AddTensor
from .tensor import Tensor
from .validators import Validator
from .ancestry import Ancestor

comptime SoftmaxBackward[dtype: DType] = SoftmaxBackwardDelegate[dtype, False]
comptime LogSoftmaxBackward[dtype: DType] = SoftmaxBackwardDelegate[dtype, True]


@fieldwise_init
struct SoftmaxBackwardDelegate[dtype: DType, is_log: Bool](
    ImplicitlyCopyable & Movable
):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = (
            output.ancestry().backward_fn_arg().get[SoftmaxArg[Self.dtype]]()
        )
        var (axes, softmax_out) = bwd_arg.axes, bwd_arg.softmax_out
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        var local_grad_ndb: NDBuffer[Self.dtype]

        comptime if Self.is_log:
            # g - softmax(x) * sum(g, axes, keepdims=True)
            var sum_grad = gradbox.buffer.sum(axes, keepdims=True)
            var softmax_sum = softmax_out * sum_grad
            local_grad_ndb = gradbox.buffer - softmax_sum
        else:
            # y * (g - sum(g * y, axes, keepdims=True))
            var gy = gradbox.buffer * softmax_out
            var gy_sum = gy.sum(axes, keepdims=True)
            var grad_diff = gradbox.buffer - gy_sum
            local_grad_ndb = softmax_out * grad_diff

        var local_grad = Gradbox[Self.dtype](local_grad_ndb^, share=False)
        return [(ancestor^, local_grad^, AddTensor)]


@fieldwise_init
struct Softmax[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        this: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var shape = this.shape()

        # Normalize axes
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)

        var ndb = this.buffer.softmax(normalized_axes, validated=True)
        var out = Tensor[Self.dtype](ndb, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(this.requires_grad)
            if grad_required:
                out.requires_grad_(True)

                # Store NDBuffer — carries device state, GPU safe
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_SOFTMAX,
                    SoftmaxArg[Self.dtype](
                        normalized_axes^,
                        ndb,
                    ),
                )

                out.add_ancestry(backwardFnArg^, this)

        return out^


@fieldwise_init
struct LogSoftmax[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        this: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var shape = this.shape()

        # Normalize axes
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)

        var (ndb, softmax_vals) = this.buffer.log_softmax(
            normalized_axes, validated=True
        )
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(this.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_LOG_SOFTMAX,
                    SoftmaxArg[Self.dtype](
                        normalized_axes^,
                        softmax_vals,
                    ),
                )

                out.add_ancestry(backwardFnArg^, this)

        return out^

