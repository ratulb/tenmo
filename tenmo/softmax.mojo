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
from .minmax import MinMax
from tenmo.kernels.reduction_kernel import Reduction
from .sum_mean_reduction import SumMeanReduction
from .common_utils import Epsilon
from std.sys import has_accelerator
from std.math import log, exp, max


@fieldwise_init
struct SoftmaxNdBuffer[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def softmax(
        ndb: NDBuffer[Self.dtype],
        axes: IntArray,
        validated: Bool = False,
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        var normalized_axes = (
            axes if validated else Validator.validate_and_normalize_axes(
                ndb.shape, axes
            )
        )
        var (_, stable_exp) = SoftmaxNdBuffer._softmax_components(
            ndb, normalized_axes
        )
        var exp_sum = SumMeanReduction[Self.dtype].sum(
            stable_exp, normalized_axes, keepdims=True
        )
        return stable_exp.arithmetic_ops[Divide](exp_sum)

    @staticmethod
    def log_sum(
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool = False,
        sync: Bool = False,
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    return Reduction[Self.dtype].launch_log_sum(
                        ndb, normalized_axes, keepdims, sync=sync
                    )
                except e:
                    print(e)
                    panic("SoftmaxNdBuffer.log_sum - GPU operation failed")
        return SoftmaxNdBuffer[Self.dtype]._log_sum_cpu(
            ndb, normalized_axes, keepdims
        )

    @staticmethod
    def _log_sum_cpu(
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        var out_shape = ndb.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var out = NDBuffer[Self.dtype].zeros(out_shape)

        if out_shape == Shape():
            var accum = Scalar[Self.dtype](0)
            for index in ndb.index_iterator():
                accum += exp(ndb.buffer[index])
            out[IntArray()] = log(max(accum, Epsilon[Self.dtype].value()))
        else:
            var reduction_axes_shape = ndb.shape.reduced_shape(normalized_axes)
            for out_coord in out_shape:
                var accum = Scalar[Self.dtype](0)
                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    accum += exp(ndb[self_coord])
                out[out_coord] = log(max(accum, Epsilon[Self.dtype].value()))

        return out^

    @staticmethod
    def log_softmax(
        ndb: NDBuffer[Self.dtype],
        axes: IntArray,
        validated: Bool = False,
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        var normalized_axes = (
            axes if validated else Validator.validate_and_normalize_axes(
                ndb.shape, axes
            )
        )
        var (stable, stable_exp) = SoftmaxNdBuffer._softmax_components(
            ndb, normalized_axes
        )
        var log_sum_exp = SoftmaxNdBuffer[Self.dtype].log_sum(
            stable, normalized_axes, keepdims=True
        )
        var exp_sum = SumMeanReduction[Self.dtype].sum(
            stable_exp, normalized_axes, keepdims=True
        )
        return stable.arithmetic_ops[Subtract](log_sum_exp), stable_exp.arithmetic_ops[Divide](exp_sum)

    @staticmethod
    def _softmax_components(
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        var (max_values, _) = MinMax[Self.dtype].minmax[is_max=True](
            ndb, normalized_axes, keepdims=True
        )
        var stable = ndb.arithmetic_ops[Subtract](max_values)
        var stable_exp = stable.exp()
        return stable, stable_exp


comptime SoftmaxBackward[dtype: DType] = SoftmaxBackwardDelegate[dtype, False]
comptime LogSoftmaxBackward[dtype: DType] = SoftmaxBackwardDelegate[dtype, True]


@fieldwise_init
struct SoftmaxBackwardDelegate[dtype: DType, is_log: Bool](
    ImplicitlyCopyable & Movable
):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var bwd_arg = (
            output.ancestry().backward_fn_arg().get[SoftmaxArg[Self.dtype]]()
        )
        var (axes, softmax_out) = bwd_arg.axes, bwd_arg.softmax_out
        ref gradbox = output.gradients()
        var ancestor = output.ancestry().get(0)
        var local_grad_ndb: NDBuffer[Self.dtype]

        comptime if Self.is_log:
            # g - softmax(x) * sum(g, axes, keepdims=True)
            var sum_grad = SumMeanReduction[Self.dtype].sum(
                gradbox.buffer(), axes, keepdims=True
            )
            var softmax_sum = softmax_out.arithmetic_ops[Multiply](sum_grad)
            local_grad_ndb = gradbox.buffer().arithmetic_ops[Subtract](softmax_sum)
        else:
            # y * (g - sum(g * y, axes, keepdims=True))
            var gy = gradbox.buffer().arithmetic_ops[Multiply](softmax_out)
            var gy_sum = SumMeanReduction[Self.dtype].sum(
                gy, axes, keepdims=True
            )
            var grad_diff = gradbox.buffer().arithmetic_ops[Subtract](gy_sum)
            local_grad_ndb = softmax_out.arithmetic_ops[Multiply](grad_diff)

        var local_grad = Gradbox[Self.dtype](local_grad_ndb^)
        ancestor.update_grad(local_grad^, AddTensor, None)
        parent_ids.append(ancestor._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct Softmax[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        this: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
        sync: Bool = False,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var shape = this.shape()

        # Normalize axes
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)

        var ndb = SoftmaxNdBuffer[Self.dtype].softmax(
            this.buffer, normalized_axes, validated=True
        )
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

        comptime if has_accelerator():
            if sync and out.is_on_gpu():
                out.buffer.sync()

        return out^


@fieldwise_init
struct LogSoftmax[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        this: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
        sync: Bool = False,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var shape = this.shape()

        # Normalize axes
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)

        var (ndb, softmax_vals) = SoftmaxNdBuffer[Self.dtype].log_softmax(
            this.buffer, normalized_axes, validated=True
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

        comptime if has_accelerator():
            if sync and out.is_on_gpu():
                out.buffer.sync()

        return out^
