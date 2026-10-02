from .shared.intarray import IntArray
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .backpropagation import (
    BackwardFnType,
    ArgumentType,
    BackwardFn,
)
from .shared.mnemonics import AddTensor, Subtract, Divide, Multiply
from .tensor import Tensor
from .validators import Validator
from .ancestry import Ancestor
from .minmax import MinMax
from .kernels.reduction_kernel import ReductionKernel
from .sum_mean_reduction import SumMeanReduction
from .shared.constants import Epsilon
from .shared.panic import panic
from .shared.shapes import Shape
from std.sys import has_accelerator
from std.math import log, exp, max
from std.sys import simd_width_of
from max.algorithm import parallelize
from std.sys.info import num_physical_cores
from std.utils.numerics import min_finite


@fieldwise_init
struct SoftmaxArg[dtype: DType](ArgumentType):
    var axes: IntArray
    var softmax_out: NDBuffer[Self.dtype]


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
        if SoftmaxNdBuffer._is_fused_eligible(ndb, normalized_axes):
            return SoftmaxNdBuffer._fused_softmax(ndb, normalized_axes)
        var (_, stable_exp) = SoftmaxNdBuffer._softmax_components(
            ndb, normalized_axes
        )
        var exp_sum = SumMeanReduction[Self.dtype].sum(
            stable_exp, normalized_axes, keepdims=True
        )
        return stable_exp.arithmetic_ops[Divide](exp_sum)

    @staticmethod
    def _is_fused_eligible(
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
    ) -> Bool:
        """Fused path requires: contiguous input, suffix-axis reduction, CPU."""
        if not ndb.is_contiguous():
            return False
        if ndb.is_on_gpu():
            return False
        var rank = ndb.shape.ndim()
        var num_axes = normalized_axes.size()
        if num_axes == 0:
            return False
        if normalized_axes[num_axes - 1] != rank - 1:
            return False
        var idx = 0
        while idx < num_axes - 1:
            if normalized_axes[idx] != rank - num_axes + idx:
                return False
            idx += 1
        return True

    @staticmethod
    def _fused_softmax(
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """Fused softmax: 3 SIMD passes, 1 output buffer, no intermediates."""
        var out = NDBuffer[Self.dtype].zeros(ndb.shape)
        var reduced_numels = (
            ndb.shape.reduced_shape(normalized_axes).product()
        )
        var num_out = ndb.numels() // reduced_numels
        comptime SIMD_WIDTH = simd_width_of[Self.dtype]()
        var simd_end = reduced_numels - (reduced_numels % SIMD_WIDTH)
        var ndb_offset = ndb.offset
        var ndb_buf = ndb.buffer

        def sm_row(oi: Int) {imm}:
            var ndb_base = ndb_offset + oi * reduced_numels
            var out_base = oi * reduced_numels

            # P1: row max
            var row_max = min_finite[Self.dtype]()
            for si in range(0, simd_end, SIMD_WIDTH):
                var vec = ndb_buf.load[simdwidth=SIMD_WIDTH](ndb_base + si)
                for i in range(SIMD_WIDTH):
                    row_max = max(row_max, vec[i])
            for ri in range(simd_end, reduced_numels):
                row_max = max(row_max, ndb_buf[ndb_base + ri])

            # P2: exp + sum, store raw e
            var sum_exp = Scalar[Self.dtype](0)
            for si in range(0, simd_end, SIMD_WIDTH):
                var vec = ndb_buf.load[simdwidth=SIMD_WIDTH](ndb_base + si)
                var e_vec = exp(vec - row_max)
                sum_exp += e_vec.reduce_add()
                out.buffer.store[simdwidth=SIMD_WIDTH](out_base + si, e_vec)
            for ri in range(simd_end, reduced_numels):
                var e = exp(ndb_buf[ndb_base + ri] - row_max)
                sum_exp += e
                out.buffer[out_base + ri] = e

            # P3: normalize in place (true division — matches old path)
            for si in range(0, simd_end, SIMD_WIDTH):
                var raw = out.buffer.load[simdwidth=SIMD_WIDTH](out_base + si)
                out.buffer.store[simdwidth=SIMD_WIDTH](
                    out_base + si, raw / sum_exp
                )
            for ri in range(simd_end, reduced_numels):
                out.buffer[out_base + ri] = out.buffer[out_base + ri] / sum_exp

        var n_threads = num_physical_cores()
        if (
            num_out >= n_threads
            and num_out * reduced_numels >= n_threads * 1024
        ):
            parallelize(sm_row, num_out, n_threads)
        else:
            for oi in range(num_out):
                sm_row(oi)
        return out^

    @staticmethod
    def log_sum(
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool = False,
        sync: Bool = True,
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    var (result_layout, result_storage) = ReductionKernel[
                        Self.dtype
                    ].launch_log_sum(
                        ndb.layout(),
                        ndb.device_state.value(),
                        normalized_axes,
                        keepdims,
                        sync=sync,
                    )
                    return NDBuffer[Self.dtype].with_layout_device_state(
                        result_layout, result_storage
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
            comptime SIMD_WIDTH = simd_width_of[Self.dtype]()
            var numels = ndb.numels()
            var simd_end = numels - (numels % SIMD_WIDTH)
            var ndb_buf = ndb.buffer
            for si in range(0, simd_end, SIMD_WIDTH):
                var vec = ndb_buf.load[simdwidth=SIMD_WIDTH](si)
                accum += exp(vec).reduce_add()
            for ri in range(simd_end, numels):
                accum += exp(ndb_buf[ri])
            out[IntArray()] = log(max(accum, Epsilon[Self.dtype].value()))
        else:
            var reduction_axes_shape = ndb.shape.reduced_shape(
                normalized_axes
            )

            # Fast path: contiguous suffix reduction with SIMD
            if ndb.is_contiguous():
                var rank = ndb.shape.ndim()
                var num_axes = normalized_axes.size()
                var is_suffix = (
                    num_axes > 0 and normalized_axes[num_axes - 1] == rank - 1
                )
                var idx = 0
                while is_suffix and idx < num_axes - 1:
                    if normalized_axes[idx] != rank - num_axes + idx:
                        is_suffix = False
                        break
                    idx += 1
                if is_suffix:
                    var reduced_numels = reduction_axes_shape.product()
                    var num_out = out.numels()
                    comptime SIMD_WIDTH = simd_width_of[Self.dtype]()
                    var ndb_offset = ndb.offset
                    var ndb_buf = ndb.buffer
                    for oi in range(num_out):
                        var base = ndb_offset + oi * reduced_numels
                        var simd_end = reduced_numels - (
                            reduced_numels % SIMD_WIDTH
                        )
                        var accum = Scalar[Self.dtype](0)
                        for si in range(0, simd_end, SIMD_WIDTH):
                            var vec = ndb_buf.load[simdwidth=SIMD_WIDTH](
                                base + si
                            )
                            accum += exp(vec).reduce_add()
                        for ri in range(simd_end, reduced_numels):
                            accum += exp(ndb_buf[base + ri])
                        out.buffer[oi] = log(
                            max(accum, Epsilon[Self.dtype].value())
                        )
                    return out^

            # Fallback: coord-by-coord
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
    def log_softmax[
        track_grad: Bool = True
    ](
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
        if SoftmaxNdBuffer._is_fused_eligible(ndb, normalized_axes):
            return SoftmaxNdBuffer._fused_log_softmax[track_grad](
                ndb, normalized_axes
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
        return stable.arithmetic_ops[Subtract](
            log_sum_exp
        ), stable_exp.arithmetic_ops[Divide](exp_sum)

    @staticmethod
    def _fused_log_softmax[
        track_grad: Bool
    ](
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        """Fused log-softmax: 3 SIMD passes, 2 output buffers, no intermediates.

        When track_grad=False the secondary softmax_vals buffer is skipped
        (returned Empty) — it is only needed by the backward pass.
        """
        var log_out = NDBuffer[Self.dtype].zeros(ndb.shape)
        var softmax_vals = NDBuffer[Self.dtype]()
        comptime if track_grad:
            softmax_vals = NDBuffer[Self.dtype].zeros(ndb.shape)
        var reduced_numels = (
            ndb.shape.reduced_shape(normalized_axes).product()
        )
        var num_out = ndb.numels() // reduced_numels
        comptime SIMD_WIDTH = simd_width_of[Self.dtype]()
        var simd_end = reduced_numels - (reduced_numels % SIMD_WIDTH)
        var ndb_offset = ndb.offset
        var ndb_buf = ndb.buffer

        def ls_row(oi: Int) {imm}:
            var ndb_base = ndb_offset + oi * reduced_numels
            var log_base = oi * reduced_numels

            # P1: row max
            var row_max = min_finite[Self.dtype]()
            for si in range(0, simd_end, SIMD_WIDTH):
                var vec = ndb_buf.load[simdwidth=SIMD_WIDTH](ndb_base + si)
                for i in range(SIMD_WIDTH):
                    row_max = max(row_max, vec[i])
            for ri in range(simd_end, reduced_numels):
                row_max = max(row_max, ndb_buf[ndb_base + ri])

            # P2: exp + sum; store stable into log_out
            var sum_exp = Scalar[Self.dtype](0)
            for si in range(0, simd_end, SIMD_WIDTH):
                var vec = ndb_buf.load[simdwidth=SIMD_WIDTH](ndb_base + si)
                var stable_vec = vec - row_max
                var e_vec = exp(stable_vec)
                sum_exp += e_vec.reduce_add()
                log_out.buffer.store[simdwidth=SIMD_WIDTH](
                    log_base + si, stable_vec
                )
                comptime if track_grad:
                    softmax_vals.buffer.store[simdwidth=SIMD_WIDTH](
                        log_base + si, e_vec
                    )
            for ri in range(simd_end, reduced_numels):
                var stable = ndb_buf[ndb_base + ri] - row_max
                var e = exp(stable)
                sum_exp += e
                log_out.buffer[log_base + ri] = stable
                comptime if track_grad:
                    softmax_vals.buffer[log_base + ri] = e

            # P3: finalize
            var log_sum_exp = log(max(sum_exp, Epsilon[Self.dtype].value()))
            for si in range(0, simd_end, SIMD_WIDTH):
                var stable_vec = log_out.buffer.load[simdwidth=SIMD_WIDTH](
                    log_base + si
                )
                log_out.buffer.store[simdwidth=SIMD_WIDTH](
                    log_base + si, stable_vec - log_sum_exp
                )
                comptime if track_grad:
                    var raw = softmax_vals.buffer.load[simdwidth=SIMD_WIDTH](
                        log_base + si
                    )
                    softmax_vals.buffer.store[simdwidth=SIMD_WIDTH](
                        log_base + si, raw / sum_exp
                    )
            for ri in range(simd_end, reduced_numels):
                log_out.buffer[log_base + ri] = (
                    log_out.buffer[log_base + ri] - log_sum_exp
                )
                comptime if track_grad:
                    softmax_vals.buffer[log_base + ri] = (
                        softmax_vals.buffer[log_base + ri] / sum_exp
                    )

        var n_threads = num_physical_cores()
        if (
            num_out >= n_threads
            and num_out * reduced_numels >= n_threads * 1024
        ):
            parallelize(ls_row, num_out, n_threads)
        else:
            for oi in range(num_out):
                ls_row(oi)
        return log_out^, softmax_vals^

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
    BackwardFnType, ImplicitlyCopyable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var bwd_arg = (
            output.ancestry().backward_fn().get[SoftmaxArg[Self.dtype]]()
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
            local_grad_ndb = gradbox.buffer().arithmetic_ops[Subtract](
                softmax_sum
            )
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
        sync: Bool = True,
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
                var backwardFn = BackwardFn(
                    SoftmaxArg[Self.dtype](
                        normalized_axes^,
                        ndb,
                    ),
                    SoftmaxBackward[Self.dtype](),
                )

                out.add_ancestry(backwardFn^, this)

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
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var shape = this.shape()

        # Normalize axes
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)

        var (ndb, softmax_vals) = SoftmaxNdBuffer[Self.dtype].log_softmax[
            track_grad
        ](this.buffer, normalized_axes, validated=True)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(this.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn(
                    SoftmaxArg[Self.dtype](
                        normalized_axes^,
                        softmax_vals,
                    ),
                    LogSoftmaxBackward[Self.dtype](),
                )

                out.add_ancestry(backwardFn^, this)

        comptime if has_accelerator():
            if sync and out.is_on_gpu():
                out.buffer.sync()

        return out^
