from summation import Summer
from subtraction import Subtractor
from division import Divider
from minmax import MinMax
from intarray import IntArray
from gradbox import Gradbox
from shapes import Shape
from ndbuffer import NDBuffer
from backpropagation import (
    BackwardFn,
    Delegate,
    BACKWARD_SOFTMAX,
    BACKWARD_LOG_SOFTMAX,
)
from mnemonics import AddTensor
from tenmo import Tensor
from sys import has_accelerator
from common_utils import panic
from validators import Validator

# ── SoftmaxBackward ───────────────────────────────────────────────────────────


@fieldwise_init
struct SoftmaxBackward[dtype: DType](ImplicitlyCopyable & Movable):
    comptime TAG = BACKWARD_SOFTMAX
    var axes: IntArray
    var softmax_out: NDBuffer[Self.dtype]  # carries device state — GPU safe

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes.copy()
        self.softmax_out = other.softmax_out

    fn __moveinit__(out self, deinit other: Self):
        self.axes = other.axes^
        self.softmax_out = other.softmax_out^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        # softmax_grad = y * (g - sum(g * y, axes, keepdims=True))
        # All ops at NDBuffer level — GPU safe, no LLVM lowering issues
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)

        # Step 1: g * y — arithmetic_ops, GPU safe
        var gy = gradbox.buffer * self.softmax_out

        # Step 2: sum(g * y, axes, keepdims=True) — axis sum, GPU safe
        var gy_sum = gy.sum(self.axes, keepdims=True)

        # Step 3: g - sum(g * y) — broadcast subtract, GPU safe
        var grad_diff = gradbox.buffer - gy_sum

        # Step 4: y * (g - sum(g * y)) — arithmetic_ops, GPU safe
        var local_grad_ndb = self.softmax_out * grad_diff

        var local_grad = Gradbox[Self.dtype](local_grad_ndb^, share=False)

        return [(ancestor^, local_grad^, AddTensor)]


# ── Softmax forward ───────────────────────────────────────────────────────────


@fieldwise_init
@register_passable
struct Softmax[dtype: DType]:
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

        # Numerical stability: subtract max along axes
        _="""var max_vals = MinMax[Self.dtype].forward[max=True, track_grad=False](
            this, normalized_axes, keepdims=True, requires_grad=False
        )

        var stable = Subtractor[Self.dtype].forward[track_grad=False](
            this, max_vals
        )

        # Compute exponentials — track_grad=False, intermediate tensor
        var stable_exp = stable.exp[track_grad=False]()

        # Sum of exponentials along axes
        var exp_sum = Summer[Self.dtype].forward[track_grad=False](
            stable_exp, normalized_axes, True
        )

        # Divide to get softmax
        var out = Divider[Self.dtype].forward[track_grad=False](
            stable_exp, exp_sum
        )"""
        var ndb = this.buffer.softmax(normalized_axes, validated=True)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(this.requires_grad)
            if grad_required:
                out.requires_grad_(True)

                # Store NDBuffer — carries device state, GPU safe
                # contiguous() ensures zero offset and contiguous layout
                var backward_fn = SoftmaxBackward[Self.dtype](
                    normalized_axes^,
                    out.buffer.contiguous(),
                ).into_backward_fn()

                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(this)

        return out^


# ── LogSoftmaxBackward ────────────────────────────────────────────────────────


@fieldwise_init
struct LogSoftmaxBackward[dtype: DType](ImplicitlyCopyable & Movable):
    comptime TAG = BACKWARD_LOG_SOFTMAX
    var axes: IntArray
    var softmax_out: NDBuffer[Self.dtype]  # carries device state — GPU safe

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes.copy()
        self.softmax_out = other.softmax_out

    fn __moveinit__(out self, deinit other: Self):
        self.axes = other.axes^
        self.softmax_out = other.softmax_out^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        # Gradient for log_softmax:
        # g - softmax(x) * sum(g, axes, keepdims=True)
        # All ops at NDBuffer level — GPU safe, no LLVM lowering issues
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)

        # Step 1: sum(g, axes, keepdims=True) — axis sum, GPU safe
        var sum_grad = gradbox.buffer.sum(self.axes, keepdims=True)

        # Step 2: softmax(x) * sum(g) — arithmetic_ops, GPU safe
        var softmax_sum = self.softmax_out * sum_grad

        # Step 3: g - softmax(x) * sum(g) — broadcast subtract, GPU safe
        var local_grad_ndb = gradbox.buffer - softmax_sum

        var local_grad = Gradbox[Self.dtype](local_grad_ndb^, share=False)

        return [(ancestor^, local_grad^, AddTensor)]


# ── LogSoftmax forward ────────────────────────────────────────────────────────


@fieldwise_init
@register_passable
struct LogSoftmax[dtype: DType]:
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

        # Numerical stability: subtract max along axes
        _="""var max_vals = MinMax[Self.dtype].forward[max=True, track_grad=False](
            this, normalized_axes, keepdims=True, requires_grad=False
        )

        var stable = Subtractor[Self.dtype].forward[track_grad=False](
            this, max_vals
        )

        # Compute exponentials — track_grad=False, intermediate tensor
        var stable_exp = stable.exp[track_grad=False]()

        # Sum of exponentials along axes
        var exp_sum = Summer[Self.dtype].forward[track_grad=False](
            stable_exp, normalized_axes, True
        )

        # log(sum(exp(x - max(x)))) — no grad tracking needed
        # Uses Logarithm.forward via Tensor.log — track_grad=False
        var log_sum_exp = exp_sum.log[track_grad=False]()

        # Log softmax: (x - max(x)) - log(sum(exp(x - max(x))))
        var out = Subtractor[Self.dtype].forward[track_grad=False](
            stable, log_sum_exp
        )"""

        var (ndb, softmax_vals) = this.buffer.log_softmax(normalized_axes, validated=True)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(this.requires_grad)
            if grad_required:
                out.requires_grad_(True)

                # Compute softmax for backward — needed for grad computation
                # Store as NDBuffer — carries device state, GPU safe
                _="""var softmax_vals = Divider[Self.dtype].forward[
                    track_grad=False
                ](stable_exp, exp_sum)

                var backward_fn = LogSoftmaxBackward[Self.dtype](
                    normalized_axes^,
                    softmax_vals.buffer.contiguous(),
                ).into_backward_fn()"""

                var backward_fn = LogSoftmaxBackward[Self.dtype](
                    normalized_axes^,
                    softmax_vals,
                ).into_backward_fn()

                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(this)

        return out^


fn main() raises:
    pass
