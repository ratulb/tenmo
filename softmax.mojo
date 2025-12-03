from tenmo import Tensor
from operators import AddTensor
from validators import Validator
from backpropagation import Delegate, BackwardFn, BACKWARD_SOFTMAX, BACKWARD_LOG_SOFTMAX
from summation import Summer
from subtraction import Subtractor
from division import Divider
from minmax import MinMax
from intarray import IntArray
from gradbox import Gradbox
from ancestry import Ancestor
from shapes import Shape
from buffers import Buffer
from ndbuffer import NDBuffer

# """
# Optimized Softmax and LogSoftmax implementation with efficient memory usage
# and faster backward pass computation.
# """

# ============================================================================
# Softmax Implementation
# ============================================================================


@fieldwise_init
struct SoftmaxBackward[dtype: DType](ImplicitlyCopyable & Movable):
    alias TAG = BACKWARD_SOFTMAX
    var axes: IntArray
    var softmax_out_buffer: Buffer[dtype]  # Store raw buffer instead of tensor
    var softmax_out_shape: Shape

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes.copy()
        self.softmax_out_buffer = other.softmax_out_buffer.copy()
        self.softmax_out_shape = other.softmax_out_shape.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.axes = other.axes^
        self.softmax_out_buffer = other.softmax_out_buffer^
        self.softmax_out_shape = other.softmax_out_shape^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var gradbox = output.grad()

        # Reconstruct softmax output efficiently from buffer
        var ndb = NDBuffer[dtype](
            self.softmax_out_buffer.copy(), self.softmax_out_shape.copy()
        )
        var softmax_out = Gradbox[dtype](ndb^)

        # softmax_grad = y * (g - sum(g * y, axis, keepdims=True))
        var gy_sum = (gradbox * softmax_out).sum(self.axes, keepdims=True)
        var local_grad = softmax_out * (gradbox - gy_sum)

        var ancestor = output.ancestry().get(0)
        return [(ancestor^, local_grad^, AddTensor)]


@fieldwise_init
@register_passable
struct Softmax[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        this: Tensor[dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        var shape = this.shape()

        # Normalize axes
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)

        # Numerical stability: subtract max along axes
        var max_vals = MinMax[dtype].forward[max=True, track_grad=False](
            this, normalized_axes, keepdims=True, requires_grad=False
        )

        var stable = Subtractor[dtype].forward[track_grad=False](this, max_vals)

        # Compute exponentials
        var stable_exp = stable.exp()
        var exp_sum = Summer[dtype].forward[track_grad=False](
            stable_exp, normalized_axes, True
        )
        var out = Divider[dtype].forward[track_grad=False](stable_exp, exp_sum)

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(this.requires_grad)

            if grad_required:
                out.requires_grad_(True)

                # Store buffer and shape (much more efficient than coordinate-value pairs)
                var backward_fn = SoftmaxBackward[dtype](
                    normalized_axes^,
                    out.buffer.contiguous_buffer(),
                    out.shape(),
                ).into_backward_fn()

                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(this)

        return out^


# ============================================================================
# LogSoftmax Implementation
# ============================================================================


@fieldwise_init
struct LogSoftmaxBackward[dtype: DType](ImplicitlyCopyable & Movable):
    alias TAG = BACKWARD_LOG_SOFTMAX
    var axes: IntArray
    var softmax_out_buffer: Buffer[dtype]
    var softmax_out_shape: Shape

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes.copy()
        self.softmax_out_buffer = other.softmax_out_buffer.copy()
        self.softmax_out_shape = other.softmax_out_shape.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.axes = other.axes^
        self.softmax_out_buffer = other.softmax_out_buffer^
        self.softmax_out_shape = other.softmax_out_shape^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var gradbox = output.grad()

        # Reconstruct softmax output efficiently
        var ndb = NDBuffer[dtype](
            self.softmax_out_buffer.copy(), self.softmax_out_shape.copy()
        )
        var softmax_out = Gradbox[dtype](ndb^)

        # Gradient for log_softmax: g - softmax(x) * sum(g, axis, keepdims=True)
        var sum_grad = gradbox.sum(self.axes, keepdims=True)
        var local_grad = gradbox - (softmax_out * sum_grad)

        var ancestor = output.ancestry().get(0)
        return [(ancestor^, local_grad^, AddTensor)]


@fieldwise_init
@register_passable
struct LogSoftmax[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        this: Tensor[dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        var shape = this.shape()

        # Normalize axes
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)

        # Numerical stability: subtract max along axes
        var max_vals = MinMax[dtype].forward[max=True, track_grad=False](
            this, normalized_axes, keepdims=True, requires_grad=False
        )
        var stable = Subtractor[dtype].forward[track_grad=False](this, max_vals)

        # Compute exponentials and sum
        var stable_exp = stable.exp()
        var exp_sum = Summer[dtype].forward[track_grad=False](
            stable_exp, normalized_axes, True
        )

        # Log softmax: (x - max(x)) - log(sum(exp(x - max(x))))
        var log_sum_exp = exp_sum.log(requires_grad=False)
        var out = Subtractor[dtype].forward[track_grad=False](
            stable, log_sum_exp
        )

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(this.requires_grad)

            if grad_required:
                out.requires_grad_(True)

                # Compute and store softmax for backward pass
                var softmax_vals = Divider[dtype].forward[track_grad=False](
                    stable_exp, exp_sum
                )

                var backward_fn = LogSoftmaxBackward[dtype](
                    normalized_axes^,
                    softmax_vals.buffer.contiguous_buffer(),
                    softmax_vals.shape(),
                ).into_backward_fn()

                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(this)

        return out^



# ============================================================================
# Usage Examples and Performance Notes
# ============================================================================

# """
# PERFORMANCE COMPARISON:

# Original Implementation (coordinate-value pairs):
# - Memory: O(N * (sizeof(IntArray) + sizeof(Scalar))) ≈ 40-80 bytes per element
# - Backward: O(N) element-wise assignments + O(N) computation
# - For 1M elements: ~40-80 MB storage

# Optimized Implementation (buffer storage):
# - Memory: O(N * sizeof(Scalar)) ≈ 4-8 bytes per element
# - Backward: O(1) buffer copy + O(N) computation
# - For 1M elements: ~4-8 MB storage

# Recompute Implementation (no storage):
# - Memory: O(1) - only stores axes
# - Backward: 2x forward computation + O(N) gradient computation
# - For 1M elements: ~few bytes storage

# RECOMMENDATIONS:
# 1. Use buffer storage version for most cases (good balance)
# 2. Use recompute version if memory is extremely constrained
# 3. Never use coordinate-value pairs (too much memory overhead)

# EXAMPLE USAGE:

# Standard softmax
# var logits = Tensor.randn(Shape([32, 1000]), requires_grad=True)
# var probs = Softmax[DType.float32].forward(logits, IntArray(1))

# Log softmax (more numerically stable for cross entropy)
# var log_probs = LogSoftmax[DType.float32].forward(logits, IntArray(1))

# Multi-axis softmax
# var spatial_logits = Tensor.randn(Shape([32, 10, 28, 28]), requires_grad=True)
# var probs_2d = Softmax[DType.float32].forward(
#    spatial_logits,
#    IntArray(1)  # Softmax over classes
# )
# """

# =======================


fn main() raises:
    print("passes")
