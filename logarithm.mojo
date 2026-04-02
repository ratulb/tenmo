from tenmo import Tensor
from mnemonics import AddTensor, LOG
from backpropagation import Delegate, BackwardFn, BACKWARD_LOG
from gradbox import Gradbox
from ndbuffer import NDBuffer
from math import log
from sys import has_accelerator
from common_utils import panic, Epsilon
from unary_ops_kernel import UnaryOpsKernel


# ── LogBackward ───────────────────────────────────────────────────────────────


@fieldwise_init
@register_passable
struct LogBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_LOG

    var epsilon: Scalar[Self.dtype]

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    @staticmethod
    fn backward(
        parent_buffer: NDBuffer[Self.dtype],
        grad_output: NDBuffer[Self.dtype],
        epsilon: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype] where Self.dtype.is_floating_point():
        """
        Core gradient computation: grad_output / max(parent, epsilon).
        grad_output is guaranteed contiguous + zero offset (Gradbox contract).

        GPU path:
            1. parent_buffer.max(epsilon)  → scalar_ops[MAX]  — GPU safe
            2. grad_output / clamped       → arithmetic_ops[Divide] — GPU safe
        CPU contiguous fast path:
            parent_buffer.data_buffer().log_back(...)
        CPU non-contiguous fallback:
            iterate strided indices
        """
        var shape = parent_buffer.shape

        @parameter
        if has_accelerator():
            if parent_buffer.is_on_gpu():
                # Step 1: clamp parent values to epsilon — scalar_ops[MAX], GPU safe
                var clamped = parent_buffer.max(epsilon)
                # Step 2: element-wise divide — arithmetic_ops[Divide], GPU safe
                var result = grad_output / clamped
                return result^

        # CPU contiguous fast path
        if parent_buffer.is_contiguous():
            var start = parent_buffer.offset
            var end = start + parent_buffer.numels()
            var parent_data = parent_buffer.data_buffer()
            var grad_data = grad_output.data_buffer()
            var result_buffer = parent_data.log_back(
                grad_data, epsilon, start, end
            )
            return NDBuffer[Self.dtype](result_buffer^, shape)

        # CPU non-contiguous fallback
        var result = NDBuffer[Self.dtype].zeros(shape)
        var index = 0
        var result_ptr = result.data_ptr()
        var grad_ptr = grad_output.data_ptr()
        for idx in parent_buffer.index_iterator():
            var input_val = (parent_buffer.data_ptr() + idx)[]
            var input_safe = max(input_val, epsilon)
            (result_ptr + index)[] = (grad_ptr + index)[] / input_safe
            index += 1
        return result^

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        """Thin wrapper — delegates to static backward(NDBuffer)."""
        ref grad_output = output.gradients()[]
        var parent = output.ancestry().get(0)

        var result_ndb = Self.backward(
            parent.buffer, grad_output.buffer, self.epsilon
        )
        var parent_gradbox = Gradbox[Self.dtype](result_ndb^, share=False)

        return [(parent^, parent_gradbox^, AddTensor)]


# ── Logarithm ─────────────────────────────────────────────────────────────────


@fieldwise_init
@register_passable
struct Logarithm[dtype: DType]:
    @staticmethod
    fn forward[
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ](
        buffer: NDBuffer[Self.dtype],
    ) -> NDBuffer[
        Self.dtype
    ] where Self.dtype.is_floating_point():
        """
        Core log computation on NDBuffer.
        epsilon is runtime — GPU path uses max(x, epsilon) then log
        via NDBuffer scalar_ops and UnaryOpsKernel[LOG] with default epsilon=0
        (clamping already done, so no double-clamping needed).

        GPU path   : clamp via scalar_ops[MAX], then log via UnaryOpsKernel[LOG, 0]
        CPU fast   : contiguous buffer log with epsilon clamping
        CPU slow   : strided index iterator fallback
        """
        var shape = buffer.shape

        @parameter
        if has_accelerator():
            if buffer.is_on_gpu():
                try:
                    # Step 1: clamp to epsilon — scalar_ops[MAX], runtime epsilon OK
                    # var clamped = buffer.max(epsilon)
                    # Step 2: log on already-clamped buffer
                    # epsilon=0 here since clamping already done
                    # comptime zero_epsilon = Scalar[Self.dtype](0)
                    return UnaryOpsKernel[Self.dtype].launch[LOG, epsilon](
                        buffer
                    )
                except e:
                    panic("Logarithm GPU forward failed: " + e.__str__())
                    return NDBuffer[Self.dtype].Empty()  # unreachable

        # CPU contiguous fast path
        if buffer.is_contiguous():
            var start = buffer.offset
            var end = start + buffer.numels()
            var input_data = buffer.data_buffer()
            var out_buffer = input_data.log[epsilon](start, end)
            return NDBuffer[Self.dtype](out_buffer^, shape)

        # CPU non-contiguous fallback
        var result = NDBuffer[Self.dtype](shape)
        var index = 0
        var result_ptr = result.data_ptr()
        for idx in buffer.index_iterator():
            var val = (buffer.data_ptr() + idx)[]
            var val_safe = max(val, epsilon)
            (result_ptr + index)[] = log(val_safe)
            index += 1
        return result^

    @staticmethod
    fn forward[
        track_grad: Bool = True,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """
        Thin wrapper — delegates compute to forward(NDBuffer).
        Attaches autograd machinery if needed.
        """
        var result_ndb = Self.forward[epsilon](self.buffer)
        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backward_fn = LogBackward[Self.dtype](
                    epsilon
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    pass
