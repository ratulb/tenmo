from tenmo import Tensor
from mnemonics import AddTensor
from backpropagation import Delegate, BackwardFn, BACKWARD_LOG
from gradbox import Gradbox
from ndbuffer import NDBuffer
from math import log


@fieldwise_init
@register_passable
struct LogBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_LOG

    var epsilon: Scalar[Self.dtype]  # Runtime epsilon value

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        """Compute gradient: ∂log(x)/∂x = 1/x."""
        ref grad_output = output.gradients()[]
        var parent = output.ancestry().get(0)
        ref shape = parent.shape()
        var parent_gradbox: Gradbox[Self.dtype]

        if parent.is_contiguous():
            var start = parent.offset()
            var end = start + parent.numels()
            var parent_buffer = parent.buffer.data_buffer()
            var grad_output_buffer = grad_output.buffer.data_buffer()
            var grad_box_buffer = parent_buffer.log_back(
                grad_output_buffer, self.epsilon, start, end
            )
            var nd_buffer = NDBuffer[Self.dtype](grad_box_buffer^, shape)
            parent_gradbox = Gradbox[Self.dtype](nd_buffer^, share=False)

        else:
            # Non-contiguous fallback
            parent_gradbox = Gradbox[Self.dtype].zeros(shape, share=False)
            var index = 0
            var parent_gradbox_data = parent_gradbox.data_ptr()
            var grad_output_data = grad_output.data_ptr()
            for idx in parent.index_iterator():
                var input_value = parent.element_at(idx)  # Original input
                var input_value_safe = max(input_value, self.epsilon)
                parent_gradbox_data[index] = (
                    grad_output_data[index] / input_value_safe
                )
                index += 1

        return [(parent^, parent_gradbox^, AddTensor)]


@fieldwise_init
@register_passable
struct Logarithm[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        epsilon: Scalar[Self.dtype] = 1e-12,  # Default epsilon
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """
        Note:
            Values less than epsilon are clamped to epsilon before taking log.
        """
        var shape = self.shape()
        var out: Tensor[Self.dtype]

        if self.is_contiguous():
            var start = self.offset()
            var end = start + self.numels()
            var input_buffer = self.buffer.data_buffer()
            var out_buffer = input_buffer.log(start, end, epsilon)
            var nd_buffer = NDBuffer[Self.dtype](out_buffer^, shape)
            out = Tensor[Self.dtype](nd_buffer^, requires_grad=False)

        else:
            # Non-contiguous fallback
            out = Tensor[Self.dtype].zeros(shape, requires_grad=False)
            var index = 0
            var out_ptr = out.data_ptr()
            var self_ptr = self.data_ptr()
            for idx in self.index_iterator():
                var input_value = (self_ptr + idx)[]
                var input_value_safe = max(input_value, epsilon)
                (out_ptr + index)[] = log(input_value_safe)
                index += 1

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
