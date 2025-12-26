from tenmo import Tensor
from operators import AddTensor
from backpropagation import Delegate, BackwardFn, BACKWARD_LOG
from gradbox import Gradbox
from math import log
from sys import simd_width_of


@fieldwise_init
@register_passable
struct LogBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_LOG

    var epsilon: Scalar[dtype]  # Runtime epsilon value

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        """Compute gradient: ∂log(x)/∂x = 1/x."""
        ref grad_output = output.gradients()[]
        var parent = output.ancestry().get(0)
        ref shape = parent.shape()
        var parent_gradbox = Gradbox[Self.dtype].zeros(shape, share=False)

        if parent.is_contiguous():
            var src = parent.buffer.data_buffer().data
            var dest = parent_gradbox.buffer.data_buffer().data
            var grad_output_data = grad_output.buffer.data_buffer().data
            var offset = parent.offset()
            var numels = parent.numels()

            alias simd_width = simd_width_of[Self.dtype]()

            for i in range(0, numels - simd_width + 1, simd_width):
                var x = src.load[width=simd_width](offset + i)  # Original input
                var grad_out = grad_output_data.load[width=simd_width](i)

                # Apply epsilon: max(x, epsilon) to avoid division by zero
                var x_safe = max(x, self.epsilon)
                dest.store[width=simd_width](i, grad_out / x_safe)

            # Handle remainder
            for i in range(numels - numels % simd_width, numels):
                var x = src[offset + i]  # Original input
                var x_safe = max(x, self.epsilon)
                dest[i] = grad_output_data[i] / x_safe
        else:
            # Non-contiguous fallback
            var index = 0
            var parent_gradbox_data = parent_gradbox.buffer.data_buffer().data
            var grad_output_data = grad_output.buffer.data_buffer().data
            for coord in shape:
                var x = parent[coord]  # Original input
                var x_safe = max(x, self.epsilon)
                parent_gradbox_data[index] = grad_output_data[index] / x_safe
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
    ) -> Tensor[Self.dtype]:
        """
        Natural logarithm: y = log(x).

        Args:
            self: Input tensor.
            requires_grad: Whether to track gradients.
            epsilon: Small value to avoid log(0) and division by zero (default: 1e-12).

        Returns:
            Logarithm of input tensor.

        Note:
            Values less than epsilon are clamped to epsilon before taking log.
        """
        var shape = self.shape()
        var out = Tensor[Self.dtype].zeros(shape, requires_grad=False)

        if self.is_contiguous():
            var src = self.buffer.data_buffer().data
            var dest = out.buffer.data_buffer().data
            var offset = self.offset()
            var numels = self.numels()

            alias simd_width = simd_width_of[Self.dtype]()

            for i in range(0, numels - simd_width + 1, simd_width):
                var chunk = src.load[width=simd_width](offset + i)
                # Clamp to epsilon to avoid log(0)
                var chunk_safe = max(chunk, epsilon)
                dest.store[width=simd_width](i, log(chunk_safe))

            # Handle remainder
            for i in range(numels - numels % simd_width, numels):
                var val = src[offset + i]
                var val_safe = max(val, epsilon)
                dest[i] = log(val_safe)
        else:
            # Non-contiguous fallback
            var index = 0
            ref out_buffer = out.buffer.data_buffer()
            for idx in self.index_iterator():
                var val = self.element_at(idx)
                var val_safe = max(val, epsilon)
                out_buffer[index] = log(val_safe)
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
