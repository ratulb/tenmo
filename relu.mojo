from tenmo import Tensor
from operators import AddTensor, ReLUForwardOp, ReLUBackwardOp
from backpropagation import Delegate, BackwardFn, BACKWARD_RELU
from gradbox import Gradbox
from ndbuffer import NDBuffer
from buffers import Buffer


@fieldwise_init
struct ReLUBackward[dtype: DType](ImplicitlyCopyable & Movable):
    alias TAG = BACKWARD_RELU
    var mask: Buffer[Self.dtype]  # Stores 0.0 or 1.0 in same dtype

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var input_tensor = output.ancestry().get(0)
        ref shape = input_tensor.shape()

        var grad_buffer = gradbox.buffer.data_buffer()
        var result_buffer = grad_buffer * self.mask

        var ndb = NDBuffer[dtype](result_buffer^, shape)
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^, share=False)

        return [(input_tensor^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
@register_passable
struct ReLU[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Apply ReLU activation: max(0, x).

        Computes output and mask simultaneously for efficiency.

        Args:
            self: Input tensor.
            requires_grad: Override gradient tracking (default: inherit from input).

        Returns:
            Output tensor with ReLU applied.
        """
        var shape = self.shape()
        var out: Tensor[Self.dtype]
        var mask: Buffer[Self.dtype]

        # Fast path: contiguous tensor - use vectorized operations
        if self.is_contiguous():
            var start = self.offset()
            var end = start + self.numels()

            # Compute both output and mask in one pass! Returns Tuple
            var result = self.buffer.data_buffer().unary_ops_with_mask[
                ReLUForwardOp
            ](start, end)
            var buffer = result[0]
            mask = result[1]

            out = Tensor[dtype](
                NDBuffer[dtype](buffer^, shape), requires_grad=False
            )
        # Slow path: non-contiguous - compute element-wise
        else:
            out = Tensor[Self.dtype].zeros(shape, requires_grad=False)
            mask = Buffer[Self.dtype](self.numels())
            ref out_buffer = out.buffer.data_buffer()
            ref self_buffer = self.buffer.data_buffer()
            var zero = Scalar[dtype](0)
            var one = Scalar[dtype](1)
            var index = 0

            # Compute output and mask together in same loop
            for idx in self.index_iterator():
                var val = self_buffer[idx]
                if val > zero:
                    out_buffer[index] = val
                    mask[index] = one
                else:
                    out_buffer[index] = zero
                    mask[index] = zero
                index += 1

        # Setup autograd if needed
        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backward_fn = ReLUBackward[Self.dtype](
                    mask=mask^
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^
