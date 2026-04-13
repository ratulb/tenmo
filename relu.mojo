from tenmo import Tensor
from mnemonics import AddTensor, RELU_FORWARD, RELU_BACKWARD
from backpropagation import BackwardFnArg, ArgumentType, BACKWARD_RELU
from gradbox import Gradbox
from ndbuffer import NDBuffer
from buffers import Buffer


@fieldwise_init
struct ReLUBackward[dtype: DType](ImplicitlyCopyable & Movable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var buffer = output.bwd_fn_arg().arg[Buffer[Self.dtype]]
        ref gradbox = output.gradients()[]
        var input_tensor = output.ancestry().get(0)
        ref shape = input_tensor.shape()

        var grad_buffer = gradbox.buffer.data_buffer()
        var result_buffer = grad_buffer * buffer

        var ndb = NDBuffer[Self.dtype](result_buffer^, shape)
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^, share=False)

        return [(input_tensor^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
struct ReLU[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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
                RELU_FORWARD
            ](start, end)
            var buffer = result[0]
            mask = result[1]

            out = Tensor[Self.dtype](
                NDBuffer[Self.dtype](buffer^, shape), requires_grad=False
            )
        # Slow path: non-contiguous - compute element-wise
        else:
            out = Tensor[Self.dtype].zeros(shape, requires_grad=False)
            mask = Buffer[Self.dtype](self.numels())
            ref out_buffer = out.buffer.data_buffer()
            ref self_buffer = self.buffer.data_buffer()
            var zero = Scalar[Self.dtype](0)
            var one = Scalar[Self.dtype](1)
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

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var bwd_fn_arg = BackwardFnArg[Self.dtype](BACKWARD_RELU,
                    ArgumentType[Self.dtype](mask^)
                )
                out.bwdFnArg = Optional(bwd_fn_arg^)
                out.add_ancestry(self)

        return out^

fn main() raises:
    pass
