from tenmo import Tensor
from mnemonics import AddTensor, RELU_FORWARD, RELU_BACKWARD
from backpropagation import BackwardFnArg, BufferArg, NDBufferArg, BACKWARD_RELU
from gradbox import Gradbox
from ndbuffer import NDBuffer
from buffers import Buffer
from ancestors_newest import AncestorRef


@fieldwise_init
struct ReLUBackward[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    fn backward(
        output: Tensor[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref arg = output.backward_fn_arg()
        ref gradbox = output.gradients()[]
        var input_tensor = output.ancestry().get(0)
        ref shape = input_tensor.shape()

        var result_ndb: NDBuffer[Self.dtype]

        if output.is_on_gpu():
            var mask_ndb = arg.get[NDBufferArg[Self.dtype]]().ndb
            result_ndb = gradbox.buffer * mask_ndb
        else:
            var mask_buf = arg.get[BufferArg[Self.dtype]]().buffer
            var result_buf = gradbox.buffer.data_buffer() * mask_buf
            result_ndb = NDBuffer[Self.dtype](result_buf^, shape)

        var gradbox_ancestor = Gradbox[Self.dtype](result_ndb^, share=False)
        return [(input_tensor^, gradbox_ancestor^, AddTensor)]

    @staticmethod
    fn backward(
        output: AncestorRef[Self.dtype],
    ) -> List[Tuple[AncestorRef[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref arg = output.backward_fn_arg()
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        ref shape = ancestor.shape()

        var result_ndb: NDBuffer[Self.dtype]

        if gradbox.is_on_gpu():
            var mask_ndb = arg.get[NDBufferArg[Self.dtype]]().ndb
            result_ndb = gradbox.buffer * mask_ndb
        else:
            var mask_buf = arg.get[BufferArg[Self.dtype]]().buffer
            var result_buf = gradbox.buffer.data_buffer() * mask_buf
            result_ndb = NDBuffer[Self.dtype](result_buf^, shape)

        var ancestor_gbx = Gradbox[Self.dtype](result_ndb^, share=False)
        return [(ancestor^, ancestor_gbx^, AddTensor)]


@fieldwise_init
struct ReLU[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Apply ReLU activation: max(0, x).

        Computes output and mask simultaneously for efficiency.
        Routes through NDBuffer.unary_ops_with_mask — GPU and CPU aware.

        Args:
            self: Input tensor.
            requires_grad: Override gradient tracking (default: inherit from input).

        Returns:
            Output tensor with ReLU applied.
        """

        var result = self.buffer.unary_ops_with_mask[RELU_FORWARD]()
        var out_ndb = result[0]  # NDBuffer — output values
        var mask_ndb = result[1]  # NDBuffer — gradient mask

        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg: BackwardFnArg[Self.dtype]

                if self.buffer.is_on_gpu():
                    backwardFnArg = BackwardFnArg[Self.dtype].from_ndbuffer(
                        BACKWARD_RELU, mask_ndb^
                    )
                else:
                    backwardFnArg = BackwardFnArg[Self.dtype].from_buffer(
                        BACKWARD_RELU, mask_ndb.data_buffer()
                    )

                out.add_ancestry(backwardFnArg^, self)

        return out^


fn main() raises:
    pass
