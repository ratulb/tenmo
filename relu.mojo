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
        ref arg = output.bwd_fn_arg().arg
        ref gradbox = output.gradients()[]
        var input_tensor = output.ancestry().get(0)
        ref shape = input_tensor.shape()

        var result_ndb: NDBuffer[Self.dtype]

        if arg.isa[NDBuffer[Self.dtype]]():
            var mask_ndb = arg[NDBuffer[Self.dtype]]
            result_ndb = gradbox.buffer * mask_ndb
        else:
            var mask_buf = arg[Buffer[Self.dtype]]
            var result_buf = gradbox.buffer.data_buffer() * mask_buf
            result_ndb = NDBuffer[Self.dtype](result_buf^, shape)

        var gradbox_ancestor = Gradbox[Self.dtype](result_ndb^, share=False)
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
        Routes through NDBuffer.unary_ops_with_mask — GPU and CPU aware.

        Args:
            self: Input tensor.
            requires_grad: Override gradient tracking (default: inherit from input).

        Returns:
            Output tensor with ReLU applied.
        """

        var result = self.buffer.unary_ops_with_mask[RELU_FORWARD]()
        var out_ndb  = result[0]   # NDBuffer — output values
        var mask_ndb = result[1]   # NDBuffer — gradient mask

        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var arg: ArgumentType[Self.dtype]

                if self.buffer.is_on_gpu():
                    arg = ArgumentType[Self.dtype](mask_ndb^)
                else:
                    arg = ArgumentType[Self.dtype](mask_ndb.data_buffer())

                var bwd_fn_arg = BackwardFnArg[Self.dtype](BACKWARD_RELU, arg^)
                out.bwdFnArg = Optional(bwd_fn_arg^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    pass
