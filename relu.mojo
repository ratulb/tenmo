from tenmo import Tensor
from mnemonics import AddTensor, RELU_FORWARD, RELU_BACKWARD
from backpropagation import Delegate, BackwardFn, BACKWARD_RELU
from gradbox import Gradbox
from ndbuffer import NDBuffer


@fieldwise_init
struct ReLUBackward[dtype: DType](ImplicitlyCopyable & Movable):
    comptime TAG = BACKWARD_RELU
    var mask_ndb: NDBuffer[Self.dtype]  # Stores 0.0 or 1.0 in same Self.dtype

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var input_tensor = output.ancestry().get(0)

        var ndb = gradbox.buffer * self.mask_ndb
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
        Routes through NDBuffer.unary_ops_with_mask — GPU and CPU aware.

        Args:
            self: Input tensor.
            requires_grad: Override gradient tracking (default: inherit from input).

        Returns:
            Output tensor with ReLU applied.
        """

        var (out_ndb, mask_ndb) = self.buffer.unary_ops_with_mask[RELU_FORWARD]()

        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backward_fn = ReLUBackward[Self.dtype](
                    mask_ndb=mask_ndb^
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^
