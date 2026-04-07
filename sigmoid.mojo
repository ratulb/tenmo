from tenmo import Tensor
from mnemonics import AddTensor, SIGMOID_BACKWARD
from backpropagation import Delegate, BackwardFn, BACKWARD_SIGMOID
from gradbox import Gradbox
from std.math import exp
from ndbuffer import NDBuffer

@fieldwise_init
struct SigmoidBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    comptime TAG = BACKWARD_SIGMOID
    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)
    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ]:
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        var ndb = output.buffer.arithmetic_ops[SIGMOID_BACKWARD](gradbox.buffer)
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^, share=False)

        return [(parent^, gradbox_ancestor^, AddTensor)]

struct Sigmoid[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var ndb = self.buffer.sigmoid()
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)
        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backward_fn = SigmoidBackward[Self.dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)
        return out^

