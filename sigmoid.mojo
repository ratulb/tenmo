from tenmo import Tensor
from mnemonics import AddTensor, SIGMOID_BACKWARD
from backpropagation import BackwardFnArg, BACKWARD_SIGMOID
from gradbox import Gradbox
from ancestry import Ancestor


@fieldwise_init
struct SigmoidBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        var ndb = output.buffer().arithmetic_ops[SIGMOID_BACKWARD](
            gradbox.buffer
        )
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^, share=False)

        return [(parent, gradbox_ancestor^, AddTensor)]


struct Sigmoid[dtype: DType](ImplicitlyCopyable, RegisterPassable):
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
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_SIGMOID
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^


fn main() raises:
    print("passes")
