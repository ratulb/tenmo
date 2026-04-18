from tenmo import Tensor
from mnemonics import AddTensor, TANH_BACKWARD
from backpropagation import BackwardFnArg, BACKWARD_TANH
from gradbox import Gradbox
from ancestors_newest import AncestorRef

@fieldwise_init
struct TanhBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Tensor[Self.dtype],
    ) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        ref gradbox = output.gradients()[]
        ref parent = output.ancestry().get(0)
        var ndb = output.buffer.arithmetic_ops[TANH_BACKWARD](gradbox.buffer)
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^, share=False)

        return [(parent, gradbox_ancestor^, AddTensor)]

    @staticmethod
    fn backward(
        output: AncestorRef[Self.dtype],
    ) -> List[
        Tuple[AncestorRef[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        ref gradbox = output.gradients()[]
        ref parent = output.ancestry().get(0)
        var ndb = output.buffer().arithmetic_ops[TANH_BACKWARD](gradbox.buffer)
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^, share=False)

        return [(parent, gradbox_ancestor^, AddTensor)]

@fieldwise_init
struct Tanh[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var ndb = self.buffer.tanh()
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg =
                    BackwardFnArg[Self.dtype].null_arg(BACKWARD_TANH)
                out.add_ancestry(backwardFnArg^, self)

        return out^


fn main() raises:
    print("passes")
