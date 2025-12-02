from tenmo import Tensor
from operators import AddTensor, ReLUForwardOp, ReLUBackwardOp
from backpropagation import Delegate, BackwardFn
from ancestry import Ancestor
from gradbox import Gradbox
from ndbuffer import NDBuffer


@fieldwise_init
@register_passable
struct ReLUBackward[dtype: DType](ImplicitlyCopyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        ref input_tensor = ancestor.tensor()
        ref shape = ancestor.shape()
        var gradbox_ancestor: Gradbox[dtype]
        var zero = Scalar[dtype](0)
        if input_tensor.is_contiguous():
            var start = input_tensor.offset()
            var end = start + input_tensor.numels()
            var grad_buffer = input_tensor.buffer.data_buffer().select[
                ReLUBackwardOp
            ](gradbox.buffer.data_buffer(), start, end)
            var ndb = NDBuffer[dtype](grad_buffer^, shape)
            gradbox_ancestor = Gradbox[dtype](ndb^, share=False)
        else:
            gradbox_ancestor = Gradbox[dtype].zeros(shape, share=False)
            ref gradbox_buffer = gradbox.buffer.data_buffer()
            var index = 0
            for coord in shape:
                if input_tensor[coord] > zero:
                    gradbox_ancestor[coord] = gradbox_buffer[index]

        return [(ancestor^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
@register_passable
struct ReLU[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        dtype
    ]:
        shape = self.shape()
        var out: Tensor[dtype]
        if self.is_contiguous():
            var start = self.offset()
            var end = start + self.numels()
            var buffer = self.buffer.data_buffer().unary_ops[ReLUForwardOp](
                start, end
            )
            out = Tensor[dtype](
                NDBuffer[dtype](buffer^, shape), requires_grad=False
            )
        else:
            out = Tensor[dtype].zeros(shape, requires_grad=False)
            ref out_buffer = out.buffer.data_buffer()
            var index = 0
            zero = Scalar[dtype](0)
            for coord in shape:
                out_buffer[index] = self[coord] if self[coord] > zero else zero
                index += 1

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                backward_fn = ReLUBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    print("passes")
