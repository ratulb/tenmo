from tenmo import Tensor
from operators import AddTensor, SigmoidOp
from backpropagation import Delegate, BackwardFn, BACKWARD_SIGMOID
from ancestry import Ancestor
from gradbox import Gradbox
from math import exp
from ndbuffer import NDBuffer

@fieldwise_init
@register_passable
struct SigmoidBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_SIGMOID
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        ref input_tensor = ancestor.tensor()
        ref shape = ancestor.shape()
        var gradbox_ancestor: Gradbox[dtype]

        if input_tensor.is_contiguous():
            var start = input_tensor.offset()
            var end = start + input_tensor.numels()
            var buffer = input_tensor.buffer.data_buffer().unary_ops[SigmoidOp](
                start, end
            )
            var ancestor_grad_buffer = gradbox.buffer.data_buffer() * (
                buffer * (1 - buffer)
            )  # These would get vectorized
            var ndb = NDBuffer[dtype](ancestor_grad_buffer^, shape)
            gradbox_ancestor = Gradbox[dtype](ndb^, share=False)

        else:
            gradbox_ancestor = Gradbox[dtype].zeros(shape, share=False)
            var index = 0
            ref gradbox_buffer = gradbox.buffer.data_buffer()
            ref ancestor_gradbox_buffer = gradbox_ancestor.buffer.data_buffer()
            for coord in shape:
                var sigmoid_value = 1 / (1 + exp(-input_tensor[coord]))
                ancestor_gradbox_buffer[index] = (
                    gradbox_buffer[index] * sigmoid_value * (1 - sigmoid_value)
                )
                index += 1
        return [(ancestor^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
@register_passable
struct Sigmoid[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        dtype
    ]:
        var out: Tensor[dtype]
        ref shape = self.shape()
        if self.is_contiguous():
            var start = self.offset()
            var end = start + self.numels()
            var buffer = self.buffer.data_buffer().unary_ops[SigmoidOp](
                start, end
            )
            out = Tensor[dtype](
                NDBuffer[dtype](buffer^, shape), requires_grad=False
            )

        else:
            out = Tensor[dtype].zeros(shape, requires_grad=False)
            var index = 0
            ref out_buffer = out.buffer.data_buffer()
            for coord in shape:
                sigmoid_value = 1 / (1 + exp(-self[coord]))
                out_buffer[index] = sigmoid_value
                index += 1

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                backward_fn = SigmoidBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    print("passes")
