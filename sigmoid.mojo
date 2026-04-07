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
    #] where Self.dtype.is_floating_point():
    ]:
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        #ref shape = input_tensor.shape()
        var ndb = output.buffer.arithmetic_ops[SIGMOID_BACKWARD](gradbox.buffer)
        var gradbox_ancestor = Gradbox[Self.dtype](ndb^, share=False)


        _="""if input_tensor.is_contiguous():
            var output_buffer = output.buffer.data_buffer()
            var ancestor_grad_buffer = gradbox.buffer.data_buffer() * (
                output_buffer * (Scalar[Self.dtype](1) - output_buffer)
            )
            var ndb = NDBuffer[Self.dtype](ancestor_grad_buffer^, shape)
            gradbox_ancestor = Gradbox[Self.dtype](ndb^, share=False)
        else:
            gradbox_ancestor = Gradbox[Self.dtype].zeros(shape, share=False)
            var index = 0
            ref gradbox_buffer = gradbox.buffer.data_buffer()
            ref ancestor_gradbox_buffer = gradbox_ancestor.buffer.data_buffer()
            for coord in output.index_iterator():
                var sigmoid_value = output[coord]
                ancestor_gradbox_buffer[index] = (
                    gradbox_buffer[index] * sigmoid_value * (Scalar[Self.dtype](1) - sigmoid_value)
                )
                index += 1"""
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
        _="""ref shape = self.shape()
        if self.is_contiguous():
            var start = self.offset()
            var end = start + self.numels()
            var buffer = self.buffer.data_buffer().sigmoid(start, end)
            out = Tensor[Self.dtype](
                NDBuffer[Self.dtype](buffer^, shape), requires_grad=False
            )
        else:
            out = Tensor[Self.dtype].zeros(shape, requires_grad=False)
            var index = 0
            ref out_buffer = out.buffer.data_buffer()
            for idx in self.index_iterator():
                var sigmoid_value = Scalar[Self.dtype](1) / (Scalar[Self.dtype](1) + exp(-self.buffer.get(idx)))
                out_buffer[index] = sigmoid_value
                index += 1"""
        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backward_fn = SigmoidBackward[Self.dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)
        return out^

