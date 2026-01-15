from tenmo import Tensor
from operators import AddTensor, SqrtForwardOp, SqrtBackwardOp
from backpropagation import Delegate, BackwardFn, BACKWARD_SQRT
from gradbox import Gradbox
from math import sqrt
from ndbuffer import NDBuffer


@fieldwise_init
@register_passable
struct SqrtBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_SQRT
    var epsilon: Scalar[Self.dtype]

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var input_tensor = output.ancestry().get(0)
        ref shape = input_tensor.shape()

        var gradbox_ancestor: Gradbox[Self.dtype]

        if input_tensor.is_contiguous():
            var start = input_tensor.offset()
            var end = start + input_tensor.numels()
            # Compute 1 / (2 * sqrt(x)) - we can not use output - it may have changed
            # output is sqrt(x), so gradient is 1 / (2 * sqrt(input))
            var buffer = input_tensor.buffer.data_buffer().unary_ops[
                SqrtBackwardOp  # This should compute: 1 / (2 * sqrt(input))
            ](start, end)
            var ancestor_grad_buffer = gradbox.buffer.data_buffer() * buffer
            var ndb = NDBuffer[Self.dtype](ancestor_grad_buffer^, shape)
            gradbox_ancestor = Gradbox[Self.dtype](ndb^, share=False)
        else:
            gradbox_ancestor = Gradbox[Self.dtype].zeros(shape, share=False)
            var index = 0
            ref gradbox_buffer = gradbox.buffer.data_buffer()
            ref ancestor_gradbox_buffer = gradbox_ancestor.buffer.data_buffer()

            for coord in shape:
                # gradient = grad_output * (1 / (2 * sqrt(x)))
                var sqrt_grad = 1.0 / (
                    self.epsilon + (2.0 * input_tensor[coord])
                )
                ancestor_gradbox_buffer[index] = (
                    gradbox_buffer[index] * sqrt_grad
                )
                index += 1

        return [(input_tensor^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
@register_passable
struct Sqrt[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-12),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var out: Tensor[Self.dtype]
        ref shape = self.shape()

        if self.is_contiguous():
            var start = self.offset()
            var end = start + self.numels()
            var buffer = self.buffer.data_buffer().unary_ops[SqrtForwardOp](
                start, end
            )
            out = Tensor[Self.dtype](
                NDBuffer[Self.dtype](buffer^, shape), requires_grad=False
            )
        else:
            out = Tensor[dtype].zeros(shape, requires_grad=False)
            var index = 0
            ref out_buffer = out.buffer.data_buffer()
            for coord in shape:
                out_buffer[index] = sqrt(self[coord])
                index += 1

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                backward_fn = SqrtBackward[Self.dtype](
                    epsilon
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^

    @staticmethod
    fn forward(
        self: Gradbox[Self.dtype],
        epsilon: Scalar[Self.dtype] = Scalar[Self.dtype](1e-12),
    ) -> Gradbox[Self.dtype]:
        var out: Gradbox[Self.dtype]
        ref shape = self.shape()

        var buffer = self.buffer.data_buffer().unary_ops[SqrtForwardOp]()
        out = Gradbox[Self.dtype](
            NDBuffer[Self.dtype](buffer^, shape), share=False
        )

        return out^
