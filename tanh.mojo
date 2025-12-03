from tenmo import Tensor
from operators import AddTensor, TanhForwardOp, TanhBackwardOp
from backpropagation import Delegate, BackwardFn, BACKWARD_TANH
from ancestry import Ancestor
from gradbox import Gradbox
from math import tanh, exp
from ndbuffer import NDBuffer


@fieldwise_init
@register_passable
struct TanhBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_TANH
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
            var buffer = input_tensor.buffer.data_buffer().unary_ops[
                TanhBackwardOp
            ](start, end)
            var ancestor_grad_buffer = gradbox.buffer.data_buffer() * buffer
            var ndb = NDBuffer[dtype](ancestor_grad_buffer^, shape)
            gradbox_ancestor = Gradbox[dtype](ndb^, share=False)

        else:
            gradbox_ancestor = Gradbox[dtype].zeros(shape, share=False)
            var index = 0
            ref gradbox_buffer = gradbox.buffer.data_buffer()
            ref ancestor_gradbox_buffer = gradbox_ancestor.buffer.data_buffer()
            for coord in shape:
                var tanh_value = 1 - Tanh[dtype].tanh_stable(input_tensor[coord]) ** 2
                ancestor_gradbox_buffer[index] = (
                    gradbox_buffer[index] * tanh_value
                )
                index += 1

        return [(ancestor^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
@register_passable
struct Tanh[dtype: DType]:
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
            var buffer = self.buffer.data_buffer().unary_ops[TanhForwardOp](
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
                out_buffer[index] = Self.tanh_stable(self[coord])
                index += 1

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                backward_fn = TanhBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^

    @staticmethod
    fn tanh_stable(x: Scalar[dtype]) -> Scalar[dtype]:
        """
        More numerically stable tanh implementation.
        """
        if x > 0:
            return (1 - exp(-2*x)) / (1 + exp(-2*x))
        else:
            return (exp(2*x) - 1) / (exp(2*x) + 1)



fn main() raises:
    #a = Tensor.arange(10, requires_grad=True)
    alias dtype = DType.float32
    a = Tensor[dtype].arange([1.0, 0.4199743, 0.070650816, 0.009866101, 0.0013411314, 0.0001816667, 2.479538e-05, 3.3378574e-06, 4.768371e-07, 0.0], requires_grad=True)

    c = a.tanh()
    c.backward()
    a.grad().print()
    print("passes")

    ll = List[Scalar[dtype]](1.0, 0.4199743, 0.070650816, 0.009866101, 0.0013411314, 0.0001816667, 2.479538e-05, 3.3378574e-06, 4.768371e-07, 0.0)

    for i in range(len(ll)):
        print(tanh(ll[i]), Tanh[dtype].tanh_stable(ll[i]))
