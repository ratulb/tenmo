from tenmo import Tensor
from mnemonics import AddTensor, SQRT, SQRT_BACKWARD
from backpropagation import BackwardFnArg, ScalarArg, BACKWARD_SQRT
from gradbox import Gradbox
from std.math import sqrt
from ndbuffer import NDBuffer
from common_utils import Epsilon
from ancestry import Ancestor


@fieldwise_init
struct SqrtBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var epsilon = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        var parent_buffer = parent.buffer()
        ref shape = parent_buffer.shape

        var gradbox_ancestor: Gradbox[Self.dtype]

        if parent_buffer.is_contiguous():
            var start = parent_buffer.offset
            var end = start + parent_buffer.numels()
            # Compute 1 / (2 * sqrt(x)) - we can not use output - it may have changed
            # output is sqrt(x), so gradient is 1 / (2 * sqrt(input))
            var buffer = parent_buffer.data_buffer().unary_ops[
                SQRT_BACKWARD  # This should compute: 1 / (2 * sqrt(input))
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
                var sqrt_grad = 1.0 / (epsilon + (2.0 * parent_buffer[coord]))
                ancestor_gradbox_buffer[index] = (
                    gradbox_buffer[index] * sqrt_grad
                )
                index += 1

        return [(parent^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
struct Sqrt[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var out: Tensor[Self.dtype]
        ref shape = self.shape()

        if self.is_contiguous():
            var start = self.offset()
            var end = start + self.numels()
            var buffer = self.buffer.data_buffer().unary_ops[SQRT](start, end)
            out = Tensor[Self.dtype](
                NDBuffer[Self.dtype](buffer^, shape), requires_grad=False
            )
        else:
            out = Tensor[Self.dtype].zeros(shape, requires_grad=False)
            var index = 0
            ref out_buffer = out.buffer.data_buffer()
            for coord in shape:
                out_buffer[index] = sqrt(self[coord])
                index += 1

        comptime if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].scalar_arg(
                    BACKWARD_SQRT, epsilon
                )

                out.add_ancestry(backwardFnArg^, self)

        return out^

    @staticmethod
    fn forward(
        self: Gradbox[Self.dtype],
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
    ) -> Gradbox[Self.dtype]:
        var out: Gradbox[Self.dtype]
        ref shape = self.shape()

        var buffer = self.buffer.data_buffer().unary_ops[SQRT]()
        out = Gradbox[Self.dtype](
            NDBuffer[Self.dtype](buffer^, shape), share=False
        )

        return out^


fn main() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].ones(10, requires_grad=True)
    var b = a.sqrt()
    b.backward()
    a.grad().print()
    s = a.std()
    s.backward()
    a.grad().print()
