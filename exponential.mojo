from tenmo import Tensor
from mnemonics import AddTensor, EXP
from backpropagation import Delegate, BackwardFn, BACKWARD_EXPONENTIAL
from gradbox import Gradbox
from ndbuffer import NDBuffer
from math import exp, log
from unary_ops_kernel import UnaryOpsKernel
from sys import has_accelerator

@fieldwise_init
@register_passable
struct ExponentialBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_EXPONENTIAL

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        # Gradient of exp: incoming grad * exp(A) = incoming grad * output
        var exp_grad = gradbox * output
        return [(parent^, exp_grad, AddTensor)]


@fieldwise_init
@register_passable
struct Exponential[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():

        var out: Tensor[Self.dtype]

        @parameter
        if has_accelerator():
            if tensor.is_on_gpu():
                try:
                    out = UnaryOpsKernel[Self.dtype].launch[EXP](
                        tensor
                    )
                except e:
                    print(e)
                    print(
                        "Exponential - GPU operation failed for opcode: ",
                        EXP,
                        ". Failling back on CPU",
                    )
                    out = Self.generate_output(tensor)

            else:
                out = Self.generate_output(tensor)

        else:
            out = Self.generate_output(tensor)

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backward_fn = ExponentialBackward[
                    Self.dtype
                ]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(tensor)

        return out^

    @staticmethod
    fn generate_output(
        ref tensor: Tensor[Self.dtype],
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        var out: Tensor[Self.dtype]

        if tensor.is_contiguous():
            var start = tensor.offset()
            var end = start + tensor.numels()
            var out_buffer = tensor.buffer.data_buffer().exp(start, end)
            var nd_buffer = NDBuffer[Self.dtype](out_buffer^, tensor.shape())
            out = Tensor[Self.dtype](nd_buffer^, requires_grad=False)

        else:
            out = Tensor[Self.dtype].zeros(tensor.shape(), requires_grad=False)
            var index = 0
            var out_ptr = out.data_ptr()
            var tensor_ptr = tensor.data_ptr()
            for idx in tensor.index_iterator():
                var input_value = (tensor_ptr + idx)[]
                (out_ptr + index)[] = log(input_value)
                index += 1

        return out^


fn main() raises:
    pass
