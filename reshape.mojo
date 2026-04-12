from tenmo import Tensor
from mnemonics import AddTensor, ZeroGrad
from backpropagation import FnArg, BACKWARD_RESHAPE
from shapes import Shape
from validators import Validator
from gradbox import Gradbox
from ndbuffer import NDBuffer
from strides import Strides


@fieldwise_init
struct ReshapeBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        var reshaped = gradbox.reshape(ancestor.shape())
        return [
            (ancestor^, reshaped^, AddTensor),
            (output, gradbox, ZeroGrad),
        ]

@fieldwise_init
struct Reshape[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        new_shape: Shape,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
    ) -> Tensor[Self.dtype]:
        var ndb = tensor.buffer.reshape(new_shape, validated)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var fn_arg = FnArg[Self.dtype].null(BACKWARD_RESHAPE)
                out.fnArg = Optional(fn_arg^)
                out.add_ancestry(tensor)

        return out^


fn main() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].arange(12, requires_grad=True).reshape(3, 4, requires_grad=True)
    #var a =aa.reshape(3, 4, requires_grad=True)
    #a.print()

    var b = a * 42
    c = b.reshape(4, 3)
    d = c.reshape(2, 2, 3)
    e = d * 5
    e.backward()
    a.grad().print()

    _="""var a_gpu = a.to_gpu()
    var a_gpu_reshaped = a_gpu.reshape(2, 2, 3)
    var res = a_gpu_reshaped * 42
    res.backward()
    a.grad().print()
    a_gpu.grad().print()
    a_gpu_reshaped.grad().print()"""

