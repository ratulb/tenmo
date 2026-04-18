from tenmo import Tensor
from intarray import IntArray
from mnemonics import AddTensor
from shapes import Shape
from backpropagation import BackwardFnArg, ReductionArg, BACKWARD_MEAN
from validators import Validator
from gradbox import Gradbox
from common_utils import panic
from ancestors_newest import AncestorRef

struct MeanBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):

    @staticmethod
    fn backward(
        output: AncestorRef[Self.dtype],
    ) -> List[Tuple[AncestorRef[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = output.backward_fn_arg().get[ReductionArg]()
        ref gradbox = output.gradients()[]
        ref gradbox_shape = gradbox.shape()
        var ancestor = output.ancestry().get(0)
        ref ancestor_shape = ancestor.buffer().shape
        if gradbox_shape == Shape():
            scalar_grad = gradbox.item() / Scalar[Self.dtype](
                ancestor_shape.num_elements()
            )
            var grad_contrib: Gradbox[Self.dtype]
            grad_contrib = Gradbox[Self.dtype].full(
                ancestor_shape,
                scalar_grad,
                share=False,
                device=gradbox.device(),
            )
            return [
                (
                    ancestor^,
                    grad_contrib^,
                    AddTensor,
                )
            ]

        var expanded = gradbox.copy()

        if not bwd_arg.keepdims:
            expanded = expanded.reshape(
                Shape(
                    gradbox_shape.intarray().insert(
                        bwd_arg.axes,
                        IntArray.filled(len(bwd_arg.axes), 1),
                    )
                )
            )

        # Broadcast and divide
        var broadcasted = expanded.broadcast_to(ancestor_shape)
        # Compute total count of elements being reduced
        var count = ancestor_shape.reduced_shape(bwd_arg.axes).product()
        count = count if count > 0 else 1
        var average = broadcasted / Scalar[Self.dtype](count)

        return [
            (
                ancestor^,
                average^,
                AddTensor,
            )
        ]

@fieldwise_init
struct Mean[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        normalized_axes = Validator.validate_and_normalize_axes(
            tensor.shape(), axes
        )
        var ndb = tensor.buffer.reduce[mean=True](normalized_axes, keepdims)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg =  BackwardFnArg[Self.dtype](
                        BACKWARD_MEAN, ReductionArg(normalized_axes, keepdims)

                )
                out.add_ancestry(backwardFnArg^, tensor)

        return out^

    @always_inline
    @staticmethod
    fn forward(
        gradbox: Gradbox[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
    ) -> Gradbox[Self.dtype]:
        var gradbox_shape = gradbox.shape()
        normalized_axes = Validator.validate_and_normalize_axes(
            gradbox_shape, axes
        )
        var ndb = gradbox.buffer.reduce[mean=True](normalized_axes, keepdims)
        var out = Gradbox[Self.dtype](ndb^, share=False)

        return out^


from std.testing import assert_true


fn main() raises:
    var A = Tensor[DType.float32].arange(5, 50, requires_grad=True)
    var B = A.reshape(5, 9, requires_grad=True)
    s = B.mean(IntArray(1), keepdims=True)
    s.backward()

    A.grad().print(num_first=1000, num_last=1000)
    _ = """a_gpu = B.to_gpu()
    b_cpu = a_gpu.reshape(3, 15)
    c_gpu = b_cpu + 10
    d_gpu = c_gpu * 42

    gpu_mean = d_gpu.mean(keepdims=True)
    gpu_mean.print()

    s = gpu_mean.sum()
    s.backward()"""
