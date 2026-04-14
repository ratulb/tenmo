from tenmo import Tensor
from intarray import IntArray
from mnemonics import AddTensor
from shapes import Shape
from backpropagation import Delegate, BackwardFn, BACKWARD_MEAN
from validators import Validator
from gradbox import Gradbox
from common_utils import panic


struct MeanBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    comptime TAG = BACKWARD_MEAN
    var axes: IntArray
    var keepdims: Bool

    fn __init__(out self, axes: IntArray = IntArray(), keepdims: Bool = False):
        self.axes = axes
        self.keepdims = keepdims

    fn __copyinit__(out self, copy: Self):
        self.axes = copy.axes.copy()
        self.keepdims = copy.keepdims

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var gradbox_shape = gradbox.shape()
        var ancestor = output.ancestry().get(0)
        if gradbox_shape == Shape():
            scalar_grad = gradbox.item() / Scalar[Self.dtype](ancestor.shape().num_elements())
            var grad_contrib: Gradbox[Self.dtype]
            grad_contrib = Gradbox[Self.dtype].full(
                ancestor.shape(),
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

        if not self.keepdims:
            expanded = expanded.reshape(
                Shape(
                    gradbox_shape.intarray().insert(
                        self.axes,
                        IntArray.filled(len(self.axes), 1),
                    )
                )
            )

        # Broadcast and divide
        var broadcasted = expanded.broadcast_to(ancestor.shape())
        # Compute total count of elements being reduced
        var count = ancestor.shape().reduced_shape(self.axes).product()
        count = count if count > 0 else 1
        var average = broadcasted / Scalar[Self.dtype](count)

        return [
            (
                ancestor^,
                average^,
                AddTensor,
            )
        ]

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
struct Mean[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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
                backward_fn = MeanBackward[Self.dtype](
                    normalized_axes.copy(), keepdims
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(tensor)

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
    A = Tensor[DType.float32].arange(5, 50, requires_grad=True)
    B = A.reshape(5, 9, requires_grad=True)
    a_gpu = B.to_gpu()
    b_cpu = a_gpu.reshape(3, 15)
    c_gpu = b_cpu + 10
    d_gpu = c_gpu * 42

    gpu_mean = d_gpu.mean(keepdims=True)
    gpu_mean.print()

    s = gpu_mean.sum()
    s.backward()

    A.grad().print()
