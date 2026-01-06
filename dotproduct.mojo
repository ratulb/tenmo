from tenmo import Tensor
from common_utils import panic
from backpropagation import Delegate, BackwardFn, BACKWARD_DOT
from operators import AddTensor
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct DotBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_DOT

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var scalar_grad_value = gradbox.item()  # Scalar
        var grad_shares: List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ] = []
        var tensor_lhs = output.ancestry().get(0)
        var tensor_rhs = output.ancestry().get(1)

        if tensor_lhs.requires_grad:
            var grad_tensor = tensor_rhs.__mul__[track_grad=False](
                scalar_grad_value
            )
            var gradbox_lhs = grad_tensor.as_gradbox(
                share=False, contiguous=False
            )
            grad_shares.append((tensor_lhs, gradbox_lhs^, AddTensor))

        if tensor_rhs.requires_grad:
            var grad_tensor = tensor_lhs.__mul__[track_grad=False](
                scalar_grad_value
            )
            var gradbox_rhs = grad_tensor.as_gradbox(
                share=False, contiguous=False
            )

            grad_shares.append((tensor_rhs^, gradbox_rhs^, AddTensor))

        return grad_shares^


@fieldwise_init
@register_passable
struct Dot[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](lhs: Tensor[Self.dtype], rhs: Tensor[Self.dtype],) -> Tensor[Self.dtype]:
        rank_lhs = lhs.rank()
        rank_rhs = rhs.rank()
        if not rank_lhs == rank_rhs and not rank_lhs <= 1:
            panic("Tensor → dot: not supported for rank > 1")
        var numels_lhs = lhs.numels()
        var numels_rhs = rhs.numels()
        if not numels_lhs == numels_rhs:
            panic(
                "Tensor → dot: size does not match",
                numels_lhs.__str__(),
                numels_rhs.__str__(),
            )

        var scalar = lhs.buffer.contiguous_buffer().dot(
            rhs.buffer.contiguous_buffer()
        )
        var out = Tensor[Self.dtype].scalar(scalar, requires_grad=False)

        @parameter
        if track_grad:
            grad_required = lhs.requires_grad or rhs.requires_grad

            if grad_required:
                out.requires_grad_(True)
                backward_fn = DotBackward[Self.dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(lhs, rhs)

        return out^
