from tenmo import Tensor
from common_utils import panic
from backpropagation import Delegate, BackwardFn, BACKWARD_DOT
from operators import AddTensor
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct DotBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_DOT

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        scalar_grad_value = gradbox.item()  # Scalar
        var grad_shares: List[Tuple[Tensor[dtype], Gradbox[dtype], Int]] = []
        tensor_lhs = output.ancestry().get(0)
        tensor_rhs = output.ancestry().get(1)

        if tensor_lhs.requires_grad:
            gradbox_lhs = Gradbox[dtype].zeros(tensor_lhs.shape(), share=False)
            for index, value in tensor_rhs:
                gradbox_lhs[index] = value * scalar_grad_value

            grad_shares.append((tensor_lhs, gradbox_lhs^, AddTensor))

        if tensor_rhs.requires_grad:
            gradbox_rhs = Gradbox[dtype].zeros(tensor_rhs.shape(), share=False)
            for index, value in tensor_lhs:
                gradbox_rhs[index] = value * scalar_grad_value

            grad_shares.append((tensor_rhs^, gradbox_rhs^, AddTensor))

        return grad_shares^


@register_passable
struct Dot[dtype: DType](Copyable):
    fn __copyinit__(out self, existing: Self):
        pass

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](lhs: Tensor[dtype], rhs: Tensor[dtype],) -> Tensor[dtype]:
        rank_lhs = lhs.rank()
        rank_rhs = rhs.rank()
        if not rank_lhs == rank_rhs and not rank_lhs <= 1:
            panic("Tensor → dot: not supported for rank > 1")
        numels_lhs = lhs.numels()
        numels_rhs = rhs.numels()
        if not numels_lhs == numels_rhs:
            panic(
                "Tensor → dot: size does not match",
                numels_lhs.__str__(),
                numels_rhs.__str__(),
            )
        scalar = lhs.buffer.contiguous_buffer().dot(
            rhs.buffer.contiguous_buffer()
        )
        out = Tensor[dtype].scalar(scalar, requires_grad=False)

        @parameter
        if track_grad:
            grad_required = lhs.requires_grad or rhs.requires_grad

            if grad_required:
                out.requires_grad_(True)
                backward_fn = DotBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(lhs, rhs)

        return out^


fn main():
    print("passes")
