from tenmo import Tensor
from common_utils import panic
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from gradbox import Gradbox
from ancestry import Ancestor


@fieldwise_init
@register_passable
struct DotBackward[dtype: DType](ImplicitlyCopyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        gradbox = output.grad().copy()
        scalar_grad_value = gradbox.item()  # Scalar
        var grad_shares: List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]] = []
        ancestor_lhs = output.ancestry().get(0)
        ancestor_rhs = output.ancestry().get(1)

        if ancestor_lhs.requires_grad():
            tensor_rhs = ancestor_rhs.tensor()
            gradbox_lhs = Gradbox[dtype].zeros(
                ancestor_lhs.shape(), share=False
            )
            for index, value in tensor_rhs:
                gradbox_lhs[index] = value * scalar_grad_value

            grad_shares.append((ancestor_lhs.copy(), gradbox_lhs^, AddTensor))

        if ancestor_rhs.requires_grad():
            tensor_lhs = ancestor_lhs.tensor()
            gradbox_rhs = Gradbox[dtype].zeros(
                ancestor_rhs.shape(), share=False
            )
            for index, value in tensor_lhs:
                gradbox_rhs[index] = value * scalar_grad_value

            grad_shares.append((ancestor_rhs^, gradbox_rhs^, AddTensor))

        return grad_shares^


@register_passable
struct Dot[dtype: DType](Copyable):
    fn __copyinit__(out self, existing: Self):
        pass

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        lhs: Tensor[dtype],
        rhs: Tensor[dtype],
    ) -> Tensor[dtype]:
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
