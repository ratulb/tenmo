from tenmo import Tensor
from backpropagation import Delegate, BackwardFn
from gradbox import Gradbox
from ancestry import Ancestor


@fieldwise_init
@register_passable
struct BroadcastBackward[dtype: DType, augment: Bool, lhs_op: Int, rhs_op: Int](
    ImplicitlyCopyable
):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var gradbox = output.grad().copy()
        var grad_shares = List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]](
            capacity=2
        )

        var ancestor_lhs = output.ancestry().get(0)
        var ancestor_rhs = output.ancestry().get(1)

        var tensor_lhs = ancestor_lhs.tensor()
        var tensor_rhs = ancestor_rhs.tensor()

        if ancestor_lhs.requires_grad():
            var lhs_share = tensor_lhs.upstream_grad_share[augment=augment](
                tensor_rhs, gradbox
            )
            grad_shares.append((ancestor_lhs^, lhs_share^, lhs_op))

        if ancestor_rhs.requires_grad():
            var rhs_share = tensor_rhs.upstream_grad_share[augment=augment](
                tensor_lhs, gradbox
            )
            grad_shares.append((ancestor_rhs^, rhs_share^, rhs_op))

        return grad_shares^


fn main():
    pass
