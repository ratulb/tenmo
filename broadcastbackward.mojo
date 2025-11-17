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

        # ------------------------------------------------------------
        # This is the gradient flowing *into* this broadcasted op.
        # We need to call copy explicitly because we have not annotated Gradbox with `ImplicitlyCopyable` yet - Intententionally
        # ------------------------------------------------------------
        ref incoming_grad = output.grad()


        # ------------------------------------------------------------
        # This will be returned to the engine to continue traversal.
        # capacity = 2 because we always have 2 parents(at most)
        # ------------------------------------------------------------
        var parent_grad_list = List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]](
            capacity=2
        )

        # ------------------------------------------------------------
        # Extract parents (ancestors)
        # ------------------------------------------------------------
        var left_parent  = output.ancestry().get(0)
        var right_parent = output.ancestry().get(1)

        # ------------------------------------------------------------
        # Extract actual tensors from ancestors
        # ------------------------------------------------------------
        var left_tensor  = left_parent.tensor()
        var right_tensor = right_parent.tensor()

        # ------------------------------------------------------------
        # For left parent: compute the gradient contribution if needed
        # ------------------------------------------------------------
        if left_parent.requires_grad():


            # This function reduces the broadcast grad back to left_tensor shape.
            var left_parent_grad = left_tensor.upstream_grad_share[
                augment = augment
            ](right_tensor, incoming_grad)

            # Append to return list:
            #   - which ancestor gets the update
            #   - the computed gradient box
            #   - the operation code (AddTensor/SubtractTensor/etc.)
            parent_grad_list.append(
                (left_parent^, left_parent_grad^, lhs_op)
            )

        # ------------------------------------------------------------
        # For right parent: compute its gradient if needed
        # ------------------------------------------------------------
        if right_parent.requires_grad():

            var right_parent_grad = right_tensor.upstream_grad_share[
                augment = augment
            ](left_tensor, incoming_grad)

            parent_grad_list.append(
                (right_parent^, right_parent_grad^, rhs_op)
            )

        # ------------------------------------------------------------
        # Return the list of gradient contributions for both parents
        # ------------------------------------------------------------
        return parent_grad_list^

fn main():
    pass
