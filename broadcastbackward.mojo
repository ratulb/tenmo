from tenmo import Tensor
from gradbox import Gradbox


@fieldwise_init
struct BroadcastBackward[
    dtype: DType, augment: Bool, lhs_op: Int, rhs_op: Int
](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        # This is the gradient flowing *into* this broadcasted op.
        # We need to call copy explicitly because we have not annotated Gradbox with `ImplicitlyCopyable` yet - Intententionally
        ref incoming_grad = output.gradients()[]

        # capacity = 2 because we always have 2 parents(at most)
        var parent_grad_list = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ](capacity=2)

        # Extract parents (ancestors)
        var left_parent = output.ancestry().get(0)
        var right_parent = output.ancestry().get(1)

        # For left parent: compute the gradient contribution if needed
        if left_parent.requires_grad:
            # This function reduces the broadcast grad back to left_tensor shape.
            var left_parent_grad = left_parent.upstream_grad_share[
                augment = Self.augment
            ](right_parent, incoming_grad)

            # Append to return list:
            #   - which ancestor gets the update
            #   - the computed gradient box
            #   - the operation code (AddTensor/SubtractTensor/etc.)
            parent_grad_list.append(
                (left_parent, left_parent_grad^, Self.lhs_op)
            )

        # For right parent: compute its gradient if needed
        if right_parent.requires_grad:
            var right_parent_grad = right_parent.upstream_grad_share[
                augment = Self.augment
            ](left_parent, incoming_grad)

            parent_grad_list.append(
                (right_parent^, right_parent_grad^, Self.rhs_op)
            )

        return parent_grad_list^
