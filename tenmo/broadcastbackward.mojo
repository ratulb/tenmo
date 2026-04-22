from tensor import Tensor
from gradbox import Gradbox
from ndbuffer import NDBuffer
from shapes import Shape
from ancestry import Ancestor
from broadcasthelper import ShapeBroadcaster


@fieldwise_init
struct BroadcastBackward[dtype: DType, augment: Bool, lhs_op: Int, rhs_op: Int](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        # This is the gradient flowing *into* this broadcasted op.
        # We need to call copy explicitly because we have not annotated Gradbox with `ImplicitlyCopyable` yet - Intententionally
        ref incoming_grad = output.gradbox[]

        # capacity = 2 because we always have 2 parents(at most)
        var parent_grad_list = List[
            Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]
        ](capacity=2)

        # Extract parents (ancestors)
        var left_parent = output.ancestry().get(0)
        var right_parent = output.ancestry().get(1)

        # For left parent: compute the gradient contribution if needed
        if left_parent.requires_grad:
            # This function reduces the broadcast grad back to left_tensor shape.
            var left_parent_grad = Self.upstream_grad_share(
                left_parent.buffer(),
                right_parent.buffer(),
                incoming_grad.buffer,
            )

            # Append to return list:
            #   - which ancestor gets the update
            #   - the computed gradient box
            #   - the operation code (AddTensor/SubtractTensor/etc.)
            parent_grad_list.append(
                (left_parent, left_parent_grad^, Self.lhs_op)
            )

        # For right parent: compute its gradient if needed
        if right_parent.requires_grad:
            var right_parent_grad = Self.upstream_grad_share(
                right_parent.buffer(),
                left_parent.buffer(),
                incoming_grad.buffer,
            )

            parent_grad_list.append(
                (right_parent^, right_parent_grad^, Self.rhs_op)
            )

        return parent_grad_list^

    @staticmethod
    fn upstream_grad_share(
        self: NDBuffer[Self.dtype],
        other: NDBuffer[Self.dtype],
        upstream_grad: NDBuffer[Self.dtype],
    ) -> Gradbox[Self.dtype]:
        var grad_contrib: Gradbox[Self.dtype]
        if upstream_grad.shape == Shape():
            grad_contrib = Gradbox[Self.dtype].full(
                self.shape,
                upstream_grad.item(),
                share=False,
                device=upstream_grad.device(),
            )
        else:
            comptime if Self.augment:
                grad_contrib = Gradbox[Self.dtype](upstream_grad * other)
            else:
                grad_contrib = Gradbox[Self.dtype](upstream_grad)

            if grad_contrib.shape() != self.shape:
                axes = ShapeBroadcaster.broadcast_mask(
                    self.shape, grad_contrib.shape()
                ).indices_of(1)
                grad_contrib = grad_contrib.sum(axes=axes, keepdims=True)
            if grad_contrib.shape() != self.shape:
                grad_contrib = grad_contrib.reshape(self.shape)

        return grad_contrib^


fn main() raises:
    print("passes")
