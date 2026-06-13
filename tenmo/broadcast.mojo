from .tensor import Tensor
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .shapes import Shape
from .ancestry import Ancestor
from .broadcasthelper import ShapeBroadcaster
from .mnemonics import AddTensor
from .common_utils import panic
from .backpropagation import BackwardFnArg, BACKWARD_BROADCAST_TO


@fieldwise_init
struct BroadcastBackward[dtype: DType, augment: Bool, lhs_op: Int, rhs_op: Int](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref incoming_grad = output.gradients()

        var left_parent = output.ancestry().get(0)
        var right_parent = output.ancestry().get(1)

        # For left parent: compute the gradient contribution if needed
        if left_parent.requires_grad:
            var left_parent_grad: Gradbox[Self.dtype]
            comptime if Self.augment:
                left_parent_grad = Self.upstream_grad_share(
                    left_parent.buffer(),
                    right_parent.buffer(),
                    incoming_grad.buffer(),
                )
            else:
                left_parent_grad = Self.upstream_grad_shape_only(
                    left_parent.shape(),
                    incoming_grad.buffer(),
                )

            left_parent.update_grad(left_parent_grad^, Self.lhs_op, None)
            parent_ids.append(left_parent._id)

        # For right parent: compute its gradient if needed
        if right_parent.requires_grad:
            var right_parent_grad: Gradbox[Self.dtype]
            comptime if Self.augment:
                right_parent_grad = Self.upstream_grad_share(
                    right_parent.buffer(),
                    left_parent.buffer(),
                    incoming_grad.buffer(),
                )
            else:
                right_parent_grad = Self.upstream_grad_shape_only(
                    right_parent.shape(),
                    incoming_grad.buffer(),
                )

            right_parent.update_grad(right_parent_grad^, Self.rhs_op, None)
            parent_ids.append(right_parent._id)
        if not retain_graph:
            incoming_grad.zero_grad()

    @staticmethod
    def upstream_grad_shape_only(
        self_shape: Shape,
        upstream_grad: NDBuffer[Self.dtype],
    ) -> Gradbox[Self.dtype]:
        var grad_contrib: Gradbox[Self.dtype]
        if upstream_grad.shape == Shape():
            grad_contrib = Gradbox[Self.dtype].full(
                self_shape,
                upstream_grad.item(),
                device=upstream_grad.device(),
            )
        else:
            var grad_ndb = upstream_grad.copy()
            grad_contrib = Gradbox[Self.dtype](grad_ndb^)
            if grad_contrib.shape() != self_shape:
                axes = ShapeBroadcaster.broadcast_mask(
                    self_shape, grad_contrib.shape()
                ).indices_of(1)
                grad_contrib = grad_contrib.sum(axes=axes, keepdims=True)
            if grad_contrib.shape() != self_shape:
                grad_contrib = grad_contrib.reshape(self_shape)
        return grad_contrib^

    @staticmethod
    def upstream_grad_share(
        self: NDBuffer[Self.dtype],
        other: NDBuffer[Self.dtype],
        upstream_grad: NDBuffer[Self.dtype],
    ) -> Gradbox[Self.dtype]:
        var grad_contrib: Gradbox[Self.dtype]
        if upstream_grad.shape == Shape():
            grad_contrib = Gradbox[Self.dtype].full(
                self.shape,
                upstream_grad.item(),
                device=upstream_grad.device(),
            )
        else:
            var product_ndb = upstream_grad * other
            grad_contrib = Gradbox[Self.dtype](product_ndb^)

            if grad_contrib.shape() != self.shape:
                var axes = ShapeBroadcaster.broadcast_mask(
                    self.shape, grad_contrib.shape()
                ).indices_of(1)
                grad_contrib = grad_contrib.sum(axes=axes, keepdims=True)
            if grad_contrib.shape() != self.shape:
                grad_contrib = grad_contrib.reshape(self.shape)
        return grad_contrib^


@fieldwise_init
struct BroadcastToBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var parent = output.ancestry().get(0)
        ref incoming_grad = output.gradients()

        var grad: Gradbox[Self.dtype]

        if incoming_grad.shape() == parent.shape():
            # No reduction needed — shapes already match
            grad = incoming_grad
        else:
            # Sum over the axes that were broadcast to reduce back
            # to the original shape
            var axes = ShapeBroadcaster.broadcast_mask(
                parent.shape(), incoming_grad.shape()
            ).indices_of(1)
            grad = incoming_grad.sum(axes=axes, keepdims=True)

            # Reshape in case keepdims left size-1 dims that need squeezing
            if grad.shape() != parent.shape():
                grad = grad.reshape(parent.shape())

        parent.update_grad(grad^, AddTensor, None)
        parent_ids.append(parent._id)
        if not retain_graph:
            incoming_grad.zero_grad()


@fieldwise_init
struct Broadcast[dtype: DType](Copyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        target_shape: Shape,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var broadcasted_buffer = self.buffer.broadcast_to(target_shape)
        var out = Tensor[Self.dtype](broadcasted_buffer^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_BROADCAST_TO
                )
                backwardFnArg.needs_parent_data = True
                out.add_ancestry(backwardFnArg^, self)

        return out^
