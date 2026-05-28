from .tensor import Tensor
from .backpropagation import (
    BackwardFnArg,
    ScalarArg,
    BACKWARD_DIVIDE,
    BACKWARD_DIV_SCALAR,
    BACKWARD_RIGHT_DIV_SCALAR,
)
from tenmo.mnemonics import AddTensor, SubtractTensor, Divide, ReverseDivide
from tenmo.common_utils import panic
from tenmo.gradbox import Gradbox
from tenmo.ancestry import Ancestor


@fieldwise_init
struct TrueDivBackwardScalar[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var scalar = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        ref gradbox = output.gradients()[]
        ancestor = output.ancestry().get(0)
        # ∂(x / s)/∂x = 1/s → incoming_grad / scalar
        var divided = gradbox / scalar
        ancestor.update_grad(divided^, AddTensor, None)
        parent_ids.append(ancestor._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct RightTrueDivBackwardScalar[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var scalar = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        var gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        # Fused: -s * grad_output / x² in one pass
        var grad_ndb = gradbox.buffer.rdiv_scalar_backward(
            ancestor.buffer(), scalar
        )
        var grad_parent = Gradbox[Self.dtype](grad_ndb^, share=False)
        ancestor.update_grad(grad_parent^, SubtractTensor, None)
        parent_ids.append(ancestor._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct DivideBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref gradbox = output.gradients()[]
        var ancestor_top = output.ancestry().get(0)
        var ancestor_bottom = output.ancestry().get(1)
        var buffer_top = ancestor_top.buffer()
        var buffer_bottom = ancestor_bottom.buffer()

        # Fused: both gradients in one pass
        var (grad_num, grad_den) = gradbox.buffer.divide_backward(
            buffer_top, buffer_bottom
        )

        if ancestor_top.requires_grad:
            if grad_num.shape != buffer_top.shape:
                grad_num = grad_num.sum_over_broadcasted_axes(buffer_top.shape)
            ancestor_top.update_grad(
                Gradbox[Self.dtype](grad_num^, share=False),
                AddTensor,
                None,
            )
            parent_ids.append(ancestor_top._id)

        if ancestor_bottom.requires_grad:
            if grad_den.shape != buffer_bottom.shape:
                grad_den = grad_den.sum_over_broadcasted_axes(
                    buffer_bottom.shape
                )
            ancestor_bottom.update_grad(
                Gradbox[Self.dtype](grad_den^, share=False),
                SubtractTensor,
                None,
            )
            parent_ids.append(ancestor_bottom._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct DivideScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        comptime assert (
            Self.dtype.is_numeric()
        ), "Tensor → __rtruediv__ is for numeric data types only"

        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[ReverseDivide](scalar), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].scalar_arg(
                    BACKWARD_RIGHT_DIV_SCALAR, scalar
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^


@fieldwise_init
struct DivideByScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        comptime assert (
            Self.dtype.is_numeric()
        ), "Tensor → __truediv__ is for numeric data types only"

        if scalar == Scalar[Self.dtype](0):
            panic("Tensor → __truediv__ : canot divide by " + String(scalar))

        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[Divide](scalar), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].scalar_arg(
                    BACKWARD_DIV_SCALAR, scalar
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^


# Element wise division of two tensors
@fieldwise_init
struct Divider[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], other: Tensor[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        if not self.broadcastable(other):
            panic(
                "Tensor division dimension mismatch: cannot broadcast shape "
                + String(self.shape())
                + " with "
                + String(other.shape()),
                "at Divider → forward",
            )

        var out = Tensor[Self.dtype](
            self.buffer.arithmetic_ops[Divide](other.buffer),
            requires_grad=False,
        )

        comptime if track_grad:
            requires_grad = self.requires_grad or other.requires_grad
            if requires_grad:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_DIVIDE
                )
                out.add_ancestry(backwardFnArg^, self, other)

        return out^
