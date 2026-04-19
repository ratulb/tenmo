from tenmo import Tensor
from backpropagation import (
    BackwardFnArg,
    ScalarArg,
    BACKWARD_DIVIDE,
    BACKWARD_DIV_SCALAR,
    BACKWARD_RIGHT_DIV_SCALAR,
)
from mnemonics import AddTensor, SubtractTensor, Divide, ReverseDivide
from common_utils import panic
from gradbox import Gradbox
from ancestry import Ancestor


@fieldwise_init
struct TrueDivBackwardScalar[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
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
        return [
            (
                ancestor^,
                divided^,
                AddTensor,
            )
        ]


@fieldwise_init
struct RightTrueDivBackwardScalar[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var scalar = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        var gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        var nd_buffer = ancestor.buffer()
        squared = nd_buffer * nd_buffer
        squared_reciprocal = Scalar[Self.dtype](1) / squared
        gradbox = (gradbox * scalar) * Gradbox[Self.dtype](
            squared_reciprocal^, share=False
        )

        return [
            (
                ancestor^,
                gradbox^,
                SubtractTensor,
            )
        ]


@fieldwise_init
struct DivideBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var ancestor_top = output.ancestry().get(0)
        var ancestor_bottom = output.ancestry().get(1)
        var buffer_top = ancestor_top.buffer()
        var buffer_bottom = ancestor_bottom.buffer()

        var grad_shares = List[
            Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]
        ](capacity=2)

        if ancestor_top.requires_grad:
            var buffer_bottom_reciprocal = Scalar[Self.dtype](1) / buffer_bottom
            ref buffer_top_shape = buffer_top.shape
            var ancestor_top_gradbox_buffer = (
                buffer_bottom_reciprocal * gradbox.buffer
            )
            if ancestor_top_gradbox_buffer.shape != buffer_top_shape:
                ancestor_top_gradbox_buffer = (
                    ancestor_top_gradbox_buffer.sum_over_broadcasted_axes(
                        buffer_top_shape
                    )
                )
            var ancestor_top_gradbox = Gradbox[Self.dtype](
                ancestor_top_gradbox_buffer^, share=False
            )
            grad_shares.append((ancestor_top, ancestor_top_gradbox^, AddTensor))

        if ancestor_bottom.requires_grad:
            var buffer_bottom_sqrd = buffer_bottom * buffer_bottom
            var buffer_bottom_sqrd_reciprocal = (
                Scalar[Self.dtype](1) / buffer_bottom_sqrd
            )
            var ancestor_bottom_grad_buffer = (
                buffer_top * buffer_bottom_sqrd_reciprocal
            )

            ancestor_bottom_grad_buffer = (
                ancestor_bottom_grad_buffer * gradbox.buffer
            )
            if ancestor_bottom_grad_buffer.shape != buffer_bottom.shape:
                ancestor_bottom_grad_buffer = (
                    ancestor_bottom_grad_buffer.sum_over_broadcasted_axes(
                        buffer_bottom.shape
                    )
                )
            var ancestor_bottom_gradbox = Gradbox[Self.dtype](
                ancestor_bottom_grad_buffer^, share=False
            )
            grad_shares.append(
                (ancestor_bottom^, ancestor_bottom_gradbox^, SubtractTensor)
            )

        return grad_shares^


@fieldwise_init
struct DivideScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
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
    fn forward[
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
    fn forward[
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


from shapes import Shape


fn main() raises:
    comptime dtype = DType.float32
    _ = """var A = Tensor[dtype].full(Shape.of(3, 3), 2, requires_grad=True)
    var B = Tensor[dtype].full(Shape.of(3, 1), 2, requires_grad=True)
    var C = (A / B) - 1 + (B * A) * 42
    #var C = (A / B) - 1
    # var C = 10/ A
    C.backward(42)
    A.grad().print()  # grad() call detaches
    B.grad().print()"""
    print("passes")
