from tenmo import Tensor
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_DIVIDE,
    BACKWARD_DIV_SCALAR,
    BACKWARD_RIGHT_DIV_SCALAR,
)
from operators import AddTensor, SubtractTensor, Divide, ReverseDivide
from common_utils import panic
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct TrueDivBackwardScalar[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_DIV_SCALAR
    var factor: Scalar[Self.dtype]

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        ancestor = output.ancestry().get(0)
        # ∂(x / s)/∂x = 1/s → incoming_grad / scalar
        var divided = gradbox / self.factor
        return [
            (
                ancestor^,
                divided^,
                AddTensor,
            )
        ]


@fieldwise_init
@register_passable
struct RightTrueDivBackwardScalar[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_RIGHT_DIV_SCALAR
    var scalar: Scalar[Self.dtype]

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var gradbox = output.grad()
        var tensor = output.ancestry().get(0)
        squared = tensor.__pow__[track_grad=False](2)
        squared_reciprocal = 1.0 / squared
        gradbox = (gradbox * self.scalar) * squared_reciprocal

        return [
            (
                tensor^,
                gradbox^,
                SubtractTensor,
            )
        ]


@fieldwise_init
@register_passable
struct DivideBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_DIVIDE

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var tensor_top = output.ancestry().get(0)
        var tensor_bottom = output.ancestry().get(1)

        grad_shares = List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]](
            capacity=2
        )

        if tensor_top.requires_grad:
            tensor_bottom_reciprocal = DivideScalar[Self.dtype].forward[
                track_grad=False
            ](tensor_bottom, 1)
            tensor_top_shape = tensor_top.shape()
            tensor_top_gradbox = tensor_bottom_reciprocal * gradbox
            if tensor_top_gradbox.shape() != tensor_top_shape:
                tensor_top_gradbox = (
                    tensor_top_gradbox.sum_over_broadcasted_axes(
                        tensor_top_shape
                    )
                )
            grad_shares.append((tensor_top, tensor_top_gradbox^, AddTensor))

        if tensor_bottom.requires_grad:
            tensor_bottom_squared = tensor_bottom.__mul__[track_grad=False](
                tensor_bottom
            )
            tensor_bottom_squared_reciprocal = DivideScalar[Self.dtype].forward[
                track_grad=False
            ](tensor_bottom_squared, 1)
            tensor_bottom_grad = tensor_top.__mul__[track_grad=False](
                tensor_bottom_squared_reciprocal
            )
            tensor_bottom_gradbox = tensor_bottom_grad * gradbox
            if tensor_bottom_gradbox.shape() != tensor_bottom.shape():
                tensor_bottom_gradbox = (
                    tensor_bottom_gradbox.sum_over_broadcasted_axes(
                        tensor_bottom.shape()
                    )
                )
            grad_shares.append(
                (tensor_bottom^, tensor_bottom_gradbox^, SubtractTensor)
            )

        return grad_shares^


@fieldwise_init
@register_passable
struct DivideScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Tensor → __rtruediv__ is for numeric data types only",
        ]()

        nd_buffer = self.buffer.scalar_ops[ReverseDivide](scalar)
        var out = Tensor[Self.dtype](nd_buffer^, requires_grad=False)

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = RightTrueDivBackwardScalar[Self.dtype](
                    scalar
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


@fieldwise_init
@register_passable
struct DivideByScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        constrained[
            Self.dtype.is_numeric(),
            "Tensor → __truediv__ is for numeric data types only",
        ]()

        if scalar == Scalar[Self.dtype](0):
            panic("Tensor → __truediv__ : canot divide by " + scalar.__str__())

        nd_buffer = self.buffer.scalar_ops[Divide](scalar)
        var out = Tensor[Self.dtype](nd_buffer^, requires_grad=False)

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)

                backward_fn = TrueDivBackwardScalar[Self.dtype](
                    scalar
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


# Element wise division of two tensors
@fieldwise_init
@register_passable
struct Divider[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], other: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        if not self.broadcastable(other):
            panic(
                "Tensor →__truediv__(self * other): dimension mismatch: "
                + self.shape().__str__()
                + " <=> "
                + other.shape().__str__(),
                "at Divider → forward",
            )
        nd_buffer = self.buffer.arithmetic_ops[Divide](other.buffer)
        var out = Tensor[Self.dtype](nd_buffer^, requires_grad=False)

        @parameter
        if track_grad:
            requires_grad = self.requires_grad or other.requires_grad
            if requires_grad:
                out.requires_grad_(True)
                backward_fn = DivideBackward[Self.dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self, other)

        return out^
