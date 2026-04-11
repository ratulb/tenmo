from tenmo import Tensor
from backpropagation import (
    FnArg,
    ScalarArg,
    BACKWARD_DIVIDE,
    BACKWARD_DIV_SCALAR,
    BACKWARD_RIGHT_DIV_SCALAR,
)
from mnemonics import AddTensor, SubtractTensor, Divide, ReverseDivide
from common_utils import panic
from gradbox import Gradbox


@fieldwise_init
struct TrueDivBackwardScalar[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var scalar = output.fn_arg().arg[ScalarArg[Self.dtype]].scalar
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
struct RightTrueDivBackwardScalar[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var scalar = output.fn_arg().arg[ScalarArg[Self.dtype]].scalar
        var gradbox = output.grad()
        var tensor = output.ancestry().get(0)
        squared = tensor.__pow__[track_grad=False](2)
        squared_reciprocal = 1.0 / squared
        gradbox = (gradbox * scalar) * squared_reciprocal

        return [
            (
                tensor^,
                gradbox^,
                SubtractTensor,
            )
        ]


@fieldwise_init
struct DivideBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
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
struct DivideScalar[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        comptime assert
            Self.dtype.is_numeric(),
            "Tensor → __rtruediv__ is for numeric data types only"

        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[ReverseDivide](scalar), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                out.fnArg = Optional(FnArg[Self.dtype].scalar(scalar, BACKWARD_RIGHT_DIV_SCALAR))
                out.add_ancestry(self)

        return out^


@fieldwise_init
struct DivideByScalar[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        comptime assert
            Self.dtype.is_numeric(),
            "Tensor → __truediv__ is for numeric data types only"

        if scalar == Scalar[Self.dtype](0):
            panic("Tensor → __truediv__ : canot divide by " + String(scalar))

        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[Divide](scalar), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                out.fnArg = Optional(FnArg[Self.dtype].scalar(scalar, BACKWARD_DIV_SCALAR))
                out.add_ancestry(self)

        return out^


# Element wise division of two tensors
@fieldwise_init
struct Divider[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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
                out.fnArg = Optional(FnArg[Self.dtype].null(BACKWARD_DIVIDE))
                out.add_ancestry(self, other)

        return out^


from common_utils import now
from std.testing import assert_true


fn main() raises:
    comptime dtype = DType.float32
    a1 = Tensor[dtype].rand(5000, 1000)
    b1 = Tensor[dtype].rand(5000, 1000)
    a = a1.transpose(0, 1)
    b = b1.transpose(0, 1)
    start = now()
    r1 = a / b
    print("CPU divide took: ", (now() - start) * 1000, "ms")
    start = now()
    ag = a.to_gpu()
    bg = b.to_gpu()
    print("Transfer to gpu took: ", (now() - start) * 1000, "ms")
    start = now()
    r2 = ag / bg
    print("Overall GPU took: ", (now() - start) * 1000, "ms")
    assert_true(r1.all_close(r2.to_cpu()))
